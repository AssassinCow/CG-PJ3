"""Cross-reference layer: join vision-critic findings with scene AST evidence.

The vision critic produces free-text findings ("label 'E' missing in frame 36").
The contract validator parses the AST. Each layer operates in isolation. This
module joins them: when both sources independently corroborate a problem, we
emit an actionable diagnostic that names the exact AST evidence so the coder
LLM does not have to do the symptom-to-cause inference itself.

The output is purely additive — it never downgrades or hides critic findings.
It sits in the coder addendum *before* the verbatim critic prose so coder reads
double-corroborated diagnostics first.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from cg_tutor.concept_metrics import (
    _attr_chain,
    _attr_chain_root,
    _keyframe_data_path_arg,
    _resolve_object_name_expr,
)
from cg_tutor.schemas import Storyboard
from cg_tutor.schemas.feedback import CriticIssue
from cg_tutor.visual_contract import VisualContract


Severity = Literal["actionable", "deferred"]


@dataclass(frozen=True)
class CrossReferenceFinding:
    rule_id: str
    severity: Severity
    diagnosis: str
    critic_source: str
    ast_evidence: str
    suggested_fix: str


@dataclass
class CrossReferenceReport:
    concept_id: str
    iteration: int
    findings: list[CrossReferenceFinding] = field(default_factory=list)

    @property
    def actionable_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "actionable")

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "iteration": self.iteration,
            "actionable_count": self.actionable_count,
            "findings": [asdict(f) for f in self.findings],
        }


def cross_reference_critic_findings(
    *,
    concept_id: str,
    iteration: int,
    critic_issues: list[CriticIssue],
    scene_code: str,
    storyboard: Storyboard,
    visual_contracts: dict[str, VisualContract] | None = None,
) -> CrossReferenceReport:
    """Run all cross-reference rules over the latest critic iteration.

    Caller is expected to pass ``critic_history[-1].report.issues`` (the current
    iter's findings). Older iterations are not consulted — that is a separate
    persistence-detection concern out of scope here.
    """
    report = CrossReferenceReport(concept_id=concept_id, iteration=iteration)
    if not critic_issues or not scene_code.strip():
        return report

    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return report

    created_names = _all_created_object_names(tree)
    schedule = _keyframe_schedule(tree)
    storyboard_names = _storyboard_object_names(storyboard)
    contract_required = _contract_required_tokens(visual_contracts)

    for issue in critic_issues:
        source = (
            f"iter{iteration:02d} shot={issue.shot_id} frame={issue.frame_idx} "
            f"{issue.severity}/{issue.category}"
        )
        tokens = _extract_candidate_tokens(issue.issue)
        contract_tokens_for_shot = contract_required.get(issue.shot_id, set())

        report.findings.extend(_rule_missing_object_creation(
            issue=issue,
            source=source,
            tokens=tokens,
            created_names=created_names,
            contract_tokens=contract_tokens_for_shot,
        ))
        report.findings.extend(_rule_misnamed_object(
            issue=issue,
            source=source,
            tokens=tokens,
            created_names=created_names,
            storyboard_names=storyboard_names,
            contract_tokens=contract_tokens_for_shot,
        ))
        report.findings.extend(_rule_keyframe_ramp_too_late(
            issue=issue,
            source=source,
            schedule=schedule,
        ))
        report.findings.extend(_rule_object_hidden_at_frame(
            issue=issue,
            source=source,
            tokens=tokens,
            schedule=schedule,
        ))

    report.findings = _dedupe_findings(report.findings)
    return report


# ---------------- Rules ----------------


_MISSING_PATTERNS = re.compile(
    r"\b(missing|absent|invisible|"
    r"not\s+(?:visible|readable|present|attached|shown|drawn|rendered)|"
    r"is\s+(?:not|never)\s+visible|"
    r"does\s+not\s+(?:show|appear|contain)|"
    r"violating\s+(?:the\s+)?(?:derived_)?visual_contract|"
    r"instead\s+of|labeled\s+\S+\s+(?:instead|but)|"
    r"wrong\s+(?:label|name|tag)|incorrect\s+(?:label|name|tag)|"
    r"should\s+be\s+labeled|required\s+(?:label|labels|anchor|anchors|to\s+be|'))",
    re.IGNORECASE,
)

_NO_X_VISIBLE_PATTERN = re.compile(
    r"\bno\b.{0,60}\b(visible|drawn|present|shown|rendered)\b",
    re.IGNORECASE,
)


def _critic_indicates_absence(text: str) -> bool:
    """True iff the critic text contains an absence/mismatch indicator.

    Broad on purpose: each rule has its own AST/contract safety check, so
    false positives in this filter cannot produce false-positive findings.
    """
    return bool(_MISSING_PATTERNS.search(text) or _NO_X_VISIBLE_PATTERN.search(text))


def _rule_missing_object_creation(
    *,
    issue: CriticIssue,
    source: str,
    tokens: list[str],
    created_names: set[str],
    contract_tokens: set[str],
) -> list[CrossReferenceFinding]:
    """Critic claims X missing AND scene AST never creates X."""
    if not _critic_indicates_absence(issue.issue):
        return []
    findings: list[CrossReferenceFinding] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        if _token_matches_created_name(token, created_names):
            continue
        # Only commit if the contract independently expects this token
        # (avoids false positives on every quoted word in the critic prose).
        if token not in contract_tokens:
            continue
        seen.add(token)
        findings.append(CrossReferenceFinding(
            rule_id="missing_object_creation",
            severity="actionable",
            diagnosis=(
                f"Vision critic reports '{token}' missing in render; AST "
                f"confirms scene never creates an object named '{token}'."
            ),
            critic_source=f"{source} — {issue.issue[:140]}",
            ast_evidence=(
                f"scene.py created objects: "
                f"{', '.join(sorted(created_names)[:12]) or '(none)'}"
                + (" ..." if len(created_names) > 12 else "")
            ),
            suggested_fix=(
                f"Create the missing object: "
                f"`bpy.data.objects.new('{token}', ...)` and link it to the "
                "scene collection. Match the contract requirement for shot "
                f"{issue.shot_id}."
            ),
        ))
    return findings


def _normalize_scene_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _token_matches_created_name(token: str, created_names: set[str]) -> bool:
    if token in created_names:
        return True
    token_norm = _normalize_scene_token(token)
    if not token_norm:
        return False
    for name in created_names:
        name_norm = _normalize_scene_token(name)
        if token_norm == name_norm:
            return True
        # Text labels are commonly named label_lod1/text_E while critics cite
        # only the visible label body ("LOD1", "E"). Treat that as present for
        # cross-ref purposes; visibility/readability remains a critic issue.
        if len(token_norm) >= 3 and name_norm.endswith(token_norm):
            prefix = name_norm[:-len(token_norm)]
            if prefix in {"label", "text", "tile", "object"}:
                return True
    return False


_INSTEAD_OF_PATTERN = re.compile(
    r"(?:labeled\s+|named\s+|tagged\s+|shows?\s+)?"
    r"['\"]?([A-Za-z][A-Za-z0-9_]{0,40})['\"]?\s+"
    r"instead\s+of\s+(?:the\s+)?(?:required\s+|expected\s+|correct\s+)?"
    r"['\"]?([A-Za-z][A-Za-z0-9_]{0,40})['\"]?",
    re.IGNORECASE,
)


def _rule_misnamed_object(
    *,
    issue: CriticIssue,
    source: str,
    tokens: list[str],
    created_names: set[str],
    storyboard_names: set[str],
    contract_tokens: set[str],
) -> list[CrossReferenceFinding]:
    """Contract expects X. Scene created Y (not X). Critic says "Y instead of X".

    Requires directional 'X instead of Y' phrasing in the critic text so the
    suggested rename is unambiguous. Without that signal we fall back to the
    primary missing_object_creation rule rather than guess a rename target.
    """
    if not _critic_indicates_absence(issue.issue):
        return []
    matches = list(_INSTEAD_OF_PATTERN.finditer(issue.issue))
    if not matches:
        return []
    findings: list[CrossReferenceFinding] = []
    seen: set[tuple[str, str]] = set()
    for m in matches:
        actual = m.group(1).strip("'\"")
        required = m.group(2).strip("'\"")
        if actual in _STOP_TOKENS or required in _STOP_TOKENS:
            continue
        if (actual, required) in seen:
            continue
        # AST must corroborate: required is missing, actual or something like
        # it is in the scene.
        if required in created_names:
            continue
        if required not in contract_tokens and required not in tokens:
            continue
        ast_has_actual = (
            actual in created_names
            or any(actual in n for n in created_names if len(actual) >= 2)
        )
        if not ast_has_actual:
            continue
        seen.add((actual, required))
        findings.append(CrossReferenceFinding(
            rule_id="misnamed_object",
            severity="actionable",
            diagnosis=(
                f"Critic reports object labeled '{actual}' where '{required}' "
                f"was required. AST confirms scene has '{actual}' but not "
                f"'{required}'."
            ),
            critic_source=f"{source} — {issue.issue[:140]}",
            ast_evidence=(
                f"contract/required: '{required}'; scene created: '{actual}' "
                f"(not '{required}')"
            ),
            suggested_fix=(
                f"Rename the existing '{actual}' references to '{required}' "
                "(update all bpy.data.objects.new / bpy.data.objects[] "
                "references, plus any label text bodies, keyframe_insert "
                "calls, and parent linkage). Do not add a duplicate object."
            ),
        ))
    return findings


_TRACE_MENTION_PATTERN = re.compile(
    r"\b(trace|trajectory|trail|growing\s+curve|arc)\b",
    re.IGNORECASE,
)


def _critic_indicates_trace_absent(text: str) -> bool:
    """True iff critic text both mentions a trace-like noun AND signals absence.

    Catches both 'trajectory trace is absent' and 'No trace ... visible'.
    """
    if not _TRACE_MENTION_PATTERN.search(text):
        return False
    return _critic_indicates_absence(text)

_RAMP_PROPERTIES: tuple[str, ...] = (
    "bevel_factor_end", "bevel_factor_start", "value", "alpha",
    "factor", "influence",
)


def _rule_keyframe_ramp_too_late(
    *,
    issue: CriticIssue,
    source: str,
    schedule: dict[str, list[tuple[str, int, object]]],
) -> list[CrossReferenceFinding]:
    """Critic sees no trace at frame F, AST keyframes ramp 0→1 too late."""
    if not _critic_indicates_trace_absent(issue.issue):
        return []
    f = issue.frame_idx
    findings: list[CrossReferenceFinding] = []
    for var, events in schedule.items():
        for data_path, frame, value in events:
            if data_path not in _RAMP_PROPERTIES:
                continue
            ramp_value_at_f = _ramp_value_at(events, data_path, f)
            if ramp_value_at_f is None or ramp_value_at_f > 0.0:
                continue
            ramp_summary = ", ".join(
                f"f{ev_f}={ev_v!r}" for dp, ev_f, ev_v in events
                if dp == data_path
            )
            findings.append(CrossReferenceFinding(
                rule_id="keyframe_ramp_too_late",
                severity="actionable",
                diagnosis=(
                    f"Critic sees no {data_path}-driven reveal at frame {f}; "
                    f"AST shows '{var}.{data_path}' is keyframed at 0.0 through "
                    f"this frame."
                ),
                critic_source=f"{source} — {issue.issue[:140]}",
                ast_evidence=(
                    f"{var}.{data_path} schedule: {ramp_summary}"
                ),
                suggested_fix=(
                    f"Move the first non-zero keyframe of '{var}.{data_path}' "
                    f"to <= frame {f}, or remove the leading zero keyframes so "
                    "the reveal starts at frame 1."
                ),
            ))
            break  # one finding per (var, data_path)
    return findings


def _rule_object_hidden_at_frame(
    *,
    issue: CriticIssue,
    source: str,
    tokens: list[str],
    schedule: dict[str, list[tuple[str, int, object]]],
) -> list[CrossReferenceFinding]:
    """Critic claims X missing at frame F, AST has X.hide_* True at frame F."""
    if not _critic_indicates_absence(issue.issue):
        return []
    f = issue.frame_idx
    findings: list[CrossReferenceFinding] = []
    for token in tokens:
        events = schedule.get(token)
        if not events:
            continue
        for data_path in ("hide_render", "hide_viewport"):
            value_at_f = _ramp_value_at(events, data_path, f)
            if value_at_f is True:
                summary = ", ".join(
                    f"f{ev_f}={ev_v!r}" for dp, ev_f, ev_v in events
                    if dp == data_path
                )
                findings.append(CrossReferenceFinding(
                    rule_id="object_hidden_at_frame",
                    severity="actionable",
                    diagnosis=(
                        f"Critic reports '{token}' not visible at frame {f}; "
                        f"AST shows '{token}.{data_path}' is True at this "
                        "frame (object hidden by design)."
                    ),
                    critic_source=f"{source} — {issue.issue[:140]}",
                    ast_evidence=(
                        f"{token}.{data_path} schedule: {summary}"
                    ),
                    suggested_fix=(
                        f"Either reveal '{token}' earlier by setting "
                        f"{data_path}=False at or before frame {f}, or "
                        "remove the hide-keyframes so it is visible "
                        "throughout."
                    ),
                ))
                break  # one finding per (token, data_path) pair
    return findings


# ---------------- AST helpers ----------------


def _all_created_object_names(tree: ast.AST) -> set[str]:
    """Set of literal object names created/referenced in scene.py.

    Covers both explicit object creation (`bpy.data.objects.new('x', ...)`)
    and the common bpy.ops pattern:

        bpy.ops.mesh.primitive_cube_add(...)
        obj = bpy.context.object
        obj.name = 'x'

    The latter is the dominant style in LLM-generated Blender scripts; missing
    it makes cross-reference diagnostics falsely report that visible objects
    were never created.
    """
    _, names = _object_aliases_and_created_names(tree)
    return names


def _keyframe_schedule(
    tree: ast.AST,
) -> dict[str, list[tuple[str, int, object]]]:
    """Return {var_name: [(data_path, frame, value_or_None), ...]}.

    var_name is resolved through bpy.data.objects['x'] / .new('x', ...) when
    possible, so the keys match _all_created_object_names. Frame is taken
    from the keyword/positional frame argument when a literal int. Value is
    the most-recent literal assignment to that (receiver, data_path) pair
    in linear source order before the keyframe_insert call.
    """
    object_var_to_name, _ = _object_aliases_and_created_names(tree)
    frame_aliases = _frame_aliases(tree)
    assigns: list[tuple[int, str, str, object]] = []
    keyframes: list[tuple[int, str, str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
            ):
                t = node.targets[0]
                root = _attr_chain_root(t.value)
                if root is None:
                    continue
                if not isinstance(node.value, ast.Constant):
                    continue
                assigns.append((
                    getattr(node, "lineno", 0),
                    root,
                    t.attr,
                    node.value.value,
                ))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "keyframe_insert":
                continue
            recv_root = _attr_chain_root(node.func.value)
            data_path = _keyframe_data_path_arg(node)
            frame = _keyframe_frame_arg(node, frame_aliases)
            if recv_root is None or data_path is None or frame is None:
                continue
            keyframes.append((
                getattr(node, "lineno", 0),
                recv_root,
                data_path,
                frame,
            ))

    assigns.sort(key=lambda x: x[0])

    schedule: dict[str, list[tuple[str, int, object]]] = {}
    for kf_line, recv_root, data_path, frame in keyframes:
        value: object | None = None
        for a_line, a_root, a_attr, a_val in assigns:
            if a_line > kf_line:
                break
            if a_root == recv_root and a_attr == data_path:
                value = a_val
        resolved_var = object_var_to_name.get(recv_root, recv_root)
        schedule.setdefault(resolved_var, []).append(
            (data_path, frame, value)
        )
    return schedule


def _frame_aliases(tree: ast.AST) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for node in _nodes_in_source_order(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = _frame_expr_value(node.value, aliases)
        if value is None:
            continue
        if isinstance(target, ast.Name):
            aliases[target.id] = value
            continue
        chain = _attr_chain(target)
        if chain:
            aliases[".".join(chain)] = value
    return aliases


def _object_aliases_and_created_names(
    tree: ast.AST,
) -> tuple[dict[str, str], set[str]]:
    names: set[str] = set()
    aliases: dict[str, str | None] = {}
    factories = _object_factory_functions(tree)

    for node in _nodes_in_source_order(tree):
        if isinstance(node, (ast.Subscript, ast.Call)):
            resolved = _resolve_object_name_expr(node)
            if resolved is not None:
                names.add(resolved)
            factory_name = _call_name(node) if isinstance(node, ast.Call) else None
            if factory_name in factories:
                param_index = factories[factory_name]
                if len(node.args) > param_index:
                    arg = node.args[param_index]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        names.add(arg.value)
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            resolved = _resolve_object_name_expr(node.value)
            if resolved is not None:
                aliases[target.id] = resolved
                names.add(resolved)
            elif _is_context_object_expr(node.value):
                aliases.setdefault(target.id, None)
            continue
        if not isinstance(target, ast.Attribute):
            continue
        if target.attr != "name":
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        literal_name = node.value.value
        if not literal_name:
            continue
        chain = _attr_chain(target.value)
        if chain in {
            ("bpy", "context", "object"),
            ("bpy", "context", "active_object"),
        }:
            names.add(literal_name)
            continue
        root = _attr_chain_root(target.value)
        if root is not None:
            aliases[root] = literal_name
            names.add(literal_name)

    return {k: v for k, v in aliases.items() if v is not None}, names


def _object_factory_functions(tree: ast.AST) -> dict[str, int]:
    """Return helper function names that create/rename objects from a param.

    LLM Blender scripts commonly wrap bpy creation in helpers such as
    `add_screen(name, ...)` or `make_label(name, ...)`; the literal object
    name then appears at the call site, not inside `bpy.data.objects.new`.
    """
    factories: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        params = [arg.arg for arg in node.args.args]
        if not params:
            continue
        param_to_index = {name: idx for idx, name in enumerate(params)}
        for inner in ast.walk(node):
            param_name: str | None = None
            if (
                isinstance(inner, ast.Assign)
                and len(inner.targets) == 1
                and isinstance(inner.targets[0], ast.Attribute)
                and inner.targets[0].attr == "name"
                and isinstance(inner.value, ast.Name)
            ):
                param_name = inner.value.id
            elif isinstance(inner, ast.Call):
                resolved_call = _attr_chain(inner.func)
                if resolved_call == ("bpy", "data", "objects", "new"):
                    if inner.args and isinstance(inner.args[0], ast.Name):
                        param_name = inner.args[0].id
            if param_name in param_to_index:
                factories[node.name] = param_to_index[param_name]
                break
    return factories


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _nodes_in_source_order(tree: ast.AST) -> list[ast.AST]:
    return sorted(
        ast.walk(tree),
        key=lambda n: (
            getattr(n, "lineno", 10**9),
            getattr(n, "col_offset", 10**9),
        ),
    )


def _is_context_object_expr(node: ast.AST) -> bool:
    return _attr_chain(node) in {
        ("bpy", "context", "object"),
        ("bpy", "context", "active_object"),
    }


def _keyframe_frame_arg(
    call: ast.Call,
    frame_aliases: dict[str, int] | None = None,
) -> int | None:
    """Extract the frame argument as int literal, else None."""
    frame_aliases = frame_aliases or {}
    # Positional second arg (after data_path)
    if len(call.args) >= 2:
        resolved = _frame_expr_value(call.args[1], frame_aliases)
        if resolved is not None:
            return resolved
    for kw in call.keywords:
        if kw.arg == "frame":
            return _frame_expr_value(kw.value, frame_aliases)
    return None


def _frame_expr_value(
    node: ast.AST,
    frame_aliases: dict[str, int],
) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return int(node.value)
    if isinstance(node, ast.Name):
        return frame_aliases.get(node.id)
    chain = _attr_chain(node)
    if chain:
        return frame_aliases.get(".".join(chain))
    return None


def _ramp_value_at(
    events: list[tuple[str, int, object]],
    data_path: str,
    frame: int,
) -> object | None:
    """Step-interpolate the keyframe value at the given frame for data_path.

    Returns the most recent keyed value at frame F, or None if there are
    no keyframes for this data_path (or all values were non-literal).
    """
    relevant = sorted(
        (f, v) for dp, f, v in events
        if dp == data_path and v is not None
    )
    if not relevant:
        return None
    if frame <= relevant[0][0]:
        return relevant[0][1]
    last_value: object = relevant[0][1]
    for f, v in relevant:
        if f <= frame:
            last_value = v
        else:
            break
    return last_value


# ---------------- Token extraction ----------------


_STOP_TOKENS: frozenset[str] = frozenset({
    "The", "A", "An", "It", "Is", "Are", "Be", "On", "In", "At", "By",
    "Of", "To", "For", "From", "With", "And", "Or", "But", "Not", "No",
    "If", "As", "So", "Do", "Does", "Did", "Has", "Have", "Had", "Will",
    "Can", "May", "Must", "Should", "Would", "Could",
    "I", "We", "You", "They",  # pronouns sometimes capitalized
})


def _extract_candidate_tokens(text: str) -> list[str]:
    """Pull identifier-like tokens out of the critic's free-text issue.

    Targets three categories of token critics use to name scene objects:
      - quoted strings ('E', "trajectory_trace", 'tip')
      - underscore-separated identifiers (trajectory_trace, joint_J1)
      - short uppercase IDs (E, J1, J2, P0, R, L, V)

    Filters short common English words via _STOP_TOKENS. We bias toward
    recall — false positives are harmless because later rules require the
    token to also be in the contract or AST.
    """
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        if not t or t in seen or t in _STOP_TOKENS:
            return
        seen.add(t)
        tokens.append(t)

    for m in re.finditer(r"['\"]([A-Za-z0-9_\-]{1,40})['\"]", text):
        _add(m.group(1))
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\b", text):
        _add(m.group(1))
    for m in re.finditer(r"\b([A-Z][A-Za-z]?[0-9]+)\b", text):
        _add(m.group(1))
    for m in re.finditer(r"(?<![A-Za-z0-9])([A-Z])(?![A-Za-z0-9_])", text):
        _add(m.group(1))
    return tokens


def _storyboard_object_names(storyboard: Storyboard) -> set[str]:
    names: set[str] = set()
    for shot in storyboard.shots:
        for obj in shot.objects:
            if obj.name:
                names.add(obj.name)
    return names


def _contract_required_tokens(
    contracts: dict[str, VisualContract] | None,
) -> dict[str, set[str]]:
    """Return {shot_id: set of required tokens} aggregated from the contract's
    required_labels / required_vectors / required_anchors. We tokenize each
    string to catch entries like 'label E' or 'arrow N' that the critic might
    reference simply as 'E' or 'N'."""
    if not contracts:
        return {}
    out: dict[str, set[str]] = {}
    for shot_id, contract in contracts.items():
        tokens: set[str] = set()
        fields = (
            contract.required_labels
            + contract.required_anchors
            + contract.required_vectors
        )
        for entry in fields:
            for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\b", entry):
                tok = m.group(1)
                if tok in _STOP_TOKENS:
                    continue
                tokens.add(tok)
            # Also keep multi-word labels themselves so 'visible text labels'
            # is matchable against a `bpy.data.objects.new('visible text labels',...)`
            # (rare but possible).
            tokens.add(entry)
        out[shot_id] = tokens
    return out


def _dedupe_findings(
    findings: list[CrossReferenceFinding],
) -> list[CrossReferenceFinding]:
    seen: set[tuple[str, str, str]] = set()
    out: list[CrossReferenceFinding] = []
    for f in findings:
        key = (f.rule_id, f.diagnosis, f.ast_evidence)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# ---------------- Formatter ----------------


def format_cross_reference_for_coder(report: CrossReferenceReport) -> str:
    """Render the cross-reference findings as a coder-facing addendum.

    Returns '' when there are no findings, so this can be safely joined with
    other addendum parts without producing dangling headers.
    """
    if not report.findings:
        return ""
    lines = [
        "CRITIC × AST CROSS-REFERENCE",
        "These findings are double-corroborated (vision critic AND scene",
        "AST agree). Fix these before applying critic prose suggestions —",
        "they are deterministic and high-confidence.",
        "",
    ]
    for f in report.findings:
        lines.append(f"[{f.severity}] {f.rule_id}")
        lines.append(f"  diagnosis: {f.diagnosis}")
        lines.append(f"  critic:    {f.critic_source}")
        lines.append(f"  ast:       {f.ast_evidence}")
        lines.append(f"  fix:       {f.suggested_fix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def cross_reference_report_to_json(report: CrossReferenceReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
