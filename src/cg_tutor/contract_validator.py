"""Structured validation of VisualContract requirements against scene.py.

Visual contracts (``visual_contract.VisualContract``) used to live as
free text in the coder addendum. The render critic was free to ignore
them — its prompt mentioned the contract but its scoring didn't
*structurally* check that the requirements actually appeared in the
generated Blender script.

This module fills that gap with deterministic, machine-readable checks
that run on the generated scene.py AST. It is intentionally narrow:

- ``required_labels`` → confirm the script creates at least N text
  objects, where N is the number of distinct required labels.
- ``required_anchors`` → confirm required scene grounding anchor names
  appear as string literals in scene.py.
- ``required_vectors`` → confirm the script creates enough arrow-like
  primitives (cones/cylinders/curve_polylines) per shot's vector count.
- ``emphasis_points`` → confirm the script assigns an Emission Color
  / Emission Strength somewhere (a "bright spot" hint).

The results are emitted as a JSON artifact next to scene_verifier so
the repair plan and critic best-selection can read them. Each finding
has the same severity/rule shape the verifier already uses; the
critic loop's existing scoring picks them up without changes.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal

from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.visual_contract import VisualContract


Severity = Literal["block", "warn"]
SanityVerdict = Literal["downgrade", "keep"]


# Registry of per-rule sanity checks. A sanity check is invoked with the
# violation, the parsed AST, and the raw source code; it returns one of:
#   "downgrade" — the rule fired but contradicting evidence in the AST says
#                 the count is wrong; downgrade severity (block -> warn).
#   "keep"      — no contradicting evidence; keep the violation as-is.
# This is a routing layer over the rules, not a new judge: the original
# violation is preserved (with prefixed message) for audit.
RULE_SANITY_CHECKS: dict[
    str,
    Callable[["ContractViolation", ast.Module, str], tuple[SanityVerdict, str]],
] = {}


def register_sanity_check(rule_id: str):
    """Decorator. Registers a callback `(violation, tree, code) -> (verdict, reason)`."""
    def deco(fn):
        RULE_SANITY_CHECKS[rule_id] = fn
        return fn
    return deco


@dataclass(frozen=True)
class ContractViolation:
    severity: Severity
    rule_id: str
    shot_id: str
    field: str
    expected: int
    found: int
    message: str
    suggested_fix: str


@dataclass
class ContractValidationReport:
    violations: list[ContractViolation] = field(default_factory=list)
    per_shot_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "block")

    @property
    def warn_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == "warn")

    @property
    def ok(self) -> bool:
        return self.block_count == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "block": self.block_count,
            "warn": self.warn_count,
            "violations": [asdict(v) for v in self.violations],
            "per_shot_counts": self.per_shot_counts,
        }


# ---------------------------------------------------------------------------
# AST scanners — global (whole-script) counts. We don't try to associate
# objects with the shot they belong to: scene.py is usually structured as
# "create all objects, then animate hide_render per shot", so a global
# count is what matters. The per-shot breakdown comes from cross-referencing
# storyboard.shots[*].objects names that appear in the AST as string
# literals (existing scene_verifier.missing_objects already does this).
# ---------------------------------------------------------------------------


def _safe_parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _attr_dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _count_calls_dotted(tree: ast.Module, dotted: str) -> int:
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _attr_dotted(node.func) == dotted:
                n += 1
    return n


def _has_emission_assignment(tree: ast.Module) -> bool:
    """Heuristic: any Subscript like ``...inputs['Emission Color']`` written to
    by an Assign, OR an attribute ``.emission_color = ...``. Either signals
    the script tries to set a self-luminous material."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                # bsdf.inputs['Emission Color'].default_value = (...)
                cur = target
                while isinstance(cur, ast.Attribute):
                    cur = cur.value
                if isinstance(cur, ast.Subscript):
                    idx = cur.slice
                    if isinstance(idx, ast.Constant) and isinstance(idx.value, str):
                        if "Emission" in idx.value:
                            return True
                # mat.emission_color = ...
                chain = _attr_dotted(target)
                if chain and chain.endswith(("emission_color", "emission_strength")):
                    return True
    return False


def _count_text_objects(tree: ast.Module) -> int:
    """Count ``bpy.ops.object.text_add(...)`` calls AND ``bpy.data.curves.new(...)``
    calls of TYPE='FONT' (the lower-level alternative). Either creates a
    Blender text data block."""
    n = _count_calls_dotted(tree, "bpy.ops.object.text_add")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _attr_dotted(node.func) != "bpy.data.curves.new":
            continue
        for kw in node.keywords:
            if kw.arg == "type" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "FONT":
                    n += 1
    n = max(n, _count_text_factory_invocations(tree))
    return n


def _count_text_factory_invocations(tree: ast.Module) -> int:
    """Count calls to local helper functions that create text objects.

    Generated scene.py often defines one helper such as ``add_text_label()``
    containing a single ``bpy.ops.object.text_add`` call, then invokes it many
    times from literal shot label tables. Counting only the helper body reports
    one text object and creates a false pre-render block. We count direct calls
    to local text factories as the more faithful static lower bound.
    """
    text_factories: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        has_text_add = any(
            isinstance(child, ast.Call)
            and _attr_dotted(child.func) == "bpy.ops.object.text_add"
            for child in ast.walk(node)
        )
        if has_text_add:
            text_factories.add(node.name)
    if not text_factories:
        return 0

    n = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in text_factories:
            n += 1
    n = max(n, _count_text_factory_invocations_with_literal_loops(tree, text_factories))
    return n


def _count_text_factory_invocations_with_literal_loops(
    tree: ast.Module,
    text_factories: set[str],
) -> int:
    literal_lengths: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        length = len(node.value.elts)
        for target in node.targets:
            if isinstance(target, ast.Name):
                literal_lengths[target.id] = length

    expanded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        loop_len = 0
        if isinstance(node.iter, ast.Name):
            loop_len = literal_lengths.get(node.iter.id, 0)
        elif (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and node.iter.args
            and isinstance(node.iter.args[0], ast.Name)
        ):
            loop_len = literal_lengths.get(node.iter.args[0].id, 0)
        elif isinstance(node.iter, (ast.List, ast.Tuple)):
            loop_len = len(node.iter.elts)
        if loop_len <= 0:
            continue
        calls_in_body = sum(
            1
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in text_factories
        )
        expanded += calls_in_body * loop_len
    return expanded


def _string_literals(tree: ast.Module) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.add(node.value)
    return values


def _anchor_is_meaningful(anchor: str) -> bool:
    """Reject anchors too short to substring-match reliably.

    A bare 'L' or 'sign' would otherwise be 'present' in almost any
    scene because the substring scan would match incidental identifiers
    ('LIGHT', 'design'). Require either >=4 chars OR an underscore /
    real camel-case boundary so each anchor token carries enough
    specificity for the match to be meaningful.

    The CamelCase test is a strict lowercase→uppercase boundary, not
    "any uppercase past position 0" — otherwise 2-letter acronyms like
    'AB' would falsely qualify.
    """
    if not anchor:
        return False
    if "_" in anchor:
        return True
    for i in range(len(anchor) - 1):
        if anchor[i].islower() and anchor[i + 1].isupper():
            return True
    return len(anchor) >= 4


def _anchor_is_present(anchor: str, string_values: set[str]) -> bool:
    if not _anchor_is_meaningful(anchor):
        # Treat short anchors as "skip rather than guess"; the prompt-layer
        # forbidden_failures still carries the constraint.
        return True
    anchor_low = anchor.lower()
    for value in string_values:
        value_low = value.lower()
        if anchor_low == value_low:
            return True
        if anchor_low in value_low or value_low in anchor_low:
            return True
    return False


def _count_named_vs_anonymous_primitives(tree: ast.Module) -> tuple[int, int]:
    """Return (primitive_creation_count, name_assignment_count).

    A "primitive creation" is any ``bpy.ops.mesh.primitive_*_add`` call;
    a "name assignment" is any ``...obj.name = '...'`` or
    ``bpy.context.object.name = '...'`` assignment with a string literal.

    Use the ratio as a proxy for the LLM "abstraction failure" mode:
    when the script spawns many primitives but names few of them, the
    scene is likely a collection of anonymous spheres/cubes rather than
    a grounded environment with real anchors.
    """
    primitive_count = 0
    name_assigns = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_dotted(node.func)
            if chain and chain.startswith("bpy.ops.mesh.primitive_") and chain.endswith("_add"):
                primitive_count += 1
        if isinstance(node, ast.Assign):
            if not (isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                continue
            for target in node.targets:
                chain = _attr_dotted(target)
                if chain and chain.endswith(".name"):
                    name_assigns += 1
                    break
    return primitive_count, name_assigns


_ARROW_PRIMITIVE_CALLS = (
    "bpy.ops.mesh.primitive_cone_add",
    "bpy.ops.mesh.primitive_cylinder_add",
    "bpy.ops.curve.primitive_bezier_curve_add",
)


def _count_arrow_or_tracer_primitives(tree: ast.Module) -> int:
    """Count primitives that *could* form an arrow or polyline tracer.

    A typical arrow uses one cylinder (shaft) and one cone (head). We
    therefore divide the raw count by 2 (rounded down) to estimate the
    number of *complete* arrows. Curve_polyline tracers are counted
    one-to-one because each call yields a full tracer.
    """
    pair_count = 0
    polyline_count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_dotted(node.func)
        if chain in (_ARROW_PRIMITIVE_CALLS[0], _ARROW_PRIMITIVE_CALLS[1]):
            pair_count += 1
        elif chain == _ARROW_PRIMITIVE_CALLS[2]:
            polyline_count += 1
    return pair_count // 2 + polyline_count


def _compiled_scene_hints(tree: ast.Module) -> dict[str, int]:
    """Read deterministic compiler count hints, when present.

    The semi-deterministic scene compiler creates many labels/tracers from
    runtime loops over ``STORYBOARD``. Static AST counting sees only the helper
    function body, so compiled scaffolds expose conservative literal counts for
    validators. We only trust the hints when the helper function and call site
    are both still present, so a later LLM edit cannot satisfy contracts by
    leaving a stale count dictionary behind.
    """
    has_helper_def = any(
        isinstance(node, ast.FunctionDef)
        and node.name == "add_teaching_helpers_for_shot"
        for node in ast.walk(tree)
    )
    has_helper_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_teaching_helpers_for_shot"
        for node in ast.walk(tree)
    )
    if not (has_helper_def and has_helper_call):
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "COMPILED_SCENE_HINTS"
            for target in node.targets
        ):
            continue
        try:
            raw = ast.literal_eval(node.value)
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(raw, dict):
            return {}
        hints: dict[str, int] = {}
        for key in ("text_objects_created", "arrow_or_tracer_primitives"):
            value = raw.get(key)
            if isinstance(value, int) and value >= 0:
                hints[key] = value
        return hints
    return {}


def _apply_sanity_checks(
    report: "ContractValidationReport",
    tree: ast.Module,
    code: str,
) -> None:
    """Run registered per-rule sanity checks and downgrade false-positive blocks.

    Mutates ``report.violations`` in place (rebuilding entries because
    ContractViolation is frozen) and records each downgrade under
    ``report.per_shot_counts["_sanity_downgrades"]`` for audit.
    """
    if not report.violations:
        return
    downgrades: dict[str, dict[str, str]] = {}
    new_violations: list[ContractViolation] = []
    for v in report.violations:
        check = RULE_SANITY_CHECKS.get(v.rule_id)
        if check is None or v.severity != "block":
            new_violations.append(v)
            continue
        try:
            verdict, reason = check(v, tree, code)
        except Exception as exc:
            # Sanity check failures must never break validation; just keep the
            # original violation and record the failure for debugging.
            downgrades[v.rule_id] = {
                "original_severity": v.severity,
                "verdict": "error",
                "reason": f"sanity check raised {type(exc).__name__}: {exc}",
            }
            new_violations.append(v)
            continue
        if verdict == "downgrade":
            new_violations.append(replace(
                v,
                severity="warn",
                message="[sanity-downgraded] " + v.message,
            ))
            downgrades[v.rule_id] = {
                "original_severity": "block",
                "verdict": "downgrade",
                "reason": reason,
            }
        else:
            new_violations.append(v)
    report.violations = new_violations
    if downgrades:
        report.per_shot_counts["_sanity_downgrades"] = downgrades


@register_sanity_check("contract_insufficient_text_objects")
def _sanity_check_text_objects(
    violation: "ContractViolation",
    tree: ast.Module,
    code: str,
) -> tuple[SanityVerdict, str]:
    """Downgrade text-object insufficiency when AST shows a factory in a loop.

    The validator's static count via ``_count_text_objects`` already handles
    literal-loop factory expansion, but misses runtime-driven loops such as
    ``for shot_id, labels in shots_dict.items(): for label in labels:
    add_text(label)`` where the iterable size is not visible to AST. When the
    code clearly defines a text-creating helper AND invokes it inside any
    loop body, we have strong evidence the actual runtime count exceeds the
    naive static count — so a block is a false positive.
    """
    # 1. find text-creating helper functions
    text_factories: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            chain = _attr_dotted(child.func)
            if chain == "bpy.ops.object.text_add":
                text_factories.add(node.name)
                break
            if chain == "bpy.data.curves.new":
                for kw in child.keywords:
                    if (
                        kw.arg == "type"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == "FONT"
                    ):
                        text_factories.add(node.name)
                        break
    if not text_factories:
        return "keep", "no text-creating helper function defined"

    # 2. find unique helper-call AST nodes that appear inside any loop type.
    #    Using id() dedupes nested-loop double-counting from ast.walk.
    LOOP_TYPES = (
        ast.For, ast.While, ast.AsyncFor,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    )
    call_ids_in_loops: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, LOOP_TYPES):
            continue
        for child in ast.walk(node):
            if child is node or not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name) and child.func.id in text_factories:
                call_ids_in_loops.add(id(child))

    # 3. count direct (non-loop-wrapped) helper invocations.
    all_call_ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in text_factories:
            all_call_ids.add(id(node))
    direct_invocations = len(all_call_ids - call_ids_in_loops)
    loop_invocations = len(call_ids_in_loops)

    # Heuristic: a single text-factory call wrapped in *any* loop is strong
    # evidence that runtime label count is plausibly >= expected. Scene.py
    # typically has 1-2 unique factory call sites called from runtime loops
    # over storyboard data the static analyzer cannot see — so a count
    # threshold based on unique AST call sites does not work. The presence
    # of the wrapping loop is what matters.
    if loop_invocations >= 1:
        return (
            "downgrade",
            (
                f"text helper(s) {sorted(text_factories)} invoked inside "
                f"{loop_invocations} loop-wrapped call site(s); runtime "
                f"label count likely >= expected={violation.expected}"
            ),
        )
    if direct_invocations >= violation.expected:
        return (
            "downgrade",
            (
                f"text helper(s) {sorted(text_factories)} statically called "
                f"{direct_invocations} time(s) >= expected={violation.expected}"
            ),
        )
    return (
        "keep",
        (
            f"factory found ({sorted(text_factories)}) but only "
            f"{direct_invocations} direct call(s) and no loop-wrapped "
            f"invocation; runtime count cannot be confirmed >= "
            f"expected={violation.expected}"
        ),
    )


def validate_visual_contracts(
    code: str,
    storyboard: Storyboard | None,
    visual_contracts: dict[str, VisualContract] | None,
    *,
    scene_profile: SceneProfile | None = None,
    success_spec_text_anchors: set[str] | None = None,
) -> ContractValidationReport:
    """Structural check that ``code`` honors ``visual_contracts``.

    Returns an empty report (and no I/O) when ``visual_contracts`` is
    empty or the code does not parse. Parse errors are scene_verifier's
    responsibility; we don't double-report them.
    """
    report = ContractValidationReport()
    if not visual_contracts:
        return report
    tree = _safe_parse(code)
    if tree is None:
        return report

    n_text = _count_text_objects(tree)
    n_arrows = _count_arrow_or_tracer_primitives(tree)
    compiled_hints = _compiled_scene_hints(tree)
    n_text = max(n_text, compiled_hints.get("text_objects_created", 0))
    n_arrows = max(n_arrows, compiled_hints.get("arrow_or_tracer_primitives", 0))
    has_emission = _has_emission_assignment(tree)
    string_values = _string_literals(tree)
    n_primitives, n_named = _count_named_vs_anonymous_primitives(tree)
    strict_teaching = (
        scene_profile is not None
        and scene_profile.base_profile in {
            "vector_teaching",
            "curve_construction",
            "transformation_demo",
        }
    )

    # Aggregate required totals across all shots. This is conservative
    # because in practice the same text/arrow can be reused via
    # hide_render gating across shots; the requirement is "at least one
    # per distinct required label / vector across the timeline".
    required_labels: list[str] = []
    required_anchors: list[str] = []
    total_required_vectors = 0
    any_emphasis_required = False
    shot_ids = []
    if storyboard is not None:
        shot_ids = [shot.node_id for shot in storyboard.shots]
    for shot_id in shot_ids:
        contract = visual_contracts.get(shot_id)
        if contract is None:
            continue
        required_anchors.extend(contract.required_anchors)
        required_labels.extend(contract.required_labels)
        total_required_vectors += len(contract.required_vectors)
        if contract.emphasis_points:
            any_emphasis_required = True

    required_anchors = sorted(set(required_anchors), key=str.lower)
    total_required_labels = len({
        " ".join(label.split()).strip().lower()
        for label in required_labels
        if label.strip()
    })
    success_spec_text_anchor_set = {
        anchor.strip()
        for anchor in (success_spec_text_anchors or set())
        if anchor and anchor.strip()
    }
    success_spec_text_required = bool(success_spec_text_anchor_set) and any(
        label.strip() in success_spec_text_anchor_set
        for label in required_labels
    )
    found_anchors = [
        anchor for anchor in required_anchors
        if _anchor_is_present(anchor, string_values)
    ]
    report.per_shot_counts = {
        "scene_grounding_anchors": {
            "actual": len(found_anchors),
            "required": len(required_anchors),
        },
        "text_objects_created": {"actual": n_text, "required": total_required_labels},
        "arrow_or_tracer_primitives": {
            "actual": n_arrows,
            "required": total_required_vectors,
        },
        "emission_material_present": {
            "actual": int(has_emission),
            "required": int(any_emphasis_required),
        },
        "named_vs_anonymous_primitives": {
            "actual": n_named,
            "required": n_primitives,
        },
    }

    missing_anchors = [
        anchor for anchor in required_anchors
        if not _anchor_is_present(anchor, string_values)
    ]
    if missing_anchors:
        report.violations.append(ContractViolation(
            severity="block",
            rule_id="contract_missing_scene_anchors",
            shot_id="*",
            field="required_anchors",
            expected=len(required_anchors),
            found=len(found_anchors),
            message=(
                "Visual contracts require persistent scene grounding anchors "
                "that are missing from scene.py string/object names: "
                + ", ".join(missing_anchors[:8])
            ),
            suggested_fix=(
                "Instantiate each missing anchor as real Blender geometry and "
                "set `obj.name` to the anchor name or an obvious variant. "
                "Do not replace scene anchors with anonymous primitives."
            ),
        ))

    # When the scene is supposed to be grounded (required_anchors set) but
    # the script spawns many primitives without naming most of them, that
    # is the classic "abstraction failure" mode — a forest of anonymous
    # spheres/cubes standing in for the real environment. We can't tell
    # from AST alone whether the unnamed primitives are decorative or
    # standing in for anchors, so this is a warning, not a block.
    if required_anchors and n_primitives >= 3 and n_named * 3 < n_primitives:
        report.violations.append(ContractViolation(
            severity="warn",
            rule_id="contract_excessive_anonymous_primitives",
            shot_id="*",
            field="required_anchors",
            expected=n_primitives,
            found=n_named,
            message=(
                f"scene.py creates {n_primitives} primitive(s) but names only "
                f"{n_named} of them. When persistent_anchors are required, "
                "anonymous primitives often signal that the script abstracted "
                "the environment into floating shapes rather than building "
                "the real scene."
            ),
            suggested_fix=(
                "After each primitive_*_add, set `bpy.context.object.name` to "
                "an anchor-matching identifier (e.g. 'window_frame_left') so "
                "the role of every primitive is explicit and inspectable."
            ),
        ))

    if total_required_labels and n_text == 0:
        report.violations.append(ContractViolation(
            severity="block",
            rule_id="contract_no_text_objects",
            shot_id="*",
            field="required_labels",
            expected=total_required_labels,
            found=0,
            message=(
                f"Visual contracts require {total_required_labels} label(s) "
                "across shots but scene.py creates zero text objects."
            ),
            suggested_fix=(
                "Add `bpy.ops.object.text_add(...)` calls with `.data.body` "
                "set to each required label; orient them toward the camera."
            ),
        ))
    elif total_required_labels and n_text < total_required_labels:
        severity: Severity = (
            "block" if strict_teaching or success_spec_text_required else "warn"
        )
        report.violations.append(ContractViolation(
            severity=severity,
            rule_id="contract_insufficient_text_objects",
            shot_id="*",
            field="required_labels",
            expected=total_required_labels,
            found=n_text,
            message=(
                f"Visual contracts require {total_required_labels} label(s) "
                f"but scene.py only creates {n_text} text object(s). Some "
                "labels may be reused across shots, but teaching profiles "
                "treat insufficient label geometry as a hard pre-render "
                "failure."
                if strict_teaching
                else (
                    "Success Spec readable text anchors require exact text "
                    "objects and cannot be treated as optional cinematic labels."
                )
                if success_spec_text_required
                else "labels may be reused across shots, so this is a warning."
            ),
            suggested_fix=(
                "Verify each required label has a corresponding text object "
                "(or shared across shots via hide_render)."
            ),
        ))

    if total_required_vectors and n_arrows == 0:
        report.violations.append(ContractViolation(
            severity="block",
            rule_id="contract_no_vector_geometry",
            shot_id="*",
            field="required_vectors",
            expected=total_required_vectors,
            found=0,
            message=(
                f"Visual contracts require {total_required_vectors} "
                "vector/ray cue(s) but scene.py has no arrow/curve_polyline "
                "geometry."
            ),
            suggested_fix=(
                "Represent each required vector either as a "
                "cone+cylinder arrow pair or a thin curve_polyline tracer."
            ),
        ))
    elif total_required_vectors and n_arrows < total_required_vectors:
        severity: Severity = "block" if strict_teaching else "warn"
        report.violations.append(ContractViolation(
            severity=severity,
            rule_id="contract_insufficient_vector_geometry",
            shot_id="*",
            field="required_vectors",
            expected=total_required_vectors,
            found=n_arrows,
            message=(
                f"Visual contracts require {total_required_vectors} "
                f"vector/ray cue(s); scene.py creates ~{n_arrows} arrow/tracer "
                "primitives. Teaching profiles treat insufficient vector "
                "geometry as a hard pre-render failure."
                if strict_teaching
                else "primitives. Some vectors may be shared, so this is a warning."
            ),
            suggested_fix=(
                "Confirm each required vector is visually distinct in the "
                "final render or share via hide_render."
            ),
        ))

    if any_emphasis_required and not has_emission:
        report.violations.append(ContractViolation(
            severity="warn",
            rule_id="contract_no_emphasis_material",
            shot_id="*",
            field="emphasis_points",
            expected=1,
            found=0,
            message=(
                "Visual contracts request a visible emphasis / highlight, "
                "but scene.py assigns no Emission Color / Emission Strength."
            ),
            suggested_fix=(
                "Set `bsdf.inputs['Emission Color']` and "
                "`bsdf.inputs['Emission Strength']` on the emphasized "
                "object's material."
            ),
        ))

    _apply_sanity_checks(report, tree, code)
    return report


def report_to_json(report: ContractValidationReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def format_contract_validation_addendum(report: ContractValidationReport) -> str:
    if not report.violations:
        return ""
    lines = [
        "STRUCTURED VISUAL CONTRACT VALIDATION:",
        "These are deterministic AST checks against the generated scene.py "
        "(not LLM judgement). Treat block-level findings as hard repairs.",
    ]
    for i, v in enumerate(report.violations, 1):
        lines.append(
            f"{i}. [{v.severity} {v.rule_id}] {v.message}"
        )
        lines.append(f"   fix: {v.suggested_fix}")
    return "\n".join(lines)
