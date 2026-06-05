"""Automatically generated soft success criteria.

This layer complements hand-written ``success_spec`` blocks.  It is designed
to be conservative: generated rules start as soft/diagnostic evidence and are
kept as run artifacts rather than written back to concept YAML.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cg_tutor.schemas import CriticReport, Narrative, Storyboard
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.success_spec import SuccessSpec


AutoRuleKind = Literal[
    "object_visible",
    "text_readable",
    "stay_in_screen_safe",
    "helper_hidden",
    "animation_coverage",
    "progressive_visual_ordering",
]
AutoRuleSource = Literal["generated", "critic_confirmed"]
AutoFailureClass = Literal[
    "success_hard",
    "success_soft",
    "aesthetic_warn",
    "diagnostic",
]


class AutoSuccessRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AutoRuleKind
    anchors: list[str] = Field(min_length=1)
    source: AutoRuleSource = "generated"
    confidence: float = 0.0
    failure_class: AutoFailureClass = "success_soft"
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _valid_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @field_validator("anchors")
    @classmethod
    def _clean_anchors(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            anchor = " ".join(str(item).strip().split())
            key = anchor.lower()
            if anchor and key not in seen:
                out.append(anchor)
                seen.add(key)
        if not out:
            raise ValueError("AutoSuccessRule.anchors must not be empty")
        return out


class AutoSuccessSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    rules: list[AutoSuccessRule] = Field(default_factory=list)


_TEXT_SUFFIX_RE = re.compile(r"(?:^|_)(?:label|labels|readout|hud|text)(?:_|$)", re.I)
_HELPER_RE = re.compile(r"(?:^|_)(?:path|guide|helper|gizmo)(?:_|$)", re.I)
_SAFE_RE = re.compile(r"(?:^|_)(?:stack|panel|hud|readout|legend|side)(?:_|$)", re.I)
_LOD_RE = re.compile(r"\bLOD[0-9]\b")


def generate_auto_success_spec(
    *,
    concept_spec: dict,
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
    storyboard: Storyboard | None = None,
    user_success_spec: SuccessSpec | None = None,
) -> tuple[AutoSuccessSpec, dict]:
    """Generate a conservative, validated AutoSuccessSpec."""
    text = _concept_text(concept_spec, narrative)
    persistent = _clean_list(concept_spec.get("persistent_anchors") or [])
    storyboard_names = _storyboard_object_names(storyboard)
    explicit_text_tokens = _explicit_text_tokens(text)
    user_anchors = user_success_spec.required_anchor_names() if user_success_spec else set()

    rules: list[AutoSuccessRule] = []

    for anchor in persistent:
        if anchor in user_anchors:
            continue
        if _looks_like_hidden_helper(anchor):
            rules.append(AutoSuccessRule(
                kind="helper_hidden",
                anchors=[anchor],
                failure_class="aesthetic_warn",
                confidence=0.45,
                reason="persistent anchor looks like a path/guide/helper",
            ))
        else:
            rules.append(AutoSuccessRule(
                kind="object_visible",
                anchors=[anchor],
                confidence=0.55,
                reason="persistent_anchors should be instantiated and visible",
            ))

    text_anchors = {
        item for item in (set(persistent) | storyboard_names)
        if _looks_like_text_anchor(item)
    } | explicit_text_tokens
    for anchor in sorted(text_anchors, key=str.lower):
        if anchor in user_anchors:
            continue
        rules.append(AutoSuccessRule(
            kind="text_readable",
            anchors=[anchor],
            confidence=0.50,
            reason="readout/label/text token should be readable",
        ))
        if _looks_screen_fixed(anchor):
            rules.append(AutoSuccessRule(
                kind="stay_in_screen_safe",
                anchors=[anchor],
                confidence=0.45,
                reason="readout/label/stack should remain in a safe screen area",
            ))

    for anchor in sorted(set(persistent) | storyboard_names, key=str.lower):
        if anchor in user_anchors:
            continue
        if _looks_screen_fixed(anchor):
            rules.append(AutoSuccessRule(
                kind="stay_in_screen_safe",
                anchors=[anchor],
                confidence=0.45,
                reason="panel/stack/HUD-like anchor should not be cropped",
            ))

    if {"LOD0", "LOD1", "LOD2"} <= explicit_text_tokens:
        rules.append(AutoSuccessRule(
            kind="progressive_visual_ordering",
            anchors=["LOD0", "LOD1", "LOD2"],
            confidence=0.35,
            failure_class="diagnostic",
            reason="explicit LOD0/LOD1/LOD2 sequence implies progressive ordering",
            metadata={"order": "decreasing_texture_frequency"},
        ))

    if storyboard is not None:
        animated = sorted(
            (
                item for item in _storyboard_animated_objects(storyboard)
                if item not in user_anchors
            ),
            key=str.lower,
        )
        if animated:
            rules.append(AutoSuccessRule(
                kind="animation_coverage",
                anchors=animated,
                confidence=0.50,
                reason="storyboard declares non-visibility keyframes",
            ))

    spec = AutoSuccessSpec(rules=_dedupe_rules(rules))
    accepted, rejected = _validate_rules(
        spec.rules,
        allowed_tokens=set(persistent) | storyboard_names | explicit_text_tokens,
    )
    validation = {
        "accepted": len(accepted),
        "rejected": rejected,
        "source": "generated",
        "scene_profile": scene_profile.profile_id if scene_profile else "",
    }
    return AutoSuccessSpec(rules=accepted), validation


def auto_success_spec_to_json(spec: AutoSuccessSpec) -> str:
    return json.dumps(spec.model_dump(mode="json"), indent=2)


def auto_success_validation_to_json(validation: dict) -> str:
    return json.dumps(validation, indent=2)


def format_auto_success_spec_for_coder(spec: AutoSuccessSpec | None) -> str:
    if spec is None or not spec.rules:
        return ""
    lines = [
        "AUTO SUCCESS SPEC (generated soft viewer-facing checks):",
        "- These checks are generated from the concept/storyboard and start as soft evidence.",
        "- Fix the listed target directly; prefer editing existing objects over adding new helpers.",
        "- Do not create extra labels unless the target itself is a readable text anchor.",
    ]
    for rule in spec.rules[:18]:
        if rule.failure_class == "diagnostic":
            continue
        anchors = ", ".join(rule.anchors)
        lines.append(
            f"- {rule.kind}: {anchors} "
            f"(failure_class={rule.failure_class}, confidence={rule.confidence:.2f})"
        )
    return "\n".join(lines)


def critic_confirmed_auto_success_issues(
    *,
    auto_success_spec: AutoSuccessSpec | None,
    critic_history: list,
    current_report: CriticReport,
    static_anchor_status: dict[str, dict] | None = None,
) -> list[dict]:
    """Return current-run escalations for repeated critic evidence."""
    if auto_success_spec is None or not auto_success_spec.rules or not critic_history:
        return []
    prior_report = critic_history[-1].report
    anchor_status = static_anchor_status or {}
    out: list[dict] = []
    for rule in auto_success_spec.rules:
        if rule.failure_class == "diagnostic":
            continue
        if rule.kind not in {
            "text_readable",
            "stay_in_screen_safe",
            "object_visible",
            "helper_hidden",
        }:
            continue
        if rule.kind == "helper_hidden":
            # Helper clutter is useful repair evidence, but generated helper
            # rules are too aesthetics-adjacent to become a hard gate.
            failure_class = "aesthetic_warn"
            severity = "warn"
            rule_id = "auto_success_helper_hidden"
            suggested_fix = _auto_rule_fix(rule)
        elif rule.kind == "object_visible" and _rule_has_static_object_evidence(
            rule, anchor_status
        ):
            failure_class = "success_soft"
            severity = "warn"
            rule_id = "auto_success_visibility_unproven"
            suggested_fix = _auto_visibility_unproven_fix(rule)
        else:
            failure_class = "success_hard"
            severity = "block"
            rule_id = f"auto_success_{rule.kind}"
            suggested_fix = _auto_rule_fix(rule)
        if (
            _report_mentions_rule(prior_report, rule)
            and _report_mentions_rule(current_report, rule)
        ):
            out.append({
                "severity": severity,
                "rule_id": rule_id,
                "anchors": rule.anchors,
                "message": (
                    f"Auto success rule {rule.kind} was confirmed by critic "
                    f"evidence in consecutive iterations: {', '.join(rule.anchors)}"
                ),
                "suggested_fix": suggested_fix,
                "failure_class": failure_class,
            })
    return out


def _report_mentions_rule(report: CriticReport, rule: AutoSuccessRule) -> bool:
    anchors = {a.lower() for a in rule.anchors}
    anchor_norms = {_norm(a) for a in rule.anchors}
    for issue in report.issues:
        evidence_kind = getattr(issue, "evidence_kind", None)
        target = str(getattr(issue, "target", "") or "").lower()
        target_norm = _norm(target)
        if (
            evidence_kind == rule.kind
            and (
                target in anchors
                or target_norm in anchor_norms
                or any(a and a in target_norm for a in anchor_norms)
            )
        ):
            return True
        text = f"{issue.issue} {issue.suggested_fix}".lower()
        if any(anchor in text for anchor in anchors):
            if rule.kind == "text_readable" and _text_problem(text):
                return True
            if rule.kind == "stay_in_screen_safe" and _screen_problem(text):
                return True
            if rule.kind == "object_visible" and _missing_problem(text):
                return True
            if rule.kind == "helper_hidden" and _helper_problem(text):
                return True
    return False


def _auto_rule_fix(rule: AutoSuccessRule) -> str:
    anchors = ", ".join(rule.anchors)
    if rule.kind == "text_readable":
        return f"Make readable text target(s) camera-facing and high contrast: {anchors}."
    if rule.kind == "stay_in_screen_safe":
        return f"Move existing target(s) inside safe frame margins without scaling them offscreen: {anchors}."
    if rule.kind == "object_visible":
        return f"Instantiate and keep visible target object(s): {anchors}."
    if rule.kind == "helper_hidden":
        return f"Hide helper/path/guide object(s) from final render: {anchors}."
    return f"Fix generated success rule target(s): {anchors}."


def _auto_visibility_unproven_fix(rule: AutoSuccessRule) -> str:
    anchors = ", ".join(rule.anchors)
    return (
        "The target object appears to be created in scene.py, so do not add a "
        f"duplicate. Move, scale, orient, unhide, or unclip the existing "
        f"object(s) so they are visibly inside the camera frame: {anchors}."
    )


def _rule_has_static_object_evidence(
    rule: AutoSuccessRule,
    static_anchor_status: dict[str, dict],
) -> bool:
    for anchor in rule.anchors:
        status = static_anchor_status.get(anchor) or static_anchor_status.get(
            anchor.lower()
        )
        if isinstance(status, dict) and bool(status.get("created")):
            return True
    return False


def _validate_rules(
    rules: list[AutoSuccessRule],
    *,
    allowed_tokens: set[str],
) -> tuple[list[AutoSuccessRule], list[dict]]:
    allowed_norm = {_norm(t) for t in allowed_tokens if t}
    accepted: list[AutoSuccessRule] = []
    rejected: list[dict] = []
    for rule in rules:
        missing = [a for a in rule.anchors if _norm(a) not in allowed_norm]
        if missing:
            rejected.append({
                "kind": rule.kind,
                "anchors": rule.anchors,
                "reason": "anchor_not_in_concept_or_storyboard_tokens",
            })
            continue
        if rule.kind == "progressive_visual_ordering" and len(rule.anchors) < 2:
            rejected.append({
                "kind": rule.kind,
                "anchors": rule.anchors,
                "reason": "progressive_visual_ordering_requires_sequence",
            })
            continue
        accepted.append(rule)
    return accepted, rejected


def _dedupe_rules(rules: list[AutoSuccessRule]) -> list[AutoSuccessRule]:
    out: list[AutoSuccessRule] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for rule in rules:
        key = (rule.kind, tuple(a.lower() for a in rule.anchors))
        if key in seen:
            continue
        seen.add(key)
        out.append(rule)
    return out


def _concept_text(concept_spec: dict, narrative: Narrative | None) -> str:
    parts: list[str] = []
    for key in (
        "concept_id", "title", "scene_profile",
        "spatial_relationships", "forbidden_abstractions", "key_points",
        "visual_style_constraints", "graphics_principles",
    ):
        value = concept_spec.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value is not None:
            parts.append(str(value))
    if narrative is not None:
        for node in narrative.nodes:
            parts.extend([node.title, node.description, node.visual_intent])
            parts.extend(node.formulas)
    return "\n".join(parts)


def _clean_list(value: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _storyboard_object_names(storyboard: Storyboard | None) -> set[str]:
    if storyboard is None:
        return set()
    return {
        obj.name
        for shot in storyboard.shots
        for obj in shot.objects
        if obj.name
    }


def _storyboard_animated_objects(storyboard: Storyboard) -> set[str]:
    out: set[str] = set()
    for shot in storyboard.shots:
        for obj in shot.objects:
            if any(k.attr not in {"hide_render", "hide_viewport"} for k in obj.keyframes):
                out.add(obj.name)
    return out


def _explicit_text_tokens(text: str) -> set[str]:
    out = set(_LOD_RE.findall(text))
    for match in re.findall(r"\b(d_(?:near|mid|far))\b", text, flags=re.I):
        out.add(match)
    return out


def _looks_like_text_anchor(anchor: str) -> bool:
    return bool(_TEXT_SUFFIX_RE.search(anchor))


def _looks_screen_fixed(anchor: str) -> bool:
    return bool(_SAFE_RE.search(anchor)) or _looks_like_text_anchor(anchor)


def _looks_like_hidden_helper(anchor: str) -> bool:
    low = anchor.lower()
    if "trail_curve" in low:
        return False
    return bool(_HELPER_RE.search(anchor))


def _text_problem(text: str) -> bool:
    return any(w in text for w in ("unreadable", "illegible", "too small", "low-contrast", "low contrast"))


def _screen_problem(text: str) -> bool:
    return any(w in text for w in ("offscreen", "off-screen", "cropped", "clipped", "outside", "edge"))


def _missing_problem(text: str) -> bool:
    return any(w in text for w in ("missing", "absent", "not visible", "invisible", "not shown"))


def _helper_problem(text: str) -> bool:
    return any(w in text for w in ("distract", "visible helper", "path is visible", "guide is visible"))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
