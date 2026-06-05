"""Structured repair plans derived from verifier and critic signals.

The plan is deterministic and JSON-serialisable. Today it is fed back to
the Blender coder as a constrained edit target; later it can become the
input to a true patch applier over Scene IR or bpy code.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from cg_tutor.critic_loop import (
    BestSelectionMode,
    CATEGORY_REPAIR_MESSAGES,
    _critic_quality_key_for,
)
from cg_tutor.schemas import FRAMING_CATEGORIES
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.visual_contract import VisualContract


class RepairTarget(BaseModel):
    priority: int
    source: Literal["scene_verifier", "contract", "critic", "object_check"]
    source_report: str = "unknown"
    shot_id: str | None = None
    category: str
    severity: str
    issue: str
    action: str


class RepairPlan(BaseModel):
    iteration: int
    base_iteration: int | None = None
    targets: list[RepairTarget] = Field(default_factory=list)


def _specific_action(
    issue_text: str,
    contract: VisualContract | None = None,
    scene_profile: SceneProfile | None = None,
) -> str:
    low = issue_text.lower()
    actions: list[str] = []
    overlay_policy = scene_profile.overlay_policy if scene_profile else {}
    rubric = scene_profile.critic_rubric if scene_profile else {}
    if any(w in low for w in ("label", "text", "caption")):
        actions.append(
            "Add readable text labels near the referenced objects/components; "
            "orient labels toward the camera and keep them outside overlay space."
        )
    if any(w in low for w in ("arrow", "vector", "ray", "normal", "reflection")):
        if scene_profile and scene_profile.base_profile in {
            "cinematic_application",
            "vector_teaching",
            "ray_tracing",
        }:
            actions.append(
                "Render ray/vector cues as thin glowing curve_polyline or dotted tracers; "
                "use only small arrowhead direction cues when the scene profile allows them; "
                "do not use thick arrows, thick arrow helper primitives, or "
                "visible light-bar helpers."
            )
        else:
            actions.append(
                "Render each vector as a distinct full arrow with visible shaft and head; "
                "separate origins/tips so arrows do not merge or overlap."
            )
    if any(w in low for w in ("highlight", "specular", "gloss")):
        actions.append(
            "Make the highlight/emphasis point visibly bright and connect any "
            "explaining vectors/rays to that point."
        )
    if any(w in low for w in ("overlap", "occlud", "sunken", "tiny", "only their heads")):
        actions.append(
            "Increase helper scale and offset helpers slightly off the surface "
            "so their full shapes remain visible."
        )
    if any(w in low for w in ("dark", "underexposed", "shadow")):
        actions.append(
            "Raise ambient/fill light or material brightness enough for the "
            "teaching cue to be readable without washing out contrast."
        )
    if any(w in low for w in ("overlay", "formula")):
        if overlay_policy.get("disable_formula_overlay") is True:
            actions.append(
                "Remove formula overlays, math panels, and contribution tiles; "
                "replace any necessary text with sparse in-world HUD cues."
            )
        else:
            actions.append(
                "Move the subject or overlay zone so the formula area is empty."
            )
    if any(w in low for w in ("component", "ambient", "diffuse", "compare")):
        actions.append(
            "Keep compared components spatially separated with stable ordering "
            "and explicit labels."
        )

    if contract:
        actions.extend(contract.required_relationships[:2])
        actions.extend(contract.forbidden_failures[:2])
    if rubric:
        for key in ("blocking_conditions", "must_have", "must_avoid"):
            values = rubric.get(key, [])
            if isinstance(values, list) and values:
                label = key.replace("_", " ")
                actions.append(
                    f"Respect adaptive critic rubric {label}: "
                    + "; ".join(str(v) for v in values[:3])
                    + "."
                )
    if scene_profile and scene_profile.repair_policy:
        actions.extend(scene_profile.repair_policy[:4])

    return " ".join(actions)


def _action_for_category(
    category: str,
    issue_text: str = "",
    contract: VisualContract | None = None,
    scene_profile: SceneProfile | None = None,
) -> str:
    specific = _specific_action(issue_text, contract, scene_profile)
    entry = CATEGORY_REPAIR_MESSAGES.get(
        category, CATEGORY_REPAIR_MESSAGES["other"]
    )
    return f"{entry['action_base']} {specific}".strip()


def build_repair_plan(
    *,
    iteration: int,
    history: list,
    verifier_report=None,
    contract_report=None,
    missing_objects: dict[str, list[str]] | None = None,
    visual_contracts: dict[str, VisualContract] | None = None,
    scene_profile: SceneProfile | None = None,
    best_selection_mode: BestSelectionMode = "framing",
    limit: int = 6,
) -> RepairPlan:
    base_iteration = None
    targets: list[RepairTarget] = []
    priority = 1

    if history:
        best = max(
            history,
            key=lambda r: _critic_quality_key_for(r, best_selection_mode),
        )
        base_iteration = best.iteration

    seen: set[tuple[str | None, str, str]] = set()

    def _append_target(
        *,
        source,
        source_report: str,
        category: str,
        severity: str,
        issue: str,
        action: str,
        shot_id: str | None = None,
    ) -> None:
        nonlocal priority
        key = (shot_id, category, issue)
        if len(targets) >= limit or key in seen:
            return
        targets.append(RepairTarget(
            priority=priority,
            source=source,
            source_report=source_report,
            shot_id=shot_id,
            category=category,
            severity=severity,
            issue=issue,
            action=action,
        ))
        seen.add(key)
        priority += 1

    def _is_fatal_verifier_issue(issue) -> bool:
        return issue.rule_id in {
            "syntax_error",
            "missing_render_call",
            "missing_out_dir",
            "missing_out_dir_env",
            "render_not_animation",
            "forbidden_engine",
            "forbidden_file_io",
            "forbidden_subprocess",
            "forbidden_save_file",
        }

    if verifier_report is not None:
        for issue in getattr(verifier_report, "issues", []):
            if issue.severity != "block":
                continue
            if not _is_fatal_verifier_issue(issue):
                continue
            _append_target(
                source="scene_verifier",
                source_report="scene_verifier",
                category=issue.rule_id,
                severity=issue.severity,
                issue=issue.message,
                action=issue.suggested_fix,
            )

    contract_priority = {
        "contract_no_vector_geometry": 0,
        "contract_missing_scene_anchors": 1,
        "contract_insufficient_vector_geometry": 2,
        "contract_insufficient_text_objects": 3,
    }
    if contract_report is not None:
        violations = sorted(
            getattr(contract_report, "violations", []),
            key=lambda v: (
                0 if v.severity == "block" else 1,
                contract_priority.get(v.rule_id, 99),
                v.shot_id,
                v.rule_id,
            ),
        )
        for violation in violations:
            if violation.severity != "block":
                continue
            if violation.rule_id not in contract_priority:
                continue
            _append_target(
                source="contract",
                source_report="contract_validation",
                shot_id=violation.shot_id,
                category=violation.rule_id,
                severity=violation.severity,
                issue=violation.message,
                action=violation.suggested_fix,
            )

    if verifier_report is not None:
        for issue in getattr(verifier_report, "issues", []):
            if issue.severity != "block":
                continue
            if _is_fatal_verifier_issue(issue):
                continue
            _append_target(
                source="scene_verifier",
                source_report="scene_verifier",
                category=issue.rule_id,
                severity=issue.severity,
                issue=issue.message,
                action=issue.suggested_fix,
            )

    for shot_id, names in sorted((missing_objects or {}).items()):
        _append_target(
            source="object_check",
            source_report="missing_objects",
            shot_id=shot_id,
            category="missing_storyboard_objects",
            severity="warn",
            issue=f"scene.py is missing storyboard object names: {', '.join(names)}",
            action="Instantiate these exact object names and make them visible in the shot.",
        )

    if history and len(targets) < limit:
        best = max(
            history,
            key=lambda r: _critic_quality_key_for(r, best_selection_mode),
        )
        issues = sorted(
            best.report.issues,
            key=lambda i: (
                0 if i.severity == "block" else 1,
                0 if i.category in FRAMING_CATEGORIES else 1,
                i.shot_id,
                i.frame_idx,
            ),
        )
        for issue in issues:
            if len(targets) >= limit:
                break
            _append_target(
                source="critic",
                source_report="critic_aggregate",
                shot_id=issue.shot_id,
                category=issue.category,
                severity=issue.severity,
                issue=issue.issue,
                action=_action_for_category(
                    issue.category,
                    issue.issue,
                    (visual_contracts or {}).get(issue.shot_id),
                    scene_profile,
                ),
            )

    return RepairPlan(
        iteration=iteration,
        base_iteration=base_iteration,
        targets=targets[:limit],
    )


def repair_plan_to_json(plan: RepairPlan) -> str:
    return plan.model_dump_json(indent=2)


def format_repair_plan_for_coder(plan: RepairPlan) -> str:
    if not plan.targets:
        return ""
    lines = [
        "STRUCTURED REPAIR PLAN:",
        f"- Base iteration: {plan.base_iteration if plan.base_iteration is not None else 'current'}",
        "- Apply targets in priority order. Make the smallest complete-code change.",
        "- Preserve clean shots from the base iteration; only touch shots named below unless required by a block fix.",
        "- Avoid regressing labels, vectors, highlights, overlays, and object visibility that are already working.",
    ]
    for target in plan.targets:
        shot = f" shot={target.shot_id}" if target.shot_id else ""
        lines.append(
            f"{target.priority}. [{target.source}/{target.source_report}{shot} "
            f"{target.severity}/{target.category}] {target.issue}"
        )
        lines.append(f"   action: {target.action}")
    return "\n".join(lines)


def load_repair_plan_json(text: str) -> RepairPlan:
    return RepairPlan.model_validate(json.loads(text))
