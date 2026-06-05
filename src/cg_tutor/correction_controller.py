"""Conservative retry-strategy correction controller.

This module does not modify storyboard/YAML/scene artifacts. It only decides
how the next coder retry should be routed when repeated evidence says the
current local repair path is not working.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from cg_tutor.concept_metrics import ConceptMetricReport
from cg_tutor.critic_cross_reference import CrossReferenceReport
from cg_tutor.schemas import CriticReport


CorrectionAction = Literal[
    "continue",
    "code_repair",
    "full_regeneration",
    "fresh_branch",
    "abort",
]


@dataclass(frozen=True)
class CorrectionDecision:
    iteration: int
    action: CorrectionAction = "continue"
    allow_diff_repair: bool = True
    force_fresh_branch: bool = False
    priority_addendum: str = ""
    evidence: list[str] = field(default_factory=list)
    suppressed_actions: list[str] = field(default_factory=list)

    @property
    def changed_strategy(self) -> bool:
        return (
            self.action != "continue"
            or not self.allow_diff_repair
            or self.force_fresh_branch
            or bool(self.priority_addendum.strip())
        )

    def to_dict(self) -> dict:
        return asdict(self)


def decide_correction(
    *,
    iteration: int,
    critic_history: list,
    metric_history: list[ConceptMetricReport],
    latest_cross_ref: CrossReferenceReport | None = None,
    max_iteration: int | None = None,
) -> CorrectionDecision:
    """Return the conservative strategy adjustment for the next retry.

    Inputs intentionally stay simple: histories are already produced by the
    pipeline. Strong actions require repeated evidence; a single noisy critic
    or metric finding only affects addendum priority.
    """
    evidence: list[str] = []
    suppressed: list[str] = []
    priority_lines: list[str] = []
    action: CorrectionAction = "continue"
    allow_diff_repair = True
    force_fresh_branch = False

    persistent_metric = _persistent_metric_blocks(metric_history, min_streak=2)
    if persistent_metric:
        action = "full_regeneration"
        allow_diff_repair = False
        suppressed.append("diff_repair")
        evidence.extend(
            f"concept_metric:{rule_id} persisted {streak} iterations"
            for rule_id, streak in persistent_metric
        )
        priority_lines.append(
            "Persistent deterministic concept metric failures were seen in "
            "consecutive iterations. Do not make a local search/replace patch; "
            "regenerate the scene with these constraints as first-class design "
            "requirements."
        )

    if _semantic_blocks_not_improving(critic_history, window=2):
        if action == "continue":
            action = "fresh_branch"
        force_fresh_branch = _can_force_fresh(iteration, max_iteration)
        allow_diff_repair = False
        if "diff_repair" not in suppressed:
            suppressed.append("diff_repair")
        evidence.append(
            "semantic concept_mismatch block count did not improve over the "
            "latest critic iterations"
        )
        priority_lines.append(
            "Repeated semantic blocks are not improving. Preserve required "
            "contracts, but avoid small local edits; rebuild the scene logic "
            "around the concept demonstration."
        )

    if critic_history and _is_compiled_origin(
        getattr(critic_history[-1], "scene_origin", "")
    ):
        allow_diff_repair = False
        if "diff_repair" not in suppressed:
            suppressed.append("diff_repair")
        evidence.append(
            f"latest scene origin is compiled ({critic_history[-1].scene_origin}); "
            "compiled fallback should not be used as a diff-repair base"
        )
        if action == "continue":
            action = "full_regeneration"

    if latest_cross_ref is not None and latest_cross_ref.actionable_count >= 3:
        evidence.append(
            f"cross_reference actionable findings={latest_cross_ref.actionable_count}"
        )
        priority_lines.append(
            "Critic x AST cross-reference produced multiple high-confidence "
            "findings. Resolve these before aesthetic or minor framing changes."
        )
        if action == "continue":
            action = "code_repair"

    if action == "continue" and not evidence:
        return CorrectionDecision(iteration=iteration)

    addendum = _format_priority_addendum(
        action=action,
        allow_diff_repair=allow_diff_repair,
        force_fresh_branch=force_fresh_branch,
        evidence=evidence,
        notes=priority_lines,
    )
    return CorrectionDecision(
        iteration=iteration,
        action=action,
        allow_diff_repair=allow_diff_repair,
        force_fresh_branch=force_fresh_branch,
        priority_addendum=addendum,
        evidence=evidence,
        suppressed_actions=suppressed,
    )


def correction_decision_to_json(decision: CorrectionDecision) -> str:
    return json.dumps(decision.to_dict(), indent=2)


def _persistent_metric_blocks(
    metric_history: list[ConceptMetricReport],
    *,
    min_streak: int,
) -> list[tuple[str, int]]:
    if len(metric_history) < min_streak:
        return []
    latest_rules = {
        issue.rule_id
        for issue in metric_history[-1].issues
        if issue.severity == "block"
    }
    out: list[tuple[str, int]] = []
    for rule_id in sorted(latest_rules):
        streak = 0
        for report in reversed(metric_history):
            rules = {
                issue.rule_id
                for issue in report.issues
                if issue.severity == "block"
            }
            if rule_id in rules:
                streak += 1
            else:
                break
        if streak >= min_streak:
            out.append((rule_id, streak))
    return out


def _semantic_blocks_not_improving(history: list, *, window: int) -> bool:
    if len(history) < window:
        return False
    counts = [_semantic_block_count(item.report) for item in history[-window:]]
    if not counts or max(counts) == 0:
        return False
    return counts[-1] >= counts[0]


def _semantic_block_count(report: CriticReport) -> int:
    return sum(
        1 for issue in report.issues
        if issue.severity == "block" and issue.category == "concept_mismatch"
    )


def _can_force_fresh(iteration: int, max_iteration: int | None) -> bool:
    if max_iteration is None:
        return True
    # Fresh branch is most useful when at least one iteration remains after
    # the current retry. On the final slot, full regeneration is less risky.
    return iteration < max_iteration


def _is_compiled_origin(origin: str) -> bool:
    return origin in {"compiled_seed", "compiled_fallback", "compiler_only"}


def _format_priority_addendum(
    *,
    action: CorrectionAction,
    allow_diff_repair: bool,
    force_fresh_branch: bool,
    evidence: list[str],
    notes: list[str],
) -> str:
    lines = [
        "CORRECTION CONTROLLER DECISION",
        f"- action: {action}",
        f"- allow_diff_repair: {str(allow_diff_repair).lower()}",
        f"- force_fresh_branch: {str(force_fresh_branch).lower()}",
        "- evidence:",
    ]
    lines.extend(f"  - {item}" for item in evidence)
    if notes:
        lines.append("- retry guidance:")
        lines.extend(f"  - {item}" for item in notes)
    return "\n".join(lines)
