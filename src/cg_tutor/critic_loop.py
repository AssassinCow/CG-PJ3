"""Pure helpers for the W3 critic loop.

Extracted from ``pipeline.py`` so the orchestration in ``pipeline.run()``
stays focused on stage wiring. Everything here is pure(-ish): no global
state, no LLM calls, no Blender subprocess — only data manipulation over
``CriticReport`` / ``CriticIteration`` / ``Storyboard`` / ``Narrative``.

The helpers split into three groups:

  - issue scoring + best-of-N selection
      ``_issue_key``, ``_critic_counts``, ``_semantic_counts``,
      ``_critic_quality_key``
  - signal extraction over the iteration history
      ``_flagged_counts``, ``_regression_keys``,
      ``_block_floor_stale_iters``, ``_scene_param_diff``
  - addendum composition (what the Coder retry actually reads)
      ``_short_issue``, ``_missing_storyboard_objects``, ``_node_by_id``,
      ``_shot_visual_contracts``, ``_issue_action_hint``,
      ``_top_retry_targets``, ``_freeze_lines``,
      ``_critic_history_addendum``

``pipeline.py`` re-exports the public names so existing
``from cg_tutor.pipeline import ...`` imports keep working.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from cg_tutor.agents.render_critic import issues_as_coder_addendum
from cg_tutor.schemas import FRAMING_CATEGORIES, CriticReport, Storyboard
from cg_tutor.scene_profiles import SceneProfile


def _issue_key(issue) -> tuple[str, str, str]:
    """Stable-ish key for comparing critic issues across iterations."""
    return (issue.shot_id, issue.category, issue.severity)


def _critic_counts(report: CriticReport) -> tuple[int, int]:
    """Framing-only (block, warn). concept_mismatch is tracked separately
    by _semantic_counts() because it is a different axis (semantic
    fidelity vs framing compliance) and feeding it into best-of-N
    selection would let a noisy single-image semantic call dominate the
    iter pick. Coder still receives concept_mismatch directives via the
    full issue list in issues_as_coder_addendum().
    """
    n_block = sum(
        1 for i in report.issues
        if i.severity == "block" and i.category in FRAMING_CATEGORIES
    )
    n_warn = sum(
        1 for i in report.issues
        if i.severity == "warn" and i.category in FRAMING_CATEGORIES
    )
    return n_block, n_warn


def _semantic_counts(report: CriticReport) -> tuple[int, int]:
    """(block, warn) restricted to concept_mismatch issues."""
    n_block = sum(
        1 for i in report.issues
        if i.severity == "block" and i.category == "concept_mismatch"
    )
    n_warn = sum(
        1 for i in report.issues
        if i.severity == "warn" and i.category == "concept_mismatch"
    )
    return n_block, n_warn


def _failure_class_counts(result: "CriticIteration") -> tuple[int, int, int, int]:
    """Return structural/success-hard/success-soft/aesthetic counts.

    Metric reports provide deterministic classes. Critic reports remain on
    the legacy block/warn schema for now; do not remap free-form critic prose
    here or framing/semantic selection semantics become unstable.
    """
    structural = (
        int(not result.render_ok)
        + int(getattr(result, "metric_structural_fatal_count", 0) or 0)
    )
    success_hard = int(getattr(result, "metric_success_hard_count", 0) or 0)
    success_soft = int(getattr(result, "metric_success_soft_count", 0) or 0)
    aesthetic = int(getattr(result, "metric_aesthetic_warn_count", 0) or 0)
    return structural, success_hard, success_soft, aesthetic


def _legacy_metric_block_count(result: "CriticIteration") -> int:
    """Metric blocks not covered by explicit failure classes.

    ``success_soft`` can still be emitted as ``severity=block`` for backwards
    compatible reporting, but it should not outrank concept-mismatch blocks in
    best selection or act like a hard gate.
    """
    metric_block = int(getattr(result, "metric_block_count", 0) or 0)
    classified = (
        int(getattr(result, "metric_structural_fatal_count", 0) or 0)
        + int(getattr(result, "metric_success_hard_count", 0) or 0)
        + int(getattr(result, "metric_success_soft_count", 0) or 0)
    )
    return max(0, metric_block - classified)


BestSelectionMode = Literal["framing", "balanced", "semantic"]


def _critic_quality_key(
    result: "CriticIteration",
) -> tuple:
    """Sort key where larger is better.

    Priority (each break only when the previous tied):
      1. fewer structural failures
      2. fewer success-hard failures
      3. prefer non-degraded LLM candidates over diagnostic fallback videos
      4. fewer cross-ref actionable findings (double-corroborated by
         vision critic AND scene AST — stricter signal than framing
         counts or vision score, so it breaks ties after structural
         and success-hard tiers)
      5. fewer framing block issues
      6. fewer legacy unclassified metric block issues
      7. higher overall_score
      8. fewer framing warn issues
      9. fewer concept_mismatch block issues   (tie-breaker only —
         never dominates framing, but when framing is equal we prefer
         the iteration that demonstrates the concept better)
      10. fewer concept_mismatch warn issues
      11. fewer success-soft failures
      12. successful render
      13. deterministic grounding anchors satisfied (final tie-breaker —
         only matters when everything above is equal, never dominates)
    """
    n_block, n_warn = _critic_counts(result.report)
    sem_block, sem_warn = _semantic_counts(result.report)
    legacy_metric_block = _legacy_metric_block_count(result)
    structural, success_hard, success_soft, _aesthetic = _failure_class_counts(result)
    cross_ref_actionable = int(
        getattr(result, "cross_ref_actionable_count", 0) or 0
    )
    non_degraded = int(not getattr(result, "fallback_degraded", False))
    return (
        -structural,
        -success_hard,
        non_degraded,
        -cross_ref_actionable,
        -n_block,
        -legacy_metric_block,
        result.report.overall_score,
        -n_warn,
        -sem_block,
        -sem_warn,
        -success_soft,
        int(result.render_ok),
        int(result.contract_anchors_ok),
    )


def _critic_quality_key_for(
    result: "CriticIteration",
    mode: BestSelectionMode = "framing",
) -> tuple:
    """Sort key for best-of-N selection under a chosen policy.

    ``framing`` preserves the original conservative behaviour: layout and
    renderability dominate, with semantic issues only as tie-breakers.

    ``balanced`` keeps framing blockers first but lets semantic blockers
    participate in the primary block floor. This avoids selecting a cleaner
    looking but more conceptually wrong video when a slightly rougher candidate
    has fewer total blockers.

    ``semantic`` is for teaching-quality experiments: total block count
    (framing + concept_mismatch) dominates, then score/framing warnings.
    """
    if mode == "framing":
        return _critic_quality_key(result)
    n_block, n_warn = _critic_counts(result.report)
    sem_block, sem_warn = _semantic_counts(result.report)
    legacy_metric_block = _legacy_metric_block_count(result)
    metric_warn = getattr(result, "metric_warn_count", 0)
    structural, success_hard, success_soft, aesthetic_warn = _failure_class_counts(result)
    cross_ref_actionable = int(
        getattr(result, "cross_ref_actionable_count", 0) or 0
    )
    non_degraded = int(not getattr(result, "fallback_degraded", False))
    if mode == "balanced":
        total_block = n_block + sem_block + legacy_metric_block
        total_warn = n_warn + sem_warn + metric_warn
        return (
            -structural,
            -success_hard,
            non_degraded,
            -cross_ref_actionable,
            -total_block,
            -sem_block,
            -n_block,
            -legacy_metric_block,
            -sem_warn,
            -success_soft,
            result.report.overall_score,
            -total_warn,
            int(result.render_ok),
            int(result.contract_anchors_ok),
        )
    if mode == "semantic":
        total_block = n_block + sem_block + legacy_metric_block
        total_warn = n_warn + sem_warn + metric_warn
        return (
            -structural,
            -success_hard,
            non_degraded,
            -cross_ref_actionable,
            -total_block,
            -sem_block,
            -n_block,
            -legacy_metric_block,
            -sem_warn,
            -success_soft,
            result.report.overall_score,
            -total_warn,
            -aesthetic_warn,
            int(result.render_ok),
            int(result.contract_anchors_ok),
        )
    raise ValueError(f"unknown best selection mode: {mode}")


def _flagged_counts(history: list["CriticIteration"]) -> dict[tuple[str, str, str], int]:
    """Trailing-streak length for each issue key in the latest report.

    A key in the latest report with streak ≥ 2 has been flagged for that
    many consecutive iterations — the prior fixes did not resolve it.
    """
    if not history:
        return {}
    latest_keys = {_issue_key(i) for i in history[-1].report.issues}
    counts: dict[tuple[str, str, str], int] = {}
    for k in latest_keys:
        streak = 1
        for h in reversed(history[:-1]):
            if k in {_issue_key(i) for i in h.report.issues}:
                streak += 1
            else:
                break
        counts[k] = streak
    return counts


def _regression_keys(history: list["CriticIteration"]) -> set[tuple[str, str, str]]:
    """Keys in the latest report that were absent last iter but present earlier.

    A regression means the Coder had this fixed and then re-broke it.
    """
    if len(history) < 3:
        return set()
    latest = {_issue_key(i) for i in history[-1].report.issues}
    prev = {_issue_key(i) for i in history[-2].report.issues}
    earlier: set[tuple[str, str, str]] = set()
    for h in history[:-2]:
        earlier |= {_issue_key(i) for i in h.report.issues}
    return (latest - prev) & earlier


def _block_floor_stale_iters(history: list["CriticIteration"]) -> tuple[int, int]:
    """Return (running_min_block_count, iters_since_floor_was_lowered).

    The first value is the lowest block count seen so far. The second
    counts how many trailing iterations did NOT set a new floor (i.e.,
    the Coder has been trading issues without net progress). Used by
    the NO PROGRESS addendum section and the pipeline early-stop.
    """
    if not history:
        return (0, 0)
    counts = [_critic_counts(h.report)[0] for h in history]
    running_min = counts[0]
    last_decrease_idx = 0
    for i in range(1, len(counts)):
        if counts[i] < running_min:
            running_min = counts[i]
            last_decrease_idx = i
    return (running_min, len(counts) - 1 - last_decrease_idx)


_SCENE_PARAM_KEYWORDS = (
    "camera", "light", "energy", "fov", "lens", "location",
    "rotation", "intensity", "shadow", "overlay_zone",
)
_ASSIGN_RE = re.compile(r"^\s*[\w.\[\]'\"]+\s*=")


def _scene_param_diff(best: "CriticIteration",
                      latest: "CriticIteration") -> str:
    """Set-based diff of camera/light assignment lines: best vs latest scene.

    Only considers lines that look like Python assignments (LHS `=` RHS)
    AND contain a camera/light/overlay keyword. This filters out tuple
    and list entries inside scene-spec data structures, which would
    otherwise dominate the diff when the Coder emits scenes as
    list-of-tuples instead of explicit assignments.

    When both sides have more than a handful of candidate lines the
    scene is almost certainly in data-structure form and the diff
    would just be noise — return empty in that case rather than
    spamming the addendum.
    """
    if best.iteration == latest.iteration:
        return ""
    if not best.scene_path.exists() or not latest.scene_path.exists():
        return ""

    def filter_lines(text: str) -> set[str]:
        out: set[str] = set()
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not _ASSIGN_RE.match(stripped):
                continue
            low = stripped.lower()
            if any(k in low for k in _SCENE_PARAM_KEYWORDS):
                out.add(stripped)
        return out

    best_lines = filter_lines(best.scene_path.read_text())
    latest_lines = filter_lines(latest.scene_path.read_text())
    only_in_best = sorted(best_lines - latest_lines)
    only_in_latest = sorted(latest_lines - best_lines)
    if not only_in_best and not only_in_latest:
        return ""
    if len(only_in_best) + len(only_in_latest) > 12:
        return ""

    lines: list[str] = []
    if only_in_best:
        lines.append(
            f"REMOVED vs best iter{best.iteration:02d} "
            "(camera/light/overlay assignments you may need to restore):"
        )
        for s in only_in_best[:5]:
            lines.append(f"  - {s}")
        if len(only_in_best) > 5:
            lines.append(f"  ... ({len(only_in_best) - 5} more)")
    if only_in_latest:
        lines.append(
            f"ADDED in latest iter{latest.iteration:02d} "
            "(these may be the regression cause):"
        )
        for s in only_in_latest[:5]:
            lines.append(f"  + {s}")
        if len(only_in_latest) > 5:
            lines.append(f"  + ... ({len(only_in_latest) - 5} more)")
    return "\n".join(lines)


def _short_issue(text: str, limit: int = 150) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


def _missing_storyboard_objects(
    code: str, storyboard: Storyboard,
) -> dict[str, list[str]]:
    """Return shot_id -> [object names the coder did not include].

    Deterministic substring scan over the generated scene.py. The coder
    prompt mandates that storyboard object names appear verbatim as
    `obj.name = "<name>"`, so a missing name almost certainly means the
    object was not instantiated. Cheap pre-render signal that catches
    cases where the LLM regenerated a scene without all required objects.
    """
    missing: dict[str, list[str]] = {}
    for shot in storyboard.shots:
        gaps: list[str] = []
        for obj in shot.objects:
            name = obj.name.strip()
            if not name:
                continue
            if name not in code:
                gaps.append(name)
        if gaps:
            missing[shot.node_id] = gaps
    return missing


def _node_by_id(narrative) -> dict[str, object]:
    return {node.id: node for node in narrative.nodes}


def _shot_visual_contracts(
    narrative,
    storyboard: Storyboard,
    *,
    scene_profile: SceneProfile | None = None,
) -> str:
    """Compact per-shot source of truth for semantic fixes.

    This is intentionally concept-agnostic: every concept already has a
    narrative visual_intent plus a storyboard object/camera contract. The
    Coder gets those fields in a compact, repeated form so retries can fix
    concept_mismatch without inventing concept-specific rules.
    """
    nodes = _node_by_id(narrative)
    lines = [
        "SHOT VISUAL CONTRACTS (source of truth for semantic correctness):",
        "- For every shot, make the narrative visual_intent literally visible.",
        "- Do not add helper arrows, markers, labels, trails, or gizmos unless "
        "the shot's visual_intent/storyboard objects call for them.",
        "- If a concept_mismatch is reported, fix it by adding/removing/"
        "modifying geometry, materials, labels, highlights, or animation; "
        "camera-only changes are not sufficient.",
    ]
    if scene_profile and scene_profile.base_profile == "cinematic_application":
        lines.extend([
            "- Cinematic application profile: use thin glowing/dotted tracers "
            "for ray cues instead of thick arrows or arrowheads.",
            "- Hide light gizmos and avoid visible foreground light bars/panels.",
        ])
    for shot in storyboard.shots:
        node = nodes.get(shot.node_id)
        objects = ", ".join(obj.name for obj in shot.objects[:10])
        if len(shot.objects) > 10:
            objects += f", ... ({len(shot.objects) - 10} more)"
        visual_intent = getattr(node, "visual_intent", "") if node else ""
        description = getattr(node, "description", "") if node else ""
        formulas = getattr(node, "formulas", []) if node else []
        formula = shot.formula or (formulas[0] if formulas else "")
        lines.extend([
            "",
            f"{shot.node_id}:",
            f"- narrative: {_short_issue(description, 180)}",
            f"- visual_intent: {_short_issue(visual_intent, 260)}",
            f"- expected visible objects: {objects}",
            f"- formula/caption: {_short_issue(formula or shot.caption or 'none', 160)}",
        ])
    return "\n".join(lines)


# Single source of truth for repair-category messages. Adding a new
# category here updates both the critic-history addendum hints (read by
# this module) and the repair-plan action text (read by repair_plan.py).
# This avoids the prior drift between two parallel dicts.
CATEGORY_REPAIR_MESSAGES: dict[str, dict[str, str]] = {
    "concept_mismatch": {
        "hint": (
            "semantic fix: compare this shot to its visual_intent; add, "
            "remove, or modify visible objects/materials/labels/highlights/"
            "animation so the intent is literally visible. Do not fix with "
            "camera movement alone."
        ),
        "action_base": (
            "Modify scene geometry/materials/labels/highlights/animation so "
            "the shot literally matches visual_intent; do not solve with "
            "camera movement alone."
        ),
    },
    "off_screen": {
        "hint": (
            "framing fix: move the camera/object, reduce scale, or widen FOV "
            "while preserving frozen shots and overlay readability."
        ),
        "action_base": (
            "Reframe by moving camera/object, changing scale, or widening FOV."
        ),
    },
    "overlay_collision": {
        "hint": (
            "overlay fix: move overlay_zone or reframe the subject so formula "
            "space is clear without hiding key objects."
        ),
        "action_base": (
            "Move overlay_zone or reframe subject to reserve formula space."
        ),
    },
    "occlusion": {
        "hint": (
            "visibility fix: separate objects, move them to the intended "
            "surface/plane, adjust camera angle, or hide construction helpers."
        ),
        "action_base": (
            "Separate objects, move helpers to surfaces, or hide clutter."
        ),
    },
    "lighting": {
        "hint": (
            "lighting fix: adjust light positions, energy, materials, or "
            "shadows so the intended visual cue is visible."
        ),
        "action_base": (
            "Adjust lights/materials/shadows so the intended cue is visible."
        ),
    },
    "other": {
        "hint": (
            "general fix: make the smallest scene change that directly "
            "addresses the reported issue."
        ),
        "action_base": (
            "Make the smallest direct change that fixes the reported issue."
        ),
    },
}


def _issue_action_hint(issue) -> str:
    entry = CATEGORY_REPAIR_MESSAGES.get(
        issue.category, CATEGORY_REPAIR_MESSAGES["other"]
    )
    return entry["hint"]


def _top_retry_targets(report: CriticReport, limit: int = 6) -> list[str]:
    """Top issues from the base report, formatted as action targets."""
    category_priority = {
        "off_screen": 0,
        "overlay_collision": 1,
        "occlusion": 2,
        "lighting": 3,
        "concept_mismatch": 4,
        "other": 5,
    }
    severity_priority = {"block": 0, "warn": 1}
    issues = sorted(
        report.issues,
        key=lambda i: (
            severity_priority.get(i.severity, 99),
            0 if i.category in FRAMING_CATEGORIES else 1,
            category_priority.get(i.category, 99),
            i.shot_id,
            i.frame_idx,
            i.issue,
        ),
    )
    out: list[str] = []
    for issue in issues[:limit]:
        line = (
            f"- {issue.shot_id}: {issue.category} {issue.severity} at frame "
            f"{issue.frame_idx} — {_short_issue(issue.issue)} "
            f"Action: {_issue_action_hint(issue)}"
        )
        if issue.suggested_fix:
            line += f" suggested={json.dumps(issue.suggested_fix)}"
        out.append(line)
    return out


def format_critic_visual_evidence_packet(
    history: list["CriticIteration"],
    *,
    limit: int = 8,
) -> str:
    """Return a high-priority visual evidence packet for the next repair.

    This deliberately duplicates some information from the softer critic
    history summary, but in a form that is harder for the coder to miss:
    concrete visual block issues plus the raw suggested_fix JSON. It is a
    routing layer, not a new judge.
    """
    if not history:
        return ""
    latest = history[-1]
    best = max(history, key=_critic_quality_key)

    selected = _critic_evidence_issues(latest.report, limit=limit)
    if best.iteration != latest.iteration and len(selected) < limit:
        seen = {_issue_key(issue) + (issue.issue,) for issue in selected}
        for issue in _critic_evidence_issues(best.report, limit=limit):
            key = _issue_key(issue) + (issue.issue,)
            if key in seen:
                continue
            selected.append(issue)
            seen.add(key)
            if len(selected) >= limit:
                break
    if not selected and not latest.report.pass_blockers:
        return ""

    lines = [
        "CRITIC VISUAL EVIDENCE PACKET:",
        "- Treat these as high-priority visual observations from rendered frames.",
        "- Apply the suggested_fix JSON literally when it names object/channel/position changes.",
        "- Do not solve concept_mismatch by only moving the camera unless the issue says framing is the root cause.",
    ]
    if latest.report.pass_blockers:
        lines.append("- Strict pass blockers: " + "; ".join(latest.report.pass_blockers[:4]))
    diagnostics = latest.report.ensemble_diagnostics or {}
    score_spread = diagnostics.get("score_spread")
    if score_spread:
        lines.append(f"- Ensemble score spread: {score_spread}")

    for idx, issue in enumerate(selected, start=1):
        lines.append(
            f"{idx}. iter{latest.iteration:02d} shot={issue.shot_id} "
            f"frame={issue.frame_idx} {issue.severity}/{issue.category}: "
            f"{_short_issue(issue.issue, limit=420)}"
        )
        if issue.suggested_fix:
            lines.append(
                "   suggested_fix: "
                + json.dumps(issue.suggested_fix, ensure_ascii=False, sort_keys=True)
            )
    return "\n".join(lines)


def _critic_evidence_issues(report: CriticReport, *, limit: int) -> list:
    severity_priority = {"block": 0, "warn": 1}
    issues = sorted(
        report.issues,
        key=lambda i: (
            severity_priority.get(i.severity, 99),
            0 if i.category in {"concept_mismatch", "occlusion", "off_screen"} else 1,
            i.shot_id,
            i.frame_idx,
            i.issue,
        ),
    )
    return issues[:limit]


def _freeze_lines(
    best_report: CriticReport,
    latest_report: CriticReport,
    shot_ids: list[str] | None,
    limit: int = 6,
) -> list[str]:
    """Soft-freeze shots whose framing is clean in the best iteration."""
    if not shot_ids:
        return []

    best_framing_flagged = {
        i.shot_id for i in best_report.issues
        if i.category in FRAMING_CATEGORIES
    }
    latest_flagged = {
        i.shot_id for i in latest_report.issues
        if i.category in FRAMING_CATEGORIES and i.severity == "block"
    }
    frozen = [
        s for s in shot_ids
        if s not in best_framing_flagged and s not in latest_flagged
    ][:limit]
    return [
        f"- {s}: preserve camera, overlay_zone, object visibility, and "
        "lighting from the best scene unless this shot is explicitly "
        "flagged later or a block fix requires touching it."
        for s in frozen
    ]


def _critic_history_addendum(
    history: list["CriticIteration"],
    shot_ids: list[str] | None = None,
) -> str:
    """Summarize critic trajectory for the next coder retry."""
    if not history:
        return ""

    latest = history[-1]
    best = max(history, key=_critic_quality_key)
    latest_block, latest_warn = _critic_counts(latest.report)
    best_block, best_warn = _critic_counts(best.report)
    latest_sem_block, latest_sem_warn = _semantic_counts(latest.report)
    best_sem_block, best_sem_warn = _semantic_counts(best.report)

    lines: list[str] = []

    if latest.missing_objects:
        lines.append(
            "MISSING FROM SCENE.PY (deterministic name check — fix these "
            "first, they are exact and 0-ambiguity):"
        )
        for shot_id, names in sorted(latest.missing_objects.items()):
            lines.append(f"- {shot_id}: {', '.join(names)}")
        lines.append(
            "→ The previous scene.py did not contain these storyboard "
            "object names verbatim. Add the objects (with these exact "
            "names via `obj.name = '<name>'`), give them visible "
            "geometry/material consistent with visual_intent, and place "
            "them on the right shot timeline. This is the most common "
            "root cause of concept_mismatch."
        )
        lines.append("")

    if latest.report.execution_errors:
        lines.append(
            f"CRITIC EXECUTION ERRORS ({len(latest.report.execution_errors)} "
            "shot call(s) failed — those shots were NOT scored):"
        )
        for err in latest.report.execution_errors[:5]:
            lines.append(f"- {err}")
        if len(latest.report.execution_errors) > 5:
            lines.append(
                f"  ... ({len(latest.report.execution_errors) - 5} more)"
            )
        lines.append(
            "→ Treat these shots as un-evaluated. Do not assume they "
            "passed; preserve their best-known versions."
        )
        lines.append("")

    lines.extend([
        "CRITIC HISTORY SUMMARY:",
        f"- Latest iter{latest.iteration:02d}: "
        f"score={latest.report.overall_score:.2f}, "
        f"framing block={latest_block}, warn={latest_warn}.",
    ])
    if latest_sem_block or latest_sem_warn:
        lines.append(
            f"- Latest concept_mismatch (semantic axis, not in score): "
            f"block={latest_sem_block}, warn={latest_sem_warn}."
        )
    lines.append(
        f"- Best so far iter{best.iteration:02d}: "
        f"score={best.report.overall_score:.2f}, "
        f"framing block={best_block}, warn={best_warn}."
    )
    if best_sem_block or best_sem_warn:
        lines.append(
            f"- Best concept_mismatch: block={best_sem_block}, "
            f"warn={best_sem_warn}."
        )

    if len(history) >= 2:
        prev = history[-2]
        prev_block, prev_warn = _critic_counts(prev.report)
        delta = latest.report.overall_score - prev.report.overall_score
        lines.append(
            f"- Last change from iter{prev.iteration:02d}: "
            f"score {delta:+.2f}, framing block {latest_block - prev_block:+d}, "
            f"warn {latest_warn - prev_warn:+d}."
        )

        prev_keys = {_issue_key(i) for i in prev.report.issues}
        latest_keys = {_issue_key(i) for i in latest.report.issues}
        resolved = sorted(prev_keys - latest_keys)
        added = sorted(latest_keys - prev_keys)
        persistent = sorted(prev_keys & latest_keys)
        if resolved:
            lines.append("- Resolved issue categories: " + ", ".join(map(str, resolved[:8])))
        if added:
            lines.append("- New/regressed issue categories: " + ", ".join(map(str, added[:8])))
        if persistent:
            lines.append("- Persistent issue categories: " + ", ".join(map(str, persistent[:8])))

    flagged = _flagged_counts(history)
    repeat_offenders = sorted(
        ((k, n) for k, n in flagged.items() if n >= 2),
        key=lambda kn: (-kn[1], kn[0]),
    )[:5]
    if repeat_offenders:
        lines.append("")
        lines.append("REPEAT OFFENDERS (flagged ≥2 iters in a row, still present):")
        for k, n in repeat_offenders:
            shot_id, category, severity = k
            lines.append(
                f"- ({shot_id}, {category}, {severity})  ← flagged {n} times"
            )
        lines.append(
            "→ Previous fix strategies have NOT worked. Try a different "
            "approach (e.g. move the camera instead of moving objects; "
            "change FOV instead of repositioning lights; reduce light "
            "energy instead of adding lights). Or, if the issue is a "
            "warn and the fix would risk a block, accept the warning "
            "and document the trade-off in a scene.py comment."
        )

    regressions = sorted(_regression_keys(history))[:5]
    if regressions:
        lines.append("")
        lines.append("REGRESSIONS (resolved in a prior iter, back this iter):")
        for k in regressions:
            shot_id, category, severity = k
            lines.append(f"- ({shot_id}, {category}, {severity})")
        lines.append(
            "→ You had these fixed before. Diff the current scene against "
            "the best iteration and preserve whatever change resolved them."
        )
        diff = _scene_param_diff(best, latest)
        if diff:
            lines.append("")
            lines.append(diff)

    concept_misses = sorted(
        _issue_key(i) for i in latest.report.issues
        if i.category == "concept_mismatch"
    )
    if concept_misses:
        lines.append("")
        lines.append(
            "CONCEPT MISMATCH (semantic gaps from visual_intent — different "
            "from framing issues):"
        )
        for k in concept_misses[:5]:
            shot_id, _cat, severity = k
            lines.append(f"- ({shot_id}, concept_mismatch, {severity})")
        if len(concept_misses) > 5:
            lines.append(f"  ... ({len(concept_misses) - 5} more)")
        lines.append(
            "→ Moving the camera or the lights will NOT fix these. Re-read "
            "the visual_intent for each listed shot and ADD or MODIFY the "
            "missing element — that usually means new geometry (a curve "
            "mesh, a trail, a highlighted region), a different material "
            "colour so the element stands out, or an animation that "
            "actually shows the described behaviour."
        )

    floor, stale = _block_floor_stale_iters(history)
    if stale >= 2:
        lines.append("")
        lines.append(
            f"NO PROGRESS: block floor = {floor}, unchanged for {stale} "
            "iterations. Trading one issue for another is not helping."
        )
        lines.append(
            "→ Stop adding new fixes. Return the iter"
            f"{best.iteration:02d} scene as-is, or change ONLY the single "
            "lowest-risk parameter most likely to clear the remaining "
            "block. If you cannot improve on the floor, prefer the "
            "best-known scene unchanged."
        )

    lines.extend([
        "",
        "BASELINE FOR THIS RETRY:",
        f"- Start from best iter{best.iteration:02d}, not the latest "
        "iteration if the latest regressed.",
        "- Make the smallest complete-code change that improves the best "
        "scene; do not redesign clean shots.",
    ])

    targets = _top_retry_targets(best.report)
    if targets:
        lines.extend([
            "",
            "TOP RETRY TARGETS (from best iteration; fix at most these first):",
            *targets,
        ])
    else:
        lines.extend([
            "",
            "TOP RETRY TARGETS:",
            "- Best iteration has no remaining critic issues. Preserve it; "
            "only make changes if they are explicitly requested elsewhere.",
        ])

    frozen = _freeze_lines(best.report, latest.report, shot_ids)
    if frozen:
        lines.extend([
            "",
            "SOFT FREEZE / PRESERVE:",
            *frozen,
        ])

    lines.extend([
        "",
        "NEXT RETRY INSTRUCTIONS:",
        "- Optimize for eliminating all block issues before improving aesthetics.",
        "- Preserve camera/framing/light changes from the best iteration when possible.",
        "- Do not introduce new overlay collisions while fixing off-screen objects.",
        "- If a prior iteration got worse, undo that kind of change rather than amplifying it.",
        "",
        "BEST ITERATION REMAINING ISSUES:",
        issues_as_coder_addendum(best.report),
    ])
    return "\n".join(lines)


def _multi_reference_retry_addendum(history: list["CriticIteration"]) -> str:
    """Give the coder multiple evaluated references without changing patch base.

    The next retry should edit the framing-safe scene, but it can borrow
    semantic/aesthetic ideas from other iterations.  This avoids the two
    common traps: getting stuck on a conservative but semantically weak base,
    or blindly continuing from a newer version that improved meaning while
    breaking framing.
    """
    if len(history) < 2:
        return ""

    safe = max(history, key=lambda r: _critic_quality_key_for(r, "framing"))
    semantic = max(history, key=lambda r: _critic_quality_key_for(r, "semantic"))
    aesthetic = max(history, key=lambda r: r.report.overall_score)

    refs: list[tuple[str, CriticIteration]] = [
        ("framing_safe_base", safe),
        ("semantic_reference", semantic),
        ("aesthetic_reference", aesthetic),
    ]
    lines = [
        "MULTI-REFERENCE RETRY STRATEGY:",
        "- Use the framing_safe_base scene as the ONLY patch/edit base.",
        "- Borrow semantic or visual ideas from references only when they do "
        "not reintroduce framing blocks.",
        "- Do not merge two full scripts. If a reference improved a concept "
        "cue, reimplement that cue cleanly inside the base scene.",
        "",
        "REFERENCE ROLES:",
    ]
    seen_iters: set[int] = set()
    for role, ref in refs:
        n_block, n_warn = _critic_counts(ref.report)
        sem_block, sem_warn = _semantic_counts(ref.report)
        marker = " (same as another role)" if ref.iteration in seen_iters else ""
        seen_iters.add(ref.iteration)
        lines.append(
            f"- {role}: iter{ref.iteration:02d}{marker}; "
            f"score={ref.report.overall_score:.2f}; "
            f"framing={n_block} block/{n_warn} warn; "
            f"semantic={sem_block} block/{sem_warn} warn."
        )

    if semantic.iteration != safe.iteration:
        lines.extend([
            "",
            "BORROW FROM SEMANTIC_REFERENCE:",
            "- It has stronger concept expression than the safe base. Borrow "
            "only the concrete visual cues that reduce concept_mismatch "
            "(for example tracers, highlights, HUD state, material changes, "
            "or animation timing).",
            "- Do not copy camera/framing/light changes that caused new "
            "framing blocks.",
        ])
        diff = _scene_param_diff(safe, semantic)
        if diff:
            lines.extend([
                "",
                "SAFE_BASE ↔ SEMANTIC_REFERENCE PARAM DIFF (inspect carefully; "
                "copy only low-risk cue changes):",
                diff,
            ])

    if aesthetic.iteration not in {safe.iteration, semantic.iteration}:
        lines.extend([
            "",
            "BORROW FROM AESTHETIC_REFERENCE:",
            "- It has the highest visual score. Borrow color/material/light "
            "polish only if it does not hurt framing_safe_base visibility.",
        ])
        diff = _scene_param_diff(safe, aesthetic)
        if diff:
            lines.extend([
                "",
                "SAFE_BASE ↔ AESTHETIC_REFERENCE PARAM DIFF (optional polish):",
                diff,
            ])

    lines.extend([
        "",
        "FUSION RULE:",
        "- The returned patch/full scene must still be based on "
        f"iter{safe.iteration:02d}. A valid improvement keeps its framing "
        "safety while adding the best semantic cues from the reference "
        "iterations.",
    ])
    return "\n".join(lines)


@dataclass
class CriticIteration:
    iteration: int
    report: CriticReport
    scene_path: Path
    render_ok: bool
    n_frames: int
    frames_hash: str = ""
    # Directory containing the exact PNG frames that were scored by the
    # critic for this iteration. When present, final composition can use this
    # snapshot directly instead of re-rendering and risking Blender/GPU
    # non-determinism.
    frames_dir: Path | None = None
    scene_origin: str = "unknown"
    # shot_id -> [object names from the storyboard that the coder's
    # scene.py for this iter did not mention verbatim]. Computed before
    # render; surfaced in the next iter's addendum so the coder fixes the
    # gap by adding geometry, not by waiting for the vision critic to
    # report concept_mismatch.
    missing_objects: dict[str, list[str]] = field(default_factory=dict)
    # Deterministic grounding signal from contract_validator. True when
    # required_anchors are all present (or no anchors were required, or
    # no contract validation was run for this iter). Used as a low-priority
    # tie-breaker in _critic_quality_key so that two iterations with
    # otherwise equal critic scores prefer the one whose script actually
    # built the required scene anchors.
    contract_anchors_ok: bool = True
    # Deterministic concept-metric failures are cheaper and more stable than
    # vision-critic judgments. They participate in pass/final selection in
    # pipeline.py while remaining separate from CriticReport's schema.
    metric_block_count: int = 0
    metric_warn_count: int = 0
    metric_structural_fatal_count: int = 0
    metric_success_hard_count: int = 0
    metric_success_soft_count: int = 0
    metric_aesthetic_warn_count: int = 0
    cross_ref_actionable_count: int = 0
    # True when a compiled scaffold was used only as a diagnostic/render-safe
    # fallback and still violates visual contracts. It may be critiqued and
    # exported, but it must not masquerade as a normal passing candidate.
    fallback_degraded: bool = False
