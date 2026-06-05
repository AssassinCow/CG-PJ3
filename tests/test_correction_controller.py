from __future__ import annotations

from pathlib import Path

from cg_tutor.concept_metrics import ConceptMetricIssue, ConceptMetricReport
from cg_tutor.correction_controller import decide_correction
from cg_tutor.critic_cross_reference import CrossReferenceFinding, CrossReferenceReport
from cg_tutor.pipeline import CriticIteration
from cg_tutor.schemas import CriticIssue, CriticReport


def _metric(rule_id: str) -> ConceptMetricReport:
    return ConceptMetricReport(
        concept_id="c",
        issues=[
            ConceptMetricIssue(
                severity="block",
                rule_id=rule_id,
                message="bad",
                suggested_fix="fix it",
            )
        ],
    )


def _iter(iteration: int, *, sem_blocks: int, origin: str = "llm") -> CriticIteration:
    issues = [
        CriticIssue(
            shot_id="s",
            frame_idx=1,
            severity="block",
            category="concept_mismatch",
            issue=f"semantic {idx}",
        )
        for idx in range(sem_blocks)
    ]
    return CriticIteration(
        iteration=iteration,
        report=CriticReport(
            concept_id="c",
            iteration=iteration,
            overall_score=0.4,
            issues=issues,
        ),
        scene_path=Path(f"scene.iter{iteration:02d}.py"),
        render_ok=True,
        n_frames=1,
        scene_origin=origin,
    )


def test_persistent_metric_block_disables_diff_repair():
    decision = decide_correction(
        iteration=2,
        critic_history=[],
        metric_history=[
            _metric("dolly_zoom_weak_camera_fov_coupling"),
            _metric("dolly_zoom_weak_camera_fov_coupling"),
        ],
    )

    assert decision.action == "full_regeneration"
    assert decision.allow_diff_repair is False
    assert "diff_repair" in decision.suppressed_actions
    assert "dolly_zoom_weak_camera_fov_coupling" in decision.priority_addendum


def test_single_metric_block_does_not_change_strategy():
    decision = decide_correction(
        iteration=1,
        critic_history=[],
        metric_history=[_metric("one_off")],
    )

    assert decision.action == "continue"
    assert decision.allow_diff_repair is True


def test_non_improving_semantic_blocks_forces_fresh_when_room_remains():
    decision = decide_correction(
        iteration=2,
        max_iteration=5,
        critic_history=[
            _iter(0, sem_blocks=4),
            _iter(1, sem_blocks=4),
        ],
        metric_history=[],
    )

    assert decision.action == "fresh_branch"
    assert decision.force_fresh_branch is True
    assert decision.allow_diff_repair is False


def test_compiled_latest_origin_disables_diff_repair():
    decision = decide_correction(
        iteration=1,
        critic_history=[_iter(0, sem_blocks=0, origin="compiled_fallback")],
        metric_history=[],
    )

    assert decision.action == "full_regeneration"
    assert decision.allow_diff_repair is False


def test_cross_reference_findings_prioritize_code_repair_without_disabling_diff():
    report = CrossReferenceReport(
        concept_id="c",
        iteration=0,
        findings=[
            CrossReferenceFinding(
                rule_id="missing_object_creation",
                severity="actionable",
                diagnosis=f"missing {idx}",
                critic_source="critic",
                ast_evidence="ast",
                suggested_fix="fix",
            )
            for idx in range(3)
        ],
    )
    decision = decide_correction(
        iteration=1,
        critic_history=[],
        metric_history=[],
        latest_cross_ref=report,
    )

    assert decision.action == "code_repair"
    assert decision.allow_diff_repair is True
    assert "cross_reference actionable findings=3" in decision.priority_addendum
