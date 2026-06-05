"""Test the per-shot keyframe sampler the Render Critic uses."""

from __future__ import annotations

import json
from pathlib import Path

from cg_tutor.agents.render_critic import (
    _aggregate_reports,
    _overall_from_scores,
    _sample_keyframes,
    _visual_contract_text,
    inspect_ensemble,
)
from cg_tutor.schemas import CriticIssue, CriticReport, Narrative, NarrativeNode, Storyboard

FIXTURE = Path(__file__).parent / "fixtures" / "phong_storyboard.json"


def test_single_shot_three_picks_inside_window():
    sb = Storyboard.model_validate_json(FIXTURE.read_text())
    samples = _sample_keyframes(sb)
    # one shot
    assert list(samples.keys()) == [0]
    # picks fall inside the shot's frame window [1, 120]
    for f in samples[0]:
        assert 1 <= f <= 120


def test_multi_shot_picks_are_per_shot_and_in_order():
    sb_data = {
        "concept_id": "c",
        "fps": 10,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a", "start_sec": 0.0, "duration_sec": 2.0,
                "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "o", "type": "primitive", "primitive": "sphere"}],
            },
            {
                "node_id": "b", "start_sec": 2.0, "duration_sec": 3.0,
                "camera": [{"time_sec": 2, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "o", "type": "primitive", "primitive": "sphere"}],
            },
        ],
    }
    sb = Storyboard.model_validate(sb_data)
    samples = _sample_keyframes(sb)
    # shot 0 has 2 s × 10 fps = 20 frames, range [1, 20]
    assert all(1 <= f <= 20 for f in samples[0])
    # shot 1 has 3 s × 10 fps = 30 frames, range [21, 50]
    assert all(21 <= f <= 50 for f in samples[1])
    # picks are sorted within a shot
    for picks in samples.values():
        assert picks == sorted(picks)


def test_sampler_handles_very_short_shot():
    """A 1-frame shot should still produce at least one keyframe pick."""
    sb_data = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a", "start_sec": 0.0, "duration_sec": 1 / 24,  # 1 frame
                "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "o", "type": "primitive", "primitive": "sphere"}],
            },
        ],
    }
    sb = Storyboard.model_validate(sb_data)
    samples = _sample_keyframes(sb)
    assert len(samples[0]) >= 1


def test_visual_contract_text_for_critic_includes_object_level_checks():
    sb_data = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a", "start_sec": 0.0, "duration_sec": 1.0,
                "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "normal_arrow", "type": "primitive", "primitive": "arrow"}],
                "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
            },
        ],
    }
    narrative = Narrative(
        concept_id="c",
        nodes=[
            NarrativeNode(
                id="a",
                title="A",
                description="Show arrow N with visible labels.",
                formulas=[],
                duration_sec=1.0,
                visual_intent="Show arrow N with visible labels and a highlight.",
            )
        ],
    )
    text = _visual_contract_text(
        Storyboard.model_validate(sb_data),
        0,
        narrative,
    )

    assert "required_vectors" in text
    assert "required_labels" in text
    assert "overlay_constraints" in text


def test_critic_ensemble_aggregate_dedupes_issues():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issue = CriticIssue(
        shot_id="a",
        frame_idx=1,
        severity="block",
        category="off_screen",
        issue="Object is outside the frame.",
    )
    report = _aggregate_reports(
        sb,
        0,
        [
            ("a", CriticReport(concept_id="c", iteration=0, overall_score=0.5, issues=[issue])),
            ("b", CriticReport(concept_id="c", iteration=0, overall_score=0.9, issues=[issue])),
        ],
    )

    assert report.overall_score == 0.7
    assert len(report.issues) == 1
    assert report.issues[0].severity == "block"


def test_critic_ensemble_downgrades_single_member_semantic_block():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issue = CriticIssue(
        shot_id="a",
        frame_idx=1,
        severity="block",
        category="concept_mismatch",
        issue="OPEN sign is rendered as a plain glowing rectangle.",
    )

    report = _aggregate_reports(
        sb,
        0,
        [
            ("claude", CriticReport(concept_id="c", iteration=0, overall_score=0.5, issues=[issue])),
            ("gemini", CriticReport(concept_id="c", iteration=0, overall_score=0.9, issues=[])),
        ],
    )

    assert len(report.issues) == 1
    assert report.issues[0].severity == "warn"
    assert report.issues[0].suggested_fix["_block_support"] == ["claude"]
    assert report.issues[0].suggested_fix["_demoted_from_block"] is True


def test_critic_ensemble_union_keeps_single_member_block():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issue = CriticIssue(
        shot_id="a",
        frame_idx=1,
        severity="block",
        category="concept_mismatch",
        issue="Required C label is absent.",
    )

    report = _aggregate_reports(
        sb,
        0,
        [
            ("claude", CriticReport(concept_id="c", iteration=0, overall_score=0.5, issues=[issue])),
            ("gpt", CriticReport(concept_id="c", iteration=0, overall_score=0.9, issues=[])),
        ],
        strictness="union",
    )

    assert report.issues[0].severity == "block"
    assert not report.issues[0].suggested_fix.get("_demoted_from_block")


def test_critic_ensemble_strict_blocks_on_member_blocks_and_score_spread():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issues = [
        CriticIssue(
            shot_id="a",
            frame_idx=i,
            severity="block",
            category="concept_mismatch",
            issue=f"Missing teaching anchor {i}.",
        )
        for i in range(3)
    ]

    report = _aggregate_reports(
        sb,
        0,
        [
            ("claude", CriticReport(concept_id="c", iteration=0, overall_score=0.6, issues=issues)),
            ("gpt", CriticReport(concept_id="c", iteration=0, overall_score=0.95, issues=[])),
        ],
        strictness="strict",
    )

    assert not report.pass_threshold
    assert any("member_block_count" in item for item in report.pass_blockers)
    assert any("member_score_spread" in item for item in report.pass_blockers)
    assert report.ensemble_diagnostics["score_spread"] == 0.35


def test_critic_ensemble_keeps_fatal_single_member_block():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issue = CriticIssue(
        shot_id="a",
        frame_idx=1,
        severity="block",
        category="other",
        issue="Blank frame: render failed to produce visible scene content.",
    )

    report = _aggregate_reports(
        sb,
        0,
        [
            ("claude", CriticReport(concept_id="c", iteration=0, overall_score=0.5, issues=[issue])),
            ("gemini", CriticReport(concept_id="c", iteration=0, overall_score=0.9, issues=[])),
        ],
    )

    assert len(report.issues) == 1
    assert report.issues[0].severity == "block"


def test_critic_report_execution_errors_do_not_pass():
    report = CriticReport(
        concept_id="c",
        iteration=0,
        overall_score=0.95,
        issues=[],
        execution_errors=["relay disconnected"],
    )

    assert not report.pass_threshold


def test_critic_error_only_score_is_not_clean_pass_score():
    assert _overall_from_scores([], ["all shot calls failed"]) == 0.0


def test_critic_ensemble_empty_aggregate_does_not_pass():
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })

    report = _aggregate_reports(sb, 0, [])

    assert report.overall_score == 0.0
    assert report.execution_errors
    assert not report.pass_threshold


def test_critic_ensemble_failed_member_is_diagnostic_but_not_scored(monkeypatch, tmp_path: Path):
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })

    def fake_inspect(sb, frames_dir, *, iteration=0, out_dir=None, backend=None,
                     narrative=None, scene_profile=None):
        if backend == "bad":
            raise RuntimeError("relay disconnected")
        return CriticReport(
            concept_id=sb.concept_id,
            iteration=iteration,
            overall_score=0.6,
            issues=[],
        )

    monkeypatch.setattr(
        "cg_tutor.agents.render_critic.inspect",
        fake_inspect,
    )

    report = inspect_ensemble(
        sb,
        tmp_path,
        backends=("good", "bad"),
        iteration=0,
        out_dir=tmp_path,
    )

    assert report.overall_score == 0.6
    assert report.execution_errors == ["bad: critic failed: relay disconnected"]
    assert not report.pass_threshold
    assert not report.has_block
    summary = json.loads((tmp_path / "critic_iter00.ensemble_summary.json").read_text())
    assert summary["execution_errors"]


def test_critic_ensemble_error_report_does_not_pollute_score(monkeypatch, tmp_path: Path):
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })

    def fake_inspect(sb, frames_dir, *, iteration=0, out_dir=None, backend=None,
                     narrative=None, scene_profile=None):
        if backend == "bad":
            return CriticReport(
                concept_id=sb.concept_id,
                iteration=iteration,
                overall_score=0.0,
                issues=[],
                execution_errors=["bad json"],
            )
        return CriticReport(
            concept_id=sb.concept_id,
            iteration=iteration,
            overall_score=0.7,
            issues=[],
        )

    monkeypatch.setattr(
        "cg_tutor.agents.render_critic.inspect",
        fake_inspect,
    )

    report = inspect_ensemble(
        sb,
        tmp_path,
        backends=("good", "bad"),
        iteration=0,
        out_dir=tmp_path,
    )

    assert report.overall_score == 0.7
    assert report.execution_errors == ["bad: bad json"]
    assert not report.pass_threshold
    summary = json.loads((tmp_path / "critic_iter00.ensemble_summary.json").read_text())
    assert summary["aggregate_score"] == 0.7
    assert summary["aggregate_execution_errors"] == ["bad: bad json"]
    assert summary["execution_errors"] == ["bad json"]


def test_critic_ensemble_partial_issue_report_still_participates(monkeypatch, tmp_path: Path):
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })
    issue = CriticIssue(
        shot_id="a",
        frame_idx=1,
        severity="block",
        category="concept_mismatch",
        issue="The mirror ray is missing.",
    )

    def fake_inspect(sb, frames_dir, *, iteration=0, out_dir=None, backend=None,
                     narrative=None, scene_profile=None):
        if backend == "partial":
            return CriticReport(
                concept_id=sb.concept_id,
                iteration=iteration,
                overall_score=0.4,
                issues=[issue],
                execution_errors=["trailing malformed shot json"],
            )
        return CriticReport(
            concept_id=sb.concept_id,
            iteration=iteration,
            overall_score=0.8,
            issues=[],
        )

    monkeypatch.setattr(
        "cg_tutor.agents.render_critic.inspect",
        fake_inspect,
    )

    report = inspect_ensemble(
        sb,
        tmp_path,
        backends=("partial", "clean"),
        iteration=0,
        out_dir=tmp_path,
        strictness="union",
    )

    assert len(report.issues) == 1
    assert report.execution_errors == ["partial: trailing malformed shot json"]
    assert not report.pass_threshold
    summary = json.loads(
        (tmp_path / "critic_iter00.member_usable_summary.json").read_text()
    )
    partial = next(m for m in summary["members"] if m["backend"] == "partial")
    assert partial["usable"] is True
    assert partial["participated_in_aggregate"] is True
    assert summary["partial_execution_errors"] == {
        "partial": ["trailing malformed shot json"]
    }


def test_inspect_ensemble_passthrough_writes_summary(tmp_path: Path):
    sb = Storyboard.model_validate({
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "a",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
            "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
        }],
    })

    report = inspect_ensemble(
        sb,
        tmp_path,
        backends=("passthrough", "passthrough"),
        iteration=0,
        out_dir=tmp_path,
    )

    assert report.overall_score == 1.0
    assert (tmp_path / "critic_iter00.ensemble_summary.json").exists()
