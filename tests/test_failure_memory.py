from pathlib import Path

from cg_tutor.critic_loop import CriticIteration
from cg_tutor.failure_memory import (
    append_failure_memory,
    format_failure_memory_for_coder,
    load_failure_memory,
    memory_from_history,
)
from cg_tutor.schemas import CriticIssue, CriticReport


def _history():
    return [
        CriticIteration(
            iteration=0,
            report=CriticReport(
                concept_id="demo",
                iteration=0,
                overall_score=0.5,
                issues=[
                    CriticIssue(
                        shot_id="node_01",
                        frame_idx=1,
                        severity="block",
                        category="concept_mismatch",
                        issue="Curve is missing.",
                    )
                ],
            ),
            scene_path=Path("/tmp/scene.py"),
            render_ok=True,
            n_frames=1,
            missing_objects={"node_01": ["curve"]},
        )
    ]


def test_failure_memory_extracts_critic_and_object_entries():
    entries = memory_from_history("demo", _history())

    categories = {e.category for e in entries}
    assert "concept_mismatch" in categories
    assert "missing_storyboard_objects" in categories


def test_failure_memory_extracts_critic_grounding_constraints():
    history = [
        CriticIteration(
            iteration=0,
            report=CriticReport(
                concept_id="demo",
                iteration=0,
                overall_score=0.5,
                issues=[
                    CriticIssue(
                        shot_id="node_01",
                        frame_idx=1,
                        severity="block",
                        category="concept_mismatch",
                        issue="object_point_B label is missing and x = f X/Z is absent.",
                        suggested_fix={
                            "object_point_b.label.visible": True,
                            "formula.text": "x = f X/Z",
                        },
                    )
                ],
            ),
            scene_path=Path("/tmp/scene.py"),
            render_ok=True,
            n_frames=1,
        )
    ]

    entries = memory_from_history("demo", history)

    grounding = [
        e for e in entries
        if e.category == "critic_grounding_constraints"
    ]
    assert grounding
    assert "object_point_b" in grounding[0].issue


def test_failure_memory_round_trips_jsonl(tmp_path: Path):
    path = tmp_path / "memory.jsonl"
    append_failure_memory(path, memory_from_history("demo", _history()))

    entries = load_failure_memory(path, "demo")

    assert entries
    assert all(e.concept_id == "demo" for e in entries)


def test_failure_memory_formats_for_coder():
    out = format_failure_memory_for_coder(memory_from_history("demo", _history()))

    assert "STRUCTURED FAILURE MEMORY" in out
    assert "Curve is missing" in out
