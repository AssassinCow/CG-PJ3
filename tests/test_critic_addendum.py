"""Unit tests for the critic-history helpers in pipeline.py.

These cover the pure-Python signal extraction (repeat offenders,
regressions, block-floor stale counter, scene param diff) without
needing Blender, an LLM, or any real critic output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cg_tutor.pipeline import (
    CriticIteration,
    _block_floor_stale_iters,
    _critic_counts,
    _critic_history_addendum,
    _critic_quality_key,
    _critic_quality_key_for,
    _flagged_counts,
    _missing_storyboard_objects,
    _multi_reference_retry_addendum,
    _regression_keys,
    _scene_param_diff,
    _semantic_counts,
    _shot_visual_contracts,
)
from cg_tutor.critic_loop import (
    CATEGORY_REPAIR_MESSAGES,
    _issue_action_hint,
    format_critic_visual_evidence_packet,
)
from cg_tutor.schemas import CriticIssue, CriticReport, Narrative, NarrativeNode, Storyboard


def _make_iter(
    iter_n: int,
    issues: list[tuple[str, str, str]],
    score: float = 0.5,
    scene_path: Path | None = None,
    execution_errors: list[str] | None = None,
    missing_objects: dict[str, list[str]] | None = None,
) -> CriticIteration:
    """Build a CriticIteration from (shot_id, category, severity) tuples."""
    return CriticIteration(
        iteration=iter_n,
        report=CriticReport(
            concept_id="test", iteration=iter_n, overall_score=score,
            issues=[
                CriticIssue(
                    shot_id=s, frame_idx=1, severity=sev, category=cat,
                    issue="x",
                )
                for s, cat, sev in issues
            ],
            execution_errors=execution_errors or [],
        ),
        scene_path=scene_path or Path("/tmp/_does_not_exist.py"),
        render_ok=True, n_frames=0, frames_hash="",
        missing_objects=missing_objects or {},
    )


def test_critic_visual_evidence_packet_includes_suggested_fix_json():
    item = CriticIteration(
        iteration=0,
        report=CriticReport(
            concept_id="test",
            iteration=0,
            overall_score=0.4,
            issues=[
                CriticIssue(
                    shot_id="node_01",
                    frame_idx=36,
                    severity="block",
                    category="concept_mismatch",
                    issue="camera icon is above the hero instead of in front.",
                    suggested_fix={
                        "camera_icon.location": [0.0, -4.0, 0.5],
                        "marker_posts.axis": "positive Y behind hero",
                    },
                )
            ],
            pass_blockers=["member_block_count>=3 (claude:5)"],
        ),
        scene_path=Path("/tmp/_does_not_exist.py"),
        render_ok=True,
        n_frames=0,
        frames_hash="",
    )

    text = format_critic_visual_evidence_packet([item])

    assert "CRITIC VISUAL EVIDENCE PACKET" in text
    assert "camera icon is above the hero" in text
    assert "camera_icon.location" in text
    assert "marker_posts.axis" in text
    assert "member_block_count>=3" in text


# ----- _flagged_counts ----------------------------------------------------


def test_flagged_counts_streak_single_iter():
    h = [_make_iter(0, [("s1", "off_screen", "block")])]
    assert _flagged_counts(h) == {("s1", "off_screen", "block"): 1}


def test_flagged_counts_streak_breaks_when_issue_gone():
    h = [
        _make_iter(0, [("s1", "off_screen", "block")]),
        _make_iter(1, []),  # resolved
        _make_iter(2, [("s1", "off_screen", "block")]),  # back, but streak = 1
    ]
    assert _flagged_counts(h) == {("s1", "off_screen", "block"): 1}


def test_flagged_counts_streak_three_in_a_row():
    key = ("s1", "off_screen", "block")
    h = [_make_iter(i, [key]) for i in range(3)]
    assert _flagged_counts(h)[key] == 3


# ----- _regression_keys ---------------------------------------------------


def test_regression_needs_three_iters():
    h = [
        _make_iter(0, [("s1", "off_screen", "block")]),
        _make_iter(1, [("s1", "off_screen", "block")]),
    ]
    assert _regression_keys(h) == set()


def test_regression_resolved_then_back():
    key = ("s1", "off_screen", "block")
    h = [
        _make_iter(0, [key]),
        _make_iter(1, []),  # resolved
        _make_iter(2, [key]),  # regressed
    ]
    assert _regression_keys(h) == {key}


def test_regression_excludes_persistent_keys():
    """A key in prev AND latest is persistent, not regressed."""
    key = ("s1", "off_screen", "block")
    h = [
        _make_iter(0, [key]),
        _make_iter(1, [key]),
        _make_iter(2, [key]),
    ]
    assert _regression_keys(h) == set()


# ----- _block_floor_stale_iters ------------------------------------------


def test_floor_empty_history():
    assert _block_floor_stale_iters([]) == (0, 0)


def test_floor_strictly_decreasing():
    counts = [5, 3, 2, 1]
    h = [_make_iter(i, [("s", "off_screen", "block")] * c)
         for i, c in enumerate(counts)]
    floor, stale = _block_floor_stale_iters(h)
    assert floor == 1
    assert stale == 0


def test_floor_stuck_for_two_iters():
    """Floor hit at iter1, then two iters at same/higher block count."""
    counts = [5, 3, 3, 4]
    h = [_make_iter(i, [("s", "off_screen", "block")] * c)
         for i, c in enumerate(counts)]
    floor, stale = _block_floor_stale_iters(h)
    assert floor == 3
    assert stale == 2  # iter2 and iter3 did not improve on iter1's floor


def test_floor_matches_bezier_trajectory():
    """Mirror the bezier run: blocks 5,3,4,2,3,1 → ends with floor=1, stale=0."""
    counts = [5, 3, 4, 2, 3, 1]
    h = [_make_iter(i, [("s", "off_screen", "block")] * c)
         for i, c in enumerate(counts)]
    floor, stale = _block_floor_stale_iters(h)
    assert floor == 1
    assert stale == 0


# ----- _scene_param_diff --------------------------------------------------


def test_scene_param_diff_extracts_camera_light_changes(tmp_path: Path):
    best = tmp_path / "scene.iter01.py"
    latest = tmp_path / "scene.iter03.py"
    best.write_text(
        "import bpy\n"
        "cam.location = (0, -8, 4)\n"
        "key_light.energy = 1500\n"
        "unrelated = 42\n"
    )
    latest.write_text(
        "import bpy\n"
        "cam.location = (0, -12, 6)\n"  # changed
        "key_light.energy = 1500\n"     # unchanged
        "unrelated = 99\n"               # not a param keyword → ignored
    )
    b = _make_iter(1, [], scene_path=best)
    l = _make_iter(3, [], scene_path=latest)
    out = _scene_param_diff(b, l)
    assert "cam.location = (0, -8, 4)" in out
    assert "cam.location = (0, -12, 6)" in out
    assert "REMOVED vs best iter01" in out
    assert "ADDED in latest iter03" in out
    assert "unrelated" not in out


def test_scene_param_diff_empty_when_same_iter():
    b = _make_iter(2, [])
    assert _scene_param_diff(b, b) == ""


def test_scene_param_diff_empty_when_no_diff(tmp_path: Path):
    best = tmp_path / "a.py"
    latest = tmp_path / "b.py"
    best.write_text("cam.location = (0, -8, 4)\n")
    latest.write_text("cam.location = (0, -8, 4)\n")
    b = _make_iter(1, [], scene_path=best)
    l = _make_iter(3, [], scene_path=latest)
    assert _scene_param_diff(b, l) == ""


# ----- _critic_history_addendum integration -------------------------------


def test_addendum_emits_no_progress_after_stale_floor():
    counts = [3, 2, 2, 2]
    h = [_make_iter(i, [("s", "off_screen", "block")] * c)
         for i, c in enumerate(counts)]
    out = _critic_history_addendum(h)
    assert "NO PROGRESS" in out
    assert "block floor = 2" in out


def test_addendum_skips_no_progress_when_improving():
    counts = [5, 3, 2]
    h = [_make_iter(i, [("s", "off_screen", "block")] * c)
         for i, c in enumerate(counts)]
    out = _critic_history_addendum(h)
    assert "NO PROGRESS" not in out


def test_addendum_includes_param_diff_under_regressions(tmp_path: Path):
    """Param diff should attach to the REGRESSIONS section when it fires."""
    best_scene = tmp_path / "scene.iter01.py"
    latest_scene = tmp_path / "scene.iter02.py"
    best_scene.write_text("camera.location = (0, 0, 10)\n")
    latest_scene.write_text("camera.location = (0, 0, 3)\n")
    key = ("s1", "off_screen", "block")
    h = [
        _make_iter(0, [key]),
        _make_iter(1, [], scene_path=best_scene),  # best (0 blocks)
        _make_iter(2, [key], scene_path=latest_scene),  # regressed
    ]
    out = _critic_history_addendum(h)
    assert "REGRESSIONS" in out
    assert "camera.location = (0, 0, 10)" in out
    assert "camera.location = (0, 0, 3)" in out


def test_addendum_empty_for_empty_history():
    assert _critic_history_addendum([]) == ""


# ----- concept_mismatch (defence line 1) ----------------------------------


def test_critic_counts_excludes_concept_mismatch():
    """Framing block/warn counts must ignore concept_mismatch issues so
    they cannot dominate best-of-N selection."""
    it = _make_iter(0, [
        ("s1", "off_screen", "block"),
        ("s2", "concept_mismatch", "block"),
        ("s3", "concept_mismatch", "warn"),
    ])
    assert _critic_counts(it.report) == (1, 0)


def test_semantic_counts_only_concept_mismatch():
    it = _make_iter(0, [
        ("s1", "off_screen", "block"),
        ("s2", "concept_mismatch", "block"),
        ("s3", "concept_mismatch", "warn"),
        ("s4", "lighting", "warn"),
    ])
    assert _semantic_counts(it.report) == (1, 1)


def test_quality_key_unaffected_by_concept_mismatch():
    """Iter with framing-block=0 must outrank iter with framing-block=1
    even if it has more concept_mismatch issues. This is the property
    that keeps semantic noise from destabilising iter selection."""
    iter_clean_framing = _make_iter(0, [
        ("s1", "concept_mismatch", "block"),
        ("s2", "concept_mismatch", "block"),
    ], score=0.5)
    iter_framing_block = _make_iter(1, [
        ("s1", "off_screen", "block"),
    ], score=0.9)
    # Larger key is better. Iter 0 (no framing blocks) wins.
    assert _critic_quality_key(iter_clean_framing) > _critic_quality_key(iter_framing_block)


def test_block_floor_unaffected_by_concept_mismatch():
    """NO PROGRESS early-stop must NOT trigger just because concept_mismatch
    is persistent — that's a different fix path and shouldn't kill the
    framing-improvement loop."""
    h = [
        _make_iter(0, [
            ("s1", "off_screen", "block"),
            ("s2", "concept_mismatch", "block"),
        ]),
        _make_iter(1, [
            # framing block drops to 0; concept_mismatch persists
            ("s2", "concept_mismatch", "block"),
        ]),
        _make_iter(2, [
            ("s2", "concept_mismatch", "block"),
        ]),
    ]
    floor, stale = _block_floor_stale_iters(h)
    assert floor == 0  # framing floor went to 0 at iter 1
    assert stale == 1  # iter 2 didn't lower it further, but only 1 stale iter


def test_addendum_lists_concept_mismatch_section():
    """The CONCEPT MISMATCH callout fires when the latest iter has any
    concept_mismatch issue, instructing the coder it's a different fix
    path from framing."""
    h = [
        _make_iter(0, [("s1", "off_screen", "block")], score=0.5),
        _make_iter(1, [
            ("s1", "off_screen", "block"),
            ("s2", "concept_mismatch", "block"),
        ], score=0.5),
    ]
    out = _critic_history_addendum(h)
    assert "CONCEPT MISMATCH" in out
    assert "(s2, concept_mismatch, block)" in out
    # The directive that distinguishes it from framing fixes:
    assert "camera" in out.lower() and "geometry" in out.lower()


def test_addendum_concept_mismatch_flows_through_repeat_offenders():
    """A concept_mismatch flagged in two consecutive iters must show up
    in REPEAT OFFENDERS — the existing 4 detections (resolved / new /
    repeat / regression) cover the new category automatically through
    _issue_key."""
    key = ("s2", "concept_mismatch", "block")
    h = [
        _make_iter(0, [key], score=0.6),
        _make_iter(1, [key], score=0.6),
    ]
    out = _critic_history_addendum(h)
    assert "REPEAT OFFENDERS" in out
    assert "(s2, concept_mismatch, block)" in out


def test_addendum_concept_mismatch_flows_through_regressions():
    """concept_mismatch resolved earlier and re-appearing must show up in
    REGRESSIONS (again, free via _issue_key being category-aware)."""
    cm = ("s2", "concept_mismatch", "block")
    fr = ("s1", "off_screen", "block")
    h = [
        _make_iter(0, [cm, fr], score=0.5),     # cm present
        _make_iter(1, [fr], score=0.55),         # cm resolved
        _make_iter(2, [cm, fr], score=0.55),     # cm back -> regression
    ]
    out = _critic_history_addendum(h)
    assert "REGRESSIONS" in out
    assert "(s2, concept_mismatch, block)" in out


def test_addendum_score_line_labels_framing():
    """Score-line breakdown should call framing counts out by name so
    the coder can't confuse them with concept_mismatch counts."""
    h = [
        _make_iter(0, [("s1", "off_screen", "block")], score=0.7),
    ]
    out = _critic_history_addendum(h)
    assert "framing block=1" in out


def test_addendum_shows_concept_mismatch_score_line_when_present():
    h = [
        _make_iter(0, [
            ("s1", "off_screen", "block"),
            ("s2", "concept_mismatch", "block"),
            ("s3", "concept_mismatch", "warn"),
        ], score=0.7),
    ]
    out = _critic_history_addendum(h)
    assert "concept_mismatch" in out.lower()
    assert "semantic axis" in out  # the "not in score" disclaimer


def test_addendum_retry_targets_come_from_best_iteration():
    """When latest regresses, retry priorities should improve the best
    scene, not chase blocks that only exist in the worse latest scene."""
    h = [
        _make_iter(0, [
            ("s1", "off_screen", "block"),
            ("s2", "overlay_collision", "block"),
        ], score=0.4),
        _make_iter(1, [
            ("s2", "overlay_collision", "block"),
        ], score=0.7),  # best by fewer framing blocks
        _make_iter(2, [
            ("s1", "off_screen", "block"),
            ("s2", "overlay_collision", "block"),
            ("s3", "occlusion", "block"),
        ], score=0.5),
    ]

    out = _critic_history_addendum(h, shot_ids=["s1", "s2", "s3"])

    assert "Start from best iter01" in out
    assert "TOP RETRY TARGETS (from best iteration" in out
    assert "- s2: overlay_collision block" in out
    assert "- s3: occlusion block" not in out


def test_addendum_retry_targets_include_semantic_blocks():
    h = [
        _make_iter(0, [
            ("s1", "concept_mismatch", "block"),
        ], score=0.7),
    ]

    out = _critic_history_addendum(h, shot_ids=["s1"])

    assert "- s1: concept_mismatch block" in out
    assert "semantic fix" in out
    assert "visual_intent" in out


def test_addendum_soft_freezes_clean_shots_from_best_iteration():
    h = [
        _make_iter(0, [
            ("s1", "off_screen", "block"),
        ], score=0.4),
        _make_iter(1, [
            ("s1", "off_screen", "block"),
        ], score=0.4),
    ]

    out = _critic_history_addendum(h, shot_ids=["s1", "s2"])

    assert "SOFT FREEZE / PRESERVE" in out
    assert "- s2: preserve camera" in out
    assert "- s1: preserve camera" not in out


def test_shot_visual_contracts_are_generated_from_narrative_and_storyboard():
    narrative = Narrative(
        concept_id="demo",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="One",
                description="Explain the visible construction.",
                formulas=["a+b"],
                duration_sec=1.0,
                visual_intent="A blue curve passes through three labelled control points.",
            )
        ],
    )
    storyboard = Storyboard.model_validate({
        "concept_id": "demo",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -5, 3],
                "look_at": [0, 0, 0],
                "fov": 50,
            }],
            "objects": [{
                "name": "curve",
                "type": "curve",
                "primitive": "curve_polyline",
                "location": [0, 0, 0],
            }],
            "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
            "formula": "a+b",
            "caption": "Curve",
        }],
    })

    out = _shot_visual_contracts(narrative, storyboard)

    assert "SHOT VISUAL CONTRACTS" in out
    assert "A blue curve passes through three labelled control points" in out
    assert "expected visible objects: curve" in out


# ----- _missing_storyboard_objects (P0-A) ---------------------------------


def _make_storyboard_with_objects(
    shot_objects: list[tuple[str, list[str]]],
) -> Storyboard:
    """Helper: build a Storyboard with arbitrary shot/object name pairs."""
    shots = []
    cursor = 0.0
    duration = 1.0
    for shot_id, names in shot_objects:
        shots.append({
            "node_id": shot_id,
            "start_sec": cursor,
            "duration_sec": duration,
            "camera": [{
                "time_sec": cursor, "position": [0, -5, 3],
                "look_at": [0, 0, 0], "fov": 50,
            }],
            "objects": [
                {"name": n, "type": "mesh", "primitive": "sphere",
                 "location": [0, 0, 0]}
                for n in names
            ],
        })
        cursor += duration
    return Storyboard.model_validate({
        "concept_id": "demo",
        "fps": 24,
        "resolution": [960, 540],
        "shots": shots,
    })


def test_missing_objects_empty_when_all_named():
    sb = _make_storyboard_with_objects([("s1", ["a", "b"])])
    code = "import bpy\nobj = bpy.context.object\nobj.name = 'a'\nobj.name = 'b'\n"
    assert _missing_storyboard_objects(code, sb) == {}


def test_missing_objects_reports_per_shot_gaps():
    sb = _make_storyboard_with_objects([
        ("s1", ["bezier_trace", "ctrl_polygon"]),
        ("s2", ["normal_arrow"]),
    ])
    # ctrl_polygon and normal_arrow are absent
    code = "obj.name = 'bezier_trace'\n# only this one named\n"
    out = _missing_storyboard_objects(code, sb)
    assert out == {"s1": ["ctrl_polygon"], "s2": ["normal_arrow"]}


def test_missing_objects_does_substring_not_word_boundary():
    """If the coder embeds the name in a longer string we accept it —
    over-permissive is fine here because false negatives (missing object
    reported as present) just degrade to vision-critic-catches-it, which
    is the current baseline."""
    sb = _make_storyboard_with_objects([("s1", ["arrow"])])
    code = "obj.name = 'normal_arrow_x'\n"
    assert _missing_storyboard_objects(code, sb) == {}


def test_missing_objects_ignores_blank_names():
    sb = _make_storyboard_with_objects([("s1", [" ", "real_obj"])])
    code = "obj.name = 'real_obj'\n"
    assert _missing_storyboard_objects(code, sb) == {}


# ----- addendum surfaces deterministic signals ---------------------------


def test_addendum_lists_missing_objects_section():
    h = [_make_iter(
        0, [], score=0.5,
        missing_objects={"s1": ["bezier_trace", "ctrl_polygon"]},
    )]
    out = _critic_history_addendum(h)
    assert "MISSING FROM SCENE.PY" in out
    assert "s1: bezier_trace, ctrl_polygon" in out
    assert "obj.name" in out  # the directive references the exact fix


def test_addendum_lists_execution_errors_section():
    h = [_make_iter(
        0, [], score=0.5,
        execution_errors=["s1: api vision critic failed: Connection error."],
    )]
    out = _critic_history_addendum(h)
    assert "CRITIC EXECUTION ERRORS" in out
    assert "Connection error" in out
    assert "Treat these shots as un-evaluated" in out


def test_addendum_empty_when_no_missing_or_errors():
    h = [_make_iter(0, [("s1", "off_screen", "block")])]
    out = _critic_history_addendum(h)
    assert "MISSING FROM SCENE.PY" not in out
    assert "CRITIC EXECUTION ERRORS" not in out


# ----- P1-D: execution_errors do NOT inflate framing counts ---------------


def test_critic_counts_unaffected_by_execution_errors():
    """The relay-failure path used to synthesize fake `category="other"
    severity="warn"` issues, which then counted as a framing warn and
    dropped the score. After the fix, those go to execution_errors and
    framing counts only reflect real critic findings."""
    report = CriticReport(
        concept_id="x", iteration=0, overall_score=0.7,
        issues=[
            CriticIssue(shot_id="s1", frame_idx=1, severity="block",
                        category="off_screen", issue="x"),
        ],
        execution_errors=[
            "s2: api vision critic failed: Connection error.",
            "s3: claude-cli returned unparseable JSON",
        ],
    )
    assert _critic_counts(report) == (1, 0)
    assert _semantic_counts(report) == (0, 0)


# ----- P1-E: cm tie-breaker in _critic_quality_key ------------------------


def test_quality_key_prefers_fewer_concept_mismatch_when_framing_ties():
    """Two iters with identical framing AND identical score should be
    tie-broken on concept_mismatch (fewer = better)."""
    a = _make_iter(
        0, [("s1", "concept_mismatch", "block")], score=0.7,
    )
    b = _make_iter(
        1, [("s1", "concept_mismatch", "block"),
            ("s2", "concept_mismatch", "block")], score=0.7,
    )
    # a has fewer cm-blocks → a wins
    assert _critic_quality_key(a) > _critic_quality_key(b)


def test_quality_key_framing_still_dominates_over_cm():
    """The cm tie-breaker must NOT override framing — that is the
    invariant test_quality_key_unaffected_by_concept_mismatch already
    asserts; re-check it through the new key shape."""
    framing_clean = _make_iter(
        0, [("s1", "concept_mismatch", "block")] * 5, score=0.5,
    )
    framing_dirty = _make_iter(
        1, [("s1", "off_screen", "block")], score=0.9,
    )
    assert _critic_quality_key(framing_clean) > _critic_quality_key(framing_dirty)


def test_quality_key_prefers_lower_cross_ref_actionable():
    """A2 regression: cross_ref_actionable_count is populated on CriticIteration
    and exported to JSON, but was previously ignored by _critic_quality_key.
    With everything else tied, an iter with fewer double-corroborated cross-ref
    findings must outrank one with more — cross-ref is stricter than vision
    score because it's vision-AND-AST agreement, not subjective."""
    fewer = _make_iter(0, [], score=0.6)
    fewer.cross_ref_actionable_count = 1
    more = _make_iter(1, [], score=0.6)
    more.cross_ref_actionable_count = 5

    assert _critic_quality_key(fewer) > _critic_quality_key(more)
    # balanced / semantic modes also honour the tier
    assert (
        _critic_quality_key_for(fewer, "balanced")
        > _critic_quality_key_for(more, "balanced")
    )
    assert (
        _critic_quality_key_for(fewer, "semantic")
        > _critic_quality_key_for(more, "semantic")
    )


def test_quality_key_success_hard_still_dominates_cross_ref():
    """cross_ref sits BELOW success_hard in the lex order — an iter with 0
    cross-ref findings but a success-hard violation must still lose to one
    with many cross-ref findings but no success-hard."""
    success_hard_dirty = _make_iter(0, [], score=0.9)
    success_hard_dirty.metric_success_hard_count = 1
    success_hard_dirty.cross_ref_actionable_count = 0

    cross_ref_dirty = _make_iter(1, [], score=0.5)
    cross_ref_dirty.cross_ref_actionable_count = 10

    assert _critic_quality_key(cross_ref_dirty) > _critic_quality_key(success_hard_dirty)


def test_quality_key_metric_block_beats_aesthetic_score_when_framing_ties():
    metric_clean_lower_score = _make_iter(0, [], score=0.55)
    metric_dirty_higher_score = _make_iter(1, [], score=0.95)
    metric_dirty_higher_score.metric_block_count = 1
    metric_dirty_higher_score.metric_success_hard_count = 1

    assert (
        _critic_quality_key_for(metric_clean_lower_score, "balanced")
        > _critic_quality_key_for(metric_dirty_higher_score, "balanced")
    )


def test_quality_key_success_hard_beats_aesthetic_score():
    clean_lower_score = _make_iter(0, [], score=0.45)
    hard_failed_high_score = _make_iter(1, [], score=0.99)
    hard_failed_high_score.metric_block_count = 1
    hard_failed_high_score.metric_success_hard_count = 1

    assert (
        _critic_quality_key_for(clean_lower_score, "balanced")
        > _critic_quality_key_for(hard_failed_high_score, "balanced")
    )


def test_quality_key_prefers_llm_over_degraded_fallback_when_hard_ties():
    llm_candidate = _make_iter(
        0, [("s1", "off_screen", "block")], score=0.3,
    )
    fallback_candidate = _make_iter(1, [], score=0.95)
    fallback_candidate.scene_origin = "compiled_fallback"
    fallback_candidate.fallback_degraded = True

    assert (
        _critic_quality_key_for(llm_candidate, "balanced")
        > _critic_quality_key_for(fallback_candidate, "balanced")
    )


def test_quality_key_success_soft_does_not_beat_success_hard_cleanliness():
    soft_failed_lower_score = _make_iter(0, [], score=0.4)
    soft_failed_lower_score.metric_success_soft_count = 3
    hard_failed_high_score = _make_iter(1, [], score=0.95)
    hard_failed_high_score.metric_success_hard_count = 1

    assert (
        _critic_quality_key_for(soft_failed_lower_score, "balanced")
        > _critic_quality_key_for(hard_failed_high_score, "balanced")
    )


def test_balanced_quality_key_lets_semantic_break_framing_ties():
    sem_clean = _make_iter(
        0, [("s1", "lighting", "warn")], score=0.7,
    )
    sem_dirty_high_score = _make_iter(
        1, [
            ("s1", "lighting", "warn"),
            ("s2", "concept_mismatch", "block"),
        ], score=0.9,
    )

    assert (
        _critic_quality_key_for(sem_clean, "balanced")
        > _critic_quality_key_for(sem_dirty_high_score, "balanced")
    )


def test_balanced_quality_key_prefers_semantic_clean_over_success_soft_only():
    sem_clean_soft = _make_iter(
        0, [("s1", "lighting", "warn")], score=0.75,
    )
    sem_clean_soft.metric_block_count = 1
    sem_clean_soft.metric_success_soft_count = 1
    sem_dirty_high_score = _make_iter(
        1,
        [("s1", "concept_mismatch", "block")] * 10,
        score=0.9,
    )

    assert (
        _critic_quality_key_for(sem_clean_soft, "balanced")
        > _critic_quality_key_for(sem_dirty_high_score, "balanced")
    )


def test_balanced_quality_key_prefers_fewer_total_blocks_over_framing_only():
    rougher_semantic_better = _make_iter(
        0,
        [("s1", "off_screen", "block")] * 3
        + [("s2", "concept_mismatch", "block")] * 8,
        score=0.57,
    )
    framing_clean_semantic_worse = _make_iter(
        1,
        [("s2", "concept_mismatch", "block")] * 13,
        score=0.8,
    )

    assert (
        _critic_quality_key_for(rougher_semantic_better, "balanced")
        > _critic_quality_key_for(framing_clean_semantic_worse, "balanced")
    )


def test_balanced_quality_key_prefers_fewer_semantic_warns_before_score():
    semantic_clean_lower_score = _make_iter(
        0, [("s1", "lighting", "warn")], score=0.55,
    )
    semantic_warn_high_score = _make_iter(
        1, [("s1", "concept_mismatch", "warn")], score=0.95,
    )

    assert (
        _critic_quality_key_for(semantic_clean_lower_score, "balanced")
        > _critic_quality_key_for(semantic_warn_high_score, "balanced")
    )


def test_semantic_quality_key_counts_total_blocks_first():
    semantic_clean_lower_score = _make_iter(
        0, [("s1", "lighting", "warn")], score=0.5,
    )
    semantic_dirty_higher_score = _make_iter(
        1, [("s1", "concept_mismatch", "block")], score=0.9,
    )

    assert (
        _critic_quality_key_for(semantic_clean_lower_score, "semantic")
        > _critic_quality_key_for(semantic_dirty_higher_score, "semantic")
    )


def test_multi_reference_addendum_uses_safe_base_and_refs(tmp_path: Path):
    safe_scene = tmp_path / "scene.iter01.py"
    sem_scene = tmp_path / "scene.iter02.py"
    safe_scene.write_text("camera.location = (0, 0, 10)\nkey_light.energy = 500\n")
    sem_scene.write_text("camera.location = (0, 0, 6)\nkey_light.energy = 900\n")

    safe = _make_iter(
        1,
        [("s1", "concept_mismatch", "block")] * 4,
        score=0.55,
        scene_path=safe_scene,
    )
    semantic_ref = _make_iter(
        2,
        [("s1", "off_screen", "block"), ("s1", "concept_mismatch", "block")],
        score=0.85,
        scene_path=sem_scene,
    )

    out = _multi_reference_retry_addendum([safe, semantic_ref])

    assert "framing_safe_base: iter01" in out
    assert "semantic_reference: iter02" in out
    assert "Use the framing_safe_base scene as the ONLY patch/edit base" in out
    assert "SAFE_BASE ↔ SEMANTIC_REFERENCE PARAM DIFF" in out
    assert "camera.location = (0, 0, 10)" in out
    assert "camera.location = (0, 0, 6)" in out


def test_quality_key_for_rejects_unknown_mode():
    it = _make_iter(0, [("s1", "off_screen", "block")])
    with pytest.raises(ValueError, match="unknown best selection mode"):
        _critic_quality_key_for(it, "lexicographic")  # type: ignore[arg-type]


# ----- contract_anchors_ok tie-breaker ------------------------------------


def test_quality_key_breaks_tie_on_contract_anchors_ok():
    """When two iters have identical critic results, the one whose
    deterministic anchor check passed should win. The anchor signal must
    NOT override any earlier priority — that is the next test."""
    anchors_ok = _make_iter(0, [], score=0.8)
    anchors_ok.contract_anchors_ok = True
    anchors_missing = _make_iter(1, [], score=0.8)
    anchors_missing.contract_anchors_ok = False

    assert _critic_quality_key(anchors_ok) > _critic_quality_key(anchors_missing)


def test_quality_key_anchors_does_not_override_framing():
    """A framing-clean iter MUST beat a framing-blocked one even if the
    blocked one has anchors_ok. Otherwise scenes with anchors but bad
    framing would erroneously be selected."""
    framing_clean_no_anchors = _make_iter(0, [], score=0.5)
    framing_clean_no_anchors.contract_anchors_ok = False
    framing_dirty_with_anchors = _make_iter(
        1, [("s1", "off_screen", "block")], score=0.9,
    )
    framing_dirty_with_anchors.contract_anchors_ok = True

    assert (
        _critic_quality_key(framing_clean_no_anchors)
        > _critic_quality_key(framing_dirty_with_anchors)
    )


def test_quality_key_anchors_does_not_override_score():
    """Anchor tie-breaker is below score in the priority order."""
    higher_score_no_anchors = _make_iter(0, [], score=0.9)
    higher_score_no_anchors.contract_anchors_ok = False
    lower_score_with_anchors = _make_iter(1, [], score=0.5)
    lower_score_with_anchors.contract_anchors_ok = True

    assert (
        _critic_quality_key(higher_score_no_anchors)
        > _critic_quality_key(lower_score_with_anchors)
    )


# ----- CATEGORY_REPAIR_MESSAGES registry (P1-4 consolidation) -------------


def test_category_repair_messages_every_entry_has_required_fields():
    """The registry is the single source of truth for both the critic-
    history hint (read by _issue_action_hint) and the repair-plan action
    text (read by repair_plan._action_for_category). Every entry must
    define both keys so adding a category can't silently break either
    consumer."""
    assert "other" in CATEGORY_REPAIR_MESSAGES, (
        "'other' is the fallback bucket — must always exist"
    )
    for category, entry in CATEGORY_REPAIR_MESSAGES.items():
        assert "hint" in entry, f"{category} missing 'hint'"
        assert "action_base" in entry, f"{category} missing 'action_base'"
        assert entry["hint"].strip(), f"{category} hint is empty"
        assert entry["action_base"].strip(), (
            f"{category} action_base is empty"
        )


def test_category_repair_messages_covers_known_framing_categories():
    """The categories the verifier and critic both produce should all
    have explicit entries (not just fall through to 'other'), so the
    coder gets a category-specific hint instead of the generic one."""
    from cg_tutor.schemas import FRAMING_CATEGORIES

    for category in FRAMING_CATEGORIES:
        assert category in CATEGORY_REPAIR_MESSAGES, (
            f"framing category {category} should have a dedicated entry "
            f"in CATEGORY_REPAIR_MESSAGES, not fall back to 'other'"
        )


def test_issue_action_hint_dispatches_via_registry():
    """_issue_action_hint must return the 'hint' field from the registry
    for a known category, and fall back to 'other' for unknown ones.
    Unknown categories use a lightweight stub since CriticIssue's
    pydantic Literal would reject them at construction time."""
    from types import SimpleNamespace

    off_screen_issue = CriticIssue(
        shot_id="s1", frame_idx=1, severity="block",
        category="off_screen", issue="x",
    )
    assert (
        _issue_action_hint(off_screen_issue)
        == CATEGORY_REPAIR_MESSAGES["off_screen"]["hint"]
    )

    unknown = SimpleNamespace(category="frobnicate")
    assert (
        _issue_action_hint(unknown)
        == CATEGORY_REPAIR_MESSAGES["other"]["hint"]
    )
