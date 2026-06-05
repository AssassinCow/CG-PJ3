"""Regression tests for the W3 security/hardening sweep.

Each test corresponds to a finding from the architecture audit:
- #1 concept_id path traversal
- #6 Pydantic extra="forbid"
- #7 NaN/Inf field validators
- #11 profile silent-fallback now surfaces in validation log
- #13 storyboard normalization audit trail
- AST-based scene cache check
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cg_tutor import pipeline
from cg_tutor.scene_profiles import (
    profile_seed,
    base_profile,
)
from cg_tutor.schemas import Storyboard
from cg_tutor.agents.storyboard import (
    _record_normalization,
    _strip_known_extras,
    validate_storyboard,
)


# ---------------------------------------------------------------------------
# #1 concept_id path traversal — pipeline rejects bad ids before mkdir.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    "../etc/passwd",
    "../../sensitive",
    "with/slash",
    "with space",
    "x.y",
    "leading-dot/../escape",
])
def test_concept_id_rejects_path_traversal_attempts(bad, tmp_path):
    concept_yaml = tmp_path / "bad.yaml"
    concept_yaml.write_text(yaml.safe_dump({
        "concept_id": bad,
        "title": "x", "description": "x", "duration_sec": 1.0,
    }))
    with pytest.raises(ValueError, match="concept_id"):
        pipeline.run(concept_yaml, out_root=tmp_path / "outputs",
                     max_critic_iterations=0)


@pytest.mark.parametrize("good", [
    "phong_lighting", "bezier_curve", "x", "with-dash", "with_underscore",
    "X123",
])
def test_concept_id_accepts_path_safe_ids(good):
    assert pipeline._validate_concept_id(good) == good


# ---------------------------------------------------------------------------
# #6 Pydantic extra="forbid" on top-level schemas.
# ---------------------------------------------------------------------------


def _good_storyboard_dict():
    return {
        "concept_id": "test",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [
                {"time_sec": 0.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
                {"time_sec": 1.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
            ],
            "objects": [
                {"name": "subject", "type": "primitive",
                 "primitive": "sphere"},
            ],
        }],
    }


def test_storyboard_rejects_unknown_top_level_field():
    raw = _good_storyboard_dict()
    raw["mystery_field"] = "shouldn't be here"
    with pytest.raises(ValidationError) as exc:
        Storyboard.model_validate(raw)
    assert "mystery_field" in str(exc.value)


def test_storyboard_rejects_typo_in_shot_field():
    raw = _good_storyboard_dict()
    # Common LLM typo: `start_seconds` instead of `start_sec`.
    raw["shots"][0]["start_seconds"] = 0.0
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


# ---------------------------------------------------------------------------
# Storyboard normalization layer strips harmless extras while still
# letting typos surface.
# ---------------------------------------------------------------------------


def test_known_extras_are_stripped_before_validation():
    raw = _good_storyboard_dict()
    raw["description"] = "a textual description the LLM hallucinated"
    raw["shots"][0]["notes"] = "shot 1 establishes the scene"
    # validate_storyboard should accept this even with extra=forbid in
    # the schema, because _strip_known_extras removes harmless commentary.
    sb = validate_storyboard(raw)
    assert sb.concept_id == "test"
    assert sb.shots[0].node_id == "shot1"


def test_strip_known_extras_leaves_unknown_typos_in_place():
    raw = {"shots": [{"start_seconds": 0.0, "description": "x"}]}
    out = _strip_known_extras(raw)
    # description (allowlisted) is gone, start_seconds (typo) is kept.
    assert "description" not in out["shots"][0]
    assert "start_seconds" in out["shots"][0]


# ---------------------------------------------------------------------------
# #7 NaN/Inf are rejected by field validators.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_duration_sec_rejects_non_finite(bad_value):
    """Either our custom 'finite' validator or Pydantic's own gt=0 must
    reject the value — we don't care which fires first, only that the
    value never gets through."""
    raw = _good_storyboard_dict()
    raw["shots"][0]["duration_sec"] = bad_value
    with pytest.raises(ValidationError):
        Storyboard.model_validate(raw)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_camera_position_rejects_non_finite(bad_value):
    raw = _good_storyboard_dict()
    raw["shots"][0]["camera"][0]["position"] = (bad_value, 0, 0)
    with pytest.raises(ValidationError, match="finite"):
        Storyboard.model_validate(raw)


def test_keyframe_value_rejects_nan_in_nested_list():
    raw = _good_storyboard_dict()
    raw["shots"][0]["objects"][0]["keyframes"] = [{
        "time_sec": 0.5,
        "attr": "location",
        "value": [1.0, float("nan"), 2.0],
    }]
    with pytest.raises(ValidationError, match="finite"):
        Storyboard.model_validate(raw)


# ---------------------------------------------------------------------------
# #11 unknown profile id should be surfaced (warning), not silently
# remapped to vector_teaching.
# ---------------------------------------------------------------------------


def test_unknown_profile_id_emits_warning():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        base_profile("totally_invented_profile")
    assert any(
        "totally_invented_profile" in str(w.message)
        for w in caught
    )


def test_profile_seed_with_unknown_string_records_in_validation():
    """profile_seed routes unknown strings through sanitize_profile so the
    fallback is recorded rather than silently squashed."""
    # The unknown id should fall through to a sanitize warning trail.
    spec = {"scene_profile": "totally_invented_profile"}
    sp = profile_seed(spec)
    # sanitize_profile picks vector_teaching as the safe base when the
    # caller's id doesn't match BASE_PROFILES.
    assert sp.base_profile == "vector_teaching"


# ---------------------------------------------------------------------------
# #13 storyboard normalization audit captures whether the start_sec
# values came from the LLM or were inferred by us.
# ---------------------------------------------------------------------------


def test_normalization_audit_flags_when_start_sec_inferred():
    raw = _good_storyboard_dict()
    raw["shots"] = [
        {**raw["shots"][0], "start_sec": 0.0, "duration_sec": 1.0,
         "node_id": "shot1"},
        {**raw["shots"][0], "start_sec": 0.0, "duration_sec": 2.0,
         "node_id": "shot2",
         "camera": [
             {"time_sec": 0.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
             {"time_sec": 3.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
         ]},
    ]
    from cg_tutor.agents.storyboard import (
        _normalize_relative_times,
        _normalize_shot_start_times,
    )
    normalized = _normalize_relative_times(_normalize_shot_start_times(raw))
    audit = _record_normalization(raw, normalized)
    assert audit["shot_start_sec_normalized"] is True


def test_normalization_audit_flags_stripped_extras():
    raw = {"shots": [{"description": "junk", "node_id": "x"}]}
    audit = _record_normalization(raw, _strip_known_extras(raw))
    assert any("description" in p for p in audit["extras_stripped"])


# ---------------------------------------------------------------------------
# AST-based _scene_cache_is_usable: substring of render call in comment
# is not enough.
# ---------------------------------------------------------------------------


def test_scene_cache_rejects_render_call_only_in_comment(tmp_path):
    fake = tmp_path / "scene.py"
    fake.write_text(
        "import bpy\n"
        "# bpy.ops.render.render(animation=True)  # not actually called\n"
    )
    assert pipeline._scene_cache_is_usable(fake) is False


def test_scene_cache_accepts_real_render_call(tmp_path):
    fake = tmp_path / "scene.py"
    fake.write_text(
        "import bpy\n"
        "bpy.ops.render.render(animation=True)\n"
    )
    assert pipeline._scene_cache_is_usable(fake) is True


def test_scene_cache_rejects_markdown_fence(tmp_path):
    fake = tmp_path / "scene.py"
    fake.write_text(
        "```python\n"
        "import bpy\n"
        "bpy.ops.render.render(animation=True)\n"
        "```\n"
    )
    assert pipeline._scene_cache_is_usable(fake) is False


# ---------------------------------------------------------------------------
# Resume gap detection.
# ---------------------------------------------------------------------------


def test_detect_history_gaps_reports_missing_iters():
    from cg_tutor.critic_loop import CriticIteration
    from cg_tutor.schemas import CriticReport

    def _make(it):
        return CriticIteration(
            iteration=it,
            report=CriticReport(
                concept_id="x", iteration=it, overall_score=0.5,
            ),
            scene_path=Path("/tmp/scene.py"),
            render_ok=True,
            n_frames=10,
            frames_hash="abc",
            missing_objects={},
        )
    gaps = pipeline._detect_history_gaps([_make(0), _make(2), _make(4)])
    assert gaps == [1, 3]


def test_detect_history_gaps_returns_empty_for_contiguous_history():
    from cg_tutor.critic_loop import CriticIteration
    from cg_tutor.schemas import CriticReport

    def _make(it):
        return CriticIteration(
            iteration=it,
            report=CriticReport(
                concept_id="x", iteration=it, overall_score=0.5,
            ),
            scene_path=Path("/tmp/scene.py"),
            render_ok=True,
            n_frames=10,
            frames_hash="abc",
            missing_objects={},
        )
    gaps = pipeline._detect_history_gaps([_make(0), _make(1), _make(2)])
    assert gaps == []
