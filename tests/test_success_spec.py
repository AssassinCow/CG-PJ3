"""Tests for the Success Spec schema and YAML loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cg_tutor.success_spec import (
    SuccessSpec,
    format_success_spec_for_coder,
    load_success_spec,
)


def _depth_of_field_success_spec_mapping() -> dict:
    return {
        "concept_id": "depth_of_field_focus_pull_fixture",
        "success_spec": {
            "version": 1,
            "success_states": [{
                "id": "early",
                "frame_range": {"fraction": [0.0, 0.3]},
                "readable_text": ["depth_label_near", "lens_readout_hud"],
            }],
            "hard_constraints": [{
                "kind": "text_faces_camera",
                "anchors": [
                    {"name": "depth_label_near", "placement": "in_scene"},
                    {"name": "lens_readout_hud", "placement": "hud_overlay"},
                ],
            }],
            "required_visual_evidence": [{
                "kind": "aperture_within_range",
                "anchor": "camera",
                "data_path": "camera.data.dof.aperture_fstop",
                "range": [1.4, 2.8],
            }],
        },
    }


def test_loads_depth_of_field_success_spec_from_mapping():
    spec = _depth_of_field_success_spec_mapping()

    success_spec = load_success_spec(spec)

    assert success_spec is not None
    assert success_spec.version == 1
    assert success_spec.aperture_fstop_range() == (1.4, 2.8)
    assert "lens_readout_hud" in success_spec.text_faces_camera_anchors()
    assert success_spec.text_anchor_placements()["lens_readout_hud"] == "hud_overlay"


def test_format_success_spec_for_coder_lists_exact_text_and_framing_contract():
    spec = _depth_of_field_success_spec_mapping()
    success_spec = load_success_spec(spec)

    text = format_success_spec_for_coder(success_spec)

    assert "SUCCESS SPEC" in text
    assert "exact readable text object names" in text
    assert "depth_label_near" in text
    assert "lens_readout_hud" in text
    assert "inside the camera frame with a clear safety margin" in text
    assert "aperture_fstop must stay in [1.4, 2.8]" in text
    assert "Pseudo-HUD anchors" in text


def test_resolves_fraction_frame_ranges():
    success_spec = SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 0.25]},
        }],
    })

    assert success_spec.resolved_state_frames(total_frames=384) == {
        "early": (1, 96),
    }


def test_text_faces_camera_accepts_legacy_anchor_list():
    success_spec = SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 0.25]},
        }],
        "hard_constraints": [{
            "kind": "text_faces_camera",
            "anchors": ["depth_label_near"],
        }],
    })

    assert success_spec.text_faces_camera_anchors() == {"depth_label_near"}
    assert success_spec.text_anchor_placements() == {
        "depth_label_near": "in_scene",
    }


def test_text_faces_camera_accepts_anchor_placement_objects():
    success_spec = SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 0.25]},
        }],
        "hard_constraints": [{
            "kind": "text_faces_camera",
            "anchors": [
                {"name": "depth_label_near", "placement": "in_scene"},
                {"name": "lens_readout_hud", "placement": "hud_overlay"},
            ],
        }],
    })

    assert success_spec.text_faces_camera_anchors() == {
        "depth_label_near",
        "lens_readout_hud",
    }
    assert success_spec.text_anchor_placements()["lens_readout_hud"] == "hud_overlay"


def test_rejects_duplicate_success_state_ids():
    with pytest.raises(ValidationError, match="success_state ids must be unique"):
        SuccessSpec.model_validate({
            "version": 1,
            "success_states": [
                {"id": "early", "frame_range": {"fraction": [0.0, 0.25]}},
                {"id": "early", "frame_range": {"fraction": [0.25, 0.5]}},
            ],
        })


def test_rejects_invalid_frame_fraction():
    with pytest.raises(ValidationError, match="frame_range.fraction"):
        SuccessSpec.model_validate({
            "version": 1,
            "success_states": [{
                "id": "middle",
                "frame_range": {"fraction": [0.8, 0.4]},
            }],
        })
