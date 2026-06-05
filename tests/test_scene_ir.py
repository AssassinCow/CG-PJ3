from cg_tutor.scene_ir import (
    build_scene_ir,
    format_scene_ir_for_coder,
    verify_scene_ir,
)
from cg_tutor.schemas import Narrative, NarrativeNode, Storyboard
from cg_tutor.scene_profiles import base_profile


def _narrative(visual_intent: str = "A red sphere is visible.") -> Narrative:
    return Narrative(
        concept_id="demo",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="One",
                description="Show the object.",
                formulas=["x"],
                duration_sec=1.0,
                visual_intent=visual_intent,
            )
        ],
    )


def _storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "demo",
        "fps": 24,
        "resolution": [320, 240],
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
                "name": "red_sphere",
                "type": "mesh",
                "primitive": "sphere",
                "location": [0, 0, 0],
            }],
            "formula": "x",
        }],
    })


def test_scene_ir_builds_from_narrative_and_storyboard():
    scene_ir = build_scene_ir(_narrative(), _storyboard())

    assert scene_ir.concept_id == "demo"
    assert scene_ir.shots[0].visual_intent == "A red sphere is visible."
    assert scene_ir.shots[0].expected_objects[0].name == "red_sphere"
    assert scene_ir.shots[0].visual_contract is not None


def test_scene_ir_verifier_blocks_missing_visual_intent():
    scene_ir = build_scene_ir(_narrative(visual_intent=""), _storyboard())

    report = verify_scene_ir(scene_ir)

    assert not report.ok
    assert any(i.rule_id == "missing_visual_intent" for i in report.issues)


def test_scene_ir_formats_contract_for_coder():
    scene_ir = build_scene_ir(
        _narrative("Show a sphere labeled surface with a vector arrow."),
        _storyboard(),
    )
    out = format_scene_ir_for_coder(scene_ir, verify_scene_ir(scene_ir))

    assert "SCENE IR CONTRACT" in out
    assert "red_sphere" in out
    assert "derived_visual_contract" in out
    assert "required_labels" in out


def test_scene_ir_adds_projection_requirement_slot():
    scene_ir = build_scene_ir(
        _narrative(
            "Show projection rays from object_point_A through the pinhole "
            "to a projected point on the image plane."
        ),
        _storyboard(),
    )

    slots = scene_ir.shots[0].requirement_slots

    assert any(s.slot_type == "projection_geometry" for s in slots)
    formatted = format_scene_ir_for_coder(scene_ir)
    assert "visual_requirement_slots" in formatted
    assert "projection_geometry" in formatted
    assert "object_point -> pinhole -> projected_point" in formatted


def test_scene_ir_adds_grounded_environment_slot_from_profile():
    profile = base_profile("vector_teaching").model_copy(update={
        "persistent_anchors": ["window_frame", "neon_sign_OPEN"],
        "spatial_relationships": ["neon_sign_OPEN sits behind window_frame"],
        "forbidden_abstractions": ["plain glowing rectangle as sign"],
    })

    scene_ir = build_scene_ir(_narrative(), _storyboard(), scene_profile=profile)
    slots = scene_ir.shots[0].requirement_slots

    grounded = [s for s in slots if s.slot_type == "grounded_environment"]
    assert grounded
    assert "window_frame" in grounded[0].requires
    assert "plain glowing rectangle as sign" in grounded[0].forbidden
