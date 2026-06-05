from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import base_profile
from cg_tutor.storyboard_sanitizer import sanitize_storyboard_for_pipeline


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
            }],
            "objects": [
                {
                    "name": "sphere",
                    "type": "mesh",
                    "primitive": "sphere",
                },
                {
                    "name": "formula_panel_diffuse",
                    "type": "text",
                    "primitive": "empty",
                },
            ],
            "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
            "formula": "I_d = k_d(N\\cdot L)I_l",
        }],
    })


def test_teaching_storyboard_removes_formula_objects():
    sb, report = sanitize_storyboard_for_pipeline(
        _storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.changed
    assert report.removed_formula_objects == {
        "node_01": ["formula_panel_diffuse"]
    }
    assert [obj.name for obj in sb.shots[0].objects] == ["sphere"]


def test_cinematic_storyboard_keeps_objects_unchanged():
    sb, report = sanitize_storyboard_for_pipeline(
        _storyboard(),
        scene_profile=base_profile("cinematic_application"),
    )

    assert not report.changed
    assert [obj.name for obj in sb.shots[0].objects] == [
        "sphere",
        "formula_panel_diffuse",
    ]
