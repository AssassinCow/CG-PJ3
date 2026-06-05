from cg_tutor.scene_state import inspect_scene_code, scene_state_report_to_json


def test_scene_state_extracts_objects_and_animation_channels():
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add(size=1.0)\n"
        "obj = bpy.context.object\n"
        "obj.name = 'moving_cube'\n"
        "obj.location = (0, 0, 0)\n"
        "obj.keyframe_insert('location', frame=1)\n"
        "obj.location = (1, 0, 0)\n"
        "obj.keyframe_insert('location', frame=24)\n"
        "obj.hide_render = True\n"
        "obj.keyframe_insert('hide_render', frame=48)\n"
    )

    report = inspect_scene_code(code)

    assert report.ok
    assert "moving_cube" in report.object_names
    assert report.metrics["non_camera_keyframe_count"] == 2
    assert "moving_cube.location" in report.metrics["animated_channels"]
    assert "moving_cube" in report.metrics["animated_non_camera_objects"]


def test_scene_state_reports_syntax_error_as_json_safe():
    report = inspect_scene_code("def \n")
    data = scene_state_report_to_json(report)

    assert not report.ok
    assert "syntax_error" in data
