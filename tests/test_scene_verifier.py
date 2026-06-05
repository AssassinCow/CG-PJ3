from cg_tutor.scene_verifier import format_verifier_addendum, verify_scene_code
from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import SceneProfile, base_profile
from cg_tutor.visual_contract import VisualContract


def _valid_scene() -> str:
    return """
import os
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.frame_start = 1
scene.frame_end = 24
out_dir = os.environ['CG_TUTOR_OUT_DIR']
scene.render.filepath = os.path.join(out_dir, 'frame_####.png')
obj = type('Obj', (), {})()
obj.name = 'red_sphere'
bpy.ops.render.render(animation=True)
"""


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
        }],
    })


def _long_storyboard_with_keyframes() -> Storyboard:
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["duration_sec"] = 3.0
    raw["shots"][0]["objects"][0]["keyframes"] = [
        {"time_sec": 0.0, "attr": "location", "value": [0, 0, 0]},
        {"time_sec": 3.0, "attr": "location", "value": [1, 0, 0]},
    ]
    return Storyboard.model_validate(raw)


def _long_storyboard_without_keyframes() -> Storyboard:
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["duration_sec"] = 3.0
    return Storyboard.model_validate(raw)


def _formula_storyboard() -> Storyboard:
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["formula"] = "I_d = k_d(N\\cdot L)I_l"
    raw["shots"][0]["overlay_zone"] = {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18}
    return Storyboard.model_validate(raw)


def _cycles_denoising_off() -> str:
    return (
        "scene.cycles.use_denoising = False\n"
        "for layer in scene.view_layers:\n"
        "    layer.cycles.use_denoising = False\n"
    )


def test_scene_verifier_accepts_valid_minimal_scene():
    report = verify_scene_code(_valid_scene(), _storyboard())

    assert report.ok
    assert report.issues == []


def test_scene_verifier_blocks_render_breaking_contracts():
    code = _valid_scene().replace("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT")

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(i.rule_id == "forbidden_eevee_next" for i in report.issues)


def test_scene_verifier_blocks_cycles_by_default():
    code = _valid_scene().replace("BLENDER_EEVEE", "CYCLES")

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(i.rule_id == "forbidden_cycles" for i in report.issues)


def test_scene_verifier_allows_cycles_when_requested():
    code = (
        _valid_scene().replace("BLENDER_EEVEE", "CYCLES")
        + "scene.cycles.device = 'GPU'\n"
        + _cycles_denoising_off()
    )

    report = verify_scene_code(code, _storyboard(), render_engine="CYCLES")

    assert report.ok
    assert not any(i.rule_id == "forbidden_cycles" for i in report.issues)
    assert any(
        i.rule_id == "cycles_gpu_device_enumeration_missing"
        for i in report.issues
    )


def test_scene_verifier_accepts_cycles_gpu_device_enumeration():
    code = (
        _valid_scene().replace("BLENDER_EEVEE", "CYCLES")
        + "scene.cycles.device = 'GPU'\n"
        + _cycles_denoising_off()
        + "prefs = bpy.context.preferences.addons.get('cycles')\n"
        + "cprefs = prefs.preferences\n"
        + "cprefs.compute_device_type = 'OPTIX'\n"
        + "cprefs.get_devices()\n"
        + "for dev in cprefs.devices:\n"
        + "    dev.use = getattr(dev, 'type', '') != 'CPU'\n"
    )

    report = verify_scene_code(code, _storyboard(), render_engine="CYCLES")

    assert report.ok
    assert not any(
        i.rule_id == "cycles_gpu_device_enumeration_missing"
        for i in report.issues
    )


def test_scene_verifier_requires_cycles_gpu_device_by_default():
    code = _valid_scene().replace("BLENDER_EEVEE", "CYCLES") + _cycles_denoising_off()

    report = verify_scene_code(code, _storyboard(), render_engine="CYCLES")

    assert not report.ok
    assert any(i.rule_id == "missing_cycles_gpu_device" for i in report.issues)


def test_scene_verifier_allows_cycles_cpu_when_requested():
    code = (
        _valid_scene().replace("BLENDER_EEVEE", "CYCLES")
        + "scene.cycles.device = 'CPU'\n"
        + _cycles_denoising_off()
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        render_engine="CYCLES",
        cycles_device="CPU",
    )

    assert report.ok
    assert not any(i.rule_id == "missing_cycles_gpu_device" for i in report.issues)


def test_scene_verifier_requires_cycles_denoising_disabled():
    code = (
        _valid_scene().replace("BLENDER_EEVEE", "CYCLES")
        + "scene.cycles.device = 'CPU'\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        render_engine="CYCLES",
        cycles_device="CPU",
    )

    assert not report.ok
    assert any(
        i.rule_id == "missing_cycles_denoising_disabled"
        for i in report.issues
    )


def test_scene_verifier_blocks_unguarded_eevee_use_property():
    code = _valid_scene() + "\nscene.eevee.use_motion_blur = False\n"

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(
        i.rule_id == "unguarded_eevee_use_property"
        for i in report.issues
    )


def test_scene_verifier_allows_guarded_eevee_use_properties():
    code = (
        _valid_scene()
        + "\nif hasattr(scene.eevee, 'use_ssr'):\n"
        + "    scene.eevee.use_ssr = True\n"
        + "if hasattr(scene.eevee, 'use_ssr_refraction'):\n"
        + "    scene.eevee.use_ssr_refraction = False\n"
        + "if hasattr(scene.eevee, 'use_motion_blur'):\n"
        + "    scene.eevee.use_motion_blur = False\n"
    )

    report = verify_scene_code(code, _storyboard())

    assert report.ok
    assert not any(
        i.rule_id == "unguarded_eevee_use_property"
        for i in report.issues
    )


def test_scene_verifier_blocks_unsafe_action_fcurves_access():
    code = (
        _valid_scene()
        + "\nfor fcu in cam.data.animation_data.action.fcurves:\n"
        + "    pass\n"
    )

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(
        i.rule_id == "unsafe_action_fcurves_access"
        for i in report.issues
    )


def test_scene_verifier_requires_matching_eevee_use_guard():
    code = (
        _valid_scene()
        + "\nif hasattr(scene.eevee, 'use_ssr'):\n"
        + "    scene.eevee.use_ssr_refraction = False\n"
    )

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(
        i.rule_id == "unguarded_eevee_use_property"
        for i in report.issues
    )


def test_scene_verifier_blocks_unguarded_material_shadow_method():
    code = _valid_scene() + "\nmat.shadow_method = 'HASHED'\n"

    report = verify_scene_code(code, _storyboard())

    assert not report.ok
    assert any(
        i.rule_id == "unguarded_material_shadow_method"
        for i in report.issues
    )


def test_scene_verifier_allows_guarded_material_shadow_method():
    code = (
        _valid_scene()
        + "\nif hasattr(mat, 'shadow_method'):\n"
        + "    mat.shadow_method = 'HASHED'\n"
    )

    report = verify_scene_code(code, _storyboard())

    assert report.ok
    assert not any(
        i.rule_id == "unguarded_material_shadow_method"
        for i in report.issues
    )


def test_scene_verifier_blocks_world_space_normal_segments_parented_to_translated_empty():
    code = (
        _valid_scene()
        + "\ndef add_dashed_line(name, a, b):\n"
        + "    return []\n"
        + "entry_point = (1.0, 0.0, 0.5)\n"
        + "n_entry_outer = (0.0, 0.0, 1.0)\n"
        + "n_entry_inner = (1.0, 0.0, 0.0)\n"
        + "surface_normal_entry_segs = add_dashed_line('surface_normal_entry_seg', n_entry_outer, n_entry_inner)\n"
        + "surface_normal_entry = bpy.data.objects.new('surface_normal_entry', None)\n"
        + "surface_normal_entry.location = entry_point\n"
        + "for s in surface_normal_entry_segs:\n"
        + "    s.parent = surface_normal_entry\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(
        i.rule_id == "normal_helper_parented_without_inverse"
        for i in report.issues
    )


def test_scene_verifier_allows_normal_parenting_with_parent_inverse():
    code = (
        _valid_scene()
        + "\ndef add_dashed_line(name, a, b):\n"
        + "    return []\n"
        + "entry_point = (1.0, 0.0, 0.5)\n"
        + "surface_normal_entry_segs = add_dashed_line('surface_normal_entry_seg', (0, 0, 0), (1, 0, 0))\n"
        + "surface_normal_entry = bpy.data.objects.new('surface_normal_entry', None)\n"
        + "surface_normal_entry.location = entry_point\n"
        + "for s in surface_normal_entry_segs:\n"
        + "    s.parent = surface_normal_entry\n"
        + "    s.matrix_parent_inverse = surface_normal_entry.matrix_world.inverted()\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert not any(
        i.rule_id == "normal_helper_parented_without_inverse"
        for i in report.issues
    )


def test_scene_verifier_does_not_block_fk_style_non_normal_parenting():
    code = (
        _valid_scene()
        + "\nJ1_pivot = bpy.data.objects.new('J1_pivot', None)\n"
        + "J2_pivot = bpy.data.objects.new('J2_pivot', None)\n"
        + "J1_pivot.location = (1.0, 0.0, 0.0)\n"
        + "J2_pivot.parent = J1_pivot\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert not any(
        i.rule_id == "normal_helper_parented_without_inverse"
        for i in report.issues
    )


def test_scene_verifier_blocks_syntax_errors():
    report = verify_scene_code("import bpy\nx = [0,\u22126]\n", _storyboard())

    assert not report.ok
    assert any(i.rule_id in {"syntax_error", "invalid_typography"}
               for i in report.issues)


def test_scene_verifier_reports_missing_storyboard_names_as_warn():
    code = _valid_scene().replace("red_sphere", "other_name")

    report = verify_scene_code(code, _storyboard())

    assert report.ok
    assert report.missing_objects == {"node_01": ["red_sphere"]}
    assert any(i.rule_id == "missing_storyboard_objects"
               and i.severity == "warn" for i in report.issues)


def test_scene_verifier_blocks_formula_text_inside_teaching_scene():
    code = (
        _valid_scene()
        + "\nobj.name = 'formula_panel_diffuse'\n"
        + "formula = 'I_d = k_d(N\\\\cdot L)I_l'\n"
    )

    report = verify_scene_code(
        code,
        _formula_storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "formula_text_in_scene" for i in report.issues)


def test_scene_verifier_blocks_formula_like_3d_text_inside_teaching_scene():
    code = (
        _valid_scene()
        + "\nbpy.ops.object.text_add(location=(0, 0, 0))\n"
        + "txt = bpy.context.object\n"
        + "txt.name = 'formula_face_point'\n"
        + "txt.data.body = 'F = (1/n) sum V_i'\n"
    )

    report = verify_scene_code(
        code,
        _formula_storyboard(),
        scene_profile=base_profile("transformation_demo"),
    )

    assert not report.ok
    assert any(i.rule_id == "formula_text_in_scene" for i in report.issues)


def test_scene_verifier_does_not_treat_api_attribute_names_as_formula_text():
    code = (
        _valid_scene()
        + "\nif hasattr(scene.cycles, 'use_denoising'):\n"
        + "    scene.cycles.use_denoising = False\n"
    )

    report = verify_scene_code(
        code,
        _formula_storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert not any(i.rule_id == "formula_text_in_scene" for i in report.issues)


def test_scene_verifier_allows_simple_topology_counters_in_scene():
    code = (
        _valid_scene()
        + "\nbpy.ops.object.text_add(location=(2, 0, 1))\n"
        + "txt = bpy.context.object\n"
        + "txt.name = 'counter_level_1'\n"
        + "txt.data.body = 'V = 26'\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert not any(i.rule_id == "formula_text_in_scene" for i in report.issues)


def test_scene_verifier_blocks_missing_storyboard_animation():
    report = verify_scene_code(
        _valid_scene(),
        _long_storyboard_with_keyframes(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "missing_storyboard_animation" for i in report.issues)


def test_scene_verifier_does_not_count_camera_keyframes_as_object_animation():
    code = (
        _valid_scene()
        + "\ncam = type('Camera', (), {})()\n"
        + "cam.data = type('CameraData', (), {})()\n"
        + "cam.keyframe_insert('location', frame=1)\n"
        + "cam.keyframe_insert('rotation_euler', frame=72)\n"
        + "cam.data.keyframe_insert('lens', frame=72)\n"
    )

    report = verify_scene_code(
        code,
        _long_storyboard_with_keyframes(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "missing_storyboard_animation" for i in report.issues)


def test_scene_verifier_blocks_static_multisecond_teaching_scene():
    code = (
        _valid_scene()
        + "\nobj.keyframe_insert('hide_render', frame=1)\n"
        + "obj.keyframe_insert('hide_viewport', frame=1)\n"
    )

    report = verify_scene_code(
        code,
        _long_storyboard_without_keyframes(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "insufficient_scene_animation" for i in report.issues)


def test_scene_verifier_accepts_object_animation():
    code = (
        _valid_scene()
        + "\nobj.location = (1, 0, 0)\n"
        + "obj.keyframe_insert('location', frame=1)\n"
        + "obj.location = (2, 0, 0)\n"
        + "obj.keyframe_insert('location', frame=72)\n"
    )

    report = verify_scene_code(
        code,
        _long_storyboard_without_keyframes(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert not any(
        i.rule_id in {"missing_storyboard_animation", "insufficient_scene_animation"}
        for i in report.issues
    )


def test_scene_verifier_addendum_is_actionable():
    report = verify_scene_code("print('no blender here')", _storyboard())

    addendum = format_verifier_addendum(report)

    assert "SCENE VERIFIER FAILED" in addendum
    assert "fix:" in addendum


def test_scene_verifier_warns_when_visual_contract_text_is_absent():
    report = verify_scene_code(
        _valid_scene(),
        _storyboard(),
        visual_contracts={
            "node_01": VisualContract(
                shot_id="node_01",
                required_labels=["surface"],
            )
        },
    )

    assert report.ok
    assert any(i.rule_id == "missing_required_labels" for i in report.issues)


def test_scene_verifier_warns_when_vector_geometry_is_absent():
    report = verify_scene_code(
        _valid_scene(),
        _storyboard(),
        visual_contracts={
            "node_01": VisualContract(
                shot_id="node_01",
                required_vectors=["N", "L"],
            )
        },
    )

    assert report.ok
    assert any(i.rule_id == "missing_vector_geometry" for i in report.issues)


def test_scene_verifier_blocks_cinematic_arrow_helpers():
    code = _valid_scene() + "\ndef create_arrow():\n    pass\ncreate_arrow()\n"

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("cinematic_application"),
    )

    assert not report.ok
    assert any(i.rule_id == "profile_forbidden_arrow" for i in report.issues)


def test_scene_verifier_forbidden_helpers_are_profile_driven():
    code = _valid_scene() + "\ndef create_arrow():\n    pass\ncreate_arrow()\n"
    profile = SceneProfile(
        profile_id="custom_no_arrows",
        base_profile="transformation_demo",
        forbidden_helpers=["arrow_primitives"],
    )

    report = verify_scene_code(code, _storyboard(), scene_profile=profile)

    assert not report.ok
    assert any(i.rule_id == "profile_forbidden_arrow" for i in report.issues)


def test_scene_verifier_allows_cinematic_searchlight_cone():
    code = (
        _valid_scene()
        + "\nbpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1.0, depth=2.0)\n"
    )

    report = verify_scene_code(
        code,
        _storyboard(),
        scene_profile=base_profile("cinematic_application"),
    )

    assert report.ok
    assert not any(i.rule_id == "profile_forbidden_arrow" for i in report.issues)


def test_scene_verifier_cinematic_vectors_expect_tracer_not_arrow():
    report = verify_scene_code(
        _valid_scene(),
        _storyboard(),
        visual_contracts={
            "node_01": VisualContract(
                shot_id="node_01",
                required_vectors=["ray_path"],
            )
        },
        scene_profile=base_profile("cinematic_application"),
    )

    assert report.ok
    assert any(i.rule_id == "missing_profile_tracer_geometry" for i in report.issues)


def test_scene_verifier_no_tracer_warning_when_profile_forbids_drawn_rays():
    profile = SceneProfile(
        profile_id="deep_sea",
        base_profile="cinematic_application",
        forbidden_helpers=[
            "ray_path_polylines",
            "schematic_light_paths",
            "thin_glowing_paths",
            "dotted_tracers",
        ],
    )
    report = verify_scene_code(
        _valid_scene(),
        _storyboard(),
        visual_contracts={
            "node_01": VisualContract(
                shot_id="node_01",
                required_vectors=["ray_path"],
            )
        },
        scene_profile=profile,
    )

    assert report.ok
    assert not any(
        i.rule_id == "missing_profile_tracer_geometry"
        for i in report.issues
    )
