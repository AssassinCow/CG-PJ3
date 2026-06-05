from cg_tutor.scene_compiler import compile_storyboard_to_bpy
from cg_tutor.schemas import Storyboard
from cg_tutor.visual_contract import VisualContract


def _storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "compiled",
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
            "objects": [
                {
                    "name": "main_sphere",
                    "type": "mesh",
                    "primitive": "sphere",
                    "location": [0, 0, 0],
                    "properties": {"color": [0.8, 0.9, 1.0]},
                },
                {
                    "name": "normal_arrow",
                    "type": "mesh",
                    "primitive": "arrow",
                    "location": [0, 0, 0.8],
                    "properties": {"direction": [0, 0, 1]},
                },
                {
                    "name": "surface_label",
                    "type": "text",
                    "primitive": None,
                    "location": [0, 0, 1.6],
                    "properties": {"text": "surface"},
                },
            ],
        }],
    })


def _long_static_storyboard() -> Storyboard:
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["duration_sec"] = 4.0
    raw["shots"][0]["camera"] = [
        {
            "time_sec": 0.0,
            "position": [0, -5, 3],
            "look_at": [0, 0, 0],
            "fov": 50,
        },
        {
            "time_sec": 4.0,
            "position": [0, -5, 3],
            "look_at": [0, 0, 0],
            "fov": 50,
        },
    ]
    for obj in raw["shots"][0]["objects"]:
        obj["keyframes"] = []
    return Storyboard.model_validate(raw)


def _shadow_softness_storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "shadow_softness_radius",
        "fps": 24,
        "resolution": [320, 240],
        "shots": [{
            "node_id": "soft_shadow",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -6, 3],
                "look_at": [0, 0, 0.8],
                "fov": 50,
            }],
            "objects": [
                {
                    "name": "key_light",
                    "type": "light",
                    "primitive": None,
                    "location": [0, -3, 5],
                    "properties": {"light_kind": "AREA", "energy": 900, "size": 5.0},
                },
                {
                    "name": "subject_pillar",
                    "type": "primitive",
                    "primitive": "cylinder",
                    "location": [0, 0, 0.8],
                    "properties": {"radius": 0.35, "depth": 1.6},
                },
                {
                    "name": "ground_plane",
                    "type": "primitive",
                    "primitive": "plane",
                    "location": [0, 0, 0],
                    "properties": {"size": 5.0},
                },
            ],
        }],
    })


def test_scene_compiler_emits_runnable_python_contracts():
    code = compile_storyboard_to_bpy(_storyboard())

    compile(code, "compiled_scene.py", "exec")
    assert "null" not in code
    assert "None" in code
    assert "CG_TUTOR_PREVIEW_FRAMES" in code
    assert "bpy.ops.render.render(animation=True)" in code
    assert "main_sphere" in code
    assert "normal_arrow" in code
    assert "surface_label" in code


def test_scene_compiler_can_emit_cycles_engine():
    code = compile_storyboard_to_bpy(_storyboard(), render_engine="CYCLES")

    assert "scene.render.engine = 'CYCLES'" in code
    assert "scene.cycles.samples = 32" in code
    assert "scene.cycles.use_denoising = False" in code
    assert "_view_layer.cycles.use_denoising = False" in code
    assert "scene.cycles.device = 'GPU'" in code
    assert "compute_device_type" in code
    assert "scene.render.engine = 'BLENDER_EEVEE'" not in code


def test_scene_compiler_adds_required_vector_placeholders_from_contracts():
    contracts = {
        "node_01": VisualContract(
            shot_id="node_01",
            required_vectors=[
                "incident_white_ray",
                "surface_normal_entry",
                "spectrum_red_ray",
            ],
        )
    }

    code = compile_storyboard_to_bpy(_storyboard(), visual_contracts=contracts)

    assert "incident_white_ray_placeholder" in code
    assert "surface_normal_entry_placeholder" in code
    assert "spectrum_red_ray_placeholder" in code
    assert "'primitive': 'curve_polyline'" in code
    assert "data.bevel_factor_end" in code


def test_scene_compiler_can_force_cycles_cpu():
    code = compile_storyboard_to_bpy(
        _storyboard(),
        render_engine="CYCLES",
        cycles_device="CPU",
    )

    assert "scene.cycles.device = 'CPU'" in code


def test_scene_compiler_emits_visual_grammar_helpers():
    code = compile_storyboard_to_bpy(_storyboard())

    assert "def add_curve_polyline" in code
    assert "def add_teaching_helpers_for_shot" in code
    assert "def add_focal_bracket" in code
    assert "def add_projection_plane_marks" in code
    assert "COMPILED_SCENE_HINTS" in code
    assert "apply_spec_keyframes(obj, spec)" in code
    assert "set_visibility_windows(obj, visibility.get(name, []))" in code


def test_scene_compiler_keeps_curve_and_group_primitives_runnable():
    sb = Storyboard.model_validate({
        "concept_id": "curve_compiled",
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
            "objects": [
                {
                    "name": "curve_path",
                    "type": "curve",
                    "primitive": "curve_polyline",
                    "location": [0, 0, 0],
                    "properties": {
                        "points": [[0, 0, 0], [1, 0, 0.5], [2, 0, 0]],
                        "bevel_depth": 0.02,
                    },
                },
                {
                    "name": "control_points",
                    "type": "mesh_group",
                    "primitive": "sphere_group",
                    "location": [0, 0, 0],
                    "properties": {
                        "points": [[0, 0, 0], [1, 0, 0.5], [2, 0, 0]],
                        "radius": 0.08,
                    },
                },
            ],
        }],
    })

    code = compile_storyboard_to_bpy(sb)

    compile(code, "compiled_curve_scene.py", "exec")
    assert "curve_path" in code
    assert "control_points" in code
    assert "add_curve_polyline(name, points, props)" in code
    assert "add_point_group(name, primitive, loc, props)" in code


def test_scene_compiler_strips_formula_text_from_runtime_storyboard():
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["formula"] = "K = diag(f,f,1)"
    raw["shots"][0]["caption"] = "do not embed me"
    raw["shots"][0]["overlay_zone"] = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    sb = Storyboard.model_validate(raw)

    code = compile_storyboard_to_bpy(sb)

    assert "K = diag" not in code
    assert "do not embed me" not in code
    assert "overlay_zone" not in code


def test_scene_compiler_enriches_static_storyboard_with_motion():
    code = compile_storyboard_to_bpy(_long_static_storyboard())

    compile(code, "compiled_motion_scene.py", "exec")
    assert "data.bevel_factor_end" in code
    assert "'attr': 'location'" in code
    assert "'attr': 'scale'" in code
    assert "apply_spec_keyframes(obj, spec)" in code


def test_scene_compiler_enriches_shadow_softness_light_size_motion():
    code = compile_storyboard_to_bpy(_shadow_softness_storyboard())

    compile(code, "compiled_shadow_softness_scene.py", "exec")
    assert "'name': 'area_light'" in code
    assert "'type': 'light'" in code
    assert "'attr': 'data.size'" in code
    assert "'value': 0.05" in code
    assert "'value': 1.25" in code
    assert "'location': [-1.7, -2.6, 4.0]" in code
    assert "if hasattr(data, 'size') and 'size' in props:" in code


def test_scene_compiler_preserves_existing_object_keyframes():
    raw = _long_static_storyboard().model_dump(mode="json")
    raw["shots"][0]["objects"][0]["keyframes"] = [
        {"time_sec": 0.0, "attr": "scale", "value": [1, 1, 1]},
        {"time_sec": 4.0, "attr": "scale", "value": [2, 2, 2]},
    ]
    sb = Storyboard.model_validate(raw)

    code = compile_storyboard_to_bpy(sb)

    assert "'value': [2.0, 2.0, 2.0]" in code


def test_scene_compiler_normalises_projection_teaching_layout():
    sb = Storyboard.model_validate({
        "concept_id": "projection_compiled",
        "fps": 24,
        "resolution": [320, 240],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -6, 3],
                "look_at": [0, 0, 1],
                "fov": 45,
            }],
            "objects": [
                {
                    "name": "black_camera_box",
                    "type": "primitive",
                    "primitive": "cube",
                    "location": [2, 0, 1],
                    "properties": {"color": [1, 0, 0], "size": 0.8},
                },
                {
                    "name": "camera_center_c",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [0, 0, 1],
                    "properties": {"radius": 0.1},
                },
                {
                    "name": "translucent_image_plane",
                    "type": "primitive",
                    "primitive": "plane",
                    "location": [-2, 0, 1],
                    "properties": {"size": 3},
                },
                {
                    "name": "object_point_a",
                    "type": "primitive",
                    "primitive": "cube",
                    "location": [2, 0, 1],
                    "properties": {"size": 0.8},
                },
                {
                    "name": "object_point_b",
                    "type": "primitive",
                    "primitive": "cube",
                    "location": [2, 0, 1],
                    "properties": {"size": 0.8},
                },
                {
                    "name": "projected_point_a",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [-2, 0, 1],
                    "properties": {"radius": 0.08},
                },
            ],
        }],
    })

    code = compile_storyboard_to_bpy(sb)

    compile(code, "compiled_projection_scene.py", "exec")
    assert "return 'C'" in code
    assert "return 'P(X,Y,Z)'" in code
    assert "return 'p(x,y)'" in code
    assert "def add_axis_glyph" in code
    assert "projected_point_b" in code
    assert "'fov': 62.0" in code
    assert "'color': [0.035, 0.035, 0.04]" in code


def test_scene_compiler_strengthens_dolly_zoom_seed():
    sb = Storyboard.model_validate({
        "concept_id": "dolly_zoom",
        "fps": 24,
        "resolution": [320, 240],
        "shots": [{
            "node_id": "node_04",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -7, 3],
                "look_at": [0, 0, 1],
                "fov": 48,
            }],
            "caption": "Demonstrate the dolly zoom.",
            "objects": [
                {
                    "name": "hero_lighthouse",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [2, 0, 1],
                    "properties": {},
                },
                {
                    "name": "marker_post_1",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [0, 0, 0],
                    "properties": {},
                },
                {
                    "name": "camera_icon",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [0, 0, 1],
                    "properties": {},
                },
            ],
        }],
    })

    code = compile_storyboard_to_bpy(sb)

    compile(code, "compiled_dolly_scene.py", "exec")
    assert "def lens_for_fov" in code
    assert "cam.data.keyframe_insert('lens'" in code
    assert "'position': [0.0, -4.8, 2.25]" in code
    assert "'position': [0.0, -10.2, 2.25]" in code
    assert "'fov': 68.0" in code
    assert "'fov': 27.0" in code
    assert "'marker_post_4'" in code
    assert "'location': [-1.8, 3.2, 0.75]" in code


def test_scene_compiler_dolly_zoom_id_match_requires_prefix_boundary():
    raw = _storyboard().model_dump(mode="json")
    raw["concept_id"] = "anti_dolly_zoom"
    raw["shots"][0]["caption"] = "Generic object turntable."
    sb = Storyboard.model_validate(raw)

    code = compile_storyboard_to_bpy(sb)

    assert "'marker_post_4'" not in code
    assert "'position': [0.0, -4.8, 2.25]" not in code
    assert "'fov': 68.0" not in code
