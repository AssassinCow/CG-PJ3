from cg_tutor.schemas import NarrativeNode, Storyboard
from cg_tutor.scene_profiles import SceneProfile, base_profile
from cg_tutor.visual_contract import build_visual_contract, format_visual_contract


def _shot(text: str = "label arrows") -> tuple:
    sb = Storyboard.model_validate({
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
            "objects": [
                {
                    "name": "arrow_N",
                    "type": "mesh",
                    "primitive": "arrow",
                    "location": [0, 0, 0],
                },
                {
                    "name": "arrow_L",
                    "type": "mesh",
                    "primitive": "arrow",
                    "location": [1, 0, 0],
                },
            ],
            "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
        }],
    })
    node = NarrativeNode(
        id="node_01",
        title="Vectors",
        description=text,
        formulas=[],
        duration_sec=1.0,
        visual_intent=text,
    )
    return sb.shots[0], node


def _plain_shot(text: str) -> tuple:
    sb = Storyboard.model_validate({
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
            "objects": [{"name": "mip_level_stack", "type": "mesh", "primitive": "cube"}],
        }],
    })
    node = NarrativeNode(
        id="node_01",
        title="Plain",
        description=text,
        formulas=[],
        duration_sec=1.0,
        visual_intent=text,
    )
    return sb.shots[0], node


def test_visual_contract_extracts_labels_vectors_highlights_and_overlay():
    shot, node = _shot(
        "Show arrows N and L with a specular highlight, labeled normal and light."
    )

    contract = build_visual_contract(shot, node)

    assert "arrow_N" in contract.required_vectors
    assert "arrow_L" in contract.required_vectors
    assert contract.required_labels
    assert contract.emphasis_points
    assert contract.overlay_constraints
    assert any("spatially separated" in s for s in contract.required_relationships)


def test_visual_contract_formats_compactly():
    shot, node = _shot("Show two vector arrows with visible labels.")

    out = "\n".join(format_visual_contract(build_visual_contract(shot, node)))

    assert "required_vectors" in out
    assert "avoid" in out


def test_visual_contract_does_not_treat_preview_as_view_vector():
    shot, node = _plain_shot(
        "Show three compact mip preview tiles labeled LOD0, LOD1, and LOD2."
    )

    contract = build_visual_contract(
        shot, node, scene_profile=base_profile("cinematic_application"),
    )

    assert contract.required_vectors == []
    assert "LOD0" in " ".join(contract.required_labels)


def test_visual_contract_extracts_quoted_label_without_reads_prefix():
    shot, node = _shot('The only nearby text label reads "shape A".')

    contract = build_visual_contract(shot, node)

    assert "shape A" in contract.required_labels
    assert not any("reads" in label.lower() for label in contract.required_labels)


def test_cinematic_profile_prefers_thin_tracers_over_arrows():
    shot, node = _shot(
        "A thin ray tracer line reaches a crystal highlight, not a classroom arrow."
    )

    contract = build_visual_contract(
        shot, node, scene_profile=base_profile("cinematic_application"),
    )

    text = " ".join(contract.required_relationships + contract.forbidden_failures)
    assert "thin glowing" in text
    assert "thick arrows" in text
    assert "full visible shaft" not in text


def test_cinematic_profile_adds_grounding_anchors_to_contract():
    shot, node = _shot("Rainy window refraction scene with neon sign.")
    profile = SceneProfile(
        profile_id="rainy_window",
        base_profile="cinematic_application",
        persistent_anchors=["window_frame", "neon_sign_OPEN"],
        spatial_relationships=[
            "raindrops_on_glass sit between camera and neon_sign_OPEN"
        ],
        forbidden_abstractions=["OPEN sign replaced by a blank glowing rectangle"],
    )

    contract = build_visual_contract(shot, node, scene_profile=profile)
    formatted = "\n".join(format_visual_contract(contract))

    assert contract.required_anchors == ["window_frame", "neon_sign_OPEN"]
    assert "required_anchors" in formatted
    assert "raindrops_on_glass" in " ".join(contract.required_relationships)
    assert "blank glowing rectangle" in " ".join(contract.forbidden_failures)


def test_teaching_profile_adds_grounding_anchors_to_contract():
    shot, node = _shot("Dolly zoom with a centered lighthouse.")
    profile = SceneProfile(
        profile_id="dolly_zoom",
        base_profile="transformation_demo",
        persistent_anchors=["hero_lighthouse", "marker_post_1"],
        spatial_relationships=["marker_post_1 sits behind hero_lighthouse"],
        forbidden_abstractions=["camera icon floating above the hero"],
    )

    contract = build_visual_contract(shot, node, scene_profile=profile)

    assert contract.required_anchors == ["hero_lighthouse", "marker_post_1"]
    assert "marker_post_1 sits behind hero_lighthouse" in contract.required_relationships
    assert "camera icon floating above the hero" in contract.forbidden_failures


def test_cinematic_profile_can_forbid_drawn_ray_helpers():
    shot, node = _shot(
        "Show reflection and refraction through a glass dome, with no ray lines."
    )
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

    contract = build_visual_contract(shot, node, scene_profile=profile)

    assert contract.required_vectors == []
    text = " ".join(contract.required_relationships + contract.forbidden_failures)
    assert "Do not draw ray/vector paths" in text
    assert "physical surfaces" in text


def test_no_drawn_ray_policy_is_not_tied_to_cinematic_base():
    shot, node = _shot(
        "Show reflection and refraction through physical glass surfaces."
    )
    profile = SceneProfile(
        profile_id="custom_physical_optics",
        base_profile="transformation_demo",
        forbidden_helpers=["ray_path_polylines"],
    )

    contract = build_visual_contract(shot, node, scene_profile=profile)

    assert contract.required_vectors == []
    text = " ".join(contract.required_relationships + contract.forbidden_failures)
    assert "Do not draw ray/vector paths" in text
