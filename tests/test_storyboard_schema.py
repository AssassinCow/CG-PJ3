"""Schema validation tests for Storyboard / Shot / OverlayZone.

The schema's job is to catch bad LLM output at the JSON boundary
before it reaches the Blender coder. These tests pin down which inputs
are rejected and which pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cg_tutor.agents import storyboard as storyboard_agent
from cg_tutor.schemas import (
    CameraKey,
    Keyframe,
    OverlayZone,
    SceneObject,
    Shot,
    Storyboard,
)
from cg_tutor.agents.storyboard import (
    _complete_and_validate_candidate,
    _deterministic_storyboard_raw,
    _ensure_storyboard_motion,
    _merge_patch_into_shot,
    _normalize_relative_times,
    _normalize_shot_start_times,
    _normalize_single_shot_root,
    _storyboard_patch_user_prompt,
    _storyboard_user_prompt,
    _storyboard_client_chain,
    _storyboard_repair_user_prompt,
    _storyboard_skeleton,
    _storyboard_candidate_slugs,
    _validate_storyboard_against_narrative,
    _validate_shot_patch,
    ShotPatch,
    validate_storyboard,
)
from cg_tutor.schemas import Narrative, NarrativeNode
from cg_tutor.scene_profiles import SceneProfile


FIXTURE = Path(__file__).parent / "fixtures" / "phong_storyboard.json"


def _demo_narrative() -> Narrative:
    return Narrative(
        concept_id="demo",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="Hit",
                description="Primary ray hits a sphere.",
                formulas=["R = V - 2(V \\cdot N)N"],
                duration_sec=2.0,
                visual_intent="Show ray hit.",
            )
        ],
    )


def _formula_free_narrative() -> Narrative:
    return Narrative(
        concept_id="demo_app",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="Alert",
                description="A museum security HUD catches a bent laser trace.",
                formulas=[],
                duration_sec=2.0,
                visual_intent="Show a cinematic application scene with no formulas.",
            )
        ],
    )


def _multi_node_narrative() -> Narrative:
    return Narrative(
        concept_id="demo_multi",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="Setup",
                description="Show the camera and image plane.",
                formulas=[],
                duration_sec=3.5,
                visual_intent="Establish the scene.",
            ),
            NarrativeNode(
                id="node_02",
                title="Projection",
                description="Trace a point through the pinhole.",
                formulas=["x' = f x / z"],
                duration_sec=4.0,
                visual_intent="Show projection geometry.",
            ),
        ],
    )


def _valid_storyboard_raw() -> dict:
    return {
        "concept_id": "demo",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "node_01",
                "start_sec": 0.0,
                "duration_sec": 2.0,
                "camera": [
                    {
                        "time_sec": 0.0,
                        "position": [0, -4, 2],
                        "look_at": [0, 0, 0],
                        "fov": 50.0,
                    },
                    {
                        "time_sec": 2.0,
                        "position": [0, -4, 2],
                        "look_at": [0, 0, 0],
                        "fov": 50.0,
                    },
                ],
                "objects": [
                    {
                        "name": "key_light",
                        "type": "light",
                        "primitive": None,
                        "location": [0, -3, 4],
                        "properties": {
                            "light_kind": "POINT",
                            "energy": 1200,
                            "light_color": [1, 1, 1],
                        },
                    }
                ],
                "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
                "formula": "R = V - 2(V \\cdot N)N",
                "caption": "Primary ray hits a sphere.",
            }
        ],
    }


class _FakeStoryClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.users: list[str] = []

    def complete_json(self, *, user: str, **_kw):
        self.users.append(user)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


# ----- happy path --------------------------------------------------------

def test_fixture_storyboard_validates():
    sb = Storyboard.model_validate_json(FIXTURE.read_text())
    assert sb.concept_id == "phong_lighting"
    assert sb.fps == 24
    assert sb.total_duration == 5.0
    assert sb.total_frames == 120


def test_storyboard_allows_application_shot_without_formula_or_overlay():
    raw = _valid_storyboard_raw()
    raw["shots"][0]["overlay_zone"] = None
    raw["shots"][0]["formula"] = None
    raw["shots"][0]["caption"] = None

    sb = validate_storyboard(raw)

    assert sb.shots[0].overlay_zone is None
    assert sb.shots[0].formula is None
    assert sb.shots[0].caption is None


def test_storyboard_candidate_mix_uses_sonnet_for_single_candidate():
    slugs = _storyboard_candidate_slugs(
        1,
        None,
        ("claude/claude-sonnet-4-6", "gemini/gemini-3.1-pro-preview"),
    )

    assert slugs == ("claude/claude-sonnet-4-6",)


def test_storyboard_candidate_mix_rotates_configured_providers():
    slugs = _storyboard_candidate_slugs(
        3,
        None,
        ("claude/claude-sonnet-4-6", "gemini/gemini-3.1-pro-preview"),
    )

    assert slugs == (
        "claude/claude-sonnet-4-6",
        "gemini/gemini-3.1-pro-preview",
        "claude/claude-sonnet-4-6",
    )


def test_storyboard_candidate_mix_rotates_official_sdk_slugs():
    slugs = _storyboard_candidate_slugs(
        3,
        None,
        ("anthropic/claude-sonnet-4.6", "google/gemini-3.1-pro-preview"),
    )

    assert slugs == (
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-sonnet-4.6",
    )


def test_storyboard_candidate_mix_respects_explicit_model():
    slugs = _storyboard_candidate_slugs(
        3,
        "gpt/gpt-5.5",
        ("claude/claude-sonnet-4-6", "gemini/gemini-3.1-pro-preview"),
    )

    assert slugs == (
        "gpt/gpt-5.5",
        "gpt/gpt-5.5",
        "gpt/gpt-5.5",
    )


def test_storyboard_default_candidate_keeps_configured_fallback():
    chain = _storyboard_client_chain(
        "claude/claude-sonnet-4-6",
        ("claude/claude-sonnet-4-6", "gemini/gemini-3.1-pro-preview"),
    )

    assert chain == (
        "claude/claude-sonnet-4-6",
        "gemini/gemini-3.1-pro-preview",
    )


def test_to_storyboard_patch_mode_merges_valid_patch(monkeypatch, tmp_path):
    calls: list[str] = []

    class FakeClient:
        def __init__(self, slug: str):
            self.slug = slug

        def complete_json(self, **_kw):
            calls.append(self.slug)
            return {
                "camera": _valid_storyboard_raw()["shots"][0]["camera"],
                "objects": _valid_storyboard_raw()["shots"][0]["objects"],
                "overlay_zone": _valid_storyboard_raw()["shots"][0]["overlay_zone"],
            }

    def fake_from_chain(chain, **_kw):
        assert len(chain) == 1
        return FakeClient(chain[0])

    monkeypatch.setattr(
        storyboard_agent.LLMClient,
        "from_chain",
        staticmethod(fake_from_chain),
    )
    monkeypatch.setattr(
        storyboard_agent,
        "get_agent_model",
        lambda _agent: SimpleNamespace(chain=(
            "claude/claude-sonnet-4-6",
            "gemini/gemini-3.1-pro-preview",
        )),
    )

    sb = storyboard_agent.to_storyboard(
        _demo_narrative(),
        out_dir=tmp_path,
        candidates=1,
    )

    assert sb.concept_id == "demo"
    assert calls == ["claude/claude-sonnet-4-6"]
    assert (tmp_path / "storyboard.patch_records.json").exists()
    assert any(obj.keyframes for obj in sb.shots[0].objects)
    selection = json.loads((tmp_path / "storyboard.selection.json").read_text())
    assert selection["motion_postprocess"]["changed"] is True


def test_storyboard_motion_postprocess_adds_camera_and_object_motion():
    raw = _deterministic_storyboard_raw(_demo_narrative())
    for shot in raw["shots"]:
        for cam in shot["camera"]:
            cam["position"] = [0, -4, 2]
            cam["look_at"] = [0, 0, 0]
        for obj in shot["objects"]:
            obj["keyframes"] = []

    audit = _ensure_storyboard_motion(raw)
    sb = validate_storyboard(raw)

    assert audit["changed"] is True
    assert any(a["camera"] for a in audit["shots"])
    assert all(any(obj.keyframes for obj in shot.objects) for shot in sb.shots)
    assert all(
        max(k.time_sec for k in shot.camera) <= shot.start_sec + shot.duration_sec
        for shot in sb.shots
    )


def test_storyboard_motion_postprocess_preserves_static_anchors():
    raw = _deterministic_storyboard_raw(_demo_narrative())
    shot = raw["shots"][0]
    shot["objects"] = [
        {
            "name": "floor_grid",
            "type": "primitive",
            "primitive": "plane",
            "location": [0, 0, 0],
            "properties": {},
            "keyframes": [],
        },
        {
            "name": "moving_object_point",
            "type": "primitive",
            "primitive": "sphere",
            "location": [1, 0, 1],
            "properties": {},
            "keyframes": [],
        },
    ]

    _ensure_storyboard_motion(raw)
    by_name = {obj["name"]: obj for obj in shot["objects"]}

    assert by_name["floor_grid"]["keyframes"] == []
    assert len(by_name["moving_object_point"]["keyframes"]) == 2


def test_to_storyboard_full_mode_tries_fallback_after_schema_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CG_TUTOR_STORYBOARD_MODE", "full")
    calls: list[str] = []

    class FakeClient:
        def __init__(self, slug: str):
            self.slug = slug

        def complete_json(self, **_kw):
            calls.append(self.slug)
            if "claude" in self.slug:
                return {"node_id": "node_01", "shots": []}
            return _valid_storyboard_raw()

    monkeypatch.setattr(
        storyboard_agent.LLMClient,
        "from_chain",
        staticmethod(lambda chain, **_kw: FakeClient(chain[0])),
    )
    monkeypatch.setattr(
        storyboard_agent,
        "get_agent_model",
        lambda _agent: SimpleNamespace(chain=(
            "claude/claude-sonnet-4-6",
            "gemini/gemini-3.1-pro-preview",
        )),
    )

    sb = storyboard_agent.to_storyboard(
        _demo_narrative(),
        out_dir=tmp_path,
        candidates=1,
    )

    assert sb.concept_id == "demo"
    assert calls == [
        "claude/claude-sonnet-4-6",
        "claude/claude-sonnet-4-6",
        "gemini/gemini-3.1-pro-preview",
    ]


def test_storyboard_skeleton_preserves_narrative_timeline():
    skel = _storyboard_skeleton(_demo_narrative())
    assert skel["concept_id"] == "demo"
    assert skel["fps"] == 24
    assert skel["resolution"] == [960, 540]
    assert skel["shots"][0]["node_id"] == "node_01"
    assert skel["shots"][0]["start_sec"] == 0.0
    assert skel["shots"][0]["duration_sec"] == 2.0
    assert skel["shots"][0]["formula"] == "R = V - 2(V \\cdot N)N"
    assert skel["shots"][0]["overlay_zone"] == {
        "x": 0.56, "y": 0.07, "w": 0.36, "h": 0.18,
    }


def test_storyboard_skeleton_uses_null_overlay_when_node_has_no_formula():
    skel = _storyboard_skeleton(_formula_free_narrative())

    assert skel["shots"][0]["formula"] is None
    assert skel["shots"][0]["overlay_zone"] is None


def test_deterministic_storyboard_fallback_covers_all_narrative_nodes():
    narrative = _multi_node_narrative()
    profile = SceneProfile(
        profile_id="demo_profile",
        base_profile="vector_teaching",
        persistent_anchors=[
            "camera_center_c",
            "pinhole_aperture",
            "translucent_image_plane",
            "colored_3d_object",
        ],
    )

    raw = _deterministic_storyboard_raw(narrative, scene_profile=profile)
    sb = validate_storyboard(raw)
    _validate_storyboard_against_narrative(sb, narrative)

    assert [shot.node_id for shot in sb.shots] == ["node_01", "node_02"]
    assert sb.total_duration == 7.5
    object_names = {obj.name for shot in sb.shots for obj in shot.objects}
    assert "camera_center_c" in object_names
    assert "translucent_image_plane" in object_names
    assert sb.shots[1].formula == "x' = f x / z"


def test_deterministic_storyboard_fallback_uses_profile_specific_anchors():
    narrative = _multi_node_narrative()
    profile = SceneProfile(
        profile_id="shape_demo",
        base_profile="transformation_demo",
    )

    raw = _deterministic_storyboard_raw(narrative, scene_profile=profile)
    sb = validate_storyboard(raw)

    object_names = {obj.name for shot in sb.shots for obj in shot.objects}
    assert "hero_mesh" in object_names
    assert "trajectory_trace" in object_names
    assert "projection_ray" not in object_names
    assert "pinhole_aperture" not in object_names


def test_deterministic_storyboard_fallback_uses_generic_vector_teaching_anchors():
    narrative = _multi_node_narrative()
    profile = SceneProfile(
        profile_id="generic_vector",
        base_profile="vector_teaching",
    )

    raw = _deterministic_storyboard_raw(narrative, scene_profile=profile)
    sb = validate_storyboard(raw)

    object_names = {obj.name for shot in sb.shots for obj in shot.objects}
    assert "main_teaching_object" in object_names
    assert "vector_cue" in object_names
    assert "projection_ray" not in object_names
    assert "pinhole_aperture" not in object_names


def test_shot_patch_rejects_structural_keys():
    with pytest.raises(ValueError, match="forbidden structural keys"):
        _validate_shot_patch({
            "node_id": "node_01",
            "camera": _valid_storyboard_raw()["shots"][0]["camera"],
            "objects": _valid_storyboard_raw()["shots"][0]["objects"],
            "overlay_zone": None,
        }, _valid_storyboard_raw()["shots"][0])


def test_shot_patch_merge_preserves_structural_timing_and_formula():
    shot = _deterministic_storyboard_raw(_demo_narrative())["shots"][0]
    patch = ShotPatch.model_validate({
        "camera": _valid_storyboard_raw()["shots"][0]["camera"],
        "objects": _valid_storyboard_raw()["shots"][0]["objects"],
        "overlay_zone": {"x": 0.5, "y": 0.06, "w": 0.3, "h": 0.12},
    })

    _merge_patch_into_shot(shot, patch)

    assert shot["node_id"] == "node_01"
    assert shot["start_sec"] == 0.0
    assert shot["duration_sec"] == 2.0
    assert shot["formula"] == "R = V - 2(V \\cdot N)N"
    assert shot["overlay_zone"]["x"] == 0.5


def test_storyboard_patch_prompt_forbids_root_keys():
    prompt = _storyboard_patch_user_prompt(
        narrative=_demo_narrative(),
        shot_idx=0,
        base_shot=_deterministic_storyboard_raw(_demo_narrative())["shots"][0],
        style_addendum="STYLE",
        scene_profile=None,
    )

    assert "Do not return root Storyboard keys" in prompt
    assert "concept_id" in prompt
    assert "STYLE" in prompt


def test_to_storyboard_uses_deterministic_fallback_after_all_providers_fail(
    monkeypatch,
    tmp_path,
):
    class FakeClient:
        def complete_json(self, **_kw):
            return {
                "node_id": "node_01",
                "start_sec": 0.0,
                "duration_sec": 3.5,
                "camera": [{
                    "time_sec": 0.0,
                    "position": [0, -5, 3],
                    "look_at": [0, 0, 0],
                    "fov": 50,
                }],
                "objects": [{
                    "name": "key_light",
                    "type": "light",
                    "primitive": None,
                }],
                "shots": [],
            }

    monkeypatch.setattr(
        storyboard_agent.LLMClient,
        "from_chain",
        staticmethod(lambda *_args, **_kw: FakeClient()),
    )
    monkeypatch.setattr(
        storyboard_agent,
        "get_agent_model",
        lambda _agent: SimpleNamespace(chain=(
            "anthropic/claude-sonnet-4.6",
            "google/gemini-3.1-pro-preview",
        )),
    )

    sb = storyboard_agent.to_storyboard(
        _multi_node_narrative(),
        out_dir=tmp_path,
        candidates=1,
    )

    assert [shot.node_id for shot in sb.shots] == ["node_01", "node_02"]
    assert (tmp_path / "storyboard.patch_records.json").exists()
    assert (tmp_path / "storyboard.json").exists()


def test_storyboard_repair_prompt_forbids_node_root():
    prompt = _storyboard_repair_user_prompt(
        _demo_narrative(),
        {"node_id": "node_01", "shots": []},
        ValueError("concept_id missing"),
    )
    assert "top-level object must NOT contain node_id" in prompt
    assert "shots must contain exactly 1 item(s)" in prompt
    assert "synthesize the missing shots" in prompt
    assert "concept_id missing" in prompt
    assert '"concept_id": "demo"' in prompt
    assert '"node_id": "node_01"' in prompt


def test_storyboard_repair_prompt_keeps_formula_free_nodes_formula_free():
    prompt = _storyboard_repair_user_prompt(
        _formula_free_narrative(),
        {"node_id": "node_01", "shots": []},
        ValueError("concept_id missing"),
    )

    assert "formulas: []" in prompt
    assert "shot.formula to null and overlay_zone to null" in prompt
    assert '"formula": null' in prompt
    assert '"overlay_zone": null' in prompt


def test_candidate_schema_failure_repairs_with_same_client(tmp_path: Path):
    invalid = {"node_id": "node_01", "shots": []}
    client = _FakeStoryClient([invalid, _valid_storyboard_raw()])

    raw, sb = _complete_and_validate_candidate(
        client=client,
        system="system",
        user='{"concept_id": "demo"}',
        narrative=_demo_narrative(),
        out_dir=tmp_path,
        artifact_prefix="storyboard.cand00.gemini",
    )

    assert raw["concept_id"] == "demo"
    assert sb.concept_id == "demo"
    assert len(client.users) == 2
    assert "Your previous response was invalid" in client.users[1]
    assert (tmp_path / "storyboard.cand00.gemini.raw.json").exists()
    assert (tmp_path / "storyboard.cand00.gemini.invalid.json").exists()
    assert (tmp_path / "storyboard.cand00.gemini.repair.raw.json").exists()


def test_candidate_rejects_single_shot_when_narrative_has_multiple_nodes(
    tmp_path: Path,
):
    narrative = _multi_node_narrative()
    single_shot_root = {
        "node_id": "node_01",
        "start_sec": 0.0,
        "duration_sec": 3.5,
        "camera": [{
            "time_sec": 0.0,
            "position": [0, -5, 3],
            "look_at": [0, 0, 0],
            "fov": 50,
        }],
        "objects": [{
            "name": "key_light",
            "type": "light",
            "primitive": None,
        }],
        "overlay_zone": None,
        "formula": None,
        "caption": "only the first node",
        "shots": [],
    }
    repaired = {
        "concept_id": "demo_multi",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "node_01",
                "start_sec": 0.0,
                "duration_sec": 3.5,
                "camera": [{
                    "time_sec": 0.0,
                    "position": [0, -5, 3],
                    "look_at": [0, 0, 0],
                    "fov": 50,
                }],
                "objects": [{
                    "name": "key_light",
                    "type": "light",
                    "primitive": None,
                }],
                "overlay_zone": None,
                "formula": None,
                "caption": "Show the camera and image plane.",
            },
            {
                "node_id": "node_02",
                "start_sec": 3.5,
                "duration_sec": 4.0,
                "camera": [{
                    "time_sec": 3.5,
                    "position": [0, -5, 3],
                    "look_at": [0, 0, 0],
                    "fov": 50,
                }],
                "objects": [{
                    "name": "projection_ray",
                    "type": "mesh",
                    "primitive": "curve_polyline",
                }],
                "overlay_zone": {"x": 0.04, "y": 0.06, "w": 0.45, "h": 0.18},
                "formula": "x' = f x / z",
                "caption": "Trace a point through the pinhole.",
            },
        ],
    }
    client = _FakeStoryClient([single_shot_root, repaired])

    raw, sb = _complete_and_validate_candidate(
        client=client,
        system="system",
        user='{"concept_id": "demo_multi"}',
        narrative=narrative,
        out_dir=tmp_path,
        artifact_prefix="storyboard.cand00.claude",
    )

    assert raw["concept_id"] == "demo_multi"
    assert [shot.node_id for shot in sb.shots] == ["node_01", "node_02"]
    assert sb.total_duration == 7.5
    assert len(client.users) == 2
    assert (tmp_path / "storyboard.cand00.claude.invalid.json").exists()


def test_storyboard_narrative_alignment_rejects_dropped_nodes():
    narrative = _multi_node_narrative()
    sb = validate_storyboard({
        "concept_id": "demo_multi",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [_valid_storyboard_raw()["shots"][0] | {"duration_sec": 3.5}],
    })

    with pytest.raises(ValueError, match="must exactly match narrative"):
        _validate_storyboard_against_narrative(sb, narrative)


def test_candidate_schema_repair_failure_records_both_errors(tmp_path: Path):
    invalid = {"node_id": "node_01", "shots": []}
    client = _FakeStoryClient([invalid, invalid])

    with pytest.raises(RuntimeError, match="schema repair also failed"):
        _complete_and_validate_candidate(
            client=client,
            system="system",
            user='{"concept_id": "demo"}',
            narrative=_demo_narrative(),
            out_dir=tmp_path,
            artifact_prefix="storyboard.cand00.gemini",
        )

    err = (tmp_path / "storyboard.cand00.gemini.error.txt").read_text()
    assert "Initial validation failed" in err
    assert "Repair validation/call failed" in err
    assert (tmp_path / "storyboard.cand00.gemini.repair.invalid.json").exists()


# ----- OverlayZone -------------------------------------------------------

def test_overlay_zone_out_of_range_x_rejected():
    with pytest.raises(ValidationError):
        OverlayZone(x=1.5, y=0.1, w=0.2, h=0.2)


def test_overlay_zone_negative_w_rejected():
    with pytest.raises(ValidationError):
        OverlayZone(x=0.0, y=0.0, w=-0.1, h=0.1)


def test_overlay_zone_overflow_right_edge_rejected():
    """x + w must stay within [0, 1]."""
    with pytest.raises(ValidationError, match="overflows right edge"):
        OverlayZone(x=0.8, y=0.1, w=0.5, h=0.2)


def test_overlay_zone_overflow_bottom_edge_rejected():
    with pytest.raises(ValidationError, match="overflows bottom edge"):
        OverlayZone(x=0.1, y=0.7, w=0.2, h=0.5)


def test_overlay_zone_at_exact_unit_passes():
    z = OverlayZone(x=0.5, y=0.5, w=0.5, h=0.5)
    assert z.x + z.w == 1.0


# ----- CameraKey / Shot --------------------------------------------------

def test_camera_fov_must_be_in_open_zero_one_eighty():
    with pytest.raises(ValidationError):
        CameraKey(time_sec=0, position=(0, 0, 0), look_at=(0, 0, 0), fov=0)
    with pytest.raises(ValidationError):
        CameraKey(time_sec=0, position=(0, 0, 0), look_at=(0, 0, 0), fov=180)


def test_keyframe_accepts_list_of_vectors_for_group_helpers():
    k = Keyframe(
        time_sec=0.0,
        attr="location",
        value=[[-2.7, -0.75, 0.13], [-0.9, 1.05, 0.13]],
    )
    assert k.value[0] == [-2.7, -0.75, 0.13]


def test_scene_object_accepts_curve_and_group_primitives():
    curve = SceneObject(
        name="control_polygon",
        type="mesh",
        primitive="curve_polyline",
        properties={"points": [[0, 0, 0], [1, 1, 0]], "bevel_depth": 0.02},
    )
    spheres = SceneObject(
        name="control_points",
        type="mesh",
        primitive="sphere_group",
        properties={"points": [[0, 0, 0], [1, 1, 0]], "radius": 0.08},
    )
    cubes = SceneObject(
        name="weight_bars",
        type="mesh",
        primitive="cube_group",
        properties={"centers": [[0, 0, 0], [1, 0, 0]]},
    )
    assert curve.primitive == "curve_polyline"
    assert spheres.primitive == "sphere_group"
    assert cubes.primitive == "cube_group"


def test_shot_camera_key_outside_window_rejected():
    obj = SceneObject(name="o", type="primitive", primitive="sphere")
    with pytest.raises(ValidationError, match="after shot end"):
        Shot(
            node_id="s",
            start_sec=0.0,
            duration_sec=5.0,
            camera=[
                CameraKey(time_sec=0, position=(0, 0, 0), look_at=(0, 0, 0)),
                CameraKey(time_sec=10, position=(0, 0, 0), look_at=(0, 0, 0)),
            ],
            objects=[obj],
        )


def test_shot_requires_at_least_one_camera_key_and_object():
    with pytest.raises(ValidationError):
        Shot(node_id="s", start_sec=0, duration_sec=1, camera=[], objects=[])


def test_shot_duration_must_be_positive():
    obj = SceneObject(name="o", type="primitive", primitive="sphere")
    cam = CameraKey(time_sec=0, position=(0, 0, 0), look_at=(0, 0, 0))
    with pytest.raises(ValidationError):
        Shot(node_id="s", start_sec=0, duration_sec=0,
             camera=[cam], objects=[obj])


# ----- Storyboard --------------------------------------------------------

def _build_shot(node_id: str, start: float, dur: float) -> Shot:
    obj = SceneObject(name="o", type="primitive", primitive="sphere")
    cam = CameraKey(time_sec=start, position=(0, 0, 0), look_at=(0, 0, 0))
    return Shot(
        node_id=node_id, start_sec=start, duration_sec=dur,
        camera=[cam], objects=[obj],
    )


def test_storyboard_shots_must_be_contiguous():
    sb_data = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            _build_shot("a", 0.0, 2.0).model_dump(),
            _build_shot("b", 3.0, 2.0).model_dump(),  # GAP between 2.0 and 3.0
        ],
    }
    with pytest.raises(ValidationError, match="contiguous"):
        Storyboard.model_validate(sb_data)


def test_storyboard_contiguous_shots_pass():
    sb_data = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            _build_shot("a", 0.0, 2.0).model_dump(),
            _build_shot("b", 2.0, 2.0).model_dump(),
        ],
    }
    sb = Storyboard.model_validate(sb_data)
    assert sb.total_duration == 4.0


def test_storyboard_relative_key_times_can_be_normalized():
    raw = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a",
                "start_sec": 0.0,
                "duration_sec": 2.0,
                "camera": [
                    {"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]},
                    {"time_sec": 2.0, "position": [1, 0, 1], "look_at": [0, 0, 0]},
                ],
                "objects": [
                    {
                        "name": "o",
                        "type": "primitive",
                        "primitive": "sphere",
                        "keyframes": [{"time_sec": 1.0, "attr": "scale", "value": 1.0}],
                    }
                ],
            },
            {
                "node_id": "b",
                "start_sec": 2.0,
                "duration_sec": 3.0,
                "camera": [
                    {"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]},
                    {"time_sec": 3.0, "position": [1, 0, 1], "look_at": [0, 0, 0]},
                ],
                "objects": [
                    {
                        "name": "o",
                        "type": "primitive",
                        "primitive": "sphere",
                        "keyframes": [{"time_sec": 1.5, "attr": "scale", "value": 2.0}],
                    }
                ],
            },
        ],
    }
    normalized = _normalize_relative_times(raw)
    sb = Storyboard.model_validate(normalized)
    assert [k.time_sec for k in sb.shots[1].camera] == [2.0, 5.0]
    assert sb.shots[1].objects[0].keyframes[0].time_sec == 3.5


def test_validate_storyboard_applies_normalization():
    raw = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a",
                "start_sec": 0.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]}],
                "objects": [{"name": "o", "type": "mesh", "primitive": "sphere"}],
            },
            {
                "node_id": "b",
                "start_sec": 1.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]}],
                "objects": [{"name": "line", "type": "mesh", "primitive": "curve_polyline"}],
            },
        ],
    }
    sb = validate_storyboard(raw)
    assert sb.shots[1].camera[0].time_sec == 1.0
    assert sb.shots[1].objects[0].primitive == "curve_polyline"


def test_normalize_shot_start_times_recomputes_when_all_zero():
    """Mirror the real gemini-3.1-pro-preview affine failure: every shot has
    start_sec=0.0 but durations are correct. Should be auto-recovered."""
    raw = {
        "concept_id": "c",
        "shots": [
            {"node_id": "a", "start_sec": 0.0, "duration_sec": 4.5},
            {"node_id": "b", "start_sec": 0.0, "duration_sec": 4.5},
            {"node_id": "c", "start_sec": 0.0, "duration_sec": 3.5},
            {"node_id": "d", "start_sec": 0.0, "duration_sec": 5.5},
        ],
    }
    fixed = _normalize_shot_start_times(raw)
    starts = [s["start_sec"] for s in fixed["shots"]]
    assert starts == [0.0, 4.5, 9.0, 12.5]


def test_normalize_shot_start_times_recomputes_when_missing():
    """start_sec key missing entirely → treat as 0 and recompute."""
    raw = {
        "concept_id": "c",
        "shots": [
            {"node_id": "a", "duration_sec": 2.0},
            {"node_id": "b", "duration_sec": 3.0},
        ],
    }
    fixed = _normalize_shot_start_times(raw)
    assert [s["start_sec"] for s in fixed["shots"]] == [0.0, 2.0]


def test_normalize_shot_start_times_trusts_non_zero_starts():
    """If the model set any non-zero start_sec we trust it and leave
    Pydantic to surface real contiguity violations."""
    raw = {
        "concept_id": "c",
        "shots": [
            {"node_id": "a", "start_sec": 0.0, "duration_sec": 2.0},
            {"node_id": "b", "start_sec": 99.0, "duration_sec": 3.0},
        ],
    }
    fixed = _normalize_shot_start_times(raw)
    assert fixed["shots"][1]["start_sec"] == 99.0


def test_normalize_shot_start_times_skips_when_durations_bad():
    """Without usable durations there's nothing to compute — pass through."""
    raw = {
        "concept_id": "c",
        "shots": [
            {"node_id": "a", "start_sec": 0.0, "duration_sec": 0.0},
            {"node_id": "b", "start_sec": 0.0, "duration_sec": 2.0},
        ],
    }
    assert _normalize_shot_start_times(raw) == raw


def test_validate_storyboard_recovers_affine_all_zero_failure():
    """End-to-end: a storyboard with all-zero starts should now validate
    instead of raising the 'shots must be contiguous' error."""
    raw = {
        "concept_id": "affine_transformation",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": f"node_{i:02d}",
                "start_sec": 0.0,
                "duration_sec": dur,
                "camera": [
                    {"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]},
                    {"time_sec": dur, "position": [0, 0, 1], "look_at": [0, 0, 0]},
                ],
                "objects": [{"name": "o", "type": "primitive", "primitive": "sphere"}],
            }
            for i, dur in enumerate([4.5, 4.5, 3.5, 5.5], 1)
        ],
    }
    sb = validate_storyboard(raw)
    assert [s.start_sec for s in sb.shots] == [0.0, 4.5, 9.0, 12.5]


def test_validate_storyboard_normalizes_curve_primitive_alias():
    raw = {
        "concept_id": "c",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [
            {
                "node_id": "a",
                "start_sec": 0.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 0.0, "position": [0, 0, 1], "look_at": [0, 0, 0]}],
                "objects": [{"name": "curve", "type": "mesh", "primitive": "curve"}],
            },
        ],
    }
    sb = validate_storyboard(raw)
    assert sb.shots[0].objects[0].primitive == "curve_polyline"


def test_validate_storyboard_wraps_single_shot_root():
    raw = {
        "node_id": "node_01",
        "start_sec": 0.0,
        "duration_sec": 3.5,
        "camera": [{
            "time_sec": 0.0,
            "position": [0, -5, 3],
            "look_at": [0, 0, 0],
            "fov": 50,
        }],
        "objects": [{
            "name": "key_light",
            "type": "light",
            "primitive": None,
        }],
        "overlay_zone": None,
        "formula": None,
        "caption": "single shot returned at root",
        "shots": [],
    }

    normalized = _normalize_single_shot_root(raw)
    sb = validate_storyboard(raw)

    assert normalized["concept_id"] == "unknown_concept"
    assert len(sb.shots) == 1
    assert sb.shots[0].node_id == "node_01"
    assert sb.shots[0].caption == "single shot returned at root"


def test_storyboard_user_prompt_is_skeleton_first():
    narrative = _demo_narrative()

    prompt = _storyboard_user_prompt(narrative, style_addendum="STYLE PROFILE")

    assert "FILL THIS STORYBOARD SKELETON" in prompt
    assert '"concept_id": "demo"' in prompt
    assert '"shots": [' in prompt
    assert "Do not remove root keys" in prompt
    assert "STYLE PROFILE" in prompt


def test_storyboard_fps_must_be_positive():
    base = json.loads(FIXTURE.read_text())
    base["fps"] = 0
    with pytest.raises(ValidationError):
        Storyboard.model_validate(base)


def test_storyboard_resolution_must_be_positive():
    base = json.loads(FIXTURE.read_text())
    base["resolution"] = [0, 540]
    with pytest.raises(ValidationError):
        Storyboard.model_validate(base)
