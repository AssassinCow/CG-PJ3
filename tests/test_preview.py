from pathlib import Path

from PIL import Image

from cg_tutor.preview import (
    PreviewIssue,
    PreviewReport,
    preview_blocks_allow_render_repair,
    select_preview_frames,
    verify_preview_frames,
)
from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import base_profile


def _storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "preview",
        "fps": 10,
        "resolution": [100, 100],
        "shots": [
            {
                "node_id": "a",
                "start_sec": 0.0,
                "duration_sec": 2.0,
                "camera": [{"time_sec": 0, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "a", "type": "mesh", "primitive": "sphere"}],
            },
            {
                "node_id": "b",
                "start_sec": 2.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 2, "position": [0, 0, 0], "look_at": [0, 0, 0]}],
                "objects": [{"name": "b", "type": "mesh", "primitive": "sphere"}],
            },
        ],
    })


def _formula_storyboard() -> Storyboard:
    raw = _storyboard().model_dump(mode="json")
    raw["shots"][0]["formula"] = "I = I_a"
    raw["shots"][0]["overlay_zone"] = {
        "x": 0.0,
        "y": 0.0,
        "w": 0.5,
        "h": 0.5,
    }
    return Storyboard.model_validate(raw)


def test_select_preview_frames_picks_midpoints():
    assert select_preview_frames(_storyboard()) == [10, 25]


def test_select_preview_frames_picks_two_samples_for_long_shot():
    raw = _storyboard().model_dump(mode="json")
    raw["shots"] = [raw["shots"][0]]
    raw["shots"][0]["duration_sec"] = 4.0
    sb = Storyboard.model_validate(raw)

    assert select_preview_frames(sb) == [10, 30]


def test_verify_preview_frames_flags_missing_and_black(tmp_path: Path):
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(tmp_path, [10, 25])

    assert not report.ok
    assert any(i.rule_id == "near_black_frame" for i in report.issues)
    assert any(i.rule_id == "missing_preview_frame" for i in report.issues)


def test_verify_preview_frames_accepts_visible_frame(tmp_path: Path):
    Image.new("RGB", (8, 8), (80, 120, 160)).save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(tmp_path, [10])

    assert report.ok
    assert report.rendered_frames == [10]


def test_verify_preview_frames_allows_dim_but_structured_cinematic_frame(tmp_path: Path):
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    img.putpixel((6, 6), (120, 20, 20))
    img.save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(
        tmp_path,
        [10],
        scene_profile=base_profile("cinematic_application"),
    )

    assert report.ok
    assert any(i.rule_id == "cinematic_low_light_frame" for i in report.issues)


def test_verify_preview_frames_blocks_pure_black_cinematic_frame(tmp_path: Path):
    Image.new("RGB", (8, 8), (0, 0, 0)).save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(
        tmp_path,
        [10],
        scene_profile=base_profile("cinematic_application"),
    )

    assert not report.ok
    assert any(i.rule_id == "near_black_frame" for i in report.issues)


def test_verify_preview_frames_warns_overlay_zone_occupied(tmp_path: Path):
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    for x in range(0, 40):
        for y in range(0, 30):
            img.putpixel((x, y), (180, 180, 180))
    img.save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(
        tmp_path,
        [10],
        storyboard=_formula_storyboard(),
    )

    assert report.ok
    assert any(i.rule_id == "overlay_zone_occupied_before_compose"
               for i in report.issues)


def test_verify_preview_frames_warns_overlay_zone_for_teaching_profile(tmp_path: Path):
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    for x in range(0, 40):
        for y in range(0, 30):
            img.putpixel((x, y), (180, 180, 180))
    img.save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(
        tmp_path,
        [10],
        storyboard=_formula_storyboard(),
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert any(
        i.rule_id == "overlay_zone_occupied_before_compose"
        and i.severity == "warn"
        for i in report.issues
    )


def test_verify_preview_frames_warns_edge_activity_for_teaching(tmp_path: Path):
    img = Image.new("RGB", (100, 100), (20, 20, 20))
    for x in range(0, 100):
        for y in range(0, 3):
            img.putpixel((x, y), (240, 240, 240))
    img.save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(tmp_path, [10])

    assert report.ok
    assert any(i.rule_id == "teaching_object_near_frame_edge"
               for i in report.issues)


def test_verify_preview_frames_warns_edge_activity_for_teaching_profile(tmp_path: Path):
    img = Image.new("RGB", (100, 100), (20, 20, 20))
    for x in range(0, 100):
        for y in range(0, 3):
            img.putpixel((x, y), (240, 240, 240))
    img.save(tmp_path / "frame_0010.png")

    report = verify_preview_frames(
        tmp_path,
        [10],
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert any(
        i.rule_id == "teaching_object_near_frame_edge"
        and i.severity == "warn"
        for i in report.issues
    )


def test_verify_preview_frames_blocks_insufficient_visible_motion(tmp_path: Path):
    raw = _storyboard().model_dump(mode="json")
    raw["shots"] = [raw["shots"][0]]
    raw["shots"][0]["duration_sec"] = 4.0
    sb = Storyboard.model_validate(raw)
    Image.new("RGB", (100, 100), (80, 120, 160)).save(tmp_path / "frame_0010.png")
    Image.new("RGB", (100, 100), (81, 121, 161)).save(tmp_path / "frame_0030.png")

    report = verify_preview_frames(
        tmp_path,
        [10, 30],
        storyboard=sb,
        scene_profile=base_profile("transformation_demo"),
    )

    assert not report.ok
    assert any(i.rule_id == "insufficient_visible_motion" for i in report.issues)
    assert preview_blocks_allow_render_repair(report)


def test_preview_render_repair_only_allows_motion_block():
    report = PreviewReport(
        ok=False,
        issues=[
            PreviewIssue(
                severity="block",
                rule_id="insufficient_visible_motion",
                frame_idx=None,
                message="too static",
                suggested_fix="move something",
            ),
            PreviewIssue(
                severity="warn",
                rule_id="teaching_object_near_frame_edge",
                frame_idx=10,
                message="edge",
                suggested_fix="reframe",
            ),
        ],
    )
    assert preview_blocks_allow_render_repair(report)

    report.issues.append(PreviewIssue(
        severity="block",
        rule_id="missing_preview_frame",
        frame_idx=20,
        message="missing",
        suggested_fix="render frame",
    ))
    assert not preview_blocks_allow_render_repair(report)


def test_verify_preview_frames_accepts_visible_motion(tmp_path: Path):
    raw = _storyboard().model_dump(mode="json")
    raw["shots"] = [raw["shots"][0]]
    raw["shots"][0]["duration_sec"] = 4.0
    sb = Storyboard.model_validate(raw)
    Image.new("RGB", (100, 100), (40, 60, 80)).save(tmp_path / "frame_0010.png")
    Image.new("RGB", (100, 100), (150, 120, 80)).save(tmp_path / "frame_0030.png")

    report = verify_preview_frames(
        tmp_path,
        [10, 30],
        storyboard=sb,
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert not any(i.rule_id == "insufficient_visible_motion" for i in report.issues)


def test_vector_teaching_allows_static_per_shot_stepwise_change(tmp_path: Path):
    raw = _storyboard().model_dump(mode="json")
    raw["fps"] = 10
    raw["shots"] = [raw["shots"][0], raw["shots"][0].copy()]
    raw["shots"][0]["node_id"] = "node_01"
    raw["shots"][0]["start_sec"] = 0.0
    raw["shots"][0]["duration_sec"] = 4.0
    raw["shots"][1]["node_id"] = "node_02"
    raw["shots"][1]["start_sec"] = 4.0
    raw["shots"][1]["duration_sec"] = 4.0
    raw["shots"][1]["camera"] = [
        {"time_sec": 4.0, "position": [0, 0, 0], "look_at": [0, 0, 0]},
    ]
    sb = Storyboard.model_validate(raw)

    # Each shot is static internally, but the diagram changes between shots.
    Image.new("RGB", (100, 100), (80, 120, 160)).save(tmp_path / "frame_0010.png")
    Image.new("RGB", (100, 100), (80, 120, 160)).save(tmp_path / "frame_0030.png")
    Image.new("RGB", (100, 100), (160, 120, 80)).save(tmp_path / "frame_0050.png")
    Image.new("RGB", (100, 100), (160, 120, 80)).save(tmp_path / "frame_0070.png")

    report = verify_preview_frames(
        tmp_path,
        [10, 30, 50, 70],
        storyboard=sb,
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert not any(i.rule_id == "insufficient_visible_motion" for i in report.issues)


def test_vector_teaching_with_storyboard_keyframes_requires_per_shot_motion(tmp_path: Path):
    raw = _storyboard().model_dump(mode="json")
    raw["fps"] = 10
    raw["shots"] = [raw["shots"][0], raw["shots"][0].copy()]
    raw["shots"][0]["node_id"] = "node_01"
    raw["shots"][0]["start_sec"] = 0.0
    raw["shots"][0]["duration_sec"] = 4.0
    raw["shots"][0]["objects"][0]["keyframes"] = [
        {"time_sec": 0.0, "attr": "scale", "value": [0.8, 0.8, 0.8]},
        {"time_sec": 3.5, "attr": "scale", "value": [1.2, 1.2, 1.2]},
    ]
    raw["shots"][1]["node_id"] = "node_02"
    raw["shots"][1]["start_sec"] = 4.0
    raw["shots"][1]["duration_sec"] = 4.0
    raw["shots"][1]["camera"] = [
        {"time_sec": 4.0, "position": [0, 0, 0], "look_at": [0, 0, 0]},
    ]
    sb = Storyboard.model_validate(raw)

    # The whole preview changes between shots, but the shot with declared
    # object keyframes is static internally and should be rejected.
    Image.new("RGB", (100, 100), (80, 120, 160)).save(tmp_path / "frame_0010.png")
    Image.new("RGB", (100, 100), (80, 120, 160)).save(tmp_path / "frame_0030.png")
    Image.new("RGB", (100, 100), (160, 120, 80)).save(tmp_path / "frame_0050.png")
    Image.new("RGB", (100, 100), (160, 120, 80)).save(tmp_path / "frame_0070.png")

    report = verify_preview_frames(
        tmp_path,
        [10, 30, 50, 70],
        storyboard=sb,
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "insufficient_visible_motion" for i in report.issues)


def test_vector_teaching_blocks_globally_static_preview(tmp_path: Path):
    raw = _storyboard().model_dump(mode="json")
    raw["fps"] = 10
    raw["shots"] = [raw["shots"][0], raw["shots"][0].copy()]
    raw["shots"][0]["node_id"] = "node_01"
    raw["shots"][0]["start_sec"] = 0.0
    raw["shots"][0]["duration_sec"] = 4.0
    raw["shots"][1]["node_id"] = "node_02"
    raw["shots"][1]["start_sec"] = 4.0
    raw["shots"][1]["duration_sec"] = 4.0
    raw["shots"][1]["camera"] = [
        {"time_sec": 4.0, "position": [0, 0, 0], "look_at": [0, 0, 0]},
    ]
    sb = Storyboard.model_validate(raw)

    for frame in [10, 30, 50, 70]:
        Image.new("RGB", (100, 100), (80, 120, 160)).save(
            tmp_path / f"frame_{frame:04d}.png"
        )

    report = verify_preview_frames(
        tmp_path,
        [10, 30, 50, 70],
        storyboard=sb,
        scene_profile=base_profile("vector_teaching"),
    )

    assert not report.ok
    assert any(i.rule_id == "insufficient_visible_motion" for i in report.issues)
