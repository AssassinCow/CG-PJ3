"""Tests for the dolly_zoom concept-metric plugins.

These are deterministic AST/storyboard checks that fire before vision
critic feedback so that high-confidence code-level failure modes (lens
never keyframed, marker posts animated, etc.) cannot be missed even if
the critic produces low-quality findings.
"""

from __future__ import annotations

import pytest

from cg_tutor.auto_success_spec import AutoSuccessRule, AutoSuccessSpec
from cg_tutor.concept_metrics import (
    ConceptMetricReport,
    _CONCEPT_METRIC_REGISTRY,
    _keyframed_object_names,
    concept_metric_report_to_json,
    format_concept_metric_report_for_coder,
    register_concept_metric,
    run_concept_metrics,
)
from cg_tutor.success_spec import SuccessSpec


def _generic_storyboard_with_object_keyframes():
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": "generic_dynamic",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [
                {
                    "time_sec": 0.0,
                    "position": (0, -3, 1.5),
                    "look_at": (0, 0, 1.5),
                    "fov": 60.0,
                },
            ],
            "objects": [{
                "name": "incident_white_ray",
                "type": "curve",
                "keyframes": [
                    {"time_sec": 0.0, "attr": "scale", "value": [0.8, 0.8, 0.8]},
                    {"time_sec": 3.0, "attr": "scale", "value": [1.0, 1.0, 1.0]},
                ],
            }],
        }],
    })


def _generic_storyboard_without_object_keyframes():
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": "generic",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": (0, -3, 1.5),
                "look_at": (0, 0, 1.5),
                "fov": 60.0,
            }],
            "objects": [{
                "name": "static_anchor",
                "type": "primitive",
                "primitive": "sphere",
            }],
        }],
    })


def test_auto_success_spec_missing_object_is_soft_not_hard():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code="import bpy\n",
        storyboard=_generic_storyboard_without_object_keyframes(),
        auto_success_spec=AutoSuccessSpec(rules=[
            AutoSuccessRule(kind="object_visible", anchors=["mip_level_stack"]),
        ]),
    )

    assert any(
        issue.rule_id == "auto_success_object_visible_missing"
        for issue in report.issues
    )
    assert report.success_soft_count == 1
    assert report.success_hard_count == 0
    assert report.ok


def test_auto_success_spec_lod_token_matches_created_text_object():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code=(
            "import bpy\n"
            "label = bpy.data.objects.new('text_lod1', None)\n"
        ),
        storyboard=_generic_storyboard_without_object_keyframes(),
        auto_success_spec=AutoSuccessSpec(rules=[
            AutoSuccessRule(kind="object_visible", anchors=["LOD1"]),
        ]),
    )

    assert not any(
        issue.rule_id == "auto_success_object_visible_missing"
        for issue in report.issues
    )


def test_auto_success_spec_reports_anchor_static_status():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code=(
            "import bpy\n"
            "stack = bpy.data.objects.new('mip_level_stack', None)\n"
        ),
        storyboard=_generic_storyboard_without_object_keyframes(),
        auto_success_spec=AutoSuccessSpec(rules=[
            AutoSuccessRule(kind="object_visible", anchors=["mip_level_stack"]),
            AutoSuccessRule(kind="object_visible", anchors=["near_checker_patch"]),
        ]),
    )

    status = report.metrics["auto_success_spec"]["anchor_status"]
    assert status["mip_level_stack"]["created"] is True
    assert status["mip_level_stack"]["matched_names"] == ["mip_level_stack"]
    assert status["near_checker_patch"]["created"] is False


def test_auto_success_spec_static_status_detects_helper_name_parameter():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code=(
            "import bpy\n"
            "def add_curve_polyline(name, points):\n"
            "    curve = bpy.data.curves.new(name, type='CURVE')\n"
            "    obj = bpy.data.objects.new(name, curve)\n"
            "    return obj\n"
            "near = add_curve_polyline('near_checker_patch', [])\n"
        ),
        storyboard=_generic_storyboard_without_object_keyframes(),
        auto_success_spec=AutoSuccessSpec(rules=[
            AutoSuccessRule(kind="object_visible", anchors=["near_checker_patch"]),
        ]),
    )

    status = report.metrics["auto_success_spec"]["anchor_status"]
    assert status["near_checker_patch"]["created"] is True
    assert status["near_checker_patch"]["matched_names"] == ["near_checker_patch"]
    assert not any(
        issue.rule_id == "auto_success_object_visible_missing"
        for issue in report.issues
    )


def test_auto_success_spec_diagnostic_rule_does_not_emit_metric_issue():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code="import bpy\n",
        storyboard=_generic_storyboard_without_object_keyframes(),
        auto_success_spec=AutoSuccessSpec(rules=[
            AutoSuccessRule(
                kind="progressive_visual_ordering",
                anchors=["LOD0", "LOD1", "LOD2"],
                failure_class="diagnostic",
            ),
        ]),
    )

    assert not report.issues


def _storyboard_with_dolly_path_keyframes():
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": "generic",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": (0, -3, 1.5),
                "look_at": (0, 0, 1.5),
                "fov": 60.0,
            }],
            "objects": [{
                "name": "camera_dolly_path",
                "type": "curve",
                "keyframes": [
                    {"time_sec": 0.0, "attr": "location", "value": [0, 0, 0]},
                    {"time_sec": 3.0, "attr": "location", "value": [1, 0, 0]},
                ],
            }],
        }],
    })


def test_storyboard_keyframe_coverage_accepts_named_proxy_animation():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code=(
            "import bpy\n"
            "path = bpy.data.objects.new('camera_dolly_path', None)\n"
            "marker = bpy.data.objects.new('camera_dolly_path_marker', None)\n"
            "marker.keyframe_insert('location', frame=1)\n"
            "marker.keyframe_insert('location', frame=72)\n"
        ),
        storyboard=_storyboard_with_dolly_path_keyframes(),
    )

    assert not any(
        issue.rule_id == "storyboard_keyframe_coverage_missing"
        for issue in report.issues
    )
    coverage = report.metrics["storyboard_keyframe_coverage"]
    assert coverage["proxy_covered_objects"] == ["camera_dolly_path"]


def test_storyboard_keyframe_coverage_still_blocks_without_anchor_or_proxy():
    report = run_concept_metrics(
        concept_id="generic",
        scene_code="import bpy\n",
        storyboard=_storyboard_with_dolly_path_keyframes(),
    )

    assert any(
        issue.rule_id == "storyboard_keyframe_coverage_missing"
        for issue in report.issues
    )


def _dolly_zoom_storyboard(camera_y_delta: float = 5.0, fov_delta: float = 30.0):
    """Storyboard with one shot whose camera plan satisfies the coupling check."""
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": "dolly_zoom",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [
                {
                    "time_sec": 0.0,
                    "position": (0, -3, 1.5),
                    "look_at": (0, 0, 1.5),
                    "fov": 60.0,
                },
                {
                    "time_sec": 4.0,
                    "position": (0, -3 - camera_y_delta, 1.5),
                    "look_at": (0, 0, 1.5),
                    "fov": 60.0 - fov_delta,
                },
            ],
            "objects": [
                {"name": "hero_lighthouse", "type": "primitive", "primitive": "cylinder"},
            ],
        }],
    })


_DEPTH_ANCHORS_CODE = (
    "hero = bpy.data.objects.new('hero_lighthouse', None)\n"
    "icon = bpy.data.objects.new('camera_icon', None)\n"
    "m1 = bpy.data.objects.new('marker_post_1', None)\n"
    "m2 = bpy.data.objects.new('marker_post_2', None)\n"
    "m3 = bpy.data.objects.new('marker_post_3', None)\n"
)


def test_keyframed_object_names_resolves_bpy_data_objects_subscript():
    code = (
        "import bpy\n"
        "mp = bpy.data.objects['marker_post_1']\n"
        "mp.keyframe_insert('location', frame=10)\n"
    )
    assert _keyframed_object_names(code) == {"marker_post_1"}


def test_keyframed_object_names_resolves_bpy_data_objects_new():
    code = (
        "import bpy\n"
        "hero = bpy.data.objects.new('hero_lighthouse', None)\n"
        "hero.keyframe_insert('rotation_euler', frame=12)\n"
    )
    assert _keyframed_object_names(code) == {"hero_lighthouse"}


def test_keyframed_object_names_resolves_bpy_context_object_rename():
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add(size=1.0)\n"
        "ray = bpy.context.object\n"
        "ray.name = 'incident_white_ray'\n"
        "ray.keyframe_insert('scale', frame=24)\n"
    )
    assert _keyframed_object_names(code) == {"incident_white_ray"}


def test_keyframed_object_names_falls_back_to_receiver_name():
    """Plain variable receiver without bpy assignment lookup yields the var
    name directly — caller will pattern-match against naming conventions."""
    code = (
        "import bpy\n"
        "marker_post_2.keyframe_insert('location', frame=1)\n"
    )
    assert _keyframed_object_names(code) == {"marker_post_2"}


def test_keyframed_object_names_resolves_for_loop_tuple_aliases():
    code = (
        "import bpy\n"
        "waypoint_1 = bpy.data.objects.new('waypoint_1', None)\n"
        "waypoint_2 = bpy.data.objects.new('waypoint_2', None)\n"
        "waypoint_3 = bpy.data.objects.new('waypoint_3', None)\n"
        "WP1 = (0, 0, 1)\n"
        "WP2 = (1, 0, 2)\n"
        "WP3 = (2, 0, 1)\n"
        "for wp_obj, wp_pos in ((waypoint_1, WP1), (waypoint_2, WP2), (waypoint_3, WP3)):\n"
        "    wp_obj.location = wp_pos\n"
        "    wp_obj.keyframe_insert('location', frame=1)\n"
    )
    assert _keyframed_object_names(code) == {
        "waypoint_1",
        "waypoint_2",
        "waypoint_3",
    }


def test_keyframed_object_names_on_syntax_error_returns_empty():
    assert _keyframed_object_names("def \n") == set()


def test_lens_keyframed_block_when_only_location_animated():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.location = (0, -5, 1.5)\n"
        "cam.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_missing_lens_animation" in rule_ids
    assert next(
        i for i in report.issues
        if i.rule_id == "dolly_zoom_missing_lens_animation"
    ).severity == "block"


def test_storyboard_keyframe_coverage_blocks_missing_anchor_animation():
    code = (
        "import bpy\n"
        "cam = bpy.data.objects['Camera']\n"
        "cam.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="prism_dispersion_teaching",
        scene_code=code,
        storyboard=_generic_storyboard_with_object_keyframes(),
    )
    issue = next(
        i for i in report.issues
        if i.rule_id == "storyboard_keyframe_coverage_missing"
    )
    assert issue.severity == "block"
    assert "incident_white_ray" in issue.message
    metrics = report.metrics["storyboard_keyframe_coverage"]
    assert metrics["coverage"] == 0.0


def test_storyboard_keyframe_coverage_passes_when_anchor_keyframed():
    code = (
        "import bpy\n"
        "ray = bpy.data.objects.new('incident_white_ray', None)\n"
        "ray.keyframe_insert('scale', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="prism_dispersion_teaching",
        scene_code=code,
        storyboard=_generic_storyboard_with_object_keyframes(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "storyboard_keyframe_coverage_missing" not in rule_ids
    metrics = report.metrics["storyboard_keyframe_coverage"]
    assert metrics["coverage"] == 1.0


def test_storyboard_keyframe_coverage_passes_with_for_loop_alias_keyframes():
    from cg_tutor.schemas import Storyboard

    storyboard = Storyboard.model_validate({
        "concept_id": "particle_trail_curve",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": (0, -3, 1.5),
                "look_at": (0, 0, 1.5),
                "fov": 50.0,
            }],
            "objects": [
                {
                    "name": "waypoint_1",
                    "type": "primitive",
                    "keyframes": [
                        {"time_sec": 0.0, "attr": "location", "value": [0, 0, 1]},
                    ],
                },
                {
                    "name": "waypoint_2",
                    "type": "primitive",
                    "keyframes": [
                        {"time_sec": 0.0, "attr": "location", "value": [1, 0, 2]},
                    ],
                },
            ],
        }],
    })
    code = (
        "import bpy\n"
        "waypoint_1 = bpy.data.objects.new('waypoint_1', None)\n"
        "waypoint_2 = bpy.data.objects.new('waypoint_2', None)\n"
        "WP1 = (0, 0, 1)\n"
        "WP2 = (1, 0, 2)\n"
        "for wp_obj, wp_pos in ((waypoint_1, WP1), (waypoint_2, WP2)):\n"
        "    wp_obj.location = wp_pos\n"
        "    wp_obj.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code + _PARTICLE_TRAIL_GOOD_SCENE,
        storyboard=storyboard,
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "storyboard_keyframe_coverage_missing" not in rule_ids
    assert report.metrics["storyboard_keyframe_coverage"]["coverage"] == 1.0


def test_lens_keyframed_passes_when_lens_animated():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.lens = 24\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
        "cam.data.lens = 50\n"
        "cam.data.keyframe_insert('lens', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_missing_lens_animation" not in rule_ids


def test_weak_camera_fov_coupling_blocks_when_storyboard_is_static():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(camera_y_delta=0.5, fov_delta=5.0),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_weak_camera_fov_coupling" in rule_ids


def test_marker_posts_static_blocks_when_marker_keyframed():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
        "mp = bpy.data.objects['marker_post_2']\n"
        "mp.location = (0, 0, 0)\n"
        "mp.keyframe_insert('location', frame=10)\n"
        "mp.location = (0, 1, 0)\n"
        "mp.keyframe_insert('location', frame=40)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    issue = next(
        i for i in report.issues if i.rule_id == "dolly_zoom_marker_posts_animated"
    )
    assert issue.severity == "block"
    assert "marker_post_2" in issue.message
    metrics = report.metrics["dolly_zoom"]
    assert "marker_post_2" in metrics["animated_marker_posts"]


def test_marker_posts_static_passes_when_no_marker_keyframes():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
        "cam.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_marker_posts_animated" not in rule_ids


def test_hero_subject_static_blocks_when_hero_animated():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
        "hero = bpy.data.objects['hero_lighthouse']\n"
        "hero.location = (0, 0, 0)\n"
        "hero.keyframe_insert('location', frame=10)\n"
        "hero.location = (1, 0, 0)\n"
        "hero.keyframe_insert('location', frame=40)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    issue = next(
        i for i in report.issues if i.rule_id == "dolly_zoom_hero_subject_animated"
    )
    assert issue.severity == "block"
    assert "hero_lighthouse" in issue.message


def test_static_hold_keyframes_do_not_count_as_anchor_animation():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.data.lens = 24\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
        "cam.data.lens = 50\n"
        "cam.data.keyframe_insert('lens', frame=96)\n"
        "mp = bpy.data.objects['marker_post_2']\n"
        "mp.location = (0, 0, 0)\n"
        "mp.keyframe_insert('location', frame=10)\n"
        "mp.location = (0, 0, 0)\n"
        "mp.keyframe_insert('location', frame=40)\n"
        "hero = bpy.data.objects['hero_lighthouse']\n"
        "hero.location = (1, 0, 0)\n"
        "hero.keyframe_insert('location', frame=10)\n"
        "hero.location = (1, 0, 0)\n"
        "hero.keyframe_insert('location', frame=40)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_marker_posts_animated" not in rule_ids
    assert "dolly_zoom_hero_subject_animated" not in rule_ids


def test_missing_depth_anchors_blocks():
    code = (
        "import bpy\n"
        "cam = bpy.data.objects['Camera']\n"
        "cam.data.keyframe_insert('lens', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dolly_zoom_missing_depth_anchors" in rule_ids


def test_non_dolly_zoom_concept_has_no_dolly_zoom_findings():
    report = run_concept_metrics(
        concept_id="bezier_curve",
        scene_code="import bpy\n",
        storyboard=_dolly_zoom_storyboard(),
    )
    assert all(not i.rule_id.startswith("dolly_zoom_") for i in report.issues)


def test_dolly_zoom_substring_match_requires_prefix_boundary():
    report = run_concept_metrics(
        concept_id="anti_dolly_zoom",
        scene_code="import bpy\n" + _DEPTH_ANCHORS_CODE,
        storyboard=_dolly_zoom_storyboard(),
    )

    assert all(not i.rule_id.startswith("dolly_zoom_") for i in report.issues)


def test_register_concept_metric_rejects_duplicate_id():
    def _dummy(scene_code, storyboard):
        return [], {}

    original = _CONCEPT_METRIC_REGISTRY.pop("particle_trail_curve")
    try:
        register_concept_metric("particle_trail_curve")(_dummy)
        with pytest.raises(ValueError, match="duplicate concept metric plugin"):
            register_concept_metric("particle_trail_curve")(_dummy)
    finally:
        _CONCEPT_METRIC_REGISTRY["particle_trail_curve"] = original


def test_format_addendum_empty_when_no_issues():
    report = ConceptMetricReport(concept_id="dolly_zoom")
    assert format_concept_metric_report_for_coder(report) == ""


def test_format_addendum_includes_rule_id_and_suggested_fix():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    text = format_concept_metric_report_for_coder(report)
    assert "AUTOMATED CONCEPT METRIC FAILURES" in text
    assert "dolly_zoom_missing_lens_animation" in text
    assert "fix:" in text


def test_report_serialization_is_json_safe():
    code = (
        "import bpy\n"
        + _DEPTH_ANCHORS_CODE
        + "cam = bpy.data.objects['Camera']\n"
        "cam.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="dolly_zoom",
        scene_code=code,
        storyboard=_dolly_zoom_storyboard(),
    )
    import json
    data = json.loads(concept_metric_report_to_json(report))
    assert data["concept_id"] == "dolly_zoom"
    assert "issues" in data and isinstance(data["issues"], list)
    assert "metrics" in data


# ---------------- particle_trail_curve ----------------


def _simple_storyboard(concept_id: str):
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": concept_id,
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [{
                "time_sec": 0.0,
                "position": (0, -3, 1.5),
                "look_at": (0, 0, 1.5),
                "fov": 50.0,
            }],
            "objects": [
                {"name": "placeholder", "type": "primitive", "primitive": "cube"},
            ],
        }],
    })


_PARTICLE_TRAIL_GOOD_SCENE = (
    "import bpy\n"
    "trail = bpy.data.objects.new('trail_curve', None)\n"
    "trail.data.bevel_factor_end = 0.0\n"
    "trail.data.keyframe_insert('bevel_factor_end', frame=1)\n"
    "trail.data.bevel_factor_end = 1.0\n"
    "trail.data.keyframe_insert('bevel_factor_end', frame=96)\n"
    "emitter = bpy.data.objects.new('glow_emitter', None)\n"
    "emitter.location = (0, 0, 0)\n"
    "emitter.keyframe_insert('location', frame=1)\n"
    "emitter.location = (2, 1, 0)\n"
    "emitter.keyframe_insert('location', frame=96)\n"
)


def test_particle_trail_passes_when_bevel_keyframed_and_emitter_animated():
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=_PARTICLE_TRAIL_GOOD_SCENE,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_missing_bevel_keyframe" not in rule_ids
    assert "particle_trail_emitter_static" not in rule_ids
    assert "particle_trail_missing_trail_curve_object" not in rule_ids
    assert "particle_trail_missing_emitter" not in rule_ids


def test_particle_trail_blocks_when_bevel_not_keyframed():
    code = (
        "import bpy\n"
        "trail = bpy.data.objects.new('trail_curve', None)\n"
        "trail.data.bevel_factor_end = 1.0\n"  # set once, no keyframe
        "emitter = bpy.data.objects.new('glow_emitter', None)\n"
        "emitter.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_missing_bevel_keyframe" in rule_ids


def test_particle_trail_blocks_when_bevel_ramp_flat():
    code = (
        "import bpy\n"
        "trail = bpy.data.objects.new('trail_curve', None)\n"
        "trail.data.bevel_factor_end = 0.1\n"
        "trail.data.keyframe_insert('bevel_factor_end', frame=1)\n"
        "trail.data.bevel_factor_end = 0.2\n"  # delta of 0.1 < 0.5 threshold
        "trail.data.keyframe_insert('bevel_factor_end', frame=96)\n"
        "emitter = bpy.data.objects.new('glow_emitter', None)\n"
        "emitter.keyframe_insert('location', frame=1)\n"
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_bevel_ramp_flat" in rule_ids


def test_particle_trail_accepts_bevel_ramp_at_boundary():
    code = _PARTICLE_TRAIL_GOOD_SCENE.replace(
        "trail.data.bevel_factor_end = 1.0",
        "trail.data.bevel_factor_end = 0.5",
    )

    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_bevel_ramp_flat" not in rule_ids


def test_particle_trail_blocks_bevel_ramp_just_below_boundary():
    code = _PARTICLE_TRAIL_GOOD_SCENE.replace(
        "trail.data.bevel_factor_end = 1.0",
        "trail.data.bevel_factor_end = 0.49",
    )

    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_bevel_ramp_flat" in rule_ids


def test_particle_trail_accepts_bevel_ramp_from_loop_variable():
    code = (
        "import bpy\n"
        "trail = bpy.data.objects.new('trail_curve', None)\n"
        "SHOT_FACTORS = [(1, 0.04), (64, 0.18), (96, 1.0)]\n"
        "for frame, factor in SHOT_FACTORS:\n"
        "    trail.data.bevel_factor_end = factor\n"
        "    trail.data.keyframe_insert('bevel_factor_end', frame=frame)\n"
        "emitter = bpy.data.objects.new('glow_emitter', None)\n"
        "emitter.location = (0, 0, 0)\n"
        "emitter.keyframe_insert('location', frame=1)\n"
        "emitter.location = (2, 1, 0)\n"
        "emitter.keyframe_insert('location', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_bevel_ramp_flat" not in rule_ids


def test_particle_trail_blocks_when_emitter_missing():
    code = (
        "import bpy\n"
        "trail = bpy.data.objects.new('trail_curve', None)\n"
        "trail.data.bevel_factor_end = 0.0\n"
        "trail.data.keyframe_insert('bevel_factor_end', frame=1)\n"
        "trail.data.bevel_factor_end = 1.0\n"
        "trail.data.keyframe_insert('bevel_factor_end', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_missing_emitter" in rule_ids


def test_particle_trail_blocks_when_emitter_static():
    code = (
        "import bpy\n"
        "trail = bpy.data.objects.new('trail_curve', None)\n"
        "trail.data.bevel_factor_end = 0.0\n"
        "trail.data.keyframe_insert('bevel_factor_end', frame=1)\n"
        "trail.data.bevel_factor_end = 1.0\n"
        "trail.data.keyframe_insert('bevel_factor_end', frame=96)\n"
        "emitter = bpy.data.objects.new('glow_emitter', None)\n"  # never keyframed
    )
    report = run_concept_metrics(
        concept_id="particle_trail_curve",
        scene_code=code,
        storyboard=_simple_storyboard("particle_trail_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "particle_trail_emitter_static" in rule_ids


# ---------------- depth_of_field_focus_pull ----------------


_DOF_GOOD_SCENE = (
    "import bpy\n"
    "fg = bpy.data.objects.new('foreground_subject', None)\n"
    "mg = bpy.data.objects.new('middleground_subject', None)\n"
    "bg = bpy.data.objects.new('background_subject', None)\n"
    "cam = bpy.data.objects.new('Camera', None)\n"
    "cam.data.dof.use_dof = True\n"
    "cam.data.dof.aperture_fstop = 1.2\n"
    "cam.data.dof.focus_distance = 1.5\n"
    "cam.data.keyframe_insert('dof.focus_distance', frame=1)\n"
    "cam.data.dof.focus_distance = 3.5\n"
    "cam.data.keyframe_insert('dof.focus_distance', frame=48)\n"
    "cam.data.dof.focus_distance = 6.5\n"
    "cam.data.keyframe_insert('dof.focus_distance', frame=96)\n"
)


def _dof_success_spec() -> SuccessSpec:
    return SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 0.25]},
        }],
        "required_visual_evidence": [{
            "kind": "aperture_within_range",
            "anchor": "main_camera",
            "data_path": "data.dof.aperture_fstop",
            "range": [1.4, 2.8],
        }],
        "hard_constraints": [{
            "kind": "text_faces_camera",
            "anchors": [
                {"name": "depth_label_near", "placement": "in_scene"},
                {"name": "lens_readout_hud", "placement": "hud_overlay"},
            ],
        }],
    })


def test_dof_passes_when_focus_keyframed_and_use_dof_enabled():
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=_DOF_GOOD_SCENE,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_missing_focus_keyframe" not in rule_ids
    assert "dof_focus_distance_flat" not in rule_ids
    assert "dof_focus_transition_not_continuous" not in rule_ids
    assert "dof_use_dof_not_enabled" not in rule_ids
    assert "dof_missing_depth_anchors" not in rule_ids
    assert "dof_aperture_too_wide" not in rule_ids


def test_dof_accepts_symbolic_focus_distance_wrapper_keyframes():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.0\n"
        "D_NEAR = dist_to(fg.location)\n"
        "D_MID = dist_to(mg.location)\n"
        "D_FAR = dist_to(bg.location)\n"
        "def key_focus(frame, distance):\n"
        "    cam.data.dof.focus_distance = distance\n"
        "    cam.data.dof.keyframe_insert('focus_distance', frame=frame)\n"
        "key_focus(1, D_NEAR)\n"
        "key_focus(96, D_MID)\n"
        "key_focus(160, D_FAR)\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_focus_distance_flat" not in rule_ids


def test_dof_blocks_symbolic_focus_distance_wrapper_when_same_target_reused():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.0\n"
        "D_NEAR = dist_to(fg.location)\n"
        "def key_focus(frame, distance):\n"
        "    cam.data.dof.focus_distance = distance\n"
        "    cam.data.dof.keyframe_insert('focus_distance', frame=frame)\n"
        "key_focus(1, D_NEAR)\n"
        "key_focus(96, D_NEAR)\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_focus_distance_flat" in rule_ids


def test_dof_blocks_when_focus_transition_too_concentrated():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.8\n"
        "cam.data.dof.focus_distance = 1.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=1)\n"
        "cam.data.dof.focus_distance = 3.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=20)\n"
        "cam.data.dof.focus_distance = 6.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=40)\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_focus_transition_not_continuous" in rule_ids
    issues = {i.rule_id: i for i in report.issues}
    assert issues["dof_focus_transition_not_continuous"].failure_class == (
        "success_soft"
    )


def test_dof_success_spec_hard_blocks_when_focus_transition_too_concentrated():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.8\n"
        "cam.data.dof.focus_distance = 1.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=1)\n"
        "cam.data.dof.focus_distance = 3.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=20)\n"
        "cam.data.dof.focus_distance = 6.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=40)\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )

    issues = {i.rule_id: i for i in report.issues}
    assert issues["dof_focus_transition_not_continuous"].failure_class == (
        "success_hard"
    )


def test_dof_blocks_when_focus_distance_not_keyframed():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.2\n"
        "cam.data.dof.focus_distance = 3.0\n"  # set once, no keyframe
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_missing_focus_keyframe" in rule_ids


def test_dof_blocks_when_use_dof_disabled():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.aperture_fstop = 1.2\n"
        "cam.data.dof.focus_distance = 1.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=1)\n"
        "cam.data.dof.focus_distance = 6.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=120)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_use_dof_not_enabled" in rule_ids


def test_dof_blocks_when_depth_anchors_missing():
    code = (
        "import bpy\n"
        "obj = bpy.data.objects.new('random_thing', None)\n"  # no anchors
        "cam = bpy.data.objects.new('Camera', None)\n"
        "cam.data.dof.use_dof = True\n"
        "cam.data.dof.aperture_fstop = 1.2\n"
        "cam.data.dof.focus_distance = 1.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=1)\n"
        "cam.data.dof.focus_distance = 6.5\n"
        "cam.data.keyframe_insert('dof.focus_distance', frame=120)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_missing_depth_anchors" in rule_ids


def test_dof_warns_when_aperture_too_wide():
    code = _DOF_GOOD_SCENE.replace(
        "cam.data.dof.aperture_fstop = 1.2",
        "cam.data.dof.aperture_fstop = 2.9",
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_aperture_too_wide" in rule_ids


@pytest.mark.parametrize("fstop", [1.4, 2.8])
def test_dof_success_spec_accepts_aperture_boundaries(fstop: float):
    code = _DOF_GOOD_SCENE.replace(
        "cam.data.dof.aperture_fstop = 1.2",
        f"cam.data.dof.aperture_fstop = {fstop}",
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_aperture_out_of_success_range" not in rule_ids


@pytest.mark.parametrize("fstop", [1.2, 2.9])
def test_dof_success_spec_blocks_aperture_outside_range(fstop: float):
    code = _DOF_GOOD_SCENE.replace(
        "cam.data.dof.aperture_fstop = 1.2",
        f"cam.data.dof.aperture_fstop = {fstop}",
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    issues = {i.rule_id: i for i in report.issues}
    assert issues["dof_aperture_out_of_success_range"].severity == "block"
    assert (
        issues["dof_aperture_out_of_success_range"].failure_class
        == "success_hard"
    )


def test_dof_legacy_aperture_warning_is_not_success_hard():
    code = _DOF_GOOD_SCENE.replace(
        "cam.data.dof.aperture_fstop = 1.2",
        "cam.data.dof.aperture_fstop = 3.5",
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    issues = {i.rule_id: i for i in report.issues}

    assert issues["dof_aperture_too_wide"].failure_class == "aesthetic_warn"
    assert report.success_hard_count == 0


def test_dof_accepts_aperture_at_readability_boundary():
    code = _DOF_GOOD_SCENE.replace(
        "cam.data.dof.aperture_fstop = 1.2",
        "cam.data.dof.aperture_fstop = 2.8",
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_aperture_too_wide" not in rule_ids


def test_dof_blocks_when_subjects_animated():
    code = _DOF_GOOD_SCENE + (
        "fg.location = (0, -1, 0)\n"
        "fg.keyframe_insert('location', frame=1)\n"
        "fg.location = (0, -2, 0)\n"
        "fg.keyframe_insert('location', frame=60)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "dof_subjects_animated" in rule_ids


def test_success_spec_blocks_mirrored_text_anchor():
    code = _DOF_GOOD_SCENE + (
        "label = bpy.data.objects.new('depth_label_near', None)\n"
        "label.scale = (-1, 1, 1)\n"
        "hud = bpy.data.objects.new('lens_readout_hud', None)\n"
        "hud.rotation_euler = (1.2, 0, 0)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "success_text_mirrored" in rule_ids


def test_success_spec_accepts_camera_facing_text_anchor():
    code = _DOF_GOOD_SCENE + (
        "label = bpy.data.objects.new('depth_label_near', None)\n"
        "label.scale = (1, 1, 1)\n"
        "label.rotation_euler = (1.2, 0, 0)\n"
        "main_camera = bpy.data.objects.new('main_camera', None)\n"
        "hud = bpy.data.objects.new('lens_readout_hud', None)\n"
        "hud.scale = (1, 1, 1)\n"
        "hud.parent = main_camera\n"
        "hud.location = (0, 0, -2.0)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "success_text_anchor_missing" not in rule_ids
    assert "success_text_mirrored" not in rule_ids
    assert "success_text_faces_camera_unproven" not in rule_ids
    assert "success_hud_overlay_not_camera_parented" not in rule_ids


def test_success_spec_blocks_hud_overlay_without_camera_parent():
    code = _DOF_GOOD_SCENE + (
        "label = bpy.data.objects.new('depth_label_near', None)\n"
        "label.scale = (1, 1, 1)\n"
        "label.rotation_euler = (1.2, 0, 0)\n"
        "hud = bpy.data.objects.new('lens_readout_hud', None)\n"
        "hud.scale = (1, 1, 1)\n"
        "hud.rotation_euler = (1.2, 0, 0)\n"
    )
    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    issues = {i.rule_id: i for i in report.issues}

    assert issues["success_hud_overlay_not_camera_parented"].failure_class == (
        "success_hard"
    )


def test_success_spec_accepts_helper_created_text_and_hud_anchors():
    code = _DOF_GOOD_SCENE + (
        "main_camera = bpy.data.objects.new('main_camera', None)\n"
        "def make_depth_label(name, body):\n"
        "    bpy.ops.object.text_add(location=(0, 0, 0))\n"
        "    obj = bpy.context.object\n"
        "    obj.name = name\n"
        "    obj.data.body = body\n"
        "    obj.rotation_euler = (1.2, 0, 0)\n"
        "    return obj\n"
        "def make_camera_hud(name, body):\n"
        "    bpy.ops.object.text_add(location=(0, 0, 0))\n"
        "    obj = bpy.context.object\n"
        "    obj.name = name\n"
        "    obj.data.body = body\n"
        "    obj.parent = main_camera\n"
        "    obj.location = (-0.3, 0.2, -1.0)\n"
        "    obj.scale = (1, 1, 1)\n"
        "    return obj\n"
        "depth_label_near = make_depth_label('depth_label_near', 'd_near')\n"
        "depth_label_mid = make_depth_label('depth_label_mid', 'd_mid')\n"
        "depth_label_far = make_depth_label('depth_label_far', 'd_far')\n"
        "lens_readout_hud = make_camera_hud('lens_readout_hud', 'f/1.8')\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    rule_ids = {i.rule_id for i in report.issues}

    assert "success_text_anchor_missing" not in rule_ids
    assert "success_hud_overlay_not_camera_parented" not in rule_ids


def test_dof_accepts_symbolic_focus_track_from_shot_ranges():
    code = (
        "import bpy\n"
        "fg = bpy.data.objects.new('foreground_subject', None)\n"
        "mg = bpy.data.objects.new('middleground_subject', None)\n"
        "bg = bpy.data.objects.new('background_subject', None)\n"
        "cam_data = bpy.data.cameras.new('main_camera')\n"
        "cam = bpy.data.objects.new('main_camera', cam_data)\n"
        "cam_data.dof.use_dof = True\n"
        "cam_data.dof.aperture_fstop = 1.8\n"
        "D_NEAR = distance_to((0, -1, 0))\n"
        "D_MID = distance_to((0, -3, 0))\n"
        "D_FAR = distance_to((0, -6, 0))\n"
        "shot_starts = []\n"
        "(s1_start, s1_end) = shot_starts[0]\n"
        "(s2_start, s2_end) = shot_starts[1]\n"
        "(s3_start, s3_end) = shot_starts[2]\n"
        "(s4_start, s4_end) = shot_starts[3]\n"
        "dof = cam_data.dof\n"
        "dof.focus_distance = D_NEAR\n"
        "dof.keyframe_insert('focus_distance', frame=s1_start)\n"
        "dof.focus_distance = D_MID\n"
        "dof.keyframe_insert('focus_distance', frame=s2_end)\n"
        "dof.focus_distance = D_FAR\n"
        "dof.keyframe_insert('focus_distance', frame=s4_end)\n"
    )

    report = run_concept_metrics(
        concept_id="depth_of_field_focus_pull",
        scene_code=code,
        storyboard=_simple_storyboard("depth_of_field_focus_pull"),
        success_spec=_dof_success_spec(),
    )
    rule_ids = {i.rule_id for i in report.issues}

    assert "dof_focus_transition_not_continuous" not in rule_ids


# ---------------- shadow_softness_radius ----------------


_SHADOW_GOOD_SCENE = (
    "import bpy\n"
    "pillar = bpy.data.objects.new('subject_pillar', None)\n"
    "ground = bpy.data.objects.new('ground_plane', None)\n"
    "light = bpy.data.objects.new('area_light', None)\n"
    "light.data.shadow_soft_size = 0.05\n"
    "light.data.keyframe_insert('shadow_soft_size', frame=1)\n"
    "light.data.shadow_soft_size = 0.9\n"
    "light.data.keyframe_insert('shadow_soft_size', frame=96)\n"
)


def test_shadow_softness_passes_when_softness_ramps():
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=_SHADOW_GOOD_SCENE,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_size_keyframe" not in rule_ids
    assert "shadow_softness_ramp_flat" not in rule_ids
    assert "shadow_softness_missing_subject" not in rule_ids
    assert "shadow_softness_missing_ground" not in rule_ids


def test_shadow_softness_accepts_light_object_alias_receiver():
    code = (
        "import bpy\n"
        "pillar = bpy.data.objects.new('subject_pillar', None)\n"
        "ground = bpy.data.objects.new('ground_plane', None)\n"
        "obj = bpy.data.objects.new('area_light', None)\n"
        "obj.data.size = 0.05\n"
        "obj.data.keyframe_insert('size', frame=1)\n"
        "obj.data.size = 0.9\n"
        "obj.data.keyframe_insert('size', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_size_keyframe" not in rule_ids
    assert "shadow_softness_ramp_flat" not in rule_ids


def test_shadow_softness_blocks_when_no_size_keyframe():
    code = (
        "import bpy\n"
        "pillar = bpy.data.objects.new('subject_pillar', None)\n"
        "ground = bpy.data.objects.new('ground_plane', None)\n"
        "light = bpy.data.objects.new('area_light', None)\n"
        "light.data.shadow_soft_size = 0.5\n"  # set once, no keyframe
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_size_keyframe" in rule_ids


def test_shadow_softness_blocks_when_ramp_flat():
    code = (
        "import bpy\n"
        "pillar = bpy.data.objects.new('subject_pillar', None)\n"
        "ground = bpy.data.objects.new('ground_plane', None)\n"
        "light = bpy.data.objects.new('area_light', None)\n"
        "light.data.shadow_soft_size = 0.10\n"
        "light.data.keyframe_insert('shadow_soft_size', frame=1)\n"
        "light.data.shadow_soft_size = 0.12\n"  # delta < 0.5 threshold
        "light.data.keyframe_insert('shadow_soft_size', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_ramp_flat" in rule_ids


def test_shadow_softness_accepts_ramp_at_boundary():
    code = _SHADOW_GOOD_SCENE.replace(
        "light.data.shadow_soft_size = 0.9",
        "light.data.shadow_soft_size = 0.55",
    )

    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_ramp_flat" not in rule_ids


def test_shadow_softness_blocks_ramp_just_below_boundary():
    code = _SHADOW_GOOD_SCENE.replace(
        "light.data.shadow_soft_size = 0.9",
        "light.data.shadow_soft_size = 0.54",
    )

    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_ramp_flat" in rule_ids


def test_shadow_softness_ignores_label_text_size_keyframes():
    code = (
        "import bpy\n"
        "pillar = bpy.data.objects.new('subject_pillar', None)\n"
        "ground = bpy.data.objects.new('ground_plane', None)\n"
        "softness_label = bpy.data.objects.new('softness_label', None)\n"
        "softness_label.data.size = 0.2\n"
        "softness_label.data.keyframe_insert('size', frame=1)\n"
        "softness_label.data.size = 1.2\n"
        "softness_label.data.keyframe_insert('size', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_size_keyframe" in rule_ids
    assert "shadow_softness_ramp_flat" not in rule_ids


def test_shadow_softness_blocks_when_subject_animated():
    code = _SHADOW_GOOD_SCENE + (
        "pillar.location = (0, 0, 0)\n"
        "pillar.keyframe_insert('location', frame=1)\n"
        "pillar.location = (0, 1, 0)\n"
        "pillar.keyframe_insert('location', frame=60)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_subject_animated" in rule_ids


def test_shadow_softness_allows_static_subject_hold_keyframes():
    code = _SHADOW_GOOD_SCENE + (
        "pillar.location = (0, 0, 0)\n"
        "pillar.keyframe_insert('location', frame=1)\n"
        "pillar.location = (0, 0, 0)\n"
        "pillar.keyframe_insert('location', frame=60)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_subject_animated" not in rule_ids


def test_shadow_softness_blocks_when_ground_missing():
    code = (
        "import bpy\n"
        "pillar = bpy.data.objects.new('subject_pillar', None)\n"
        "light = bpy.data.objects.new('area_light', None)\n"
        "light.data.shadow_soft_size = 0.05\n"
        "light.data.keyframe_insert('shadow_soft_size', frame=1)\n"
        "light.data.shadow_soft_size = 0.9\n"
        "light.data.keyframe_insert('shadow_soft_size', frame=96)\n"
    )
    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=_simple_storyboard("shadow_softness_radius"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_ground" in rule_ids


def test_shadow_softness_accepts_compiled_embedded_light_size_keyframes():
    from cg_tutor.scene_compiler import compile_storyboard_to_bpy
    from cg_tutor.schemas import Storyboard

    storyboard = Storyboard.model_validate({
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
                    "name": "subject_pillar",
                    "type": "primitive",
                    "primitive": "cylinder",
                    "location": [0, 0, 0.8],
                    "properties": {"radius": 0.35, "depth": 1.6},
                    "keyframes": [],
                },
                {
                    "name": "ground_plane",
                    "type": "primitive",
                    "primitive": "plane",
                    "location": [0, 0, 0],
                    "properties": {"size": 5.0},
                    "keyframes": [],
                },
                {
                    "name": "area_light",
                    "type": "primitive",
                    "primitive": "sphere",
                    "location": [-1, -2, 4],
                    "properties": {"radius": 0.08},
                    "keyframes": [],
                },
            ],
        }],
    })
    code = compile_storyboard_to_bpy(storyboard, render_engine="CYCLES")

    report = run_concept_metrics(
        concept_id="shadow_softness_radius",
        scene_code=code,
        storyboard=storyboard,
    )

    rule_ids = {i.rule_id for i in report.issues}
    assert "shadow_softness_missing_size_keyframe" not in rule_ids
    assert "shadow_softness_ramp_flat" not in rule_ids
    assert "shadow_softness_subject_animated" not in rule_ids
    assert report.metrics["shadow_softness_radius"]["softness_from_embedded_storyboard"]


# ---------------- registry isolation ----------------


def test_unknown_concept_emits_no_concept_specific_findings():
    report = run_concept_metrics(
        concept_id="bezier_curve",
        scene_code="import bpy\n",
        storyboard=_simple_storyboard("bezier_curve"),
    )
    rule_ids = {i.rule_id for i in report.issues}
    assert not any(rid.startswith("particle_trail_") for rid in rule_ids)
    assert not any(rid.startswith("dof_") for rid in rule_ids)
    assert not any(rid.startswith("shadow_softness_") for rid in rule_ids)
    assert not any(rid.startswith("dolly_zoom_") for rid in rule_ids)
