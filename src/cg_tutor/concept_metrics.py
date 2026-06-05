"""Concept-level executable checks for rendered teaching scenes.

These checks sit between static AST validation and the expensive vision
critic. They intentionally stay small: each plugin catches high-confidence
failure modes that are hard to express with generic schema checks.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

from cg_tutor.preview import PreviewReport
from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.auto_success_spec import AutoSuccessSpec
from cg_tutor.success_spec import SuccessSpec


Severity = Literal["block", "warn"]
FailureClass = Literal[
    "structural_fatal",
    "success_hard",
    "success_soft",
    "aesthetic_warn",
]

ConceptMetricRunner = Callable[
    [str, Storyboard, SuccessSpec | None], "tuple[list[ConceptMetricIssue], dict]"
]


@dataclass(frozen=True)
class ConceptMetricIssue:
    severity: Severity
    rule_id: str
    message: str
    suggested_fix: str
    failure_class: FailureClass | None = None

    def __post_init__(self) -> None:
        if self.failure_class is None:
            object.__setattr__(
                self,
                "failure_class",
                _default_failure_class(self.rule_id, self.severity),
            )


@dataclass
class ConceptMetricReport:
    concept_id: str
    issues: list[ConceptMetricIssue] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "block")

    @property
    def warn_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warn")

    @property
    def structural_fatal_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.failure_class == "structural_fatal"
        )

    @property
    def success_hard_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.failure_class == "success_hard"
        )

    @property
    def success_soft_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.failure_class == "success_soft"
        )

    @property
    def aesthetic_warn_count(self) -> int:
        return sum(
            1 for issue in self.issues
            if issue.failure_class == "aesthetic_warn"
        )

    @property
    def ok(self) -> bool:
        return self.structural_fatal_count == 0 and self.success_hard_count == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "concept_id": self.concept_id,
            "block": self.block_count,
            "warn": self.warn_count,
            "structural_fatal": self.structural_fatal_count,
            "success_hard": self.success_hard_count,
            "success_soft": self.success_soft_count,
            "aesthetic_warn": self.aesthetic_warn_count,
            "metrics": self.metrics,
            "issues": [asdict(issue) for issue in self.issues],
        }


_SUCCESS_HARD_RULES = {
    "success_text_anchor_missing",
    "success_text_mirrored",
    "success_hud_overlay_not_camera_parented",
    "dof_aperture_out_of_success_range",
    "dof_focus_transition_not_continuous",
}

_SUCCESS_SOFT_RULES = {
    "success_text_faces_camera_unproven",
    "dolly_zoom_lens_not_keyframed",
    "concept_metric_visible_motion_missing",
}

_AESTHETIC_WARN_RULES = {
    "dof_aperture_too_wide",
}


def _default_failure_class(rule_id: str, severity: Severity) -> FailureClass:
    if rule_id in _SUCCESS_HARD_RULES:
        return "success_hard"
    if rule_id in _SUCCESS_SOFT_RULES:
        return "success_soft"
    if rule_id in _AESTHETIC_WARN_RULES:
        return "aesthetic_warn"
    if severity == "warn":
        return "aesthetic_warn"
    return "success_soft"


@dataclass(frozen=True)
class _ConceptMetricPlugin:
    concept_id: str
    runner: ConceptMetricRunner
    metrics_key: str
    match_substring: bool = False

    def matches(self, cid: str) -> bool:
        if self.match_substring:
            return self.concept_id == cid or cid.startswith(f"{self.concept_id}_")
        return self.concept_id == cid


_CONCEPT_METRIC_REGISTRY: dict[str, _ConceptMetricPlugin] = {}

_MIN_TRAIL_REVEAL_DELTA = 0.5
_MIN_SHADOW_SOFTNESS_DELTA = 0.5
_DOF_MAX_READABLE_FSTOP = 2.8


def register_concept_metric(
    concept_id: str,
    *,
    metrics_key: str | None = None,
    match_substring: bool = False,
) -> Callable[[ConceptMetricRunner], ConceptMetricRunner]:
    """Register a concept-specific metric runner.

    The runner takes ``(scene_code, storyboard, success_spec)`` and returns
    ``(issues, metrics_dict)``. Plugins fire only when ``matches(concept_id)``
    returns True; ``match_substring=True`` lets the same plugin handle
    concept ids that contain the registration key as a substring
    (used historically for the dolly_zoom variant ids).
    """
    def decorator(fn: ConceptMetricRunner) -> ConceptMetricRunner:
        if concept_id in _CONCEPT_METRIC_REGISTRY:
            raise ValueError(f"duplicate concept metric plugin: {concept_id}")
        _CONCEPT_METRIC_REGISTRY[concept_id] = _ConceptMetricPlugin(
            concept_id=concept_id,
            runner=fn,
            metrics_key=metrics_key or concept_id,
            match_substring=match_substring,
        )
        return fn
    return decorator


def run_concept_metrics(
    *,
    concept_id: str,
    scene_code: str,
    storyboard: Storyboard,
    scene_profile: SceneProfile | None = None,
    preview_report: PreviewReport | None = None,
    success_spec: SuccessSpec | None = None,
    auto_success_spec: AutoSuccessSpec | None = None,
) -> ConceptMetricReport:
    report = ConceptMetricReport(concept_id=concept_id)
    report.issues.extend(
        _generic_dynamic_metrics(
            storyboard=storyboard,
            scene_profile=scene_profile,
            preview_report=preview_report,
        )
    )
    coverage_issues, coverage_metrics = _storyboard_keyframe_coverage_metrics(
        scene_code,
        storyboard,
    )
    report.issues.extend(coverage_issues)
    if coverage_metrics:
        report.metrics["storyboard_keyframe_coverage"] = coverage_metrics

    text_issues, text_metrics = _success_spec_text_faces_camera_metrics(
        scene_code,
        success_spec,
    )
    report.issues.extend(text_issues)
    if text_metrics:
        report.metrics["success_spec_text_faces_camera"] = text_metrics

    auto_issues, auto_metrics = _auto_success_spec_metrics(
        scene_code,
        auto_success_spec,
    )
    report.issues.extend(auto_issues)
    if auto_metrics:
        report.metrics["auto_success_spec"] = auto_metrics

    for plugin in _CONCEPT_METRIC_REGISTRY.values():
        if not plugin.matches(concept_id):
            continue
        issues, metrics = plugin.runner(scene_code, storyboard, success_spec)
        report.issues.extend(issues)
        if metrics:
            report.metrics[plugin.metrics_key] = metrics

    return report


def _generic_dynamic_metrics(
    *,
    storyboard: Storyboard,
    scene_profile: SceneProfile | None,
    preview_report: PreviewReport | None,
) -> list[ConceptMetricIssue]:
    if scene_profile is None or preview_report is None:
        return []
    if scene_profile.base_profile not in {
        "transformation_demo",
        "vector_teaching",
        "curve_construction",
    }:
        return []
    issues: list[ConceptMetricIssue] = []
    for issue in preview_report.issues:
        if issue.rule_id == "insufficient_visible_motion":
            issues.append(ConceptMetricIssue(
                severity="block",
                rule_id="concept_metric_visible_motion_missing",
                message=issue.message,
                suggested_fix=(
                    "Increase the actual main-frame animation amplitude for "
                    f"{storyboard.concept_id}: animate concept anchors, camera, "
                    "lens, shape keys, modifier levels, curve reveal, or "
                    "material state so sampled frames visibly differ."
                ),
            ))
    return issues


def _success_spec_text_faces_camera_metrics(
    scene_code: str,
    success_spec: SuccessSpec | None,
) -> tuple[list[ConceptMetricIssue], dict]:
    if success_spec is None:
        return [], {}
    anchors = success_spec.text_faces_camera_anchors()
    if not anchors:
        return [], {}
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return [], {"expected": sorted(anchors), "checked": [], "missing": sorted(anchors)}

    assignments = _assignment_targets(tree)
    object_names = _object_name_assignments(tree)
    helper_calls = _helper_object_calls(tree)
    helper_evidence = _helper_object_evidence(tree)
    placements = success_spec.text_anchor_placements()
    checked: set[str] = set()
    mirrored: set[str] = set()
    missing: set[str] = set(anchors)
    unconstrained: set[str] = set()
    hud_not_parented: set[str] = set()
    for var_name, object_name in object_names.items():
        if object_name not in anchors:
            continue
        checked.add(object_name)
        missing.discard(object_name)
        scale_values = assignments.get((var_name, "scale"), [])
        if any(_has_negative_component(value) for value in scale_values):
            mirrored.add(object_name)
        if placements.get(object_name, "in_scene") == "hud_overlay":
            if not _text_anchor_has_camera_parent_evidence(
                tree, assignments, var_name, object_name,
                helper_calls=helper_calls,
                helper_evidence=helper_evidence,
            ):
                hud_not_parented.add(object_name)
        elif not _text_anchor_has_camera_facing_evidence(scene_code, var_name, object_name):
            unconstrained.add(object_name)

    issues: list[ConceptMetricIssue] = []
    if missing:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="success_text_anchor_missing",
            message=(
                "Success Spec requires text anchor(s) "
                f"{', '.join(sorted(missing))}, but scene.py does not create "
                "objects with those exact names."
            ),
            suggested_fix=(
                "Create the required text objects with exact names from "
                "success_spec.hard_constraints[text_faces_camera].anchors."
            ),
            failure_class="success_hard",
        ))
    if mirrored:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="success_text_mirrored",
            message=(
                "Success Spec text anchor(s) appear to use negative scale, "
                f"which mirrors text and makes it unreadable: "
                f"{', '.join(sorted(mirrored))}."
            ),
            suggested_fix=(
                "Remove negative scale on required text anchors. Face the "
                "camera with rotation/Track To constraints instead of "
                "flipping scale."
            ),
            failure_class="success_hard",
        ))
    if unconstrained:
        issues.append(ConceptMetricIssue(
            severity="warn",
            rule_id="success_text_faces_camera_unproven",
            message=(
                "Success Spec text anchor(s) do not show explicit camera-facing "
                f"evidence: {', '.join(sorted(unconstrained))}."
            ),
            suggested_fix=(
                "Add Track To constraints toward the render camera, or set a "
                "clear rotation_euler for text anchors so they face the camera "
                "without mirrored scale."
            ),
            failure_class="success_soft",
        ))
    if hud_not_parented:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="success_hud_overlay_not_camera_parented",
            message=(
                "Success Spec HUD text anchor(s) are not implemented as "
                "camera-parented pseudo-HUD objects with fixed local "
                f"placement: {', '.join(sorted(hud_not_parented))}."
            ),
            suggested_fix=(
                "Parent each hud_overlay text anchor to the render camera, "
                "set a fixed local location in front of the camera, avoid "
                "negative scale, and do not create duplicate replacement labels."
            ),
            failure_class="success_hard",
        ))
    return issues, {
        "expected": sorted(anchors),
        "checked": sorted(checked),
        "missing": sorted(missing),
        "mirrored": sorted(mirrored),
        "camera_facing_unproven": sorted(unconstrained),
        "hud_overlay_expected": sorted(
            name for name, placement in placements.items()
            if placement == "hud_overlay"
        ),
        "hud_overlay_not_camera_parented": sorted(hud_not_parented),
    }


def _auto_success_spec_metrics(
    scene_code: str,
    auto_success_spec: AutoSuccessSpec | None,
) -> tuple[list[ConceptMetricIssue], dict]:
    """Run low-risk checks for generated success rules.

    Generated specs are intentionally soft at this stage.  The AST checks
    below provide retry evidence, but they must not turn a first iteration
    into a hard failure without repeated critic confirmation.
    """
    if auto_success_spec is None or not auto_success_spec.rules:
        return [], {}
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return [], {
            "rules": len(auto_success_spec.rules),
            "checked": False,
            "reason": "scene_code_syntax_error",
        }

    created_names = set(_object_aliases(tree).values())
    code_lower = scene_code.lower()
    anchor_status: dict[str, dict] = {}
    for rule in auto_success_spec.rules:
        for anchor in rule.anchors:
            matches = _auto_anchor_matched_names(anchor, created_names)
            anchor_status[anchor] = {
                "created": bool(matches),
                "matched_names": matches,
            }
    issues: list[ConceptMetricIssue] = []
    metrics = {
        "rules": len(auto_success_spec.rules),
        "checked": True,
        "created_names_sample": sorted(created_names)[:80],
        "anchor_status": anchor_status,
    }

    for rule in auto_success_spec.rules:
        if rule.failure_class == "diagnostic":
            continue

        if rule.kind == "object_visible":
            missing = [
                anchor for anchor in rule.anchors
                if not anchor_status.get(anchor, {}).get("created")
            ]
            if missing:
                issues.append(ConceptMetricIssue(
                    severity="warn",
                    rule_id="auto_success_object_visible_missing",
                    message=(
                        "Generated success spec expects visible object "
                        "anchor(s) that are not clearly created in scene.py: "
                        + ", ".join(missing)
                    ),
                    suggested_fix=(
                        "Instantiate the generated anchor(s) with exact or "
                        "obvious matching names. Keep this as a small targeted "
                        "change; do not add unrelated labels or decorations."
                    ),
                    failure_class="success_soft",
                ))

        elif rule.kind == "helper_hidden":
            visible_helpers = [
                anchor for anchor in rule.anchors
                if anchor_status.get(anchor, {}).get("created")
                and _auto_helper_lacks_hide_render_evidence(anchor, code_lower)
            ]
            if visible_helpers:
                issues.append(ConceptMetricIssue(
                    severity="warn",
                    rule_id="auto_success_helper_may_be_visible",
                    message=(
                        "Generated success spec marks helper/path/guide "
                        "anchor(s) as possible final-render clutter: "
                        + ", ".join(visible_helpers)
                    ),
                    suggested_fix=(
                        "Hide helper/path/guide anchors from final render "
                        "unless they are the main teaching object."
                    ),
                    failure_class="aesthetic_warn",
                ))

    return issues, metrics


def _auto_anchor_matches_created_name(anchor: str, created_names: set[str]) -> bool:
    return bool(_auto_anchor_matched_names(anchor, created_names))


def _auto_anchor_matched_names(anchor: str, created_names: set[str]) -> list[str]:
    anchor_norm = _auto_norm(anchor)
    if not anchor_norm:
        return []
    matches: list[str] = []
    for name in created_names:
        name_norm = _auto_norm(name)
        if not name_norm:
            continue
        if anchor_norm == name_norm:
            matches.append(name)
            continue
        # LOD/readout tokens are often embedded in helper variable names,
        # e.g. critic target LOD1 and object name text_lod1.
        if len(anchor_norm) >= 3 and anchor_norm in name_norm:
            matches.append(name)
    return sorted(set(matches))


def _auto_helper_lacks_hide_render_evidence(anchor: str, code_lower: str) -> bool:
    anchor_lower = anchor.lower()
    if "hide_render" not in code_lower:
        return True
    # Conservative: accept either direct object evidence or broad hide_render
    # evidence in the same generated scene.  This avoids false hardening on
    # helper rules, which are only aesthetic warnings in this rollout.
    return anchor_lower in code_lower and f"{anchor_lower}.hide_render" not in code_lower


def _auto_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@register_concept_metric("dolly_zoom", match_substring=True)
def _dolly_zoom_metrics(
    scene_code: str,
    storyboard: Storyboard,
    success_spec: SuccessSpec | None = None,
) -> tuple[list[ConceptMetricIssue], dict]:
    low = scene_code.lower()
    camera_y_delta = _storyboard_camera_y_delta(storyboard)
    fov_delta = _storyboard_fov_delta(storyboard)
    lens_keyframed = _has_lens_keyframe(scene_code) and fov_delta >= 20.0
    has_depth_anchors = all(
        name in low
        for name in (
            "hero_lighthouse",
            "camera_icon",
            "marker_post_1",
            "marker_post_2",
            "marker_post_3",
        )
    )
    animated_markers = sorted(
        n for n in _objects_with_progressing_keyframes(
            scene_code,
            set(),
            {"location", "scale", "rotation_euler"},
            min_delta=0.05,
            name_predicate=lambda n: bool(re.match(r"marker_post_\d+", n)),
        )
    )
    animated_hero = sorted(
        n for n in _objects_with_progressing_keyframes(
            scene_code,
            set(),
            {"location", "scale", "rotation_euler"},
            min_delta=0.05,
            name_predicate=lambda n: n.startswith("hero_"),
        )
    )
    metrics = {
        "lens_keyframed": lens_keyframed,
        "storyboard_camera_y_delta": round(camera_y_delta, 4),
        "storyboard_fov_delta": round(fov_delta, 4),
        "has_depth_anchors": has_depth_anchors,
        "animated_marker_posts": animated_markers,
        "animated_hero_objects": animated_hero,
    }
    issues: list[ConceptMetricIssue] = []
    if not lens_keyframed:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dolly_zoom_missing_lens_animation",
            message=(
                "Dolly zoom scene does not keyframe camera.data.lens. "
                "The Vertigo effect requires focal length animation, not "
                "only object or icon motion."
            ),
            suggested_fix=(
                "Convert FOV keys to camera.data.lens values and insert "
                "`camera.data.keyframe_insert('lens', frame=...)` during "
                "the dolly-zoom segment."
            ),
        ))
    if camera_y_delta < 2.0 or fov_delta < 20.0:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dolly_zoom_weak_camera_fov_coupling",
            message=(
                "Storyboard/camera plan has too little camera-depth or FOV "
                f"change for a readable dolly zoom (y_delta={camera_y_delta:.2f}, "
                f"fov_delta={fov_delta:.2f})."
            ),
            suggested_fix=(
                "Use a strong coupled move: render camera moves backward "
                "along the depth axis while lens/FOV narrows enough that the "
                "foreground subject stays nearly constant and background "
                "markers visibly breathe."
            ),
        ))
    if not has_depth_anchors:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dolly_zoom_missing_depth_anchors",
            message=(
                "Dolly zoom scene code does not contain the required depth "
                "anchors hero_lighthouse, camera_icon, and marker_post_1..3."
            ),
            suggested_fix=(
                "Instantiate one centered hero_lighthouse, a camera_icon in "
                "front of it, and at least three stationary marker_post_N "
                "objects behind it on the depth axis."
            ),
        ))
    if animated_markers:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dolly_zoom_marker_posts_animated",
            message=(
                "Background marker posts must stay planted in world space "
                "so they read as fixed depth anchors. Found keyframes on "
                f"{', '.join(animated_markers)}."
            ),
            suggested_fix=(
                "Remove all keyframe_insert calls on marker_post_* objects. "
                "Only the render camera and (optionally) the camera_icon "
                "should move along the depth axis."
            ),
        ))
    if animated_hero:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dolly_zoom_hero_subject_animated",
            message=(
                "The hero subject must remain stationary so dolly-zoom "
                "coupling reads as constant screen height. Found keyframes "
                f"on {', '.join(animated_hero)}."
            ),
            suggested_fix=(
                "Remove keyframe_insert calls on hero_* objects. The "
                "perceptual cue of dolly zoom comes from the hero staying "
                "fixed while background markers breathe."
            ),
        ))
    return issues, metrics


@register_concept_metric("particle_trail_curve")
def _particle_trail_curve_metrics(
    scene_code: str,
    storyboard: Storyboard,
    success_spec: SuccessSpec | None = None,
) -> tuple[list[ConceptMetricIssue], dict]:
    low = scene_code.lower()
    bevel_keyframed = _any_keyframe_on_data_path(
        scene_code,
        {"bevel_factor_end", "data.bevel_factor_end"},
    )
    emitter_motion = _objects_with_progressing_keyframes(
        scene_code,
        {"glow_emitter", "particle_emitter", "trail_head"},
        {"location", "scale", "rotation_euler"},
        min_delta=0.05,
    )
    emitter_animated = bool(emitter_motion)
    has_curve = "trail_curve" in low or "trajectory_curve" in low
    has_emitter = (
        "glow_emitter" in low or "particle_emitter" in low or "trail_head" in low
    )
    bevel_progresses = _data_path_value_progresses(
        scene_code,
        {"bevel_factor_end", "data.bevel_factor_end"},
        min_delta=_MIN_TRAIL_REVEAL_DELTA,
    )
    metrics = {
        "bevel_keyframed": bevel_keyframed,
        "bevel_progresses": bevel_progresses,
        "has_trail_curve_object": has_curve,
        "has_emitter_object": has_emitter,
        "emitter_animated": emitter_animated,
    }
    issues: list[ConceptMetricIssue] = []
    if not bevel_keyframed:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="particle_trail_missing_bevel_keyframe",
            message=(
                "Particle trail scene does not keyframe "
                "data.bevel_factor_end on any curve. The growing-trail "
                "effect requires the curve reveal to be animated."
            ),
            suggested_fix=(
                "On the trail curve object call "
                "`trail.data.bevel_factor_end = 0.0; "
                "trail.data.keyframe_insert('bevel_factor_end', frame=1)` "
                "then again with value 1.0 at the final frame."
            ),
        ))
    elif not bevel_progresses:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="particle_trail_bevel_ramp_flat",
            message=(
                "bevel_factor_end keyframes exist but the value never "
                f"progresses by at least {_MIN_TRAIL_REVEAL_DELTA:.1f}, so "
                "the trail will not appear to grow."
            ),
            suggested_fix=(
                "Use a strong ramp: 0.0 at frame 1 and 1.0 at the final "
                "frame, with the first non-zero value before duration*0.2 "
                "so the trail starts revealing early."
            ),
        ))
    if not has_curve:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="particle_trail_missing_trail_curve_object",
            message=(
                "Scene code does not create a `trail_curve` (or "
                "`trajectory_curve`) object. The trail itself is the "
                "primary teaching anchor."
            ),
            suggested_fix=(
                "Create a bezier or poly curve named 'trail_curve' with "
                "`bevel_depth > 0` and `bevel_factor_end` keyframed from "
                "0 to 1 across the shot."
            ),
        ))
    if not has_emitter:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="particle_trail_missing_emitter",
            message=(
                "Scene code does not create a glow_emitter / "
                "particle_emitter / trail_head object. The viewer needs "
                "the leading point to anchor the reveal."
            ),
            suggested_fix=(
                "Add a small emissive sphere named 'glow_emitter' and "
                "keyframe its `location` to follow the curve waypoints."
            ),
        ))
    if has_emitter and not emitter_animated:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="particle_trail_emitter_static",
            message=(
                "glow_emitter / particle_emitter exists but is never "
                "keyframed. The leading point must travel along the "
                "trajectory in sync with the curve reveal."
            ),
            suggested_fix=(
                "Keyframe `glow_emitter.location` at each waypoint "
                "matching the curve's bevel_factor_end progression."
            ),
        ))
    return issues, metrics


@register_concept_metric("depth_of_field_focus_pull")
def _depth_of_field_focus_pull_metrics(
    scene_code: str,
    storyboard: Storyboard,
    success_spec: SuccessSpec | None = None,
) -> tuple[list[ConceptMetricIssue], dict]:
    low = scene_code.lower()
    focus_keyframed = _any_keyframe_on_data_path(
        scene_code,
        {"dof.focus_distance", "focus_distance"},
    )
    focus_progresses = _data_path_value_progresses(
        scene_code,
        {"dof.focus_distance", "focus_distance"},
        min_delta=2.0,
    ) or _keyframed_data_path_uses_distinct_values(
        scene_code,
        {"dof.focus_distance", "focus_distance"},
        min_numeric_delta=2.0,
    )
    focus_track = _data_path_keyframe_track(
        scene_code,
        {"dof.focus_distance", "focus_distance"},
        total_frames=storyboard.total_frames,
    )
    continuity = _focus_track_continuity(focus_track, total_frames=storyboard.total_frames)
    use_dof_enabled = _dof_use_dof_enabled(scene_code)
    aperture_metric = _dof_aperture_fstop(scene_code)
    aperture_range = (
        success_spec.aperture_fstop_range()
        if success_spec is not None
        else None
    )
    has_depth_anchors = all(
        name in low
        for name in (
            "foreground_subject",
            "middleground_subject",
            "background_subject",
        )
    )
    animated_subjects = sorted(
        _objects_with_progressing_keyframes(
            scene_code,
            {"foreground_subject", "middleground_subject", "background_subject"},
            {"location", "scale", "rotation_euler"},
            min_delta=0.05,
        )
    )
    metrics = {
        "focus_distance_keyframed": focus_keyframed,
        "focus_distance_progresses": focus_progresses,
        "focus_distance_track": focus_track,
        "focus_distance_continuity": continuity,
        "use_dof_enabled": use_dof_enabled,
        "aperture_fstop": aperture_metric,
        "aperture_fstop_range": list(aperture_range) if aperture_range else None,
        "has_depth_anchors": has_depth_anchors,
        "animated_subjects": animated_subjects,
    }
    issues: list[ConceptMetricIssue] = []
    if not focus_keyframed:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_missing_focus_keyframe",
            message=(
                "Scene does not keyframe camera.data.dof.focus_distance. "
                "The focus pull effect requires animating the focus plane."
            ),
            suggested_fix=(
                "Set `camera.data.dof.focus_distance = <near>` then "
                "`camera.data.keyframe_insert('dof.focus_distance', "
                "frame=1)`; repeat with the far value at the final frame."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    elif not focus_progresses:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_focus_distance_flat",
            message=(
                "focus_distance is keyframed but never traverses a "
                "meaningful range (need at least 2 units from near to far "
                "subject)."
            ),
            suggested_fix=(
                "Pick focus_distance values that match the actual depth "
                "of foreground_subject (close) and background_subject "
                "(far) so the visible defocus area shifts across them."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    elif not continuity["ok"]:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_focus_transition_not_continuous",
            message=(
                "focus_distance changes, but the keyframe path does not "
                "cover a smooth early -> middle -> late focus pull. "
                f"Reason: {continuity['reason']}."
            ),
            suggested_fix=(
                "Keyframe camera.data.dof.focus_distance at early, middle, "
                "and late/final frames with monotonic near -> mid -> far "
                "values. Spread the transition across most of the timeline "
                "instead of holding for long intervals or jumping at shot cuts."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    if not use_dof_enabled:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_use_dof_not_enabled",
            message=(
                "camera.data.dof.use_dof is not set to True. Without it "
                "Blender ignores focus_distance and aperture entirely."
            ),
            suggested_fix=(
                "Set `camera.data.dof.use_dof = True` once before the "
                "focus_distance keyframes."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    if not has_depth_anchors:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_missing_depth_anchors",
            message=(
                "Scene must contain foreground_subject, "
                "middleground_subject, and background_subject placed at "
                "distinct depths so the focus pull has visible targets."
            ),
            suggested_fix=(
                "Create three named objects at increasing distance from "
                "the camera (near, middle, far) so the viewer can see "
                "each one come into focus in turn."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    if aperture_metric is not None:
        if aperture_range is not None:
            min_fstop, max_fstop = aperture_range
            if aperture_metric < min_fstop or aperture_metric > max_fstop:
                issues.append(ConceptMetricIssue(
                    severity="block",
                    rule_id="dof_aperture_out_of_success_range",
                    message=(
                        f"camera.data.dof.aperture_fstop = {aperture_metric:.1f} "
                        f"is outside the Success Spec range "
                        f"[{min_fstop:.1f}, {max_fstop:.1f}]."
                    ),
                    suggested_fix=(
                        "Set camera.data.dof.aperture_fstop inside the "
                        "success_spec required_visual_evidence "
                        "aperture_within_range interval."
                    ),
                    failure_class="success_hard",
                ))
        elif aperture_metric > _DOF_MAX_READABLE_FSTOP:
            issues.append(ConceptMetricIssue(
                severity="warn",
                rule_id="dof_aperture_too_wide",
                message=(
                    f"camera.data.dof.aperture_fstop = {aperture_metric:.1f} "
                    "is too narrow — depth of field will be too deep and the "
                    "blur change between focus distances will be invisible."
                ),
                suggested_fix=(
                    "Use a fast but physically plausible aperture_fstop around "
                    "1.4–2.8 for this demo so unfocused subjects clearly blur "
                    "out in preview frames and the focus pull is readable."
                ),
                failure_class="aesthetic_warn",
            ))
    if animated_subjects:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="dof_subjects_animated",
            message=(
                "Depth-anchor subjects must stay planted so the focus "
                f"pull is the only visible change. Found keyframes on "
                f"{', '.join(animated_subjects)}."
            ),
            suggested_fix=(
                "Remove keyframe_insert calls on foreground/"
                "middleground/background_subject. Only camera "
                "focus_distance should animate."
            ),
            failure_class="success_soft" if success_spec is None else "success_hard",
        ))
    return issues, metrics


@register_concept_metric("shadow_softness_radius")
def _shadow_softness_radius_metrics(
    scene_code: str,
    storyboard: Storyboard,
    success_spec: SuccessSpec | None = None,
) -> tuple[list[ConceptMetricIssue], dict]:
    low = scene_code.lower()
    embedded_storyboard = _embedded_storyboard_data(scene_code)
    embedded_keyframed, embedded_progresses = _storyboard_light_softness_progress(
        embedded_storyboard,
    )
    embedded_subject_animated = _storyboard_subject_motion(
        embedded_storyboard,
        {"subject_pillar", "subject_sphere"},
    )
    softness_keyframed = embedded_keyframed or _any_keyframe_on_data_path(
        scene_code,
        {"shadow_soft_size", "data.shadow_soft_size", "size", "data.size"},
        receiver_predicate=_is_light_receiver_name,
    )
    softness_progresses = embedded_progresses or _data_path_value_progresses(
        scene_code,
        {"shadow_soft_size", "data.shadow_soft_size", "size", "data.size"},
        min_delta=_MIN_SHADOW_SOFTNESS_DELTA,
        receiver_predicate=_is_light_receiver_name,
    )
    has_subject = "subject_pillar" in low or "subject_sphere" in low
    has_ground = "ground_plane" in low or "floor_plane" in low
    subject_animated = embedded_subject_animated or bool(
        _objects_with_progressing_keyframes(
            scene_code,
            {"subject_pillar", "subject_sphere"},
            {"location", "scale", "rotation_euler"},
            min_delta=0.05,
        )
    )
    metrics = {
        "softness_keyframed": softness_keyframed,
        "softness_progresses": softness_progresses,
        "softness_from_embedded_storyboard": embedded_keyframed,
        "has_subject": has_subject,
        "has_ground": has_ground,
        "subject_animated": subject_animated,
    }
    issues: list[ConceptMetricIssue] = []
    if not softness_keyframed:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="shadow_softness_missing_size_keyframe",
            message=(
                "No light keyframes `size` or `shadow_soft_size`. The "
                "hard-to-soft shadow transition cannot read without "
                "animating one of these properties."
            ),
            suggested_fix=(
                "On an area or point light, set "
                "`light.data.shadow_soft_size = 0.05` (hard) then "
                "`light.data.keyframe_insert('shadow_soft_size', frame=1)` "
                "and again with 0.8 (soft) at the final frame."
            ),
        ))
    elif not softness_progresses:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="shadow_softness_ramp_flat",
            message=(
                "Light size / shadow_soft_size keyframes exist but never "
                f"change by at least {_MIN_SHADOW_SOFTNESS_DELTA:.1f} units. "
                "The shadow edge will look identical at every frame."
            ),
            suggested_fix=(
                "Pick a small radius (~0.05) at the start and a large "
                "one (~0.8) at the end so the penumbra clearly expands."
            ),
        ))
    if not has_subject:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="shadow_softness_missing_subject",
            message=(
                "Scene needs a subject_pillar (or subject_sphere) casting "
                "the shadow so the softness change has a visible anchor."
            ),
            suggested_fix=(
                "Add a centered upright primitive named 'subject_pillar' "
                "positioned between the light and the ground plane."
            ),
        ))
    if not has_ground:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="shadow_softness_missing_ground",
            message=(
                "Scene needs a ground_plane (or floor_plane) to receive "
                "the cast shadow."
            ),
            suggested_fix=(
                "Add a large flat plane named 'ground_plane' below the "
                "subject so the shadow has somewhere to land."
            ),
        ))
    if subject_animated:
        issues.append(ConceptMetricIssue(
            severity="block",
            rule_id="shadow_softness_subject_animated",
            message=(
                "The shadow-casting subject must stay static so the only "
                "visible change is shadow softness."
            ),
            suggested_fix=(
                "Remove keyframe_insert calls on subject_pillar / "
                "subject_sphere. Only the light's size / shadow_soft_size "
                "should animate."
            ),
        ))
    return issues, metrics


def _embedded_storyboard_data(scene_code: str) -> dict | None:
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "STORYBOARD" for t in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None
    return None


def _storyboard_light_softness_progress(data: dict | None) -> tuple[bool, bool]:
    if not data:
        return False, False
    values: list[float] = []
    keyframed = False
    for shot in data.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        for obj in shot.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", ""))
            kind = str(obj.get("type", "")).lower()
            if not (_is_light_receiver_name(name) or kind == "light"):
                continue
            for key in obj.get("keyframes", []) or []:
                if not isinstance(key, dict):
                    continue
                attr = str(key.get("attr", ""))
                if attr not in {"size", "data.size", "shadow_soft_size", "data.shadow_soft_size"}:
                    continue
                value = key.get("value")
                if isinstance(value, (int, float)):
                    keyframed = True
                    values.append(float(value))
    return keyframed, len(values) >= 2 and (max(values) - min(values)) >= 0.5


def _storyboard_subject_motion(data: dict | None, names: set[str]) -> bool:
    if not data:
        return False
    values_by_name_attr: dict[tuple[str, str], list[tuple[float, ...]]] = {}
    for shot in data.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        for obj in shot.get("objects", []) or []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name", "")).lower()
            if name not in names:
                continue
            for key in obj.get("keyframes", []) or []:
                if not isinstance(key, dict):
                    continue
                attr = str(key.get("attr", ""))
                if attr in {"hide_render", "hide_viewport"}:
                    continue
                value = key.get("value")
                if isinstance(value, (int, float)):
                    vec = (float(value),)
                elif isinstance(value, list) and all(
                    isinstance(v, (int, float)) for v in value
                ):
                    vec = tuple(float(v) for v in value)
                else:
                    continue
                values_by_name_attr.setdefault((name, attr), []).append(vec)
    return any(_vector_delta(values) >= 0.05 for values in values_by_name_attr.values())


def _storyboard_keyframe_coverage_metrics(
    scene_code: str,
    storyboard: Storyboard,
) -> tuple[list[ConceptMetricIssue], dict]:
    expected = _storyboard_keyframed_object_names(storyboard)
    if not expected:
        return [], {}
    animated = _keyframed_object_names(scene_code)
    if "apply_spec_keyframes(obj, spec)" in scene_code:
        return [], {
            "expected_objects": sorted(expected),
            "animated_objects": sorted(animated),
            "coverage": 1.0,
            "via_helper": "apply_spec_keyframes",
        }
    created = _created_object_names(scene_code)
    proxy_covered = {
        name for name in expected - animated
        if name in created and _has_animated_proxy(name, animated)
    }
    missing = sorted(expected - animated - proxy_covered)
    coverage = (len(expected) - len(missing)) / max(1, len(expected))
    metrics = {
        "expected_objects": sorted(expected),
        "animated_objects": sorted(animated),
        "created_objects_sample": sorted(created)[:80],
        "proxy_covered_objects": sorted(proxy_covered),
        "missing_objects": missing,
        "coverage": round(coverage, 4),
    }
    if not missing:
        return [], metrics
    return [ConceptMetricIssue(
        severity="block",
        rule_id="storyboard_keyframe_coverage_missing",
        message=(
            "Storyboard declares non-visibility keyframes for object(s) "
            f"{', '.join(missing[:8])}, but scene.py does not keyframe "
            "those exact teaching anchors."
        ),
        suggested_fix=(
            "Implement the storyboard motion on the named anchors with "
            "`obj.keyframe_insert('location'/'scale'/'rotation_euler', ...)` "
            "or, for curve/ray objects, animate `obj.data.bevel_factor_end` "
            "with data keyframes across the shot."
        ),
    )], metrics


def _storyboard_keyframed_object_names(storyboard: Storyboard) -> set[str]:
    names: set[str] = set()
    for shot in storyboard.shots:
        for obj in shot.objects:
            if not obj.name:
                continue
            if any(
                key.attr not in {"hide_render", "hide_viewport"}
                for key in obj.keyframes
            ):
                names.add(obj.name)
    return names


def _created_object_names(scene_code: str) -> set[str]:
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return set()
    return set(_object_aliases(tree).values())


def _has_animated_proxy(anchor: str, animated_names: set[str]) -> bool:
    proxy_names = {
        f"{anchor}_marker",
        f"{anchor}_proxy",
        f"{anchor}_handle",
    }
    return bool(proxy_names & animated_names)


def _keyframed_object_names(scene_code: str) -> set[str]:
    """Return the set of variable names whose attributes call keyframe_insert.

    Resolves ``obj.location = (...)\\nobj.keyframe_insert('location')`` style
    code by walking the AST. We collect the variable receiving the
    ``keyframe_insert`` call; if that variable was assigned from
    ``bpy.data.objects[...]`` or ``bpy.data.objects.new(name, ...)``, we
    resolve to the underlying object name. Plain receiver names are kept
    so callers can match against naming conventions like ``marker_post_1``.
    """
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return set()

    object_var_to_name = _object_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        target_var = node.targets[0].id
        resolved = _resolve_object_name_expr(node.value)
        if resolved is not None:
            object_var_to_name[target_var] = resolved

    loop_var_to_object_names = _loop_object_aliases(tree, object_var_to_name)
    receivers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "keyframe_insert":
            continue
        recv = node.func.value
        recv_name = _attr_chain_root(recv)
        if recv_name is None:
            continue
        receivers.add(recv_name)

    resolved_names: set[str] = set()
    for name in receivers:
        if name in loop_var_to_object_names:
            resolved_names.update(loop_var_to_object_names[name])
        else:
            resolved_names.add(object_var_to_name.get(name, name))
    return resolved_names


def _object_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str | None] = {}
    helper_name_arg_indices = _helper_object_name_arg_indices(tree)
    for node in sorted(
        ast.walk(tree),
        key=lambda n: (
            getattr(n, "lineno", 10**9),
            getattr(n, "col_offset", 10**9),
        ),
    ):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            resolved = _resolve_object_name_expr(node.value)
            if resolved is None:
                resolved = _resolve_helper_object_name_expr(
                    node.value,
                    helper_name_arg_indices,
                )
            if resolved is not None:
                aliases[target.id] = resolved
            elif _attr_chain(node.value) in {
                ("bpy", "context", "object"),
                ("bpy", "context", "active_object"),
            }:
                aliases.setdefault(target.id, None)
            continue
        if not isinstance(target, ast.Attribute) or target.attr != "name":
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        root = _attr_chain_root(target.value)
        if root is not None:
            aliases[root] = node.value.value
    return {k: v for k, v in aliases.items() if v is not None}


def _object_name_assignments(tree: ast.AST) -> dict[str, str]:
    """Return variable name -> exact object name for simple creation patterns."""
    return _object_aliases(tree)


def _assignment_targets(tree: ast.AST) -> dict[tuple[str, str], list[ast.AST]]:
    out: dict[tuple[str, str], list[ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            root = _attr_chain_root(target.value)
            if root is None:
                continue
            out.setdefault((root, target.attr), []).append(node.value)
    return out


def _has_negative_component(node: ast.AST) -> bool:
    values: list[float] = []
    if isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            literal = _literal_number(item)
            if literal is not None:
                values.append(literal)
    else:
        literal = _literal_number(node)
        if literal is not None:
            values.append(literal)
    return any(value < 0 for value in values)


def _text_anchor_has_camera_facing_evidence(
    scene_code: str,
    var_name: str,
    object_name: str,
) -> bool:
    if f"{var_name}.rotation_euler" in scene_code:
        return True
    if f"{object_name}.rotation_euler" in scene_code:
        return True
    if "Track To" in scene_code and var_name in scene_code:
        return True
    if "track_axis" in scene_code and var_name in scene_code:
        return True
    if "look_at(" in scene_code and (var_name in scene_code or object_name in scene_code):
        return True
    return False


def _text_anchor_has_camera_parent_evidence(
    tree: ast.AST,
    assignments: dict[tuple[str, str], list[ast.AST]],
    var_name: str,
    object_name: str,
    *,
    helper_calls: dict[str, str],
    helper_evidence: dict[str, dict[str, bool]],
) -> bool:
    parent_values = assignments.get((var_name, "parent"), [])
    has_camera_parent = any(
        _is_camera_reference(value) for value in parent_values
    )
    helper_name = helper_calls.get(var_name)
    evidence = helper_evidence.get(helper_name or "", {})
    has_camera_parent = has_camera_parent or bool(evidence.get("camera_parent"))
    if not has_camera_parent:
        return False
    location_values = assignments.get((var_name, "location"), [])
    has_fixed_local_location = any(
        _tuple_has_depth_component(value) for value in location_values
    )
    has_fixed_local_location = has_fixed_local_location or bool(
        evidence.get("fixed_location")
    )
    return has_fixed_local_location


def _is_camera_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "cam" in node.id.lower() or "camera" in node.id.lower()
    resolved = _resolve_object_name_expr(node)
    if resolved is not None:
        return "camera" in resolved.lower() or resolved.lower() == "main_camera"
    chain = _attr_chain(node)
    return bool(chain and any("camera" in part.lower() for part in chain))


def _tuple_has_depth_component(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) < 3:
        return False
    z = _literal_number(node.elts[2])
    return z is not None and abs(z) > 0.01


def _loop_object_aliases(
    tree: ast.AST,
    object_var_to_name: dict[str, str],
) -> dict[str, set[str]]:
    """Resolve simple loop variables used to keyframe several objects.

    LLM-generated scenes often use compact patterns like::

        for wp_obj, wp_pos in ((waypoint_1, WP1), (waypoint_2, WP2)):
            wp_obj.keyframe_insert('location', frame=frame)

    The generic keyframe coverage metric should credit this as keyframing
    both ``waypoint_1`` and ``waypoint_2``. We intentionally keep the
    recognizer conservative: it only handles literal tuple/list iterables
    whose elements are names, or tuples/lists whose first fields are names.
    """
    aliases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        target_names = _loop_target_names(node.target)
        if not target_names:
            continue
        per_slot_names: list[set[str]] = [set() for _ in target_names]
        for item in _literal_iter_items(node.iter):
            fields = _literal_iter_items(item)
            if not fields:
                fields = [item]
            for idx, field in enumerate(fields[:len(target_names)]):
                object_name = _object_name_from_loop_field(field, object_var_to_name)
                if object_name:
                    per_slot_names[idx].add(object_name)
        for name, object_names in zip(target_names, per_slot_names):
            if object_names:
                aliases.setdefault(name, set()).update(object_names)
    return aliases


def _loop_target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            else:
                return []
        return names
    return []


def _literal_iter_items(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return list(node.elts)
    return []


def _object_name_from_loop_field(
    node: ast.AST,
    object_var_to_name: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return object_var_to_name.get(node.id, node.id)
    return _resolve_object_name_expr(node)


def _resolve_object_name_expr(value: ast.AST) -> str | None:
    """Recognize ``bpy.data.objects['x']`` and ``bpy.data.objects.new('x', ...)``.

    Returns the literal name argument if the expression is one of these
    bpy patterns, otherwise None.
    """
    if isinstance(value, ast.Subscript):
        if _attr_chain(value.value) == ("bpy", "data", "objects"):
            slc = value.slice
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                return slc.value
    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Attribute) and value.func.attr == "new":
            if _attr_chain(value.func.value) == ("bpy", "data", "objects"):
                if value.args and isinstance(value.args[0], ast.Constant):
                    if isinstance(value.args[0].value, str):
                        return value.args[0].value
    return None


def _resolve_helper_object_name_expr(
    value: ast.AST,
    helper_name_arg_indices: dict[str, int],
) -> str | None:
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    name_index = helper_name_arg_indices.get(value.func.id)
    if name_index is None or name_index >= len(value.args):
        return None
    arg = value.args[name_index]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _helper_object_name_arg_indices(tree: ast.AST) -> dict[str, int]:
    helpers: dict[str, int] = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        params = [arg.arg for arg in fn.args.args]
        assigned_name_params: set[str] = set()
        creates_object = False
        returns_named_object = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "text_add"
                    and _attr_chain(node.func.value) == ("bpy", "ops", "object")
                ):
                    creates_object = True
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "new"
                    and _attr_chain(node.func.value) == ("bpy", "data", "objects")
                ):
                    creates_object = True
                    if node.args and isinstance(node.args[0], ast.Name):
                        if node.args[0].id in params:
                            assigned_name_params.add(node.args[0].id)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Attribute) or target.attr != "name":
                        continue
                    if isinstance(node.value, ast.Name) and node.value.id in params:
                        assigned_name_params.add(node.value.id)
                continue
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                returns_named_object = True
        if creates_object and returns_named_object and assigned_name_params:
            helpers[fn.name] = min(params.index(name) for name in assigned_name_params)
    return helpers


def _helper_object_calls(tree: ast.AST) -> dict[str, str]:
    calls: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            calls[target.id] = node.value.func.id
    return calls


def _helper_object_evidence(tree: ast.AST) -> dict[str, dict[str, bool]]:
    evidence: dict[str, dict[str, bool]] = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        data = {"camera_parent": False, "fixed_location": False}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if target.attr == "parent" and _is_camera_reference(node.value):
                    data["camera_parent"] = True
                if target.attr == "location" and _tuple_has_depth_component(node.value):
                    data["fixed_location"] = True
        evidence[fn.name] = data
    return evidence


def _attr_chain(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return tuple(reversed(parts))
    return ()


def _attr_chain_root(node: ast.AST) -> str | None:
    chain = _attr_chain(node)
    if chain:
        return chain[0]
    if isinstance(node, ast.Name):
        return node.id
    return None


def _has_lens_keyframe(scene_code: str) -> bool:
    """True iff ``keyframe_insert`` is actually called with ``lens`` as
    the data_path argument. Substring presence of ``data.lens`` and
    ``keyframe_insert`` elsewhere in the file is not enough — that pattern
    matches scenes that only keyframe ``location`` while assigning ``lens``
    once outside any frame loop, which is the dolly_zoom failure mode."""
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "keyframe_insert":
            continue
        data_path = _keyframe_data_path_arg(node)
        if data_path is None:
            continue
        if data_path == "lens" or data_path.endswith(".lens"):
            return True
    return False


def _any_keyframe_on_data_path(
    scene_code: str,
    paths: set[str],
    *,
    receiver_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """True iff some ``keyframe_insert`` call uses one of the given paths.

    A literal match against the data_path arg, plus an ``endswith('.' + p)``
    fallback for cases like ``camera.data.keyframe_insert('dof.focus_distance')``
    where the chain includes a parent attribute prefix.
    """
    return _any_keyframe_on_data_path_filtered(
        scene_code,
        paths,
        receiver_predicate=receiver_predicate,
    )


def _any_keyframe_on_data_path_filtered(
    scene_code: str,
    paths: set[str],
    *,
    receiver_predicate: Callable[[str], bool] | None = None,
) -> bool:
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return False
    object_var_to_name = _object_aliases(tree)
    suffixes = {"." + p for p in paths}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "keyframe_insert":
            continue
        path = _keyframe_data_path_arg(node)
        if path is None:
            continue
        if not (path in paths or any(path.endswith(suf) for suf in suffixes)):
            continue
        receiver = _attr_chain_root(node.func.value)
        resolved_receiver = object_var_to_name.get(receiver, receiver) if receiver else None
        if receiver_predicate is not None:
            if resolved_receiver is None or not receiver_predicate(resolved_receiver):
                continue
        return True
    return False


def _data_path_value_progresses(
    scene_code: str,
    paths: set[str],
    *,
    min_delta: float,
    receiver_predicate: Callable[[str], bool] | None = None,
) -> bool:
    """True iff a value assigned to one of ``paths`` later differs from an
    earlier assigned value by at least ``min_delta``.

    We look for ``X = literal`` followed (in source order) by
    ``X.keyframe_insert('path', ...)``. The chain prefix on X is ignored;
    only the trailing attribute name must match an entry in ``paths`` (or
    in its ``.``-suffix form, e.g. ``dof.focus_distance``).
    """
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return False
    loop_numeric_aliases = _loop_numeric_aliases(tree)
    object_var_to_name = _object_aliases(tree)
    values_per_path: dict[str, list[float]] = {p: [] for p in paths}
    last_assigned: dict[str, dict[str, float]] = {p: {} for p in paths}
    nodes = sorted(
        ast.walk(tree),
        key=lambda n: (getattr(n, "lineno", 10**9),
                       getattr(n, "col_offset", 10**9)),
    )
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute):
                values = _numeric_values(node.value, loop_numeric_aliases)
                if values:
                    chain = _attr_chain(target)
                    if chain:
                        receiver = chain[0]
                        resolved_receiver = object_var_to_name.get(receiver, receiver)
                        trailing_attr = chain[-1]
                        if (
                            receiver_predicate is not None
                            and not receiver_predicate(resolved_receiver)
                        ):
                            continue
                        for p in paths:
                            tail = p.split(".")[-1]
                            if trailing_attr == tail:
                                last_assigned[p][receiver] = values[-1]
                                values_per_path[p].extend(values)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "keyframe_insert":
                path = _keyframe_data_path_arg(node)
                if path is not None:
                    receiver = _attr_chain_root(node.func.value)
                    if receiver is None:
                        continue
                    resolved_receiver = object_var_to_name.get(receiver, receiver)
                    if (
                        receiver_predicate is not None
                        and not receiver_predicate(resolved_receiver)
                    ):
                        continue
                    for p in paths:
                        if path == p or path.endswith("." + p):
                            val = last_assigned.get(p, {}).get(receiver)
                            if val is None and "." in p:
                                # try last segment match for fallback receivers
                                tail = p.split(".")[-1]
                                for rec_name, recorded in last_assigned[p].items():
                                    if rec_name == receiver:
                                        val = recorded
                                        break
                            if val is not None:
                                values_per_path[p].append(val)
    for vals in values_per_path.values():
        if len(vals) >= 2 and (max(vals) - min(vals)) >= min_delta:
            return True
    return False


def _keyframed_data_path_uses_distinct_values(
    scene_code: str,
    paths: set[str],
    *,
    min_numeric_delta: float,
) -> bool:
    """Detect value progress when generated code hides values behind names.

    LLM scenes often wrap keyframing in helpers such as::

        def key_focus(frame, distance):
            cam.data.dof.focus_distance = distance
            cam.data.dof.keyframe_insert('focus_distance', frame=frame)
        key_focus(s1_start, D_NEAR)
        key_focus(s3_end, D_MID)

    The ordinary numeric pass cannot evaluate ``D_NEAR = dist_to(...)``, but
    distinct semantic targets (D_NEAR/D_MID/D_FAR) are still strong evidence
    that the focus plane is intended to traverse the depth anchors.
    """
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return False

    tokens: list[str] = []
    numeric_values: list[float] = []
    last_assigned: dict[str, str] = {}

    wrapper_param_indices = _keyframe_wrapper_value_params(tree, paths)
    nodes_inside_functions = {
        id(child)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for child in ast.walk(fn)
        if child is not fn
    }
    nodes = sorted(
        ast.walk(tree),
        key=lambda n: (getattr(n, "lineno", 10**9),
                       getattr(n, "col_offset", 10**9)),
    )
    for node in nodes:
        if id(node) in nodes_inside_functions:
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Attribute):
                continue
            chain = _attr_chain(target)
            if not chain:
                continue
            trailing_attr = chain[-1]
            if not any(trailing_attr == p.split(".")[-1] for p in paths):
                continue
            receiver = chain[0]
            token = _value_identity_token(node.value)
            if token is not None:
                last_assigned[receiver] = token
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "keyframe_insert":
                continue
            path = _keyframe_data_path_arg(node)
            if path is None or not _path_matches(path, paths):
                continue
            receiver = _attr_chain_root(node.func.value)
            if receiver is None:
                continue
            token = last_assigned.get(receiver)
            if token is not None:
                tokens.append(token)
                numeric = _numeric_token_value(token)
                if numeric is not None:
                    numeric_values.append(numeric)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            param_index = wrapper_param_indices.get(node.func.id)
            if param_index is None or param_index >= len(node.args):
                continue
            token = _value_identity_token(node.args[param_index])
            if token is not None:
                tokens.append(token)
                numeric = _numeric_token_value(token)
                if numeric is not None:
                    numeric_values.append(numeric)

    if len(numeric_values) >= 2 and max(numeric_values) - min(numeric_values) >= min_numeric_delta:
        return True
    symbolic = {
        token for token in tokens
        if _numeric_token_value(token) is None
    }
    return len(symbolic) >= 2


def _data_path_keyframe_track(
    scene_code: str,
    paths: set[str],
    *,
    total_frames: int | None = None,
) -> list[dict[str, float]]:
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return []
    loop_numeric_aliases = _loop_numeric_aliases(tree)
    last_assigned: dict[str, float] = {}
    last_symbol_assigned: dict[str, str] = {}
    wrapper_param_indices = _keyframe_wrapper_value_params(tree, paths)
    track: list[tuple[int, float | str]] = []
    numeric_symbols = _symbolic_numeric_assignments(tree)
    frame_symbols = _symbolic_frame_assignments(tree, total_frames=total_frames)
    nodes_inside_functions = {
        id(child)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for child in ast.walk(fn)
        if child is not fn
    }
    nodes = sorted(
        ast.walk(tree),
        key=lambda n: (getattr(n, "lineno", 10**9),
                       getattr(n, "col_offset", 10**9)),
    )
    for node in nodes:
        if id(node) in nodes_inside_functions:
            continue
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Attribute):
                continue
            chain = _attr_chain(target)
            if not chain or not any(chain[-1] == p.split(".")[-1] for p in paths):
                continue
            values = _numeric_values(node.value, loop_numeric_aliases)
            if values:
                last_assigned[chain[0]] = values[-1]
            token = _value_identity_token(node.value)
            if token is not None:
                last_symbol_assigned[chain[0]] = token
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "keyframe_insert":
                continue
            path = _keyframe_data_path_arg(node)
            if path is None or not _path_matches(path, paths):
                continue
            receiver = _attr_chain_root(node.func.value)
            if receiver is None:
                continue
            frame = _call_frame_arg(node)
            if frame is None:
                frame = _call_frame_arg(node, frame_symbols=frame_symbols)
            if frame is not None and receiver in last_assigned:
                track.append((frame, last_assigned[receiver]))
            elif frame is not None and receiver in last_symbol_assigned:
                token = last_symbol_assigned[receiver]
                track.append((frame, numeric_symbols.get(token, token)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            param_index = wrapper_param_indices.get(node.func.id)
            if param_index is None or param_index >= len(node.args):
                continue
            frame = _call_frame_arg(node)
            if frame is None:
                frame = _call_frame_arg(node, frame_symbols=frame_symbols)
            if frame is None:
                continue
            value = _literal_number(node.args[param_index])
            if value is not None:
                track.append((frame, value))
                continue
            token = _value_identity_token(node.args[param_index])
            if token is not None:
                track.append((frame, numeric_symbols.get(token, token)))
    return [
        {"frame": frame, "value": value}
        for frame, value in sorted(set(track))
    ]


def _call_frame_arg(
    call: ast.Call,
    *,
    frame_symbols: dict[str, int] | None = None,
) -> int | None:
    for kw in call.keywords:
        if kw.arg == "frame":
            value = _literal_number(kw.value)
            if value is not None:
                return int(value)
            if isinstance(kw.value, ast.Name) and frame_symbols is not None:
                return frame_symbols.get(kw.value.id)
            return None
    if call.args:
        value = _literal_number(call.args[0])
        if value is not None:
            return int(value)
        if isinstance(call.args[0], ast.Name) and frame_symbols is not None:
            return frame_symbols.get(call.args[0].id)
    return None


def _focus_track_continuity(
    track: list[dict[str, float]],
    *,
    total_frames: int,
) -> dict[str, object]:
    if len(track) < 3:
        return {
            "ok": False,
            "reason": "need at least three numeric focus_distance keyframes",
        }
    frames = [int(item["frame"]) for item in track]
    values = [_focus_track_value_order(item["value"]) for item in track]
    if any(value is None for value in values):
        return {
            "ok": False,
            "reason": "focus_distance track contains unordered symbolic values",
        }
    ordered_values = [float(value) for value in values if value is not None]
    span = max(frames) - min(frames)
    min_span = max(1, int(total_frames * 0.55))
    if span < min_span:
        return {
            "ok": False,
            "reason": (
                f"focus keyframes span {span} frame(s), below required "
                f"{min_span} frame(s)"
            ),
            "span": span,
            "min_span": min_span,
        }
    deltas = [b - a for a, b in zip(ordered_values, ordered_values[1:])]
    nonzero = [d for d in deltas if abs(d) > 1e-6]
    if not nonzero:
        return {"ok": False, "reason": "focus_distance values are flat"}
    signs = {1 if d > 0 else -1 for d in nonzero}
    if len(signs) > 1:
        return {
            "ok": False,
            "reason": "focus_distance reverses direction instead of moving monotonically",
        }
    max_gap = max(b - a for a, b in zip(frames, frames[1:]))
    max_allowed_gap = max(1, int(total_frames * 0.55))
    if max_gap > max_allowed_gap:
        return {
            "ok": False,
            "reason": (
                f"largest focus keyframe gap is {max_gap} frame(s), above "
                f"allowed {max_allowed_gap}"
            ),
            "max_gap": max_gap,
            "max_allowed_gap": max_allowed_gap,
        }
    return {
        "ok": True,
        "reason": "focus keyframes span the timeline monotonically",
        "span": span,
        "max_gap": max_gap,
    }


def _focus_track_value_order(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    token = value.upper()
    if "NEAR" in token:
        return 1.0
    if "MID" in token or "MIDDLE" in token:
        return 2.0
    if "FAR" in token or "BACK" in token:
        return 3.0
    return _numeric_token_value(value)


def _symbolic_numeric_assignments(tree: ast.AST) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _literal_number(node.value)
        if value is not None:
            values[target.id] = value
    return values


def _symbolic_frame_assignments(
    tree: ast.AST,
    *,
    total_frames: int | None = None,
) -> dict[str, int]:
    values: dict[str, int] = {}
    tuple_values: dict[str, list[tuple[int, ...]]] = _literal_tuple_list_assignments(tree)
    if total_frames is not None:
        tuple_values.update(_shot_start_tuple_assignments(tree, total_frames=total_frames))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Subscript):
            source = node.value
            if not isinstance(source.slice, ast.Constant):
                continue
            idx = source.slice.value
            if not isinstance(idx, int):
                continue
            source_name = _attr_chain_root(source.value)
            if source_name is None:
                continue
            assigned = tuple_values.get(source_name, [])
            if idx < 0 or idx >= len(assigned):
                continue
            fields = assigned[idx]
            tuple_names = [
                elt.id for elt in node.targets[0].elts
                if isinstance(elt, ast.Name)
            ]
            for name, value in zip(tuple_names, fields):
                if isinstance(value, int):
                    values[name] = value
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            value = _literal_number(node.value)
            if value is not None:
                values[target.id] = int(value)
    return values


def _shot_start_tuple_assignments(
    tree: ast.AST,
    *,
    total_frames: int,
) -> dict[str, list[tuple[int, ...]]]:
    names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and "shot" in target.id.lower()
    }
    if not names:
        names = {"shot_starts"}
    # Conservative fallback for storyboard-derived shot ranges. This handles
    # generated code that builds shot_starts in a loop from STORYBOARD data,
    # which is common and not literal enough for static AST evaluation.
    q = max(1, total_frames // 4)
    ranges = [(1, q), (q + 1, q * 2), (q * 2 + 1, q * 3), (q * 3 + 1, total_frames)]
    return {name: ranges for name in names}


def _literal_tuple_list_assignments(tree: ast.AST) -> dict[str, list[tuple[int, ...]]]:
    values: dict[str, list[tuple[int, ...]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        rows: list[tuple[int, ...]] = []
        for elt in node.value.elts:
            if not isinstance(elt, (ast.List, ast.Tuple)):
                continue
            row: list[int] = []
            for item in elt.elts:
                number = _literal_number(item)
                if number is None:
                    break
                row.append(int(number))
            else:
                rows.append(tuple(row))
        if rows:
            values[target.id] = rows
    return values


def _keyframe_wrapper_value_params(tree: ast.AST, paths: set[str]) -> dict[str, int]:
    wrappers: dict[str, int] = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        params = [arg.arg for arg in fn.args.args]
        assigned_params: set[str] = set()
        keyframes_path = False
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Attribute):
                    continue
                chain = _attr_chain(target)
                if not chain or not any(chain[-1] == p.split(".")[-1] for p in paths):
                    continue
                if isinstance(node.value, ast.Name) and node.value.id in params:
                    assigned_params.add(node.value.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr != "keyframe_insert":
                    continue
                path = _keyframe_data_path_arg(node)
                if path is not None and _path_matches(path, paths):
                    keyframes_path = True
        if not keyframes_path:
            continue
        for name in assigned_params:
            wrappers[fn.name] = params.index(name)
            break
    return wrappers


def _value_identity_token(node: ast.AST) -> str | None:
    literal = _literal_number(node)
    if literal is not None:
        return repr(literal)
    if isinstance(node, ast.Name):
        return node.id
    try:
        token = ast.unparse(node).strip()
    except Exception:
        return None
    return token or None


def _numeric_token_value(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _objects_with_progressing_keyframes(
    scene_code: str,
    object_names: set[str],
    paths: set[str],
    *,
    min_delta: float,
    name_predicate: Callable[[str], bool] | None = None,
) -> set[str]:
    """Return object names whose keyed values actually change.

    This distinguishes a static hold keyframe from real animation. The
    generic `_keyframed_object_names()` is intentionally broad; concept
    metrics use this helper when a planted anchor must remain still.
    """
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return set()
    object_var_to_name = _object_aliases(tree)
    loop_numeric_aliases = _loop_numeric_aliases(tree)
    wanted = {name.lower() for name in object_names}
    keyed_values: dict[str, dict[str, list[tuple[float, ...]]]] = {}
    last_assigned: dict[tuple[str, str], tuple[float, ...]] = {}
    nodes = sorted(
        ast.walk(tree),
        key=lambda n: (getattr(n, "lineno", 10**9),
                       getattr(n, "col_offset", 10**9)),
    )
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Attribute):
                continue
            chain = _attr_chain(target)
            if not chain:
                continue
            root = chain[0]
            path = ".".join(chain[1:]) if len(chain) > 1 else chain[-1]
            if not _path_matches(path, paths):
                continue
            value = _numeric_vector(node.value, loop_numeric_aliases)
            if value is not None:
                last_assigned[(root, path)] = value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "keyframe_insert":
                continue
            path = _keyframe_data_path_arg(node)
            root = _attr_chain_root(node.func.value)
            if path is None or root is None or not _path_matches(path, paths):
                continue
            resolved = object_var_to_name.get(root, root)
            low = resolved.lower()
            if wanted and low not in wanted:
                continue
            if name_predicate is not None and not name_predicate(resolved):
                continue
            value = (
                last_assigned.get((root, path))
                or last_assigned.get((root, path.split(".")[-1]))
            )
            if value is None:
                continue
            keyed_values.setdefault(resolved, {}).setdefault(path, []).append(value)
    progressing: set[str] = set()
    for name, by_path in keyed_values.items():
        for values in by_path.values():
            if _vector_delta(values) >= min_delta:
                progressing.add(name)
                break
    return progressing


def _path_matches(path: str, paths: set[str]) -> bool:
    return path in paths or any(path.endswith("." + p) for p in paths)


def _numeric_vector(
    node: ast.AST,
    loop_numeric_aliases: dict[str, list[float]],
) -> tuple[float, ...] | None:
    scalar_values = _numeric_values(node, loop_numeric_aliases)
    if scalar_values:
        return tuple(scalar_values)
    if isinstance(node, (ast.Tuple, ast.List)):
        vals: list[float] = []
        for item in node.elts:
            literal = _literal_number(item)
            if literal is None:
                return None
            vals.append(literal)
        return tuple(vals)
    return None


def _vector_delta(values: list[tuple[float, ...]]) -> float:
    if len(values) < 2:
        return 0.0
    max_delta = 0.0
    for i, a in enumerate(values):
        for b in values[i + 1:]:
            n = min(len(a), len(b))
            if n == 0:
                continue
            delta = sum((a[j] - b[j]) ** 2 for j in range(n)) ** 0.5
            max_delta = max(max_delta, delta)
    return max_delta


def _is_light_receiver_name(receiver: str) -> bool:
    low = receiver.lower()
    if "label" in low or "text" in low:
        return False
    return "light" in low or "lamp" in low or "area" in low


def _loop_numeric_aliases(tree: ast.AST) -> dict[str, list[float]]:
    """Return possible numeric values for simple loop variables.

    Handles common generated patterns such as::

        SHOT_FACTORS = [(frame1, 0.0), (frame2, 1.0)]
        for frame, factor in SHOT_FACTORS:
            curve.data.bevel_factor_end = factor

    The analysis is deliberately shallow and deterministic; it does not
    evaluate expressions or function calls.
    """
    constants: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = node.value

    aliases: dict[str, list[float]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        target_names = _loop_target_names(node.target)
        if not target_names:
            continue
        iter_node = constants.get(node.iter.id) if isinstance(node.iter, ast.Name) else node.iter
        items = _literal_iter_items(iter_node) if iter_node is not None else []
        if not items:
            continue
        per_slot_values: list[list[float]] = [[] for _ in target_names]
        for item in items:
            fields = _literal_iter_items(item)
            if not fields:
                fields = [item]
            for idx, field in enumerate(fields[:len(target_names)]):
                value = _literal_number(field)
                if value is not None:
                    per_slot_values[idx].append(value)
        for name, values in zip(target_names, per_slot_values):
            if values:
                aliases.setdefault(name, []).extend(values)
    return aliases


def _numeric_values(
    node: ast.AST,
    loop_numeric_aliases: dict[str, list[float]],
) -> list[float]:
    literal = _literal_number(node)
    if literal is not None:
        return [literal]
    if isinstance(node, ast.Name):
        return list(loop_numeric_aliases.get(node.id, []))
    return []


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_number(node.operand)
        if value is not None:
            return -value
    return None


def _dof_use_dof_enabled(scene_code: str) -> bool:
    """Detect ``camera.data.dof.use_dof = True`` or any ``X.use_dof = True``."""
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute):
            continue
        if target.attr != "use_dof":
            continue
        if isinstance(node.value, ast.Constant) and node.value.value is True:
            return True
    return False


def _dof_aperture_fstop(scene_code: str) -> float | None:
    """Return the most recent ``X.dof.aperture_fstop = literal`` value, if any."""
    try:
        tree = ast.parse(scene_code)
    except SyntaxError:
        return None
    found: float | None = None
    for node in sorted(
        ast.walk(tree),
        key=lambda n: (getattr(n, "lineno", 10**9),
                       getattr(n, "col_offset", 10**9)),
    ):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Attribute) or target.attr != "aperture_fstop":
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
            found = float(node.value.value)
    return found


def _keyframe_data_path_arg(call: ast.Call) -> str | None:
    """Return the literal data_path passed to keyframe_insert, if any."""
    # Positional: first arg
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    # Keyword: data_path=...
    for kw in call.keywords:
        if kw.arg == "data_path":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
    return None


def _storyboard_camera_y_delta(storyboard: Storyboard) -> float:
    values: list[float] = []
    for shot in storyboard.shots:
        for key in shot.camera:
            values.append(float(key.position[1]))
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _storyboard_fov_delta(storyboard: Storyboard) -> float:
    values: list[float] = []
    for shot in storyboard.shots:
        for key in shot.camera:
            if key.fov is not None:
                values.append(float(key.fov))
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def concept_metric_report_to_json(report: ConceptMetricReport) -> str:
    return json.dumps(report.to_dict(), indent=2)


def format_concept_metric_report_for_coder(report: ConceptMetricReport) -> str:
    """Render the metric findings as a coder-facing addendum section.

    Returns an empty string when there are no findings, so this can be
    safely joined with other addendum parts without producing dangling
    headers.
    """
    if not report.issues:
        return ""
    lines = [
        "AUTOMATED CONCEPT METRIC FAILURES",
        "These are deterministic AST/storyboard checks. Treat 'block'",
        "findings with failure_class=success_hard or structural_fatal as",
        "hard pre-render failures: fix them before relying on critic visual feedback.",
        "Apply the smallest possible code change for each success-hard finding.",
        "",
    ]
    for issue in report.issues:
        tag = f"{issue.severity.upper()} / {issue.failure_class}"
        lines.append(f"[{tag}] {issue.rule_id}")
        lines.append(f"  {issue.message}")
        if issue.suggested_fix:
            lines.append(f"  fix: {issue.suggested_fix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
