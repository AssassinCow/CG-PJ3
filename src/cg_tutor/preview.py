"""Cheap keyframe preview rendering and deterministic image checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.schemas import Storyboard


@dataclass
class PreviewIssue:
    severity: str
    rule_id: str
    frame_idx: int | None
    message: str
    suggested_fix: str


@dataclass
class PreviewReport:
    ok: bool
    rendered_frames: list[int] = field(default_factory=list)
    issues: list[PreviewIssue] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def block_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "block")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rendered_frames": self.rendered_frames,
            "block": self.block_count,
            "warn": self.warn_count,
            "skipped_reason": self.skipped_reason,
            "issues": [asdict(i) for i in self.issues],
        }


_RENDER_REPAIRABLE_BLOCK_RULES = {
    "insufficient_visible_motion",
}


def preview_blocks_allow_render_repair(report: PreviewReport) -> bool:
    """Return True when preview blocks should enter render+critic repair.

    Missing frames, black frames, unreadable PNGs, and Blender errors are hard
    pre-render failures. Weak visible motion is different: it is a semantic
    quality signal that the critic/retry loop can often repair better than an
    iter00 early exit.
    """
    block_rule_ids = {
        issue.rule_id
        for issue in report.issues
        if issue.severity == "block"
    }
    return bool(block_rule_ids) and block_rule_ids <= _RENDER_REPAIRABLE_BLOCK_RULES


def select_preview_frames(storyboard: Storyboard) -> list[int]:
    frames: list[int] = []
    cursor = 0
    for shot in storyboard.shots:
        n = max(1, round(shot.duration_sec * storyboard.fps))
        start = cursor + 1
        end = cursor + n
        if shot.duration_sec > 2.0 and n >= 6:
            frames.append(int(start + 0.25 * (end - start)))
            frames.append(int(start + 0.75 * (end - start)))
        else:
            frames.append(int(start + 0.5 * (end - start)))
        cursor = end
    return sorted(set(frames))


def verify_preview_frames(
    frames_dir: Path,
    expected_frames: list[int],
    *,
    scene_profile: SceneProfile | None = None,
    storyboard: Storyboard | None = None,
) -> PreviewReport:
    issues: list[PreviewIssue] = []
    rendered: list[int] = []
    is_cinematic = (
        scene_profile is not None
        and scene_profile.base_profile == "cinematic_application"
    )
    for frame in expected_frames:
        path = frames_dir / f"frame_{frame:04d}.png"
        if not path.exists():
            issues.append(PreviewIssue(
                severity="block",
                rule_id="missing_preview_frame",
                frame_idx=frame,
                message=f"Preview frame {frame:04d} was not rendered.",
                suggested_fix="Ensure scene.py honors CG_TUTOR_PREVIEW_FRAMES and writes frame_####.png.",
            ))
            continue
        rendered.append(frame)
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                stat = ImageStat.Stat(img)
                mean = sum(stat.mean) / 3.0
                extrema = img.getextrema()
                dynamic = max(hi - lo for lo, hi in extrema)
                edge_risk = _edge_activity_risk(img)
                overlay_risk = _overlay_occupancy_risk(
                    img, frame, storyboard,
                )
        except Exception as e:  # noqa: BLE001
            issues.append(PreviewIssue(
                severity="block",
                rule_id="unreadable_preview_frame",
                frame_idx=frame,
                message=f"Preview frame {frame:04d} could not be read: {e}",
                suggested_fix="Render valid PNG files.",
            ))
            continue
        if mean < 8:
            if is_cinematic:
                if mean < 4 and dynamic < 4:
                    issues.append(PreviewIssue(
                        severity="block",
                        rule_id="near_black_frame",
                        frame_idx=frame,
                        message=f"Preview frame {frame:04d} is nearly black.",
                        suggested_fix=(
                            "Add a little more fill/key light or ensure the "
                            "scene includes at least one visible luminous cue."
                        ),
                    ))
                else:
                    issues.append(PreviewIssue(
                        severity="warn",
                        rule_id="dim_but_structured_frame",
                        frame_idx=frame,
                        message=(
                            f"Preview frame {frame:04d} is dim but has visible "
                            "structure."
                        ),
                        suggested_fix=(
                            "If this is a cinematic scene, the darkness is probably "
                            "intentional; otherwise raise fill/key light slightly."
                        ),
                    ))
            elif dynamic < 12:
                issues.append(PreviewIssue(
                    severity="block",
                    rule_id="near_black_frame",
                    frame_idx=frame,
                    message=f"Preview frame {frame:04d} is nearly black.",
                    suggested_fix="Add/raise key or ambient lighting before full render.",
                ))
            else:
                issues.append(PreviewIssue(
                    severity="warn",
                    rule_id="dim_but_structured_frame",
                    frame_idx=frame,
                    message=(
                        f"Preview frame {frame:04d} is dim but has visible "
                        "structure."
                    ),
                    suggested_fix=(
                        "If this is a cinematic scene, the darkness is probably "
                        "intentional; otherwise raise fill/key light slightly."
                    ),
                ))
        elif mean > 247:
            issues.append(PreviewIssue(
                severity="warn",
                rule_id="near_white_frame",
                frame_idx=frame,
                message=f"Preview frame {frame:04d} is nearly white.",
                suggested_fix="Lower light/material intensity to avoid blown-out teaching cues.",
            ))
        if dynamic < 6:
            issues.append(PreviewIssue(
                severity="warn",
                rule_id="low_contrast_frame",
                frame_idx=frame,
                message=f"Preview frame {frame:04d} has very low contrast.",
                suggested_fix="Increase contrast between background and teaching objects.",
            ))
        if not is_cinematic and edge_risk:
            issues.append(PreviewIssue(
                severity="warn",
                rule_id="teaching_object_near_frame_edge",
                frame_idx=frame,
                message=(
                    f"Preview frame {frame:04d} has bright/structured content "
                    "touching the frame edge; labels or arrows may be clipped."
                ),
                suggested_fix=(
                    "Keep labels, counters, arrows, and hero objects inside an "
                    "8% screen-safe margin; pull helper text inward, reduce "
                    "its size, or widen/shift the camera framing."
                ),
            ))
        if not is_cinematic and overlay_risk:
            issues.append(PreviewIssue(
                severity="warn",
                rule_id="overlay_zone_occupied_before_compose",
                frame_idx=frame,
                message=(
                    f"Preview frame {frame:04d} already has visible scene "
                    "content inside the formula overlay zone."
                ),
                suggested_fix=(
                    "Keep Blender labels, counters, helper geometry, and hero "
                    "objects outside shot.overlay_zone; formula text is added "
                    "later by ffmpeg and needs a clean reserved area."
                ),
            ))
        if is_cinematic and mean < 8 and dynamic >= 4:
            issues.append(PreviewIssue(
                severity="warn",
                rule_id="cinematic_low_light_frame",
                frame_idx=frame,
                message=(
                    f"Preview frame {frame:04d} is dark, but this may be "
                    "intentional for a cinematic application scene."
                ),
                suggested_fix=(
                    "Keep it if the scene is meant to be dark; otherwise add "
                    "slightly more fill/key light."
                ),
            ))
    motion_mode = _visible_motion_mode(scene_profile, storyboard)
    if motion_mode is not None:
        issues.extend(_visible_motion_issues(
            frames_dir,
            rendered,
            storyboard,
            mode=motion_mode,
        ))
    return PreviewReport(
        ok=not any(i.severity == "block" for i in issues),
        rendered_frames=rendered,
        issues=issues,
    )


def _visible_motion_mode(
    scene_profile: SceneProfile | None,
    storyboard: Storyboard | None,
) -> str | None:
    if scene_profile is None or storyboard is None:
        return None
    if storyboard.total_duration <= 2.0:
        return None
    if scene_profile.base_profile in {"transformation_demo", "curve_construction"}:
        return "per_shot"
    if scene_profile.base_profile == "vector_teaching":
        if _storyboard_has_object_keyframes(storyboard):
            return "per_shot"
        return "whole_preview"
    return None


def _storyboard_has_object_keyframes(storyboard: Storyboard) -> bool:
    for shot in storyboard.shots:
        for obj in shot.objects:
            for key in obj.keyframes:
                if key.attr not in {"hide_render", "hide_viewport"}:
                    return True
    return False


def _visible_motion_issues(
    frames_dir: Path,
    rendered: list[int],
    storyboard: Storyboard,
    *,
    mode: str,
) -> list[PreviewIssue]:
    if mode == "whole_preview":
        return _whole_preview_motion_issues(frames_dir, rendered)

    by_shot: dict[str, list[int]] = {}
    rendered_set = set(rendered)
    for frame in rendered:
        shot = _shot_for_frame(storyboard, frame)
        if shot is None or shot.duration_sec <= 2.0:
            continue
        by_shot.setdefault(shot.node_id, []).append(frame)

    diffs: list[tuple[str, int, int, float]] = []
    for shot_id, frames in by_shot.items():
        available = [f for f in sorted(frames) if f in rendered_set]
        if len(available) < 2:
            continue
        a, b = available[0], available[-1]
        diff = _mean_frame_diff(
            frames_dir / f"frame_{a:04d}.png",
            frames_dir / f"frame_{b:04d}.png",
        )
        if diff is not None:
            diffs.append((shot_id, a, b, diff))
    if not diffs:
        return []

    max_diff = max(diff for *_rest, diff in diffs)
    if max_diff >= 2.0:
        return []
    details = "; ".join(
        f"{shot_id} {a:04d}->{b:04d} mean_diff={diff:.2f}"
        for shot_id, a, b, diff in diffs[:4]
    )
    return [PreviewIssue(
        severity="block",
        rule_id="insufficient_visible_motion",
        frame_idx=None,
        message=(
            "Preview frames show too little visible change for a dynamic "
            f"teaching scene ({details})."
        ),
        suggested_fix=(
            "Increase the visible animation amplitude in the main frame: "
            "animate the render camera and/or teaching anchors with clear "
            "position, lens, scale, shape, curve reveal, or material changes. "
            "Do not rely only on hide_render shot gating or tiny inset motion."
        ),
    )]


def _whole_preview_motion_issues(
    frames_dir: Path,
    rendered: list[int],
) -> list[PreviewIssue]:
    """For vector teaching, allow static per-shot diagrams.

    A stepwise teaching scene such as Phong lighting may deliberately hold each
    shot still while changing the diagram between shots. Block only when the
    entire preview is visually static.
    """
    frames = sorted(rendered)
    if len(frames) < 2:
        return []
    diffs: list[tuple[int, int, float]] = []
    for a, b in zip(frames, frames[1:]):
        diff = _mean_frame_diff(
            frames_dir / f"frame_{a:04d}.png",
            frames_dir / f"frame_{b:04d}.png",
        )
        if diff is not None:
            diffs.append((a, b, diff))
    if not diffs:
        return []
    max_diff = max(diff for *_rest, diff in diffs)
    if max_diff >= 2.0:
        return []
    details = "; ".join(
        f"{a:04d}->{b:04d} mean_diff={diff:.2f}"
        for a, b, diff in diffs[:4]
    )
    return [PreviewIssue(
        severity="block",
        rule_id="insufficient_visible_motion",
        frame_idx=None,
        message=(
            "Preview frames show too little visible change across the whole "
            f"vector-teaching scene ({details})."
        ),
        suggested_fix=(
            "Make at least one visible teaching state change across the preview: "
            "reveal vectors, change lighting components, move a cue, or switch "
            "component/composite geometry. Static per-shot diagrams are allowed, "
            "but the full sequence cannot be visually identical."
        ),
    )]


def _mean_frame_diff(path_a: Path, path_b: Path) -> float | None:
    try:
        with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
            a = img_a.convert("RGB").resize((160, 90))
            b = img_b.convert("RGB").resize((160, 90))
            diff = ImageChops.difference(a, b)
            stat = ImageStat.Stat(diff)
            return float(sum(stat.mean) / 3.0)
    except Exception:  # noqa: BLE001
        return None


def _edge_activity_risk(img: Image.Image) -> bool:
    w, h = img.size
    margin = max(2, int(min(w, h) * 0.04))
    gray = img.convert("L")
    center = gray.crop((margin, margin, w - margin, h - margin))
    edge = Image.new("L", (w, h), 0)
    edge.paste(gray.crop((0, 0, w, margin)), (0, 0))
    edge.paste(gray.crop((0, h - margin, w, h)), (0, h - margin))
    edge.paste(gray.crop((0, 0, margin, h)), (0, 0))
    edge.paste(gray.crop((w - margin, 0, w, h)), (w - margin, 0))
    center_mean = ImageStat.Stat(center).mean[0]
    edge_stat = ImageStat.Stat(edge)
    edge_mean = edge_stat.mean[0]
    edge_max = edge.getextrema()[1]
    return edge_max > max(90, center_mean + 45) and edge_mean > 1.0


def _overlay_occupancy_risk(
    img: Image.Image,
    frame: int,
    storyboard: Storyboard | None,
) -> bool:
    if storyboard is None:
        return False
    shot = _shot_for_frame(storyboard, frame)
    if shot is None or shot.overlay_zone is None or not shot.formula:
        return False
    w, h = img.size
    z = shot.overlay_zone
    x0 = max(0, min(w - 1, int(z.x * w)))
    y0 = max(0, min(h - 1, int(z.y * h)))
    x1 = max(x0 + 1, min(w, int((z.x + z.w) * w)))
    y1 = max(y0 + 1, min(h, int((z.y + z.h) * h)))
    crop = img.convert("L").crop((x0, y0, x1, y1))
    stat = ImageStat.Stat(crop)
    mean = stat.mean[0]
    extrema = crop.getextrema()
    return mean > 14 and (extrema[1] - extrema[0]) > 35


def _shot_for_frame(storyboard: Storyboard, frame: int):
    for shot in storyboard.shots:
        start = round(shot.start_sec * storyboard.fps) + 1
        end = start + round(shot.duration_sec * storyboard.fps) - 1
        if start <= frame <= end:
            return shot
    return None


def preview_report_to_json(report: PreviewReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
