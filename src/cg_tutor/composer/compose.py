"""High-level compose step: storyboard + frames → final mp4.

Encapsulates the overlay-spec building + ffmpeg call that both
``pipeline.py`` (W3 critic loop) and ``scripts/continue_critic.py``
were duplicating.
"""

from __future__ import annotations

from pathlib import Path

from cg_tutor.agents.latex_overlay import render_overlays
from cg_tutor.composer.ffmpeg_wrapper import ComposeResult, OverlaySpec, frames_to_mp4
from cg_tutor.schemas import Storyboard


def compose_storyboard_video(
    storyboard: Storyboard,
    frames_dir: Path,
    out_dir: Path,
    *,
    pattern: str = "frame_%04d.png",
    out_name: str = "final.mp4",
) -> ComposeResult:
    """Render per-shot formula PNGs, build OverlaySpecs gated by shot
    timing, and call ffmpeg. Returns the ComposeResult; the caller can
    inspect ``ok`` and ``output`` to handle errors.
    """
    overlay_paths = render_overlays(storyboard, out_dir)
    overlay_specs = _build_overlay_specs(storyboard, overlay_paths)
    return frames_to_mp4(
        frames_dir,
        out_dir / out_name,
        fps=storyboard.fps,
        pattern=pattern,
        overlays=overlay_specs,
    )


def _build_overlay_specs(
    storyboard: Storyboard,
    overlay_paths: list[Path | None],
) -> list[OverlaySpec]:
    specs: list[OverlaySpec] = []
    res_w, res_h = storyboard.resolution
    cursor = 0.0
    for shot, png in zip(storyboard.shots, overlay_paths):
        start, end = cursor, cursor + shot.duration_sec
        cursor = end
        if png is None or shot.overlay_zone is None:
            continue
        oz = shot.overlay_zone
        x, y, width = _safe_overlay_geometry(
            res_w, res_h, oz.x, oz.y, oz.w,
        )
        specs.append(OverlaySpec(
            png=png,
            xy=(x, y),
            width=width,
            start_sec=start,
            end_sec=end,
        ))
    return specs


def _safe_overlay_geometry(
    res_w: int,
    res_h: int,
    x_norm: float,
    y_norm: float,
    w_norm: float,
) -> tuple[int, int, int]:
    """Clamp formula overlays into a conservative screen-safe area.

    Storyboard overlay zones are model-generated and occasionally hug frame
    edges. The rendered formula PNG is scaled by width in ffmpeg, so clamping
    x/y/width here prevents off-screen formula crops even when the storyboard
    selected an aggressive zone.
    """
    margin_x = max(1, int(res_w * 0.06))
    margin_y = max(1, int(res_h * 0.06))
    max_width = max(1, res_w - 2 * margin_x)
    width = max(1, min(int(w_norm * res_w), max_width))
    x = int(x_norm * res_w)
    y = int(y_norm * res_h)
    x = max(margin_x, min(x, res_w - margin_x - width))
    y = max(margin_y, min(y, res_h - margin_y))
    return x, y, width
