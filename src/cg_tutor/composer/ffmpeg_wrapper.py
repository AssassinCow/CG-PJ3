"""Compose rendered Blender frames into an mp4.

Why ffmpeg instead of letting Blender output mp4 directly?
- Frame-level overlays (formulas, subtitles) are easier as a second pass.
- Easier to swap encoders / try drawtext filter for captions.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegNotFound(RuntimeError):
    pass


@dataclass
class ComposeResult:
    returncode: int
    output: Path
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output.exists()


@dataclass
class OverlaySpec:
    """A single time-gated overlay layer for the compositor."""
    png: Path
    xy: tuple[int, int]
    width: int | None = None   # scale overlay to this px width before placing
    start_sec: float = 0.0
    end_sec: float | None = None   # None = "until end of clip"


def _resolve_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise FfmpegNotFound("ffmpeg not on PATH (sudo apt install ffmpeg)")
    return found


def frames_to_mp4(
    frames_dir: Path,
    out_path: Path,
    *,
    fps: int = 30,
    pattern: str = "frame_%04d.png",
    overlays: list[OverlaySpec] | None = None,
    # Legacy single-overlay kwargs (still honoured by upgrading to list).
    overlay_png: Path | None = None,
    overlay_xy: tuple[int, int] | None = None,
    overlay_width: int | None = None,
    crf: int = 20,
) -> ComposeResult:
    """Encode frames matching `pattern` into mp4, with optional overlays.

    `overlays` is a list of OverlaySpec. Each overlay is scaled (if
    `width` given) and composited at `xy`, time-gated to its
    [start_sec, end_sec) window via ffmpeg's `enable` expression.

    Backward compatibility: the older `overlay_png/xy/width` kwargs
    behave as a single overlay covering the full clip.
    """
    ffmpeg = _resolve_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overlays = list(overlays or [])
    if overlay_png and overlay_xy is not None:
        overlays.append(OverlaySpec(
            png=overlay_png, xy=overlay_xy, width=overlay_width,
        ))

    cmd = [ffmpeg, "-y", "-framerate", str(fps),
           "-i", str(frames_dir / pattern)]
    for ov in overlays:
        cmd += ["-i", str(ov.png)]

    if overlays:
        cmd += ["-filter_complex", _build_overlay_filter(overlays)]
        cmd += ["-map", f"[v{len(overlays)}]"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            str(out_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return ComposeResult(
        returncode=proc.returncode, output=out_path, stderr=proc.stderr,
    )


def _build_overlay_filter(overlays: list[OverlaySpec]) -> str:
    """Build an ffmpeg `-filter_complex` string for N time-gated overlays.

    Stream 0 is the frame sequence; streams 1..N are the overlay PNGs.
    Each overlay is first scaled (if requested) into [ovlk], then
    composited onto the running [vk-1] producing [vk] with an `enable`
    gate. Final output stream is [vN].
    """
    parts: list[str] = []
    for k, ov in enumerate(overlays, start=1):
        if ov.width:
            parts.append(f"[{k}:v]scale={ov.width}:-1[ovl{k}]")
        else:
            parts.append(f"[{k}:v]null[ovl{k}]")

    prev_label = "0:v"
    for k, ov in enumerate(overlays, start=1):
        if ov.end_sec is not None:
            gate = f":enable='between(t,{ov.start_sec},{ov.end_sec})'"
        elif ov.start_sec > 0:
            gate = f":enable='gte(t,{ov.start_sec})'"
        else:
            gate = ""
        parts.append(
            f"[{prev_label}][ovl{k}]overlay={ov.xy[0]}:{ov.xy[1]}{gate}[v{k}]"
        )
        prev_label = f"v{k}"
    return ";".join(parts)
