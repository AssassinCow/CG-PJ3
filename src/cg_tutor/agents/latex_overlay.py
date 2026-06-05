"""LaTeX Overlay Agent (deterministic).

Reads each shot's `formula` and renders a transparent PNG per shot.
Naming: ``<out_dir>/overlays/shot_<idx>_<node_id>.png``.

Return order matches ``storyboard.shots``; entries are ``None`` for
shots without a formula so callers can ``zip(shots, paths)`` and skip.
"""

from __future__ import annotations

from pathlib import Path

from cg_tutor.composer.formula_render import render_formula_to_png
from cg_tutor.schemas import Storyboard


def render_overlays(storyboard: Storyboard, out_dir: Path) -> list[Path | None]:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path | None] = []
    for idx, shot in enumerate(storyboard.shots):
        if not shot.formula:
            paths.append(None)
            continue
        out_path = overlay_dir / f"shot_{idx:02d}_{shot.node_id}.png"
        render_formula_to_png(shot.formula, out_path)
        paths.append(out_path)
    return paths
