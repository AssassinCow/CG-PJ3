"""Render LaTeX math to a transparent PNG using matplotlib's mathtext.

Mathtext covers most of inline LaTeX without a TeX install. For broader
support, swap in `manim` or a real LaTeX pipeline later.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_formula_to_png(
    formula: str,
    out_path: Path,
    *,
    fontsize: int = 14,
    fg: str = "white",
    dpi: int = 150,
    pad_inches: float = 0.02,
) -> Path:
    """Render a LaTeX-style formula to a transparent PNG.

    matplotlib's `mathtext` is a strict subset of LaTeX — it lacks
    `\\begin{bmatrix}...\\end{bmatrix}`, multi-line aligns, and most
    package-specific commands. If mathtext refuses the input, we fall
    back to rendering the formula's LaTeX source as plain monospace text
    so the overlay is informative rather than absent.

    Final on-screen size is *also* constrained by the ffmpeg compositor —
    it rescales this PNG to fit the storyboard's `overlay_zone.w`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _try_render(formula, out_path, fontsize=fontsize, fg=fg,
                dpi=dpi, pad_inches=pad_inches, mathtext=True) or \
        _try_render(_simplify_to_plain(formula), out_path,
                    fontsize=fontsize, fg=fg, dpi=dpi,
                    pad_inches=pad_inches, mathtext=False)
    return out_path


def _try_render(text: str, out_path: Path, *, fontsize: int, fg: str,
                dpi: int, pad_inches: float, mathtext: bool) -> bool:
    fig_w = _estimate_fig_width(text)
    fig_h = 0.85 if len(text) < 80 else 1.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)
    body = f"${text}$" if mathtext else text
    kwargs = {
        "fontsize": fontsize,
        "color": fg,
        "ha": "left",
        "va": "center",
        "clip_on": False,
    }
    if not mathtext:
        kwargs["family"] = "monospace"
    fig.text(0.03, 0.5, body, **kwargs)
    try:
        fig.savefig(out_path, dpi=dpi, transparent=True,
                    bbox_inches="tight", pad_inches=pad_inches)
        return True
    except (ValueError, RuntimeError):
        return False
    finally:
        plt.close(fig)


def _estimate_fig_width(text: str) -> float:
    """Give mathtext enough canvas to avoid clipping before ffmpeg scaling."""
    stripped = text.replace("\\", "").replace("{", "").replace("}", "")
    # Long formulas are later scaled down to the storyboard overlay width; a
    # wider source canvas is preferable to clipped glyphs at the PNG edge.
    return min(9.0, max(4.0, 0.075 * len(stripped) + 1.2))


def _simplify_to_plain(formula: str) -> str:
    """Last-ditch plain-text rendering for formulas mathtext can't parse.

    Strips the most common LaTeX commands so the result reads naturally,
    then surrounds with brackets to signal the math context. We do NOT
    try to make this pretty — the goal is "viewer can still tell what
    formula it is", not "publication quality".
    """
    import re
    s = formula
    # bracket constructions
    s = re.sub(r"\\begin\{[bp]?matrix\}", "[", s)
    s = re.sub(r"\\end\{[bp]?matrix\}", "]", s)
    s = re.sub(r"\\begin\{[a-zA-Z*]+\}", "", s)
    s = re.sub(r"\\end\{[a-zA-Z*]+\}", "", s)
    # font / formatting commands -> argument only
    s = re.sub(r"\\(mathbf|mathrm|mathit|text|operatorname)\{([^{}]*)\}",
               r"\2", s)
    # common operators
    s = (s.replace("\\cdot", "·")
           .replace("\\times", "x")
           .replace("\\theta", "θ")
           .replace("\\sin", "sin").replace("\\cos", "cos")
           .replace("\\sum", "Σ").replace("\\int", "∫")
           .replace("\\\\", "; ")   # row break
           .replace("&", " "))
    s = re.sub(r"\\[a-zA-Z]+", "", s)   # strip remaining commands
    s = re.sub(r"\s+", " ", s).strip()
    return s
