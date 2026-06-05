"""Formula renderer fallback: mathtext-compatible inputs go through
the LaTeX path; macros mathtext can't parse (like \\begin{bmatrix})
go through the plain-text simplifier."""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip everything if matplotlib isn't installed in this venv.
matplotlib = pytest.importorskip("matplotlib")

from cg_tutor.composer.formula_render import (  # noqa: E402
    _simplify_to_plain,
    render_formula_to_png,
)


def test_simplify_strips_bmatrix_and_keeps_structure():
    formula = r"\mathbf{T}(\mathbf{t}) = \begin{bmatrix} I & t \\ 0 & 1 \end{bmatrix}"
    out = _simplify_to_plain(formula)
    # mathbf wrapper gone, bmatrix → [ ... ], row sep ; , col sep space
    assert "T(t)" in out
    assert out.startswith("T(t)") or "T(t)" in out
    assert "[" in out and "]" in out
    assert ";" in out  # row break
    assert "\\" not in out  # no leftover LaTeX commands


def test_simplify_translates_common_unicode_operators():
    out = _simplify_to_plain(r"a \cdot b + \theta + \sin x")
    assert "·" in out
    assert "θ" in out
    assert "sin" in out


def test_simplify_idempotent_on_plain_text():
    out = _simplify_to_plain("just text")
    assert out == "just text"


def test_render_plain_mathtext_formula(tmp_path: Path):
    out = render_formula_to_png(r"a = b + c", tmp_path / "ok.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_unsupported_formula_falls_back_to_plain_text(tmp_path: Path):
    """Matplotlib mathtext rejects \\begin{bmatrix}; the renderer must
    fall back rather than raise."""
    formula = r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}"
    out = render_formula_to_png(formula, tmp_path / "fallback.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_long_formula_has_transparent_safe_margin(tmp_path: Path):
    from PIL import Image

    formula = r"E = \frac{V_1 + V_2 + F_1 + F_2}{4} \quad \text{(edge point)}"
    out = render_formula_to_png(formula, tmp_path / "long.png")

    with Image.open(out) as img:
        alpha = img.convert("RGBA").getchannel("A")
        w, h = alpha.size
        left = alpha.crop((0, 0, 1, h)).getbbox()
        right = alpha.crop((w - 1, 0, w, h)).getbbox()

    assert left is None
    assert right is None
