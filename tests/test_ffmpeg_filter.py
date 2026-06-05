"""Unit tests for ffmpeg filter_complex construction.

We don't run ffmpeg here; we only verify that the string passed to
`-filter_complex` is shaped correctly for 0, 1, and N overlay specs
with the various time-gating cases.
"""

from __future__ import annotations

from pathlib import Path

from cg_tutor.composer.compose import _build_overlay_specs
from cg_tutor.composer.ffmpeg_wrapper import OverlaySpec, _build_overlay_filter
from cg_tutor.schemas import Storyboard


def _ovl(start, end, *, width=200, xy=(10, 20), name="o.png"):
    return OverlaySpec(png=Path(name), xy=xy, width=width,
                       start_sec=start, end_sec=end)


def test_overlay_specs_skip_application_shot_without_formula_overlay():
    storyboard = Storyboard.model_validate({
        "concept_id": "heist_demo",
        "fps": 24,
        "resolution": [960, 540],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 2.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -5, 3],
                "look_at": [0, 0, 0],
                "fov": 50,
            }],
            "objects": [{
                "name": "key_light",
                "type": "light",
                "location": [0, -3, 4],
            }],
            "overlay_zone": None,
            "formula": None,
        }],
    })

    assert _build_overlay_specs(storyboard, [None]) == []
    assert _build_overlay_specs(storyboard, [Path("formula.png")]) == []


def test_overlay_specs_clamp_formula_to_screen_safe_area():
    storyboard = Storyboard.model_validate({
        "concept_id": "safe_overlay",
        "fps": 24,
        "resolution": [1000, 500],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 2.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -5, 3],
                "look_at": [0, 0, 0],
                "fov": 50,
            }],
            "objects": [{
                "name": "hero",
                "type": "mesh",
                "primitive": "cube",
            }],
            "overlay_zone": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.2},
            "formula": "a=b+c",
        }],
    })

    specs = _build_overlay_specs(storyboard, [Path("formula.png")])

    assert len(specs) == 1
    assert specs[0].xy == (60, 30)
    assert specs[0].width == 880


def test_single_overlay_bounded_window():
    f = _build_overlay_filter([_ovl(1.0, 4.0)])
    assert "[1:v]scale=200:-1[ovl1]" in f
    assert "[0:v][ovl1]overlay=10:20:enable='between(t,1.0,4.0)'[v1]" in f


def test_overlay_without_width_passes_null_filter():
    f = _build_overlay_filter([_ovl(0.0, 1.0, width=None)])
    assert "[1:v]null[ovl1]" in f
    # start_sec == 0 with end_sec set → still uses between()
    assert "enable='between(t,0.0,1.0)'" in f


def test_overlay_unbounded_uses_gte():
    """end_sec=None means 'until end of clip' → enable='gte(t,start)'."""
    f = _build_overlay_filter([OverlaySpec(
        png=Path("o.png"), xy=(0, 0), width=100,
        start_sec=2.5, end_sec=None,
    )])
    assert "enable='gte(t,2.5)'" in f


def test_overlay_starts_at_zero_unbounded_has_no_gate():
    f = _build_overlay_filter([OverlaySpec(
        png=Path("o.png"), xy=(0, 0), width=100,
        start_sec=0.0, end_sec=None,
    )])
    # neither gate present
    assert "enable=" not in f


def test_multiple_overlays_chain_through_intermediate_streams():
    f = _build_overlay_filter([
        _ovl(0.0, 2.0, name="a.png"),
        _ovl(2.0, 4.0, name="b.png", xy=(5, 5), width=300),
    ])
    # First overlay consumes 0:v, produces v1
    assert "[0:v][ovl1]overlay=10:20" in f
    # Second overlay consumes v1, produces v2
    assert "[v1][ovl2]overlay=5:5" in f
    # Both overlays scaled
    assert "[1:v]scale=200:-1[ovl1]" in f
    assert "[2:v]scale=300:-1[ovl2]" in f
