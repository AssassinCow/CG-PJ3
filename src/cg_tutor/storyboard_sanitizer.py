"""Deterministic storyboard cleanup before code generation.

Storyboard objects are meant to describe 3D scene geometry. Formula overlays
are handled later by the compositor from ``shot.formula`` and
``shot.overlay_zone``. If the LLM also adds ``formula_panel_*`` text/planes to
the 3D storyboard, the final video can show doubled or garbled formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import SceneProfile


_FORMULA_OBJECT_MARKERS = (
    "formula",
    "equation",
    "latex",
    "math_overlay",
    "formula_panel",
)


@dataclass
class StoryboardSanitizationReport:
    removed_formula_objects: dict[str, list[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return any(self.removed_formula_objects.values())

    def to_dict(self) -> dict:
        return {
            "changed": self.changed,
            "removed_formula_objects": self.removed_formula_objects,
        }


def sanitize_storyboard_for_pipeline(
    storyboard: Storyboard,
    *,
    scene_profile: SceneProfile | None = None,
) -> tuple[Storyboard, StoryboardSanitizationReport]:
    """Remove deterministic non-geometry clutter from storyboard objects."""
    report = StoryboardSanitizationReport()
    if not _should_remove_formula_objects(scene_profile):
        return storyboard, report

    raw = storyboard.model_dump(mode="json")
    for shot in raw.get("shots", []):
        kept = []
        removed = []
        for obj in shot.get("objects", []):
            name = str(obj.get("name", ""))
            kind = str(obj.get("type", ""))
            primitive = str(obj.get("primitive", ""))
            if _is_formula_object(name, kind, primitive):
                removed.append(name)
            else:
                kept.append(obj)
        if removed:
            shot["objects"] = kept
            report.removed_formula_objects[str(shot.get("node_id", ""))] = removed

    if not report.changed:
        return storyboard, report
    return Storyboard.model_validate(raw), report


def _should_remove_formula_objects(scene_profile: SceneProfile | None) -> bool:
    if scene_profile is None:
        return True
    return scene_profile.base_profile in {
        "vector_teaching",
        "curve_construction",
        "transformation_demo",
    }


def _is_formula_object(name: str, kind: str, primitive: str) -> bool:
    text = " ".join([name, kind, primitive]).lower()
    return any(marker in text for marker in _FORMULA_OBJECT_MARKERS)
