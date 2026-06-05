"""Static checks for concept YAML files.

These catch configuration-level contradictions before an expensive API/Blender
run, such as asking a cinematic no-formula profile to render formulas.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cg_tutor.scene_profiles import BASE_PROFILES


ROOT = Path(__file__).resolve().parents[1]
CONCEPT_DIR = ROOT / "configs" / "concepts"


def _concept_specs() -> list[tuple[Path, dict]]:
    specs: list[tuple[Path, dict]] = []
    for path in sorted(CONCEPT_DIR.glob("*.yaml")):
        specs.append((path, yaml.safe_load(path.read_text()) or {}))
    return specs


def test_concept_ids_match_filenames_and_required_fields():
    for path, spec in _concept_specs():
        assert spec.get("concept_id") == path.stem
        assert spec.get("title")
        assert spec.get("target_audience")
        assert float(spec.get("duration_sec", 0)) > 0
        assert spec.get("key_points")


def test_concept_scene_profiles_are_valid_and_formula_compatible():
    valid_profiles = set(BASE_PROFILES) | {"auto", None}
    for path, spec in _concept_specs():
        profile = spec.get("scene_profile")
        assert profile in valid_profiles, f"{path.name}: invalid scene_profile={profile!r}"
        formulas = spec.get("must_show_formulas") or []
        if formulas:
            assert profile != "cinematic_application", (
                f"{path.name}: cinematic_application disables formula overlays; "
                "use a teaching profile or remove must_show_formulas"
            )
            assert spec.get("presentation_mode") != "application_scene", (
                f"{path.name}: application_scene should communicate through the "
                "scene, not required formula overlays"
            )
