"""Scene-level visual policy profiles.

Profiles are a compact style contract shared by storyboard generation,
coding, verification, critic feedback, and repair.  They intentionally use
plain string lists so LLM-generated profiles can be sanitized safely.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


BASE_PROFILE_IDS = {
    "vector_teaching",
    "cinematic_application",
    "curve_construction",
    "transformation_demo",
}

_RUBRIC_LIST_KEYS = (
    "must_have",
    "must_avoid",
    "aesthetic_goals",
    "blocking_conditions",
)
_MAX_RUBRIC_ITEMS = 6
_MAX_RUBRIC_CHARS = 180
_GROUNDING_LIST_KEYS = (
    "persistent_anchors",
    "spatial_relationships",
    "forbidden_abstractions",
)
_MAX_GROUNDING_ITEMS = 10
_MAX_GROUNDING_CHARS = 160

HELPER_GROUPS: dict[str, set[str]] = {
    "drawn_rays": {
        "thin_glowing_paths",
        "dotted_tracers",
        "ray_path_polylines",
        "schematic_light_paths",
        "thin_curve_polyline_tracers",
        "long_shadow_arrows",
    },
    "arrow_helpers": {
        "arrows",
        "thick_arrows",
        "arrow_primitives",
        "cone_arrowheads",
        "vector_shafts",
        "long_shadow_arrows",
        "large_axes",
        "vector_labels",
        "normal_vector_labels",
    },
    "visible_light_gizmos": {
        "visible_light_bars",
        "visible_area_light_rectangles",
        "foreground_light_panels",
        "light_bars",
        "light_panels",
        "softbox_panel",
        "floating_emissive_slabs",
    },
}


class SceneProfile(BaseModel):
    profile_id: str
    base_profile: Literal[
        "vector_teaching",
        "cinematic_application",
        "curve_construction",
        "transformation_demo",
    ]
    visual_goal: str = ""
    allowed_helpers: list[str] = Field(default_factory=list)
    forbidden_helpers: list[str] = Field(default_factory=list)
    preferred_geometry: list[str] = Field(default_factory=list)
    overlay_policy: dict[str, Any] = Field(default_factory=dict)
    critic_priorities: list[str] = Field(default_factory=list)
    critic_rubric: dict[str, Any] = Field(default_factory=dict)
    repair_policy: list[str] = Field(default_factory=list)
    persistent_anchors: list[str] = Field(default_factory=list)
    spatial_relationships: list[str] = Field(default_factory=list)
    forbidden_abstractions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SceneProfileResolution:
    profile: SceneProfile
    source: str
    raw: dict[str, Any] | None
    validation: dict[str, Any]


def normalized_helper_set(items: list[str]) -> set[str]:
    return {
        str(item).strip().lower().replace("-", "_").replace(" ", "_")
        for item in items
        if str(item).strip()
    }


def profile_forbids_helper_group(
    scene_profile: SceneProfile | None,
    group: str,
) -> bool:
    if scene_profile is None:
        return False
    group_items = HELPER_GROUPS.get(group)
    if not group_items:
        return False
    forbidden = normalized_helper_set(scene_profile.forbidden_helpers)
    return bool(forbidden & group_items)


def profile_allows_helper_group(
    scene_profile: SceneProfile | None,
    group: str,
) -> bool:
    if scene_profile is None:
        return False
    group_items = HELPER_GROUPS.get(group)
    if not group_items:
        return False
    allowed = normalized_helper_set(scene_profile.allowed_helpers)
    forbidden = normalized_helper_set(scene_profile.forbidden_helpers)
    return bool((allowed & group_items) - forbidden)


BASE_PROFILES: dict[str, dict[str, Any]] = {
    "vector_teaching": {
        "profile_id": "vector_teaching",
        "base_profile": "vector_teaching",
        "visual_goal": "Clear explanatory scene with readable vectors, labels, formulas, and teaching helpers.",
        "allowed_helpers": [
            "arrows",
            "vector_labels",
            "axes",
            "formula_overlay",
            "highlight_markers",
        ],
        "forbidden_helpers": [
            "unlabeled_vectors",
            "overlapping_formula_overlay",
        ],
        "preferred_geometry": [
            "simple_primitives",
            "stable_ground_plane",
            "readable_arrow_vectors",
        ],
        "overlay_policy": {
            "formula_style": "teaching_overlay",
            "avoid_center_occlusion": True,
        },
        "critic_priorities": [
            "make vectors and labels readable",
            "keep formula overlay clear",
            "preserve pedagogical helper geometry",
        ],
        "critic_rubric": {
            "must_have": [
                "readable teaching helpers requested by the storyboard",
                "clear formula overlay when a shot has a formula",
                "visible vectors/labels for explicit vector concepts",
            ],
            "must_avoid": [
                "unreadable labels",
                "formula overlay colliding with the main subject",
                "missing or merged vector arrows",
            ],
            "aesthetic_goals": [
                "clean educational composition",
                "stable camera and legible helper geometry",
            ],
            "blocking_conditions": [
                "a required formula/vector/label is absent or unreadable",
                "main teaching object is off-screen or occluded",
            ],
        },
        "repair_policy": [
            "vectors should be full visible arrows with shaft and head",
            "separate labels and helper arrows so each cue is readable",
        ],
    },
    "cinematic_application": {
        "profile_id": "cinematic_application",
        "base_profile": "cinematic_application",
        "visual_goal": "Cinematic object-focused application scene with minimal HUD helpers.",
        "allowed_helpers": [
            "thin_glowing_paths",
            "dotted_tracers",
            "small_hud_labels",
            "diegetic_hud",
            "subtle_highlight_markers",
        ],
        "forbidden_helpers": [
            "thick_arrows",
            "arrow_primitives",
            "formula_overlay",
            "math_formula_overlay",
            "explanatory_labels",
            "visible_light_bars",
            "visible_area_light_rectangles",
            "large_axes",
            "classroom_panels",
            "contribution_breakdown_tiles",
            "foreground_light_panels",
        ],
        "preferred_geometry": [
            "cinematic_subject",
            "glossy_surfaces",
            "natural_lighting",
            "thin_curve_polyline_tracers",
        ],
        "overlay_policy": {
            "formula_style": "none",
            "disable_formula_overlay": True,
            "allow_diegetic_hud": True,
            "avoid_center_occlusion": True,
        },
        "critic_priorities": [
            "preserve cinematic scene mood",
            "make graphics principles visible through scene phenomena",
            "avoid visible light gizmos and thick arrows",
            "avoid mathematical overlays and classroom explanation panels",
        ],
        "critic_rubric": {
            "must_have": [
                "cinematic application scene matching the concept setting",
                "graphics principles visible through natural scene phenomena",
                "thin tracers or diegetic HUD cues only when they support the scene",
            ],
            "must_avoid": [
                "mathematical formula overlays",
                "classroom explanation panels",
                "thick arrow primitives or arrowheads",
                "visible light bars or foreground light panels",
                "contribution breakdown tiles",
            ],
            "aesthetic_goals": [
                "premium cinematic mood",
                "object-focused composition",
                "sparse in-world HUD text",
            ],
            "blocking_conditions": [
                "the scene becomes a classroom diagram instead of an application scene",
                "formula overlays or contribution tiles dominate the frame",
                "required reflection/refraction/shadow cues are absent",
            ],
        },
        "repair_policy": [
            "replace arrows with thin glowing curve_polyline or dotted tracers",
            "hide visible light panels and use invisible light objects",
            "remove formula overlays and contribution breakdown tiles",
            "keep any text as sparse in-world HUD, not classroom labels",
        ],
        "persistent_anchors": [
            "recognizable environment frame",
            "main reflective/refractive subject",
            "practical light source",
            "background object that reveals reflection/refraction",
        ],
        "spatial_relationships": [
            "environment frame stays visible across shots",
            "reflective/refractive subject sits between camera and background cue",
            "practical light source illuminates the subject without appearing as a floating panel",
        ],
        "forbidden_abstractions": [
            "black void with isolated primitives",
            "blank glowing rectangles used as signs",
            "floating transparent spheres without an environment",
            "visible light panels replacing practical lights",
        ],
    },
    "curve_construction": {
        "profile_id": "curve_construction",
        "base_profile": "curve_construction",
        "visual_goal": "Curve construction scene with control structure and progressive tracing.",
        "allowed_helpers": [
            "control_points",
            "control_polygon",
            "curve_traces",
            "parameter_labels",
            "formula_overlay",
        ],
        "forbidden_helpers": [
            "unreadable_control_labels",
            "overlapping_curve_samples",
        ],
        "preferred_geometry": [
            "curve_polyline",
            "sphere_group_control_points",
            "thin_construction_lines",
        ],
        "overlay_policy": {
            "formula_style": "teaching_overlay",
            "avoid_center_occlusion": True,
        },
        "critic_priorities": [
            "keep control points visible",
            "show curve progression clearly",
        ],
        "critic_rubric": {
            "must_have": [
                "visible control points and control polygon when requested",
                "readable curve trace or construction progression",
                "parameter/formula overlay when the shot has a formula",
            ],
            "must_avoid": [
                "overlapping or hidden control points",
                "curve samples too dense to read",
                "formula overlay covering the construction",
            ],
            "aesthetic_goals": [
                "clean curve construction layout",
                "clear color separation between control structure and curve",
            ],
            "blocking_conditions": [
                "the curve or its control structure is missing",
                "construction stages are visually indistinguishable",
            ],
        },
        "repair_policy": [
            "preserve control polygon and curve trace",
            "separate labels from curve samples",
        ],
    },
    "transformation_demo": {
        "profile_id": "transformation_demo",
        "base_profile": "transformation_demo",
        "visual_goal": "Spatial transformation scene with grid, axes, before/after states, and clear motion.",
        "allowed_helpers": [
            "axes",
            "grid",
            "ghost_objects",
            "before_after_labels",
            "motion_trails",
            "formula_overlay",
        ],
        "forbidden_helpers": [
            "unlabeled_axes",
            "ambiguous_before_after_state",
        ],
        "preferred_geometry": [
            "grid_plane",
            "axis_arrows",
            "semi_transparent_ghosts",
        ],
        "overlay_policy": {
            "formula_style": "teaching_overlay",
            "avoid_center_occlusion": True,
        },
        "critic_priorities": [
            "preserve coordinate frame readability",
            "keep before and after objects distinct",
        ],
        "critic_rubric": {
            "must_have": [
                "readable coordinate/grid reference when requested",
                "distinct before and after states",
                "visible transformation motion or correspondence",
            ],
            "must_avoid": [
                "ambiguous before/after states",
                "missing axes or grid for coordinate demonstrations",
                "formula overlay colliding with transformed objects",
            ],
            "aesthetic_goals": [
                "clear spatial layout",
                "stable transformation comparison",
            ],
            "blocking_conditions": [
                "the transformation effect is not visible",
                "before and after objects cannot be distinguished",
            ],
        },
        "repair_policy": [
            "keep axes/grid visible when they explain the transformation",
            "separate before and after states spatially or by material",
        ],
    },
}


_HELPER_ALIASES = {
    "arrow": "arrows",
    "thick_arrow": "thick_arrows",
    "thick arrows": "thick_arrows",
    "arrow_primitive": "arrow_primitives",
    "arrow primitives": "arrow_primitives",
    "ray_arrow": "thick_arrows",
    "visible light bar": "visible_light_bars",
    "visible_light_bar": "visible_light_bars",
    "area_light_rect": "visible_area_light_rectangles",
    "area light rectangle": "visible_area_light_rectangles",
    "light panel": "foreground_light_panels",
    "foreground light panel": "foreground_light_panels",
    "thin line": "thin_glowing_paths",
    "thin tracer": "thin_glowing_paths",
    "dotted line": "dotted_tracers",
    "formula overlay": "formula_overlay",
    "formula_overlay": "formula_overlay",
    "math overlay": "math_formula_overlay",
    "mathematical overlay": "math_formula_overlay",
    "classroom panel": "classroom_panels",
    "classroom_panel": "classroom_panels",
    "explanatory label": "explanatory_labels",
    "explanatory_label": "explanatory_labels",
    "contribution tiles": "contribution_breakdown_tiles",
    "contribution_tiles": "contribution_breakdown_tiles",
    "contribution breakdown tiles": "contribution_breakdown_tiles",
    "contribution_breakdown_tiles": "contribution_breakdown_tiles",
}


def _norm_item(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = "_".join(part for part in text.split("_") if part)
    return _HELPER_ALIASES.get(text, text)


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _norm_item(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _clean_text_list(values: Any, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        key = text.lower()
        if key not in seen:
            out.append(text)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _sanitize_critic_rubric(
    value: Any,
    *,
    base_id: str,
    validation: dict[str, Any],
) -> dict[str, list[str]]:
    base = deepcopy(BASE_PROFILES[base_id]["critic_rubric"])
    if not isinstance(value, dict):
        validation["warnings"].append(
            "critic_rubric was not an object; using base critic rubric"
        )
        return base

    unknown = sorted(set(value) - set(_RUBRIC_LIST_KEYS))
    if unknown:
        validation["warnings"].append(
            "dropped unsupported critic_rubric key(s): " + ", ".join(unknown)
        )

    out: dict[str, list[str]] = {}
    for key in _RUBRIC_LIST_KEYS:
        cleaned = _clean_text_list(
            value.get(key),
            max_items=_MAX_RUBRIC_ITEMS,
            max_chars=_MAX_RUBRIC_CHARS,
        )
        if cleaned:
            out[key] = cleaned
        else:
            out[key] = list(base.get(key, []))
            if key in value:
                validation["warnings"].append(
                    f"critic_rubric.{key} was empty/invalid; using base values"
                )
    return out


def base_profile(profile_id: str) -> SceneProfile:
    if profile_id not in BASE_PROFILES:
        # Surface the silent fallback so unknown ids don't masquerade as
        # vector_teaching. Callers that catch the warning to a validation
        # log are still served by SceneProfile being returned.
        import warnings as _warnings
        _warnings.warn(
            f"scene_profile id={profile_id!r} not in BASE_PROFILES; "
            "falling back to 'vector_teaching'",
            stacklevel=2,
        )
        key = "vector_teaching"
    else:
        key = profile_id
    return SceneProfile.model_validate(deepcopy(BASE_PROFILES[key]))


def choose_base_profile(concept_spec: dict[str, Any]) -> str:
    text = json.dumps(concept_spec, ensure_ascii=False).lower()
    if str(concept_spec.get("presentation_mode", "")).lower() == "application_scene":
        return "cinematic_application"
    if any(word in text for word in ("cinematic", "showcase", "jewelry", "premium", "no thick arrow", "no classroom")):
        return "cinematic_application"
    if any(word in text for word in ("whitted", "ray tracing", "ray_tracing")) and any(
        word in text for word in ("thin", "dotted", "no arrow", "invisible light", "showcase")
    ):
        return "cinematic_application"
    if any(word in text for word in ("bezier", "curve", "spline")):
        return "curve_construction"
    if any(word in text for word in ("affine", "transformation", "matrix", "rotate", "scale", "translate")):
        return "transformation_demo"
    return "vector_teaching"


def _grounding_seed_from_concept(concept_spec: dict[str, Any]) -> dict[str, list[str]]:
    """Infer concrete application-scene anchors as deterministic fallback."""
    text = json.dumps(concept_spec, ensure_ascii=False).lower()
    anchors: list[str] = []
    relationships: list[str] = []
    forbidden: list[str] = []

    if "rainy_window" in text or "rainy window" in text:
        anchors.extend([
            "window_frame",
            "window_pane",
            "mullion",
            "raindrops_on_glass",
            "neon_sign_OPEN",
            "wet_sidewalk",
            "interior_sill",
        ])
        relationships.extend([
            "window_frame and mullion outline the glass pane in every shot",
            "raindrops sit on the window pane between camera and neon_sign_OPEN",
            "neon_sign_OPEN remains behind the glass so droplets can distort it",
            "wet_sidewalk reflects the neon color below the window",
        ])
        forbidden.extend([
            "OPEN sign replaced by a blank glowing rectangle",
            "raindrops floating in a black void",
            "missing window frame or glass boundary",
        ])

    if "windshield" in text or "车窗" in text or "挡风玻璃" in text:
        anchors.extend([
            "dashboard_silhouette",
            "windshield_frame",
            "wiper_blade",
            "raindrops_on_windshield",
            "wet_street",
            "neon_sign_OPEN",
            "headlight_glow",
        ])
        relationships.extend([
            "dashboard_silhouette stays in the lower foreground",
            "windshield_frame encloses the rainy glass plane in every shot",
            "raindrops sit on the windshield between camera and neon/background lights",
            "wiper_blade crosses or frames the glass without hiding the main cue",
        ])
        forbidden.extend([
            "car scene reduced to floating droplets and glowing rectangles",
            "missing dashboard or windshield frame",
            "signage represented only as blank emissive slabs",
        ])

    if "deep_sea" in text or "deep sea" in text or "submersible" in text:
        anchors.extend([
            "submersible_hull",
            "viewport_glass_dome",
            "seafloor_silt",
            "floating_particles",
            "blue_water_volume",
            "red_beacon",
            "distant_rock_wall",
        ])
        relationships.extend([
            "viewport_glass_dome is attached to the submersible_hull",
            "blue_water_volume and particles surround the subject in every shot",
            "red_beacon sits on or near the hull rather than floating alone",
            "seafloor_silt or rock wall gives depth behind the glass",
        ])
        forbidden.extend([
            "glass sphere floating in empty black space",
            "red cube or beacon without a submersible body",
            "missing water volume or seafloor context",
        ])

    if "water_glass" in text or "water glass" in text:
        anchors.extend([
            "table_surface",
            "glass_tumbler",
            "water_volume",
            "background_checker_card",
            "caustic_patch",
            "rim_highlight",
        ])
        relationships.extend([
            "glass_tumbler sits on table_surface throughout the scene",
            "background_checker_card is behind the glass to reveal distortion",
            "caustic_patch appears on the table below or near the glass",
        ])
        forbidden.extend([
            "single transparent cylinder without a table or background card",
            "refraction shown only as abstract rays",
        ])

    return {
        "persistent_anchors": _clean_text_list(
            anchors, max_items=_MAX_GROUNDING_ITEMS, max_chars=_MAX_GROUNDING_CHARS
        ),
        "spatial_relationships": _clean_text_list(
            relationships, max_items=_MAX_GROUNDING_ITEMS, max_chars=_MAX_GROUNDING_CHARS
        ),
        "forbidden_abstractions": _clean_text_list(
            forbidden, max_items=_MAX_GROUNDING_ITEMS, max_chars=_MAX_GROUNDING_CHARS
        ),
    }


def _merge_grounding_seed(
    profile_data: dict[str, Any],
    concept_spec: dict[str, Any],
) -> dict[str, Any]:
    seed = {
        "persistent_anchors": _clean_text_list(
            concept_spec.get("persistent_anchors") or [],
            max_items=_MAX_GROUNDING_ITEMS,
            max_chars=_MAX_GROUNDING_CHARS,
        ),
        "spatial_relationships": _clean_text_list(
            concept_spec.get("spatial_relationships") or [],
            max_items=_MAX_GROUNDING_ITEMS,
            max_chars=_MAX_GROUNDING_CHARS,
        ),
        "forbidden_abstractions": _clean_text_list(
            concept_spec.get("forbidden_abstractions") or [],
            max_items=_MAX_GROUNDING_ITEMS,
            max_chars=_MAX_GROUNDING_CHARS,
        ),
    }
    if profile_data.get("base_profile") == "cinematic_application":
        inferred = _grounding_seed_from_concept(concept_spec)
        for key, values in inferred.items():
            seed[key].extend(values)
    for key, values in seed.items():
        profile_data[key] = list(profile_data.get(key, [])) + values
    return profile_data


def profile_requires_llm(concept_spec: dict[str, Any]) -> bool:
    """Fresh profile resolution uses LLM review by default.

    Built-in and inline profiles are treated as seeds/fallbacks, not final
    policy, so the system can adapt to new scene types automatically.
    """
    if concept_spec.get("scene_profile_llm") is False:
        return False
    return True


def profile_seed(concept_spec: dict[str, Any]) -> SceneProfile:
    requested = concept_spec.get("scene_profile")
    if isinstance(requested, str):
        if requested not in BASE_PROFILES:
            # Don't go through base_profile() with an unknown id and just
            # squash to vector_teaching; let sanitize_profile record the
            # fallback in its validation["warnings"] trail instead.
            return sanitize_profile(
                {"base_profile": requested},
                fallback_base=choose_base_profile(concept_spec),
                source="builtin_unknown_id",
            ).profile
        return sanitize_profile(
            _merge_grounding_seed(base_profile(requested).model_dump(), concept_spec),
            fallback_base=requested,
        ).profile
    if isinstance(requested, dict):
        base_id = requested.get("base_profile") or choose_base_profile(concept_spec)
        merged = deepcopy(BASE_PROFILES.get(base_id, BASE_PROFILES["vector_teaching"]))
        merged.update(requested)
        _merge_grounding_seed(merged, concept_spec)
        return sanitize_profile(merged, fallback_base=base_id).profile
    base_id = choose_base_profile(concept_spec)
    merged = _merge_grounding_seed(base_profile(base_id).model_dump(), concept_spec)
    return sanitize_profile(merged, fallback_base=base_id).profile


def sanitize_profile(
    raw: dict[str, Any],
    *,
    fallback_base: str = "vector_teaching",
    source: str = "sanitized",
) -> SceneProfileResolution:
    validation: dict[str, Any] = {
        "source": source,
        "fallback_base": fallback_base,
        "warnings": [],
    }
    base_id = raw.get("base_profile") if isinstance(raw, dict) else None
    if base_id not in BASE_PROFILES:
        validation["warnings"].append(f"invalid base_profile={base_id!r}; using {fallback_base}")
        base_id = fallback_base if fallback_base in BASE_PROFILES else "vector_teaching"
    merged = deepcopy(BASE_PROFILES[base_id])
    if isinstance(raw, dict):
        for key in (
            "profile_id",
            "base_profile",
            "visual_goal",
            "allowed_helpers",
            "forbidden_helpers",
            "preferred_geometry",
            "overlay_policy",
            "critic_priorities",
            "critic_rubric",
            "repair_policy",
            "persistent_anchors",
            "spatial_relationships",
            "forbidden_abstractions",
        ):
            if key in raw:
                merged[key] = raw[key]

    merged["base_profile"] = base_id
    if not str(merged.get("profile_id", "")).strip():
        merged["profile_id"] = base_id
    for key in (
        "allowed_helpers",
        "forbidden_helpers",
        "preferred_geometry",
        "critic_priorities",
        "repair_policy",
    ):
        if key in ("critic_priorities", "repair_policy"):
            values = merged.get(key, [])
            merged[key] = [str(v).strip() for v in values if str(v).strip()] if isinstance(values, list) else []
        else:
            merged[key] = _clean_list(merged.get(key, []))
    if not isinstance(merged.get("overlay_policy"), dict):
        validation["warnings"].append("overlay_policy was not an object; using base overlay policy")
        merged["overlay_policy"] = deepcopy(BASE_PROFILES[base_id]["overlay_policy"])
    merged["critic_rubric"] = _sanitize_critic_rubric(
        merged.get("critic_rubric"),
        base_id=base_id,
        validation=validation,
    )
    for key in _GROUNDING_LIST_KEYS:
        merged[key] = _clean_text_list(
            merged.get(key),
            max_items=_MAX_GROUNDING_ITEMS,
            max_chars=_MAX_GROUNDING_CHARS,
        )

    forbidden = set(merged["forbidden_helpers"])
    allowed_before = list(merged["allowed_helpers"])
    merged["allowed_helpers"] = [item for item in allowed_before if item not in forbidden]
    removed = sorted(set(allowed_before) - set(merged["allowed_helpers"]))
    if removed:
        validation["warnings"].append(
            "removed helpers from allowed_helpers because forbidden wins: "
            + ", ".join(removed)
        )

    profile = SceneProfile.model_validate(merged)
    validation["profile_id"] = profile.profile_id
    validation["base_profile"] = profile.base_profile
    return SceneProfileResolution(
        profile=profile,
        source=source,
        raw=raw,
        validation=validation,
    )


def format_scene_profile_for_prompt(profile: SceneProfile | None) -> str:
    if profile is None:
        return ""
    lines = [
        "STYLE PROFILE:",
        f"- profile_id: {profile.profile_id}",
        f"- base_profile: {profile.base_profile}",
        f"- visual_goal: {profile.visual_goal}",
        "- allowed_helpers: " + ", ".join(profile.allowed_helpers or ["none"]),
        "- forbidden_helpers: " + ", ".join(profile.forbidden_helpers or ["none"]),
        "- preferred_geometry: " + ", ".join(profile.preferred_geometry or ["none"]),
        "- overlay_policy: " + json.dumps(profile.overlay_policy, ensure_ascii=False),
    ]
    if profile.critic_priorities:
        lines.append("- critic_priorities: " + "; ".join(profile.critic_priorities[:5]))
    if profile.critic_rubric:
        lines.append(
            "- adaptive_critic_rubric: "
            + json.dumps(profile.critic_rubric, ensure_ascii=False)
        )
    if profile.persistent_anchors:
        lines.append(
            "- persistent_anchors: "
            + "; ".join(profile.persistent_anchors[:_MAX_GROUNDING_ITEMS])
        )
    if profile.spatial_relationships:
        lines.append(
            "- spatial_relationships: "
            + "; ".join(profile.spatial_relationships[:_MAX_GROUNDING_ITEMS])
        )
    if profile.forbidden_abstractions:
        lines.append(
            "- forbidden_abstractions: "
            + "; ".join(profile.forbidden_abstractions[:_MAX_GROUNDING_ITEMS])
        )
    if profile.repair_policy:
        lines.append("- repair_policy: " + "; ".join(profile.repair_policy[:5]))
    lines.extend([
        "Treat forbidden_helpers as hard visual constraints. If a storyboard "
        "object or code helper conflicts with this profile, replace it with an "
        "allowed helper that preserves the same concept meaning.",
        "Treat persistent_anchors as hard scene grounding constraints for "
        "application/cinematic scenes. These anchors should appear by exact "
        "or near-exact object name in every shot unless a shot explicitly "
        "hides one for narrative reasons.",
    ])
    return "\n".join(lines)
