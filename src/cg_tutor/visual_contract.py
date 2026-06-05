"""Deterministic visual contracts derived from narrative/storyboard text.

This module intentionally stays concept-agnostic. It converts common
teaching-video language (labels, vectors, highlights, comparisons, overlay
space) into explicit constraints the coder and repair loop can reuse.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from cg_tutor.schemas import NarrativeNode, Shot
from cg_tutor.scene_profiles import (
    SceneProfile,
    profile_allows_helper_group,
    profile_forbids_helper_group,
)


class VisualContract(BaseModel):
    shot_id: str
    required_anchors: list[str] = Field(default_factory=list)
    required_labels: list[str] = Field(default_factory=list)
    required_vectors: list[str] = Field(default_factory=list)
    required_relationships: list[str] = Field(default_factory=list)
    emphasis_points: list[str] = Field(default_factory=list)
    overlay_constraints: list[str] = Field(default_factory=list)
    forbidden_failures: list[str] = Field(default_factory=list)

    @property
    def has_constraints(self) -> bool:
        return any([
            self.required_labels,
            self.required_anchors,
            self.required_vectors,
            self.required_relationships,
            self.emphasis_points,
            self.overlay_constraints,
            self.forbidden_failures,
        ])


_VECTOR_WORDS = (
    "arrow", "vector", "ray", "direction", "normal", "tangent",
    "reflection", "view", "axis", "axes",
)
_LABEL_WORDS = (
    "label", "labels", "labeled", "labelled", "text",
    "annotation", "annotations", "annotate", "annotated",
)
_HIGHLIGHT_WORDS = ("highlight", "specular", "gloss", "bright spot")
_COMPARE_WORDS = (
    "compare", "comparison", "components", "decompose", "side by side",
    "side-by-side", "three", "multiple",
)


def _contains_terms(text_low: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        escaped = re.escape(term.lower())
        if re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text_low):
            return True
    return False


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        norm = " ".join(item.split()).strip()
        key = norm.lower()
        if norm and key not in seen:
            out.append(norm)
            seen.add(key)
    return out


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9]+", name) if t]


def _label_candidates_from_text(text: str) -> list[str]:
    """Extract short label-like words after labelled/labeled/list phrases.

    This is a heuristic, not NLP: it targets common prompt text such as
    "labeled ambient, diffuse, and specular" without overfitting to one
    concept.
    """
    candidates: list[str] = []
    patterns = [
        r"\b(?:labeled|labelled|labels?|text)\s+(?:as\s+|with\s+)?([^.;:]+)",
        r"\b(?:show|compare)\s+([^.;:]+?)\s+(?:labels?|components?)\b",
    ]
    stop = {
        "and", "or", "the", "a", "an", "with", "near", "for", "each",
        "visible", "clearly", "component", "components", "label", "labels",
        "as", "of", "to", "from", "reads", "says", "showing",
    }
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.I):
            phrase = match.group(1)
            phrase = re.split(
                r"\b(?:while|where|that|so|but|then)\b",
                phrase,
                maxsplit=1,
                flags=re.I,
            )[0]
            quoted = re.findall(r"['\"]([^'\"]{1,32})['\"]", phrase)
            if quoted:
                candidates.extend(q.strip() for q in quoted if q.strip())
                continue
            for token in re.split(r"[,/]| and | or ", phrase):
                token = token.strip(" '\"()[]{}")
                words = [
                    w for w in re.split(r"\s+", token)
                    if w and w.lower() not in stop
                ]
                if 1 <= len(words) <= 3:
                    label = " ".join(words)
                    if re.search(r"[A-Za-z]", label):
                        candidates.append(label)
    return _dedupe(candidates)


def _object_label_candidates(shot: Shot) -> list[str]:
    labels: list[str] = []
    for obj in shot.objects:
        if obj.type in {"annotation", "text"}:
            labels.append(obj.name)
        props = obj.properties or {}
        for key in ("text", "label", "caption"):
            value = props.get(key)
            if isinstance(value, str) and value.strip():
                labels.append(value.strip())
    return labels


def _vector_candidates(shot: Shot, text: str) -> list[str]:
    vectors: list[str] = []
    for obj in shot.objects:
        name_low = obj.name.lower()
        if obj.primitive == "arrow" or any(w in name_low for w in _VECTOR_WORDS):
            vectors.append(obj.name)

    # Capture symbolic vector names in prose/formulae, e.g. N, L, R, V.
    for token in re.findall(r"\b[A-Z]\b", text):
        if token not in {"A", "I"}:
            vectors.append(token)
    return _dedupe(vectors)


def build_visual_contract(
    shot: Shot,
    node: NarrativeNode | None = None,
    *,
    scene_profile: SceneProfile | None = None,
) -> VisualContract:
    visual_intent = node.visual_intent if node else ""
    description = node.description if node else ""
    formulas = " ".join(node.formulas) if node else ""
    text = " ".join(
        p for p in [visual_intent, description, shot.caption or "", shot.formula or "", formulas]
        if p
    )
    low = text.lower()

    required_labels: list[str] = []
    required_vectors: list[str] = []
    required_relationships: list[str] = []
    emphasis_points: list[str] = []
    overlay_constraints: list[str] = []
    forbidden_failures: list[str] = []
    required_anchors: list[str] = []

    if _contains_terms(low, _LABEL_WORDS):
        required_labels.extend(_label_candidates_from_text(text))
        required_labels.extend(_object_label_candidates(shot))
        if not required_labels:
            required_labels.append("visible text labels")
        forbidden_failures.append("Do not omit required labels or make them too small to read.")

    is_cinematic = (
        scene_profile is not None
        and scene_profile.base_profile == "cinematic_application"
    )
    no_drawn_rays = profile_forbids_helper_group(scene_profile, "drawn_rays")
    prefer_thin_tracers = (
        profile_allows_helper_group(scene_profile, "drawn_rays")
        and not no_drawn_rays
    )
    forbid_arrows = profile_forbids_helper_group(scene_profile, "arrow_helpers")

    if _contains_terms(low, _VECTOR_WORDS) or any(
        obj.primitive == "arrow" for obj in shot.objects
    ):
        if no_drawn_rays:
            required_relationships.append(
                "Do not draw ray/vector paths; express reflection, refraction, shadow, and Fresnel through physical surfaces, glass distortion, shadows, and highlights."
            )
            forbidden_failures.append(
                "Do not add drawn ray paths, dotted tracers, arrows, schematic polylines, axes, or classroom vector helpers."
            )
        else:
            required_vectors.extend(_vector_candidates(shot, text))
            if not required_vectors:
                required_vectors.append("visible ray/vector cues")
        if (is_cinematic or prefer_thin_tracers or forbid_arrows) and not no_drawn_rays:
            required_relationships.append(
                "Represent ray/vector cues as thin glowing curve_polyline or dotted tracers, not thick arrow primitives."
            )
            forbidden_failures.append(
                "Do not use thick arrows, arrow primitives, visible light bars, or foreground light panels."
            )
        elif not is_cinematic:
            required_relationships.append(
                "Draw each required vector/arrow as a full visible shaft plus head."
            )
            forbidden_failures.append("Do not merge, overlap, or hide distinct vectors/arrows.")
        if len(required_vectors) >= 2:
            required_relationships.append(
                "Keep distinct vector/ray cues spatially separated enough to read individually."
            )

    if _contains_terms(low, _HIGHLIGHT_WORDS):
        emphasis_points.append("visible highlight / bright spot")
        forbidden_failures.append("Do not let the requested highlight disappear into dark shading.")
        if required_vectors:
            required_relationships.append(
                "When vectors/rays explain a highlight, anchor them at or visibly connect them to the highlighted point."
            )

    if _contains_terms(low, _COMPARE_WORDS):
        required_relationships.append(
            "Lay compared components out with clear separation and stable ordering."
        )
        if "label" in low or "component" in low:
            forbidden_failures.append("Do not show component comparisons without labels.")

    if scene_profile is not None and scene_profile.persistent_anchors:
        storyboard_names = {obj.name for obj in shot.objects}
        for anchor in scene_profile.persistent_anchors:
            anchor_low = anchor.lower()
            if anchor in storyboard_names or any(
                anchor_low in name.lower() or name.lower() in anchor_low
                for name in storyboard_names
            ):
                required_anchors.append(anchor)
        if not required_anchors:
            required_anchors.extend(scene_profile.persistent_anchors)
        required_relationships.extend(scene_profile.spatial_relationships)
        forbidden_failures.extend(
            f"Do not abstract away scene anchor: {anchor}."
            for anchor in required_anchors
        )
        forbidden_failures.extend(scene_profile.forbidden_abstractions)

    if (
        scene_profile is not None
        and scene_profile.base_profile == "cinematic_application"
    ):
        forbidden_failures.extend(scene_profile.forbidden_abstractions)

    if shot.overlay_zone is not None:
        z = shot.overlay_zone
        overlay_constraints.append(
            f"Reserve overlay zone x={z.x:.2f}, y={z.y:.2f}, w={z.w:.2f}, h={z.h:.2f}; keep key objects and labels outside it."
        )
        forbidden_failures.append("Do not place the main subject or labels under the formula overlay.")

    return VisualContract(
        shot_id=shot.node_id,
        required_anchors=_dedupe(required_anchors),
        required_labels=_dedupe(required_labels),
        required_vectors=_dedupe(required_vectors),
        required_relationships=_dedupe(required_relationships),
        emphasis_points=_dedupe(emphasis_points),
        overlay_constraints=_dedupe(overlay_constraints),
        forbidden_failures=_dedupe(forbidden_failures),
    )


def format_visual_contract(contract: VisualContract) -> list[str]:
    if not contract.has_constraints:
        return []
    lines: list[str] = []
    if contract.required_labels:
        lines.append("- required_labels: " + ", ".join(contract.required_labels[:8]))
    if contract.required_anchors:
        lines.append("- required_anchors: " + ", ".join(contract.required_anchors[:10]))
    if contract.required_vectors:
        lines.append("- required_vectors: " + ", ".join(contract.required_vectors[:8]))
    if contract.emphasis_points:
        lines.append("- emphasis_points: " + "; ".join(contract.emphasis_points[:4]))
    if contract.required_relationships:
        lines.append("- required_relationships: " + "; ".join(contract.required_relationships[:5]))
    if contract.overlay_constraints:
        lines.append("- overlay_constraints: " + "; ".join(contract.overlay_constraints[:3]))
    if contract.forbidden_failures:
        lines.append("- avoid: " + "; ".join(contract.forbidden_failures[:5]))
    return lines
