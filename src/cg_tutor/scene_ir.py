"""Scene intermediate representation used across agents.

The current renderer still relies on an LLM-generated ``scene.py`` for the
full Blender implementation, but this IR gives the pipeline a stable,
schema-checked contract between narrative/storyboard/coder/critic. It is a
practical stepping stone toward a future deterministic IR -> bpy compiler.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from cg_tutor.schemas import Narrative, Storyboard
from cg_tutor.scene_profiles import SceneProfile
from cg_tutor.visual_contract import (
    VisualContract,
    build_visual_contract,
    format_visual_contract,
)


class SceneIRObject(BaseModel):
    name: str
    object_type: str
    primitive: str | None = None
    role: Literal["visual", "light", "camera_helper", "annotation"] = "visual"
    required: bool = True


class VisualRequirementSlot(BaseModel):
    slot_type: Literal[
        "projection_geometry",
        "curve_construction",
        "transformation_sequence",
        "lighting_components",
        "ray_path",
        "grounded_environment",
        "comparison_state",
        "custom",
    ]
    priority: Literal["hard", "soft"] = "hard"
    requires: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    natural_language_contract: str = ""


class SceneIRShot(BaseModel):
    node_id: str
    start_sec: float
    duration_sec: float
    visual_intent: str
    narrative: str = ""
    formula: str | None = None
    caption: str | None = None
    expected_objects: list[SceneIRObject] = Field(default_factory=list)
    requirement_slots: list[VisualRequirementSlot] = Field(default_factory=list)
    visual_contract: VisualContract | None = None


class SceneIR(BaseModel):
    concept_id: str
    fps: int
    resolution: tuple[int, int]
    scene_profile_id: str | None = None
    scene_profile_base: str | None = None
    shots: list[SceneIRShot]


class SceneIRIssue(BaseModel):
    severity: Literal["block", "warn"]
    shot_id: str | None = None
    rule_id: str
    message: str
    suggested_fix: str


class SceneIRVerification(BaseModel):
    ok: bool
    issues: list[SceneIRIssue] = Field(default_factory=list)

    @property
    def block_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "block")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")


def _object_role(obj) -> str:
    if obj.type == "light":
        return "light"
    if obj.type in {"annotation", "text"}:
        return "annotation"
    return "visual"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = " ".join(str(item).strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(word in low for word in words)


def _build_requirement_slots(
    *,
    text: str,
    object_names: list[str],
    visual_contract: VisualContract | None,
    scene_profile: SceneProfile | None,
) -> list[VisualRequirementSlot]:
    slots: list[VisualRequirementSlot] = []
    object_text = " ".join(object_names)
    combined = f"{text} {object_text}"

    def add_slot(
        slot_type: VisualRequirementSlot.model_fields["slot_type"].annotation,
        *,
        requires: list[str],
        relationships: list[str],
        forbidden: list[str] | None = None,
        priority: Literal["hard", "soft"] = "hard",
        natural_language_contract: str = "",
    ) -> None:
        slots.append(VisualRequirementSlot(
            slot_type=slot_type,
            priority=priority,
            requires=_dedupe(requires),
            relationships=_dedupe(relationships),
            forbidden=_dedupe(forbidden or []),
            natural_language_contract=natural_language_contract,
        ))

    if _has_any(combined, ("pinhole", "image plane", "projection", "projected point", "perspective divide")):
        add_slot(
            "projection_geometry",
            requires=[
                "camera_center or pinhole",
                "image_plane",
                "object_point",
                "projection_ray",
                "projected_point",
            ],
            relationships=[
                "projection_ray connects object_point -> pinhole -> projected_point",
                "projected_point lies on image_plane",
                "focal length bracket spans pinhole to image_plane when focal length is discussed",
            ],
            forbidden=[
                "Do not show projection rays detached from the pinhole.",
                "Do not show projected points floating away from the image plane.",
            ],
            natural_language_contract="Preserve geometric projection relationships as visible spatial constraints.",
        )

    if _has_any(combined, ("bezier", "control point", "control polygon", "de casteljau", "curve")):
        add_slot(
            "curve_construction",
            requires=["control_points", "control_polygon", "curve_path"],
            relationships=[
                "curve remains visually tied to its control points",
                "control polygon stays visible when explaining construction",
                "interpolation helpers should connect neighboring construction points",
            ],
            forbidden=["Do not hide or occlude control points under the curve."],
            natural_language_contract="Make the curve construction legible, not just the final curve.",
        )

    if _has_any(combined, ("affine", "transform", "translation", "rotation", "scaling", "matrix")):
        add_slot(
            "transformation_sequence",
            requires=["source_shape", "transformed_shape", "coordinate_frame"],
            relationships=[
                "before and after states remain comparable",
                "transformation path/order is visible when sequence matters",
            ],
            forbidden=["Do not show only the final object when order/comparison is being taught."],
        )

    if _has_any(combined, ("phong", "ambient", "diffuse", "specular", "normal", "highlight", "n·l", "r·v")):
        add_slot(
            "lighting_components",
            requires=["surface", "key_light", "normal", "light_vector", "view_vector", "highlight"],
            relationships=[
                "light/vector cues visibly connect to the lit surface point",
                "highlight changes consistently with light/view geometry",
            ],
            forbidden=["Do not show lighting terms without a visible lit surface relationship."],
        )

    if _has_any(combined, ("ray", "reflection", "refraction", "whitted", "trace", "tracer")):
        add_slot(
            "ray_path",
            requires=["ray_path", "surface_or_medium", "interaction_point"],
            relationships=[
                "ray path visibly meets the reflecting/refracting surface",
                "secondary rays branch from the interaction point when shown",
            ],
            forbidden=["Do not draw rays floating independently of scene geometry."],
        )

    if _has_any(combined, ("compare", "comparison", "before", "after", "increase", "decrease", "shrinks", "grows", "larger", "smaller")):
        add_slot(
            "comparison_state",
            requires=["reference_state", "current_state"],
            relationships=[
                "reference and current states are visible in the same shot or with stable continuity",
                "differences are spatially aligned enough to compare",
            ],
            forbidden=["Do not replace comparison with a single unlabelled state."],
        )

    if scene_profile and (
        scene_profile.persistent_anchors
        or scene_profile.spatial_relationships
        or scene_profile.forbidden_abstractions
    ):
        add_slot(
            "grounded_environment",
            requires=list(scene_profile.persistent_anchors),
            relationships=list(scene_profile.spatial_relationships),
            forbidden=list(scene_profile.forbidden_abstractions),
            natural_language_contract="Build recognizable scene anchors before transient effects.",
        )

    if visual_contract:
        contract_requires = (
            visual_contract.required_anchors
            + visual_contract.required_labels
            + visual_contract.required_vectors
        )
        if contract_requires or visual_contract.required_relationships:
            add_slot(
                "custom",
                priority="hard",
                requires=contract_requires,
                relationships=list(visual_contract.required_relationships),
                forbidden=list(visual_contract.forbidden_failures),
                natural_language_contract="Derived visual contract requirements that do not fit a more specific slot.",
            )

    return slots


def build_scene_ir(
    narrative: Narrative,
    storyboard: Storyboard,
    *,
    scene_profile: SceneProfile | None = None,
) -> SceneIR:
    nodes = {n.id: n for n in narrative.nodes}
    shots: list[SceneIRShot] = []
    for shot in storyboard.shots:
        node = nodes.get(shot.node_id)
        formulas = node.formulas if node else []
        expected = [
            SceneIRObject(
                name=obj.name,
                object_type=obj.type,
                primitive=obj.primitive,
                role=_object_role(obj),
                required=True,
            )
            for obj in shot.objects
        ]
        visual_contract = build_visual_contract(
            shot, node, scene_profile=scene_profile,
        )
        slot_text = " ".join(
            p for p in [
                node.visual_intent if node else "",
                node.description if node else "",
                " ".join(node.formulas) if node else "",
                shot.caption or "",
                shot.formula or "",
            ]
            if p
        )
        shots.append(SceneIRShot(
            node_id=shot.node_id,
            start_sec=shot.start_sec,
            duration_sec=shot.duration_sec,
            visual_intent=node.visual_intent if node else "",
            narrative=node.description if node else "",
            formula=shot.formula or (formulas[0] if formulas else None),
            caption=shot.caption,
            expected_objects=expected,
            requirement_slots=_build_requirement_slots(
                text=slot_text,
                object_names=[obj.name for obj in shot.objects],
                visual_contract=visual_contract,
                scene_profile=scene_profile,
            ),
            visual_contract=visual_contract,
        ))
    return SceneIR(
        concept_id=storyboard.concept_id,
        fps=storyboard.fps,
        resolution=storyboard.resolution,
        scene_profile_id=scene_profile.profile_id if scene_profile else None,
        scene_profile_base=scene_profile.base_profile if scene_profile else None,
        shots=shots,
    )


def verify_scene_ir(scene_ir: SceneIR) -> SceneIRVerification:
    issues: list[SceneIRIssue] = []
    if not scene_ir.shots:
        issues.append(SceneIRIssue(
            severity="block",
            rule_id="no_shots",
            message="Scene IR has no shots.",
            suggested_fix="Regenerate storyboard and scene IR with at least one shot.",
        ))
    if scene_ir.fps <= 0:
        issues.append(SceneIRIssue(
            severity="block",
            rule_id="bad_fps",
            message=f"Scene IR fps must be positive, got {scene_ir.fps}.",
            suggested_fix="Use the storyboard fps, normally 24.",
        ))
    for shot in scene_ir.shots:
        if not shot.visual_intent.strip():
            issues.append(SceneIRIssue(
                severity="block",
                shot_id=shot.node_id,
                rule_id="missing_visual_intent",
                message="Shot has no visual_intent.",
                suggested_fix="Regenerate narrative or carry the node visual_intent into Scene IR.",
            ))
        if not shot.expected_objects:
            issues.append(SceneIRIssue(
                severity="block",
                shot_id=shot.node_id,
                rule_id="no_expected_objects",
                message="Shot has no expected visible objects.",
                suggested_fix="Regenerate storyboard with at least one visible object for the shot.",
            ))
        if shot.visual_contract is None:
            issues.append(SceneIRIssue(
                severity="warn",
                shot_id=shot.node_id,
                rule_id="missing_visual_contract",
                message="Shot has no derived visual contract.",
                suggested_fix="Rebuild scene IR from narrative and storyboard.",
            ))
        if scene_ir.scene_profile_base == "cinematic_application":
            for obj in shot.expected_objects:
                low_name = obj.name.lower()
                if obj.primitive == "arrow" or "arrow" in low_name:
                    issues.append(SceneIRIssue(
                        severity="block",
                        shot_id=shot.node_id,
                        rule_id="forbidden_profile_arrow",
                        message=(
                            "Cinematic profile forbids arrow primitives or "
                            "arrow-named helper objects."
                        ),
                        suggested_fix=(
                            "Use curve_polyline thin glowing/dotted tracers "
                            "or small HUD line segments instead."
                        ),
                    ))
                if any(w in low_name for w in ("area_light_rect", "light_bar", "light_panel")):
                    issues.append(SceneIRIssue(
                        severity="block",
                        shot_id=shot.node_id,
                        rule_id="forbidden_profile_light_gizmo",
                        message=(
                            "Cinematic profile forbids visible light bars or "
                            "foreground light-panel helper objects."
                        ),
                        suggested_fix=(
                            "Use ordinary invisible light objects and keep "
                            "softbox glow subtle, small, and non-dominant."
                        ),
                    ))
        names = [o.name for o in shot.expected_objects if o.name.strip()]
        if len(names) != len(set(names)):
            issues.append(SceneIRIssue(
                severity="warn",
                shot_id=shot.node_id,
                rule_id="duplicate_object_names",
                message="Shot repeats one or more object names.",
                suggested_fix="Keep names stable across shots only for the same logical object.",
            ))
    return SceneIRVerification(ok=not any(i.severity == "block" for i in issues),
                               issues=issues)


def scene_ir_to_json(scene_ir: SceneIR) -> str:
    return scene_ir.model_dump_json(indent=2)


def scene_ir_verification_to_json(report: SceneIRVerification) -> str:
    return report.model_dump_json(indent=2)


def format_scene_ir_for_coder(
    scene_ir: SceneIR,
    verification: SceneIRVerification | None = None,
) -> str:
    lines = [
        "SCENE IR CONTRACT:",
        "This structured IR is the stable semantic contract. The generated "
        "scene.py must implement it faithfully; do not invent extra helpers "
        "unless visual_intent or expected_objects require them.",
    ]
    if verification and verification.issues:
        lines.append("")
        lines.append("SCENE IR VERIFIER:")
        for issue in verification.issues[:8]:
            shot = f"{issue.shot_id}: " if issue.shot_id else ""
            lines.append(
                f"- [{issue.severity} {issue.rule_id}] {shot}{issue.message} "
                f"Fix: {issue.suggested_fix}"
            )
    for shot in scene_ir.shots:
        objects = ", ".join(o.name for o in shot.expected_objects[:12])
        if len(shot.expected_objects) > 12:
            objects += f", ... ({len(shot.expected_objects) - 12} more)"
        lines.extend([
            "",
            f"{shot.node_id}:",
            f"- visual_intent: {shot.visual_intent}",
            f"- expected_objects: {objects}",
            f"- formula/caption: {shot.formula or shot.caption or 'none'}",
        ])
        if shot.visual_contract:
            contract_lines = format_visual_contract(shot.visual_contract)
            if contract_lines:
                lines.append("- derived_visual_contract:")
                lines.extend(f"  {line}" for line in contract_lines)
        if shot.requirement_slots:
            lines.append("- visual_requirement_slots:")
            for slot in shot.requirement_slots[:5]:
                lines.append(
                    f"  - type={slot.slot_type}, priority={slot.priority}"
                )
                if slot.requires:
                    lines.append(
                        "    requires: " + ", ".join(slot.requires[:8])
                    )
                if slot.relationships:
                    lines.append(
                        "    relationships: "
                        + "; ".join(slot.relationships[:4])
                    )
                if slot.forbidden:
                    lines.append(
                        "    forbidden: " + "; ".join(slot.forbidden[:4])
                    )
                if slot.natural_language_contract:
                    lines.append(
                        "    contract: " + slot.natural_language_contract
                    )
    return "\n".join(lines)


def load_scene_ir_json(text: str) -> SceneIR:
    return SceneIR.model_validate(json.loads(text))
