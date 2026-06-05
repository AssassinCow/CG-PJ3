"""Machine-readable success criteria for concept videos.

The Success Spec sits between concept YAML and the generated storyboard/scene.
It describes what the viewer must be able to observe, not just which object
names should exist.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StateId = Literal["early", "middle", "late", "final"]
ObjectState = Literal["visible", "hidden", "sharp", "blurred", "static", "moving", "glow"]
EvidenceKind = Literal["frame_difference", "sharpness_ordering", "aperture_within_range"]
ConstraintKind = Literal["object_static", "camera_static", "text_faces_camera"]
TextPlacement = Literal["in_scene", "hud_overlay"]


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameRange(_Forbid):
    fraction: tuple[float, float] | None = None
    frames: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _exactly_one_range(self) -> "FrameRange":
        if (self.fraction is None) == (self.frames is None):
            raise ValueError("frame_range must provide exactly one of fraction or frames")
        if self.fraction is not None:
            start, end = self.fraction
            if not (0.0 <= start < end <= 1.0):
                raise ValueError("frame_range.fraction must satisfy 0 <= start < end <= 1")
        if self.frames is not None:
            start, end = self.frames
            if not (start >= 1 and start <= end):
                raise ValueError("frame_range.frames must satisfy 1 <= start <= end")
        return self

    def resolve_frames(self, *, total_frames: int) -> tuple[int, int]:
        if self.frames is not None:
            start, end = self.frames
            return max(1, start), min(total_frames, end)
        assert self.fraction is not None
        start_f, end_f = self.fraction
        start = max(1, int(total_frames * start_f) + 1)
        end = max(start, int(round(total_frames * end_f)))
        return start, min(total_frames, end)


class SuccessState(_Forbid):
    id: StateId
    frame_range: FrameRange
    object_states: dict[str, list[ObjectState]] = Field(default_factory=dict)
    readable_text: list[str] = Field(default_factory=list)


class FrameDifferenceEvidence(_Forbid):
    kind: Literal["frame_difference"]
    between: list[StateId] = Field(min_length=2)
    min_mean_diff: float = Field(gt=0)


class SharpnessOrdering(_Forbid):
    state: StateId
    sharpest: str = Field(min_length=1)


class SharpnessOrderingEvidence(_Forbid):
    kind: Literal["sharpness_ordering"]
    orderings: list[SharpnessOrdering] = Field(min_length=1)


class ApertureWithinRangeEvidence(_Forbid):
    kind: Literal["aperture_within_range"]
    anchor: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    range: tuple[float, float]

    @field_validator("range")
    @classmethod
    def _valid_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low > high:
            raise ValueError("range lower bound must be <= upper bound")
        return value


RequiredVisualEvidence = (
    FrameDifferenceEvidence | SharpnessOrderingEvidence | ApertureWithinRangeEvidence
)


class ObjectStaticConstraint(_Forbid):
    kind: Literal["object_static"]
    anchors: list[str] = Field(min_length=1)


class CameraStaticConstraint(_Forbid):
    kind: Literal["camera_static"]
    anchor: str = Field(min_length=1)


class TextAnchorPlacement(_Forbid):
    name: str = Field(min_length=1)
    placement: TextPlacement = "in_scene"


class TextFacesCameraConstraint(_Forbid):
    kind: Literal["text_faces_camera"]
    anchors: list[str | TextAnchorPlacement] = Field(min_length=1)

    def anchor_names(self) -> set[str]:
        out: set[str] = set()
        for anchor in self.anchors:
            if isinstance(anchor, str):
                out.add(anchor)
            else:
                out.add(anchor.name)
        return out

    def anchor_placements(self) -> dict[str, TextPlacement]:
        out: dict[str, TextPlacement] = {}
        for anchor in self.anchors:
            if isinstance(anchor, str):
                out[anchor] = "in_scene"
            else:
                out[anchor.name] = anchor.placement
        return out


HardConstraint = ObjectStaticConstraint | CameraStaticConstraint | TextFacesCameraConstraint


class SuccessSpec(_Forbid):
    version: int = Field(default=1)
    success_states: list[SuccessState] = Field(min_length=1)
    required_visual_evidence: list[RequiredVisualEvidence] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("only success_spec version=1 is supported")
        return value

    @model_validator(mode="after")
    def _state_ids_unique(self) -> "SuccessSpec":
        ids = [state.id for state in self.success_states]
        if len(ids) != len(set(ids)):
            raise ValueError("success_state ids must be unique")
        return self

    def aperture_fstop_range(self) -> tuple[float, float] | None:
        for evidence in self.required_visual_evidence:
            if isinstance(evidence, ApertureWithinRangeEvidence):
                if evidence.data_path.endswith("aperture_fstop"):
                    return evidence.range
        return None

    def object_static_anchors(self) -> set[str]:
        anchors: set[str] = set()
        for constraint in self.hard_constraints:
            if isinstance(constraint, ObjectStaticConstraint):
                anchors.update(constraint.anchors)
        return anchors

    def camera_static_anchors(self) -> set[str]:
        anchors: set[str] = set()
        for constraint in self.hard_constraints:
            if isinstance(constraint, CameraStaticConstraint):
                anchors.add(constraint.anchor)
        return anchors

    def text_faces_camera_anchors(self) -> set[str]:
        anchors: set[str] = set()
        for constraint in self.hard_constraints:
            if isinstance(constraint, TextFacesCameraConstraint):
                anchors.update(constraint.anchor_names())
        return anchors

    def text_anchor_placements(self) -> dict[str, TextPlacement]:
        placements: dict[str, TextPlacement] = {}
        for constraint in self.hard_constraints:
            if isinstance(constraint, TextFacesCameraConstraint):
                placements.update(constraint.anchor_placements())
        return placements

    def state_object_anchors(self) -> set[str]:
        anchors: set[str] = set()
        for state in self.success_states:
            anchors.update(state.object_states)
        return anchors

    def readable_text_anchors(self) -> set[str]:
        anchors: set[str] = set()
        for state in self.success_states:
            anchors.update(state.readable_text)
        return anchors

    def required_anchor_names(self) -> set[str]:
        return (
            self.state_object_anchors()
            | self.object_static_anchors()
            | self.text_faces_camera_anchors()
            | self.readable_text_anchors()
        )

    def resolved_state_frames(self, *, total_frames: int) -> dict[str, tuple[int, int]]:
        return {
            state.id: state.frame_range.resolve_frames(total_frames=total_frames)
            for state in self.success_states
        }


def load_success_spec(concept_spec: dict) -> SuccessSpec | None:
    raw = concept_spec.get("success_spec")
    if raw is None:
        return None
    return SuccessSpec.model_validate(raw)


def success_spec_to_json(success_spec: SuccessSpec) -> str:
    return json.dumps(success_spec.model_dump(mode="json"), indent=2)


def format_success_spec_for_coder(success_spec: SuccessSpec | None) -> str:
    if success_spec is None:
        return ""

    lines = [
        "SUCCESS SPEC (hard viewer-facing contract):",
        "- Create exact Blender object names for every listed anchor; do not use aliases.",
        "- Keep required subjects and readable text inside the camera frame with a clear safety margin; do not crop them at screen edges.",
        "- Keep HUD/readable text camera-facing and readable. Prefer camera-parented foreground text or overlay-safe placement so scene depth of field does not blur it.",
        "- Do not make text glow into bokeh: use modest emission, dark outline/backplate if needed, and a stable foreground pseudo-HUD placement.",
    ]

    anchors = sorted(success_spec.required_anchor_names())
    if anchors:
        lines.append("- exact required object names: " + ", ".join(anchors[:24]))
        if len(anchors) > 24:
            lines.append(f"  ... ({len(anchors) - 24} more)")

    text_anchors = sorted(
        success_spec.text_faces_camera_anchors()
        | success_spec.readable_text_anchors()
    )
    if text_anchors:
        lines.append(
            "- exact readable text object names: " + ", ".join(text_anchors)
        )
        lines.append(
            "- For each readable text anchor, set the text body explicitly, avoid negative scale, and orient it toward the render camera."
        )
        hud_anchors = sorted(
            name for name, placement in success_spec.text_anchor_placements().items()
            if placement == "hud_overlay"
        )
        if hud_anchors:
            lines.append(
                "- Pseudo-HUD anchors must be 3D text parented to the render camera with fixed local placement in front of the camera, small scale, low emission, and no DOF blur: "
                + ", ".join(hud_anchors)
            )

    aperture = success_spec.aperture_fstop_range()
    if aperture is not None:
        lines.append(
            f"- camera.data.dof.aperture_fstop must stay in [{aperture[0]:.1f}, {aperture[1]:.1f}]."
        )

    static_anchors = sorted(success_spec.object_static_anchors())
    if static_anchors:
        lines.append(
            "- keep these scene objects spatially static unless only hide_render is animated: "
            + ", ".join(static_anchors)
        )
    camera_static = sorted(success_spec.camera_static_anchors())
    if camera_static:
        lines.append(
            "- keep these cameras spatially static; animate lens/focus properties only: "
            + ", ".join(camera_static)
        )

    for state in success_spec.success_states:
        start, end = state.frame_range.fraction or state.frame_range.frames or ("?", "?")
        state_bits: list[str] = []
        for anchor, states in sorted(state.object_states.items()):
            state_bits.append(f"{anchor}={'+'.join(states)}")
        if state.readable_text:
            state_bits.append("readable_text=" + ",".join(state.readable_text))
        if state_bits:
            lines.append(
                f"- state {state.id} ({start}-{end}): " + "; ".join(state_bits[:8])
            )

    return "\n".join(lines)
