from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]
KeyframeValue = float | list[float] | list[list[float]]
PrimitiveKind = Literal[
    "sphere", "plane", "cube", "arrow", "empty",
    "cylinder", "cone", "torus", "monkey",
    # Higher-level helpers commonly emitted for curve-heavy concepts.
    "curve_polyline", "sphere_group", "cube_group",
]
ObjectKind = Literal[
    "mesh", "light", "primitive", "curve",
    "mesh_group", "group", "annotation", "text",
]

# Concept ids flow directly into filesystem paths (out_dir = out_root /
# concept_id). Restricting the alphabet here is the schema-level half of
# defense; pipeline.run() applies the same regex on the raw YAML so users
# also get a clear early error when the spec is invalid.
_CONCEPT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _require_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name}={value!r} must be a finite float (no NaN/Inf)")
    return value


def _require_finite_vec3(name: str, value: tuple) -> tuple:
    if value is None:
        return value
    if len(value) != 3:
        raise ValueError(f"{name} must have 3 components, got {len(value)}")
    for i, v in enumerate(value):
        if not math.isfinite(float(v)):
            raise ValueError(
                f"{name}[{i}]={v!r} must be a finite float (no NaN/Inf)"
            )
    return value


class _Forbid(BaseModel):
    """Base class: reject unknown fields so LLM typos surface immediately."""

    model_config = ConfigDict(extra="forbid")


class CameraKey(_Forbid):
    time_sec: float = Field(ge=0)
    position: Vec3
    look_at: Vec3
    fov: float = Field(default=50.0, gt=0, lt=180)

    @field_validator("time_sec", "fov")
    @classmethod
    def _finite_scalar(cls, v: float, info) -> float:
        return _require_finite(info.field_name, v)

    @field_validator("position", "look_at")
    @classmethod
    def _finite_vec(cls, v: tuple, info) -> tuple:
        return _require_finite_vec3(info.field_name, v)


class Keyframe(_Forbid):
    time_sec: float = Field(ge=0)
    attr: str = Field(min_length=1)
    # Scalar attributes (data.energy, scale, lens) use a number; vector
    # attributes (location, rotation_euler, color) use a list. Some
    # generated group/curve helpers use a list of vectors to represent
    # multiple control points or bar scales.
    value: KeyframeValue

    @field_validator("time_sec")
    @classmethod
    def _finite_time(cls, v: float) -> float:
        return _require_finite("time_sec", v)

    @field_validator("value")
    @classmethod
    def _finite_value(cls, v):
        def _walk(x):
            if isinstance(x, (list, tuple)):
                for item in x:
                    _walk(item)
            elif isinstance(x, (int, float)) and not math.isfinite(float(x)):
                raise ValueError(f"keyframe value {x!r} must be finite")
        _walk(v)
        return v


class SceneObject(_Forbid):
    name: str = Field(min_length=1)
    type: ObjectKind
    primitive: PrimitiveKind | None = None
    location: Vec3 = (0.0, 0.0, 0.0)
    properties: dict = Field(default_factory=dict)
    keyframes: list[Keyframe] = Field(default_factory=list)

    @field_validator("location")
    @classmethod
    def _finite_location(cls, v: tuple) -> tuple:
        return _require_finite_vec3("location", v)


class OverlayZone(_Forbid):
    """All values are normalized screen coordinates in [0, 1] with
    origin at top-left. `x + w` and `y + h` must stay within [0, 1]."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @field_validator("x", "y", "w", "h")
    @classmethod
    def _finite(cls, v: float, info) -> float:
        return _require_finite(info.field_name, v)

    @model_validator(mode="after")
    def _fits_in_frame(self) -> "OverlayZone":
        if self.x + self.w > 1.0 + 1e-6:
            raise ValueError(
                f"overlay_zone overflows right edge: x+w={self.x + self.w:.3f} > 1"
            )
        if self.y + self.h > 1.0 + 1e-6:
            raise ValueError(
                f"overlay_zone overflows bottom edge: y+h={self.y + self.h:.3f} > 1"
            )
        return self


class Shot(_Forbid):
    node_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0)
    duration_sec: float = Field(gt=0)
    camera: list[CameraKey] = Field(min_length=1)
    objects: list[SceneObject] = Field(min_length=1)
    overlay_zone: OverlayZone | None = None
    formula: str | None = None
    caption: str | None = None

    @field_validator("start_sec", "duration_sec")
    @classmethod
    def _finite_timing(cls, v: float, info) -> float:
        return _require_finite(info.field_name, v)

    @model_validator(mode="after")
    def _camera_keys_inside_shot(self) -> "Shot":
        for k in self.camera:
            # Camera key times are expressed in the global timeline.
            if k.time_sec < self.start_sec - 1e-6:
                raise ValueError(
                    f"shot {self.node_id}: camera key at t={k.time_sec} is "
                    f"before shot start_sec={self.start_sec}"
                )
            if k.time_sec > self.start_sec + self.duration_sec + 1e-6:
                raise ValueError(
                    f"shot {self.node_id}: camera key at t={k.time_sec} is "
                    f"after shot end={self.start_sec + self.duration_sec}"
                )
        return self


class Storyboard(_Forbid):
    concept_id: str = Field(min_length=1)
    fps: int = Field(default=30, gt=0, le=120)
    resolution: tuple[int, int] = (1280, 720)
    shots: list[Shot] = Field(min_length=1)

    @field_validator("concept_id")
    @classmethod
    def _concept_id_is_path_safe(cls, v: str) -> str:
        if not _CONCEPT_ID_RE.match(v):
            raise ValueError(
                f"concept_id={v!r} must match [A-Za-z0-9_-]+ "
                "(no path separators, no '..' segments)"
            )
        return v

    @model_validator(mode="after")
    def _resolution_positive(self) -> "Storyboard":
        w, h = self.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        return self

    @model_validator(mode="after")
    def _shots_contiguous(self) -> "Storyboard":
        """Shots should tile [0, total_duration) contiguously: each
        shot's start_sec equals the previous shot's end_sec (within
        a small float tolerance).
        """
        cursor = 0.0
        eps = 1e-3
        for idx, shot in enumerate(self.shots):
            if abs(shot.start_sec - cursor) > eps:
                raise ValueError(
                    f"shot {idx} ({shot.node_id}) start_sec={shot.start_sec} "
                    f"does not match expected {cursor:.3f}; shots must be "
                    f"contiguous"
                )
            cursor = shot.start_sec + shot.duration_sec
        return self

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec for s in self.shots)

    @property
    def total_frames(self) -> int:
        return int(self.total_duration * self.fps)
