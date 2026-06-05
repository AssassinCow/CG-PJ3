from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NarrativeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str
    formulas: list[str] = Field(default_factory=list)
    duration_sec: float
    visual_intent: str

    @field_validator("duration_sec")
    @classmethod
    def _finite_duration(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"duration_sec={v!r} must be finite (no NaN/Inf)")
        if v <= 0:
            raise ValueError(f"duration_sec={v!r} must be > 0")
        return v


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    nodes: list[NarrativeNode]

    @property
    def total_duration(self) -> float:
        return sum(n.duration_sec for n in self.nodes)
