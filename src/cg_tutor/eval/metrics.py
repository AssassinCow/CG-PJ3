"""Lightweight evaluation scaffold for CG-Tutor outputs.

W2: human 5-axis scoring stored as YAML. W3 will add VLM auto-scoring
on the same axes and a correlation report.

Axes (1-5 integer scale):
  clarity        — Can a viewer identify the geometric setup?
  correctness    — Does the visual match the concept being explained?
  aesthetics     — Lighting, framing, composition.
  formula_sync   — Does the formula overlay appear when relevant
                   without occluding the subject?
  pacing         — Are shot durations appropriate (not rushed/slow)?

A pass is mean >= 3.5 and no single axis < 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


AXES = ("clarity", "correctness", "aesthetics", "formula_sync", "pacing")


@dataclass
class ConceptEval:
    concept_id: str
    scores: dict[str, int]            # axis -> 1..5
    notes: str = ""
    rater: str = "anon"

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores)

    @property
    def pass_(self) -> bool:
        return self.mean >= 3.5 and min(self.scores.values()) >= 2


@dataclass
class EvalSet:
    name: str
    entries: list[ConceptEval] = field(default_factory=list)

    def to_yaml(self, path: Path) -> None:
        data = {
            "name": self.name,
            "axes": list(AXES),
            "entries": [
                {
                    "concept_id": e.concept_id,
                    "rater": e.rater,
                    "scores": e.scores,
                    "notes": e.notes,
                    "mean": round(e.mean, 2),
                    "pass": e.pass_,
                } for e in self.entries
            ],
        }
        path.write_text(yaml.safe_dump(data, sort_keys=False,
                                       allow_unicode=True))

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalSet":
        data = yaml.safe_load(path.read_text())
        return cls(
            name=data["name"],
            entries=[
                ConceptEval(
                    concept_id=e["concept_id"],
                    rater=e.get("rater", "anon"),
                    scores=e["scores"],
                    notes=e.get("notes", ""),
                ) for e in data.get("entries", [])
            ],
        )

    def summary(self) -> dict:
        n = len(self.entries)
        if n == 0:
            return {"n": 0}
        means = {a: sum(e.scores[a] for e in self.entries) / n for a in AXES}
        return {
            "n": n,
            "axis_means": {a: round(v, 2) for a, v in means.items()},
            "overall_mean": round(sum(e.mean for e in self.entries) / n, 2),
            "pass_rate": round(sum(1 for e in self.entries if e.pass_) / n, 2),
        }


def blank_entry(concept_id: str, rater: str = "anon") -> ConceptEval:
    """Pre-filled stub for manual rating."""
    return ConceptEval(
        concept_id=concept_id,
        rater=rater,
        scores={a: 0 for a in AXES},
        notes="(fill me)",
    )
