from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Severity = Literal["block", "warn"]
IssueCategory = Literal[
    "occlusion", "off_screen", "overlay_collision", "lighting",
    "concept_mismatch", "other",
]
EvidenceKind = Literal[
    "object_visible",
    "text_readable",
    "stay_in_screen_safe",
    "helper_hidden",
    "animation_coverage",
    "progressive_visual_ordering",
]

# Categories that count toward "framing" compliance, i.e. how well the
# render obeys the storyboard's layout/lighting contract. `concept_mismatch`
# is intentionally excluded: it tracks semantic fidelity (does the picture
# actually show the concept) and we treat it as a separate axis so it does
# not destabilise best-of-N selection.
FRAMING_CATEGORIES: tuple[str, ...] = (
    "occlusion", "off_screen", "overlay_collision", "lighting", "other",
)


class CriticIssue(BaseModel):
    # CriticIssue is parsed directly from LLM output. LLMs commonly emit
    # extra annotation fields (rationale, frame_evidence, ...) we don't
    # need; rejecting them would drop the whole issue. Allow + ignore.
    model_config = ConfigDict(extra="ignore")

    shot_id: str
    frame_idx: int
    severity: Severity
    category: IssueCategory
    issue: str
    suggested_fix: dict = Field(default_factory=dict)
    evidence_kind: EvidenceKind | None = None
    target: str | None = None
    expected: str | None = None
    observed: str | None = None
    confidence: float | None = None


class CriticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    iteration: int
    overall_score: float
    issues: list[CriticIssue] = Field(default_factory=list)
    # Critic-side failures (relay 504, JSON parse, subprocess crash). Kept
    # separate from `issues` so a flaky vision call does not inflate the
    # framing-warn count or pull overall_score down via best-of-N selection.
    execution_errors: list[str] = Field(default_factory=list)
    # Non-issue reasons to keep iterating. Used for ensemble uncertainty:
    # e.g. member critics disagree strongly even after aggregate severity
    # dedupe.
    pass_blockers: list[str] = Field(default_factory=list)
    ensemble_diagnostics: dict = Field(default_factory=dict)

    @property
    def has_block(self) -> bool:
        return any(i.severity == "block" for i in self.issues)

    @property
    def pass_threshold(self) -> bool:
        return (
            self.overall_score > 0.7
            and not self.has_block
            and not self.execution_errors
            and not self.pass_blockers
        )
