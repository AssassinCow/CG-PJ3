"""Structured failure memory for repeated concept-rendering runs.

Cross-run state lives in a shared JSONL file. Two concerns are handled
here that single-writer scripts usually ignore:

1. Concurrency: multiple pipelines (different concepts running in
   parallel, or accidental re-launch) may all try to append at the same
   time. We take an advisory lock via ``fcntl.flock`` on POSIX so
   concurrent appends never interleave a half-written line.

2. Growth: the file is append-only by design (history is signal). We
   compact in-place when it crosses a soft cap by collapsing identical
   ``key`` tuples into one entry with the summed ``count`` and the
   latest ``last_seen``. This bounds disk + load time without losing
   the "we have seen this issue before" signal.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

try:  # POSIX only — Windows runs without locking, which matches today's
    # WSL-but-not-native-Windows deployment.
    import fcntl  # type: ignore
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows fallback
    _HAS_FCNTL = False


# Soft cap. Crossing it triggers an in-place compaction (dedupe by key,
# keep the most recent N entries per concept). Picked so that ~50 runs
# per concept × ~10 concepts × ~20 issues/run stays well under it.
_COMPACTION_LINE_CAP = 5000
_COMPACTION_KEEP_PER_CONCEPT = 200


class FailureMemoryEntry(BaseModel):
    concept_id: str
    shot_id: str | None = None
    category: str
    severity: str
    issue: str
    suggested_action: str
    source: str = "critic"
    count: int = 1
    last_seen: float = Field(default_factory=time.time)

    @property
    def key(self) -> tuple[str, str | None, str, str, str]:
        return (
            self.concept_id,
            self.shot_id,
            self.category,
            self.severity,
            self.issue[:160],
        )


def _action_for_category(category: str) -> str:
    if category == "concept_mismatch":
        return (
            "Compare against visual_intent and add/remove/modify visible "
            "geometry/material/label/highlight/animation; camera-only fixes "
            "are usually insufficient."
        )
    if category == "off_screen":
        return "Preserve clean shots while reframing the flagged object/shot."
    if category == "overlay_collision":
        return "Reserve overlay space by moving overlay_zone or subject."
    if category == "occlusion":
        return "Separate overlapping objects or hide construction helpers."
    if category == "lighting":
        return "Adjust light/material setup so the intended cue is visible."
    return "Make the smallest direct fix and avoid unrelated redesign."


def memory_from_history(concept_id: str, history: list) -> list[FailureMemoryEntry]:
    entries: dict[tuple[str, str | None, str, str, str], FailureMemoryEntry] = {}
    for item in history:
        for issue in item.report.issues:
            entry = FailureMemoryEntry(
                concept_id=concept_id,
                shot_id=issue.shot_id,
                category=issue.category,
                severity=issue.severity,
                issue=issue.issue,
                suggested_action=_action_for_category(issue.category),
            )
            if entry.key in entries:
                old = entries[entry.key]
                old.count += 1
                old.last_seen = time.time()
            else:
                entries[entry.key] = entry
            patch = _grounding_patch_from_issue(issue)
            if patch:
                patch_issue = json.dumps(patch, sort_keys=True)
                patch_entry = FailureMemoryEntry(
                    concept_id=concept_id,
                    shot_id=issue.shot_id,
                    category="critic_grounding_constraints",
                    severity=issue.severity,
                    issue=patch_issue,
                    suggested_action=(
                        "Treat these critic-derived anchors, labels, and "
                        "relationships as required grounding constraints for "
                        "this concept."
                    ),
                    source="critic_grounding_patch",
                )
                if patch_entry.key in entries:
                    entries[patch_entry.key].count += 1
                else:
                    entries[patch_entry.key] = patch_entry
        for shot_id, names in getattr(item, "missing_objects", {}).items():
            issue = f"Missing storyboard object names: {', '.join(names)}"
            entry = FailureMemoryEntry(
                concept_id=concept_id,
                shot_id=shot_id,
                category="missing_storyboard_objects",
                severity="warn",
                issue=issue,
                suggested_action="Instantiate exact storyboard object names and make them visible.",
                source="object_check",
            )
            if entry.key in entries:
                entries[entry.key].count += 1
            else:
                entries[entry.key] = entry
    return sorted(
        entries.values(),
        key=lambda e: (0 if e.severity == "block" else 1, -e.count, e.category),
    )


def _grounding_patch_from_issue(issue) -> dict:
    if issue.category != "concept_mismatch":
        return {}
    if issue.severity != "block" and not issue.suggested_fix.get(
        "_demoted_from_block"
    ):
        return {}
    anchors: list[str] = []
    labels: list[str] = []
    relationships: list[str] = []
    for raw_key, value in (issue.suggested_fix or {}).items():
        key = str(raw_key)
        if key.startswith("_"):
            continue
        root = key.split(".")[0]
        if "_" in root and any(ch.isalpha() for ch in root):
            anchors.append(root)
        if any(marker in key.lower() for marker in ("label", "formula", "text")):
            if isinstance(value, str):
                labels.append(value)
            else:
                labels.append(root)
        if any(
            marker in key.lower()
            for marker in (
                "pass_through", "on_image_plane", "projection_ray",
                "projected_silhouette",
            )
        ):
            relationships.append(key.replace("_", " "))
    low = issue.issue.lower()
    if "x = f" in low or "x=f" in low:
        labels.append("x = f X/Z")
    if "object_point_b" in low or "point b" in low:
        anchors.append("object_point_b")
        labels.append("B")
    if "projection ray" in low or "projection rays" in low:
        relationships.append(issue.issue[:180])
    def dedupe(values: list[str]) -> list[str]:
        out = []
        seen = set()
        for value in values:
            norm = " ".join(str(value).strip().split())
            key = norm.lower()
            if norm and key not in seen:
                out.append(norm)
                seen.add(key)
        return out
    patch = {
        "required_anchors": dedupe(anchors),
        "required_labels": dedupe(labels),
        "required_relationships": dedupe(relationships),
    }
    return {k: v for k, v in patch.items() if v}


@contextmanager
def _locked(path: Path, mode: str):
    """Open `path` with an advisory exclusive lock held for the duration.

    On non-POSIX systems we just open without locking — the WSL/Linux
    primary target gets the safety net, Windows-native is rare enough
    that we don't block it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open(mode, encoding="utf-8")
    try:
        if _HAS_FCNTL:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield fh
    finally:
        try:
            if _HAS_FCNTL:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _load_all_entries(path: Path) -> list[FailureMemoryEntry]:
    out: list[FailureMemoryEntry] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(FailureMemoryEntry.model_validate_json(line))
        except Exception:
            continue
    return out


def load_failure_memory(path: Path, concept_id: str, limit: int = 8) -> list[FailureMemoryEntry]:
    if not path.exists():
        return []
    # Read under lock so we don't observe a partial line written by a
    # concurrent appender.
    if _HAS_FCNTL:
        try:
            with _locked(path, "r") as fh:
                lines = fh.read().splitlines()
        except FileNotFoundError:
            return []
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[FailureMemoryEntry] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = FailureMemoryEntry.model_validate_json(line)
        except Exception:
            continue
        if entry.concept_id == concept_id:
            entries.append(entry)
    entries.sort(
        key=lambda e: (0 if e.severity == "block" else 1, -e.count, -e.last_seen),
    )
    return entries[:limit]


def _maybe_compact(path: Path) -> None:
    """If the file is bigger than the soft cap, dedupe by key and keep
    the most recent N entries per concept. Called under the same lock
    that owns the append, so no other writer can race us."""
    if not path.exists():
        return
    try:
        # Cheap pre-check: count newlines without parsing.
        line_count = sum(1 for _ in path.open("rb"))
    except OSError:
        return
    if line_count <= _COMPACTION_LINE_CAP:
        return

    all_entries = _load_all_entries(path)
    by_key: dict[tuple, FailureMemoryEntry] = {}
    for e in all_entries:
        existing = by_key.get(e.key)
        if existing is None:
            by_key[e.key] = e
        else:
            existing.count += e.count
            existing.last_seen = max(existing.last_seen, e.last_seen)

    per_concept: dict[str, list[FailureMemoryEntry]] = {}
    for e in by_key.values():
        per_concept.setdefault(e.concept_id, []).append(e)

    kept: list[FailureMemoryEntry] = []
    for concept_entries in per_concept.values():
        concept_entries.sort(
            key=lambda e: (
                0 if e.severity == "block" else 1,
                -e.count,
                -e.last_seen,
            )
        )
        kept.extend(concept_entries[:_COMPACTION_KEEP_PER_CONCEPT])

    tmp = path.with_suffix(path.suffix + ".compact.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for e in kept:
            fh.write(e.model_dump_json() + "\n")
    os.replace(tmp, path)


def append_failure_memory(path: Path, entries: list[FailureMemoryEntry]) -> None:
    if not entries:
        return
    with _locked(path, "a") as fh:
        for entry in entries:
            fh.write(entry.model_dump_json() + "\n")
        fh.flush()
    # Compact *after* releasing the append lock; compaction itself
    # takes a new exclusive lock for the rewrite.
    if _HAS_FCNTL:
        with _locked(path, "a"):
            _maybe_compact(path)
    else:
        _maybe_compact(path)


def save_failure_memory_snapshot(path: Path, entries: list[FailureMemoryEntry]) -> None:
    path.write_text(
        json.dumps([e.model_dump() for e in entries], indent=2),
        encoding="utf-8",
    )


def format_failure_memory_for_coder(entries: list[FailureMemoryEntry]) -> str:
    if not entries:
        return ""
    grouped: dict[tuple[str | None, str], list[FailureMemoryEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.shot_id, entry.category), []).append(entry)
    lines = [
        "STRUCTURED FAILURE MEMORY FROM PRIOR RUNS:",
        "Use these as cautionary constraints; do not overfit if they conflict with the current storyboard.",
        "Repeated patterns to actively avoid:",
    ]
    for (shot_id, category), group in sorted(
        grouped.items(),
        key=lambda item: (
            0 if any(e.severity == "block" for e in item[1]) else 1,
            -sum(e.count for e in item[1]),
            item[0][0] or "",
            item[0][1],
        ),
    )[:6]:
        shot = f" shot={shot_id}" if shot_id else ""
        top = sorted(group, key=lambda e: (-e.count, e.issue))[0]
        total = sum(e.count for e in group)
        lines.append(
            f"- [{category}{shot} seen={total}] {top.issue[:180]}"
        )
    lines.append("")
    lines.append("Detailed prior failures:")
    for i, entry in enumerate(entries[:8], 1):
        shot = f" shot={entry.shot_id}" if entry.shot_id else ""
        lines.append(
            f"{i}. [{entry.severity}/{entry.category}{shot} count={entry.count}] "
            f"{entry.issue[:220]}"
        )
        lines.append(f"   avoid by: {entry.suggested_action}")
    return "\n".join(lines)
