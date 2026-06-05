"""Concurrency + size-cap behaviour for failure_memory.

These two were the audit's primary concerns about ``failure_memory.jsonl``:
appends from concurrent runs must not interleave half-lines, and the
file must not grow without bound.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cg_tutor import failure_memory
from cg_tutor.failure_memory import (
    FailureMemoryEntry,
    append_failure_memory,
    load_failure_memory,
)


def _entry(concept_id: str, issue: str) -> FailureMemoryEntry:
    return FailureMemoryEntry(
        concept_id=concept_id,
        category="other",
        severity="block",
        issue=issue,
        suggested_action="x",
    )


def test_concurrent_appends_produce_well_formed_json_lines(tmp_path):
    path = tmp_path / "memory.jsonl"
    n_threads = 8
    n_per_thread = 25

    def worker(i):
        entries = [
            _entry("concept", f"issue from t{i} #{k}")
            for k in range(n_per_thread)
        ]
        append_failure_memory(path, entries)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every line must be valid JSON of the expected shape.
    lines = path.read_text().splitlines()
    assert len(lines) == n_threads * n_per_thread
    for line in lines:
        parsed = json.loads(line)
        assert parsed["concept_id"] == "concept"
        assert parsed["category"] == "other"


def test_compaction_keeps_per_concept_cap(tmp_path, monkeypatch):
    """Compaction is amortized: when a fresh append crosses the soft cap,
    we rewrite the file to keep at most ``_COMPACTION_KEEP_PER_CONCEPT``
    entries per concept. Subsequent appends accumulate again until the
    next threshold crossing — so the steady-state size is
    ``KEEP_PER_CONCEPT + (LINE_CAP - KEEP_PER_CONCEPT)`` at most."""
    path = tmp_path / "memory.jsonl"
    monkeypatch.setattr(failure_memory, "_COMPACTION_LINE_CAP", 50)
    monkeypatch.setattr(failure_memory, "_COMPACTION_KEEP_PER_CONCEPT", 5)
    for i in range(80):
        append_failure_memory(path, [_entry("concept", f"distinct_issue_{i}")])
    lines = path.read_text().splitlines()
    # Upper bound is (LINE_CAP - 1) + new appends since last compaction.
    # With 80 distinct entries and compaction at 50, exactly one
    # compaction round fires (collapsing 51 → 5), then 29 more appends
    # leave us with 34 lines, well below the cap.
    assert len(lines) <= 50
    # And the file MUST have been compacted at least once — without
    # compaction we'd have 80 lines, with compaction at most ~35.
    assert len(lines) < 80
    for line in lines:
        parsed = json.loads(line)
        assert parsed["concept_id"] == "concept"


def test_compaction_collapses_duplicate_keys_with_summed_count(tmp_path, monkeypatch):
    """Duplicate keys must collapse to one entry with the count summed
    across every appended copy. After enough appends, the surviving
    entry's count reflects ALL of the appends, not just the last batch."""
    path = tmp_path / "memory.jsonl"
    monkeypatch.setattr(failure_memory, "_COMPACTION_LINE_CAP", 5)
    monkeypatch.setattr(failure_memory, "_COMPACTION_KEEP_PER_CONCEPT", 10)
    for _ in range(30):
        append_failure_memory(path, [_entry("concept", "same issue")])
    lines = path.read_text().splitlines()
    # Same key across all appends → at most LINE_CAP-ish surviving lines,
    # but the compacted line's count should reflect the cumulative seen.
    by_key = [json.loads(line) for line in lines]
    total_seen = sum(p["count"] for p in by_key)
    assert total_seen == 30


def test_load_filters_by_concept_id_after_compaction(tmp_path, monkeypatch):
    path = tmp_path / "memory.jsonl"
    monkeypatch.setattr(failure_memory, "_COMPACTION_LINE_CAP", 5)
    monkeypatch.setattr(failure_memory, "_COMPACTION_KEEP_PER_CONCEPT", 3)
    for i in range(10):
        append_failure_memory(path, [_entry("alpha", f"x{i}")])
        append_failure_memory(path, [_entry("beta", f"y{i}")])
    alpha = load_failure_memory(path, "alpha")
    beta = load_failure_memory(path, "beta")
    assert all(e.concept_id == "alpha" for e in alpha)
    assert all(e.concept_id == "beta" for e in beta)
