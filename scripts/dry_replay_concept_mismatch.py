"""Dry-replay the new concept_mismatch critic check against existing runs.

This re-runs `render_critic.inspect` against the rendered frames of an
already-completed concept (e.g. `outputs/mirror_reflection/`)
using the *new* prompt + visual_intent wiring, but without touching
storyboards, scene.py, or the original critic outputs.

Use it to answer one question before trusting concept_mismatch in a real
critic loop: does the LLM critic actually emit concept_mismatch on the
shots where we know the concept is not visually demonstrated?

Usage:
    python scripts/dry_replay_concept_mismatch.py \\
        outputs/mirror_reflection \\
        --backend claude

Reports per-shot which categories fired. Saves nothing — pure dry-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cg_tutor.agents import render_critic  # noqa: E402
from cg_tutor.schemas import Narrative, Storyboard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("concept_dir", type=Path,
                    help="e.g. outputs/mirror_reflection")
    ap.add_argument("--backend", default="claude",
                    help="render_critic backend (default: claude). "
                         "Use 'passthrough' for a no-LLM smoke test.")
    args = ap.parse_args()

    cd = args.concept_dir
    narrative = Narrative.model_validate_json(
        (cd / "narrative.json").read_text()
    )
    storyboard = Storyboard.model_validate_json(
        (cd / "storyboard.json").read_text()
    )
    frames_dir = cd / "frames"

    print(f"[dry-replay] concept = {cd.name}")
    print(f"[dry-replay] backend = {args.backend}")
    print(f"[dry-replay] {len(narrative.nodes)} narrative nodes, "
          f"{len(storyboard.shots)} shots, "
          f"{len(list(frames_dir.glob('frame_*.png')))} frames")
    print()
    print("visual_intent per shot (what concept_mismatch will judge against):")
    for node in narrative.nodes:
        intent_preview = node.visual_intent[:120].replace("\n", " ")
        print(f"  {node.id}: {intent_preview!r}")
    print()

    # Run the critic WITHOUT writing critic_iterNN.json (out_dir=None).
    # This protects existing artifacts and keeps the replay purely
    # observational.
    report = render_critic.inspect(
        storyboard, frames_dir, iteration=999,
        out_dir=None, backend=args.backend,
        narrative=narrative,
    )

    print(f"overall_score = {report.overall_score:.3f}")
    print(f"n_issues = {len(report.issues)}")
    print()

    by_cat: Counter[str] = Counter()
    for i in report.issues:
        by_cat[i.category] += 1
    if by_cat:
        print("issues by category:")
        for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
            marker = "  ←" if cat == "concept_mismatch" else ""
            print(f"  {cat:>20}: {n}{marker}")
    print()

    cm_issues = [i for i in report.issues if i.category == "concept_mismatch"]
    if cm_issues:
        print(f"concept_mismatch detail ({len(cm_issues)} issues):")
        for i in cm_issues:
            print(f"  [{i.severity}] {i.shot_id} (frame {i.frame_idx}): "
                  f"{i.issue[:180]}")
    else:
        print("concept_mismatch: 0 issues fired.")
        print("  (this may mean the model is too lenient, the prompt rule "
              "needs sharpening, or the rendered shots actually do "
              "demonstrate the concept.)")

    print()
    print("==========")
    print("DECISION HEURISTIC:")
    print("  >=2 concept_mismatch fires on a known-bad concept")
    print("  → defence-line-1 is worth keeping in the loop.")
    print("  0 fires on a known-bad concept → tighten the prompt or try a")
    print("  different vision model before enabling in production runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
