"""Manage the manual evaluation YAML.

Usage:
    # Create a blank rating sheet for all concepts under outputs/
    python scripts/eval_concepts.py init

    # Print a summary
    python scripts/eval_concepts.py summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cg_tutor.eval.metrics import AXES, EvalSet, blank_entry  # noqa: E402


DEFAULT_SHEET = REPO_ROOT / "eval" / "manual_scores.yaml"


def _list_concepts(outputs_dir: Path) -> list[str]:
    return sorted(
        d.name for d in outputs_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "final.mp4").exists()
    )


def cmd_init(args: argparse.Namespace) -> None:
    sheet_path = Path(args.path)
    outputs = Path(args.outputs_dir)
    concept_ids = _list_concepts(outputs)
    if not concept_ids:
        print(f"no concepts with final.mp4 under {outputs}")
        return
    existing = set()
    if sheet_path.exists():
        existing = {e.concept_id for e in EvalSet.from_yaml(sheet_path).entries}
        es = EvalSet.from_yaml(sheet_path)
    else:
        es = EvalSet(name="cg-tutor manual scores")
    for cid in concept_ids:
        if cid in existing:
            continue
        es.entries.append(blank_entry(cid, rater=args.rater))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    es.to_yaml(sheet_path)
    print(f"wrote {sheet_path}  ({len(es.entries)} entries; "
          f"axes={list(AXES)})")
    print("open the file and fill `scores` (1..5 ints) plus `notes`.")


def cmd_summary(args: argparse.Namespace) -> None:
    sheet_path = Path(args.path)
    if not sheet_path.exists():
        print(f"no eval sheet at {sheet_path}; run `init` first")
        sys.exit(1)
    es = EvalSet.from_yaml(sheet_path)
    s = es.summary()
    print(f"sheet: {sheet_path}")
    print(f"n={s['n']}  overall_mean={s.get('overall_mean')}  "
          f"pass_rate={s.get('pass_rate')}")
    print("per-axis means:")
    for a, v in s.get("axis_means", {}).items():
        print(f"  {a:14s} {v}")


def cmd_critic_summary(args: argparse.Namespace) -> None:
    outputs = Path(args.outputs_dir)
    if not outputs.exists():
        print(f"no outputs dir at {outputs}")
        sys.exit(1)
    print(f"outputs: {outputs}")
    for concept_dir in sorted(d for d in outputs.iterdir() if d.is_dir()):
        critics = sorted(concept_dir.glob("critic_iter*.json"))
        if not critics:
            continue
        last = critics[-1]
        data = json.loads(last.read_text())
        issues = data.get("issues", [])
        blocks = sum(1 for i in issues if i.get("severity") == "block")
        warns = sum(1 for i in issues if i.get("severity") == "warn")
        score = float(data.get("overall_score", 0.0))
        passed = score > 0.7 and blocks == 0
        print(
            f"{concept_dir.name:24s} {last.stem:13s} "
            f"score={score:.2f} pass={passed} block={blocks} warn={warns}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEFAULT_SHEET))
    sub = ap.add_subparsers(dest="cmd", required=True)
    init_p = sub.add_parser("init")
    init_p.add_argument("--rater", default="anon")
    init_p.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    init_p.set_defaults(func=cmd_init)
    sum_p = sub.add_parser("summary")
    sum_p.set_defaults(func=cmd_summary)
    critic_p = sub.add_parser("critic-summary")
    critic_p.add_argument("--outputs-dir", default=str(REPO_ROOT / "outputs"))
    critic_p.set_defaults(func=cmd_critic_summary)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
