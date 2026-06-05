"""Continue a critic-loop run with one more iteration without re-doing
prior iterations.

Reads the latest critic_iterNN.json + storyboard.json from <out_dir>,
asks the Coder for ONE more retry, renders, runs Critic, composes mp4.

Usage:
    python scripts/continue_critic.py outputs/mirror_reflection
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cg_tutor.agents import blender_coder, render_critic  # noqa: E402
from cg_tutor.agents.base import save_artifact  # noqa: E402
from cg_tutor.agents.render_critic import issues_as_coder_addendum  # noqa: E402
from cg_tutor.blender import runtime as blender_runtime  # noqa: E402
from cg_tutor.composer.compose import compose_storyboard_video  # noqa: E402
from cg_tutor.schemas import CriticReport, Storyboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="e.g. outputs/mirror_reflection")
    ap.add_argument("--blender-timeout", type=int, default=3600)
    ap.add_argument("--coder", default=None)
    ap.add_argument("--only-block", action="store_true",
                    help="Filter critic feedback to severity=block before "
                         "sending to coder. Keeps the retry prompt small and "
                         "focuses the model on critical issues.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    sb = Storyboard.model_validate_json((out_dir / "storyboard.json").read_text())

    # Find the highest existing iter and load its critic report.
    crit_files = sorted(out_dir.glob("critic_iter*.json"))
    if not crit_files:
        sys.exit("no critic_iter*.json found in out_dir")
    last_crit_path = crit_files[-1]
    last_iter = int(last_crit_path.stem.replace("critic_iter", ""))
    next_iter = last_iter + 1
    print(f"continuing from iter{last_iter:02d}; running iter{next_iter:02d}")
    critic_report = CriticReport.model_validate_json(last_crit_path.read_text())
    print(f"  prior score={critic_report.overall_score:.2f}  "
          f"block={sum(1 for i in critic_report.issues if i.severity == 'block')}  "
          f"warn={sum(1 for i in critic_report.issues if i.severity == 'warn')}")

    if args.only_block:
        critic_report = critic_report.model_copy(update={
            "issues": [i for i in critic_report.issues if i.severity == "block"],
        })
        print(f"  filtered to {len(critic_report.issues)} block issue(s) for retry")

    t0 = time.time()

    # Coder retry
    print("\n[continue] (3/7) coder retry with critic feedback")
    addendum = issues_as_coder_addendum(critic_report)
    coder_kwargs = dict(out_dir=out_dir, iteration=next_iter, addendum=addendum)
    if args.coder:
        coder_kwargs["model"] = args.coder
    code = blender_coder.to_bpy_script(sb, **coder_kwargs)
    print(f"           scene.py: {len(code.splitlines())} lines")

    # Render
    print("[continue] (4/7) render")
    frames_dir = out_dir / "frames"
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()
    os.environ["CG_TUTOR_OUT_DIR"] = str(frames_dir.resolve())
    rr = blender_runtime.run_script(
        out_dir / "scene.py", frames_dir, timeout_sec=args.blender_timeout,
    )
    n_frames = len(list(frames_dir.glob("frame_*.png")))
    save_artifact(out_dir, "blender_stderr.txt", rr.stderr or "")
    save_artifact(out_dir, "blender_stdout.txt", rr.stdout or "")
    if n_frames == 0:
        sys.exit(f"render produced 0 frames; see {out_dir}/blender_stderr.txt")
    print(f"           rendered {n_frames} frames  (ok={rr.ok})")

    # Critic
    print("[continue] (6/7) critic")
    new_report = render_critic.inspect(
        sb, frames_dir, iteration=next_iter, out_dir=out_dir,
    )
    n_block = sum(1 for i in new_report.issues if i.severity == "block")
    n_warn = sum(1 for i in new_report.issues if i.severity == "warn")
    print(f"           score={new_report.overall_score:.2f}  "
          f"block={n_block} warn={n_warn}  "
          f"pass={new_report.pass_threshold}")

    # Overlay + compose (shared helper with the main pipeline).
    print("[continue] (5+7/7) overlay + compose")
    res = compose_storyboard_video(sb, frames_dir, out_dir)
    if not res.ok:
        sys.exit(f"ffmpeg failed: {res.stderr[:400]}")
    size_kb = (out_dir / "final.mp4").stat().st_size // 1024
    print(f"\n[continue] DONE in {time.time() - t0:.1f}s — "
          f"{out_dir}/final.mp4 ({size_kb} KB)")


if __name__ == "__main__":
    main()
