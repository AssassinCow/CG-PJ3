"""CLI entry: run the full pipeline on one concept YAML.

Examples:
    python scripts/run_concept.py prism_dispersion_teaching
    python scripts/run_concept.py mirror_reflection --render-engine CYCLES
    python scripts/run_concept.py configs/concepts/shape_morphing.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cg_tutor import pipeline  # noqa: E402
from cg_tutor._logging import configure_logging  # noqa: E402


def _resolve_concept(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    candidate = REPO_ROOT / "configs" / "concepts" / f"{arg}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"concept not found: tried {p} and {candidate}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("concept", help="concept_id (looked up in configs/concepts/) "
                                    "or a path to a concept YAML")
    ap.add_argument("--decomposer", default=None)
    ap.add_argument("--storyboard", default=None)
    ap.add_argument("--coder", default=None)
    ap.add_argument("--out-root", default=str(REPO_ROOT / "outputs"),
                    help="Where outputs/<concept>/ goes (default outputs/)")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse cached narrative/storyboard/scene if present "
                         "and continue from saved critic history.")
    ap.add_argument("--keep-frames", action="store_true",
                    help="Skip Blender; reuse PNGs already in outputs/<id>/frames/")
    ap.add_argument("--blender-timeout", type=int, default=3600,
                    help="Per-render timeout in seconds (default 3600)")
    ap.add_argument("--critic", default=None,
                    choices=[
                        None,
                        "passthrough",
                        "api",
                        "claude",
                        "gpt",
                        "gemini",
                        "claude-api",
                        "codex-api",
                        "gemini-api",
                        "google",
                        "claude-cli",
                    ],
                    help="Critic backend. None = auto "
                         "(configured API vision, else claude-cli, else google, "
                         "else passthrough)")
    ap.add_argument("--max-critic-iters", type=int, default=5,
                    help="Max coder->render->critic retries (0 = no critic loop). "
                         "Default 5 for quality-first runs.")
    ap.add_argument("--early-stop-stale-iters", type=int, default=0,
                    help="Exit the critic loop after this many iterations "
                         "with no block-floor improvement (0 = disable). "
                         "Default 0, disabled.")
    ap.add_argument("--best-selection", default="balanced",
                    choices=["framing", "balanced", "semantic"],
                    help="How to choose the final/baseline best critic iter. "
                         "framing = conservative layout-first default; "
                         "balanced = semantic blockers matter before score; "
                         "semantic = total block count first.")
    ap.add_argument("--compiler-seed", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Generate a deterministic scene.compiled.py and feed "
                         "it to the coder as a scaffold.")
    ap.add_argument("--compiler-only", action="store_true",
                    help="Use the deterministic Scene IR -> bpy compiler for "
                         "iter00 instead of calling the coder.")
    ap.add_argument("--preview-render", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Before each full render, render one keyframe per "
                         "shot when scene.py supports CG_TUTOR_PREVIEW_FRAMES.")
    ap.add_argument("--diff-repair", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="On critic retries, ask the coder for a unified diff "
                         "against the best scene; fall back to full script "
                         "regeneration if patching fails.")
    ap.add_argument("--critic-ensemble", default="claude,gpt",
                    help="Comma-separated critic backends to run and "
                         "aggregate, e.g. claude,gpt or claude,gemini. "
                         "When set, this overrides --critic for scoring.")
    ap.add_argument("--critic-strictness", default="strict",
                    choices=["consensus", "union", "strict"],
                    help="How ensemble critic blocks are aggregated. "
                         "consensus requires multi-critic agreement; union "
                         "keeps any member block; strict = union plus score/"
                         "member-block disagreement blockers. Default strict.")
    ap.add_argument("--render-engine", default="BLENDER_EEVEE",
                    choices=["BLENDER_EEVEE", "CYCLES"],
                    help="Blender render engine for generated scenes. "
                         "Default BLENDER_EEVEE for speed; pass CYCLES for "
                         "scenes that need physically cleaner shadows/DoF.")
    ap.add_argument("--cycles-device", default="AUTO",
                    choices=["AUTO", "GPU", "CPU"],
                    help="Cycles render device when --render-engine CYCLES. "
                         "AUTO/GPU try Blender's GPU backends (OPTIX, CUDA, "
                         "HIP, METAL, ONEAPI) before render; CPU forces CPU. "
                         "Default AUTO.")
    ap.add_argument("--strict-best-replay", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Abort before composing final.mp4 if the re-rendered "
                         "best scene's frame hash differs from the critic-"
                         "scored hash (Blender non-determinism). Default ON: "
                         "the critic's score and the composed mp4 must match. "
                         "Pass --no-strict-best-replay to fall back to "
                         "WARN-and-continue.")
    ap.add_argument("--log-level", default=None,
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity. Falls back to "
                         "CG_TUTOR_LOG_LEVEL env var, then INFO.")
    ap.add_argument("--max-verifier-repair-iters", type=int, default=2,
                    help="Max passes through the verifier repair loop when "
                         "the deterministic verifier reports block issues. "
                         "Default 2: a Tier-1 cheap pre-render loop that "
                         "absorbs structural failures (missing anchors, "
                         "absent text/vector objects) before any Blender "
                         "render or VLM critic is invoked.")
    args = ap.parse_args()

    configure_logging(args.log_level)
    concept_path = _resolve_concept(args.concept)
    result = pipeline.run(
        concept_path,
        out_root=Path(args.out_root),
        decomposer_model=args.decomposer,
        storyboard_model=args.storyboard,
        coder_model=args.coder,
        resume=args.resume,
        keep_frames=args.keep_frames,
        blender_timeout_sec=args.blender_timeout,
        critic_backend=args.critic,
        max_critic_iterations=args.max_critic_iters,
        early_stop_stale_iters=args.early_stop_stale_iters,
        best_selection_mode=args.best_selection,
        compiler_seed=args.compiler_seed,
        compiler_only=args.compiler_only,
        preview_render=args.preview_render,
        diff_repair=args.diff_repair,
        critic_ensemble=tuple(
            s.strip() for s in args.critic_ensemble.split(",") if s.strip()
        ),
        critic_strictness=args.critic_strictness,
        render_engine=args.render_engine,
        cycles_device=args.cycles_device,
        strict_best_replay=args.strict_best_replay,
        max_verifier_repair_iters=args.max_verifier_repair_iters,
    )

    if not result.ok:
        sys.exit(1)
    # Final path goes to stdout (not the logging stderr stream) so shell
    # consumers can capture it cleanly: ``mp4=$(... | tail -1)``.
    print(result.final_mp4)


if __name__ == "__main__":
    main()
