"""Render Critic Agent.

Inspects rendered keyframes against the storyboard and emits a
CriticReport. Three backends:

  - `inspect_passthrough`: returns a perfect score; used when the user
    disables the critic loop.
  - `inspect_api_vision`: calls an OpenAI-compatible vision endpoint
    with PNG frames as data URLs.
  - `inspect_claude_cli`: shells out to `claude -p`, passing PNG file
    paths in the prompt. Claude Code's Read tool loads them, and the
    Claude model itself does the vision analysis.
  - `inspect_gemini`: direct Google Gemini Vision SDK call. Cheaper
    per call than claude-cli but requires GOOGLE_API_KEY.

Auto-select order: configured API vision chain → claude-cli (if
`claude` on PATH) → google (if GOOGLE_API_KEY set) → passthrough.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from cg_tutor.agents.base import load_prompt, save_artifact
from cg_tutor.config import get_agent_model
from cg_tutor.llm_client import LLMClient
from cg_tutor.scene_profiles import SceneProfile, format_scene_profile_for_prompt
from cg_tutor.schemas import CriticIssue, CriticReport, Narrative, Storyboard
from cg_tutor.visual_contract import build_visual_contract, format_visual_contract


AGENT = "render_critic"
CriticStrictness = str
_CRITIC_STRICTNESS_VALUES = {"consensus", "union", "strict"}
_STRICT_BLOCK_COUNT_THRESHOLD = 3
_STRICT_SCORE_SPREAD_THRESHOLD = 0.25


# Mapping shot index → list of frame indices to inspect (1-based, matches
# Blender's frame numbering). We pick 3 frames per shot by default:
# 25 %, 50 %, 75 % of the shot's duration.
def _sample_keyframes(storyboard: Storyboard) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    cursor_frames = 0
    for idx, shot in enumerate(storyboard.shots):
        n_frames = max(1, round(shot.duration_sec * storyboard.fps))
        start = cursor_frames + 1
        end = cursor_frames + n_frames
        # 25%, 50%, 75% through the shot, but never the first or last
        # (transient camera setup at boundaries skews critic).
        picks = [
            int(start + 0.25 * (end - start)),
            int(start + 0.50 * (end - start)),
            int(start + 0.75 * (end - start)),
        ]
        out[idx] = sorted(set(picks))
        cursor_frames = end
    return out


def _frame_path(frames_dir: Path, frame_idx: int) -> Path:
    return frames_dir / f"frame_{frame_idx:04d}.png"


def _overall_from_scores(scores: list[float], exec_errors: list[str]) -> float:
    """Keep critic transport failures distinct from a clean no-issue score."""
    if scores:
        return sum(scores) / len(scores)
    return 0.0 if exec_errors else 1.0


def _visual_intent_lookup(narrative: Narrative | None) -> dict[str, str]:
    """Build node_id -> visual_intent map. Empty dict when narrative is None."""
    if narrative is None:
        return {}
    return {n.id: n.visual_intent for n in narrative.nodes if n.visual_intent}


def _node_lookup(narrative: Narrative | None) -> dict[str, object]:
    if narrative is None:
        return {}
    return {n.id: n for n in narrative.nodes}


def _visual_contract_text(
    storyboard: Storyboard,
    shot_idx: int,
    narrative: Narrative | None,
    scene_profile: SceneProfile | None = None,
) -> str:
    nodes = _node_lookup(narrative)
    shot = storyboard.shots[shot_idx]
    contract = build_visual_contract(
        shot, nodes.get(shot.node_id), scene_profile=scene_profile,
    )
    lines = format_visual_contract(contract)
    if not lines:
        return ""
    return "\n".join(lines)


# ----- backends -----------------------------------------------------------


def inspect_passthrough(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    iteration: int = 0,
    out_dir: Path | None = None,
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
) -> CriticReport:
    """Always returns a perfect score. Used when critic is disabled."""
    report = CriticReport(
        concept_id=storyboard.concept_id, iteration=iteration,
        overall_score=1.0, issues=[],
    )
    if out_dir is not None:
        save_artifact(out_dir, f"critic_iter{iteration:02d}.json",
                      report.model_dump_json(indent=2))
    return report


def inspect_claude_cli(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    iteration: int = 0,
    out_dir: Path | None = None,
    model: str = "sonnet",
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
) -> CriticReport:
    """Use Claude Code CLI as a VLM. Each shot gets one `claude -p` call
    that references its sampled keyframe paths; Claude's Read tool loads
    the PNGs as image inputs, and the model returns a per-shot
    CriticReport-shaped JSON. We aggregate across shots.
    """
    if not shutil.which("claude"):
        raise RuntimeError("`claude` CLI not on PATH")

    system = load_prompt("critic")
    samples = _sample_keyframes(storyboard)
    intents = _visual_intent_lookup(narrative)
    all_issues: list[CriticIssue] = []
    exec_errors: list[str] = []
    scores: list[float] = []

    for shot_idx, frame_indices in samples.items():
        shot = storyboard.shots[shot_idx]
        frame_files = [_frame_path(frames_dir, f) for f in frame_indices]
        frame_files = [(f, fi) for f, fi in zip(frame_files, frame_indices)
                       if f.exists()]
        if not frame_files:
            exec_errors.append(
                f"{shot.node_id}: no sampled frames available for critic"
            )
            continue

        # Prompt: Claude Code agent will Read each PNG (its tool loads
        # them as images for the model), then emit the CriticReport JSON.
        user_lines = [
            "Inspect this shot against the rendered frames.",
            "",
            "Shot JSON:",
            shot.model_dump_json(indent=2),
        ]
        intent = intents.get(shot.node_id)
        if intent:
            user_lines += [
                "",
                "visual_intent (what the viewer should see for the concept "
                "to be demonstrated — use for concept_mismatch judgement):",
                intent,
            ]
        contract_text = _visual_contract_text(
            storyboard, shot_idx, narrative, scene_profile,
        )
        if contract_text:
            user_lines += [
                "",
                "derived_visual_contract (object-level checks; report "
                "concept_mismatch when these required elements are absent "
                "or unreadable):",
                contract_text,
            ]
        profile_text = format_scene_profile_for_prompt(scene_profile)
        if profile_text:
            user_lines += [
                "",
                "scene_profile (style policy; report concept_mismatch when "
                "the render violates forbidden helpers, style policy, or "
                "adaptive_critic_rubric):",
                profile_text,
            ]
        user_lines += [
            "",
            "Read each of these PNG files (use your Read tool — they are "
            "image files, not text) and analyse them collectively:",
        ]
        for f, fi in frame_files:
            user_lines.append(f"  - {f.resolve()}   (frame_idx={fi})")
        user_lines.append("")
        user_lines.append(
            "Return ONLY the JSON object matching the CriticReport schema "
            "described in your system prompt. Do not add prose."
        )
        user = "\n".join(user_lines)

        cmd = ["claude", "-p", "--output-format", "json",
               "--model", model, "--system-prompt", system]
        proc = subprocess.run(
            cmd, input=user, capture_output=True, text=True,
            timeout=int(os.environ.get("CG_TUTOR_LLM_TIMEOUT", "300")),
            check=False,
        )
        if proc.returncode != 0:
            exec_errors.append(
                f"{shot.node_id}: claude-cli failed: {proc.stderr[:200]}"
            )
            continue
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            exec_errors.append(
                f"{shot.node_id}: claude-cli envelope unparseable"
            )
            continue
        if envelope.get("is_error"):
            exec_errors.append(
                f"{shot.node_id}: claude-cli reported is_error"
            )
            continue
        result = envelope.get("result", "")
        try:
            data = _parse_json_from_text(result)
        except (json.JSONDecodeError, ValueError):
            exec_errors.append(
                f"{shot.node_id}: critic returned unparseable JSON"
            )
            continue

        scores.append(float(data.get("overall_score", 1.0)))
        for raw in data.get("issues", []):
            try:
                # Coerce missing required fields with reasonable defaults
                raw.setdefault("shot_id", shot.node_id)
                raw.setdefault("frame_idx", frame_indices[0])
                raw.setdefault("severity", "warn")
                raw.setdefault("category", "other")
                all_issues.append(CriticIssue.model_validate(raw))
            except Exception:  # noqa: BLE001
                pass

    overall = _overall_from_scores(scores, exec_errors)
    report = CriticReport(
        concept_id=storyboard.concept_id, iteration=iteration,
        overall_score=overall, issues=all_issues,
        execution_errors=exec_errors,
    )
    if out_dir is not None:
        save_artifact(out_dir, f"critic_iter{iteration:02d}.json",
                      report.model_dump_json(indent=2))
    return report


def inspect_api_vision(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    iteration: int = 0,
    out_dir: Path | None = None,
    model: str | None = None,
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
) -> CriticReport:
    """Use OpenAI-compatible vision chat completions to score frames."""
    cfg = get_agent_model(AGENT)
    if model:
        chain = (model,)
    else:
        chain = cfg.chain or ("anthropic/claude-sonnet-4.6",)
    client = LLMClient.from_chain(chain)
    system = load_prompt("critic")
    samples = _sample_keyframes(storyboard)
    intents = _visual_intent_lookup(narrative)
    all_issues: list[CriticIssue] = []
    exec_errors: list[str] = []
    scores: list[float] = []

    for shot_idx, frame_indices in samples.items():
        shot = storyboard.shots[shot_idx]
        frame_files = [_frame_path(frames_dir, f) for f in frame_indices]
        frame_pairs = [(f, fi) for f, fi in zip(frame_files, frame_indices)
                       if f.exists()]
        if not frame_pairs:
            exec_errors.append(
                f"{shot.node_id}: no sampled frames available for critic"
            )
            continue
        frame_pair = frame_pairs[len(frame_pairs) // 2]

        user_lines = [
            "Inspect this shot against the rendered frames.",
            "",
            "Shot summary JSON:",
            json.dumps(
                {
                    "concept_id": storyboard.concept_id,
                    "iteration": iteration,
                    "shot_id": shot.node_id,
                    "start_sec": shot.start_sec,
                    "duration_sec": shot.duration_sec,
                    "formula": shot.formula,
                    "overlay_zone": (
                        shot.overlay_zone.model_dump(mode="json")
                        if shot.overlay_zone is not None
                        else None
                    ),
                    "object_names": [obj.name for obj in shot.objects],
                    "visual_intent": intents.get(shot.node_id, ""),
                    "derived_visual_contract": _visual_contract_text(
                        storyboard, shot_idx, narrative, scene_profile,
                    ),
                    "scene_profile": (
                        scene_profile.model_dump()
                        if scene_profile is not None
                        else None
                    ),
                },
                indent=2,
            ),
            "",
            "The attached image is a representative sampled frame for this shot:",
        ]
        user_lines.append(f"  - frame_idx={frame_pair[1]}")
        user_lines.append("")
        user_lines.append(
            "Return ONLY the JSON object matching the CriticReport schema "
            "described in your system prompt. Do not add prose."
        )

        try:
            result = client.complete_with_images(
                system=system,
                user="\n".join(user_lines),
                # API vision requests can be noticeably slower than text
                # calls. One representative mid-shot frame catches the
                # common failures while keeping latency predictable.
                image_paths=[frame_pair[0]],
                response_format="json",
                temperature=0.2,
                max_tokens=2048,
                raw_out_dir=out_dir,
            )
            data = _parse_json_from_text(result)
        except Exception as e:  # noqa: BLE001
            exec_errors.append(
                f"{shot.node_id}: api vision critic failed: {str(e)[:200]}"
            )
            continue

        scores.append(float(data.get("overall_score", 1.0)))
        for raw in data.get("issues", []):
            try:
                raw.setdefault("shot_id", shot.node_id)
                raw.setdefault("frame_idx", frame_pairs[0][1])
                raw.setdefault("severity", "warn")
                raw.setdefault("category", "other")
                all_issues.append(CriticIssue.model_validate(raw))
            except Exception:  # noqa: BLE001
                pass

    overall = _overall_from_scores(scores, exec_errors)
    report = CriticReport(
        concept_id=storyboard.concept_id, iteration=iteration,
        overall_score=overall, issues=all_issues,
        execution_errors=exec_errors,
    )
    if out_dir is not None:
        save_artifact(out_dir, f"critic_iter{iteration:02d}.json",
                      report.model_dump_json(indent=2))
    return report


def _parse_json_from_text(text: str) -> dict:
    """Best-effort JSON extraction tolerating ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (and language tag)
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Find the outermost {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def inspect_gemini(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    iteration: int = 0,
    out_dir: Path | None = None,
    model: str = "gemini-2.0-flash",
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
) -> CriticReport:
    """Use Google Gemini's vision endpoint to score rendered keyframes.

    Sends one request per shot containing the shot's storyboard fragment
    + the sampled keyframe images. Aggregates issues across shots.
    """
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gen_model = genai.GenerativeModel(model)

    system = load_prompt("critic")
    samples = _sample_keyframes(storyboard)
    intents = _visual_intent_lookup(narrative)
    all_issues: list[CriticIssue] = []
    exec_errors: list[str] = []
    scores: list[float] = []

    for shot_idx, frame_indices in samples.items():
        shot = storyboard.shots[shot_idx]
        frame_files = [_frame_path(frames_dir, f) for f in frame_indices]
        frame_files = [f for f in frame_files if f.exists()]
        if not frame_files:
            exec_errors.append(
                f"{shot.node_id}: no sampled frames available for critic"
            )
            continue

        contents = [system, "\n\nShot JSON:\n",
                    shot.model_dump_json(indent=2)]
        intent = intents.get(shot.node_id)
        if intent:
            contents += [
                "\n\nvisual_intent (what the viewer should see for the "
                "concept to be demonstrated; use for concept_mismatch "
                "judgement):\n",
                intent,
            ]
        contract_text = _visual_contract_text(
            storyboard, shot_idx, narrative, scene_profile,
        )
        if contract_text:
            contents += [
                "\n\nderived_visual_contract (object-level checks; report "
                "concept_mismatch when these required elements are absent "
                "or unreadable):\n",
                contract_text,
            ]
        profile_text = format_scene_profile_for_prompt(scene_profile)
        if profile_text:
            contents += [
                "\n\nscene_profile (style policy; report concept_mismatch "
                "when the render violates forbidden helpers, style policy, "
                "or adaptive_critic_rubric):\n",
                profile_text,
            ]
        contents.append("\n\nFrames (in order):")
        for f, idx in zip(frame_files, frame_indices):
            contents.append(f"\nframe_idx={idx}:")
            contents.append(Image.open(f))

        resp = gen_model.generate_content(
            contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, ValueError):
            exec_errors.append(
                f"{shot.node_id}: gemini VLM returned unparseable JSON"
            )
            continue

        scores.append(float(data.get("overall_score", 1.0)))
        for raw in data.get("issues", []):
            try:
                all_issues.append(CriticIssue.model_validate(raw))
            except Exception:  # noqa: BLE001
                pass  # skip malformed issue entries

    overall = _overall_from_scores(scores, exec_errors)
    report = CriticReport(
        concept_id=storyboard.concept_id, iteration=iteration,
        overall_score=overall, issues=all_issues,
        execution_errors=exec_errors,
    )
    if out_dir is not None:
        save_artifact(out_dir, f"critic_iter{iteration:02d}.json",
                      report.model_dump_json(indent=2))
    return report


# ----- dispatch -----------------------------------------------------------


def _select_backend(prefer: str | None) -> Callable:
    """Pick a backend by name, or auto-detect."""
    if prefer == "passthrough":
        return inspect_passthrough
    if prefer in (
        "api", "claude", "gpt", "gemini",
        "gemini-api", "claude-api", "codex-api",
        "openai", "anthropic", "google-api",
    ):
        return inspect_api_vision
    if prefer == "claude-cli":
        return inspect_claude_cli
    if prefer == "google":
        return inspect_gemini
    # auto: use the configured critic chain kind first, then local Claude,
    # then Gemini, then passthrough.
    chain = get_agent_model(AGENT).chain
    if any(s.startswith((
        "claude/", "gpt/", "gemini/",
        "claude-api/", "codex-api/", "openai/",
        "anthropic/", "google/",
    ))
           for s in chain):
        return inspect_api_vision
    if any(s.startswith("claude-cli/") for s in chain):
        return inspect_claude_cli
    if shutil.which("claude"):
        return inspect_claude_cli
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            import google.generativeai  # noqa: F401
            return inspect_gemini
        except ImportError:
            pass
    return inspect_passthrough


def _slug_matches_model_family(slug: str, family: str) -> bool:
    if family == "claude":
        return slug.startswith(("claude/", "anthropic/")) or (
            slug.startswith("openai/") and "claude" in slug.lower()
        )
    if family == "gemini":
        return slug.startswith(("gemini/", "google/")) or (
            slug.startswith("openai/") and "gemini" in slug.lower()
        )
    if family == "gpt":
        return slug.startswith("gpt/") or (
            slug.startswith("openai/") and "gpt" in slug.lower()
        )
    if family == "openai":
        return slug.startswith("openai/")
    if family == "anthropic":
        return slug.startswith("anthropic/")
    if family == "google":
        return slug.startswith("google/")
    return slug.startswith(f"{family}/")


def inspect(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    iteration: int = 0,
    out_dir: Path | None = None,
    backend: str | None = None,
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
) -> CriticReport:
    """Public entry point.

    `backend` ∈ {None, 'api', 'claude', 'gpt', 'gemini',
    'claude-api', 'codex-api', 'google', 'claude-cli', 'passthrough'}.
    None means auto-select based on config,
    env, and available SDKs.

    `narrative` (optional) is forwarded to the backend so the LLM critic
    can judge concept_mismatch per shot against the node's visual_intent.
    When None, concept_mismatch will not be assessed.
    """
    impl = _select_backend(backend)
    model = None
    backend_aliases = {
        "claude-api": "claude",
        "codex-api": "gpt",
        "gemini-api": "gemini",
        "openai": "openai",
        "anthropic": "anthropic",
        "google-api": "google",
    }
    model_family = backend_aliases.get(backend or "", backend)
    if model_family in ("claude", "gpt", "gemini", "openai", "anthropic", "google"):
        cfg = get_agent_model(AGENT)
        chain = cfg.chain
        first = chain[0] if chain else ""
        # When the configured primary already matches the requested family
        # and a fallback exists, leave model=None so inspect_api_vision can
        # walk the whole chain on provider failure.
        use_chain_fallback = (
            _slug_matches_model_family(first, model_family) and len(chain) > 1
        )
        if not use_chain_fallback:
            model = next(
                (s for s in chain if _slug_matches_model_family(s, model_family)),
                None,
            )
            if model is None and backend in ("claude-api", "codex-api"):
                model = next(
                    (s for s in chain if s.startswith(f"{backend}/")),
                    None,
                )
            if model is None:
                default_model = {
                    "claude": "claude-sonnet-4-6",
                    "gpt": "gpt-5.5",
                    "gemini": "gemini-3.1-flash-lite-preview",
                    "openai": "gpt-5.5",
                    "anthropic": "claude-sonnet-4.6",
                    "google": "gemini-3.1-flash-lite-preview",
                }[model_family]
                provider = {
                    "claude": "openai",
                    "gpt": "openai",
                    "gemini": "openai",
                }.get(model_family, model_family)
                model = f"{provider}/{default_model}"
    if model:
        return impl(
            storyboard, frames_dir, iteration=iteration, out_dir=out_dir,
            model=model, narrative=narrative, scene_profile=scene_profile,
        )
    return impl(
        storyboard, frames_dir, iteration=iteration, out_dir=out_dir,
        narrative=narrative, scene_profile=scene_profile,
    )


def _aggregate_reports(
    storyboard: Storyboard,
    iteration: int,
    reports: list[tuple[str, CriticReport]],
    *,
    strictness: CriticStrictness = "consensus",
    partial_execution_errors: dict[str, list[str]] | None = None,
    unusable_members: dict[str, list[str]] | None = None,
) -> CriticReport:
    if strictness not in _CRITIC_STRICTNESS_VALUES:
        raise ValueError(
            "critic strictness must be one of: consensus, union, strict"
        )
    if not reports:
        return CriticReport(
            concept_id=storyboard.concept_id,
            iteration=iteration,
            overall_score=0.0,
            issues=[],
            execution_errors=["critic ensemble had no successful members"],
            ensemble_diagnostics={
                "strictness": strictness,
                "usable_members": [],
                "unusable_members": unusable_members or {},
                "partial_execution_errors": partial_execution_errors or {},
            },
        )
    scores = [r.overall_score for _name, r in reports]
    issues: dict[tuple[str, str, int, str], list[tuple[str, CriticIssue]]] = {}
    for name, report in reports:
        for issue in report.issues:
            key = (
                issue.shot_id,
                issue.category,
                issue.frame_idx,
                _normalise_issue_text(issue.issue)[:120],
            )
            issues.setdefault(key, []).append((name, issue))
    aggregate_issues = [
        _aggregate_issue_severity(
            group,
            total_members=len(reports),
            strictness=strictness,
        )
        for group in issues.values()
    ]
    member_block_counts = {
        name: sum(1 for i in report.issues if i.severity == "block")
        for name, report in reports
    }
    member_warn_counts = {
        name: sum(1 for i in report.issues if i.severity == "warn")
        for name, report in reports
    }
    score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    pass_blockers = _ensemble_pass_blockers(
        member_block_counts,
        score_spread,
        strictness=strictness,
    )
    demoted_blocks = [
        issue.model_dump()
        for issue in aggregate_issues
        if issue.suggested_fix.get("_demoted_from_block")
    ]
    aggregate_execution_errors = [
        f"{name}: {err}"
        for name, errors in sorted((partial_execution_errors or {}).items())
        for err in errors
    ] + [
        f"{name}: {err}"
        for name, errors in sorted((unusable_members or {}).items())
        for err in errors
    ]
    return CriticReport(
        concept_id=storyboard.concept_id,
        iteration=iteration,
        overall_score=sum(scores) / len(scores),
        issues=aggregate_issues,
        execution_errors=aggregate_execution_errors,
        pass_blockers=pass_blockers,
        ensemble_diagnostics={
            "strictness": strictness,
            "usable_members": [name for name, _report in reports],
            "unusable_members": unusable_members or {},
            "partial_execution_errors": partial_execution_errors or {},
            "member_block_counts": member_block_counts,
            "member_warn_counts": member_warn_counts,
            "score_spread": score_spread,
            "demoted_blocks": demoted_blocks,
        },
    )


def _ensemble_pass_blockers(
    member_block_counts: dict[str, int],
    score_spread: float,
    *,
    strictness: CriticStrictness,
) -> list[str]:
    if strictness != "strict":
        return []
    blockers: list[str] = []
    high_block_members = [
        f"{name}:{count}"
        for name, count in sorted(member_block_counts.items())
        if count >= _STRICT_BLOCK_COUNT_THRESHOLD
    ]
    if high_block_members:
        blockers.append(
            "member_block_count>="
            f"{_STRICT_BLOCK_COUNT_THRESHOLD} ({', '.join(high_block_members)})"
        )
    if score_spread > _STRICT_SCORE_SPREAD_THRESHOLD:
        blockers.append(
            f"member_score_spread>{_STRICT_SCORE_SPREAD_THRESHOLD:.2f} "
            f"({score_spread:.2f})"
        )
    return blockers


def _normalise_issue_text(text: str) -> str:
    return " ".join(text.lower().split())


def _is_fatal_critic_issue(issue: CriticIssue) -> bool:
    text = issue.issue.lower()
    return issue.category == "other" and any(
        marker in text
        for marker in (
            "blank frame",
            "black frame",
            "render failed",
            "no frames",
            "execution error",
        )
    )


def _aggregate_issue_severity(
    group: list[tuple[str, CriticIssue]],
    *,
    total_members: int,
    strictness: CriticStrictness = "consensus",
) -> CriticIssue:
    """Aggregate one deduplicated ensemble issue.

    For semantic/layout judgements, a block requires at least two critic
    members to agree. A single-member block is downgraded to warn, which makes
    the ensemble reduce variance instead of simply stacking strictness. Fatal
    render failures stay block even when only one backend reports them.
    """
    backends = {name for name, _issue in group}
    block_backends = {
        name for name, issue in group
        if issue.severity == "block"
    }
    representative = next(
        (issue for _name, issue in group if issue.severity == "block"),
        group[0][1],
    )
    severity = representative.severity
    demoted = (
        strictness == "consensus"
        and
        total_members > 1
        and severity == "block"
        and len(block_backends) < 2
        and not _is_fatal_critic_issue(representative)
    )
    if demoted:
        severity = "warn"
    suggested_fix = dict(representative.suggested_fix)
    suggested_fix.setdefault("_ensemble_support", sorted(backends))
    suggested_fix.setdefault("_block_support", sorted(block_backends))
    if demoted:
        suggested_fix["_demoted_from_block"] = True
    return representative.model_copy(
        update={
            "severity": severity,
            "suggested_fix": suggested_fix,
        }
    )


def inspect_ensemble(
    storyboard: Storyboard,
    frames_dir: Path,
    *,
    backends: tuple[str, ...],
    iteration: int = 0,
    out_dir: Path | None = None,
    narrative: Narrative | None = None,
    scene_profile: SceneProfile | None = None,
    strictness: CriticStrictness = "consensus",
) -> CriticReport:
    """Run multiple critic backends/models and aggregate conservatively."""
    if strictness not in _CRITIC_STRICTNESS_VALUES:
        raise ValueError(
            "critic strictness must be one of: consensus, union, strict"
        )
    reports: list[tuple[str, CriticReport]] = []
    member_reports: list[tuple[str, CriticReport]] = []
    ensemble_errors: list[str] = []
    partial_execution_errors: dict[str, list[str]] = {}
    unusable_members: dict[str, list[str]] = {}
    for idx, backend in enumerate(backends):
        try:
            report = inspect(
                storyboard,
                frames_dir,
                iteration=iteration,
                out_dir=None,
                backend=backend,
                narrative=narrative,
                scene_profile=scene_profile,
            )
            if report.issues or not report.execution_errors:
                reports.append((backend, report))
                if report.execution_errors:
                    partial_execution_errors[backend] = list(report.execution_errors)
                    ensemble_errors.extend(
                        f"{backend}: {err}" for err in report.execution_errors
                    )
            else:
                unusable_members[backend] = list(report.execution_errors)
                ensemble_errors.extend(
                    f"{backend}: {err}" for err in report.execution_errors
                )
        except Exception as e:  # noqa: BLE001
            report = CriticReport(
                concept_id=storyboard.concept_id,
                iteration=iteration,
                overall_score=0.0,
                issues=[],
                execution_errors=[f"critic failed: {str(e)[:200]}"],
            )
            unusable_members[backend] = list(report.execution_errors)
            ensemble_errors.extend(report.execution_errors)
        member_reports.append((backend, report))
        if out_dir is not None:
            save_artifact(
                out_dir,
                f"critic_iter{iteration:02d}.ensemble{idx:02d}_{backend.replace('/', '_')}.json",
                report.model_dump_json(indent=2),
            )
    aggregate = _aggregate_reports(
        storyboard,
        iteration,
        reports,
        strictness=strictness,
        partial_execution_errors=partial_execution_errors,
        unusable_members=unusable_members,
    )
    if ensemble_errors and not reports:
        aggregate.execution_errors.extend(ensemble_errors)
    if out_dir is not None:
        member_usable_summary = {
            "members": [
                {
                    "backend": name,
                    "usable": bool(report.issues or not report.execution_errors),
                    "participated_in_aggregate": any(
                        name == usable_name for usable_name, _report in reports
                    ),
                    "issues": len(report.issues),
                    "errors": len(report.execution_errors),
                    "execution_errors": list(report.execution_errors),
                }
                for name, report in member_reports
            ],
            "usable_members": [name for name, _report in reports],
            "unusable_members": unusable_members,
            "partial_execution_errors": partial_execution_errors,
        }
        save_artifact(
            out_dir,
            f"critic_iter{iteration:02d}.json",
            aggregate.model_dump_json(indent=2),
        )
        save_artifact(
            out_dir,
            f"critic_iter{iteration:02d}.member_usable_summary.json",
            json.dumps(member_usable_summary, indent=2),
        )
        save_artifact(
            out_dir,
            f"critic_iter{iteration:02d}.ensemble_summary.json",
            json.dumps({
                "backends": list(backends),
                "strictness": strictness,
                "member_scores": [
                    {"backend": name, "overall_score": report.overall_score}
                    for name, report in member_reports
                ],
                "member_block_counts": {
                    name: sum(
                        1 for issue in report.issues
                        if issue.severity == "block"
                    )
                    for name, report in member_reports
                },
                "member_warn_counts": {
                    name: sum(
                        1 for issue in report.issues
                        if issue.severity == "warn"
                    )
                    for name, report in member_reports
                },
                "aggregate_score": aggregate.overall_score,
                "aggregate_issues": len(aggregate.issues),
                "usable_members": [name for name, _report in reports],
                "unusable_members": unusable_members,
                "partial_execution_errors": partial_execution_errors,
                "score_spread": aggregate.ensemble_diagnostics.get(
                    "score_spread", 0.0,
                ),
                "demoted_blocks": aggregate.ensemble_diagnostics.get(
                    "demoted_blocks", [],
                ),
                "pass_blockers": aggregate.pass_blockers,
                "execution_errors": [
                    err
                    for _name, report in member_reports
                    for err in report.execution_errors
                ],
                "aggregate_execution_errors": aggregate.execution_errors,
            }, indent=2),
        )
    return aggregate


def issues_as_coder_addendum(report: CriticReport) -> str:
    """Format a critic report as a prompt addendum for a coder retry.

    The Blender Coder agent receives the original storyboard + this
    addendum, asking it to re-emit scene.py addressing the issues.
    """
    if not report.issues and not report.pass_blockers:
        return ""
    lines = [
        "PREVIOUS ATTEMPT had visual problems. Address each issue below "
        "by editing object positions, camera positions, light energy, or "
        "object visibility in the regenerated bpy script.\n",
    ]
    if report.pass_blockers:
        lines.append("STRICT CRITIC PASS BLOCKERS:")
        lines.extend(f"- {item}" for item in report.pass_blockers)
        score_spread = report.ensemble_diagnostics.get("score_spread")
        if isinstance(score_spread, (int, float)):
            lines.append(f"- member score spread: {score_spread:.2f}")
        lines.append("")
    for i, issue in enumerate(report.issues, 1):
        prefix = "single-critic BLOCK demoted to warn" if (
            issue.suggested_fix.get("_demoted_from_block")
        ) else issue.severity
        lines.append(
            f"{i}. [shot={issue.shot_id} frame={issue.frame_idx} "
            f"severity={prefix} category={issue.category}] "
            f"{issue.issue}"
        )
        if issue.suggested_fix:
            lines.append(f"   suggested: {json.dumps(issue.suggested_fix)}")
    return "\n".join(lines)
