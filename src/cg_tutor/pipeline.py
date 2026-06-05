"""Top-level orchestration: concept YAML → mp4.

Seven stages:
    1. Concept Decomposer  → narrative.json
    2. Storyboard          → storyboard.json (+ storyboard.raw.json)
    3. Blender Coder       → scene.py
    4. Blender runtime     → frames/*.png
    5. LaTeX Overlay       → overlays/*.png
    6. Render Critic       → critic_iterNN.json
    7. ffmpeg composer     → final.mp4

When ``max_critic_iterations > 0``, steps 3-4-6 form a loop: the Coder
retry receives the previous Critic's issues plus a history-aware trend
summary as an addendum. The loop exits when the Critic passes
(score ≥ 0.7, no block issues) or the iteration budget is exhausted.

Per-iteration artifacts (under ``out_dir``):
    scene.iterNN.py                    Coder output for iter N
    critic_iterNN.json                 Critic report for iter N
    trend_iterNN.txt                   addendum the Coder saw for iter N
                                       (skipped for iter 00)
    blender_stderr.iterNN.txt          Blender logs for iter N
    blender_stdout.iterNN.txt
    blender_stderr.bestreplay.txt      logs from the best-of replay render
    blender_stdout.bestreplay.txt      (only when best != last iter)
    critic_best.json                   summary across iters + frame hashes

Default Critic backend: the configured API ensemble, currently
``claude`` + ``gpt`` with strict aggregation. If no API model is configured, auto-select falls
back to ``claude-cli`` → ``google`` → ``passthrough``.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
import json

import yaml


# Concept ids flow directly into filesystem paths; reject anything that
# could escape `out_root` before we mkdir or save_artifact.
_CONCEPT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _validate_concept_id(concept_id: str) -> str:
    if not isinstance(concept_id, str) or not _CONCEPT_ID_RE.match(concept_id):
        raise ValueError(
            f"concept_id={concept_id!r} must match [A-Za-z0-9_-]+ "
            "(no path separators, no '..' segments). This is enforced "
            "because concept_id is used as a filesystem path component."
        )
    return concept_id

from cg_tutor.agents import (
    blender_coder,
    concept_decomposer,
    profile_generator,
    render_critic,
    storyboard as storyboard_agent,
)
from cg_tutor.agents.base import save_artifact
from cg_tutor.auto_success_spec import (
    auto_success_spec_to_json,
    auto_success_validation_to_json,
    critic_confirmed_auto_success_issues,
    format_auto_success_spec_for_coder,
    generate_auto_success_spec,
)
from cg_tutor.blender import runtime as blender_runtime
from cg_tutor.composer.compose import compose_storyboard_video
from cg_tutor.critic_loop import (
    BestSelectionMode,
    CriticIteration,
    _block_floor_stale_iters,
    _critic_counts,
    _critic_history_addendum,
    _critic_quality_key,
    _critic_quality_key_for,
    _failure_class_counts,
    _flagged_counts,
    format_critic_visual_evidence_packet,
    _issue_key,
    _missing_storyboard_objects,
    _multi_reference_retry_addendum,
    _regression_keys,
    _scene_param_diff,
    _semantic_counts,
    _shot_visual_contracts,
)
from cg_tutor.concept_metrics import (
    ConceptMetricIssue,
    ConceptMetricReport,
    concept_metric_report_to_json,
    format_concept_metric_report_for_coder,
    run_concept_metrics,
)
from cg_tutor.correction_controller import (
    CorrectionDecision,
    correction_decision_to_json,
    decide_correction,
)
from cg_tutor.critic_cross_reference import (
    CrossReferenceFinding,
    CrossReferenceReport,
    cross_reference_critic_findings,
    cross_reference_report_to_json,
    format_cross_reference_for_coder,
)
from cg_tutor.schemas import CriticReport, Storyboard
from cg_tutor.visual_contract import VisualContract
from cg_tutor.failure_memory import (
    append_failure_memory,
    format_failure_memory_for_coder,
    load_failure_memory,
    memory_from_history,
    save_failure_memory_snapshot,
)
from cg_tutor.repair_plan import (
    build_repair_plan,
    format_repair_plan_for_coder,
    repair_plan_to_json,
)
from cg_tutor.scene_ir import (
    build_scene_ir,
    format_scene_ir_for_coder,
    scene_ir_to_json,
    scene_ir_verification_to_json,
    verify_scene_ir,
)
from cg_tutor.scene_state import inspect_scene_code, scene_state_report_to_json
from cg_tutor.scene_profiles import (
    SceneProfile,
    format_scene_profile_for_prompt,
)
from cg_tutor.success_spec import (
    SuccessSpec,
    format_success_spec_for_coder,
    load_success_spec,
    success_spec_to_json,
)
from cg_tutor.storyboard_sanitizer import sanitize_storyboard_for_pipeline
from cg_tutor import terminal_ui as ui
from cg_tutor.scene_compiler import (
    compile_storyboard_to_bpy,
)
from cg_tutor.scene_verifier import (
    SceneVerificationIssue,
    SceneVerificationReport,
    format_verifier_addendum,
    report_to_json,
    verify_scene_code,
)
from cg_tutor.contract_validator import (
    format_contract_validation_addendum,
    report_to_json as contract_report_to_json,
    validate_visual_contracts,
)
from cg_tutor.preview import (
    PreviewIssue,
    PreviewReport,
    preview_blocks_allow_render_repair,
    preview_report_to_json,
    select_preview_frames,
    verify_preview_frames,
)
from cg_tutor._logging import get_logger


log = get_logger(__name__)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "BestSelectionMode",
    "CriticIteration",
    "PipelineResult",
    "_block_floor_stale_iters",
    "_critic_counts",
    "_critic_history_addendum",
    "_critic_quality_key",
    "_critic_quality_key_for",
    "_flagged_counts",
    "_issue_key",
    "_missing_storyboard_objects",
    "_multi_reference_retry_addendum",
    "_regression_keys",
    "_scene_param_diff",
    "_semantic_counts",
    "_shot_visual_contracts",
    "compose_storyboard_video",
    "run",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"


def _early_exit(out_dir, scene_path, frames_dir, t0):
    """Build a failed PipelineResult when we can't proceed past step 4."""
    return PipelineResult(
        concept_id=out_dir.name, out_dir=out_dir, final_mp4=None,
        narrative_path=out_dir / "narrative.json",
        storyboard_path=out_dir / "storyboard.json",
        scene_path=scene_path, frames_dir=frames_dir,
        elapsed_sec=time.time() - t0,
    )


def _scene_cache_is_usable(scene_path: Path) -> bool:
    """Cheap guard against cached truncated / fenced LLM code.

    Uses AST so a stray ``bpy.ops.render.render`` mentioned in a
    comment or docstring does not falsely vouch for a script that
    actually never reaches the render call. The check is intentionally
    cheap (single ast.parse + one walk) — anything more thorough lives
    in scene_verifier.verify_scene_code.
    """
    import ast as _ast
    if not scene_path.exists():
        return False
    text = scene_path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("```"):
        return False
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            func = node.func
            parts: list[str] = []
            while isinstance(func, _ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, _ast.Name):
                parts.append(func.id)
                dotted = ".".join(reversed(parts))
                if dotted == "bpy.ops.render.render":
                    return True
    return False


def _critic_iteration_from_path(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("critic_iter"):
        return None
    suffix = stem.removeprefix("critic_iter")
    if len(suffix) != 2 or not suffix.isdigit():
        return None
    return int(suffix)


def _load_critic_history_from_disk(out_dir: Path) -> list[CriticIteration]:
    """Reconstruct critic history from saved artifacts when resuming.

    We prefer the per-iteration critic JSONs as the source of truth and
    layer in the summary metadata from ``critic_best.json`` when present.
    """
    summary_meta: dict[int, dict] = {}
    critic_best_path = out_dir / "critic_best.json"
    if critic_best_path.exists():
        try:
            best_data = json.loads(critic_best_path.read_text())
            for item in best_data.get("all_iterations", []):
                try:
                    summary_meta[int(item["iteration"])] = item
                except Exception as e:  # noqa: BLE001
                    log.warning(f"bad iteration entry in "
                          f"critic_best.json, skipping: {e!r}")
                    continue
        except Exception as e:  # noqa: BLE001
            log.warning(f"critic_best.json unreadable, "
                  f"resume will rebuild metadata from per-iter files: {e!r}")
            summary_meta = {}

    history: list[CriticIteration] = []
    for path in sorted(out_dir.glob("critic_iter??.json")):
        if path.name.endswith(".ensemble_summary.json"):
            continue
        iteration = _critic_iteration_from_path(path)
        if iteration is None:
            continue
        try:
            report = CriticReport.model_validate_json(path.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning(f"{path.name} failed to parse, skipping "
                  f"(iteration chain may have a gap at iter{iteration:02d}): "
                  f"{e!r}")
            continue
        meta = summary_meta.get(iteration, {})
        scene_iter_path = out_dir / f"scene.iter{iteration:02d}.py"
        frames_iter_dir = _frame_snapshot_dir(out_dir, iteration)
        n_frames = int(meta.get("n_frames", 0) or 0)
        frames_hash = str(meta.get("frames_hash", ""))
        if not scene_iter_path.exists():
            log.warning(
                f"{path.name} has no matching {scene_iter_path.name}; "
                "skipping this critic record for resume/best selection"
            )
            continue
        history.append(CriticIteration(
            iteration=iteration,
            report=report,
            scene_path=scene_iter_path,
            render_ok=bool(meta.get("render_ok", True)),
            n_frames=n_frames,
            frames_hash=frames_hash,
            frames_dir=frames_iter_dir if frames_iter_dir.exists() else None,
            scene_origin=str(meta.get("scene_origin", "unknown")),
            missing_objects={},
            metric_block_count=int(meta.get("metric_block", 0) or 0),
            metric_warn_count=int(meta.get("metric_warn", 0) or 0),
            metric_structural_fatal_count=int(
                meta.get("metric_structural_fatal", 0) or 0
            ),
            metric_success_hard_count=int(
                meta.get("metric_success_hard", 0) or 0
            ),
            metric_success_soft_count=int(
                meta.get("metric_success_soft", 0) or 0
            ),
            metric_aesthetic_warn_count=int(
                meta.get("metric_aesthetic_warn", meta.get("metric_warn", 0)) or 0
            ),
            cross_ref_actionable_count=int(
                meta.get("cross_ref_actionable", 0) or 0
            ),
            fallback_degraded=bool(meta.get("fallback_degraded", False)),
        ))
    history.sort(key=lambda r: r.iteration)
    return history


def _load_metric_history_from_disk(
    out_dir: Path,
) -> list[ConceptMetricReport]:
    history: list[ConceptMetricReport] = []
    for path in sorted(out_dir.glob("concept_metric_report.iter??.json")):
        try:
            raw = json.loads(path.read_text())
            issues = [
                ConceptMetricIssue(
                    severity=item.get("severity", "warn"),
                    rule_id=str(item.get("rule_id", "")),
                    message=str(item.get("message", "")),
                    suggested_fix=str(item.get("suggested_fix", "")),
                    failure_class=item.get("failure_class"),
                )
                for item in raw.get("issues", [])
                if isinstance(item, dict)
            ]
            history.append(ConceptMetricReport(
                concept_id=str(raw.get("concept_id", "")),
                issues=issues,
                metrics={
                    **(raw.get("metrics", {}) if isinstance(raw.get("metrics"), dict) else {}),
                    "_iteration": _artifact_iteration_from_name(path.name),
                },
            ))
        except Exception as e:  # noqa: BLE001
            log.warning(f"{path.name} failed to parse, skipping metric resume: {e!r}")
    return history


def _load_cross_ref_history_from_disk(
    out_dir: Path,
) -> list[CrossReferenceReport]:
    history: list[CrossReferenceReport] = []
    for path in sorted(out_dir.glob("critic_cross_reference.iter??.json")):
        try:
            raw = json.loads(path.read_text())
            findings = [
                CrossReferenceFinding(
                    rule_id=str(item.get("rule_id", "")),
                    severity=item.get("severity", "deferred"),
                    diagnosis=str(item.get("diagnosis", "")),
                    critic_source=str(item.get("critic_source", "")),
                    ast_evidence=str(item.get("ast_evidence", "")),
                    suggested_fix=str(item.get("suggested_fix", "")),
                )
                for item in raw.get("findings", [])
                if isinstance(item, dict)
            ]
            history.append(CrossReferenceReport(
                concept_id=str(raw.get("concept_id", "")),
                iteration=int(raw.get("iteration", 0) or 0),
                findings=findings,
            ))
        except Exception as e:  # noqa: BLE001
            log.warning(f"{path.name} failed to parse, skipping cross-ref resume: {e!r}")
    return history


def _hydrate_history_diagnostics(
    critic_history: list[CriticIteration],
    metric_history: list[ConceptMetricReport],
    cross_ref_history: list[CrossReferenceReport],
) -> None:
    metrics_by_iter = {
        int(report.metrics.get("_iteration", idx)): report
        for idx, report in enumerate(metric_history)
    }
    # Metric artifacts are loaded in sorted filename order; if the JSON lacks
    # an iteration field, their sorted index matches iterNN for contiguous runs.
    for item in critic_history:
        metric = metrics_by_iter.get(item.iteration)
        if metric is not None:
            item.metric_block_count = metric.block_count
            item.metric_warn_count = metric.warn_count
            item.metric_structural_fatal_count = metric.structural_fatal_count
            item.metric_success_hard_count = metric.success_hard_count
            item.metric_success_soft_count = metric.success_soft_count
            item.metric_aesthetic_warn_count = metric.aesthetic_warn_count
        for report in cross_ref_history:
            if report.iteration == item.iteration:
                item.cross_ref_actionable_count = report.actionable_count
                break
def _artifact_iteration_from_name(name: str) -> int:
    match = re.search(r"\.iter(\d{2})\.", name)
    return int(match.group(1)) if match else 0


def _select_resume_scene_source(
    scene_path: Path,
    critic_history: list[CriticIteration],
) -> Path | None:
    """Pick the best cached scene to continue from on resume."""
    if _scene_cache_is_usable(scene_path):
        return scene_path
    for item in reversed(critic_history):
        if _scene_cache_is_usable(item.scene_path):
            return item.scene_path
    return None


def _compat_scene_text(text: str) -> str:
    return blender_coder.compatibilize_blender_code(text)


def _write_compat_scene(path: Path, text: str) -> None:
    path.write_text(_compat_scene_text(text))


def _frames_hash(frames_dir: Path) -> str:
    """SHA-256 over sorted frame contents. Empty string when no frames.

    Used to detect non-determinism between the originally critic-scored
    render and the best-of replay render: if the hash changes, the
    composed mp4 may not match what the critic actually evaluated.
    """
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        return ""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _frame_snapshot_dir(out_dir: Path, iteration: int) -> Path:
    return out_dir / f"frames.iter{iteration:02d}"


def _copy_frame_snapshot(src_dir: Path, dst_dir: Path) -> int:
    """Copy rendered PNG frames so critic-scored frames can be reused exactly."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.glob("frame_*.png"):
        old.unlink()
    count = 0
    for src in sorted(src_dir.glob("frame_*.png")):
        shutil.copy2(src, dst_dir / src.name)
        count += 1
    return count


class _FramesHashCache:
    """Memoize _frames_hash within a single pipeline.run().

    Frame contents are large (tens of MB per render) so SHA-256-ing the
    same directory three times — which the current code does in the
    best-replay flow — is wasted I/O. Keyed by (path, mtime_ns,
    file_count) so the cache is invalidated whenever frames change.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def get(self, frames_dir: Path) -> str:
        try:
            paths = sorted(frames_dir.glob("frame_*.png"))
            if not paths:
                return ""
            latest_mtime = max(p.stat().st_mtime_ns for p in paths)
            key = (str(frames_dir.resolve()), latest_mtime, len(paths))
        except OSError:
            return _frames_hash(frames_dir)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        digest = _frames_hash(frames_dir)
        self._cache[key] = digest
        return digest

    def invalidate(self, frames_dir: Path) -> None:
        prefix = str(frames_dir.resolve())
        self._cache = {k: v for k, v in self._cache.items() if k[0] != prefix}


def _detect_history_gaps(history: list[CriticIteration]) -> list[int]:
    """Return the iteration indices that are missing between iter00 and
    the last present iteration. An empty list means the chain is
    contiguous from 0..last."""
    if not history:
        return []
    iters = sorted({h.iteration for h in history})
    expected = set(range(iters[0], iters[-1] + 1))
    return sorted(expected - set(iters))


def _save_scene_verification(out_dir: Path, iteration: int, report, suffix: str = ""):
    name = f"scene_verifier.iter{iteration:02d}{suffix}.json"
    save_artifact(out_dir, name, report_to_json(report))
    return name


def _merge_contract_blocks_into_verifier(verifier_report, contract_report) -> None:
    existing = {
        (issue.rule_id, issue.message)
        for issue in verifier_report.issues
    }
    for violation in contract_report.violations:
        if violation.severity != "block":
            continue
        key = (violation.rule_id, violation.message)
        if key in existing:
            continue
        verifier_report.issues.append(SceneVerificationIssue(
            severity="block",
            rule_id=violation.rule_id,
            message=violation.message,
            suggested_fix=violation.suggested_fix,
        ))
        existing.add(key)


def _repair_quality_key(verifier_report, contract_report) -> tuple[int, ...]:
    """Lower is better for deterministic pre-render repair candidates."""
    syntax_blocks = sum(
        1 for issue in verifier_report.issues
        if issue.severity == "block" and issue.rule_id == "syntax_error"
    )
    missing_objects = sum(
        len(names) for names in verifier_report.missing_objects.values()
    )
    total_blocks = verifier_report.block_count + contract_report.block_count
    hard_failures = total_blocks + missing_objects
    return (
        syntax_blocks,
        hard_failures,
        missing_objects,
        total_blocks,
        verifier_report.block_count,
        contract_report.block_count,
        verifier_report.warn_count + contract_report.warn_count,
    )


def _syntax_block_count(verifier_report) -> int:
    return sum(
        1 for issue in verifier_report.issues
        if issue.severity == "block" and issue.rule_id == "syntax_error"
    )


def _render_contract_block_count(verifier_report) -> int:
    fatal_rules = {
        "syntax_error",
        "missing_render_call",
        "missing_out_dir",
        "missing_out_dir_env",
        "render_not_animation",
        "forbidden_engine",
        "forbidden_file_io",
        "forbidden_subprocess",
        "forbidden_save_file",
    }
    return sum(
        1 for issue in verifier_report.issues
        if issue.severity == "block" and issue.rule_id in fatal_rules
    )


def _fatal_verifier_rule_ids(verifier_report) -> list[str]:
    fatal_rules = {
        "syntax_error",
        "missing_render_call",
        "missing_out_dir",
        "missing_out_dir_env",
        "render_not_animation",
        "forbidden_engine",
        "forbidden_file_io",
        "forbidden_subprocess",
        "forbidden_save_file",
    }
    return [
        issue.rule_id
        for issue in verifier_report.issues
        if issue.severity == "block" and issue.rule_id in fatal_rules
    ]


def _missing_object_count(verifier_report) -> int:
    return sum(len(names) for names in verifier_report.missing_objects.values())


def _repair_regression_reason(
    current_verifier,
    current_contract,
    candidate_verifier,
    candidate_contract,
) -> str | None:
    current_syntax = _syntax_block_count(current_verifier)
    candidate_syntax = _syntax_block_count(candidate_verifier)
    current_hard = (
        current_verifier.block_count
        + current_contract.block_count
        + _missing_object_count(current_verifier)
    )
    candidate_hard = (
        candidate_verifier.block_count
        + candidate_contract.block_count
        + _missing_object_count(candidate_verifier)
    )
    if candidate_syntax > current_syntax:
        return "syntax block count increased"
    current_fatal = _fatal_verifier_rule_ids(current_verifier)
    candidate_fatal = _fatal_verifier_rule_ids(candidate_verifier)
    new_fatal = sorted(set(candidate_fatal) - set(current_fatal))
    if current_fatal:
        if len(candidate_fatal) > len(current_fatal):
            return "fatal verifier block count increased"
        if len(candidate_fatal) == len(current_fatal) and new_fatal:
            return "fatal verifier block traded for " + ", ".join(new_fatal)
        if new_fatal:
            return "new fatal verifier block introduced: " + ", ".join(new_fatal)
    syntax_repaired = current_syntax > 0 and candidate_syntax < current_syntax
    if syntax_repaired and candidate_hard <= current_hard:
        return None
    if candidate_verifier.block_count > current_verifier.block_count:
        return "verifier block count increased"
    if candidate_contract.block_count > current_contract.block_count:
        return "contract block count increased"
    if _missing_object_count(candidate_verifier) > _missing_object_count(current_verifier):
        return "missing storyboard object count increased"
    return None


def _has_fatal_verifier_block(verifier_report) -> bool:
    return _render_contract_block_count(verifier_report) > 0


def _can_try_render_tail_repair(verifier_report, code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    fatal_ids = set(_fatal_verifier_rule_ids(verifier_report))
    return bool(fatal_ids) and fatal_ids.issubset({
        "missing_render_call",
        "missing_out_dir",
        "missing_out_dir_env",
        "render_not_animation",
    })


def _append_preview_aware_render_tail(code: str) -> str:
    tail = r'''

# CG-Tutor deterministic render tail repair.
import os

_cg_out_dir = os.environ.get('CG_TUTOR_OUT_DIR')
if _cg_out_dir:
    _cg_frame_pattern = os.path.join(_cg_out_dir, 'frame_####.png')
    if not _cg_frame_pattern.startswith('\\\\'):
        _cg_frame_pattern = _cg_frame_pattern.replace('\\', '/')
    bpy.context.scene.render.filepath = _cg_frame_pattern

_cg_preview_raw = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if _cg_preview_raw and _cg_out_dir:
    for _cg_raw in _cg_preview_raw.split(','):
        _cg_raw = _cg_raw.strip()
        if not _cg_raw:
            continue
        _cg_frame = int(_cg_raw)
        bpy.context.scene.frame_set(_cg_frame)
        _cg_preview_path = os.path.join(_cg_out_dir, f'frame_{_cg_frame:04d}.png')
        if not _cg_preview_path.startswith('\\\\'):
            _cg_preview_path = _cg_preview_path.replace('\\', '/')
        bpy.context.scene.render.filepath = _cg_preview_path
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
'''
    return code.rstrip() + "\n" + tail.lstrip()


def _try_render_tail_repair(
    *,
    out_dir: Path,
    iteration: int,
    code: str,
    storyboard: Storyboard,
    visual_contracts,
    scene_profile: SceneProfile | None,
    render_engine: str,
    cycles_device: str,
    success_spec: SuccessSpec | None,
):
    repaired = _append_preview_aware_render_tail(code)
    raw_report = verify_scene_code(
        repaired,
        storyboard,
        visual_contracts=visual_contracts,
        scene_profile=scene_profile,
        render_engine=render_engine,
        cycles_device=cycles_device,
    )
    verifier_name = _save_scene_verification(
        out_dir,
        iteration,
        raw_report,
        suffix=".render_tail",
    )
    contract_report = validate_visual_contracts(
        repaired,
        storyboard,
        visual_contracts,
        scene_profile=scene_profile,
        success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
    )
    contract_name = f"contract_validation.iter{iteration:02d}.render_tail.json"
    save_artifact(out_dir, contract_name, contract_report_to_json(contract_report))
    _merge_contract_blocks_into_verifier(raw_report, contract_report)
    if _has_fatal_verifier_block(raw_report):
        return None
    save_artifact(out_dir, "scene.py", repaired)
    save_artifact(
        out_dir,
        f"scene.render_tail.used.iter{iteration:02d}.txt",
        f"verifier={verifier_name}\ncontract={contract_name}\n",
    )
    return repaired, raw_report, contract_report, verifier_name


def _repair_quality_text(verifier_report, contract_report) -> str:
    return (
        f"verifier={verifier_report.block_count}b/"
        f"{verifier_report.warn_count}w; "
        f"contract={contract_report.block_count}b/"
        f"{contract_report.warn_count}w"
    )


def _fallback_contract_is_render_safe(contract_report) -> bool:
    allowed_blocks = {
        "contract_insufficient_vector_geometry",
        "contract_insufficient_text_objects",
    }
    for violation in contract_report.violations:
        if violation.severity != "block":
            continue
        if violation.rule_id not in allowed_blocks:
            return False
    return True


def _try_compiled_scene_fallback(
    *,
    out_dir: Path,
    iteration: int,
    compiled_scene: str,
    storyboard: Storyboard,
    visual_contracts,
    scene_profile: SceneProfile | None,
    render_engine: str,
    cycles_device: str,
    reason: str,
    success_spec: SuccessSpec | None = None,
):
    verifier_report = verify_scene_code(
        compiled_scene,
        storyboard,
        visual_contracts=visual_contracts,
        scene_profile=scene_profile,
        render_engine=render_engine,
        cycles_device=cycles_device,
    )
    verifier_name = _save_scene_verification(
        out_dir,
        iteration,
        verifier_report,
        suffix=".compiled_fallback",
    )
    contract_report = validate_visual_contracts(
        compiled_scene,
        storyboard,
        visual_contracts,
        scene_profile=scene_profile,
        success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
    )
    contract_name = (
        f"contract_validation.iter{iteration:02d}.compiled_fallback.json"
    )
    save_artifact(out_dir, contract_name, contract_report_to_json(contract_report))
    if verifier_report.block_count or not _fallback_contract_is_render_safe(contract_report):
        return None
    save_artifact(out_dir, "scene.py", compiled_scene)
    save_artifact(
        out_dir,
        "scene.compiled.used.txt",
        f"reason={reason}\n"
        f"verifier={verifier_name}\n"
        f"contract={contract_name}\n",
    )
    return compiled_scene, verifier_report, contract_report, verifier_name


def _format_vector_ray_minimum_scaffold(visual_contracts) -> str:
    if not visual_contracts:
        return ""
    by_shot: list[str] = []
    for shot_id, contract in sorted(visual_contracts.items()):
        vectors = list(getattr(contract, "required_vectors", []) or [])
        if not vectors:
            continue
        labels = list(getattr(contract, "required_labels", []) or [])
        by_shot.append(
            f"- {shot_id}: vectors="
            + ", ".join(str(v) for v in vectors[:8])
            + (
                "; labels=" + ", ".join(str(v) for v in labels[:6])
                if labels else ""
            )
        )
    if not by_shot:
        return ""
    return (
        "VECTOR/RAY MINIMUM SCAFFOLD:\n"
        "This scene has required vector/ray visual contracts. Before improving "
        "style, create the minimal named teaching skeleton.\n"
        "- Each required vector/ray cue must be a real thin curve_polyline or "
        "arrow-like object, not only a label.\n"
        "- For ray/light paths, prefer curve_polyline tracers with bevel_depth "
        "and high-contrast/emissive material.\n"
        "- Object names must include the required vector token, such as "
        "incident_white_ray, surface_normal_entry, spectrum_red_ray.\n"
        "- For each vector-heavy shot, include incoming/outgoing/internal path "
        "cues when relevant, a surface normal cue, an angle arc cue, and a "
        "short readable label cue.\n"
        "- Do not replace all vectors with one generic helper unless shot-local "
        "visibility and names remain traceable.\n"
        "Required cues:\n"
        + "\n".join(by_shot)
    )


def _run_keyframe_preview(
    *,
    scene_path: Path,
    preview_dir: Path,
    preview_frames: list[int],
    scene_profile,
    storyboard,
    blender_runtime_mod,
    blender_timeout_sec: float,
    out_dir: Path,
    iteration: int,
    artifact_suffix: str = "",
):
    """Render keyframe preview, save stdout/stderr/report, return (report, blender_ok).

    Extracted so the same logic can run twice in one iter: once against the
    coder candidate, and once against the compiled scaffold as a fallback
    when the candidate crashes Blender. ``artifact_suffix`` distinguishes
    the two artifact sets (e.g. ``".compiled_fallback"``).
    """
    if preview_dir.exists():
        for f in preview_dir.glob("frame_*.png"):
            f.unlink()
    rr_preview = blender_runtime_mod.run_script(
        scene_path,
        preview_dir,
        env_overrides={
            "CG_TUTOR_PREVIEW_FRAMES": ",".join(
                str(f) for f in preview_frames
            ),
        },
        timeout_sec=min(blender_timeout_sec, 600),
    )
    suf = artifact_suffix
    save_artifact(
        out_dir,
        f"blender_stdout.preview.iter{iteration:02d}{suf}.txt",
        rr_preview.stdout or "",
    )
    save_artifact(
        out_dir,
        f"blender_stderr.preview.iter{iteration:02d}{suf}.txt",
        rr_preview.stderr or "",
    )
    report = verify_preview_frames(
        preview_dir,
        preview_frames,
        scene_profile=scene_profile,
        storyboard=storyboard,
    )
    if not rr_preview.ok:
        report.issues.append(PreviewIssue(
            severity="block",
            rule_id="preview_blender_error",
            frame_idx=None,
            message="Blender failed during keyframe preview.",
            suggested_fix=(
                "Inspect blender_stderr.preview for the Python "
                "or render error before full render."
            ),
        ))
        report.ok = False
    save_artifact(
        out_dir,
        f"preview_report.iter{iteration:02d}{suf}.json",
        preview_report_to_json(report),
    )
    return report, rr_preview.ok


def _is_compiled_origin(origin: str) -> bool:
    return origin in {"compiled_seed", "compiled_fallback", "compiler_only"}


def _latest_non_compiled_iteration(
    history: list[CriticIteration],
) -> CriticIteration | None:
    for item in reversed(history):
        if not _is_compiled_origin(item.scene_origin):
            return item
    return None


def _join_addenda(*parts: str) -> str:
    return "\n\n---\n".join(p for p in parts if p and p.strip())


_MINIMAL_SUCCESS_REPAIR_PRINCIPLE = """SUCCESS-HARD REPAIR PRINCIPLE:
Each success-hard violation has exactly one simplest fix.
- Missing anchor: add the exact named object only.
- Mirrored text: fix the existing object's transform only.
- Aperture out of range: change the one literal assignment.
- Focus transition broken: add/adjust exactly the required focus keyframes.
- Prefer editing existing objects over creating new ones.
- Do not add labels, decorations, HUD text, or helper objects not listed in success_spec."""


@dataclass
class AddendumBundle:
    priority: str = ""
    metric: str = ""
    cross_ref: str = ""
    shot_contract: str = ""
    grounding_patch: str = ""
    critic_visual_evidence: str = ""
    multi_reference: str = ""
    history: str = ""
    repair_plan: str = ""
    has_success_hard: bool = False

    def _needs_success_repair_principle(self) -> bool:
        # Only inject the success-hard repair guardrail when the bundle is
        # actually carrying a success-hard signal. Caller sets has_success_hard
        # explicitly from the typed metric report; the formatted metric text
        # contains the literal "success_hard" in its descriptive header, so
        # substring-sniffing self.metric would fire on any non-empty metric
        # (including aesthetic_warn-only) and defeat the conditional.
        if self.has_success_hard:
            return True
        if self.cross_ref.strip():
            return True
        return False

    def join(self) -> str:
        principle = (
            _MINIMAL_SUCCESS_REPAIR_PRINCIPLE
            if self._needs_success_repair_principle()
            else ""
        )
        return _join_addenda(
            principle,
            self.priority,
            self.metric,
            self.cross_ref,
            self.shot_contract,
            self.grounding_patch,
            self.critic_visual_evidence,
            self.multi_reference,
            self.history,
            self.repair_plan,
        )


def _onoff(value: bool) -> str:
    return "on" if value else "off"


def _count_text(block: int, warn: int) -> str:
    return f"{block} block / {warn} warn"


def _critic_status_text(report: CriticReport) -> str:
    framing_block, framing_warn = _critic_counts(report)
    semantic_block, semantic_warn = _semantic_counts(report)
    total_block = sum(1 for issue in report.issues if issue.severity == "block")
    total_warn = sum(1 for issue in report.issues if issue.severity == "warn")
    return (
        f"score={report.overall_score:.2f}; "
        f"framing={_count_text(framing_block, framing_warn)}; "
        f"semantic={_count_text(semantic_block, semantic_warn)}; "
        f"total={_count_text(total_block, total_warn)}; "
        f"errors={len(report.execution_errors)}; "
        f"blockers={len(report.pass_blockers)}"
    )


def _failure_class_status_text(item: CriticIteration) -> str:
    structural, success_hard, success_soft, aesthetic_warn = (
        _failure_class_counts(item)
    )
    return (
        f"structural={structural}; success_hard={success_hard}; "
        f"success_soft={success_soft}; aesthetic_warn={aesthetic_warn}"
    )


def _critic_fail_reasons(report: CriticReport) -> list[str]:
    reasons: list[str] = []
    if report.overall_score <= 0.7:
        reasons.append("score <= 0.70")
    if report.has_block:
        reasons.append("has block issue(s)")
    if report.execution_errors:
        reasons.append("critic execution error(s)")
    if report.pass_blockers:
        reasons.extend(report.pass_blockers)
    return reasons


def _iteration_pass_threshold(
    report: CriticReport,
    metric_report: ConceptMetricReport | None = None,
) -> bool:
    return report.pass_threshold and (
        metric_report is None
        or (
            metric_report.block_count == 0
            and
            metric_report.structural_fatal_count == 0
            and metric_report.success_hard_count == 0
        )
    )


def _iteration_fail_reasons(
    report: CriticReport,
    metric_report: ConceptMetricReport | None = None,
) -> list[str]:
    reasons = _critic_fail_reasons(report)
    if metric_report is not None:
        if metric_report.structural_fatal_count:
            reasons.append("concept metric structural fatal issue(s)")
        if metric_report.success_hard_count:
            reasons.append("concept metric success-hard issue(s)")
        elif metric_report.block_count:
            reasons.append("concept metric block issue(s)")
    return reasons


def _append_auto_success_issues(
    metric_report: ConceptMetricReport,
    items: list[dict],
) -> None:
    for item in items:
        metric_report.issues.append(ConceptMetricIssue(
            severity=str(item.get("severity", "block")),  # type: ignore[arg-type]
            rule_id=str(item["rule_id"]),
            message=str(item["message"]),
            suggested_fix=str(item["suggested_fix"]),
            failure_class=str(item.get("failure_class", "success_hard")),  # type: ignore[arg-type]
        ))


def _critic_iteration_pass(item: CriticIteration) -> bool:
    return (
        not item.fallback_degraded
        and
        item.report.pass_threshold
        and item.metric_block_count == 0
        and item.metric_structural_fatal_count == 0
        and item.metric_success_hard_count == 0
    )


_FIX_KEY_ANCHOR_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b"
)
_RELATIONSHIP_MARKERS = (
    "pass_through",
    "pass through",
    "through c",
    "on_image_plane",
    "on image plane",
    "projection ray",
    "projection rays",
    "projected silhouette",
)
_LABEL_KEY_MARKERS = ("label", "formula", "text")
_ANCHOR_KEY_STOPWORDS = {
    "label_text", "label_visible", "label_attached", "label_offset",
    "font_size", "screen_position", "base_color",
}


def _critic_grounding_patch(critic_history: list[CriticIteration]) -> dict:
    """Derive a run-local grounding patch from repeated critic evidence."""
    evidence: dict[str, dict] = {}
    required_anchors: list[str] = []
    required_labels: list[str] = []
    required_relationships: list[str] = []
    source_issues: list[dict] = []
    for item in critic_history:
        for issue in item.report.issues:
            if issue.category != "concept_mismatch":
                continue
            text = issue.issue.strip()
            low = text.lower()
            if issue.severity == "block" or issue.suggested_fix.get(
                "_demoted_from_block"
            ):
                extracted = _extract_grounding_constraints_from_issue(issue)
                required_anchors.extend(extracted["required_anchors"])
                required_labels.extend(extracted["required_labels"])
                required_relationships.extend(
                    extracted["required_relationships"]
                )
                if any(extracted.values()):
                    source_issues.append({
                        "shot_id": issue.shot_id,
                        "severity": issue.severity,
                        "issue": text[:240],
                        "suggested_fix": issue.suggested_fix,
                    })
            if not any(
                marker in low
                for marker in (
                    "glowing rectangle",
                    "plain rectangle",
                    "floating sphere",
                    "black void",
                    "abstract primitive",
                    "generic primitive",
                    "missing",
                    "absent",
                )
            ):
                continue
            key = _issue_key(issue)
            entry = evidence.setdefault(key, {
                "count": 0,
                "shots": set(),
                "examples": [],
                "severity": issue.severity,
            })
            entry["count"] += 1
            entry["shots"].add(issue.shot_id)
            if len(entry["examples"]) < 2:
                entry["examples"].append(text)
            if issue.severity == "block":
                entry["severity"] = "block"

    forbidden: list[str] = []
    priorities: list[str] = []
    for entry in evidence.values():
        if entry["count"] < 2 and entry["severity"] != "block":
            continue
        forbidden.append(
            "Avoid this critic-observed abstraction failure: "
            + entry["examples"][0]
        )
        priorities.append(
            "Ground the scene with concrete named geometry before adding "
            "effects; repeated critic evidence from shot(s) "
            + ", ".join(sorted(entry["shots"]))
            + "."
        )
    return {
        "forbidden_abstractions_addendum": forbidden[:6],
        "critic_priorities_addendum": priorities[:4],
        "required_anchors": _dedupe_strings(required_anchors)[:12],
        "required_labels": _dedupe_strings(required_labels)[:12],
        "required_relationships": _dedupe_strings(required_relationships)[:12],
        "source_issues": source_issues[:12],
    }


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        norm = " ".join(str(item).strip().split())
        key = norm.lower()
        if norm and key not in seen:
            out.append(norm)
            seen.add(key)
    return out


def _extract_grounding_constraints_from_issue(issue) -> dict[str, list[str]]:
    anchors: list[str] = []
    labels: list[str] = []
    relationships: list[str] = []
    text = issue.issue or ""
    fix = issue.suggested_fix or {}
    for raw_key, value in fix.items():
        key = str(raw_key)
        if key.startswith("_"):
            continue
        root = key.split(".")[0]
        if _looks_like_anchor(root):
            anchors.append(root)
        for match in _FIX_KEY_ANCHOR_RE.findall(key):
            if _looks_like_anchor(match):
                anchors.append(match)
        low_key = key.lower()
        if any(marker in low_key for marker in _LABEL_KEY_MARKERS):
            if isinstance(value, str):
                labels.append(value)
            elif key.endswith(".visible") or key.endswith("_visible"):
                labels.append(root)
        if isinstance(value, str) and any(
            marker in value.lower() for marker in ("=", "label", "formula")
        ):
            labels.append(value)
        if any(marker in low_key for marker in _RELATIONSHIP_MARKERS):
            relationships.append(key.replace("_", " "))
        if isinstance(value, bool) and value and any(
            marker in low_key for marker in _RELATIONSHIP_MARKERS
        ):
            relationships.append(key.replace("_", " "))
    for match in _FIX_KEY_ANCHOR_RE.findall(text):
        if _looks_like_anchor(match):
            anchors.append(match)
    low_text = text.lower()
    if "x = f" in low_text or "x=f" in low_text:
        labels.append("x = f X/Z")
    if "p(x,y)" in low_text:
        labels.append("p(x,y)")
    if "p(x, y)" in low_text:
        labels.append("p(x,y)")
    if re.search(r"\bC\b", text) and "pinhole" in low_text:
        labels.append("C")
    if "object_point_b" in low_text or "point b" in low_text:
        anchors.append("object_point_b")
        labels.append("B")
    if any(marker in low_text for marker in _RELATIONSHIP_MARKERS):
        relationships.append(text[:180])
    return {
        "required_anchors": _dedupe_strings(anchors),
        "required_labels": _dedupe_strings(labels),
        "required_relationships": _dedupe_strings(relationships),
    }


def _looks_like_anchor(value: str) -> bool:
    low = value.lower()
    if low in _ANCHOR_KEY_STOPWORDS:
        return False
    if len(value) < 3:
        return False
    return "_" in value and any(ch.isalpha() for ch in value)


def _merge_grounding_patch_into_visual_contracts(
    visual_contracts: dict,
    patch: dict,
) -> dict:
    if not patch:
        return visual_contracts
    anchors = _dedupe_strings(patch.get("required_anchors") or [])
    labels = _dedupe_strings(patch.get("required_labels") or [])
    relationships = _dedupe_strings(patch.get("required_relationships") or [])
    if not anchors and not labels and not relationships:
        return visual_contracts
    out = dict(visual_contracts)
    for shot_id, contract in list(out.items()):
        if not isinstance(contract, VisualContract):
            continue
        out[shot_id] = contract.model_copy(update={
            "required_anchors": _dedupe_strings(
                list(contract.required_anchors) + anchors
            ),
            "required_labels": _dedupe_strings(
                list(contract.required_labels) + labels
            ),
            "required_relationships": _dedupe_strings(
                list(contract.required_relationships) + relationships
            ),
        })
    return out


def _success_spec_text_anchor_names(
    success_spec: SuccessSpec | None,
) -> set[str]:
    if success_spec is None:
        return set()
    return (
        success_spec.text_faces_camera_anchors()
        | success_spec.readable_text_anchors()
    )


def _merge_success_spec_into_visual_contracts(
    visual_contracts: dict[str, VisualContract],
    success_spec: SuccessSpec | None,
) -> dict[str, VisualContract]:
    if success_spec is None or not visual_contracts:
        return visual_contracts
    anchors = sorted(success_spec.required_anchor_names())
    text_anchors = sorted(
        success_spec.text_faces_camera_anchors()
        | success_spec.readable_text_anchors()
    )
    if not anchors and not text_anchors:
        return visual_contracts

    out: dict[str, VisualContract] = {}
    for shot_id, contract in visual_contracts.items():
        merged = contract.model_copy(deep=True)
        merged.required_anchors = sorted(
            set(merged.required_anchors) | set(anchors),
            key=str.lower,
        )
        merged.required_labels = sorted(
            set(merged.required_labels) | set(text_anchors),
            key=str.lower,
        )
        if text_anchors:
            merged.required_relationships = sorted(
                set(merged.required_relationships)
                | {
                    "Success Spec readable text anchors must be exact-named Blender text objects, camera-facing, non-mirrored, legible, and inside the frame safety margin.",
                },
                key=str.lower,
            )
            merged.forbidden_failures = sorted(
                set(merged.forbidden_failures)
                | {
                    "Do not crop, occlude, mirror, defocus, or alias Success Spec readable text anchors.",
                },
                key=str.lower,
            )
        out[shot_id] = merged
    return out


def _format_grounding_patch_for_coder(patch: dict) -> str:
    forbidden = patch.get("forbidden_abstractions_addendum") or []
    priorities = patch.get("critic_priorities_addendum") or []
    anchors = patch.get("required_anchors") or []
    labels = patch.get("required_labels") or []
    relationships = patch.get("required_relationships") or []
    if not forbidden and not priorities and not anchors and not labels and not relationships:
        return ""
    lines = [
        "RUN-LOCAL GROUNDING PATCH:",
        "Treat this as an extension of the STYLE PROFILE for this retry. "
        "It comes from repeated pixel-level critic evidence in the current run.",
    ]
    if forbidden:
        lines.append("- forbidden_abstractions_addendum:")
        lines.extend(f"  - {item}" for item in forbidden)
    if priorities:
        lines.append("- critic_priorities_addendum:")
        lines.extend(f"  - {item}" for item in priorities)
    if anchors:
        lines.append("- required_anchors_from_critic:")
        lines.extend(f"  - {item}" for item in anchors)
    if labels:
        lines.append("- required_labels_from_critic:")
        lines.extend(f"  - {item}" for item in labels)
    if relationships:
        lines.append("- required_relationships_from_critic:")
        lines.extend(f"  - {item}" for item in relationships)
    return "\n".join(lines)


def _storyboard_names_by_shot(storyboard: Storyboard) -> dict[str, set[str]]:
    return {
        shot.node_id: {obj.name for obj in shot.objects}
        for shot in storyboard.shots
    }


def _missing_storyboard_contract_anchors(
    storyboard: Storyboard,
    visual_contracts: dict,
) -> dict[str, list[str]]:
    names_by_shot = _storyboard_names_by_shot(storyboard)
    missing: dict[str, list[str]] = {}
    for shot_id, contract in sorted((visual_contracts or {}).items()):
        shot_names = names_by_shot.get(shot_id, set())
        for anchor in getattr(contract, "required_anchors", []) or []:
            anchor_low = anchor.lower()
            if not any(
                anchor_low in name.lower() or name.lower() in anchor_low
                for name in shot_names
            ):
                missing.setdefault(shot_id, []).append(anchor)
    return missing


def _anchor_storyboard_object(anchor: str, shot_index: int, shot_start: float) -> dict:
    low = anchor.lower()
    x = -2.5 + 0.35 * (shot_index % 4)
    y = 1.8
    z = 0.65 + 0.12 * (shot_index % 3)
    if "open" in low or "sign" in low or "label" in low:
        text = "OPEN" if "open" in low else anchor.replace("_", " ")
        return {
            "name": anchor,
            "type": "text",
            "primitive": None,
            "location": [x, y, z + 0.5],
            "properties": {
                "text": text,
                "size": 0.28,
                "color": [0.2, 0.9, 1.0],
                "emission": 0.4,
            },
            "keyframes": [{
                "time_sec": shot_start,
                "attr": "location",
                "value": [x, y, z + 0.5],
            }],
        }
    primitive = "plane" if any(w in low for w in ("pane", "plane", "screen")) else "cube"
    scale = (
        [1.3, 0.06, 0.75]
        if any(w in low for w in ("frame", "wall", "plane", "pane"))
        else [0.35, 0.35, 0.35]
    )
    return {
        "name": anchor,
        "type": "mesh",
        "primitive": primitive,
        "location": [x, y, z],
        "properties": {
            "size": 1.0,
            "color": [0.25, 0.32, 0.38],
            "alpha": 0.45 if primitive == "plane" else 1.0,
            "scale": scale,
        },
        "keyframes": [
            {"time_sec": shot_start, "attr": "location", "value": [x, y, z]},
            {"time_sec": shot_start, "attr": "scale", "value": scale},
        ],
    }


def _augment_storyboard_with_contract_anchors(
    storyboard: Storyboard,
    missing_by_shot: dict[str, list[str]],
) -> tuple[Storyboard, dict]:
    if not missing_by_shot:
        return storyboard, {"changed": False, "added": {}}
    raw = storyboard.model_dump(mode="json")
    added: dict[str, list[str]] = {}
    for idx, shot in enumerate(raw.get("shots", [])):
        shot_id = shot.get("node_id")
        anchors = missing_by_shot.get(shot_id, [])
        if not anchors:
            continue
        existing = {obj.get("name") for obj in shot.get("objects", [])}
        for anchor in anchors:
            if anchor in existing:
                continue
            shot.setdefault("objects", []).append(
                _anchor_storyboard_object(
                    anchor, idx, float(shot.get("start_sec", 0.0)),
                )
            )
            added.setdefault(shot_id, []).append(anchor)
    if not added:
        return storyboard, {"changed": False, "added": {}}
    return Storyboard.model_validate(raw), {"changed": True, "added": added}


def _should_use_fresh_retry(
    critic_history: list[CriticIteration],
    *,
    iteration: int,
    max_iteration: int,
) -> bool:
    if iteration <= 0 or iteration != max_iteration:
        return False
    if len(critic_history) < 3:
        return False
    _floor, stale = _block_floor_stale_iters(critic_history)
    return stale >= 2


def _critic_iteration_summary(item: CriticIteration) -> dict:
    block, warn = _critic_counts(item.report)
    sem_block, sem_warn = _semantic_counts(item.report)
    structural, success_hard, success_soft, aesthetic_warn = (
        _failure_class_counts(item)
    )
    return {
        "iteration": item.iteration,
        "overall_score": item.report.overall_score,
        "block": block,
        "warn": warn,
        "concept_mismatch_block": sem_block,
        "concept_mismatch_warn": sem_warn,
        "render_ok": item.render_ok,
        "n_frames": item.n_frames,
        "frames_hash": item.frames_hash,
        "scene_origin": item.scene_origin,
        "fallback_degraded": item.fallback_degraded,
        "frames_dir": (
            item.frames_dir.name
            if item.frames_dir is not None and item.frames_dir.exists()
            else None
        ),
        "metric_block": item.metric_block_count,
        "metric_warn": item.metric_warn_count,
        "metric_structural_fatal": item.metric_structural_fatal_count,
        "metric_success_hard": item.metric_success_hard_count,
        "metric_success_soft": item.metric_success_soft_count,
        "metric_aesthetic_warn": item.metric_aesthetic_warn_count,
        "structural_fatal": structural,
        "success_hard": success_hard,
        "success_soft": success_soft,
        "aesthetic_warn": aesthetic_warn,
        "cross_ref_actionable": item.cross_ref_actionable_count,
        "pass": _critic_iteration_pass(item),
    }


def _selected_reason(
    best: CriticIteration,
    history: list[CriticIteration],
) -> str:
    if _critic_iteration_pass(best):
        return "pass"
    if best.fallback_degraded:
        renderable_llm = [
            item for item in history
            if item.render_ok
            and not item.fallback_degraded
            and not _is_compiled_origin(item.scene_origin)
        ]
        if not renderable_llm:
            return "no_renderable_llm_candidate"
        return "fallback_diagnostic"
    return "least_violating_llm"


def _select_video_variants(
    history: list[CriticIteration],
    *,
    primary_mode: BestSelectionMode,
) -> dict[str, CriticIteration]:
    """Pick named export variants from the evaluated critic history."""
    primary = max(history, key=lambda r: _critic_quality_key_for(r, primary_mode))
    return {
        "selected": primary,
        "balanced": max(history, key=lambda r: _critic_quality_key_for(r, "balanced")),
        "compliance": max(history, key=lambda r: _critic_quality_key_for(r, "framing")),
        "aesthetic": max(history, key=lambda r: r.report.overall_score),
        "semantic": max(history, key=lambda r: _critic_quality_key_for(r, "semantic")),
    }


def _strict_selection_history(
    history: list[CriticIteration],
) -> list[CriticIteration]:
    """Prefer iterations with exact critic-scored frame snapshots."""
    snapshotted = [
        item for item in history
        if item.frames_dir is not None and item.frames_dir.exists()
    ]
    return snapshotted or history


@dataclass
class PipelineResult:
    concept_id: str
    out_dir: Path
    final_mp4: Path | None
    narrative_path: Path
    storyboard_path: Path
    scene_path: Path
    frames_dir: Path
    elapsed_sec: float

    @property
    def ok(self) -> bool:
        return self.final_mp4 is not None and self.final_mp4.exists()


@dataclass
class ResumeCheckpoint:
    iteration: int
    stage: str
    scene_origin: str = "unknown"
    render_ok: bool = True
    n_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "stage": self.stage,
            "scene_origin": self.scene_origin,
            "render_ok": self.render_ok,
            "n_frames": self.n_frames,
        }


def _resume_checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "resume_checkpoint.json"


def _save_resume_checkpoint(out_dir: Path, checkpoint: ResumeCheckpoint) -> None:
    save_artifact(
        out_dir,
        "resume_checkpoint.json",
        json.dumps(checkpoint.to_dict(), indent=2),
    )


def _clear_resume_checkpoint(out_dir: Path) -> None:
    path = _resume_checkpoint_path(out_dir)
    if path.exists():
        path.unlink()


def _load_resume_checkpoint(out_dir: Path) -> ResumeCheckpoint | None:
    path = _resume_checkpoint_path(out_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return ResumeCheckpoint(
            iteration=int(raw.get("iteration", 0)),
            stage=str(raw.get("stage", "") or ""),
            scene_origin=str(raw.get("scene_origin", "unknown") or "unknown"),
            render_ok=bool(raw.get("render_ok", True)),
            n_frames=int(raw.get("n_frames", 0) or 0),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"resume checkpoint unreadable, ignoring: {e!r}")
        return None


def _load_preview_report(path: Path) -> PreviewReport | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        issues = [
            PreviewIssue(
                severity=str(item.get("severity", "warn")),
                rule_id=str(item.get("rule_id", "")),
                frame_idx=item.get("frame_idx"),
                message=str(item.get("message", "")),
                suggested_fix=str(item.get("suggested_fix", "")),
            )
            for item in raw.get("issues", [])
        ]
        return PreviewReport(
            ok=bool(raw.get("ok", False)),
            rendered_frames=[int(x) for x in raw.get("rendered_frames", [])],
            issues=issues,
            skipped_reason=str(raw.get("skipped_reason", "")),
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"{path.name} unreadable preview report, ignoring: {e!r}")
        return None


def run(
    concept_path: Path,
    *,
    out_root: Path = DEFAULT_OUTPUT_ROOT,
    decomposer_model: str | None = None,
    storyboard_model: str | None = None,
    coder_model: str | None = None,
    critic_backend: str | None = None,
    max_critic_iterations: int = 5,
    early_stop_stale_iters: int = 0,
    best_selection_mode: BestSelectionMode = "balanced",
    compiler_seed: bool = True,
    compiler_only: bool = False,
    preview_render: bool = True,
    diff_repair: bool = True,
    critic_ensemble: tuple[str, ...] | None = None,
    resume: bool = False,
    keep_frames: bool = False,
    blender_timeout_sec: int = 3600,
    strict_best_replay: bool = True,
    max_verifier_repair_iters: int = 2,
    critic_strictness: str = "strict",
    render_engine: str = "BLENDER_EEVEE",
    cycles_device: str = "AUTO",
) -> PipelineResult:
    """Run end-to-end. With `resume=True`, reuse cached artifacts on disk
    (narrative.json / storyboard.json / scene.py), restore saved critic
    history when available, and only re-do stages whose output is missing.
    With `keep_frames=True`, skip the Blender render entirely and reuse
    whatever PNGs are already in frames/.

    Critic loop (W3): set `max_critic_iterations > 0` to enable. The
    pipeline runs Coder → Render → Critic up to (max + 1) times. Each
    retry asks the Coder to address the previous Critic's issues.
    `critic_backend` ∈ {None, 'passthrough', 'claude', 'gpt', 'gemini',
    'google'}; None uses the configured API ensemble by default.

    `early_stop_stale_iters`: exit the loop early when the running min
    block count has not decreased for this many consecutive iterations.
    Saves LLM tokens when the Coder is stuck trading issues without net
    progress. Set to 0 to disable."""
    spec = yaml.safe_load(concept_path.read_text())
    concept_id = _validate_concept_id(spec["concept_id"])
    out_root = Path(out_root).resolve()
    if best_selection_mode not in {"framing", "balanced", "semantic"}:
        raise ValueError(
            "best_selection_mode must be one of: framing, balanced, semantic"
        )
    if critic_strictness not in {"consensus", "union", "strict"}:
        raise ValueError(
            "critic_strictness must be one of: consensus, union, strict"
        )
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be one of: BLENDER_EEVEE, CYCLES")
    if cycles_device not in {"AUTO", "GPU", "CPU"}:
        raise ValueError("cycles_device must be one of: AUTO, GPU, CPU")
    max_verifier_repair_iters = max(1, int(max_verifier_repair_iters))
    if critic_ensemble is None:
        critic_ensemble = (
            ("claude", "gpt") if critic_backend is None else ()
        )
    out_dir = (out_root / concept_id).resolve()
    # Defense in depth: even after the regex check, ensure no symlink
    # trickery walks us out of out_root.
    if out_root != out_dir.parent and out_root not in out_dir.parents:
        raise ValueError(
            f"resolved out_dir={out_dir} escapes out_root={out_root}; "
            "refusing to proceed"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    success_spec = load_success_spec(spec)
    if success_spec is not None:
        save_artifact(out_dir, "success_spec.json", success_spec_to_json(success_spec))

    log.info(ui.banner(f"CG-Tutor | {concept_id} | {'resume' if resume else 'fresh'}"))
    log.info(ui.kv("output", out_dir))
    log.info(ui.kv(
        "config",
        (
            f"critic_iters=0..{max_critic_iterations}, "
            f"early_stop={early_stop_stale_iters or 'off'}, "
            f"best={best_selection_mode}, "
            f"preview={_onoff(preview_render)}, "
            f"render_engine={render_engine}, "
            f"cycles_device={cycles_device if render_engine == 'CYCLES' else 'n/a'}, "
            f"critic={','.join(critic_ensemble) if critic_ensemble else 'off'}, "
            f"critic_strictness={critic_strictness}"
        ),
    ))

    narrative_path = out_dir / "narrative.json"
    scene_profile_path = out_dir / "scene_profile.json"
    storyboard_path = out_dir / "storyboard.json"
    storyboard_raw_path = out_dir / "storyboard.raw.json"
    scene_path = out_dir / "scene.py"

    # 1. Decompose
    if resume and narrative_path.exists():
        from cg_tutor.schemas import Narrative
        narrative = Narrative.model_validate_json(narrative_path.read_text())
        log.info(ui.step(1, 7, "Concept decomposer", "cached"))
        log.info(ui.detail("narrative", f"{len(narrative.nodes)} nodes"))
    else:
        log.info(ui.step(1, 7, "Concept decomposer"))
        decomposer_kwargs = {"out_dir": out_dir}
        if decomposer_model:
            decomposer_kwargs["model"] = decomposer_model
        narrative = concept_decomposer.decompose(spec, **decomposer_kwargs)
        log.info(ui.ok(
            f"narrative: {len(narrative.nodes)} node(s), "
            f"{narrative.total_duration:.1f}s"
        ))

    # 1b. Scene profile: deterministic base policy plus optional LLM
    # validation/completion. This happens after narrative so the profile can
    # use the actual shot intent, but before storyboard so it can constrain
    # object/helper choices.
    if resume and scene_profile_path.exists():
        scene_profile = SceneProfile.model_validate_json(
            scene_profile_path.read_text()
        )
        profile_source = "cached"
    else:
        resolved_profile = profile_generator.resolve_scene_profile(
            spec,
            narrative,
            out_dir=out_dir,
        )
        scene_profile = resolved_profile.profile
        profile_source = resolved_profile.source
    style_profile_text = format_scene_profile_for_prompt(scene_profile)
    log.info(ui.detail(
        "scene profile",
        f"{scene_profile.profile_id} ({scene_profile.base_profile}, {profile_source})",
    ))

    # 2. Storyboard
    if resume and storyboard_path.exists():
        sb = Storyboard.model_validate_json(storyboard_path.read_text())
        log.info(ui.step(2, 7, "Storyboard", "cached"))
        log.info(ui.detail("shots", f"{len(sb.shots)}, {sb.total_frames} frames"))
    elif resume and storyboard_raw_path.exists():
        raw = json.loads(storyboard_raw_path.read_text())
        try:
            sb = storyboard_agent.validate_storyboard(raw)
            save_artifact(out_dir, "storyboard.json", sb.model_dump_json(indent=2))
            log.info(ui.step(2, 7, "Storyboard", "cached raw"))
            log.info(ui.detail("shots", f"{len(sb.shots)}, {sb.total_frames} frames"))
        except Exception as e:  # noqa: BLE001
            log.info(ui.warn(f"cached raw storyboard invalid: {e}"))
            log.info(ui.step(2, 7, "Storyboard", "regenerating"))
            sb_kwargs = {
                "out_dir": out_dir,
                "candidates": 1,
                "style_addendum": style_profile_text,
                "scene_profile": scene_profile,
            }
            if storyboard_model:
                sb_kwargs["model"] = storyboard_model
            sb = storyboard_agent.to_storyboard(narrative, **sb_kwargs)
            log.info(ui.ok(
                f"storyboard: {len(sb.shots)} shot(s), "
                f"{sb.total_duration:.1f}s, {sb.total_frames} frames"
            ))
    else:
        log.info(ui.step(2, 7, "Storyboard"))
        sb_kwargs = {
            "out_dir": out_dir,
            "candidates": 1,
            "style_addendum": style_profile_text,
            "scene_profile": scene_profile,
        }
        if storyboard_model:
            sb_kwargs["model"] = storyboard_model
        sb = storyboard_agent.to_storyboard(narrative, **sb_kwargs)
        log.info(ui.ok(
            f"storyboard: {len(sb.shots)} shot(s), "
            f"{sb.total_duration:.1f}s, {sb.total_frames} frames"
        ))

    sb, storyboard_sanitization = sanitize_storyboard_for_pipeline(
        sb,
        scene_profile=scene_profile,
    )
    if storyboard_sanitization.changed:
        save_artifact(
            out_dir,
            "storyboard.sanitization.json",
            json.dumps(storyboard_sanitization.to_dict(), indent=2),
        )
        save_artifact(out_dir, "storyboard.json", sb.model_dump_json(indent=2))
        removed_count = sum(
            len(names)
            for names in storyboard_sanitization.removed_formula_objects.values()
        )
        log.info(ui.warn(
            f"storyboard sanitizer removed {removed_count} formula object(s); "
            "see storyboard.sanitization.json"
        ))

    auto_success_spec, auto_success_validation = generate_auto_success_spec(
        concept_spec=spec,
        narrative=narrative,
        scene_profile=scene_profile,
        storyboard=sb,
        user_success_spec=success_spec,
    )
    save_artifact(
        out_dir,
        "success_spec.generated.json",
        auto_success_spec_to_json(auto_success_spec),
    )
    save_artifact(
        out_dir,
        "success_spec.validation.json",
        auto_success_validation_to_json(auto_success_validation),
    )
    save_artifact(
        out_dir,
        "success_spec.effective.json",
        json.dumps(
            {
                "user_success_spec_present": success_spec is not None,
                "user_success_spec": (
                    success_spec.model_dump(mode="json")
                    if success_spec is not None else None
                ),
                "generated": auto_success_spec.model_dump(mode="json"),
            },
            indent=2,
        ),
    )

    scene_ir = build_scene_ir(narrative, sb, scene_profile=scene_profile)
    scene_ir_report = verify_scene_ir(scene_ir)
    visual_contracts = {
        shot.node_id: shot.visual_contract
        for shot in scene_ir.shots
        if shot.visual_contract is not None
    }
    visual_contracts = _merge_success_spec_into_visual_contracts(
        visual_contracts,
        success_spec,
    )
    save_artifact(out_dir, "scene_ir.json", scene_ir_to_json(scene_ir))
    save_artifact(
        out_dir,
        "visual_contracts.json",
        json.dumps(
            {
                shot_id: contract.model_dump()
                for shot_id, contract in visual_contracts.items()
            },
            indent=2,
        ),
    )
    save_artifact(
        out_dir,
        "scene_ir_verifier.json",
        scene_ir_verification_to_json(scene_ir_report),
    )
    if scene_ir_report.issues:
        log.info(ui.detail(
            "scene IR verifier",
            f"{scene_ir_report.block_count}b/{scene_ir_report.warn_count}w",
        ))
    compiled_scene = compile_storyboard_to_bpy(
        sb,
        render_engine=render_engine,
        cycles_device=cycles_device,
        visual_contracts=visual_contracts,
    )
    save_artifact(out_dir, "scene.compiled.py", compiled_scene)

    # 3 + 4 + 6 form the Critic loop. Iteration 0 either uses cached
    # scene.py (if --resume) or generates fresh. Iterations 1..N
    # regenerate scene.py with a history-aware critic summary, then
    # re-render. When critic is enabled, every rendered iteration is
    # evaluated and the final mp4 is composed from the best-scoring scene,
    # not blindly from the last retry.
    frames_dir = out_dir / "frames"
    critic_report: CriticReport | None = None
    critic_history: list[CriticIteration] = []
    metric_history: list[ConceptMetricReport] = []
    cross_ref_history: list[CrossReferenceReport] = []
    last_metric_addendum: str = ""
    last_cross_ref_addendum: str = ""
    last_contract_report = None
    prev_iter_code_hash: str | None = None
    frames_hash_cache = _FramesHashCache()
    current_missing_objects: dict[str, list[str]] = {}
    memory_path = out_root / "_failure_memory.jsonl"
    prior_memory = load_failure_memory(memory_path, concept_id)
    if prior_memory:
        save_failure_memory_snapshot(
            out_dir / "failure_memory.prior.json",
            prior_memory,
        )
    shot_contract_text = _join_addenda(
        style_profile_text,
        format_success_spec_for_coder(success_spec),
        format_auto_success_spec_for_coder(auto_success_spec),
        format_scene_ir_for_coder(scene_ir, scene_ir_report),
        _shot_visual_contracts(narrative, sb, scene_profile=scene_profile),
        _format_vector_ray_minimum_scaffold(visual_contracts),
        format_failure_memory_for_coder(prior_memory),
    )

    if resume:
        critic_history = _load_critic_history_from_disk(out_dir)
        metric_history = _load_metric_history_from_disk(out_dir)
        cross_ref_history = _load_cross_ref_history_from_disk(out_dir)
        _hydrate_history_diagnostics(
            critic_history,
            metric_history,
            cross_ref_history,
        )
        if metric_history:
            last_metric_addendum = format_concept_metric_report_for_coder(
                metric_history[-1]
            )
        if cross_ref_history:
            last_cross_ref_addendum = format_cross_reference_for_coder(
                cross_ref_history[-1]
            )
        if critic_history:
            gaps = _detect_history_gaps(critic_history)
            if gaps:
                # A gap means one of the per-iter files was unreadable or
                # never written. Continuing risks treating a stale "best"
                # as a contiguous record. Refuse rather than silently
                # paper over it; the user can delete the bad file or
                # start fresh with --no-resume.
                gap_list = ", ".join(f"iter{g:02d}" for g in gaps)
                raise RuntimeError(
                    f"resume: critic history has gaps ({gap_list}); refuse "
                    "to continue. Either remove the broken critic_iterNN.json "
                    "files and resume to backfill from the next contiguous "
                    "iter, or restart without --resume."
                )
            resume_best = max(
                critic_history,
                key=lambda r: _critic_quality_key_for(r, best_selection_mode),
            )
            log.info(ui.step(0, 7, "Resume state"))
            log.info(ui.detail(
                "critic history",
                (
                    f"{len(critic_history)} iteration(s); "
                    f"last=iter{critic_history[-1].iteration:02d}; "
                    f"best=iter{resume_best.iteration:02d}"
                ),
            ))
            log.info(ui.detail("best score", _critic_status_text(resume_best.report)))
        else:
            log.info(ui.detail("critic history", "none cached"))
        resume_scene_source = _select_resume_scene_source(scene_path, critic_history)
        if resume_scene_source is not None:
            restored_text = resume_scene_source.read_text()
            compat_text = _compat_scene_text(restored_text)
            if resume_scene_source != scene_path:
                scene_path.write_text(compat_text)
                log.info(ui.detail("scene.py", f"restored from {resume_scene_source.name}"))
            elif compat_text != restored_text:
                scene_path.write_text(compat_text)
                log.info(ui.detail("scene.py", "patched for Blender API compatibility"))
            current_missing_objects = _missing_storyboard_objects(
                scene_path.read_text(), sb,
            )
            prev_iter_code_hash = hashlib.sha256(
                scene_path.read_bytes()
            ).hexdigest()

    start_iter = max((r.iteration for r in critic_history), default=-1) + 1
    resume_checkpoint = _load_resume_checkpoint(out_dir) if resume else None
    resumed_partial_iter = False
    resumed_partial_stage = ""
    if resume_checkpoint is not None:
        if resume_checkpoint.iteration < start_iter:
            log.info(ui.detail(
                "resume checkpoint",
                (
                    f"stale iter{resume_checkpoint.iteration:02d}/"
                    f"{resume_checkpoint.stage}; ignoring"
                ),
            ))
        elif resume_checkpoint.iteration > max_critic_iterations:
            log.info(ui.detail(
                "resume checkpoint",
                (
                    f"iter{resume_checkpoint.iteration:02d} beyond max iter "
                    f"{max_critic_iterations:02d}; ignoring"
                ),
            ))
        else:
            start_iter = resume_checkpoint.iteration
            resumed_partial_iter = True
            resumed_partial_stage = resume_checkpoint.stage
            log.info(ui.detail(
                "resume checkpoint",
                (
                    f"resume iter{resume_checkpoint.iteration:02d} from "
                    f"{resume_checkpoint.stage}"
                ),
            ))
    if start_iter > 0:
        if start_iter <= max_critic_iterations:
            log.info(ui.detail(
                "resume cursor",
                f"continue at iter{start_iter:02d} / max iter{max_critic_iterations:02d}",
            ))
        else:
            log.info(ui.detail(
                "resume cursor",
                (
                    f"no remaining critic iterations "
                    f"(next iter{start_iter:02d} > max iter{max_critic_iterations:02d}); "
                    "compose saved best"
                ),
            ))

    for it in range(start_iter, max_critic_iterations + 1):
        log.info(ui.iter_header(it, max_critic_iterations))

        # Default optimistically; the normal coder path overrides this with
        # the post-repair contract validation result.
        contract_anchors_ok = True
        scene_origin = "unknown"
        grounding_patch_text = ""
        grounding_patch: dict = {}
        resume_stage_for_iter = ""
        if (
            resumed_partial_iter
            and resume_checkpoint is not None
            and it == resume_checkpoint.iteration
        ):
            resume_stage_for_iter = resumed_partial_stage
        correction_decision = CorrectionDecision(iteration=it)
        if critic_history:
            grounding_patch = _critic_grounding_patch(critic_history)
            visual_contracts = _merge_grounding_patch_into_visual_contracts(
                visual_contracts,
                grounding_patch,
            )
            grounding_patch_text = _format_grounding_patch_for_coder(
                grounding_patch,
            )
            if grounding_patch_text:
                save_artifact(
                    out_dir,
                    f"grounding_patch.iter{it:02d}.json",
                    json.dumps(grounding_patch, indent=2),
                )

        # 3. Coder (skipped on it=0 if resume+cached)
        if it == 0 and resume and _scene_cache_is_usable(scene_path):
            scene_origin = "cached"
            log.info(ui.step(3, 7, "Blender coder", "cached"))
            log.info(ui.detail("scene.py", f"{len(scene_path.read_text().splitlines())} lines"))
            # Re-derive the deterministic name check from the cached
            # scene.py so the addendum still gets accurate "missing
            # objects" data on an addendum-driven retry.
            current_missing_objects = _missing_storyboard_objects(
                scene_path.read_text(), sb,
            )
            prev_iter_code_hash = hashlib.sha256(
                scene_path.read_bytes()
            ).hexdigest()
            # Re-validate the cached scene against the current contracts
            # so a profile change between runs (e.g. new persistent_anchors)
            # doesn't silently inherit a stale pass.
            cached_contract_report = validate_visual_contracts(
                scene_path.read_text(), sb, visual_contracts,
                scene_profile=scene_profile,
                success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
            )
            contract_anchors_ok = not any(
                v.severity == "block"
                and v.rule_id == "contract_missing_scene_anchors"
                for v in cached_contract_report.violations
            )
        else:
            log.info(ui.step(
                3, 7, "Blender coder",
                "retry with critic feedback" if it > 0 else None,
            ))
            coder_kwargs = {
                "out_dir": out_dir,
                "iteration": it,
                "render_engine": render_engine,
                "cycles_device": cycles_device,
                "addendum": _join_addenda(
                    shot_contract_text,
                    grounding_patch_text,
                ),
            }
            if coder_model:
                coder_kwargs["model"] = coder_model
            if compiler_only and it == 0:
                code = compiled_scene
                scene_origin = "compiler_only"
                save_artifact(out_dir, "scene.py", code)
                save_artifact(out_dir, "scene.compiled.used.txt", "compiler_only=true\n")
                log.info(ui.ok(f"scene.py: {len(code.splitlines())} lines (compiled)"))
            else:
                if compiler_seed and it == 0:
                    coder_kwargs["base_scene"] = compiled_scene
                if it > 0 and critic_history:
                    correction_decision = decide_correction(
                        iteration=it,
                        critic_history=critic_history,
                        metric_history=metric_history,
                        latest_cross_ref=(
                            cross_ref_history[-1] if cross_ref_history else None
                        ),
                        max_iteration=max_critic_iterations,
                    )
                    save_artifact(
                        out_dir,
                        f"correction_decision.iter{it:02d}.json",
                        correction_decision_to_json(correction_decision),
                    )
                    if correction_decision.changed_strategy:
                        log.info(ui.detail(
                            "correction",
                            (
                                f"{correction_decision.action}; "
                                f"diff={_onoff(correction_decision.allow_diff_repair)}; "
                                f"fresh={_onoff(correction_decision.force_fresh_branch)}"
                            ),
                        ))
                    fresh_retry = _should_use_fresh_retry(
                        critic_history,
                        iteration=it,
                        max_iteration=max_critic_iterations,
                    ) or correction_decision.force_fresh_branch
                    if fresh_retry:
                        log.info(ui.detail(
                            "retry mode",
                            "fresh branch from storyboard/compiler scaffold",
                        ))
                        if compiler_seed:
                            coder_kwargs["base_scene"] = compiled_scene
                        fresh_addendum = AddendumBundle(
                            priority=correction_decision.priority_addendum,
                            metric=last_metric_addendum,
                            cross_ref=last_cross_ref_addendum,
                            shot_contract=shot_contract_text,
                            grounding_patch=grounding_patch_text,
                            history="FRESH RETRY MODE:\n"
                            "- Start from the storyboard, Scene IR, visual contracts, and deterministic scaffold.\n"
                            "- Do not anchor on prior scene parameter choices or local diffs.\n"
                            "- Preserve hard render boilerplate, object names, visibility gating, and grounding constraints.\n"
                            "- Use this branch to escape a stale local optimum; still satisfy every required label/vector/anchor.",
                            has_success_hard=bool(
                                metric_history
                                and metric_history[-1].success_hard_count > 0
                            ),
                        ).join()
                        coder_kwargs["addendum"] = fresh_addendum
                        save_artifact(out_dir, f"trend_iter{it:02d}.txt", fresh_addendum)
                        code = blender_coder.to_bpy_script(sb, **coder_kwargs)
                        scene_origin = "fresh"
                        log.info(ui.ok(
                            f"scene.py: {len(code.splitlines())} lines "
                            "(fresh branch)"
                        ))
                    else:
                        best_for_retry = max(
                            critic_history,
                            key=lambda r: _critic_quality_key_for(r, "framing"),
                        )
                        log.info(ui.detail(
                            "retry base",
                            (
                                f"iter{best_for_retry.iteration:02d} "
                                f"(framing-safe; {_critic_status_text(best_for_retry.report)})"
                            ),
                        ))
                        semantic_ref = max(
                            critic_history,
                            key=lambda r: _critic_quality_key_for(r, "semantic"),
                        )
                        aesthetic_ref = max(
                            critic_history,
                            key=lambda r: r.report.overall_score,
                        )
                        log.info(ui.detail(
                            "references",
                            (
                                f"semantic=iter{semantic_ref.iteration:02d} "
                                f"({_critic_status_text(semantic_ref.report)}); "
                                f"aesthetic=iter{aesthetic_ref.iteration:02d} "
                                f"(score={aesthetic_ref.report.overall_score:.2f})"
                            ),
                        ))
                        base_scene_text = ""
                        diff_base = best_for_retry
                        if _is_compiled_origin(best_for_retry.scene_origin):
                            diff_base = _latest_non_compiled_iteration(critic_history)
                            if diff_base is None:
                                log.info(ui.detail(
                                    "retry base",
                                    "best prior is compiled; forcing full regeneration",
                                ))
                        if diff_base is not None and diff_base.scene_path.exists():
                            base_scene_text = diff_base.scene_path.read_text()
                            coder_kwargs["base_scene"] = base_scene_text
                        history_addendum_text = _critic_history_addendum(
                            critic_history,
                            shot_ids=[shot.node_id for shot in sb.shots],
                        )
                        plan = build_repair_plan(
                            iteration=it,
                            history=critic_history,
                            contract_report=last_contract_report,
                            missing_objects=current_missing_objects,
                            visual_contracts=visual_contracts,
                            scene_profile=scene_profile,
                            best_selection_mode=best_selection_mode,
                        )
                        save_artifact(
                            out_dir,
                            f"repair_plan.iter{it:02d}.json",
                            repair_plan_to_json(plan),
                        )
                        log.info(ui.detail(
                            "repair plan",
                            f"{len(plan.targets)} target(s); repair_plan.iter{it:02d}.json",
                        ))
                        addendum_text = AddendumBundle(
                            priority=correction_decision.priority_addendum,
                            metric=last_metric_addendum,
                            cross_ref=last_cross_ref_addendum,
                            shot_contract=shot_contract_text,
                            grounding_patch=grounding_patch_text,
                            critic_visual_evidence=format_critic_visual_evidence_packet(
                                critic_history
                            ),
                            multi_reference=_multi_reference_retry_addendum(
                                critic_history
                            ),
                            history=history_addendum_text,
                            repair_plan=format_repair_plan_for_coder(plan),
                            has_success_hard=bool(
                                metric_history
                                and metric_history[-1].success_hard_count > 0
                            ),
                        ).join()
                        coder_kwargs["addendum"] = addendum_text
                        # Persist the exact addendum the Coder saw, so ablation can
                        # answer "did the Coder ignore the addendum, or was the
                        # addendum itself misleading?" without re-deriving it.
                        save_artifact(out_dir, f"trend_iter{it:02d}.txt", addendum_text)
                        use_diff_repair = (
                            diff_repair
                            and correction_decision.allow_diff_repair
                            and base_scene_text
                        )
                        if use_diff_repair:
                            try:
                                code = blender_coder.repair_bpy_script_diff(
                                    sb,
                                    base_scene=base_scene_text,
                                    out_dir=out_dir,
                                    iteration=it,
                                    addendum=addendum_text,
                                    model=coder_model,
                                    render_engine=render_engine,
                                    cycles_device=cycles_device,
                                )
                                scene_origin = "diff_repair"
                                log.info(ui.ok(
                                    f"scene.py: {len(code.splitlines())} lines "
                                    "(search/replace repair)"
                                ))
                            except Exception as e:  # noqa: BLE001
                                log.info(ui.warn(
                                    f"search/replace repair failed: {str(e)[:200]}"
                                ))
                                log.info(ui.detail(
                                    "fallback",
                                    "full scene regeneration",
                                ))
                                code = blender_coder.to_bpy_script(sb, **coder_kwargs)
                                scene_origin = "llm"
                                log.info(ui.ok(
                                    f"scene.py: {len(code.splitlines())} lines"
                                ))
                        else:
                            code = blender_coder.to_bpy_script(sb, **coder_kwargs)
                            scene_origin = "llm"
                            log.info(ui.ok(f"scene.py: {len(code.splitlines())} lines"))
                else:
                    try:
                        code = blender_coder.to_bpy_script(sb, **coder_kwargs)
                    except Exception as e:  # noqa: BLE001
                        if it == 0:
                            raise
                        log.info(ui.warn(
                            f"coder failed on retry ({type(e).__name__}): "
                            f"{str(e)[:200]}"
                        ))
                        log.info(ui.detail(
                            "fallback",
                            f"iter{it - 1:02d} result",
                        ))
                        break
                    scene_origin = "llm"
                    log.info(ui.ok(f"scene.py: {len(code.splitlines())} lines"))

            verifier_report = verify_scene_code(
                code, sb, visual_contracts=visual_contracts,
                scene_profile=scene_profile,
                render_engine=render_engine,
                cycles_device=cycles_device,
            )
            verifier_name = _save_scene_verification(
                out_dir, it, verifier_report,
            )
            if verifier_report.issues:
                log.info(ui.detail(
                    "verifier",
                    (
                        f"{verifier_report.block_count}b/"
                        f"{verifier_report.warn_count}w; see {verifier_name}"
                    ),
                ))
            # Structural visual-contract validation. Independent of the
            # LLM critic — surfaces "the script does NOT create the
            # required labels/vectors" with no inference involved.
            contract_report = validate_visual_contracts(
                code, sb, visual_contracts, scene_profile=scene_profile,
                success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
            )
            if contract_report.violations:
                contract_name = f"contract_validation.iter{it:02d}.json"
                save_artifact(out_dir, contract_name,
                              contract_report_to_json(contract_report))
                log.info(ui.detail(
                    "contract validator",
                    (
                        f"{contract_report.block_count}b/"
                        f"{contract_report.warn_count}w; see {contract_name}"
                    ),
                ))
                if any(
                    v.severity == "block"
                    and v.rule_id == "contract_missing_scene_anchors"
                    for v in contract_report.violations
                ):
                    missing_storyboard_anchors = (
                        _missing_storyboard_contract_anchors(sb, visual_contracts)
                    )
                    if missing_storyboard_anchors:
                        sb, storyboard_repair = (
                            _augment_storyboard_with_contract_anchors(
                                sb, missing_storyboard_anchors,
                            )
                        )
                        if storyboard_repair.get("changed"):
                            save_artifact(
                                out_dir,
                                f"storyboard.repair.iter{it:02d}.json",
                                sb.model_dump_json(indent=2),
                            )
                            save_artifact(
                                out_dir,
                                f"storyboard.repair.iter{it:02d}.summary.json",
                                json.dumps(storyboard_repair, indent=2),
                            )
                            log.info(ui.warn(
                                "storyboard repair: added missing contract "
                                "anchor(s); rebuilding Scene IR/contracts"
                            ))
                            scene_ir = build_scene_ir(
                                narrative, sb, scene_profile=scene_profile,
                            )
                            scene_ir_report = verify_scene_ir(scene_ir)
                            save_artifact(
                                out_dir,
                                "scene_ir.json",
                                scene_ir_to_json(scene_ir),
                            )
                            save_artifact(
                                out_dir,
                                "scene_ir_verifier.json",
                                scene_ir_verification_to_json(scene_ir_report),
                            )
                            visual_contracts = {
                                shot.node_id: shot.visual_contract
                                for shot in scene_ir.shots
                                if shot.visual_contract is not None
                            }
                            visual_contracts = _merge_success_spec_into_visual_contracts(
                                visual_contracts,
                                success_spec,
                            )
                            auto_success_spec, auto_success_validation = (
                                generate_auto_success_spec(
                                    concept_spec=spec,
                                    narrative=narrative,
                                    scene_profile=scene_profile,
                                    storyboard=sb,
                                    user_success_spec=success_spec,
                                )
                            )
                            save_artifact(
                                out_dir,
                                "success_spec.generated.json",
                                auto_success_spec_to_json(auto_success_spec),
                            )
                            save_artifact(
                                out_dir,
                                "success_spec.validation.json",
                                auto_success_validation_to_json(auto_success_validation),
                            )
                            save_artifact(
                                out_dir,
                                "success_spec.effective.json",
                                json.dumps(
                                    {
                                        "user_success_spec_present": success_spec is not None,
                                        "user_success_spec": (
                                            success_spec.model_dump(mode="json")
                                            if success_spec is not None else None
                                        ),
                                        "generated": auto_success_spec.model_dump(mode="json"),
                                    },
                                    indent=2,
                                ),
                            )
                            save_artifact(
                                out_dir,
                                "visual_contracts.json",
                                json.dumps({
                                    shot_id: contract.model_dump()
                                    for shot_id, contract in visual_contracts.items()
                                }, indent=2),
                            )
                            compiled_scene = compile_storyboard_to_bpy(
                                sb,
                                render_engine=render_engine,
                                cycles_device=cycles_device,
                                visual_contracts=visual_contracts,
                            )
                            save_artifact(out_dir, "scene.compiled.py", compiled_scene)
                            shot_contract_text = _join_addenda(
                                style_profile_text,
                                format_success_spec_for_coder(success_spec),
                                format_auto_success_spec_for_coder(auto_success_spec),
                                format_scene_ir_for_coder(scene_ir, scene_ir_report),
                                _shot_visual_contracts(
                                    narrative, sb, scene_profile=scene_profile,
                                ),
                                _format_vector_ray_minimum_scaffold(visual_contracts),
                                format_failure_memory_for_coder(prior_memory),
                            )
                            verifier_report = verify_scene_code(
                                code, sb, visual_contracts=visual_contracts,
                                scene_profile=scene_profile,
                                render_engine=render_engine,
                                cycles_device=cycles_device,
                            )
                            contract_report = validate_visual_contracts(
                                code, sb, visual_contracts,
                                scene_profile=scene_profile,
                                success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
                            )
                            save_artifact(
                                out_dir,
                                f"contract_validation.iter{it:02d}.after_storyboard_repair.json",
                                contract_report_to_json(contract_report),
                            )
                # Merge block violations into the verifier flow so the
                # existing repair loop picks them up. We synthesise a
                # SceneVerificationIssue per contract block.
                _merge_contract_blocks_into_verifier(
                    verifier_report, contract_report,
                )
            if verifier_report.block_count:
                log.info(ui.warn(
                    "verifier repair: regenerating before Blender "
                    f"(max {max_verifier_repair_iters} attempt(s))"
                ))
                best_repair_key = _repair_quality_key(
                    verifier_report, contract_report,
                )
                repair_attempts = (
                    1 if scene_origin == "fresh"
                    else max_verifier_repair_iters
                )
                for attempt in range(1, repair_attempts + 1):
                    attempt_tag = (
                        "" if attempt == 1 else f".attempt{attempt:02d}"
                    )
                    verifier_plan = build_repair_plan(
                        iteration=it,
                        history=critic_history,
                        verifier_report=verifier_report,
                        contract_report=contract_report,
                        missing_objects=verifier_report.missing_objects,
                        visual_contracts=visual_contracts,
                        scene_profile=scene_profile,
                        best_selection_mode=best_selection_mode,
                    )
                    save_artifact(
                        out_dir,
                        f"repair_plan.iter{it:02d}.verifier{attempt_tag}.json",
                        repair_plan_to_json(verifier_plan),
                    )
                    repair_addendum = _join_addenda(
                        coder_kwargs.get("addendum", ""),
                        format_verifier_addendum(verifier_report),
                        format_contract_validation_addendum(contract_report),
                        format_repair_plan_for_coder(verifier_plan),
                    )
                    repair_kwargs = dict(coder_kwargs)
                    repair_kwargs["base_scene"] = code
                    repair_kwargs["addendum"] = repair_addendum
                    save_artifact(
                        out_dir,
                        f"scene_verifier.iter{it:02d}.addendum{attempt_tag}.txt",
                        repair_addendum,
                    )
                    try:
                        candidate_code = blender_coder.to_bpy_script(
                            sb, **repair_kwargs,
                        )
                    except Exception as e:  # noqa: BLE001
                        log.info(ui.warn(
                            f"verifier repair attempt {attempt} failed "
                            f"({type(e).__name__}): {str(e)[:200]}"
                        ))
                        continue
                    raw_verifier_report = verify_scene_code(
                        candidate_code, sb, visual_contracts=visual_contracts,
                        scene_profile=scene_profile,
                        render_engine=render_engine,
                        cycles_device=cycles_device,
                    )
                    repair_suffix = (
                        ".repair" if attempt == 1
                        else f".repair.attempt{attempt:02d}"
                    )
                    verifier_name = _save_scene_verification(
                        out_dir, it, raw_verifier_report, suffix=repair_suffix,
                    )
                    candidate_contract_report = validate_visual_contracts(
                        candidate_code, sb, visual_contracts,
                        scene_profile=scene_profile,
                        success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
                    )
                    if candidate_contract_report.violations:
                        contract_name = (
                            f"contract_validation.iter{it:02d}"
                            f"{repair_suffix}.json"
                        )
                        save_artifact(
                            out_dir,
                            contract_name,
                            contract_report_to_json(candidate_contract_report),
                        )
                    _merge_contract_blocks_into_verifier(
                        raw_verifier_report, candidate_contract_report,
                    )
                    regression_reason = _repair_regression_reason(
                        verifier_report,
                        contract_report,
                        raw_verifier_report,
                        candidate_contract_report,
                    )
                    candidate_key = _repair_quality_key(
                        raw_verifier_report,
                        candidate_contract_report,
                    )
                    accepted = (
                        regression_reason is None
                        and candidate_key < best_repair_key
                    )
                    if accepted:
                        code = candidate_code
                        verifier_report = raw_verifier_report
                        contract_report = candidate_contract_report
                        best_repair_key = candidate_key
                    else:
                        save_artifact(out_dir, "scene.py", code)
                    status = (
                        "accepted" if accepted
                        else "regressed" if regression_reason
                        else "rejected"
                    )
                    quality_text = _repair_quality_text(
                        raw_verifier_report,
                        candidate_contract_report,
                    )
                    if regression_reason:
                        quality_text += f"; regression={regression_reason}"
                    log.info(ui.detail(
                        f"verifier repair {attempt}/{repair_attempts}",
                        (
                            f"{quality_text}; "
                            f"{status}; "
                            f"see {verifier_name}"
                        ),
                    ))
                    if accepted and not verifier_report.block_count:
                        break
                if verifier_report.block_count:
                    if scene_origin == "fresh" and critic_history:
                        log.info(ui.warn(
                            "fresh branch verifier remained blocked; "
                            "discarding fresh candidate and falling back to "
                            "best prior critic result"
                        ))
                        break
                    if _can_try_render_tail_repair(verifier_report, code):
                        tail_repair = _try_render_tail_repair(
                            out_dir=out_dir,
                            iteration=it,
                            code=code,
                            storyboard=sb,
                            visual_contracts=visual_contracts,
                            scene_profile=scene_profile,
                            render_engine=render_engine,
                            cycles_device=cycles_device,
                            success_spec=success_spec,
                        )
                        if tail_repair is not None:
                            (
                                code,
                                verifier_report,
                                contract_report,
                                verifier_name,
                            ) = tail_repair
                            log.info(ui.warn(
                                "deterministic render-tail repair fixed "
                                "missing render contract"
                            ))
                    if _has_fatal_verifier_block(verifier_report):
                        fallback = _try_compiled_scene_fallback(
                            out_dir=out_dir,
                            iteration=it,
                            compiled_scene=compiled_scene,
                            storyboard=sb,
                            visual_contracts=visual_contracts,
                            scene_profile=scene_profile,
                            render_engine=render_engine,
                            cycles_device=cycles_device,
                            reason="fatal_verifier_repair_exhausted",
                            success_spec=success_spec,
                        )
                    else:
                        fallback = None
                        log.info(ui.warn(
                            "verifier still has non-fatal block issue(s); "
                            "keeping LLM candidate for render+critic"
                        ))
                    if fallback is not None:
                        (
                            code,
                            verifier_report,
                            contract_report,
                            verifier_name,
                        ) = fallback
                        scene_origin = "compiled_fallback"
                        log.info(ui.warn(
                            "compiled fallback: fatal verifier repair remained "
                            "blocked; using deterministic scaffold"
                        ))
                    elif _has_fatal_verifier_block(verifier_report) and it > 0 and critic_history:
                        log.info(ui.warn(
                            "verifier still has block issue(s) "
                            f"after {max_verifier_repair_iters} attempt(s); "
                            "falling back to best prior critic result"
                        ))
                        break
                    elif _has_fatal_verifier_block(verifier_report):
                        log.info(ui.fail(
                            "scene verifier still has block issue(s) after "
                            f"{max_verifier_repair_iters} attempt(s); "
                            f"see {verifier_name}"
                        ))
                        return _early_exit(out_dir, scene_path, frames_dir, t0)

            # Final contract validation pass — the verifier-repair loop
            # may have regenerated `code`, so the early contract_report
            # could be stale. Overwrite the saved artifact with the
            # post-repair state, which is what we are about to render,
            # and capture an anchor-OK signal for best-iteration scoring.
            contract_report = validate_visual_contracts(
                code, sb, visual_contracts, scene_profile=scene_profile,
                success_spec_text_anchors=_success_spec_text_anchor_names(success_spec),
            )
            save_artifact(
                out_dir,
                f"contract_validation.iter{it:02d}.json",
                contract_report_to_json(contract_report),
            )
            contract_anchors_ok = not any(
                v.severity == "block"
                and v.rule_id == "contract_missing_scene_anchors"
                for v in contract_report.violations
            )
            last_contract_report = contract_report

            # P0-B: if the coder returned a scene byte-identical to the
            # previous iteration's, the addendum is not changing behaviour
            # — skip the render + critic call rather than burning Blender
            # time and another vision request for an identical result.
            code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if it > 0 and prev_iter_code_hash == code_hash:
                log.info(ui.warn(
                    "coder returned scene identical to prior iter; "
                    "skipping render+critic"
                ))
                continue
            prev_iter_code_hash = code_hash

            # P0-A: deterministic pre-render check that the coder
            # mentioned each storyboard object name verbatim. Cheap
            # (substring scan, 0 tokens) and feeds the next addendum.
            current_missing_objects = dict(verifier_report.missing_objects)
            if current_missing_objects:
                n_missing = sum(
                    len(v) for v in current_missing_objects.values()
                )
                log.info(ui.detail(
                    "object-name check",
                    (
                        f"{n_missing} expected name(s) missing across "
                        f"{len(current_missing_objects)} shot(s)"
                    ),
                ))
                save_artifact(
                    out_dir, f"missing_objects.iter{it:02d}.json",
                    json.dumps(current_missing_objects, indent=2),
                )
            _save_resume_checkpoint(
                out_dir,
                ResumeCheckpoint(
                    iteration=it,
                    stage="scene_ready",
                    scene_origin=scene_origin,
                    render_ok=True,
                    n_frames=0,
                ),
            )

        # 4. Render (keep_frames only honoured on it=0)
        if it == 0 and keep_frames and frames_dir.exists():
            n_frames = len(list(frames_dir.glob("frame_*.png")))
            log.info(ui.step(4, 7, "Blender render", "kept"))
            log.info(ui.detail("frames", f"{n_frames} pre-existing"))
            if n_frames == 0:
                log.info(ui.fail("keep_frames set but frames/ empty"))
                return _early_exit(out_dir, scene_path, frames_dir, t0)
            rr_ok = True
        else:
            if preview_render:
                preview_dir = out_dir / f"preview_frames.iter{it:02d}"
                preview_frames = select_preview_frames(sb)
                preview_report_path = out_dir / f"preview_report.iter{it:02d}.json"
                cached_preview_report = None
                if resume_stage_for_iter in {"preview_done", "render_done"}:
                    cached_preview_report = _load_preview_report(preview_report_path)
                scene_text = scene_path.read_text() if scene_path.exists() else ""
                if cached_preview_report is not None:
                    report = cached_preview_report
                    log.info(ui.step(
                        4, 7, "Keyframe preview",
                        f"{len(preview_frames)} frame(s), cached",
                    ))
                    if report.issues:
                        log.info(ui.detail(
                            "preview",
                            (
                                f"{report.block_count}b/"
                                f"{report.warn_count}w; see "
                                f"preview_report.iter{it:02d}.json"
                            ),
                        ))
                elif "CG_TUTOR_PREVIEW_FRAMES" not in scene_text:
                    report = PreviewReport(
                        ok=True,
                        skipped_reason=(
                            "scene.py does not advertise CG_TUTOR_PREVIEW_FRAMES support"
                        ),
                    )
                    save_artifact(
                        out_dir,
                        f"preview_report.iter{it:02d}.json",
                        preview_report_to_json(report),
                    )
                    log.info(ui.detail(
                        "preview",
                        "skipped: scene.py lacks preview-frame support",
                    ))
                else:
                    log.info(ui.step(
                        4, 7, "Keyframe preview",
                        f"{len(preview_frames)} frame(s)",
                    ))
                    report, _ = _run_keyframe_preview(
                        scene_path=scene_path,
                        preview_dir=preview_dir,
                        preview_frames=preview_frames,
                        scene_profile=scene_profile,
                        storyboard=sb,
                        blender_runtime_mod=blender_runtime,
                        blender_timeout_sec=blender_timeout_sec,
                        out_dir=out_dir,
                        iteration=it,
                    )
                    if report.issues:
                        log.info(ui.detail(
                            "preview",
                            (
                                f"{report.block_count}b/"
                                f"{report.warn_count}w; see "
                                f"preview_report.iter{it:02d}.json"
                            ),
                        ))
                    if not report.ok:
                        if preview_blocks_allow_render_repair(report):
                            log.info(ui.warn(
                                "preview reports weak visible motion; "
                                "continuing to render+critic so the retry "
                                "loop can repair animation amplitude"
                            ))
                        elif it > 0 and critic_history:
                            log.info(ui.warn(
                                "preview has block issue(s); falling back "
                                "to best prior critic result"
                            ))
                            break
                        else:
                            # iter00 has no prior critic result to fall back to.
                            # Before giving up, try the deterministic compiled
                            # scaffold — same safety net that fatal verifier
                            # blocks already use. If the LLM scene crashed
                            # Blender (runtime bug invisible to the static
                            # verifier), the scaffold often still renders.
                            fallback_ok = False
                            if not _is_compiled_origin(scene_origin):
                                log.info(ui.warn(
                                    "preview crash on coder candidate; "
                                    "trying compiled scaffold fallback"
                                ))
                                fb = _try_compiled_scene_fallback(
                                    out_dir=out_dir,
                                    iteration=it,
                                    compiled_scene=compiled_scene,
                                    storyboard=sb,
                                    visual_contracts=visual_contracts,
                                    scene_profile=scene_profile,
                                    render_engine=render_engine,
                                    cycles_device=cycles_device,
                                    reason="preview_crash",
                                    success_spec=success_spec,
                                )
                                if fb is not None:
                                    (
                                        code,
                                        verifier_report,
                                        contract_report,
                                        verifier_name,
                                    ) = fb
                                    report, _ = _run_keyframe_preview(
                                        scene_path=scene_path,
                                        preview_dir=preview_dir,
                                        preview_frames=preview_frames,
                                        scene_profile=scene_profile,
                                        storyboard=sb,
                                        blender_runtime_mod=blender_runtime,
                                        blender_timeout_sec=blender_timeout_sec,
                                        out_dir=out_dir,
                                        iteration=it,
                                        artifact_suffix=".compiled_fallback",
                                    )
                                    if report.ok:
                                        scene_origin = "compiled_fallback"
                                        current_missing_objects = dict(
                                            verifier_report.missing_objects
                                        )
                                        contract_anchors_ok = not any(
                                            v.severity == "block"
                                            and v.rule_id
                                            == "contract_missing_scene_anchors"
                                            for v in contract_report.violations
                                        )
                                        log.info(ui.ok(
                                            "compiled scaffold passed preview; "
                                            "proceeding to render+critic with "
                                            "scene_origin=compiled_fallback"
                                        ))
                                        fallback_ok = True
                                    else:
                                        log.info(ui.detail(
                                            "compiled preview",
                                            (
                                                f"{report.block_count}b/"
                                                f"{report.warn_count}w; "
                                                "see preview_report.iter"
                                                f"{it:02d}.compiled_fallback.json"
                                            ),
                                        ))
                            if not fallback_ok:
                                log.info(ui.fail(
                                    "preview has block issue(s); "
                                    f"see preview_report.iter{it:02d}.json"
                                ))
                                return _early_exit(
                                    out_dir, scene_path, frames_dir, t0,
                                )
                _save_resume_checkpoint(
                    out_dir,
                    ResumeCheckpoint(
                        iteration=it,
                        stage="preview_done",
                        scene_origin=scene_origin,
                        render_ok=True,
                        n_frames=0,
                    ),
                )
            log.info(ui.step(4, 7, "Blender render"))
            reuse_render = (
                resume_stage_for_iter == "render_done"
                and any(frames_dir.glob("frame_*.png"))
            )
            if reuse_render:
                n_frames = len(list(frames_dir.glob("frame_*.png")))
                rr_ok = bool(
                    resume_checkpoint.render_ok if resume_checkpoint is not None
                    else True
                )
                log.info(ui.detail(
                    "render",
                    f"reusing cached frames for iter{it:02d} ({n_frames} frame(s))",
                ))
            else:
                if frames_dir.exists():
                    for f in frames_dir.glob("frame_*.png"):
                        f.unlink()
                # CG_TUTOR_OUT_DIR is set by blender_runtime.run_script from
                # the out_dir argument; do NOT mutate the parent process
                # env here — it would leak across concurrent pipeline calls.
                rr = blender_runtime.run_script(scene_path, frames_dir,
                                                timeout_sec=blender_timeout_sec)
                n_frames = len(list(frames_dir.glob("frame_*.png")))
                stderr_name = f"blender_stderr.iter{it:02d}.txt"
                stdout_name = f"blender_stdout.iter{it:02d}.txt"
                save_artifact(out_dir, stderr_name, rr.stderr or "")
                save_artifact(out_dir, stdout_name, rr.stdout or "")
                if n_frames == 0:
                    if it > 0 and critic_history:
                        log.info(ui.warn(
                            f"retry render failed: blender ok={rr.ok}, "
                            f"frames=0; see {stderr_name}"
                        ))
                        log.info(ui.detail(
                            "fallback",
                            "best prior critic result",
                        ))
                        break
                    log.info(ui.fail(
                        f"blender ok={rr.ok}, frames=0; see {stderr_name}"
                    ))
                    return _early_exit(out_dir, scene_path, frames_dir, t0)
                rr_ok = rr.ok
                if not rr_ok:
                    log.info(ui.warn(
                        f"partial render: returncode={rr.returncode}, "
                        f"frames={n_frames}; composing what we have"
                    ))
                else:
                    log.info(ui.ok(f"rendered {n_frames} frames"))
                _save_resume_checkpoint(
                    out_dir,
                    ResumeCheckpoint(
                        iteration=it,
                        stage="render_done",
                        scene_origin=scene_origin,
                        render_ok=rr_ok,
                        n_frames=n_frames,
                    ),
                )
        scene_iter_path = out_dir / f"scene.iter{it:02d}.py"
        if scene_path.exists():
            scene_iter_path.write_text(scene_path.read_text())
            scene_state_report = inspect_scene_code(scene_iter_path.read_text())
            save_artifact(
                out_dir,
                f"scene_state.iter{it:02d}.json",
                scene_state_report_to_json(scene_state_report),
            )

        # 6. Critic. With critic enabled, score every rendered iteration
        # so we can select the best scene even if the final retry regresses.
        # Skip when keep_frames is set on it=0 since the frames don't
        # correspond to the current scene.py.
        if max_critic_iterations == 0 or (it == 0 and keep_frames):
            _clear_resume_checkpoint(out_dir)
            log.info(ui.step(6, 7, "Critic", "skipped"))
            break

        log.info(ui.step(6, 7, "Critic"))
        if critic_ensemble:
            critic_report = render_critic.inspect_ensemble(
                sb, frames_dir, iteration=it,
                out_dir=out_dir, backends=critic_ensemble,
                narrative=narrative,
                scene_profile=scene_profile,
                strictness=critic_strictness,
            )
        else:
            critic_report = render_critic.inspect(
                sb, frames_dir, iteration=it,
                out_dir=out_dir, backend=critic_backend,
                narrative=narrative,
                scene_profile=scene_profile,
            )
        frames_snapshot_dir = _frame_snapshot_dir(out_dir, it)
        copied_frames = _copy_frame_snapshot(frames_dir, frames_snapshot_dir)
        if copied_frames != n_frames:
            log.info(ui.warn(
                f"frames snapshot copied {copied_frames}/{n_frames} frames "
                f"to {frames_snapshot_dir.name}"
            ))
        iter_hash = frames_hash_cache.get(frames_snapshot_dir)
        critic_history.append(CriticIteration(
            iteration=it,
            report=critic_report,
            scene_path=scene_iter_path,
            render_ok=rr_ok,
            n_frames=n_frames,
            frames_hash=iter_hash,
            frames_dir=frames_snapshot_dir,
            scene_origin=scene_origin,
            missing_objects=dict(current_missing_objects),
            contract_anchors_ok=contract_anchors_ok,
            fallback_degraded=(
                scene_origin == "compiled_fallback"
                and contract_report.block_count > 0
            ),
        ))
        _clear_resume_checkpoint(out_dir)
        try:
            scene_code_for_metrics = scene_iter_path.read_text()
        except FileNotFoundError:
            scene_code_for_metrics = ""
        metric_report = run_concept_metrics(
            concept_id=concept_id,
            scene_code=scene_code_for_metrics,
            storyboard=sb,
            scene_profile=scene_profile,
            success_spec=success_spec,
            auto_success_spec=auto_success_spec,
        )
        auto_anchor_status = (
            metric_report.metrics
            .get("auto_success_spec", {})
            .get("anchor_status", {})
        )
        _append_auto_success_issues(
            metric_report,
            critic_confirmed_auto_success_issues(
                auto_success_spec=auto_success_spec,
                critic_history=critic_history[:-1],
                current_report=critic_report,
                static_anchor_status=auto_anchor_status,
            ),
        )
        save_artifact(
            out_dir,
            f"concept_metric_report.iter{it:02d}.json",
            concept_metric_report_to_json(metric_report),
        )
        last_metric_addendum = format_concept_metric_report_for_coder(metric_report)
        metric_history.append(metric_report)
        critic_history[-1].metric_block_count = metric_report.block_count
        critic_history[-1].metric_warn_count = metric_report.warn_count
        critic_history[-1].metric_structural_fatal_count = (
            metric_report.structural_fatal_count
        )
        critic_history[-1].metric_success_hard_count = (
            metric_report.success_hard_count
        )
        critic_history[-1].metric_success_soft_count = (
            metric_report.success_soft_count
        )
        critic_history[-1].metric_aesthetic_warn_count = (
            metric_report.aesthetic_warn_count
        )
        if metric_report.issues:
            log.info(ui.detail(
                "concept metric",
                (
                    f"block={metric_report.block_count} "
                    f"warn={metric_report.warn_count}; "
                    f"success_hard={metric_report.success_hard_count} "
                    f"success_soft={metric_report.success_soft_count}"
                ),
            ))
        cross_ref_report = cross_reference_critic_findings(
            concept_id=concept_id,
            iteration=it,
            critic_issues=list(critic_report.issues),
            scene_code=scene_code_for_metrics,
            storyboard=sb,
            visual_contracts=visual_contracts,
        )
        save_artifact(
            out_dir,
            f"critic_cross_reference.iter{it:02d}.json",
            cross_reference_report_to_json(cross_ref_report),
        )
        last_cross_ref_addendum = format_cross_reference_for_coder(cross_ref_report)
        cross_ref_history.append(cross_ref_report)
        critic_history[-1].cross_ref_actionable_count = (
            cross_ref_report.actionable_count
        )
        if cross_ref_report.findings:
            log.info(ui.detail(
                "cross-ref",
                f"{len(cross_ref_report.findings)} actionable findings",
            ))
        iteration_pass = _iteration_pass_threshold(critic_report, metric_report)
        log.info(ui.detail(
            "critic result",
            f"{_critic_status_text(critic_report)}; pass={iteration_pass}",
        ))
        if not iteration_pass:
            reasons = ", ".join(
                _iteration_fail_reasons(critic_report, metric_report)
            )
            log.info(ui.warn(f"not passed: {reasons}"))
        if iteration_pass:
            break

        if it >= max_critic_iterations:
            log.info(ui.detail("critic loop", f"reached max iter {max_critic_iterations}"))
            break

        if early_stop_stale_iters > 0:
            floor, stale = _block_floor_stale_iters(critic_history)
            if stale >= early_stop_stale_iters:
                log.info(ui.warn(
                    f"early stop: best framing block floor is {floor}; "
                    f"no lower floor for {stale} iteration(s) "
                    f"(threshold={early_stop_stale_iters})"
                ))
                break

    # If critic retries regressed, roll back to the best evaluated scene and
    # re-render its frames before composing. This makes max_critic_iters>1
    # a best-of search instead of a risky last-writer-wins loop.
    variant_selections: dict[str, CriticIteration] = {}
    rendered_variant_frames: dict[int, Path] = {}
    final_status: str = "no_critic_history"
    if critic_history:
        selection_history = _strict_selection_history(critic_history)
        if len(selection_history) != len(critic_history):
            log.info(ui.warn(
                "strict final selection uses only iterations with "
                "frames.iterNN snapshots; older unsnapshotted critic records "
                "remain available for repair context"
            ))
        variant_selections = _select_video_variants(
            selection_history,
            primary_mode=best_selection_mode,
        )
        best = variant_selections["selected"]
        best_block, best_warn = _critic_counts(best.report)
        best_sem_block, best_sem_warn = _semantic_counts(best.report)
        selected_reason = _selected_reason(best, selection_history)
        final_status = (
            "pass" if selected_reason == "pass"
            else (
                "fallback_diagnostic_video"
                if selected_reason in {
                    "fallback_diagnostic",
                    "no_renderable_llm_candidate",
                }
                else "best_with_violations"
            )
        )
        compliance_best = variant_selections["compliance"]
        aesthetic_best = variant_selections["aesthetic"]
        semantic_best = variant_selections["semantic"]
        aesthetic_block, aesthetic_warn = _critic_counts(aesthetic_best.report)
        trade_off_note = ""
        if aesthetic_best.iteration != best.iteration:
            trade_off_note = (
                f"Selected best (iter{best.iteration:02d}) has "
                f"{aesthetic_block - best_block:+d} block vs aesthetic best "
                f"(iter{aesthetic_best.iteration:02d}), and "
                f"{best.report.overall_score - aesthetic_best.report.overall_score:+.2f} "
                "overall_score. final.mp4 is composed from selected best."
            )
        floor, stale = _block_floor_stale_iters(critic_history)
        memory_entries = memory_from_history(concept_id, critic_history)
        save_failure_memory_snapshot(
            out_dir / "failure_memory.current.json",
            memory_entries,
        )
        append_failure_memory(memory_path, memory_entries)
        log.info(ui.step(0, 7, "Best iteration"))
        log.info(ui.detail(
            "selected",
            (
                f"iter{best.iteration:02d} by {best_selection_mode}; "
                f"{_critic_status_text(best.report)}; "
                f"metric={best.metric_block_count}b/"
                f"{best.metric_warn_count}w; "
                f"{_failure_class_status_text(best)}; "
                f"final_status={final_status}; "
                f"selected_reason={selected_reason}; "
                f"fallback_degraded={best.fallback_degraded}; "
                f"pass={_critic_iteration_pass(best)}"
            ),
        ))
        if not _critic_iteration_pass(best):
            reasons = ", ".join(
                _iteration_fail_reasons(best.report)
                + (
                    ["metric structural fatal issue(s)"]
                    if best.metric_structural_fatal_count else []
                )
                + (
                    ["metric success-hard issue(s)"]
                    if best.metric_success_hard_count else []
                )
                + (
                    ["concept metric success-soft issue(s)"]
                    if (
                        best.metric_success_soft_count
                        and not best.metric_structural_fatal_count
                        and not best.metric_success_hard_count
                    )
                    else []
                )
            )
            log.info(ui.warn(f"best result is not pass: {reasons}"))
            log.info(ui.warn(
                f"final_status={final_status} (iter{best.iteration:02d}); "
                f"selected_reason={selected_reason}; delivered video is not "
                "a spec-passing one"
            ))
        else:
            log.info(ui.ok(
                f"final_status=pass (iter{best.iteration:02d})"
            ))
        if trade_off_note:
            log.info(ui.detail("trade-off", trade_off_note))
        # Replay the best scene first when current frames don't match it, so
        # critic_best.json can record the exact final-mp4 source state (the
        # replay frame hash and whether it differs from the critic-scored hash).
        current_frames_hash = frames_hash_cache.get(frames_dir)
        frames_match_best = bool(
            best.frames_hash and current_frames_hash == best.frames_hash
        )
        replay_ran = False
        replay_frames_hash = ""
        replay_hash_mismatch = False
        if (
            best.frames_dir is not None
            and best.frames_dir.exists()
            and not frames_match_best
        ):
            log.info(ui.step(
                4, 7, "Restore critic-scored frames",
                f"iter{best.iteration:02d}",
            ))
            if frames_dir.exists():
                for f in frames_dir.glob("frame_*.png"):
                    f.unlink()
            copied_frames = _copy_frame_snapshot(best.frames_dir, frames_dir)
            frames_hash_cache.invalidate(frames_dir)
            replay_frames_hash = frames_hash_cache.get(frames_dir)
            replay_hash_mismatch = bool(
                best.frames_hash and replay_frames_hash != best.frames_hash
            )
            if copied_frames == 0:
                log.info(ui.fail(
                    f"{best.frames_dir.name} has no frames to restore"
                ))
                return _early_exit(out_dir, scene_path, frames_dir, t0)
            if replay_hash_mismatch:
                log.info(ui.fail(
                    "critic-scored frame snapshot hash mismatch after copy "
                    f"(original={best.frames_hash[:8]}, "
                    f"restored={replay_frames_hash[:8]})"
                ))
                return _early_exit(out_dir, scene_path, frames_dir, t0)
            _write_compat_scene(scene_path, best.scene_path.read_text())
            log.info(ui.ok(
                f"restored {copied_frames} critic-scored frames from "
                f"{best.frames_dir.name}"
            ))
        elif best.scene_path.exists() and not frames_match_best:
            replay_ran = True
            log.info(ui.step(
                4, 7, "Best-scene replay",
                f"iter{best.iteration:02d}",
            ))
            _write_compat_scene(scene_path, best.scene_path.read_text())
            if frames_dir.exists():
                for f in frames_dir.glob("frame_*.png"):
                    f.unlink()
            frames_hash_cache.invalidate(frames_dir)
            # See note above: runtime sets CG_TUTOR_OUT_DIR per-call.
            rr = blender_runtime.run_script(scene_path, frames_dir,
                                            timeout_sec=blender_timeout_sec)
            n_frames = len(list(frames_dir.glob("frame_*.png")))
            save_artifact(out_dir, "blender_stderr.bestreplay.txt", rr.stderr or "")
            save_artifact(out_dir, "blender_stdout.bestreplay.txt", rr.stdout or "")
            if n_frames == 0:
                log.info(ui.fail(
                    "best-scene render "
                    f"ok={rr.ok}, frames=0; see blender_stderr.bestreplay.txt"
                ))
                return _early_exit(out_dir, scene_path, frames_dir, t0)
            # Sanity check: if the replay frames don't match the originally
            # critic-scored frames, Blender is non-deterministic on this
            # scene and the final mp4 may differ from what the critic saw.
            replay_frames_hash = frames_hash_cache.get(frames_dir)
            if best.frames_hash and replay_frames_hash != best.frames_hash:
                replay_hash_mismatch = True
                log.info(ui.warn(
                    "best-replay frame hash mismatch "
                    f"(original={best.frames_hash[:8]}, "
                    f"replay={replay_frames_hash[:8]}); "
                    "Blender non-determinism; final.mp4 may differ "
                    "from the critic-scored best"
                ))
            log.info(ui.ok(f"restored render frames={n_frames} ok={rr.ok}"))
        rendered_variant_frames[best.iteration] = frames_dir
        save_artifact(out_dir, "critic_best.json", json.dumps({
            "best_iteration": best.iteration,
            "best_selection_mode": best_selection_mode,
            "final_status": final_status,
            "selected_reason": selected_reason,
            "pass": _critic_iteration_pass(best),
            "overall_score": best.report.overall_score,
            "block": best_block,
            "warn": best_warn,
            "concept_mismatch_block": best_sem_block,
            "concept_mismatch_warn": best_sem_warn,
            "structural_fatal": _failure_class_counts(best)[0],
            "success_hard": _failure_class_counts(best)[1],
            "success_soft": _failure_class_counts(best)[2],
            "aesthetic_warn": _failure_class_counts(best)[3],
            "render_ok": best.render_ok,
            "scene_origin": best.scene_origin,
            "fallback_degraded": best.fallback_degraded,
            "n_frames": best.n_frames,
            "frames_hash": best.frames_hash,
            "balanced_best": _critic_iteration_summary(
                variant_selections["balanced"]
            ),
            "compliance_best": _critic_iteration_summary(compliance_best),
            "aesthetic_best": _critic_iteration_summary(aesthetic_best),
            "semantic_best": _critic_iteration_summary(semantic_best),
            "differs": aesthetic_best.iteration != best.iteration,
            "trade_off_note": trade_off_note,
            "block_floor": floor,
            "iters_since_floor_lowered": stale,
            "failure_memory_entries": len(memory_entries),
            "replay_ran": replay_ran,
            "replay_frames_hash": replay_frames_hash,
            "replay_hash_mismatch": replay_hash_mismatch,
            "all_iterations": [
                _critic_iteration_summary(r) for r in selection_history
            ],
            "all_history_iterations": [
                _critic_iteration_summary(r) for r in critic_history
            ],
        }, indent=2))
        if strict_best_replay and replay_hash_mismatch:
            log.info(ui.fail(
                "--strict-best-replay set and best-replay frames diverged "
                "from the critic-scored hash; final.mp4 would not match "
                "the score. See critic_best.json "
                "(replay_hash_mismatch=true) and blender_stderr.bestreplay.txt."
            ))
            return _early_exit(out_dir, scene_path, frames_dir, t0)

    # 5 + 7. Overlays and mp4 compose (the helper does both).
    log.info(ui.step(5, 7, "Overlay + ffmpeg compose"))
    final_mp4 = out_dir / "final.mp4"
    video_exports: list[dict] = []

    def _compose_variant(
        role: str,
        item: CriticIteration | None,
        out_name: str,
        *,
        primary: bool = False,
    ) -> None:
        nonlocal final_mp4
        source_frames = frames_dir
        render_ok = True
        n_variant_frames = len(list(source_frames.glob("frame_*.png")))
        variant_hash = frames_hash_cache.get(source_frames)
        if item is not None:
            cached = rendered_variant_frames.get(item.iteration)
            if cached is not None:
                source_frames = cached
                n_variant_frames = len(list(source_frames.glob("frame_*.png")))
                variant_hash = frames_hash_cache.get(source_frames)
            elif item.frames_dir is not None and item.frames_dir.exists():
                source_frames = item.frames_dir
                n_variant_frames = len(list(source_frames.glob("frame_*.png")))
                variant_hash = frames_hash_cache.get(source_frames)
            elif item.scene_path.exists():
                source_frames = out_dir / f"frames_{role}"
                source_frames.mkdir(parents=True, exist_ok=True)
                for f in source_frames.glob("frame_*.png"):
                    f.unlink()
                frames_hash_cache.invalidate(source_frames)
                log.info(ui.detail(
                    f"export {role}",
                    f"render iter{item.iteration:02d}",
                ))
                # See note above: runtime sets CG_TUTOR_OUT_DIR per-call.
                rr_variant = blender_runtime.run_script(
                    item.scene_path,
                    source_frames,
                    timeout_sec=blender_timeout_sec,
                )
                save_artifact(
                    out_dir,
                    f"blender_stderr.export_{role}.txt",
                    rr_variant.stderr or "",
                )
                save_artifact(
                    out_dir,
                    f"blender_stdout.export_{role}.txt",
                    rr_variant.stdout or "",
                )
                render_ok = bool(rr_variant.ok)
                n_variant_frames = len(list(source_frames.glob("frame_*.png")))
                variant_hash = frames_hash_cache.get(source_frames)
                if n_variant_frames == 0:
                    video_exports.append({
                        "role": role,
                        "output": out_name,
                        "iteration": item.iteration,
                        "ok": False,
                        "stage": "render",
                        "reason": "no frames rendered",
                    })
                    if primary:
                        final_mp4 = None
                    return
                rendered_variant_frames[item.iteration] = source_frames
            else:
                video_exports.append({
                    "role": role,
                    "output": out_name,
                    "iteration": item.iteration,
                    "ok": False,
                    "stage": "render",
                    "reason": f"{item.scene_path.name} missing",
                })
                if primary:
                    final_mp4 = None
                return

        res = compose_storyboard_video(
            sb, source_frames, out_dir, out_name=out_name,
        )
        output_path = out_dir / out_name
        if not res.ok:
            stderr_name = (
                "ffmpeg_stderr.txt" if primary
                else f"ffmpeg_stderr.{role}.txt"
            )
            save_artifact(out_dir, stderr_name, res.stderr or "")
            log.info(ui.fail(f"ffmpeg for {role}; see {stderr_name}"))
            if primary:
                final_mp4 = None
        video_exports.append({
            "role": role,
            "output": out_name,
            "iteration": item.iteration if item is not None else None,
            "ok": bool(res.ok and output_path.exists()),
            "stage": "compose",
            "render_ok": render_ok,
            "n_frames": n_variant_frames,
            "frames_hash": variant_hash,
            "summary": _critic_iteration_summary(item) if item is not None else None,
        })

    primary_item = variant_selections.get("selected")
    _compose_variant("selected", primary_item, "final.mp4", primary=True)
    if variant_selections:
        for role in ("balanced", "compliance", "aesthetic", "semantic"):
            _compose_variant(
                role,
                variant_selections[role],
                f"final_{role}.mp4",
            )
        save_artifact(
            out_dir,
            "video_exports.json",
            json.dumps(video_exports, indent=2),
        )
        critic_best_path = out_dir / "critic_best.json"
        if critic_best_path.exists():
            try:
                data = json.loads(critic_best_path.read_text())
                data["video_exports"] = video_exports
                save_artifact(out_dir, "critic_best.json", json.dumps(data, indent=2))
            except Exception as e:  # noqa: BLE001
                log.warning(ui.warn(
                    f"could not append video exports to critic_best.json: {e!r}"
                ))
        ok_exports = [e for e in video_exports if e.get("ok")]
        if len(ok_exports) > 1:
            log.info(ui.step(7, 7, "Exported alternatives"))
            for e in ok_exports:
                if e["role"] == "selected":
                    continue
                s = e.get("summary") or {}
                log.info(ui.detail(
                    e["output"],
                    (
                        f"iter{s.get('iteration', -1):02d} "
                        f"score={s.get('overall_score', 0):.2f}; "
                        f"framing={s.get('block', 0)}b/"
                        f"{s.get('warn', 0)}w; "
                        f"semantic={s.get('concept_mismatch_block', 0)}b/"
                        f"{s.get('concept_mismatch_warn', 0)}w"
                    ),
                ))

    elapsed = time.time() - t0
    if final_mp4 and final_mp4.exists():
        size_kb = final_mp4.stat().st_size // 1024
        log.info(ui.done(
            f"in {elapsed:.1f}s - {final_mp4} ({size_kb} KB) "
            f"[final_status={final_status}]"
        ))

    return PipelineResult(
        concept_id=concept_id, out_dir=out_dir, final_mp4=final_mp4,
        narrative_path=out_dir / "narrative.json",
        storyboard_path=out_dir / "storyboard.json",
        scene_path=scene_path, frames_dir=frames_dir,
        elapsed_sec=elapsed,
    )
