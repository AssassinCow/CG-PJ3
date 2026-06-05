"""End-to-end smoke tests for `pipeline.run()`'s critic loop.

We stub every external dependency — concept_decomposer, storyboard agent,
blender_coder, render_critic, Blender runtime, ffmpeg compose — so the
test exercises only the orchestration logic in pipeline.run().

These tests are the ones that would have caught the v5/phong_lighting
incident (coder returned scene byte-identical to iter00, pipeline still
ran render+critic for a second iter and wasted time/tokens).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from cg_tutor import pipeline
from cg_tutor.agents import blender_coder, concept_decomposer, render_critic
from cg_tutor.agents import profile_generator
from cg_tutor.agents import storyboard as storyboard_agent
from cg_tutor.blender import runtime as blender_runtime
from cg_tutor.composer import compose as compose_mod
from cg_tutor.concept_metrics import ConceptMetricIssue, ConceptMetricReport
from cg_tutor.schemas import (
    CriticIssue, CriticReport, Narrative, NarrativeNode, Storyboard,
)
from cg_tutor.scene_profiles import SceneProfileResolution, base_profile
from cg_tutor.scene_verifier import SceneVerificationIssue, SceneVerificationReport
from cg_tutor.contract_validator import ContractValidationReport, ContractViolation
from cg_tutor.visual_contract import VisualContract


@pytest.fixture
def fake_concept_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "fake.yaml"
    p.write_text(yaml.safe_dump({
        "concept_id": "fake",
        "title": "Fake Concept",
        "duration_sec": 2.0,
        "scene_profile": "vector_teaching",
    }))
    return p


@pytest.fixture
def fake_narrative() -> Narrative:
    return Narrative(
        concept_id="fake",
        nodes=[
            NarrativeNode(
                id="node_01", title="One",
                description="Show one thing.",
                formulas=[], duration_sec=2.0,
                visual_intent="A red sphere on a black floor.",
            ),
        ],
    )


@pytest.fixture
def fake_storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "fake",
        "fps": 24,
        "resolution": [320, 240],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 2.0,
            "camera": [{
                "time_sec": 0.0, "position": [0, -5, 3],
                "look_at": [0, 0, 0], "fov": 50,
            }],
            "objects": [{
                "name": "red_sphere", "type": "mesh",
                "primitive": "sphere", "location": [0, 0, 0],
            }],
        }],
    })


def _install_stubs(
    monkeypatch,
    narrative: Narrative,
    storyboard: Storyboard,
    scripted_code: list[str | BaseException],
    scripted_reports: list[CriticReport],
):
    """Wire fake versions of every agent the pipeline calls."""
    monkeypatch.setattr(
        concept_decomposer, "decompose",
        lambda spec, **kw: (
            _save_narrative(narrative, kw.get("out_dir")) or narrative
        ),
    )
    monkeypatch.setattr(
        profile_generator,
        "resolve_scene_profile",
        lambda spec, narrative, **kw: SceneProfileResolution(
            profile=base_profile(spec.get("scene_profile") or "vector_teaching"),
            source="test_stub",
            raw=None,
            validation={"source": "test_stub"},
        ),
    )
    monkeypatch.setattr(
        storyboard_agent, "to_storyboard",
        lambda n, **kw: (
            _save_storyboard(storyboard, kw.get("out_dir")) or storyboard
        ),
    )

    code_iter = iter(scripted_code)
    coder_call_count = {"n": 0}

    def fake_coder(sb, *, out_dir=None, iteration=0, **kw):
        coder_call_count["n"] += 1
        try:
            code = next(code_iter)
        except StopIteration:
            code = scripted_code[-1]
        if isinstance(code, BaseException):
            raise code
        if out_dir is not None:
            suffix = "" if iteration == 0 else f".iter{iteration:02d}"
            (out_dir / f"scene{suffix}.py").write_text(code)
            if iteration > 0:
                (out_dir / "scene.py").write_text(code)
        return code

    monkeypatch.setattr(blender_coder, "to_bpy_script", fake_coder)

    def fake_repair(sb, *, base_scene, out_dir=None, iteration=0,
                    addendum="", model=None):
        return fake_coder(sb, out_dir=out_dir, iteration=iteration)

    monkeypatch.setattr(
        blender_coder, "repair_bpy_script_diff", fake_repair
    )

    report_iter = iter(scripted_reports)

    def fake_critic(sb, frames_dir, *, iteration=0, out_dir=None, backend=None,
                    narrative=None, scene_profile=None):
        try:
            r = next(report_iter)
        except StopIteration:
            r = scripted_reports[-1]
        r2 = r.model_copy(update={"iteration": iteration})
        if out_dir is not None:
            (out_dir / f"critic_iter{iteration:02d}.json").write_text(
                r2.model_dump_json(indent=2)
            )
        return r2

    monkeypatch.setattr(render_critic, "inspect", fake_critic)

    def fake_runtime(scene_path, frames_dir, **kw):
        frames_dir.mkdir(parents=True, exist_ok=True)
        # Write fake PNGs (content varies with scene.py content so the
        # frame hash changes when the coder changes).
        preview_raw = (kw.get("env_overrides") or {}).get("CG_TUTOR_PREVIEW_FRAMES")
        frame_ids = [
            int(x) for x in preview_raw.split(",") if x.strip()
        ] if preview_raw else list(range(1, 5))
        seed = sum(scene_path.read_bytes()) % 255
        for i in frame_ids:
            Image.new(
                "RGB",
                (8, 8),
                ((seed + i) % 255, (seed + 40 + i) % 255, (seed + 80 + i) % 255),
            ).save(frames_dir / f"frame_{i:04d}.png")

        class _RR:
            ok = True
            returncode = 0
            stdout = ""
            stderr = ""
        return _RR()

    monkeypatch.setattr(blender_runtime, "run_script", fake_runtime)

    def fake_compose(sb, frames_dir, out_dir, *, out_name="final.mp4"):
        mp4 = out_dir / out_name
        mp4.write_bytes(b"FAKEMP4")

        class _R:
            ok = True
            stderr = ""
        return _R()

    monkeypatch.setattr(compose_mod, "compose_storyboard_video", fake_compose)
    monkeypatch.setattr(
        pipeline, "compose_storyboard_video", fake_compose
    )

    return coder_call_count


def _save_narrative(n: Narrative, out_dir: Path | None):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "narrative.json").write_text(n.model_dump_json(indent=2))


def _save_storyboard(s: Storyboard, out_dir: Path | None):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "storyboard.json").write_text(s.model_dump_json(indent=2))


_OK_REPORT = CriticReport(
    concept_id="fake", iteration=0, overall_score=0.9, issues=[],
)


def _valid_scene(marker: str, *, include_obj: bool = True) -> str:
    obj_line = "obj.name = 'red_sphere'\n" if include_obj else ""
    return (
        "import os\n"
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "scene = bpy.context.scene\n"
        "scene.render.engine = 'BLENDER_EEVEE'\n"
        "scene.frame_start = 1\n"
        "scene.frame_end = 4\n"
        "out_dir = os.environ['CG_TUTOR_OUT_DIR']\n"
        "scene.render.filepath = os.path.join(out_dir, 'frame_####.png')\n"
        "obj = type('Obj', (), {})()\n"
        f"{obj_line}"
        "bpy.ops.render.render(animation=True)\n"
        f"# {marker}\n"
    )


def _block_report(score: float = 0.4) -> CriticReport:
    return CriticReport(
        concept_id="fake", iteration=0, overall_score=score,
        issues=[
            CriticIssue(shot_id="node_01", frame_idx=1, severity="block",
                        category="off_screen", issue="x"),
        ],
    )


def _metric_block_report(rule_id: str = "deterministic_failure") -> ConceptMetricReport:
    return ConceptMetricReport(
        concept_id="fake",
        issues=[
            ConceptMetricIssue(
                severity="block",
                rule_id=rule_id,
                message="metric says this scene is structurally wrong",
                suggested_fix="fix deterministic concept metric failure",
            )
        ],
    )


def test_fresh_retry_keeps_metric_and_cross_ref_addenda(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    _install_stubs(
        monkeypatch,
        fake_narrative,
        fake_storyboard,
        scripted_code=[
            _valid_scene("iter00"),
            _valid_scene("iter01"),
            _valid_scene("iter02"),
            _valid_scene("iter03 fresh"),
        ],
        scripted_reports=[
            _block_report(0.4),
            _block_report(0.4),
            _block_report(0.4),
            _OK_REPORT,
        ],
    )
    metric_reports = iter([
        _metric_block_report("deterministic_scene_wrong"),
        _metric_block_report("deterministic_scene_wrong"),
        _metric_block_report("deterministic_scene_wrong"),
        ConceptMetricReport(concept_id="fake"),
    ])

    def fake_metrics(**kwargs):
        try:
            return next(metric_reports)
        except StopIteration:
            return ConceptMetricReport(concept_id="fake")

    monkeypatch.setattr(pipeline, "run_concept_metrics", fake_metrics)
    monkeypatch.setattr(
        pipeline,
        "format_cross_reference_for_coder",
        lambda report: "CRITIC x AST CROSS-REFERENCE\nfix cross-ref item\n",
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=3,
        critic_backend="passthrough",
        preview_render=False,
        diff_repair=False,
    )

    trend = (res.out_dir / "trend_iter03.txt").read_text()
    assert "FRESH RETRY MODE" in trend
    assert "AUTOMATED CONCEPT METRIC FAILURES" in trend
    assert "deterministic_scene_wrong" in trend
    assert "CRITIC x AST CROSS-REFERENCE" in trend


def test_loop_runs_until_pass(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """Pass on iter00 → no retries."""
    code_iter0 = _valid_scene("iter00")
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[code_iter0],
        scripted_reports=[_OK_REPORT],
    )
    res = pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=2,
        critic_backend="passthrough", preview_render=False,
    )
    assert res.ok
    assert counts["n"] == 1
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 0
    assert (res.out_dir / "visual_contracts.json").exists()


def test_concept_metric_block_prevents_critic_pass_and_retries(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("iter00"), _valid_scene("iter01")],
        scripted_reports=[_OK_REPORT, _OK_REPORT],
    )
    metric_reports = iter([
        _metric_block_report("deterministic_scene_wrong"),
        ConceptMetricReport(concept_id="fake"),
    ])

    def fake_metrics(**kwargs):
        try:
            return next(metric_reports)
        except StopIteration:
            return ConceptMetricReport(concept_id="fake")

    monkeypatch.setattr(pipeline, "run_concept_metrics", fake_metrics)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_backend="passthrough",
        preview_render=False,
        diff_repair=False,
    )

    assert res.ok
    assert counts["n"] == 2
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 1
    assert cb["all_history_iterations"][0]["metric_block"] == 1
    assert cb["all_history_iterations"][0]["pass"] is False


def test_success_spec_is_written_and_passed_to_concept_metrics(
    monkeypatch, tmp_path, fake_narrative, fake_storyboard,
):
    concept_yaml = tmp_path / "fake_success.yaml"
    concept_yaml.write_text(yaml.safe_dump({
        "concept_id": "fake",
        "title": "Fake Concept",
        "duration_sec": 2.0,
        "scene_profile": "vector_teaching",
        "persistent_anchors": ["status_readout"],
        "success_spec": {
            "version": 1,
            "success_states": [{
                "id": "early",
                "frame_range": {"fraction": [0.0, 0.5]},
            }],
            "required_visual_evidence": [{
                "kind": "aperture_within_range",
                "anchor": "main_camera",
                "data_path": "data.dof.aperture_fstop",
                "range": [1.4, 2.8],
            }],
        },
    }))
    _install_stubs(
        monkeypatch,
        fake_narrative,
        fake_storyboard,
        scripted_code=[_valid_scene("iter00"), _valid_scene("iter01")],
        scripted_reports=[_block_report(), _OK_REPORT],
    )
    captured = {}

    def fake_metrics(**kwargs):
        captured["success_spec"] = kwargs.get("success_spec")
        captured["auto_success_spec"] = kwargs.get("auto_success_spec")
        return ConceptMetricReport(concept_id="fake")

    monkeypatch.setattr(pipeline, "run_concept_metrics", fake_metrics)

    res = pipeline.run(
        concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_backend="passthrough",
        preview_render=False,
    )

    assert res.ok
    artifact = json.loads((res.out_dir / "success_spec.json").read_text())
    assert artifact["version"] == 1
    generated = json.loads((res.out_dir / "success_spec.generated.json").read_text())
    effective = json.loads((res.out_dir / "success_spec.effective.json").read_text())
    validation = json.loads((res.out_dir / "success_spec.validation.json").read_text())
    assert generated["rules"]
    assert effective["user_success_spec_present"] is True
    assert validation["source"] == "generated"
    assert captured["success_spec"].aperture_fstop_range() == (1.4, 2.8)
    assert captured["auto_success_spec"].rules
    trend = (res.out_dir / "trend_iter01.txt").read_text()
    assert "SUCCESS SPEC" in trend
    assert "AUTO SUCCESS SPEC" in trend
    assert "aperture_fstop must stay in [1.4, 2.8]" in trend


def test_success_spec_text_anchors_merge_into_visual_contracts():
    from cg_tutor.success_spec import SuccessSpec

    contracts = {
        "node_01": VisualContract(
            shot_id="node_01",
            required_anchors=["foreground_subject"],
        ),
    }
    success_spec = SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 0.5]},
            "object_states": {"foreground_subject": ["visible", "sharp"]},
            "readable_text": ["depth_label_near"],
        }],
        "hard_constraints": [{
            "kind": "text_faces_camera",
            "anchors": ["depth_label_near", "lens_readout_hud"],
        }],
    })

    merged = pipeline._merge_success_spec_into_visual_contracts(
        contracts,
        success_spec,
    )

    contract = merged["node_01"]
    assert "foreground_subject" in contract.required_anchors
    assert "depth_label_near" in contract.required_anchors
    assert "lens_readout_hud" in contract.required_anchors
    assert "depth_label_near" in contract.required_labels
    assert any("frame safety margin" in s for s in contract.required_relationships)


def test_loop_skips_identical_scene_and_uses_remaining_budget(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """Identical iter01 should not waste render+critic, but it should not
    consume the rest of the retry budget either."""
    same_code = _valid_scene("stuck")
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            same_code,
            same_code,
            _valid_scene("iter02 escape"),
        ],
        scripted_reports=[_block_report(), _OK_REPORT],
    )
    res = pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=2,
        critic_backend="passthrough", early_stop_stale_iters=0,
        diff_repair=False, preview_render=False,
    )
    assert res.ok
    # iter01 is skipped as identical, then iter02 gets rendered/evaluated.
    assert (res.out_dir / "critic_iter00.json").exists()
    assert not (res.out_dir / "critic_iter01.json").exists()
    assert (res.out_dir / "critic_iter02.json").exists()
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 2
    assert [item["iteration"] for item in cb["all_iterations"]] == [0, 2]


def test_loop_picks_best_when_last_iter_regresses(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """Best-of-N: iter01 clean, iter02 regresses → final.mp4 must be
    composed from iter01's scene, not iter02's."""
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            _valid_scene("iter00"),
            _valid_scene("iter01 clean"),
            _valid_scene("iter02 regressed"),
        ],
        scripted_reports=[
            _block_report(0.3),
            CriticReport(
                concept_id="fake",
                iteration=1,
                overall_score=0.9,
                issues=[],
                execution_errors=["simulated critic timeout"],
            ),
            _block_report(0.4),
        ],
    )
    res = pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=2,
        critic_backend="passthrough", early_stop_stale_iters=0,
        diff_repair=False, preview_render=False,
    )
    assert res.ok
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 1
    # Final scene.py must equal iter01.py, not iter02.py.
    assert "iter01" in (res.out_dir / "scene.py").read_text()
    assert (res.out_dir / "frames.iter01").exists()


def test_loop_exports_scored_video_alternatives(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """When aesthetic and compliance picks differ, keep both videos."""
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            _valid_scene("iter00 compliant"),
            _valid_scene("iter01 pretty but blocked"),
        ],
        scripted_reports=[
            CriticReport(concept_id="fake", iteration=0,
                         overall_score=0.50, issues=[]),
            _block_report(0.65),
        ],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_backend="passthrough",
        early_stop_stale_iters=0,
        diff_repair=False,
        preview_render=False,
    )

    assert res.ok
    assert (res.out_dir / "final.mp4").exists()
    assert (res.out_dir / "final_compliance.mp4").exists()
    assert (res.out_dir / "final_aesthetic.mp4").exists()
    exports = json.loads((res.out_dir / "video_exports.json").read_text())
    by_role = {e["role"]: e for e in exports}
    assert by_role["compliance"]["iteration"] == 0
    assert by_role["aesthetic"]["iteration"] == 1
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["video_exports"] == exports


def test_loop_uses_frame_snapshot_for_best_without_replay(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    runtime_calls = {"n": 0}
    composed = {}
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            _valid_scene("iter00 weak"),
            _valid_scene("iter01 best"),
            _valid_scene("iter02 regressed"),
        ],
        scripted_reports=[
            _block_report(0.3),
            CriticReport(
                concept_id="fake",
                iteration=1,
                overall_score=0.9,
                issues=[],
                execution_errors=["simulated critic timeout"],
            ),
            _block_report(0.4),
        ],
    )

    original_run_script = blender_runtime.run_script

    def counting_runtime(scene_path, frames_dir, **kw):
        runtime_calls["n"] += 1
        return original_run_script(scene_path, frames_dir, **kw)

    def fake_compose(sb, frames_dir, out_dir, *, out_name="final.mp4"):
        if out_name == "final.mp4":
            composed["frames_dir"] = frames_dir
        mp4 = out_dir / out_name
        mp4.write_bytes(b"FAKEMP4")

        class _R:
            ok = True
            stderr = ""
        return _R()

    monkeypatch.setattr(blender_runtime, "run_script", counting_runtime)
    monkeypatch.setattr(compose_mod, "compose_storyboard_video", fake_compose)
    monkeypatch.setattr(pipeline, "compose_storyboard_video", fake_compose)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=2,
        critic_backend="passthrough",
        early_stop_stale_iters=0,
        diff_repair=False,
        preview_render=False,
    )

    assert res.ok
    # One render per evaluated iteration; no extra best-scene replay render.
    assert runtime_calls["n"] == 3
    assert composed["frames_dir"] == res.out_dir / "frames"
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 1
    assert cb["replay_ran"] is False
    assert cb["replay_hash_mismatch"] is False
    assert cb["frames_hash"] == cb["replay_frames_hash"]
    assert cb["all_iterations"][1]["frames_dir"] == "frames.iter01"


def test_loop_can_pick_semantic_best_when_requested(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """semantic best selection lets teaching fidelity beat a higher score
    when framing is otherwise acceptable."""
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            _valid_scene("iter00 semantic dirty"),
            _valid_scene("iter01 semantic clean"),
        ],
        scripted_reports=[
            CriticReport(
                concept_id="fake",
                iteration=0,
                overall_score=0.95,
                issues=[
                    CriticIssue(
                        shot_id="node_01",
                        frame_idx=1,
                        severity="block",
                        category="concept_mismatch",
                        issue="Required label missing.",
                    )
                ],
            ),
            CriticReport(
                concept_id="fake",
                iteration=1,
                overall_score=0.75,
                issues=[],
            ),
        ],
    )
    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_backend="passthrough",
        early_stop_stale_iters=0,
        best_selection_mode="semantic",
    )

    assert res.ok
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_selection_mode"] == "semantic"
    assert cb["best_iteration"] == 1
    assert "iter01" in (res.out_dir / "scene.py").read_text()


def test_loop_writes_missing_objects_artifact(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """P0-A: coder forgets a storyboard object → missing_objects.iter00.json
    is written and lists the gap."""
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        # Note: scene does NOT mention "red_sphere"
        scripted_code=[_valid_scene("missing object", include_obj=False)],
        scripted_reports=[_OK_REPORT],
    )
    pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=0,
        critic_backend="passthrough",
    )
    missing_path = tmp_path / "fake" / "missing_objects.iter00.json"
    assert missing_path.exists()
    data = json.loads(missing_path.read_text())
    assert data == {"node_01": ["red_sphere"]}


def test_loop_no_missing_objects_artifact_when_clean(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("clean")],
        scripted_reports=[_OK_REPORT],
    )
    pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=0,
        critic_backend="passthrough",
    )
    assert not (tmp_path / "fake" / "missing_objects.iter00.json").exists()


def test_loop_repairs_scene_verifier_blocks_before_render(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            "import bpy\n# missing render contracts\n",
            _valid_scene("verifier repaired"),
        ],
        scripted_reports=[_OK_REPORT],
    )

    res = pipeline.run(
        fake_concept_yaml, out_root=tmp_path, max_critic_iterations=0,
        critic_backend="passthrough", preview_render=False,
    )

    assert res.ok
    assert counts["n"] == 2
    assert (res.out_dir / "scene_verifier.iter00.json").exists()
    assert (res.out_dir / "scene_verifier.iter00.repair.json").exists()
    assert "verifier repaired" in (res.out_dir / "scene.py").read_text()


def test_loop_rejects_syntax_bad_repair_and_uses_compiled_fallback(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled fallback"),
    )
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            "import bpy\n# missing render call\n",
            "import bpy\nif True\n    bpy.ops.render.render(animation=True)\n",
        ],
        scripted_reports=[_OK_REPORT],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=False,
        max_verifier_repair_iters=1,
    )

    assert res.ok
    assert counts["n"] == 2
    assert (res.out_dir / "scene_verifier.iter00.repair.json").exists()
    assert (res.out_dir / "scene_verifier.iter00.render_tail.json").exists()
    assert (res.out_dir / "scene.render_tail.used.iter00.txt").exists()
    assert "CG-Tutor deterministic render tail repair" in (
        res.out_dir / "scene.py"
    ).read_text()


def test_repair_regression_rejects_syntax_to_missing_render_trade():
    current = SceneVerificationReport(issues=[
        SceneVerificationIssue(
            severity="block",
            rule_id="syntax_error",
            message="bad syntax",
            suggested_fix="fix syntax",
        )
    ])
    candidate = SceneVerificationReport(issues=[
        SceneVerificationIssue(
            severity="block",
            rule_id="missing_render_call",
            message="missing render",
            suggested_fix="call render",
        )
    ])

    reason = pipeline._repair_regression_reason(
        current,
        ContractValidationReport(),
        candidate,
        ContractValidationReport(),
    )

    assert reason is not None
    assert "fatal verifier block traded" in reason


def test_compiled_fallback_allows_insufficient_vectors_but_not_missing_anchors():
    allowed = ContractValidationReport(violations=[
        ContractViolation(
            severity="block",
            rule_id="contract_insufficient_vector_geometry",
            shot_id="node_01",
            field="required_vectors",
            expected=3,
            found=1,
            message="not enough vectors",
            suggested_fix="add vectors",
        )
    ])
    rejected = ContractValidationReport(violations=[
        ContractViolation(
            severity="block",
            rule_id="contract_missing_scene_anchors",
            shot_id="node_01",
            field="required_anchors",
            expected=1,
            found=0,
            message="missing anchor",
            suggested_fix="add anchor",
        )
    ])

    assert pipeline._fallback_contract_is_render_safe(allowed)
    assert not pipeline._fallback_contract_is_render_safe(rejected)


def test_loop_rejects_repair_that_trades_block_for_missing_objects(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled fallback"),
    )
    no_render = _valid_scene("needs repair").replace(
        "bpy.ops.render.render(animation=True)\n",
        "",
    )
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            no_render,
            _valid_scene("dropped storyboard object", include_obj=False),
        ],
        scripted_reports=[_OK_REPORT],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=False,
        max_verifier_repair_iters=1,
    )

    assert res.ok
    assert counts["n"] == 2
    repair = json.loads(
        (res.out_dir / "scene_verifier.iter00.repair.json").read_text()
    )
    assert repair["missing_objects"] == {"node_01": ["red_sphere"]}
    assert "CG-Tutor deterministic render tail repair" in (
        res.out_dir / "scene.py"
    ).read_text()


def test_loop_rejects_repair_that_increases_verifier_blocks(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    nonfatal = _valid_scene("one nonfatal block") + (
        "\n# missing_profile_tracer_geometry proxy\n"
    )
    worse = nonfatal.replace(
        "bpy.ops.render.render(animation=True)\n",
        "# render removed by bad repair\n",
    )
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            nonfatal,
            worse,
        ],
        scripted_reports=[_OK_REPORT],
    )

    def fake_verify(
        code, sb, visual_contracts=None, scene_profile=None, **kwargs,
    ):
        if "render removed" in code:
            return SceneVerificationReport(
                issues=[
                    SceneVerificationIssue(
                        severity="block",
                        rule_id="missing_render_call",
                        message="no render",
                        suggested_fix="add render",
                    ),
                    SceneVerificationIssue(
                        severity="block",
                        rule_id="missing_scene_camera",
                        message="no camera",
                        suggested_fix="add camera",
                    ),
                ],
                missing_objects={},
            )
        return SceneVerificationReport(
            issues=[
                SceneVerificationIssue(
                    severity="block",
                    rule_id="missing_profile_tracer_geometry",
                    message="no tracer",
                    suggested_fix="add tracer",
                )
            ],
            missing_objects={},
        )

    monkeypatch.setattr(pipeline, "verify_scene_code", fake_verify)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=False,
        max_verifier_repair_iters=1,
    )

    assert res.ok
    assert counts["n"] == 2
    assert "one nonfatal block" in (res.out_dir / "scene.py").read_text()
    assert "render removed" not in (res.out_dir / "scene.py").read_text()
    assert not (res.out_dir / "scene_verifier.iter00.compiled_fallback.json").exists()


def test_loop_uses_compiled_fallback_when_iter00_repair_call_fails(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled fallback"),
    )
    no_render = _valid_scene("needs repair").replace(
        "bpy.ops.render.render(animation=True)\n",
        "",
    )
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[no_render, RuntimeError("simulated repair outage")],
        scripted_reports=[_OK_REPORT],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=False,
        max_verifier_repair_iters=1,
    )

    assert res.ok
    assert counts["n"] == 2
    assert (res.out_dir / "scene_verifier.iter00.render_tail.json").exists()
    assert "CG-Tutor deterministic render tail repair" in (
        res.out_dir / "scene.py"
    ).read_text()


def test_loop_does_not_diff_repair_against_compiled_fallback(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled fallback"),
    )
    no_render = _valid_scene("needs repair").replace(
        "bpy.ops.render.render(animation=True)\n",
        "",
    )
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            no_render,
            RuntimeError("simulated repair outage"),
            _valid_scene("iter01 llm regeneration"),
        ],
        scripted_reports=[_block_report(), _OK_REPORT],
    )

    def fail_if_diff_repair(*_args, **_kw):
        raise AssertionError("diff repair should not target compiled fallback")

    monkeypatch.setattr(
        blender_coder,
        "repair_bpy_script_diff",
        fail_if_diff_repair,
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_backend="passthrough",
        preview_render=False,
        max_verifier_repair_iters=1,
    )

    assert res.ok
    best = json.loads((res.out_dir / "critic_best.json").read_text())
    assert best["all_iterations"][0]["scene_origin"] == "llm"
    assert best["all_iterations"][1]["scene_origin"] == "llm"
    assert "iter01 llm regeneration" in (res.out_dir / "scene.py").read_text()


def test_loop_preview_crash_at_iter00_falls_back_to_compiled(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """LLM scene passes static checks but crashes Blender at preview time.
    Pipeline should swap to the compiled scaffold and continue to render +
    critic instead of early-exiting. Regression test for the iter00 dead-
    end where verifier_clean + preview_crash meant the run terminated with
    no video at all."""
    compiled_marker = "compiled fallback OK"
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene(compiled_marker)
        + "# CG_TUTOR_PREVIEW_FRAMES support\n",
    )
    llm_scene = (
        _valid_scene("llm preview crash") + "# CG_TUTOR_PREVIEW_FRAMES support\n"
    )
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[llm_scene],
        scripted_reports=[_OK_REPORT],
    )

    def crash_on_llm_preview(scene_path, frames_dir, **kw):
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene_text = Path(scene_path).read_text()
        preview_raw = (kw.get("env_overrides") or {}).get(
            "CG_TUTOR_PREVIEW_FRAMES"
        )
        is_preview = preview_raw is not None
        llm_crash = is_preview and "llm preview crash" in scene_text
        if not llm_crash:
            frame_ids = (
                [int(x) for x in preview_raw.split(",") if x.strip()]
                if preview_raw else list(range(1, 5))
            )
            seed = sum(scene_text.encode()) % 255
            for i in frame_ids:
                Image.new(
                    "RGB",
                    (8, 8),
                    (
                        (seed + i) % 255,
                        (seed + 40 + i) % 255,
                        (seed + 80 + i) % 255,
                    ),
                ).save(frames_dir / f"frame_{i:04d}.png")

        class _RR:
            ok = not llm_crash
            returncode = 1 if llm_crash else 0
            stdout = ""
            stderr = "Traceback ...\nKeyError: 'sphere'" if llm_crash else ""
        return _RR()

    monkeypatch.setattr(blender_runtime, "run_script", crash_on_llm_preview)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=True,
    )

    assert res.ok
    assert res.final_mp4 is not None and res.final_mp4.exists()
    fb_report = res.out_dir / "preview_report.iter00.compiled_fallback.json"
    assert fb_report.exists()
    assert json.loads(fb_report.read_text())["ok"] is True
    # LLM preview report should still be there, recording the crash.
    crash_report = json.loads(
        (res.out_dir / "preview_report.iter00.json").read_text()
    )
    assert crash_report["ok"] is False
    assert any(
        i["rule_id"] == "preview_blender_error" for i in crash_report["issues"]
    )
    # scene.py should now be the compiled scaffold.
    assert compiled_marker in (res.out_dir / "scene.py").read_text()
    used = (res.out_dir / "scene.compiled.used.txt").read_text()
    assert "preview_crash" in used


def test_loop_preview_crash_when_compiled_also_fails_exits_cleanly(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    """If both the LLM scene AND the compiled scaffold crash preview, the
    pipeline must early-exit cleanly (no half-rendered video) rather than
    trying to render a known-broken scene."""
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled also crashes")
        + "# CG_TUTOR_PREVIEW_FRAMES support\n",
    )
    llm_scene = (
        _valid_scene("llm preview crash") + "# CG_TUTOR_PREVIEW_FRAMES support\n"
    )
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[llm_scene],
        scripted_reports=[_OK_REPORT],
    )

    def crash_on_all_previews(scene_path, frames_dir, **kw):
        frames_dir.mkdir(parents=True, exist_ok=True)
        preview_raw = (kw.get("env_overrides") or {}).get(
            "CG_TUTOR_PREVIEW_FRAMES"
        )
        is_preview = preview_raw is not None

        class _RR:
            ok = not is_preview
            returncode = 1 if is_preview else 0
            stdout = ""
            stderr = "Traceback ...\nRuntimeError" if is_preview else ""
        return _RR()

    monkeypatch.setattr(blender_runtime, "run_script", crash_on_all_previews)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        preview_render=True,
    )

    assert not res.ok
    assert res.final_mp4 is None
    # Both LLM and compiled fallback should have left a failed preview report.
    assert (res.out_dir / "preview_report.iter00.json").exists()
    assert (
        res.out_dir / "preview_report.iter00.compiled_fallback.json"
    ).exists()
    fb = json.loads(
        (res.out_dir / "preview_report.iter00.compiled_fallback.json").read_text()
    )
    assert fb["ok"] is False


def test_fresh_branch_verifier_failure_does_not_use_compiled_fallback(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    monkeypatch.setattr(
        pipeline,
        "compile_storyboard_to_bpy",
        lambda sb, **kwargs: _valid_scene("compiled fallback should not be used"),
    )
    no_render = _valid_scene("fresh needs repair").replace(
        "bpy.ops.render.render(animation=True)\n",
        "",
    )
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[
            _valid_scene("iter00"),
            _valid_scene("iter01"),
            _valid_scene("iter02"),
            no_render,
            RuntimeError("fresh repair outage"),
        ],
        scripted_reports=[
            _block_report(0.3),
            _block_report(0.35),
            _block_report(0.4),
        ],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=3,
        critic_backend="passthrough",
        preview_render=False,
        diff_repair=False,
        max_verifier_repair_iters=2,
    )

    assert res.ok
    assert not (res.out_dir / "scene_verifier.iter03.compiled_fallback.json").exists()
    best = json.loads((res.out_dir / "critic_best.json").read_text())
    assert [item["iteration"] for item in best["all_iterations"]] == [0, 1, 2]
    assert "compiled fallback should not be used" not in (
        res.out_dir / "scene.py"
    ).read_text()


def test_loop_can_use_compiler_only_for_iter00(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("should not be used")],
        scripted_reports=[_OK_REPORT],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=0,
        critic_backend="passthrough",
        compiler_only=True,
    )

    assert res.ok
    assert counts["n"] == 0
    assert (res.out_dir / "scene.compiled.py").exists()
    assert (res.out_dir / "scene.compiled.used.txt").exists()


def test_loop_uses_critic_ensemble_when_requested(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("iter00")],
        scripted_reports=[_OK_REPORT],
    )
    seen = {}

    def fake_ensemble(sb, frames_dir, *, backends, iteration=0, out_dir=None,
                      narrative=None, scene_profile=None, strictness="consensus"):
        seen["backends"] = backends
        seen["strictness"] = strictness
        report = CriticReport(
            concept_id="fake",
            iteration=iteration,
            overall_score=0.9,
            issues=[],
        )
        if out_dir is not None:
            (out_dir / f"critic_iter{iteration:02d}.json").write_text(
                report.model_dump_json(indent=2)
            )
        return report

    monkeypatch.setattr(render_critic, "inspect_ensemble", fake_ensemble)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        max_critic_iterations=1,
        critic_ensemble=("passthrough", "api"),
    )

    assert res.ok
    assert seen["backends"] == ("passthrough", "api")
    assert seen["strictness"] == "strict"


def test_resume_continues_from_existing_critic_history(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    out_dir = tmp_path / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_narrative(fake_narrative, out_dir)
    _save_storyboard(fake_storyboard, out_dir)
    (out_dir / "scene.py").write_text(_valid_scene("resumed baseline"))
    (out_dir / "scene.iter00.py").write_text(_valid_scene("iter00 old"))
    (out_dir / "scene.iter01.py").write_text(_valid_scene("iter01 old"))
    (out_dir / "critic_iter00.json").write_text(
        _block_report(0.3).model_copy(update={"iteration": 0}).model_dump_json(
            indent=2
        )
    )
    (out_dir / "critic_iter01.json").write_text(
        CriticReport(
            concept_id="fake",
            iteration=1,
            overall_score=0.85,
            issues=[],
        ).model_dump_json(indent=2)
    )

    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("iter02 resumed")],
        scripted_reports=[
            CriticReport(
                concept_id="fake",
                iteration=2,
                overall_score=0.95,
                issues=[],
            )
        ],
    )

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        resume=True,
        max_critic_iterations=2,
        critic_backend="passthrough",
        diff_repair=False,
        preview_render=False,
    )

    assert res.ok
    assert counts["n"] == 1
    assert (res.out_dir / "critic_iter02.json").exists()
    cb = json.loads((res.out_dir / "critic_best.json").read_text())
    assert cb["best_iteration"] == 2
    assert len(cb["all_iterations"]) == 1
    assert len(cb["all_history_iterations"]) == 3


def test_resume_rehydrates_metric_and_cross_ref_diagnostics(tmp_path):
    out_dir = tmp_path / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.iter00.py").write_text(_valid_scene("iter00"))
    (out_dir / "critic_iter00.json").write_text(_OK_REPORT.model_dump_json(indent=2))
    (out_dir / "concept_metric_report.iter00.json").write_text(json.dumps({
        "ok": False,
        "concept_id": "fake",
        "block": 1,
        "warn": 0,
        "metrics": {},
        "issues": [{
            "severity": "block",
            "rule_id": "deterministic_scene_wrong",
            "message": "bad",
            "suggested_fix": "fix",
        }],
    }, indent=2))
    (out_dir / "critic_cross_reference.iter00.json").write_text(json.dumps({
        "concept_id": "fake",
        "iteration": 0,
        "actionable_count": 1,
        "findings": [{
            "rule_id": "missing_object_creation",
            "severity": "actionable",
            "diagnosis": "missing x",
            "critic_source": "critic",
            "ast_evidence": "ast",
            "suggested_fix": "fix",
        }],
    }, indent=2))

    history = pipeline._load_critic_history_from_disk(out_dir)
    metrics = pipeline._load_metric_history_from_disk(out_dir)
    cross_refs = pipeline._load_cross_ref_history_from_disk(out_dir)
    pipeline._hydrate_history_diagnostics(history, metrics, cross_refs)

    assert history[0].metric_block_count == 1
    assert history[0].metric_success_soft_count == 1
    assert history[0].cross_ref_actionable_count == 1
    assert pipeline._critic_iteration_summary(history[0])["pass"] is False


def test_iteration_pass_requires_no_success_hard_metric():
    report = _OK_REPORT
    metric_report = ConceptMetricReport(
        concept_id="fake",
        issues=[
            ConceptMetricIssue(
                severity="block",
                rule_id="success_text_anchor_missing",
                message="missing",
                suggested_fix="add exact anchor",
                failure_class="success_hard",
            )
        ],
    )

    assert pipeline._iteration_pass_threshold(report, metric_report) is False


def test_critic_iteration_pass_rejects_degraded_fallback():
    item = pipeline.CriticIteration(
        iteration=0,
        report=_OK_REPORT,
        scene_path=Path("scene.py"),
        render_ok=True,
        n_frames=1,
        scene_origin="compiled_fallback",
        fallback_degraded=True,
    )

    assert pipeline._critic_iteration_pass(item) is False
    assert pipeline._critic_iteration_summary(item)["fallback_degraded"] is True


def test_selected_reason_marks_fallback_diagnostic_video():
    fallback = pipeline.CriticIteration(
        iteration=0,
        report=_OK_REPORT,
        scene_path=Path("scene.py"),
        render_ok=True,
        n_frames=1,
        scene_origin="compiled_fallback",
        fallback_degraded=True,
    )

    assert pipeline._selected_reason(fallback, [fallback]) == (
        "no_renderable_llm_candidate"
    )


def test_append_auto_success_issues_preserves_returned_classification():
    metric_report = ConceptMetricReport(concept_id="fake")

    pipeline._append_auto_success_issues(
        metric_report,
        [{
            "severity": "warn",
            "rule_id": "auto_success_visibility_unproven",
            "message": "created but not visible",
            "suggested_fix": "move existing object",
            "failure_class": "success_soft",
        }],
    )

    assert metric_report.warn_count == 1
    assert metric_report.block_count == 0
    assert metric_report.success_soft_count == 1
    assert metric_report.success_hard_count == 0


def test_addendum_bundle_includes_minimal_success_repair_principle():
    metric_text = (
        "AUTOMATED CONCEPT METRIC FAILURES\n"
        "[BLOCK / success_hard] success_text_anchor_missing\n"
        "  required anchors are absent from scene.py"
    )
    text = pipeline.AddendumBundle(
        metric=metric_text,
        has_success_hard=True,
    ).join()

    assert "SUCCESS-HARD REPAIR PRINCIPLE" in text
    assert "Missing anchor: add the exact named object only." in text
    assert "Do not add labels, decorations" in text


def test_addendum_bundle_omits_repair_principle_without_success_signal():
    text = pipeline.AddendumBundle(
        metric="AUTOMATED CONCEPT METRIC FAILURES\n[WARN / aesthetic_warn] mild",
        shot_contract="shot grounding text",
    ).join()

    assert "SUCCESS-HARD REPAIR PRINCIPLE" not in text


def test_addendum_bundle_omits_repair_principle_with_real_metric_formatter():
    """A1 regression: format_concept_metric_report_for_coder embeds the literal
    string 'success_hard' in its descriptive header. AddendumBundle must NOT
    inject the repair principle just because the formatted metric text mentions
    'success_hard' in boilerplate — only when has_success_hard is True (i.e.,
    an actual success-hard issue fired)."""
    from cg_tutor.concept_metrics import (
        ConceptMetricIssue,
        ConceptMetricReport,
        format_concept_metric_report_for_coder,
    )

    aesthetic_only_report = ConceptMetricReport(
        concept_id="some_concept",
        issues=[
            ConceptMetricIssue(
                severity="warn",
                rule_id="dof_aperture_too_wide",
                message="aperture is wider than recommended",
                suggested_fix="narrow the aperture",
                failure_class="aesthetic_warn",
            ),
        ],
    )
    metric_text = format_concept_metric_report_for_coder(aesthetic_only_report)

    # Sanity: the boilerplate header DOES contain the literal substring.
    # If this assertion ever fails, the formatter changed and the A1 bug is
    # gone — but the test still serves as an interface contract.
    assert "success_hard" in metric_text

    text = pipeline.AddendumBundle(
        metric=metric_text,
        has_success_hard=False,
    ).join()

    assert "SUCCESS-HARD REPAIR PRINCIPLE" not in text


def test_addendum_bundle_includes_repair_principle_when_cross_ref_present():
    text = pipeline.AddendumBundle(
        cross_ref="CRITIC × AST CROSS-REFERENCE\n[actionable] missing_object_creation",
    ).join()

    assert "SUCCESS-HARD REPAIR PRINCIPLE" in text


def test_resume_recovers_mid_iteration_after_render_before_critic(
    monkeypatch, tmp_path, fake_concept_yaml, fake_narrative, fake_storyboard,
):
    out_dir = tmp_path / "fake"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_narrative(fake_narrative, out_dir)
    _save_storyboard(fake_storyboard, out_dir)
    scene_code = _valid_scene("iter00 interrupted")
    (out_dir / "scene.py").write_text(scene_code)
    (out_dir / "scene.iter00.py").write_text(scene_code)
    (out_dir / "resume_checkpoint.json").write_text(json.dumps({
        "iteration": 0,
        "stage": "render_done",
        "scene_origin": "llm",
        "render_ok": True,
        "n_frames": 4,
    }, indent=2))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 5):
        Image.new("RGB", (8, 8), (idx, idx, idx)).save(
            frames_dir / f"frame_{idx:04d}.png"
        )

    counts = _install_stubs(
        monkeypatch, fake_narrative, fake_storyboard,
        scripted_code=[_valid_scene("should not be used on resume")],
        scripted_reports=[_OK_REPORT],
    )

    render_calls = {"n": 0}

    def _boom_runtime(*args, **kwargs):
        render_calls["n"] += 1
        raise AssertionError("resume should not re-render when render_done checkpoint exists")

    monkeypatch.setattr(blender_runtime, "run_script", _boom_runtime)

    res = pipeline.run(
        fake_concept_yaml,
        out_root=tmp_path,
        resume=True,
        max_critic_iterations=0,
        keep_frames=False,
        preview_render=False,
        critic_backend="passthrough",
    )

    assert res.ok
    assert counts["n"] == 0
    assert render_calls["n"] == 0
    assert not (out_dir / "resume_checkpoint.json").exists()


def test_grounding_patch_promotes_repeated_critic_abstraction():
    issue = CriticIssue(
        shot_id="shot1",
        frame_idx=1,
        severity="warn",
        category="concept_mismatch",
        issue="OPEN sign is rendered as a plain glowing rectangle.",
    )
    history = [
        pipeline.CriticIteration(
            iteration=0,
            report=CriticReport(
                concept_id="fake",
                iteration=0,
                overall_score=0.6,
                issues=[issue],
            ),
            scene_path=Path("scene0.py"),
            render_ok=True,
            n_frames=1,
        ),
        pipeline.CriticIteration(
            iteration=1,
            report=CriticReport(
                concept_id="fake",
                iteration=1,
                overall_score=0.6,
                issues=[issue],
            ),
            scene_path=Path("scene1.py"),
            render_ok=True,
            n_frames=1,
        ),
    ]

    patch = pipeline._critic_grounding_patch(history)
    text = pipeline._format_grounding_patch_for_coder(patch)

    assert "forbidden_abstractions_addendum" in text
    assert "plain glowing rectangle" in text


def test_grounding_patch_extracts_critic_required_anchors_and_labels():
    issue = CriticIssue(
        shot_id="shot1",
        frame_idx=1,
        severity="block",
        category="concept_mismatch",
        issue="The required object_point_B label is absent and x = f X/Z is missing.",
        suggested_fix={
            "object_point_b.label.visible": True,
            "camera_center_c.label.text": "C",
            "formula.text": "x = f X/Z",
            "projection_rays.pass_through_C": True,
        },
    )
    history = [
        pipeline.CriticIteration(
            iteration=0,
            report=CriticReport(
                concept_id="fake",
                iteration=0,
                overall_score=0.6,
                issues=[issue],
            ),
            scene_path=Path("scene0.py"),
            render_ok=True,
            n_frames=1,
        )
    ]

    patch = pipeline._critic_grounding_patch(history)
    merged = pipeline._merge_grounding_patch_into_visual_contracts(
        {"shot1": VisualContract(shot_id="shot1")},
        patch,
    )
    text = pipeline._format_grounding_patch_for_coder(patch)

    assert "object_point_b" in patch["required_anchors"]
    assert "camera_center_c" in patch["required_anchors"]
    assert "x = f X/Z" in patch["required_labels"]
    assert "required_anchors_from_critic" in text
    assert "object_point_b" in merged["shot1"].required_anchors


def test_storyboard_anchor_augmentation_adds_missing_contract_anchor():
    sb = Storyboard.model_validate({
        "concept_id": "fake",
        "fps": 24,
        "resolution": [320, 240],
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -5, 3],
                "look_at": [0, 0, 0],
            }],
            "objects": [{
                "name": "subject",
                "type": "mesh",
                "primitive": "sphere",
            }],
        }],
    })
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["neon_sign_OPEN"],
        )
    }

    missing = pipeline._missing_storyboard_contract_anchors(sb, contracts)
    repaired, summary = pipeline._augment_storyboard_with_contract_anchors(
        sb, missing,
    )

    assert missing == {"shot1": ["neon_sign_OPEN"]}
    assert summary["changed"]
    assert "neon_sign_OPEN" in {obj.name for obj in repaired.shots[0].objects}


def test_fresh_retry_only_uses_final_stale_slot():
    def item(iteration: int, block: int) -> pipeline.CriticIteration:
        issues = [
            CriticIssue(
                shot_id="shot1",
                frame_idx=1,
                severity="block",
                category="off_screen",
                issue=f"block {idx}",
            )
            for idx in range(block)
        ]
        return pipeline.CriticIteration(
            iteration=iteration,
            report=CriticReport(
                concept_id="fake",
                iteration=iteration,
                overall_score=0.4,
                issues=issues,
            ),
            scene_path=Path(f"scene{iteration}.py"),
            render_ok=True,
            n_frames=1,
        )

    history = [item(0, 2), item(1, 2), item(2, 2)]

    assert not pipeline._should_use_fresh_retry(
        history, iteration=2, max_iteration=3,
    )
    assert pipeline._should_use_fresh_retry(
        history, iteration=3, max_iteration=3,
    )
