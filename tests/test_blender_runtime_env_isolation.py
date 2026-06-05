"""Make sure runtime.run_script sets CG_TUTOR_OUT_DIR per-call rather
than relying on a global mutation that leaks between callers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cg_tutor.blender import runtime as blender_runtime


def test_run_script_sets_out_dir_in_env_without_polluting_parent(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(blender_runtime, "resolve_blender_bin",
                        lambda: sys.executable)

    out = tmp_path / "frames"
    out.mkdir()

    # We replace `_run_with_bounded_capture` with a spy that captures the
    # env it was given without actually launching a subprocess.
    seen_env = {}

    def fake_run(cmd, *, env, timeout_sec, out_dir):
        seen_env.update(env)
        return blender_runtime.RunResult(
            returncode=0, stdout="", stderr="", frames_dir=out_dir,
        )

    monkeypatch.setattr(blender_runtime, "_run_with_bounded_capture", fake_run)

    # Capture the parent process state before/after.
    before = os.environ.get("CG_TUTOR_OUT_DIR")
    blender_runtime.run_script(Path("/tmp/scene.py"), out)
    after = os.environ.get("CG_TUTOR_OUT_DIR")

    # The subprocess saw the out_dir,
    assert seen_env["CG_TUTOR_OUT_DIR"] == str(out.resolve())
    # but the parent process state is unchanged.
    assert before == after
