"""Bounded stdout/stderr capture in blender_runtime.

The audit flagged ``capture_output=True`` as a memory risk because
Blender can emit GB-scale logs. Our replacement streams output through
a deque that keeps only the most recent N bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cg_tutor.blender import runtime as blender_runtime


def _stub_python_runtime(monkeypatch):
    """Treat the local Python interpreter as if it were Blender, so we
    can test ``_run_with_bounded_capture`` without a real Blender install.

    The runtime constructs the command as ``[blender, "-b", ...]``; we
    pass scripts using the ``-c`` style instead by replacing the binary
    resolver and the script-call shape entirely.
    """
    monkeypatch.setattr(blender_runtime, "resolve_blender_bin",
                        lambda: sys.executable)


def test_bounded_capture_keeps_tail_of_large_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr(blender_runtime, "_MAX_CAPTURED_BYTES", 1024)
    out = tmp_path / "frames"
    out.mkdir()

    # A Python script that emits ~10 KB on stdout.
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "for i in range(2000):\n"
        "    sys.stdout.write(f'line {i:05d}\\n')\n"
        "    sys.stdout.flush()\n"
    )

    rr = blender_runtime._run_with_bounded_capture(
        [sys.executable, str(script)],
        env={},
        timeout_sec=10,
        out_dir=out,
    )
    assert rr.returncode == 0
    # Captured stdout should be capped well under the raw 10 KB output,
    # and end with the *last* lines (tail-preserving behaviour).
    assert len(rr.stdout.encode("utf-8")) <= 1024 + 64 * 1024  # 64 KB chunk + cap slack
    assert "line 01999" in rr.stdout
    # Early lines fell out of the bounded buffer.
    assert "line 00000" not in rr.stdout


def test_timeout_kills_and_returns_negative_returncode(tmp_path, monkeypatch):
    out = tmp_path / "frames"
    out.mkdir()
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    rr = blender_runtime._run_with_bounded_capture(
        [sys.executable, str(script)],
        env={},
        timeout_sec=1,
        out_dir=out,
    )
    assert rr.returncode < 0
    assert "killed after" in rr.stderr


def test_bounded_capture_no_data_loss_for_small_output(tmp_path, monkeypatch):
    out = tmp_path / "frames"
    out.mkdir()
    script = tmp_path / "small.py"
    script.write_text("print('hello')\nprint('world')\n")
    rr = blender_runtime._run_with_bounded_capture(
        [sys.executable, str(script)],
        env={},
        timeout_sec=10,
        out_dir=out,
    )
    assert rr.returncode == 0
    assert "hello" in rr.stdout
    assert "world" in rr.stdout
