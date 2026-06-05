"""Subprocess-level tests for the ffmpeg wrapper.

Existing ``test_ffmpeg_filter.py`` covers the filter_complex string
builder. This file complements it with tests on how
``frames_to_mp4`` assembles the actual command list and dispatches
to ``subprocess.run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cg_tutor.composer import ffmpeg_wrapper


def test_frames_to_mp4_includes_libx264_yuv420p_and_crf(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw

        class P:
            returncode = 0
            stderr = ""
        return P()

    monkeypatch.setattr(ffmpeg_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(ffmpeg_wrapper, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")

    out = tmp_path / "out.mp4"
    out.write_bytes(b"placeholder")  # ok flag checks .exists()
    res = ffmpeg_wrapper.frames_to_mp4(
        tmp_path / "frames", out, fps=24, crf=18,
    )
    assert res.ok
    cmd = captured["cmd"]
    assert "-c:v" in cmd and "libx264" in cmd
    assert "-pix_fmt" in cmd and "yuv420p" in cmd
    assert "-crf" in cmd and "18" in cmd
    assert "-framerate" in cmd and "24" in cmd


def test_frames_to_mp4_propagates_nonzero_returncode(tmp_path, monkeypatch):
    def fake_run(cmd, **kw):
        class P:
            returncode = 1
            stderr = "ffmpeg crashed"
        return P()

    monkeypatch.setattr(ffmpeg_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(ffmpeg_wrapper, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")

    out = tmp_path / "out.mp4"
    # Don't pre-create out; .ok should still be False.
    res = ffmpeg_wrapper.frames_to_mp4(tmp_path / "frames", out)
    assert not res.ok
    assert res.returncode == 1
    assert "crashed" in res.stderr


def test_frames_to_mp4_legacy_overlay_kwargs_upgrade_to_overlay_list(
    tmp_path, monkeypatch
):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd

        class P:
            returncode = 0
            stderr = ""
        return P()

    monkeypatch.setattr(ffmpeg_wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(ffmpeg_wrapper, "_resolve_ffmpeg", lambda: "/usr/bin/ffmpeg")

    out = tmp_path / "out.mp4"
    out.write_bytes(b"placeholder")
    ffmpeg_wrapper.frames_to_mp4(
        tmp_path / "frames", out,
        overlay_png=tmp_path / "ov.png",
        overlay_xy=(10, 20),
        overlay_width=300,
    )
    # The legacy path adds a single overlay input and a filter_complex.
    assert "-filter_complex" in captured["cmd"]


def test_ffmpeg_not_found_raises():
    import shutil
    saved = ffmpeg_wrapper._resolve_ffmpeg.__globals__["shutil"]
    try:
        class FakeShutil:
            @staticmethod
            def which(_name):
                return None
        ffmpeg_wrapper._resolve_ffmpeg.__globals__["shutil"] = FakeShutil
        with pytest.raises(ffmpeg_wrapper.FfmpegNotFound):
            ffmpeg_wrapper._resolve_ffmpeg()
    finally:
        ffmpeg_wrapper._resolve_ffmpeg.__globals__["shutil"] = saved
