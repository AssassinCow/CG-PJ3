"""Run bpy scripts via headless Blender subprocess.

Blender's pip-installable `bpy` module is fragile (Python-version tied,
multi-GB install). We invoke a real Blender executable instead so the
scaffolding works on stock machines.

Usage:
    runtime.run_script(Path("scene.py"), out_dir=Path("outputs/phong/frames"))
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


# Hard cap on captured Blender stdout/stderr (per stream) to avoid OOM
# when Blender renders many frames with verbose logging. We keep the
# *tail* because the interesting traceback / API errors are always at
# the end. ~4 MB is plenty for a 10k-line traceback while bounding
# memory at ~8 MB per render call.
_MAX_CAPTURED_BYTES = 4 * 1024 * 1024


class BlenderNotFound(RuntimeError):
    pass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    frames_dir: Path

    @property
    def ok(self) -> bool:
        # Blender returns 0 even when the user script raises, so we also
        # treat a Python traceback in stderr/stdout as failure.
        if self.returncode != 0:
            return False
        if "Traceback (most recent call last)" in self.stderr:
            return False
        if "Traceback (most recent call last)" in self.stdout:
            return False
        return True


def resolve_blender_bin() -> str:
    explicit = os.environ.get("BLENDER_BIN")
    if explicit:
        if shutil.which(explicit) or Path(explicit).exists():
            return explicit
        raise BlenderNotFound(f"BLENDER_BIN={explicit!r} is set but not executable")
    found = shutil.which("blender")
    if found:
        return found
    raise BlenderNotFound(
        "Blender not on PATH. Install it (apt install blender, "
        "or download from blender.org) and set BLENDER_BIN if not on PATH."
    )


def _looks_like_windows_exe(path: str) -> bool:
    low = path.lower()
    return low.endswith(".exe") or low.startswith("/mnt/")


def _add_wslenv_entry(existing: str, entry: str) -> str:
    entries = [e for e in existing.split(":") if e]
    name = entry.split("/", 1)[0]
    if not any(e.split("/", 1)[0] == name for e in entries):
        entries.append(entry)
    return ":".join(entries)


def _prepare_windows_interop_env(env: dict[str, str]) -> None:
    wslenv = env.get("WSLENV", "")
    wslenv = _add_wslenv_entry(wslenv, "CG_TUTOR_OUT_DIR/p")
    if "CG_TUTOR_PREVIEW_FRAMES" in env:
        wslenv = _add_wslenv_entry(wslenv, "CG_TUTOR_PREVIEW_FRAMES")
    env["WSLENV"] = wslenv


def _wslpath_windows(path: Path) -> str:
    """Convert a WSL path for a Windows Blender executable.

    WSLENV converts environment values, but command-line arguments such as
    Blender's ``-P scene.py`` are not converted. Passing a Linux absolute path
    to Windows Blender can be interpreted relative to its current UNC working
    directory, producing doubled paths like ``...cg-tutor\\home\\...``.
    """
    resolved = str(path.resolve())
    try:
        proc = subprocess.run(
            ["wslpath", "-w", resolved],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return resolved
    converted = proc.stdout.strip()
    return converted or resolved


def run_script(
    script_path: Path,
    out_dir: Path,
    *,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
    timeout_sec: int = 600,
) -> RunResult:
    """Run a bpy script with `blender -b -P script.py`.

    The script is expected to drive its own rendering and write frames into
    `out_dir`. We pass `out_dir` via env var `CG_TUTOR_OUT_DIR` so the script
    knows where to write without hard-coding paths.

    On timeout: SIGKILL Blender, then return a RunResult with
    returncode=-9. The caller can still use any frames that were written to
    `out_dir` before the kill — partial renders are useful for diagnostics
    and can still be composed into a (shorter) mp4.
    """
    blender = resolve_blender_bin()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CG_TUTOR_OUT_DIR"] = str(out_dir.resolve())
    if env_overrides:
        env.update(env_overrides)
    if _looks_like_windows_exe(blender):
        _prepare_windows_interop_env(env)
        script_arg = _wslpath_windows(script_path)
    else:
        script_arg = str(script_path)

    cmd = [blender, "-b", "--factory-startup", "-P", script_arg]
    if extra_args:
        cmd.extend(extra_args)

    return _run_with_bounded_capture(
        cmd, env=env, timeout_sec=timeout_sec, out_dir=out_dir,
    )


def _bounded_drain(stream, sink: list[bytes], cap: int) -> None:
    """Read from `stream` into `sink` while bounding total bytes to `cap`.

    Streaming equivalent of ``data[-cap:]``: read in small chunks,
    concatenate, then trim to the last ``cap`` bytes after each read.
    This handles the pathological case where a single chunk is larger
    than ``cap`` (we trim mid-chunk instead of dropping the whole thing
    and producing an empty buffer).
    """
    chunk_size = min(64 * 1024, max(cap, 4096))
    data = b""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        data += chunk
        if len(data) > cap:
            data = data[-cap:]
    if data:
        sink.append(data)


def _run_with_bounded_capture(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout_sec: int,
    out_dir: Path,
) -> RunResult:
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    t_out = threading.Thread(
        target=_bounded_drain,
        args=(proc.stdout, stdout_chunks, _MAX_CAPTURED_BYTES),
        daemon=True,
    )
    t_err = threading.Thread(
        target=_bounded_drain,
        args=(proc.stderr, stderr_chunks, _MAX_CAPTURED_BYTES),
        daemon=True,
    )
    t_out.start()
    t_err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    stdout_text = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    if timed_out:
        stderr_text += f"\n[runtime] killed after {timeout_sec}s timeout\n"
        returncode = -9
    else:
        returncode = proc.returncode
    return RunResult(
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        frames_dir=out_dir,
    )
