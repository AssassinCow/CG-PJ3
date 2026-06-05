from cg_tutor.blender.runtime import (
    _add_wslenv_entry,
    _prepare_windows_interop_env,
)
from cg_tutor.blender import runtime as blender_runtime


def test_add_wslenv_entry_preserves_existing_entries():
    assert (
        _add_wslenv_entry("PATH/l:CG_TUTOR_OUT_DIR/p", "CG_TUTOR_OUT_DIR/p")
        == "PATH/l:CG_TUTOR_OUT_DIR/p"
    )
    assert (
        _add_wslenv_entry("PATH/l", "CG_TUTOR_PREVIEW_FRAMES")
        == "PATH/l:CG_TUTOR_PREVIEW_FRAMES"
    )


def test_prepare_windows_interop_env_exports_render_vars():
    env = {
        "WSLENV": "PATH/l",
        "CG_TUTOR_OUT_DIR": "/home/me/out",
        "CG_TUTOR_PREVIEW_FRAMES": "1,2,3",
    }

    _prepare_windows_interop_env(env)

    assert "CG_TUTOR_OUT_DIR/p" in env["WSLENV"].split(":")
    assert "CG_TUTOR_PREVIEW_FRAMES" in env["WSLENV"].split(":")


def test_run_script_converts_script_arg_for_windows_blender(tmp_path, monkeypatch):
    blender = "/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
    monkeypatch.setattr(blender_runtime, "resolve_blender_bin", lambda: blender)
    monkeypatch.setattr(
        blender_runtime,
        "_wslpath_windows",
        lambda path: r"\\wsl.localhost\Ubuntu\home\me\scene.py",
    )

    seen = {}

    def fake_run(cmd, *, env, timeout_sec, out_dir):
        seen["cmd"] = cmd
        seen["env"] = env
        return blender_runtime.RunResult(
            returncode=0, stdout="", stderr="", frames_dir=out_dir,
        )

    monkeypatch.setattr(blender_runtime, "_run_with_bounded_capture", fake_run)

    blender_runtime.run_script(tmp_path / "scene.py", tmp_path / "frames")

    assert seen["cmd"][:5] == [
        blender,
        "-b",
        "--factory-startup",
        "-P",
        r"\\wsl.localhost\Ubuntu\home\me\scene.py",
    ]
    assert "CG_TUTOR_OUT_DIR/p" in seen["env"]["WSLENV"].split(":")
