from cg_tutor import terminal_ui as ui


def test_terminal_ui_formats_stage_and_details() -> None:
    rule = "=" * 72
    assert ui.rule() == rule
    assert ui.step(3, 7, "Blender coder") == f"\n{rule}\n[3/7] Blender coder"
    assert ui.step(2, 7, "Storyboard", "cached") == (
        f"\n{rule}\n[2/7] Storyboard  [cached]"
    )
    assert ui.iter_header(4, 5) == f"\n{rule}\niter04 / iter05"
    assert ui.detail("critic result", "score=0.88") == "    - critic result      score=0.88"


def test_terminal_ui_status_lines_are_plain_ascii() -> None:
    assert ui.ok("rendered 480 frames") == "    OK  rendered 480 frames"
    assert ui.warn("preview has block issue(s)") == "    WARN preview has block issue(s)"
    assert ui.fail("frames=0") == "    FAIL frames=0"
    assert ui.done("in 1.0s - final.mp4") == (
        f"\n{'=' * 72}\nDONE in 1.0s - final.mp4"
    )
