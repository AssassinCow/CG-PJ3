import pytest

from cg_tutor.agents.blender_coder import (
    PatchApplyError,
    _render_engine_policy,
    apply_search_replace_blocks,
    apply_unified_diff,
    compatibilize_blender_code,
    normalize_python_text,
)


def test_normalize_python_text_replaces_llm_typography():
    raw = "pos = [0,\u22126,1.5]\nlabel = \u201cPhong\u201d\ncenter\uff1a tuple\uff081, 2\uff09"

    assert normalize_python_text(raw) == (
        'pos = [0,-6,1.5]\nlabel = "Phong"\ncenter: tuple(1, 2)'
    )


def test_compatibilize_blender_code_guards_eevee_use_properties():
    code = (
        "scene = bpy.context.scene\n"
        "scene.eevee.use_ssr = True\n"
        "scene.eevee.use_ssr_refraction = False\n"
        "scene.eevee.use_motion_blur = False\n"
    )

    fixed = compatibilize_blender_code(code)

    assert "if hasattr(scene.eevee, 'use_ssr'):" in fixed
    assert "if hasattr(scene.eevee, 'use_ssr_refraction'):" in fixed
    assert "if hasattr(scene.eevee, 'use_motion_blur'):" in fixed
    assert "    scene.eevee.use_ssr = True" in fixed


def test_compatibilize_blender_code_guards_material_shadow_method():
    fixed = compatibilize_blender_code("mat.shadow_method = 'HASHED'\n")

    assert "if hasattr(mat, 'shadow_method'):" in fixed
    assert "    mat.shadow_method = 'HASHED'" in fixed


def test_compatibilize_blender_code_preserves_unc_render_filepath():
    fixed = compatibilize_blender_code(
        "scene.render.filepath = os.path.join(out_dir, 'frame_####.png').replace('\\\\', '/')\n"
    )

    assert "_render_path = os.path.join(out_dir, 'frame_####.png')" in fixed
    assert "if not _render_path.startswith('\\\\\\\\'):" in fixed
    assert "scene.render.filepath = _render_path" in fixed


def test_compatibilize_blender_code_falls_back_to_agx_view_look():
    fixed = compatibilize_blender_code(
        "scene.view_settings.look = 'Medium High Contrast'\n"
    )

    assert "try:" in fixed
    assert "scene.view_settings.look = 'Medium High Contrast'" in fixed
    assert "except TypeError:" in fixed
    assert "scene.view_settings.look = 'AgX - Medium High Contrast'" in fixed


def test_compatibilize_blender_code_disables_cycles_denoising():
    fixed = compatibilize_blender_code(
        "scene = bpy.context.scene\n"
        "scene.render.engine = 'CYCLES'\n"
        "scene.cycles.samples = 32\n"
    )

    assert "scene.cycles.use_denoising = False" in fixed
    assert "_cg_view_layer.cycles.use_denoising = False" in fixed


def test_render_engine_policy_requests_cycles_gpu_setup():
    text = _render_engine_policy("CYCLES", "AUTO")

    assert "Requested Cycles device: GPU" in text
    assert "OPTIX/CUDA/HIP/METAL/ONEAPI" in text
    assert "scene.cycles.device = 'GPU'" in text
    assert "use_denoising = False" in text
    assert "action.fcurves" in text


def test_apply_unified_diff_applies_single_file_patch():
    base = "a = 1\nb = 2\nc = 3\n"
    diff = """--- scene.py
+++ scene.py
@@ -1,3 +1,4 @@
 a = 1
-b = 2
+b = 20
+bb = 22
 c = 3
"""

    assert apply_unified_diff(base, diff) == "a = 1\nb = 20\nbb = 22\nc = 3\n"


def test_apply_unified_diff_rejects_context_mismatch():
    base = "a = 1\nb = 2\n"
    diff = """--- scene.py
+++ scene.py
@@ -1,2 +1,2 @@
 a = 999
-b = 2
+b = 3
"""

    with pytest.raises(PatchApplyError):
        apply_unified_diff(base, diff)


def test_apply_search_replace_basic():
    base = "a = 1\nb = 2\nc = 3\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "b = 2\n"
        "=======\n"
        "b = 20\n"
        "bb = 22\n"
        ">>>>>>> REPLACE\n"
    )
    assert apply_search_replace_blocks(base, patch) == (
        "a = 1\nb = 20\nbb = 22\nc = 3\n"
    )


def test_apply_search_replace_multi_block():
    base = "a = 1\nb = 2\nc = 3\nd = 4\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "=======\n"
        "a = 10\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "d = 4\n"
        "=======\n"
        "d = 40\n"
        ">>>>>>> REPLACE\n"
    )
    assert apply_search_replace_blocks(base, patch) == (
        "a = 10\nb = 2\nc = 3\nd = 40\n"
    )


def test_apply_search_replace_rejects_ambiguous():
    base = "x = 1\ny = 1\nx = 1\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "x = 1\n"
        "=======\n"
        "x = 99\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(PatchApplyError, match="matches 2 places"):
        apply_search_replace_blocks(base, patch)


def test_apply_search_replace_rejects_missing():
    base = "a = 1\nb = 2\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "nonexistent_line\n"
        "=======\n"
        "anything\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(PatchApplyError, match="not found"):
        apply_search_replace_blocks(base, patch)


def test_apply_search_replace_tolerates_trailing_whitespace():
    # base has a line with trailing spaces; SEARCH does not.
    base = "def f():\n    return 1   \n"
    patch = (
        "<<<<<<< SEARCH\n"
        "    return 1\n"
        "=======\n"
        "    return 2\n"
        ">>>>>>> REPLACE\n"
    )
    assert apply_search_replace_blocks(base, patch) == "def f():\n    return 2\n"


def test_apply_search_replace_empty_replace_deletes():
    base = "keep_me\ndrop_me\nkeep_too\n"
    patch = (
        "<<<<<<< SEARCH\n"
        "drop_me\n"
        "=======\n"
        ">>>>>>> REPLACE\n"
    )
    assert apply_search_replace_blocks(base, patch) == "keep_me\nkeep_too\n"


def test_apply_search_replace_requires_at_least_one_block():
    with pytest.raises(PatchApplyError, match="no search/replace blocks"):
        apply_search_replace_blocks("a = 1\n", "no markers here")
