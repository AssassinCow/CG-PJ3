"""Deterministic pre-render checks for generated Blender scripts.

The vision critic is useful, but it is late and expensive: Blender has
already rendered frames and a VLM call has already happened. This module
implements a small verifier layer for hard pipeline contracts that can be
checked from ``scene.py`` text alone. It is intentionally conservative:
only clear render-breaking or contract-breaking problems are blocks.
Ambiguous visual quality problems stay with the render critic.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from cg_tutor.critic_loop import _missing_storyboard_objects
from cg_tutor.schemas import Storyboard
from cg_tutor.scene_profiles import (
    SceneProfile,
    profile_allows_helper_group,
    profile_forbids_helper_group,
)
from cg_tutor.visual_contract import VisualContract


Severity = Literal["block", "warn"]


@dataclass(frozen=True)
class SceneVerificationIssue:
    severity: Severity
    rule_id: str
    message: str
    suggested_fix: str


@dataclass
class SceneVerificationReport:
    issues: list[SceneVerificationIssue] = field(default_factory=list)
    missing_objects: dict[str, list[str]] = field(default_factory=dict)

    @property
    def block_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "block")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")

    @property
    def ok(self) -> bool:
        return self.block_count == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "block": self.block_count,
            "warn": self.warn_count,
            "issues": [asdict(i) for i in self.issues],
            "missing_objects": self.missing_objects,
        }


_BAD_TYPOGRAPHY = {
    "\u2212": "Unicode minus sign",
    "\u201c": "smart double quote",
    "\u201d": "smart double quote",
    "\u2018": "smart single quote",
    "\u2019": "smart single quote",
    "\uff0c": "fullwidth comma",
    "\uff1a": "fullwidth colon",
}


def _issue(
    severity: Severity,
    rule_id: str,
    message: str,
    suggested_fix: str,
) -> SceneVerificationIssue:
    return SceneVerificationIssue(
        severity=severity,
        rule_id=rule_id,
        message=message,
        suggested_fix=suggested_fix,
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
#
# Earlier versions of this verifier used substring scans (``"import bpy" in
# code``) which an LLM could trivially defeat by mentioning the marker
# inside a comment or docstring. We now parse the script once and ask
# structural questions: "is there an Import node that brings in bpy?",
# "is there a Call to bpy.ops.wm.read_factory_settings?", etc. Comments
# and string literals are part of the AST but in places where ``ast.walk``
# never confuses them with code (string literals show up as Constant
# nodes inside Expression / Assign, never as Import).


def _safe_parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _attr_dotted(node: ast.AST) -> str | None:
    """Return ``bpy.ops.render.render`` for the matching Attribute chain.

    Walks the leftmost Name → Attribute spine. Returns None if the chain
    is rooted in anything other than a bare Name (e.g. function calls)."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))


def _imports_module(tree: ast.Module, name: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Match ``import bpy`` and ``import bpy as X`` and
                # submodule imports like ``import bpy.types``.
                if alias.name == name or alias.name.startswith(name + "."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == name or node.module.startswith(name + "."):
                return True
    return False


def _calls_dotted(tree: ast.Module, dotted: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attr_dotted(node.func)
            if chain == dotted:
                return True
    return False


def _assigns_string_value(
    tree: ast.Module, dotted_suffix: str, *, value: str
) -> bool:
    """Return True iff some assignment writes ``value`` to an attribute
    chain ending in ``dotted_suffix`` (e.g. ``...render.engine``)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or node.value.value != value:
            continue
        for target in node.targets:
            chain = _attr_dotted(target)
            if chain and chain.endswith(dotted_suffix):
                return True
    return False


def _has_cycles_gpu_device_enumeration(code: str) -> bool:
    required_markers = (
        "compute_device_type",
        "get_devices(",
        ".devices",
        ".use",
    )
    return all(marker in code for marker in required_markers) and any(
        backend in code for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")
    )


def _string_constants_contain(tree: ast.Module, needle: str) -> bool:
    """Look only inside string-typed Constant nodes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if needle in node.value:
                return True
    return False


def _formula_like_scene_text_present(
    code: str,
    tree: ast.Module,
    storyboard: Storyboard | None,
    scene_profile: SceneProfile | None,
) -> list[str]:
    if storyboard is None:
        return []
    if scene_profile is not None and scene_profile.base_profile == "cinematic_application":
        return []
    findings: list[str] = []
    lower = code.lower()
    if (
        "formula_panel" in lower
        or "math_overlay" in lower
        or "formula_" in lower
    ):
        findings.append("formula_panel/math_overlay/formula_ marker")
    formulas = [
        shot.formula
        for shot in storyboard.shots
        if shot.formula and shot.overlay_zone is not None
    ]
    for formula in formulas:
        if _string_constants_contain(tree, formula):
            findings.append(f"formula string {formula!r}")
    for value in _string_constants(tree):
        if _looks_like_formula_text(value):
            findings.append(f"formula-like scene text {value!r}")
            if len(findings) >= 4:
                break
    return findings


def _string_constants(tree: ast.Module) -> list[str]:
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
    return values


def _looks_like_formula_text(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 160:
        return False
    # Strings passed to Blender/Python APIs as attribute names, e.g.
    # hasattr(scene.cycles, "use_denoising"), are not visible scene text.
    # Treat bare identifiers as API tokens rather than formula candidates.
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return False
    low = text.lower()
    # Allow short counters/labels such as "V = 26" or "Level 2"; block
    # actual equations that belong in the compositor overlay.
    if re.fullmatch(r"[vef]\s*=\s*\d+", text, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"(level|shape|frame)\s*\d+", text, flags=re.IGNORECASE):
        return False
    markers = (
        "\\frac", "\\sum", "\\sin", "\\cos", "\\theta", "\\mathbf",
        "sum ", " slerp", "cos", "sin", "theta", "omega",
        "v_i", "f_i", "v1", "v2", "f1", "f2", "grad", "∇",
    )
    if any(marker in low for marker in markers):
        return True
    if re.search(r"\bn_(?:\{|[a-z])", low):
        return True
    return bool(
        "=" in text
        and any(op in text for op in ("+", "-", "*", "/", "^", "_", "·"))
        and any(ch.isalpha() for ch in text)
    )


def _calls_render_with_animation_true(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _attr_dotted(node.func) != "bpy.ops.render.render":
            continue
        for kw in node.keywords:
            if kw.arg == "animation" and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    return True
    return False


def _storyboard_duration(storyboard: Storyboard | None) -> float:
    if storyboard is None:
        return 0.0
    return sum(float(shot.duration_sec) for shot in storyboard.shots)


def _storyboard_has_object_keyframes(storyboard: Storyboard | None) -> bool:
    if storyboard is None:
        return False
    for shot in storyboard.shots:
        for obj in shot.objects:
            for key in obj.keyframes:
                if key.attr not in {"hide_render", "hide_viewport"}:
                    return True
    return False


def _scene_has_effective_animation(code: str, tree: ast.Module) -> bool:
    lower = code.lower()
    if "bevel_factor_end" in lower and "keyframe_insert" in lower:
        return True
    if "apply_spec_keyframes(obj, spec)" in code:
        return True
    for line in code.splitlines():
        low = line.lower()
        if "keyframe_insert" not in low:
            continue
        if "hide_render" in low or "hide_viewport" in low:
            continue
        if "cam.keyframe_insert" in low or "camera.keyframe_insert" in low:
            continue
        if any(
            marker in low
            for marker in (
                "'location'", '"location"',
                "'scale'", '"scale"',
                "'rotation_euler'", '"rotation_euler"',
                "'data.energy'", '"data.energy"',
                "'data.color'", '"data.color"',
                "data_path=leaf",
                "data_path='bevel_factor_end'",
                'data_path="bevel_factor_end"',
            )
        ):
            return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_dotted(node.func)
        if not chain or not chain.endswith(".keyframe_insert"):
            continue
        if _is_camera_keyframe_insert(chain):
            continue
        target = None
        if node.args and isinstance(node.args[0], ast.Constant):
            target = node.args[0].value
        for kw in node.keywords:
            if kw.arg == "data_path" and isinstance(kw.value, ast.Constant):
                target = kw.value.value
        if target in {
            "location", "scale", "rotation_euler",
            "data.energy", "data.color", "bevel_factor_end",
        }:
            return True
    return False


def _is_camera_keyframe_insert(chain: str) -> bool:
    root = chain.split(".", 1)[0]
    if root in {"cam", "camera", "render_camera", "main_camera"}:
        return True
    return chain.startswith("bpy.context.scene.camera.")


def _scene_has_camera_motion(code: str) -> bool:
    lower = code.lower()
    return (
        "cam.keyframe_insert" in lower
        or "camera.keyframe_insert" in lower
        or ".timeline_markers.new" in lower
    )


def _has_cycles_denoising_disabled(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            chain = _attr_dotted(target)
            if chain is None:
                continue
            if chain.endswith("use_denoising"):
                if isinstance(node.value, ast.Constant) and node.value.value is False:
                    return True
            if chain.endswith("denoiser"):
                if isinstance(node.value, ast.Constant) and node.value.value == "NONE":
                    return True
    return False


def _animation_severity(scene_profile: SceneProfile | None) -> Severity:
    if scene_profile is None:
        return "warn"
    if scene_profile.base_profile in {
        "vector_teaching",
        "transformation_demo",
        "lighting_decomposition",
    }:
        return "block"
    return "warn"


def _reads_environ_key(tree: ast.Module, key: str) -> bool:
    """Detect ``os.environ['key']`` subscripts and ``os.environ.get('key')``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            chain = _attr_dotted(node.value)
            if chain == "os.environ":
                idx = node.slice
                if isinstance(idx, ast.Constant) and idx.value == key:
                    return True
        elif isinstance(node, ast.Call):
            chain = _attr_dotted(node.func)
            if chain in ("os.environ.get", "os.getenv"):
                if node.args and isinstance(node.args[0], ast.Constant):
                    if node.args[0].value == key:
                        return True
    return False


def _has_unguarded_eevee_use_assignment(code: str) -> bool:
    assignment_re = re.compile(
        r"^\s*[A-Za-z_][\w.]*\.eevee\."
        r"(?P<prop>use_[A-Za-z_][\w]*)\s*=",
    )
    prev = ""
    for line in code.splitlines():
        stripped = line.strip()
        m = assignment_re.match(line)
        if m:
            prop = m.group("prop")
            if "hasattr" not in prev or prop not in prev:
                return True
        if stripped:
            prev = stripped
    return False


def _has_unguarded_material_shadow_assignment(code: str) -> bool:
    assignment_re = re.compile(
        r"^\s*[A-Za-z_][\w.]*\.shadow_method\s*=",
    )
    prev = ""
    for line in code.splitlines():
        stripped = line.strip()
        if assignment_re.match(line):
            if "hasattr" not in prev or "shadow_method" not in prev:
                return True
        if stripped:
            prev = stripped
    return False


def _has_unsafe_action_fcurves_access(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attr_dotted(node)
        if chain is None:
            continue
        if chain.endswith(".animation_data.action.fcurves") or chain.endswith(".action.fcurves"):
            return True
    return False


def _normal_helper_parenting_issues(tree: ast.Module, code: str) -> list[str]:
    """Detect world-space normal helper lines parented to translated empties.

    A common LLM Blender bug is:

        segs = add_dashed_line('surface_normal_seg', world_a, world_b)
        normal = bpy.data.objects.new('surface_normal', None)
        normal.location = entry_point
        for s in segs:
            s.parent = normal

    The dashed-line segments were already created in world coordinates; direct
    parenting to a non-origin empty can offset them away from the surface unless
    matrix_parent_inverse is set. Scope this rule to normal helper geometry so
    we do not block legitimate FK/rig parent-child hierarchies.
    """
    if "matrix_parent_inverse" in code:
        return []

    located_vars: set[str] = set()
    normal_groups: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "location"
            ):
                root = _attr_root(node.targets[0].value)
                if root and not _is_zero_vector_literal(node.value):
                    located_vars.add(root)
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _looks_like_normal_helper_name(node.targets[0].id)
            ):
                normal_groups.add(node.targets[0].id)

    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        iter_name = node.iter.id if isinstance(node.iter, ast.Name) else ""
        if not _looks_like_normal_helper_name(iter_name) and iter_name not in normal_groups:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign):
                continue
            if (
                len(inner.targets) != 1
                or not isinstance(inner.targets[0], ast.Attribute)
                or inner.targets[0].attr != "parent"
                or not isinstance(inner.value, ast.Name)
            ):
                continue
            parent = inner.value.id
            if parent not in located_vars:
                continue
            if not _looks_like_normal_helper_name(parent):
                continue
            findings.append(f"{iter_name} parented to translated {parent}")
    return sorted(set(findings))


def _attr_root(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _looks_like_normal_helper_name(name: str) -> bool:
    low = name.lower()
    return "normal" in low and any(
        marker in low for marker in ("surface", "seg", "line", "helper", "entry", "exit")
    )


def _is_zero_vector_literal(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) != 3:
        return False
    for elt in node.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, (int, float)):
            return False
        if abs(float(elt.value)) > 1e-9:
            return False
    return True


def verify_scene_code(
    code: str,
    storyboard: Storyboard | None = None,
    visual_contracts: dict[str, VisualContract] | None = None,
    scene_profile: SceneProfile | None = None,
    render_engine: str = "BLENDER_EEVEE",
    cycles_device: str = "AUTO",
) -> SceneVerificationReport:
    """Check generated ``scene.py`` before spending Blender/VLM time."""
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be one of: BLENDER_EEVEE, CYCLES")
    if cycles_device not in {"AUTO", "GPU", "CPU"}:
        raise ValueError("cycles_device must be one of: AUTO, GPU, CPU")
    issues: list[SceneVerificationIssue] = []
    stripped = code.lstrip()

    if stripped.startswith("```"):
        issues.append(_issue(
            "block",
            "markdown_fence",
            "scene.py still starts with a Markdown code fence.",
            "Output only runnable Python. Remove all ``` fences and prose.",
        ))

    for ch, label in _BAD_TYPOGRAPHY.items():
        if ch in code:
            issues.append(_issue(
                "block",
                "invalid_typography",
                f"scene.py contains {label} {ch!r}, which can break Python.",
                "Use ASCII Python syntax: '-', straight quotes, commas, and colons.",
            ))
            break

    tree = _safe_parse(code)
    if tree is None:
        try:
            compile(code, "scene.py", "exec")
        except SyntaxError as e:
            loc = f"line {e.lineno}" if e.lineno else "unknown line"
            issues.append(_issue(
                "block",
                "syntax_error",
                f"scene.py does not compile ({loc}): {e.msg}",
                "Return a complete syntactically valid Python script.",
            ))
        # If the script doesn't parse we can't reason about the rest of
        # the structural checks; surface the failures we have and return.
        return SceneVerificationReport(issues=issues)

    if not _imports_module(tree, "bpy"):
        issues.append(_issue(
            "block",
            "missing_bpy_import",
            "scene.py does not import bpy.",
            "Add `import bpy` at the top of the script.",
        ))

    if not _calls_dotted(tree, "bpy.ops.wm.read_factory_settings"):
        issues.append(_issue(
            "block",
            "missing_factory_reset",
            "scene.py does not reset Blender factory settings.",
            "Start with `bpy.ops.wm.read_factory_settings(use_empty=True)`.",
        ))

    if _assigns_string_value(tree, "render.engine", value="BLENDER_EEVEE_NEXT"):
        issues.append(_issue(
            "block",
            "forbidden_eevee_next",
            "scene.py uses BLENDER_EEVEE_NEXT, which is unavailable in Blender 4.0.",
            "Use `scene.render.engine = 'BLENDER_EEVEE'`.",
        ))

    if render_engine == "BLENDER_EEVEE" and _assigns_string_value(
        tree, "render.engine", value="CYCLES",
    ):
        issues.append(_issue(
            "block",
            "forbidden_cycles",
            "scene.py uses CYCLES, which is too slow for this pipeline.",
            "Use `scene.render.engine = 'BLENDER_EEVEE'` with low sample counts.",
        ))

    # Use AST: only count a *string-typed* assignment to a `render.engine`
    # attribute as evidence. This avoids matching a comment that mentions
    # BLENDER_EEVEE without actually setting it.
    if not _assigns_string_value(tree, "render.engine", value=render_engine):
        label = "BLENDER_EEVEE" if render_engine == "BLENDER_EEVEE" else "CYCLES"
        issues.append(_issue(
            "block",
            "missing_eevee" if render_engine == "BLENDER_EEVEE" else "missing_cycles",
            f"scene.py does not explicitly select {label}.",
            f"Set `bpy.context.scene.render.engine = '{label}'`.",
        ))
    if (
        render_engine == "CYCLES"
        and cycles_device in {"AUTO", "GPU"}
        and not _assigns_string_value(tree, "cycles.device", value="GPU")
    ):
        issues.append(_issue(
            "block",
            "missing_cycles_gpu_device",
            "scene.py selects CYCLES but does not request GPU rendering.",
            "Set `bpy.context.scene.cycles.device = 'GPU'` and enable "
            "available non-CPU Cycles addon devices before rendering.",
        ))
    if (
        render_engine == "CYCLES"
        and cycles_device in {"AUTO", "GPU"}
        and _assigns_string_value(tree, "cycles.device", value="GPU")
        and not _has_cycles_gpu_device_enumeration(code)
    ):
        issues.append(_issue(
            "warn",
            "cycles_gpu_device_enumeration_missing",
            "scene.py requests Cycles GPU but does not enumerate/enable "
            "Blender Cycles addon devices, so Blender may silently render "
            "on CPU.",
            "Configure Cycles preferences: set compute_device_type "
            "(OPTIX/CUDA/HIP/METAL/ONEAPI), call get_devices(), enable "
                "non-CPU devices, and disable CPU devices when a GPU exists.",
        ))
    if render_engine == "CYCLES" and not _has_cycles_denoising_disabled(tree):
        issues.append(_issue(
            "block",
            "missing_cycles_denoising_disabled",
            "scene.py selects CYCLES but does not disable denoising. This "
            "Blender build may not include OpenImageDenoiser, causing preview "
            "render to fail before writing frames.",
            "Set `scene.cycles.use_denoising = False` when available and "
            "also disable `view_layer.cycles.use_denoising` for every view "
            "layer before rendering.",
        ))

    if _has_unguarded_eevee_use_assignment(code):
        issues.append(_issue(
            "block",
            "unguarded_eevee_use_property",
            "scene.py writes EEVEE use_* properties without an API guard.",
            "Wrap `scene.eevee.use_*` assignments in "
            "`if hasattr(scene.eevee, '<property>'):` for Blender 5.x.",
        ))

    if _has_unguarded_material_shadow_assignment(code):
        issues.append(_issue(
            "block",
            "unguarded_material_shadow_method",
            "scene.py writes Material.shadow_method without an API guard.",
            "Wrap `mat.shadow_method = ...` in "
            "`if hasattr(mat, 'shadow_method'):` for Blender 5.x.",
        ))

    if _has_unsafe_action_fcurves_access(tree):
        issues.append(_issue(
            "block",
            "unsafe_action_fcurves_access",
            "scene.py directly reads animation_data.action.fcurves, which is "
            "not compatible with Blender 5.x Actions and can crash preview.",
            "Do not introspect Action.fcurves in scene.py. Track your own "
            "keyframe values in Python variables, or guard any animation_data "
            "inspection with `hasattr(action, 'fcurves')` before access.",
        ))

    normal_parenting = _normal_helper_parenting_issues(tree, code)
    if normal_parenting:
        issues.append(_issue(
            "block",
            "normal_helper_parented_without_inverse",
            (
                "scene.py appears to create surface-normal helper line "
                "segments in world coordinates and then parent them to a "
                "translated normal object, which can visibly displace the "
                "normal away from the prism/surface contact point: "
                + "; ".join(normal_parenting[:3])
            ),
            (
                "For surface normals and dashed helper lines, either keep the "
                "segments unparented in world coordinates, or create them in "
                "local coordinates under the parent, or set "
                "`child.matrix_parent_inverse = parent.matrix_world.inverted()` "
                "immediately after assigning `child.parent = parent`."
            ),
        ))

    formula_findings = _formula_like_scene_text_present(
        code, tree, storyboard, scene_profile,
    )
    if formula_findings:
        issues.append(_issue(
            "block",
            "formula_text_in_scene",
            (
                "Teaching formulas must be rendered only by the LaTeX/ffmpeg "
                "overlay stage, but scene.py appears to create formula text: "
                + ", ".join(formula_findings[:4])
            ),
            (
                "Remove formula_panel/math text from scene.py. Keep "
                "shot.formula and shot.overlay_zone in storyboard; the "
                "composer will add readable formula overlays after rendering."
            ),
        ))

    if _calls_dotted(tree, "bpy.ops.wm.save_as_mainfile"):
        issues.append(_issue(
            "block",
            "forbidden_save_file",
            "scene.py calls save_as_mainfile, which is forbidden.",
            "Remove all `.blend` file saving; render PNG frames only.",
        ))

    if not _reads_environ_key(tree, "CG_TUTOR_OUT_DIR"):
        issues.append(_issue(
            "block",
            "missing_out_dir_env",
            "scene.py does not read CG_TUTOR_OUT_DIR.",
            "Read `out_dir = os.environ['CG_TUTOR_OUT_DIR']` and render frames there.",
        ))

    if not _calls_dotted(tree, "bpy.ops.render.render"):
        issues.append(_issue(
            "block",
            "missing_render_call",
            "scene.py does not call bpy.ops.render.render.",
            "End the script with `bpy.ops.render.render(animation=True)`.",
        ))
    elif not _calls_render_with_animation_true(tree):
        issues.append(_issue(
            "block",
            "render_not_animation",
            "scene.py calls render, but not with animation=True.",
            "Use `bpy.ops.render.render(animation=True)` so every frame is written.",
        ))

    if re.search(r"\.data\.angle[^\n]*keyframe_insert|keyframe_insert\s*\([^)]*['\"](?:data\.)?angle['\"]", code):
        issues.append(_issue(
            "block",
            "animates_camera_angle",
            "scene.py attempts to keyframe camera.data.angle, which is not animatable.",
            "Animate camera.data.lens instead, or keep FOV constant.",
        ))

    if "frame_####.png" not in code and "####" not in code:
        issues.append(_issue(
            "warn",
            "missing_frame_pattern",
            "scene.py does not visibly use a frame_####.png output pattern.",
            "Set `scene.render.filepath = os.path.join(out_dir, 'frame_####.png')`.",
        ))

    if "frame_start" not in code or "frame_end" not in code:
        issues.append(_issue(
            "warn",
            "missing_frame_range",
            "scene.py does not explicitly set frame_start/frame_end.",
            "Set frame_start=1 and frame_end=round(total_duration * fps).",
        ))

    has_effective_animation = _scene_has_effective_animation(code, tree)
    if _storyboard_has_object_keyframes(storyboard) and not has_effective_animation:
        issues.append(_issue(
            "block",
            "missing_storyboard_animation",
            "Storyboard contains object keyframes, but scene.py does not implement non-visibility object animation.",
            "Apply storyboard object keyframes with `obj.keyframe_insert(...)` or animate curve reveal with `data.bevel_factor_end`.",
        ))
    elif (
        storyboard is not None
        and _storyboard_duration(storyboard) > 2.0
        and not has_effective_animation
    ):
        severity = _animation_severity(scene_profile)
        issues.append(_issue(
            severity,
            "insufficient_scene_animation",
            (
                "scene.py appears static for a multi-second storyboard: it has "
                "no non-visibility object/ray/light animation."
            ),
            (
                "Add object/light keyframes or curve reveal animation; "
                "hide_render/hide_viewport shot gating does not count as motion."
            ),
        ))

    missing_objects = (
        _missing_storyboard_objects(code, storyboard) if storyboard is not None
        else {}
    )
    if missing_objects:
        total = sum(len(v) for v in missing_objects.values())
        issues.append(_issue(
            "warn",
            "missing_storyboard_objects",
            f"{total} storyboard object name(s) are absent from scene.py.",
            "Instantiate the missing objects with exact `obj.name = '<name>'` names.",
        ))

    is_cinematic = (
        scene_profile is not None
        and scene_profile.base_profile == "cinematic_application"
    )
    no_drawn_rays = profile_forbids_helper_group(scene_profile, "drawn_rays")
    forbid_arrows = profile_forbids_helper_group(scene_profile, "arrow_helpers")
    forbid_light_gizmos = profile_forbids_helper_group(
        scene_profile, "visible_light_gizmos",
    )
    prefer_thin_tracers = (
        profile_allows_helper_group(scene_profile, "drawn_rays")
        and not no_drawn_rays
    )
    lower_code = code.lower()
    if forbid_arrows:
        forbidden_arrow_markers = (
            "add_arrow",
            "create_arrow",
            "primitive_arrow",
            "arrowhead",
            "arrow_head",
        )
        if any(marker in lower_code for marker in forbidden_arrow_markers):
            issues.append(_issue(
                "block",
                "profile_forbidden_arrow",
                "The active scene profile forbids arrow helper geometry.",
                (
                    "Remove drawn ray/arrow helpers and show the optical effect through physical surfaces, shadows, and highlights."
                    if no_drawn_rays
                    else "Use thin curve_polyline / bevel_depth tracer lines or subtle dotted HUD paths instead of arrow shafts/heads."
                ),
            ))
    if forbid_light_gizmos:
        forbidden_light_markers = (
            "area_light_rect",
            "light_bar",
            "light_panel",
            "foreground_light",
            "softbox_panel",
        )
        if any(marker in lower_code for marker in forbidden_light_markers):
            issues.append(_issue(
                "block",
                "profile_forbidden_light_gizmo",
                "The active scene profile forbids visible light bars or foreground light panels.",
                "Keep Blender light objects invisible to camera and avoid mesh panels used as main visual light sources.",
            ))

    if visual_contracts:
        text_markers = (
            "bpy.ops.object.text_add",
            "bpy.ops.object.text_add(",
            "FONT",
            ".data.body",
        )
        has_text = any(marker in code for marker in text_markers)
        arrow_markers = (
            "primitive_cone_add",
            "primitive_cylinder_add",
            "primitive_arrow",
            "add_arrow",
            "create_arrow",
        )
        arrow_marker_count = sum(code.count(marker) for marker in arrow_markers)
        for shot_id, contract in sorted(visual_contracts.items()):
            if contract.required_labels and not has_text:
                issues.append(_issue(
                    "warn",
                    "missing_required_labels",
                    f"{shot_id} requires visible labels, but scene.py does not appear to create text objects.",
                    "Create readable Blender text labels, face them toward the camera, and name them explicitly.",
                ))
            if contract.required_vectors:
                if no_drawn_rays:
                    continue
                if is_cinematic or prefer_thin_tracers or forbid_arrows:
                    line_markers = (
                        "curve_polyline",
                        "bevel_depth",
                        "polyline",
                        "tracer",
                        "dotted",
                    )
                    if not any(marker in lower_code for marker in line_markers):
                        issues.append(_issue(
                            "warn",
                            "missing_profile_tracer_geometry",
                            f"{shot_id} requires ray/vector cues, but scene.py has no obvious thin tracer/curve geometry.",
                            "Represent ray cues as thin glowing curve_polyline or dotted tracer paths, not arrow geometry.",
                        ))
                elif arrow_marker_count == 0:
                    issues.append(_issue(
                        "warn",
                        "missing_vector_geometry",
                        f"{shot_id} requires vectors/arrows, but scene.py does not appear to create arrow geometry.",
                        "Represent each vector as a visible shaft and head, not a tiny cone or color patch.",
                    ))
            if contract.emphasis_points and not any(
                word in lower_code for word in ("highlight", "emission", "specular", "roughness")
            ):
                issues.append(_issue(
                    "warn",
                    "missing_emphasis_cue",
                    f"{shot_id} requires a visible emphasis point/highlight, but scene.py has no obvious highlight material cue.",
                    "Add a small bright marker or glossy material cue at the emphasized point.",
                ))

    return SceneVerificationReport(
        issues=issues,
        missing_objects=missing_objects,
    )


def format_verifier_addendum(report: SceneVerificationReport) -> str:
    """Turn verifier failures into a compact Coder retry prompt."""
    if not report.issues:
        return ""
    lines = [
        "SCENE VERIFIER FAILED BEFORE BLENDER RENDER:",
        "Fix these deterministic code/contract issues before changing visual style.",
    ]
    for i, issue in enumerate(report.issues, 1):
        lines.append(
            f"{i}. [{issue.severity} {issue.rule_id}] {issue.message}"
        )
        lines.append(f"   fix: {issue.suggested_fix}")
    if report.missing_objects:
        lines.append("")
        lines.append("Missing storyboard object names:")
        for shot_id, names in sorted(report.missing_objects.items()):
            lines.append(f"- {shot_id}: {', '.join(names)}")
    return "\n".join(lines)


def report_to_json(report: SceneVerificationReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
