"""Blender Coder Agent.

Storyboard → self-contained bpy 4.0 Python script. The script reads
CG_TUTOR_OUT_DIR from env and renders PNG frames there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from cg_tutor.agents.base import load_prompt, save_artifact, strip_code_fence
from cg_tutor.config import get_agent_model
from cg_tutor.llm_client import LLMClient
from cg_tutor.schemas import Storyboard


AGENT = "blender_coder"


_PYTHON_TEXT_TRANSLATION = str.maketrans({
    "\u2212": "-",   # minus sign
    "\u2010": "-",   # hyphen
    "\u2011": "-",   # non-breaking hyphen
    "\u2012": "-",   # figure dash
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\uff0c": ",",   # fullwidth comma
    "\uff1a": ":",   # fullwidth colon
    "\uff08": "(",   # fullwidth left paren
    "\uff09": ")",   # fullwidth right paren
})


def normalize_python_text(code: str) -> str:
    """Normalize LLM typography that is invalid or risky in Python code."""
    return code.translate(_PYTHON_TEXT_TRANSLATION)


def compatibilize_blender_code(code: str) -> str:
    """Patch known Blender API drift in generated scripts.

    Blender 5.x removed some Blender 4.0 EEVEE attributes. The prompt still
    targets the older API for broad compatibility, so guard those assignments
    before runtime instead of letting a valid scene fail with AttributeError.
    """
    out: list[str] = []
    guarded_assignment_re = re.compile(
        r"^(?P<indent>\s*)(?P<target>[A-Za-z_][\w.]*\.eevee)"
        r"\.(?P<prop>use_[A-Za-z_][\w]*)\s*=\s*(?P<value>.+)$"
    )
    material_shadow_re = re.compile(
        r"^(?P<indent>\s*)(?P<target>[A-Za-z_][\w.]*)"
        r"\.(?P<prop>shadow_method)\s*=\s*(?P<value>.+)$"
    )
    filepath_replace_re = re.compile(
        r"^(?P<indent>\s*)scene\.render\.filepath\s*=\s*"
        r"(?P<path_expr>.+)\.replace\(['\"]\\\\['\"],\s*['\"]/['\"]\)$"
    )
    view_look_re = re.compile(
        r"^(?P<indent>\s*)scene\.view_settings\.look\s*=\s*"
        r"(?P<quote>['\"])(?P<look>Medium High Contrast|High Contrast|"
        r"Very High Contrast|Base Contrast|Medium Low Contrast|Low Contrast|"
        r"Very Low Contrast)(?P=quote)$"
    )
    cycles_engine_re = re.compile(
        r"^(?P<indent>\s*)(?P<scene>[A-Za-z_][\w.]*"
        r"(?:\.context\.scene)?)\.render\.engine\s*=\s*['\"]CYCLES['\"]\s*$"
    )
    has_denoising_guard = (
        "use_denoising = False" in code
        or "denoiser = 'NONE'" in code
        or 'denoiser = "NONE"' in code
    )
    for line in code.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        newline = line[len(raw):]
        cycles_match = cycles_engine_re.match(raw)
        if cycles_match and not has_denoising_guard:
            indent = cycles_match.group("indent")
            scene_expr = cycles_match.group("scene")
            out.append(line)
            out.extend(_cycles_denoising_disable_lines(indent, scene_expr, newline))
            has_denoising_guard = True
            continue
        look_match = view_look_re.match(raw)
        if look_match:
            indent = look_match.group("indent")
            look = look_match.group("look")
            quote = look_match.group("quote")
            agx_look = f"AgX - {look}"
            out.append(f"{indent}try:{newline}")
            out.append(f"{indent}    scene.view_settings.look = {quote}{look}{quote}{newline}")
            out.append(f"{indent}except TypeError:{newline}")
            out.append(f"{indent}    scene.view_settings.look = {quote}{agx_look}{quote}{newline}")
            continue
        fp = filepath_replace_re.match(raw)
        if fp:
            indent = fp.group("indent")
            path_expr = fp.group("path_expr")
            out.append(f"{indent}_render_path = {path_expr}{newline}")
            out.append(f"{indent}if not _render_path.startswith('\\\\\\\\'):{newline}")
            out.append(f"{indent}    _render_path = _render_path.replace('\\\\', '/'){newline}")
            out.append(f"{indent}scene.render.filepath = _render_path{newline}")
            continue
        m = guarded_assignment_re.match(raw) or material_shadow_re.match(raw)
        if not m:
            out.append(line)
            continue
        prop = m.group("prop")
        target = m.group("target")
        prev = out[-1].strip() if out else ""
        if "hasattr" in prev and prop in prev:
            out.append(line)
            continue
        indent = m.group("indent")
        value = m.group("value")
        out.append(f"{indent}if hasattr({target}, '{prop}'):{newline}")
        out.append(f"{indent}    {target}.{prop} = {value}{newline}")
    return "".join(out)


def _cycles_denoising_disable_lines(
    indent: str,
    scene_expr: str,
    newline: str,
) -> list[str]:
    return [
        f"{indent}if hasattr({scene_expr}, 'cycles'):{newline}",
        f"{indent}    if hasattr({scene_expr}.cycles, 'use_denoising'):{newline}",
        f"{indent}        {scene_expr}.cycles.use_denoising = False{newline}",
        f"{indent}    if hasattr({scene_expr}.cycles, 'denoiser'):{newline}",
        f"{indent}        try:{newline}",
        f"{indent}            {scene_expr}.cycles.denoiser = 'NONE'{newline}",
        f"{indent}        except Exception:{newline}",
        f"{indent}            pass{newline}",
        f"{indent}for _cg_view_layer in {scene_expr}.view_layers:{newline}",
        f"{indent}    if hasattr(_cg_view_layer, 'cycles') and hasattr(_cg_view_layer.cycles, 'use_denoising'):{newline}",
        f"{indent}        _cg_view_layer.cycles.use_denoising = False{newline}",
    ]


class PatchApplyError(RuntimeError):
    pass


def _strip_diff_fence(text: str) -> str:
    text = strip_code_fence(text).strip()
    if text.startswith("diff --git"):
        return text
    start = text.find("--- ")
    if start > 0:
        text = text[start:]
    return text


def apply_unified_diff(base: str, diff_text: str) -> str:
    """Apply a simple single-file unified diff to ``base``.

    This intentionally supports the subset we ask LLMs to emit. On any
    mismatch we raise and let the pipeline fall back to complete-script
    regeneration.
    """
    diff_text = _strip_diff_fence(diff_text)
    lines = base.splitlines(keepends=True)
    diff = diff_text.splitlines(keepends=True)
    hunks: list[tuple[int, list[str]]] = []
    i = 0
    while i < len(diff):
        line = diff[i]
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if not m:
                raise PatchApplyError(f"unsupported hunk header: {line.strip()}")
            old_start = int(m.group(1)) - 1
            body: list[str] = []
            i += 1
            while i < len(diff) and not diff[i].startswith("@@"):
                if diff[i].startswith(("--- ", "+++ ")):
                    i += 1
                    continue
                body.append(diff[i])
                i += 1
            hunks.append((old_start, body))
            continue
        i += 1
    if not hunks:
        raise PatchApplyError("no unified-diff hunks found")

    out: list[str] = []
    cursor = 0
    def _same_line(a: str, b: str) -> bool:
        return a == b or a.rstrip("\n") == b.rstrip("\n")

    for old_start, body in hunks:
        if old_start < cursor:
            raise PatchApplyError("overlapping hunks")
        out.extend(lines[cursor:old_start])
        cursor = old_start
        for raw in body:
            if raw.startswith("\\ No newline at end of file"):
                continue
            if not raw:
                continue
            tag = raw[0]
            payload = raw[1:]
            if tag == " ":
                if cursor >= len(lines) or not _same_line(lines[cursor], payload):
                    got = lines[cursor].rstrip() if cursor < len(lines) else "<eof>"
                    raise PatchApplyError(
                        f"context mismatch at line {cursor + 1}: expected {payload.rstrip()!r}, got {got!r}"
                    )
                out.append(lines[cursor])
                cursor += 1
            elif tag == "-":
                if cursor >= len(lines) or not _same_line(lines[cursor], payload):
                    got = lines[cursor].rstrip() if cursor < len(lines) else "<eof>"
                    raise PatchApplyError(
                        f"delete mismatch at line {cursor + 1}: expected {payload.rstrip()!r}, got {got!r}"
                    )
                cursor += 1
            elif tag == "+":
                out.append(payload)
            else:
                raise PatchApplyError(f"bad diff line: {raw.rstrip()!r}")
    out.extend(lines[cursor:])
    return "".join(out)


_SR_BLOCK_RE = re.compile(
    r"^[ \t]*<{5,}[ \t]*SEARCH[ \t]*\r?\n"
    r"(?P<search>.*?)"
    r"^[ \t]*={5,}[ \t]*\r?\n"
    r"(?P<replace>.*?)"
    r"^[ \t]*>{5,}[ \t]*REPLACE[ \t]*\r?\n?",
    re.DOTALL | re.MULTILINE,
)


def _parse_sr_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for m in _SR_BLOCK_RE.finditer(text):
        blocks.append((m.group("search"), m.group("replace")))
    return blocks


def apply_search_replace_blocks(base: str, text: str) -> str:
    """Apply Aider-style SEARCH/REPLACE blocks to ``base``.

    Each block looks like::

        <<<<<<< SEARCH
        ...verbatim slice of base...
        =======
        ...replacement lines...
        >>>>>>> REPLACE

    The SEARCH side must match exactly ONE location in the current scene
    (after prior blocks have been applied). Trailing whitespace on each
    line is ignored when matching; leading whitespace (indentation) must
    match exactly because Python depends on it.
    """
    blocks = _parse_sr_blocks(text)
    if not blocks:
        raise PatchApplyError("no search/replace blocks found")
    current = base
    for idx, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            raise PatchApplyError(f"block #{idx}: empty SEARCH")
        current = _apply_one_sr(current, search, replace, idx)
    return current


def _apply_one_sr(current: str, search: str, replace: str, idx: int) -> str:
    if current.count(search) == 1:
        return current.replace(search, replace, 1)
    cur_lines = current.splitlines()
    s_lines = search.splitlines()
    r_lines = replace.splitlines()
    n = len(s_lines)
    if n == 0:
        raise PatchApplyError(f"block #{idx}: empty SEARCH")
    s_norm = [ln.rstrip() for ln in s_lines]
    matches: list[int] = []
    for i in range(len(cur_lines) - n + 1):
        window = [ln.rstrip() for ln in cur_lines[i:i + n]]
        if window == s_norm:
            matches.append(i)
    if not matches:
        first = s_lines[0][:80].rstrip()
        raise PatchApplyError(
            f"block #{idx}: SEARCH not found (first line: {first!r})"
        )
    if len(matches) > 1:
        raise PatchApplyError(
            f"block #{idx}: SEARCH matches {len(matches)} places — "
            "add more surrounding context to make it unique"
        )
    i = matches[0]
    out_lines = cur_lines[:i] + r_lines + cur_lines[i + n:]
    result = "\n".join(out_lines)
    if current.endswith("\n"):
        result += "\n"
    return result


def to_bpy_script(
    storyboard: Storyboard,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
    addendum: str = "",
    base_scene: str = "",
    iteration: int = 0,
    render_engine: str = "BLENDER_EEVEE",
    cycles_device: str = "AUTO",
) -> str:
    """Return a runnable bpy script as a string.

    `addendum`: an optional string appended to the user message, typically
    formatted issues from a prior critic pass. `base_scene` is the current
    best known runnable scene.py; on retry it nudges the model toward a
    minimal patch instead of a full reinvention.
    """
    cfg = get_agent_model(AGENT)
    if model:
        chain = (model,)
    else:
        chain = cfg.chain or ("claude-cli/opus",)
    client = LLMClient.from_chain(chain, max_retries=1)
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be one of: BLENDER_EEVEE, CYCLES")
    if cycles_device not in {"AUTO", "GPU", "CPU"}:
        raise ValueError("cycles_device must be one of: AUTO, GPU, CPU")
    user_msg = storyboard.model_dump_json(indent=2)
    user_msg = (
        f"{user_msg}\n\n---\n"
        f"{_render_engine_policy(render_engine, cycles_device)}"
    )
    if base_scene:
        user_msg = (
            f"{user_msg}\n\n---\n"
            "BASE scene.py TO IMPROVE FROM:\n"
            "The script below is either a deterministic scaffold generated "
            "from the storyboard/Scene IR or the current best critic-scored "
            "iteration. Treat your task like a conservative code review "
            "patch: preserve its working camera/framing/lighting/object-"
            "visibility choices, preview-frame support, render boilerplate, "
            "and reusable teaching helpers such as labeled points, thin "
            "curve_polyline tracers, focal brackets, and image-plane marks "
            "unless the retry instructions explicitly require changing them. "
            "Make the smallest local edits needed for the named failing shots; "
            "do not rewrite clean shots, rename working objects, or redesign "
            "the whole scene. Output the COMPLETE runnable scene.py after your "
            "patch, not a diff or explanation.\n\n"
            "```python\n"
            f"{base_scene}\n"
            "```"
        )
    if addendum:
        user_msg = f"{user_msg}\n\n---\n{addendum}"
    complete_kwargs = dict(
        system=load_prompt("coder"),
        user=user_msg,
        response_format="text",
        temperature=0.4,
    )
    max_tokens = int(os.environ.get("CG_TUTOR_CODER_MAX_TOKENS", "16000"))
    if max_tokens > 0:
        complete_kwargs["max_tokens"] = max_tokens
    raw = client.complete(**complete_kwargs)
    code = compatibilize_blender_code(normalize_python_text(strip_code_fence(raw)))
    if out_dir is not None:
        suffix = "" if iteration == 0 else f".iter{iteration:02d}"
        save_artifact(out_dir, f"scene{suffix}.py", code)
        save_artifact(out_dir, f"scene{suffix}.raw.txt", raw)
        # Always also keep "scene.py" pointing at the latest version for
        # downstream tools that look for it by canonical name.
        if iteration > 0:
            save_artifact(out_dir, "scene.py", code)
    return code


def repair_bpy_script_diff(
    storyboard: Storyboard,
    *,
    base_scene: str,
    model: str | None = None,
    out_dir: Path | None = None,
    addendum: str = "",
    iteration: int = 0,
    render_engine: str = "BLENDER_EEVEE",
    cycles_device: str = "AUTO",
) -> str:
    """Ask the coder for SEARCH/REPLACE blocks, apply them, return full code.

    Uses Aider-style blocks because LLMs are unreliable at producing valid
    unified-diff line numbers — the previous unified-diff path mismatched
    context regularly. SEARCH/REPLACE forces the model to anchor on
    unique verbatim snippets instead of line counts.
    """
    if not base_scene:
        raise PatchApplyError("repair requires base_scene")
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be one of: BLENDER_EEVEE, CYCLES")
    if cycles_device not in {"AUTO", "GPU", "CPU"}:
        raise ValueError("cycles_device must be one of: AUTO, GPU, CPU")
    cfg = get_agent_model(AGENT)
    chain = (model,) if model else (cfg.chain or ("claude-cli/opus",))
    client = LLMClient.from_chain(chain, max_retries=1)
    user_msg = (
        "Storyboard JSON:\n"
        f"{storyboard.model_dump_json(indent=2)}\n\n---\n"
        "BASE scene.py:\n"
        "```python\n"
        f"{base_scene}\n"
        "```\n\n---\n"
        f"{_render_engine_policy(render_engine, cycles_device)}\n\n---\n"
        f"{addendum}\n\n"
        "Return ONE OR MORE SEARCH/REPLACE blocks that patch BASE scene.py.\n"
        "Each block MUST use this exact format:\n\n"
        "<<<<<<< SEARCH\n"
        "{a contiguous slice copied verbatim from BASE scene.py}\n"
        "=======\n"
        "{the lines that will replace it}\n"
        ">>>>>>> REPLACE\n\n"
        "Rules:\n"
        "- The SEARCH side MUST be copied byte-for-byte from BASE. Do not "
        "paraphrase, do not change indentation, do not reflow whitespace.\n"
        "- The SEARCH side MUST match exactly ONE location in BASE. If a "
        "snippet appears more than once, include more surrounding lines "
        "until it is unique.\n"
        "- To insert new code, use an existing line as the anchor on the "
        "SEARCH side; repeat that anchor line at the top of the REPLACE "
        "side and put the new lines below it.\n"
        "- To delete code, leave the REPLACE side empty.\n"
        "- Patch only the failing shots named in the repair instructions. "
        "Do not rewrite clean shots, rename working objects, or touch the "
        "render boilerplate or visibility windows.\n"
        "- Output ONLY the blocks. No prose, no code fences, no full file."
    )
    complete_kwargs = dict(
        system=(
            "You are a careful Blender Python maintainer. Produce minimal "
            "SEARCH/REPLACE blocks against the provided scene.py. SEARCH "
            "must be a verbatim slice of BASE that occurs exactly once."
        ),
        user=user_msg,
        response_format="text",
        temperature=0.2,
    )
    max_tokens = int(os.environ.get("CG_TUTOR_PATCH_MAX_TOKENS", "8000"))
    if max_tokens > 0:
        complete_kwargs["max_tokens"] = max_tokens
    raw = client.complete(**complete_kwargs)
    patch = normalize_python_text(strip_code_fence(raw))
    code = compatibilize_blender_code(apply_search_replace_blocks(base_scene, patch))
    compile(code, "scene.py", "exec")
    if out_dir is not None:
        suffix = f".iter{iteration:02d}" if iteration > 0 else ""
        save_artifact(out_dir, f"scene{suffix}.patch.txt", patch)
        save_artifact(out_dir, f"scene{suffix}.py", code)
        save_artifact(out_dir, "scene.py", code)
    return code


def _render_engine_policy(render_engine: str, cycles_device: str = "AUTO") -> str:
    if render_engine == "CYCLES":
        resolved_device = "GPU" if cycles_device == "AUTO" else cycles_device
        gpu_policy = (
            "- Configure GPU rendering before render: set "
            "`scene.cycles.device = 'GPU'`, iterate Cycles addon preferences "
            "over OPTIX/CUDA/HIP/METAL/ONEAPI, call `get_devices()`, enable "
            "non-CPU devices and disable CPU devices when at least one GPU "
            "device is found."
            if resolved_device == "GPU"
            else "- Use `scene.cycles.device = 'CPU'`; do not configure GPU devices."
        )
        return (
            "RENDER ENGINE OVERRIDE FOR THIS RUN:\n"
            "- Use `scene.render.engine = 'CYCLES'`.\n"
            "- Keep Cycles cheap: set `scene.cycles.samples` to 32 or lower "
            "unless a later verifier addendum explicitly requests otherwise.\n"
            "- Disable Cycles denoising for this environment: set "
            "`scene.cycles.use_denoising = False` when available and also "
            "disable `view_layer.cycles.use_denoising`; this Blender build "
            "may not include OpenImageDenoiser.\n"
            f"- Requested Cycles device: {resolved_device}.\n"
            f"{gpu_policy}\n"
            "- Blender API compatibility: do not read "
            "`animation_data.action.fcurves` or `action.fcurves` directly. "
            "If you need to reason about animation, keep your own keyframe "
            "value lists in Python variables.\n"
            "- Do not switch back to BLENDER_EEVEE in this run."
        )
    return (
        "RENDER ENGINE POLICY FOR THIS RUN:\n"
        "- Use `scene.render.engine = 'BLENDER_EEVEE'`.\n"
        "- Do not use `BLENDER_EEVEE_NEXT` or `CYCLES` unless the CLI "
        "explicitly requests Cycles.\n"
        "- Blender API compatibility: do not read "
        "`animation_data.action.fcurves` or `action.fcurves` directly. "
        "If you need to reason about animation, keep your own keyframe "
        "value lists in Python variables."
    )
