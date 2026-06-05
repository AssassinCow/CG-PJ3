"""Shared helpers for agent implementations.

We deliberately avoid an Agent base class — every agent is a single
top-level function `run(input, model=...) -> output`. Simpler call sites,
easier to test, and the agents don't share enough state to justify a
class hierarchy.
"""

from __future__ import annotations

import re
from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt template by basename (without extension)."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text()


_FENCE_RE = re.compile(r"```(?:python|py|json)?[^\n]*\n(.*?)```", re.DOTALL)
_OPENING_FENCE_RE = re.compile(r"^```(?:python|py|json)?[^\n]*\n", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    """Return the largest fenced block, or the trimmed text if unfenced.

    Some providers truncate long code responses before the closing fence.
    In that case, still remove the opening fence so Blender does not see
    a Markdown marker as Python syntax.
    """
    candidates = _FENCE_RE.findall(text)
    if candidates:
        return max(candidates, key=len).strip()
    stripped = text.strip()
    return _OPENING_FENCE_RE.sub("", stripped, count=1).strip()


def save_artifact(out_dir: Path, name: str, content: str | bytes) -> Path:
    """Persist an agent intermediate artifact for debugging / inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path
