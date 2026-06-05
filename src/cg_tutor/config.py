"""Load per-agent model assignments from a YAML config file.

Single entry point: :func:`get_models()`. The result is a dict keyed by
agent name (``concept_decomposer``, ``storyboard``, ``blender_coder``,
``render_critic``, ``latex_overlay``); each value is an ``AgentModel``
with ``primary`` and ``fallback`` slugs (either may be ``None``).

Resolution order:
1. Explicit path passed to ``get_models(path=...)``.
2. ``CG_TUTOR_MODELS_YAML`` environment variable.
3. ``<repo_root>/configs/models_api.yaml`` (default).

The repo also ships ``configs/models_cli.yaml`` for the CLI-subprocess
provider mode; switch to it by setting ``CG_TUTOR_MODELS_YAML``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentModel:
    primary: str | None
    fallback: str | None = None

    @property
    def chain(self) -> tuple[str, ...]:
        """Provider slugs to try in order, skipping unset entries."""
        return tuple(s for s in (self.primary, self.fallback) if s)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models_api.yaml"


def _resolve_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("CG_TUTOR_MODELS_YAML")
    if env:
        return Path(env)
    return DEFAULT_CONFIG


@lru_cache(maxsize=8)
def _load_yaml(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def get_models(path: Path | str | None = None) -> dict[str, AgentModel]:
    p = _resolve_path(path)
    data = _load_yaml(str(p))
    agents = data.get("agents", {}) or {}
    out: dict[str, AgentModel] = {}
    for name, spec in agents.items():
        spec = spec or {}
        out[name] = AgentModel(
            primary=spec.get("primary"),
            fallback=spec.get("fallback"),
        )
    return out


def get_agent_model(agent: str, path: Path | str | None = None) -> AgentModel:
    """Look up a single agent's slugs. Missing agent → empty AgentModel."""
    return get_models(path).get(agent, AgentModel(primary=None, fallback=None))


def get_defaults(path: Path | str | None = None) -> dict:
    p = _resolve_path(path)
    data = _load_yaml(str(p))
    return data.get("defaults", {}) or {}
