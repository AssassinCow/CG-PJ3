"""LLM-assisted scene profile generation and repair."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from cg_tutor.agents.base import load_prompt, save_artifact
from cg_tutor.config import get_agent_model
from cg_tutor.llm_client import LLMClient
from cg_tutor.scene_profiles import (
    BASE_PROFILES,
    SceneProfileResolution,
    format_scene_profile_for_prompt,
    profile_requires_llm,
    profile_seed,
    sanitize_profile,
)
from cg_tutor.schemas import Narrative
from cg_tutor._logging import get_logger


log = get_logger(__name__)


AGENT = "scene_profile"


def resolve_scene_profile(
    concept_spec: dict[str, Any],
    narrative: Narrative,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
) -> SceneProfileResolution:
    """Resolve a profile through LLM review/completion with safe fallback.

    Built-in profiles are only seeds. Fresh runs still ask the scene_profile
    model to adapt the policy and critic rubric to the specific concept, so
    style and evaluation remain automatic instead of purely hand-written.
    """
    seed = profile_seed(concept_spec)
    requested = concept_spec.get("scene_profile")
    if not profile_requires_llm(concept_spec):
        raw = seed.model_dump()
        resolved = sanitize_profile(raw, fallback_base=seed.base_profile, source="builtin")
        _save_resolution(out_dir, resolved)
        return resolved

    try:
        raw = _complete_profile_json(
            concept_spec=concept_spec,
            narrative=narrative,
            seed=seed.model_dump(),
            model=model,
        )
        resolved = sanitize_profile(raw, fallback_base=seed.base_profile, source="llm")
        _preserve_seed_grounding(resolved, seed.model_dump())
        _save_resolution(out_dir, resolved)
        return resolved
    except Exception as e:  # noqa: BLE001
        raw = seed.model_dump()
        resolved = sanitize_profile(raw, fallback_base=seed.base_profile, source="fallback")
        resolved.validation["llm_error"] = f"{type(e).__name__}: {e}"
        resolved.validation["requested_scene_profile"] = requested
        _save_resolution(out_dir, resolved)
        log.info("LLM review failed; using deterministic "
            f"{seed.base_profile} fallback: {type(e).__name__}: {e}",
        )
        return resolved


def _preserve_seed_grounding(
    resolved: SceneProfileResolution,
    seed: dict[str, Any],
) -> None:
    """Keep explicit concept grounding constraints after LLM profile review."""
    changed: list[str] = []
    for key in (
        "persistent_anchors",
        "spatial_relationships",
        "forbidden_abstractions",
    ):
        existing = list(getattr(resolved.profile, key))
        for value in seed.get(key) or []:
            if value and value not in existing:
                existing.append(value)
                changed.append(key)
        setattr(resolved.profile, key, existing)
    if changed:
        resolved.validation["preserved_seed_grounding"] = sorted(set(changed))


def _complete_profile_json(
    *,
    concept_spec: dict[str, Any],
    narrative: Narrative,
    seed: dict[str, Any],
    model: str | None,
) -> dict[str, Any]:
    cfg = get_agent_model(AGENT)
    chain = (model,) if model else (cfg.chain or ("gpt/gpt-5.5", "claude/claude-sonnet-4-6"))
    client = LLMClient.from_chain(chain, max_retries=1)
    user = "\n\n".join([
        "Concept YAML:",
        yaml.safe_dump(concept_spec, sort_keys=False, allow_unicode=True),
        "Narrative JSON:",
        narrative.model_dump_json(indent=2),
        "Seed profile to validate/complete:",
        json.dumps(seed, indent=2, ensure_ascii=False),
        "Available base profiles:",
        json.dumps(BASE_PROFILES, indent=2, ensure_ascii=False),
    ])
    return client.complete_json(
        system=load_prompt("profile_generator"),
        user=user,
        max_tokens=int(os.environ.get("CG_TUTOR_PROFILE_MAX_TOKENS", "3500")),
    )


def _save_resolution(out_dir: Path | None, resolved: SceneProfileResolution) -> None:
    if out_dir is None:
        return
    if resolved.raw is not None:
        save_artifact(
            out_dir,
            "scene_profile.raw.json",
            json.dumps(resolved.raw, indent=2, ensure_ascii=False),
        )
    save_artifact(
        out_dir,
        "scene_profile.json",
        resolved.profile.model_dump_json(indent=2),
    )
    save_artifact(
        out_dir,
        "scene_profile.validation.json",
        json.dumps(resolved.validation, indent=2, ensure_ascii=False),
    )
    save_artifact(
        out_dir,
        "scene_profile.prompt.txt",
        format_scene_profile_for_prompt(resolved.profile),
    )
