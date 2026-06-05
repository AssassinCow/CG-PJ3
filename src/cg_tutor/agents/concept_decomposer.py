"""Concept Decomposer Agent.

Concept spec (YAML/dict) → Narrative (a tree of 2-5 nodes, each one
visual scene with formulas).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from cg_tutor.agents.base import load_prompt, save_artifact
from cg_tutor.config import get_agent_model
from cg_tutor.llm_client import LLMClient
from cg_tutor.schemas import Narrative, NarrativeNode


AGENT = "concept_decomposer"


def _normalise_narrative_payload(raw: object, concept_spec: dict) -> dict:
    """Coerce common LLM schema drift back to Narrative shape.

    The decomposer prompt asks for {"concept_id": ..., "nodes": [...]}, but
    some models occasionally return a single NarrativeNode dict or a bare
    list of nodes. Keep strict Pydantic validation for the node fields while
    repairing only this outer wrapper.
    """
    concept_id = str(concept_spec.get("concept_id") or concept_spec.get("id") or "")
    if isinstance(raw, dict) and "concept_id" in raw and "nodes" in raw:
        return raw
    if isinstance(raw, dict) and "nodes" in raw:
        out = dict(raw)
        out["concept_id"] = concept_id
        return out
    if isinstance(raw, dict) and "id" in raw and "duration_sec" in raw:
        node = NarrativeNode.model_validate(raw)
        return {"concept_id": concept_id, "nodes": [node.model_dump()]}
    if isinstance(raw, list):
        nodes = [NarrativeNode.model_validate(item).model_dump() for item in raw]
        return {"concept_id": concept_id, "nodes": nodes}
    return raw  # type: ignore[return-value]


def decompose(
    concept_spec: dict,
    *,
    model: str | None = None,
    out_dir: Path | None = None,
) -> Narrative:
    cfg = get_agent_model(AGENT)
    if model:
        chain = (model,)
    else:
        chain = cfg.chain or ("codex-cli/gpt-5.5",)
    client = LLMClient.from_chain(chain)
    raw = client.complete_json(
        system=load_prompt("decomposer"),
        user=yaml.safe_dump(concept_spec, sort_keys=False, allow_unicode=True),
        max_tokens=int(os.environ.get("CG_TUTOR_DECOMPOSER_MAX_TOKENS", "3000")),
    )
    narrative = Narrative.model_validate(
        _normalise_narrative_payload(raw, concept_spec),
    )
    if out_dir is not None:
        save_artifact(out_dir, "narrative.json", narrative.model_dump_json(indent=2))
    return narrative
