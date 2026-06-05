import pytest

from cg_tutor.agents.concept_decomposer import _normalise_narrative_payload
from cg_tutor.schemas import Narrative


def _node(node_id: str = "node_01") -> dict:
    return {
        "id": node_id,
        "title": "One",
        "description": "Show one concept.",
        "formulas": [],
        "duration_sec": 3.0,
        "visual_intent": "A clear visual setup.",
    }


def test_decomposer_wraps_single_node_payload():
    payload = _normalise_narrative_payload(
        _node(),
        {"concept_id": "depth_of_field_focus_pull"},
    )

    narrative = Narrative.model_validate(payload)

    assert narrative.concept_id == "depth_of_field_focus_pull"
    assert [node.id for node in narrative.nodes] == ["node_01"]


def test_decomposer_wraps_bare_node_list_payload():
    payload = _normalise_narrative_payload(
        [_node("node_01"), _node("node_02")],
        {"concept_id": "fake"},
    )

    narrative = Narrative.model_validate(payload)

    assert narrative.concept_id == "fake"
    assert [node.id for node in narrative.nodes] == ["node_01", "node_02"]


def test_decomposer_preserves_full_narrative_payload():
    raw = {"concept_id": "fake", "nodes": [_node()]}

    assert _normalise_narrative_payload(raw, {"concept_id": "other"}) == raw


def test_decomposer_still_rejects_invalid_node_payload():
    with pytest.raises(Exception):
        _normalise_narrative_payload(
            {"id": "node_01", "duration_sec": 3.0},
            {"concept_id": "fake"},
        )
