"""Tests for generated soft Success Spec rules."""

from __future__ import annotations

from cg_tutor.auto_success_spec import (
    AutoSuccessRule,
    AutoSuccessSpec,
    critic_confirmed_auto_success_issues,
    generate_auto_success_spec,
)
from cg_tutor.schemas import (
    CriticIssue,
    CriticReport,
    Narrative,
    NarrativeNode,
    Storyboard,
)
from cg_tutor.success_spec import SuccessSpec


def _narrative() -> Narrative:
    return Narrative(
        concept_id="texture_mipmap_lod",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="LOD stack",
                description="Compare LOD0, LOD1, and LOD2 in a side panel.",
                formulas=[],
                duration_sec=3.0,
                visual_intent="Show mip_level_stack and lod_readout.",
            ),
        ],
    )


def _storyboard() -> Storyboard:
    return Storyboard.model_validate({
        "concept_id": "texture_mipmap_lod",
        "fps": 24,
        "resolution": [640, 360],
        "shots": [{
            "node_id": "node_01",
            "start_sec": 0.0,
            "duration_sec": 3.0,
            "camera": [{
                "time_sec": 0.0,
                "position": [0, -4, 2],
                "look_at": [0, 0, 0],
                "fov": 55,
            }],
            "objects": [
                {
                    "name": "mip_level_stack",
                    "type": "mesh",
                    "primitive": "cube",
                },
                {
                    "name": "lod_readout",
                    "type": "text",
                    "keyframes": [
                        {"time_sec": 0.0, "attr": "location", "value": [0, 0, 0]},
                        {"time_sec": 2.0, "attr": "location", "value": [1, 0, 0]},
                    ],
                },
            ],
        }],
    })


def test_auto_success_spec_generates_soft_rules_without_user_spec():
    spec, validation = generate_auto_success_spec(
        concept_spec={
            "concept_id": "texture_mipmap_lod",
            "title": "Texture Mipmap LOD",
            "persistent_anchors": [
                "mip_level_stack",
                "lod_readout",
                "camera_dolly_path",
            ],
            "key_points": ["Show LOD0, LOD1, and LOD2."],
        },
        narrative=_narrative(),
        storyboard=_storyboard(),
    )

    rule_pairs = {(rule.kind, tuple(rule.anchors)) for rule in spec.rules}

    assert validation["rejected"] == []
    assert ("object_visible", ("mip_level_stack",)) in rule_pairs
    assert ("text_readable", ("lod_readout",)) in rule_pairs
    assert ("stay_in_screen_safe", ("lod_readout",)) in rule_pairs
    assert ("helper_hidden", ("camera_dolly_path",)) in rule_pairs
    assert ("progressive_visual_ordering", ("LOD0", "LOD1", "LOD2")) in rule_pairs
    assert all(
        rule.failure_class in {"success_soft", "aesthetic_warn", "diagnostic"}
        for rule in spec.rules
    )


def test_auto_success_spec_does_not_invent_view_vector_from_preview():
    spec, _validation = generate_auto_success_spec(
        concept_spec={
            "concept_id": "preview_demo",
            "title": "Preview demo",
            "persistent_anchors": ["preview_panel"],
            "key_points": ["A preview panel shows the result."],
        },
        narrative=None,
        storyboard=None,
    )

    anchors = {anchor for rule in spec.rules for anchor in rule.anchors}

    assert "view_vector" not in anchors
    assert "view" not in anchors


def test_user_success_spec_anchors_are_not_duplicated_by_generated_spec():
    user_spec = SuccessSpec.model_validate({
        "version": 1,
        "success_states": [{
            "id": "early",
            "frame_range": {"fraction": [0.0, 1.0]},
        }],
        "hard_constraints": [{
            "kind": "text_faces_camera",
            "anchors": ["lod_readout"],
        }],
    })

    spec, _validation = generate_auto_success_spec(
        concept_spec={
            "concept_id": "texture_mipmap_lod",
            "persistent_anchors": ["lod_readout", "mip_level_stack"],
        },
        narrative=_narrative(),
        storyboard=_storyboard(),
        user_success_spec=user_spec,
    )

    generated_targets = [anchor for rule in spec.rules for anchor in rule.anchors]

    assert "lod_readout" not in generated_targets
    assert "mip_level_stack" in generated_targets


def test_repeated_critic_evidence_escalates_generated_rule_current_run_only():
    auto_spec = AutoSuccessSpec(rules=[
        AutoSuccessRule(kind="text_readable", anchors=["lod_readout"]),
    ])
    prior = CriticReport(
        concept_id="texture_mipmap_lod",
        iteration=0,
        overall_score=0.6,
        issues=[
            CriticIssue(
                shot_id="node_01",
                frame_idx=1,
                severity="warn",
                category="concept_mismatch",
                issue="lod_readout is unreadable.",
                evidence_kind="text_readable",
                target="lod_readout",
                expected="readable readout",
                observed="projected on floor",
                confidence=0.8,
            ),
        ],
    )
    current = prior.model_copy(update={"iteration": 1})

    issues = critic_confirmed_auto_success_issues(
        auto_success_spec=auto_spec,
        critic_history=[type("Iter", (), {"report": prior})()],
        current_report=current,
    )

    assert issues
    assert issues[0]["failure_class"] == "success_hard"


def _repeated_auto_issue(
    *,
    kind: str,
    anchor: str,
    static_anchor_status: dict[str, dict] | None = None,
    issue_text: str = "target is missing from the frame",
    category: str = "concept_mismatch",
) -> list[dict]:
    auto_spec = AutoSuccessSpec(rules=[
        AutoSuccessRule(kind=kind, anchors=[anchor]),  # type: ignore[arg-type]
    ])
    prior = CriticReport(
        concept_id="texture_mipmap_lod",
        iteration=0,
        overall_score=0.6,
        issues=[
            CriticIssue(
                shot_id="node_01",
                frame_idx=1,
                severity="block",
                category=category,  # type: ignore[arg-type]
                issue=issue_text,
                evidence_kind=kind,  # type: ignore[arg-type]
                target=anchor,
                expected="expected target",
                observed="not visible",
                confidence=0.85,
            ),
        ],
    )
    current = prior.model_copy(update={"iteration": 1})
    return critic_confirmed_auto_success_issues(
        auto_success_spec=auto_spec,
        critic_history=[type("Iter", (), {"report": prior})()],
        current_report=current,
        static_anchor_status=static_anchor_status,
    )


def test_repeated_object_visible_with_static_object_stays_soft_visibility_unproven():
    issues = _repeated_auto_issue(
        kind="object_visible",
        anchor="mip_level_stack",
        static_anchor_status={
            "mip_level_stack": {
                "created": True,
                "matched_names": ["mip_level_stack"],
            },
        },
    )

    assert issues
    assert issues[0]["rule_id"] == "auto_success_visibility_unproven"
    assert issues[0]["severity"] == "warn"
    assert issues[0]["failure_class"] == "success_soft"
    assert "do not add a duplicate" in issues[0]["suggested_fix"]


def test_repeated_object_visible_without_static_object_escalates_hard():
    issues = _repeated_auto_issue(
        kind="object_visible",
        anchor="near_checker_patch",
        static_anchor_status={
            "near_checker_patch": {
                "created": False,
                "matched_names": [],
            },
        },
    )

    assert issues
    assert issues[0]["rule_id"] == "auto_success_object_visible"
    assert issues[0]["severity"] == "block"
    assert issues[0]["failure_class"] == "success_hard"


def test_repeated_stay_in_screen_safe_escalates_hard():
    issues = _repeated_auto_issue(
        kind="stay_in_screen_safe",
        anchor="mip_level_stack",
        issue_text="mip_level_stack is clipped offscreen at the edge",
        category="off_screen",
    )

    assert issues
    assert issues[0]["rule_id"] == "auto_success_stay_in_screen_safe"
    assert issues[0]["failure_class"] == "success_hard"


def test_repeated_helper_hidden_does_not_escalate_hard():
    issues = _repeated_auto_issue(
        kind="helper_hidden",
        anchor="camera_dolly_path",
        issue_text="visible helper path is distracting",
    )

    assert issues
    assert issues[0]["rule_id"] == "auto_success_helper_hidden"
    assert issues[0]["severity"] == "warn"
    assert issues[0]["failure_class"] == "aesthetic_warn"
