from pathlib import Path

from cg_tutor.critic_loop import CATEGORY_REPAIR_MESSAGES, CriticIteration
from cg_tutor.repair_plan import (
    _action_for_category,
    build_repair_plan,
    format_repair_plan_for_coder,
)
from cg_tutor.scene_verifier import verify_scene_code
from cg_tutor.schemas import CriticIssue, CriticReport
from cg_tutor.scene_profiles import base_profile
from cg_tutor.contract_validator import ContractValidationReport, ContractViolation
from cg_tutor.visual_contract import VisualContract


def _iter_with_issue(category: str = "concept_mismatch") -> CriticIteration:
    return CriticIteration(
        iteration=2,
        report=CriticReport(
            concept_id="demo",
            iteration=2,
            overall_score=0.6,
            issues=[
                CriticIssue(
                    shot_id="node_01",
                    frame_idx=1,
                    severity="block",
                    category=category,
                    issue="The required visual element is missing.",
                )
            ],
        ),
        scene_path=Path("/tmp/missing.py"),
        render_ok=True,
        n_frames=10,
    )


def test_repair_plan_includes_critic_targets():
    plan = build_repair_plan(iteration=3, history=[_iter_with_issue()])

    assert plan.base_iteration == 2
    assert plan.targets[0].source == "critic"
    assert plan.targets[0].category == "concept_mismatch"
    assert "visual_intent" in plan.targets[0].action


def test_repair_plan_includes_verifier_blocks_first():
    report = verify_scene_code("print('bad')", None)

    plan = build_repair_plan(iteration=0, history=[], verifier_report=report)

    assert plan.targets
    assert plan.targets[0].source == "scene_verifier"
    assert plan.targets[0].severity == "block"


def test_repair_plan_formats_for_coder():
    plan = build_repair_plan(iteration=3, history=[_iter_with_issue("off_screen")])

    out = format_repair_plan_for_coder(plan)

    assert "STRUCTURED REPAIR PLAN" in out
    assert "off_screen" in out
    assert "critic/critic_aggregate" in out


def test_repair_plan_turns_issue_text_into_specific_actions():
    history = [_iter_with_issue()]
    history[0].report.issues[0].issue = (
        "The required labels are missing and arrows overlap the highlight."
    )

    plan = build_repair_plan(
        iteration=3,
        history=history,
        visual_contracts={
            "node_01": VisualContract(
                shot_id="node_01",
                required_relationships=["Keep arrows separated."],
                forbidden_failures=["Do not omit labels."],
            )
        },
    )

    action = plan.targets[0].action
    assert "text labels" in action
    assert "distinct full arrow" in action
    assert "highlight" in action
    assert "Keep arrows separated" in action


def test_repair_plan_uses_cinematic_tracer_policy():
    history = [_iter_with_issue()]
    history[0].report.issues[0].issue = (
        "The ray arrow is too thick and overlaps the crystal reflection."
    )

    plan = build_repair_plan(
        iteration=3,
        history=history,
        scene_profile=base_profile("cinematic_application"),
    )

    action = plan.targets[0].action
    assert "thin glowing" in action
    assert "do not use thick arrows" in action
    assert "distinct full arrow" not in action


def test_repair_plan_uses_vector_teaching_tracer_policy():
    history = [_iter_with_issue()]
    history[0].report.issues[0].issue = (
        "The incident ray arrow and normal vector are missing."
    )

    plan = build_repair_plan(
        iteration=3,
        history=history,
        scene_profile=base_profile("vector_teaching"),
    )

    action = plan.targets[0].action
    assert "thin glowing" in action
    assert "do not use thick arrows" in action
    assert "distinct full arrow" not in action


def test_repair_plan_prioritizes_contract_blocks_before_critic():
    contract = ContractValidationReport(violations=[
        ContractViolation(
            severity="block",
            rule_id="contract_insufficient_vector_geometry",
            shot_id="node_01",
            field="required_vectors",
            expected=3,
            found=1,
            message="not enough vector geometry",
            suggested_fix="add named curve_polyline rays",
        )
    ])

    plan = build_repair_plan(
        iteration=3,
        history=[_iter_with_issue("off_screen")],
        contract_report=contract,
    )

    assert plan.targets[0].source == "contract"
    assert plan.targets[0].source_report == "contract_validation"
    assert plan.targets[0].category == "contract_insufficient_vector_geometry"
    assert plan.targets[1].source == "critic"


def test_repair_plan_uses_profile_rubric_and_removes_formula_overlay():
    history = [_iter_with_issue()]
    history[0].report.issues[0].issue = (
        "The frame contains a formula overlay and contribution tiles in a cinematic scene."
    )

    plan = build_repair_plan(
        iteration=3,
        history=history,
        scene_profile=base_profile("cinematic_application"),
    )

    action = plan.targets[0].action
    assert "Remove formula overlays" in action
    assert "adaptive critic rubric" in action
    assert "formula area is empty" not in action


def test_action_for_category_uses_registry_action_base():
    """_action_for_category must prefix its output with the shared
    registry's action_base text, so a registry edit propagates to the
    repair plan without code changes here."""
    out = _action_for_category("off_screen", issue_text="")
    assert out.startswith(CATEGORY_REPAIR_MESSAGES["off_screen"]["action_base"])


def test_action_for_category_falls_back_to_other_for_unknown():
    """Unknown categories must dispatch to the 'other' bucket — this is
    the contract that keeps the coder addendum non-empty when a new
    issue category appears in the wild before the registry is updated."""
    out = _action_for_category("brand_new_category", issue_text="")
    assert out.startswith(CATEGORY_REPAIR_MESSAGES["other"]["action_base"])
