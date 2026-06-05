"""Tests for the validator rule sanity-check framework.

The framework lets each validator rule register a callback that inspects
the AST for contradicting evidence and can downgrade a BLOCK to a WARN.
The motivating case is ``contract_insufficient_text_objects`` reporting
``actual=1, required=30`` when scene.py defines a text-creating helper
and invokes it inside a runtime loop the static counter can't see.
"""

from __future__ import annotations

from cg_tutor.contract_validator import validate_visual_contracts
from cg_tutor.scene_profiles import base_profile
from cg_tutor.visual_contract import VisualContract


def _minimal_storyboard():
    from cg_tutor.schemas import Storyboard
    return Storyboard.model_validate({
        "concept_id": "test_concept",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": "shot1",
            "start_sec": 0.0,
            "duration_sec": 1.0,
            "camera": [
                {"time_sec": 0.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
                {"time_sec": 1.0, "position": (0, 0, 5), "look_at": (0, 0, 0)},
            ],
            "objects": [
                {"name": "subject", "type": "primitive", "primitive": "sphere"},
            ],
        }],
    })


def _many_label_contracts(n: int) -> dict[str, VisualContract]:
    labels = [f"label_{i}" for i in range(n)]
    return {
        "shot1": VisualContract(shot_id="shot1", required_labels=labels),
    }


def test_text_objects_block_downgraded_when_factory_in_runtime_loop():
    """The dolly_zoom failure mode: helper defined, called inside a runtime
    loop over shot_id keys whose length the static counter cannot see."""
    sb = _minimal_storyboard()
    contracts = _many_label_contracts(30)
    code = (
        "import bpy\n"
        "def add_label(text):\n"
        "    bpy.ops.object.text_add()\n"
        "    bpy.context.object.data.body = text\n"
        "\n"
        "shots_dict = build_label_table_at_runtime()\n"
        "for shot_id, labels in shots_dict.items():\n"
        "    for label in labels:\n"
        "        add_label(label)\n"
    )
    report = validate_visual_contracts(
        code, sb, contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    # The rule still fires (static count is 1) but should now be a warn.
    rule = next(
        v for v in report.violations
        if v.rule_id == "contract_insufficient_text_objects"
    )
    assert rule.severity == "warn"
    assert rule.message.startswith("[sanity-downgraded] ")
    assert "_sanity_downgrades" in report.per_shot_counts
    downgrade = report.per_shot_counts["_sanity_downgrades"]
    assert "contract_insufficient_text_objects" in downgrade
    assert downgrade["contract_insufficient_text_objects"]["verdict"] == "downgrade"


def test_text_objects_block_preserved_when_helper_never_called():
    """Helper defined but never invoked — no contradicting evidence, keep block."""
    sb = _minimal_storyboard()
    contracts = _many_label_contracts(30)
    code = (
        "import bpy\n"
        "def add_label(text):\n"
        "    bpy.ops.object.text_add()\n"
        "    bpy.context.object.data.body = text\n"
        "\n"
        "# helper never invoked\n"
    )
    report = validate_visual_contracts(
        code, sb, contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    rule = next(
        v for v in report.violations
        if v.rule_id == "contract_insufficient_text_objects"
    )
    assert rule.severity == "block"
    assert not rule.message.startswith("[sanity-downgraded]")
    assert "_sanity_downgrades" not in report.per_shot_counts


def test_text_objects_block_preserved_when_no_text_helper_at_all():
    """Pure single text_add call with required=30 — no sanity contradiction."""
    sb = _minimal_storyboard()
    contracts = _many_label_contracts(30)
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
    )
    report = validate_visual_contracts(
        code, sb, contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    rule = next(
        v for v in report.violations
        if v.rule_id == "contract_insufficient_text_objects"
    )
    assert rule.severity == "block"


def test_text_objects_block_downgraded_when_listcomp_with_factory():
    """List-comprehension over runtime data also counts as loop evidence."""
    sb = _minimal_storyboard()
    contracts = _many_label_contracts(30)
    code = (
        "import bpy\n"
        "def add_label(text):\n"
        "    bpy.ops.object.text_add()\n"
        "\n"
        "all_labels = collect_labels_at_runtime()\n"
        "_ = [add_label(s) for s in all_labels]\n"
    )
    report = validate_visual_contracts(
        code, sb, contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    rule = next(
        v for v in report.violations
        if v.rule_id == "contract_insufficient_text_objects"
    )
    assert rule.severity == "warn"


def test_text_objects_block_downgraded_for_curves_font_helper():
    """Lower-level ``bpy.data.curves.new(type='FONT')`` helper also counts."""
    sb = _minimal_storyboard()
    contracts = _many_label_contracts(30)
    code = (
        "import bpy\n"
        "def make_text(text):\n"
        "    curve = bpy.data.curves.new(name='t', type='FONT')\n"
        "    obj = bpy.data.objects.new('text_obj', curve)\n"
        "    bpy.context.collection.objects.link(obj)\n"
        "\n"
        "for shot_id in storyboard_shot_ids:\n"
        "    for label in labels_by_shot[shot_id]:\n"
        "        make_text(label)\n"
    )
    report = validate_visual_contracts(
        code, sb, contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    rule = next(
        v for v in report.violations
        if v.rule_id == "contract_insufficient_text_objects"
    )
    assert rule.severity == "warn"
    assert "make_text" in (
        report.per_shot_counts["_sanity_downgrades"][rule.rule_id]["reason"]
    )


def test_sanity_check_errors_do_not_break_validation():
    """A buggy sanity check must never make validation crash; the original
    violation is preserved and the error is recorded for debugging."""
    from cg_tutor.contract_validator import (
        RULE_SANITY_CHECKS, register_sanity_check,
    )

    # Register a buggy check for a fake rule, fire validation that emits
    # that rule, and confirm the original violation survives.
    # We'll piggyback on the real text rule by temporarily replacing its
    # check with a raising one.
    original = RULE_SANITY_CHECKS["contract_insufficient_text_objects"]

    @register_sanity_check("contract_insufficient_text_objects")
    def _buggy(violation, tree, code):
        raise RuntimeError("boom")

    try:
        sb = _minimal_storyboard()
        contracts = _many_label_contracts(30)
        code = "import bpy\nbpy.ops.object.text_add()\n"
        report = validate_visual_contracts(
            code, sb, contracts,
            scene_profile=base_profile("vector_teaching"),
        )
        rule = next(
            v for v in report.violations
            if v.rule_id == "contract_insufficient_text_objects"
        )
        # validation did not crash; block stayed block
        assert rule.severity == "block"
        # error was recorded for audit
        assert "_sanity_downgrades" in report.per_shot_counts
        entry = report.per_shot_counts["_sanity_downgrades"][rule.rule_id]
        assert entry["verdict"] == "error"
        assert "boom" in entry["reason"]
    finally:
        RULE_SANITY_CHECKS["contract_insufficient_text_objects"] = original
