"""Tests for the structured visual contract validator.

These checks run scene.py source through Python's AST and verify that
the script honors VisualContract requirements (required_labels,
required_vectors, emphasis_points). The validator is the deterministic
counterpart to the LLM render critic — its findings should not depend
on the model.
"""

from __future__ import annotations

from cg_tutor.contract_validator import (
    format_contract_validation_addendum,
    validate_visual_contracts,
)
from cg_tutor.scene_profiles import base_profile
from cg_tutor.visual_contract import VisualContract


def _minimal_storyboard():
    """A storyboard with one shot keyed shot_id='shot1'."""
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


def test_empty_contracts_returns_empty_report():
    report = validate_visual_contracts("import bpy\n", _minimal_storyboard(), {})
    assert report.ok
    assert report.violations == []


def test_unparseable_code_returns_empty_report_silently():
    report = validate_visual_contracts(
        "import bpy\ndef \n",  # syntax error
        _minimal_storyboard(),
        {"shot1": VisualContract(shot_id="shot1", required_labels=["L"])},
    )
    # Parse errors are scene_verifier's job; we just return clean.
    assert report.ok
    assert report.violations == []


def test_required_labels_block_when_no_text_objects():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["incoming ray", "normal"],
        ),
    }
    code = "import bpy\nbpy.ops.mesh.primitive_sphere_add()\n"
    report = validate_visual_contracts(code, sb, contracts)
    assert not report.ok
    assert report.block_count == 1
    violation = report.violations[0]
    assert violation.rule_id == "contract_no_text_objects"
    assert violation.expected == 2
    assert violation.found == 0


def test_required_anchors_block_when_missing_from_scene_code():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["window_frame", "neon_sign_OPEN"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add()\n"
        "bpy.context.object.name = 'anonymous_cube'\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    assert not report.ok
    violation = report.violations[0]
    assert violation.rule_id == "contract_missing_scene_anchors"
    assert violation.expected == 2
    assert violation.found == 0
    assert "window_frame" in violation.message


def test_required_anchors_accept_exact_or_obvious_variant_names():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["window_frame", "neon_sign_OPEN"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.context.object.name = 'window_frame_left'\n"
        "sign = 'neon_sign_OPEN'\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    assert report.ok
    assert report.per_shot_counts["scene_grounding_anchors"] == {
        "actual": 2,
        "required": 2,
    }


def test_short_anchor_is_skipped_not_falsely_substring_matched():
    """A 2-letter anchor like 'AB' would substring-match almost any
    identifier ('label', 'cabin', 'tab'). We treat short anchors as
    'skip the structural check' rather than guess — the prompt-layer
    forbidden_failures still carries the constraint."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["AB"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_sphere_add()\n"
        "bpy.context.object.name = 'anonymous'\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    # Short anchor "AB" is skipped → no missing-anchor block fires,
    # so the report stays ok.
    assert report.ok
    block_rules = {v.rule_id for v in report.violations if v.severity == "block"}
    assert "contract_missing_scene_anchors" not in block_rules


def test_short_anchor_with_underscore_is_still_checked():
    """'a_b' is short in chars but has an underscore — treat as meaningful
    and check substring matching normally."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["a_b"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.context.object.name = 'unrelated_shape'\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    assert not report.ok
    assert report.violations[0].rule_id == "contract_missing_scene_anchors"


def test_excessive_anonymous_primitives_warns_when_anchors_required():
    """When required_anchors are set, a scene with many primitive_*_add
    calls but few obj.name assignments signals the LLM is abstracting
    the environment into anonymous shapes. Warn-level."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_anchors=["window_frame"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.context.object.name = 'window_frame'\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_cube_add()\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    # Anchor present → no block. But 4 primitives, only 0 named after
    # creation → excessive anonymous primitives warning.
    warn_rules = {v.rule_id for v in report.violations if v.severity == "warn"}
    assert "contract_excessive_anonymous_primitives" in warn_rules


def test_no_excessive_anonymous_warning_without_required_anchors():
    """The anonymous-primitive heuristic is only relevant when the scene
    is supposed to be grounded. Without required_anchors, decorative
    primitives are fine."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L"],  # other contracts present, but no anchors
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_uv_sphere_add()\n"
        "bpy.ops.mesh.primitive_cube_add()\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    rule_ids = {v.rule_id for v in report.violations}
    assert "contract_excessive_anonymous_primitives" not in rule_ids


def test_required_labels_warn_when_insufficient_text_objects():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["a", "b", "c"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok  # warn-only
    assert report.warn_count == 1
    assert report.violations[0].rule_id == "contract_insufficient_text_objects"


def test_required_labels_block_when_insufficient_for_teaching_profile():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["a", "b", "c"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
    )
    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    assert not report.ok
    assert report.violations[0].severity == "block"
    assert report.violations[0].rule_id == "contract_insufficient_text_objects"


def test_required_labels_satisfied_by_text_add():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["a", "b"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
        "bpy.ops.object.text_add()\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok
    assert report.warn_count == 0


def test_required_labels_count_helper_function_calls():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["d", "f", "FOV"],
        ),
    }
    code = (
        "import bpy\n"
        "def add_text_label(name, body):\n"
        "    bpy.ops.object.text_add()\n"
        "    obj = bpy.context.object\n"
        "    obj.name = name\n"
        "    obj.data.body = body\n"
        "add_text_label('label_d', 'd')\n"
        "add_text_label('label_f', 'f')\n"
        "add_text_label('label_fov', 'FOV')\n"
    )

    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 3,
        "required": 3,
    }


def test_required_labels_count_helper_calls_in_literal_loop():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["d", "f", "FOV", "dolly zoom"],
        ),
    }
    code = (
        "import bpy\n"
        "def add_text_label(name, body):\n"
        "    bpy.ops.object.text_add()\n"
        "shot_labels = [('a', 'd'), ('b', 'f'), ('c', 'FOV'), ('d', 'dolly zoom')]\n"
        "for name, body in shot_labels:\n"
        "    add_text_label(name, body)\n"
    )

    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 4,
        "required": 4,
    }


def test_required_labels_count_helper_calls_in_enumerated_literal_loop():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["d", "f", "FOV", "dolly zoom"],
        ),
    }
    code = (
        "import bpy\n"
        "def add_text_label(name, body):\n"
        "    bpy.ops.object.text_add()\n"
        "shot_labels = [('a', 'd'), ('b', 'f'), ('c', 'FOV'), ('d', 'dolly zoom')]\n"
        "for idx, (name, body) in enumerate(shot_labels):\n"
        "    add_text_label(name, body)\n"
    )

    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 4,
        "required": 4,
    }


def test_required_labels_are_deduped_across_shots():
    from cg_tutor.schemas import Storyboard

    sb = Storyboard.model_validate({
        "concept_id": "label_reuse",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [
            {
                "node_id": "shot1",
                "start_sec": 0.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 0.0, "position": (0, 0, 5), "look_at": (0, 0, 0)}],
                "objects": [{"name": "subject", "type": "primitive", "primitive": "sphere"}],
            },
            {
                "node_id": "shot2",
                "start_sec": 1.0,
                "duration_sec": 1.0,
                "camera": [{"time_sec": 1.0, "position": (0, 0, 5), "look_at": (0, 0, 0)}],
                "objects": [{"name": "subject", "type": "primitive", "primitive": "sphere"}],
            },
        ],
    })
    contracts = {
        "shot1": VisualContract(shot_id="shot1", required_labels=["d", "f"]),
        "shot2": VisualContract(shot_id="shot2", required_labels=["d", "f"]),
    }
    code = (
        "import bpy\n"
        "bpy.ops.object.text_add()\n"
        "bpy.ops.object.text_add()\n"
    )

    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("transformation_demo"),
    )

    assert report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 2,
        "required": 2,
    }


def test_required_labels_satisfied_by_lower_level_curves_new():
    """Some scripts skip the op and use bpy.data.curves.new(type='FONT')."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L"],
        ),
    }
    code = (
        "import bpy\n"
        "data = bpy.data.curves.new('label_curve', type='FONT')\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok


def test_required_vectors_block_with_no_arrows():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_vectors=["N", "L"],
        ),
    }
    code = "import bpy\nbpy.ops.mesh.primitive_sphere_add()\n"
    report = validate_visual_contracts(code, sb, contracts)
    assert not report.ok
    assert report.violations[0].rule_id == "contract_no_vector_geometry"


def test_required_vectors_satisfied_by_cone_cylinder_pair():
    """An arrow needs both shaft (cylinder) and head (cone). One pair
    counts as one arrow."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_vectors=["N"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cylinder_add()\n"
        "bpy.ops.mesh.primitive_cone_add()\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok


def test_required_vectors_satisfied_by_curve_polyline():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_vectors=["ray1", "ray2"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.curve.primitive_bezier_curve_add()\n"
        "bpy.ops.curve.primitive_bezier_curve_add()\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok


def test_compiled_scene_hints_satisfy_dynamic_helper_counts():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L", "N"],
            required_vectors=["ray1", "ray2", "ray3"],
        ),
    }
    code = (
        "import bpy\n"
        "def add_teaching_helpers_for_shot(shot, object_map):\n"
        "    return []\n"
        "add_teaching_helpers_for_shot({}, {})\n"
        "COMPILED_SCENE_HINTS = {\n"
        "  'text_objects_created': 2,\n"
        "  'arrow_or_tracer_primitives': 3,\n"
        "}\n"
    )

    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("vector_teaching"),
    )

    assert report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 2,
        "required": 2,
    }
    assert report.per_shot_counts["arrow_or_tracer_primitives"] == {
        "actual": 3,
        "required": 3,
    }


def test_compiled_scene_hints_ignored_without_helper_callsite():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L"],
            required_vectors=["ray"],
        ),
    }
    code = (
        "import bpy\n"
        "COMPILED_SCENE_HINTS = {\n"
        "  'text_objects_created': 10,\n"
        "  'arrow_or_tracer_primitives': 10,\n"
        "}\n"
    )

    report = validate_visual_contracts(code, sb, contracts)

    assert not report.ok
    assert report.per_shot_counts["text_objects_created"] == {
        "actual": 0,
        "required": 1,
    }


def test_required_vectors_block_when_insufficient_for_teaching_profile():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_vectors=["N", "L"],
        ),
    }
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cylinder_add()\n"
        "bpy.ops.mesh.primitive_cone_add()\n"
    )
    report = validate_visual_contracts(
        code,
        sb,
        contracts,
        scene_profile=base_profile("vector_teaching"),
    )
    assert not report.ok
    assert report.violations[0].severity == "block"
    assert report.violations[0].rule_id == "contract_insufficient_vector_geometry"


def test_emphasis_points_warn_without_emission_assignment():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            emphasis_points=["bright spot"],
        ),
    }
    code = "import bpy\nbpy.ops.mesh.primitive_sphere_add()\n"
    report = validate_visual_contracts(code, sb, contracts)
    assert report.warn_count == 1
    assert report.violations[0].rule_id == "contract_no_emphasis_material"


def test_emphasis_points_satisfied_by_emission_color_input():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            emphasis_points=["x"],
        ),
    }
    code = (
        "import bpy\n"
        "mat = bpy.data.materials.new('m')\n"
        "bsdf = mat.node_tree.nodes['Principled BSDF']\n"
        "bsdf.inputs['Emission Color'].default_value = (1,1,1,1)\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert report.ok


def test_format_addendum_lists_each_violation():
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L"],
            required_vectors=["N"],
        ),
    }
    code = "import bpy\n"
    report = validate_visual_contracts(code, sb, contracts)
    text = format_contract_validation_addendum(report)
    assert "STRUCTURED VISUAL CONTRACT" in text
    assert "contract_no_text_objects" in text
    assert "contract_no_vector_geometry" in text


def test_substring_match_in_comment_is_not_counted():
    """Regression: the old substring scan would treat a commented mention
    of text_add as if it were a real call. AST counts the actual nodes."""
    sb = _minimal_storyboard()
    contracts = {
        "shot1": VisualContract(
            shot_id="shot1",
            required_labels=["L"],
        ),
    }
    code = (
        "import bpy\n"
        "# bpy.ops.object.text_add() — disabled for now\n"
        "bpy.ops.mesh.primitive_sphere_add()\n"
    )
    report = validate_visual_contracts(code, sb, contracts)
    assert not report.ok
    assert report.violations[0].rule_id == "contract_no_text_objects"
