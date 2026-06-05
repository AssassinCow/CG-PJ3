"""Tests for the critic × AST cross-reference layer.

These verify the join logic between vision-critic findings (free-text) and
scene.py AST evidence. Each rule should only fire when *both* sources
independently corroborate the same problem.
"""

from __future__ import annotations

from cg_tutor.critic_cross_reference import (
    CrossReferenceFinding,
    CrossReferenceReport,
    _all_created_object_names,
    _extract_candidate_tokens,
    _keyframe_schedule,
    _ramp_value_at,
    cross_reference_critic_findings,
    cross_reference_report_to_json,
    format_cross_reference_for_coder,
)
from cg_tutor.schemas import Storyboard
from cg_tutor.schemas.feedback import CriticIssue
from cg_tutor.visual_contract import VisualContract


def _storyboard(*, object_names: list[str], shot_id: str = "node_01"):
    objects = [
        {"name": name, "type": "primitive", "primitive": "cube"}
        for name in object_names
    ] or [{"name": "__placeholder__", "type": "primitive", "primitive": "cube"}]
    return Storyboard.model_validate({
        "concept_id": "test_concept",
        "fps": 24,
        "resolution": (1280, 720),
        "shots": [{
            "node_id": shot_id,
            "start_sec": 0.0,
            "duration_sec": 4.0,
            "camera": [
                {
                    "time_sec": 0.0,
                    "position": (0, -3, 1.5),
                    "look_at": (0, 0, 1.5),
                    "fov": 60.0,
                },
            ],
            "objects": objects,
        }],
    })


def _contract(*, shot_id: str = "node_01", required_labels=None, required_anchors=None):
    return VisualContract(
        shot_id=shot_id,
        required_labels=required_labels or [],
        required_anchors=required_anchors or [],
    )


def _issue(text: str, *, shot_id="node_01", frame_idx=36, severity="block",
           category="concept_mismatch") -> CriticIssue:
    return CriticIssue(
        shot_id=shot_id,
        frame_idx=frame_idx,
        severity=severity,
        category=category,
        issue=text,
        suggested_fix={},
    )


# ---------------- Token extraction ----------------


def test_extract_candidate_tokens_pulls_quoted_strings():
    text = "The label 'E' is absent, only 'tip' is visible."
    tokens = _extract_candidate_tokens(text)
    assert "E" in tokens
    assert "tip" in tokens


def test_extract_candidate_tokens_pulls_uppercase_short_ids():
    text = "Labels J1 and J2 are visible but E is missing."
    tokens = _extract_candidate_tokens(text)
    assert "J1" in tokens
    assert "J2" in tokens
    assert "E" in tokens


def test_extract_candidate_tokens_filters_stop_words():
    text = "The trace is absent. It does not appear."
    tokens = _extract_candidate_tokens(text)
    assert "The" not in tokens
    assert "It" not in tokens


def test_extract_candidate_tokens_pulls_snake_case_identifiers():
    text = "trajectory_trace and joint_J1 are not drawn."
    tokens = _extract_candidate_tokens(text)
    assert "trajectory_trace" in tokens
    assert "joint_J1" in tokens


# ---------------- AST helpers ----------------


def test_all_created_object_names_finds_both_new_and_subscript():
    code = (
        "import bpy\n"
        "a = bpy.data.objects.new('hero', None)\n"
        "b = bpy.data.objects['camera_icon']\n"
        "c = bpy.data.objects.new('marker_1', None)\n"
    )
    import ast
    tree = ast.parse(code)
    assert _all_created_object_names(tree) == {"hero", "camera_icon", "marker_1"}


def test_all_created_object_names_finds_bpy_ops_context_object_renames():
    code = (
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add(size=1.0)\n"
        "hero = bpy.context.object\n"
        "hero.name = 'hero_lighthouse'\n"
        "bpy.ops.object.text_add(location=(0, 0, 0))\n"
        "bpy.context.object.name = 'label_E'\n"
        "bpy.ops.object.light_add(type='POINT')\n"
        "light = bpy.context.active_object\n"
        "light.name = 'key_light'\n"
    )
    import ast
    tree = ast.parse(code)
    assert _all_created_object_names(tree) == {
        "hero_lighthouse",
        "label_E",
        "key_light",
    }


def test_all_created_object_names_finds_helper_factory_literal_calls():
    code = (
        "import bpy\n"
        "def add_screen(name, location):\n"
        "    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)\n"
        "    obj = bpy.context.object\n"
        "    obj.name = name\n"
        "    return obj\n"
        "def add_curve(name, points):\n"
        "    curve = bpy.data.curves.new(name + '_curve', 'CURVE')\n"
        "    obj = bpy.data.objects.new(name, curve)\n"
        "    return obj\n"
        "screen = add_screen('projection_screen', (0, 0, 0))\n"
        "ray = add_curve('incident_white_ray', [])\n"
    )
    import ast
    tree = ast.parse(code)
    assert {"projection_screen", "incident_white_ray"} <= _all_created_object_names(tree)


def test_keyframe_schedule_records_value_from_preceding_assignment():
    code = (
        "import bpy\n"
        "trace = bpy.data.objects.new('trace_curve', None)\n"
        "trace.bevel_factor_end = 0.0\n"
        "trace.keyframe_insert('bevel_factor_end', frame=1)\n"
        "trace.bevel_factor_end = 1.0\n"
        "trace.keyframe_insert('bevel_factor_end', frame=120)\n"
    )
    import ast
    sched = _keyframe_schedule(ast.parse(code))
    events = sched["trace_curve"]
    assert ("bevel_factor_end", 1, 0.0) in events
    assert ("bevel_factor_end", 120, 1.0) in events


def test_keyframe_schedule_resolves_bpy_context_object_rename():
    code = (
        "import bpy\n"
        "bpy.ops.curve.primitive_bezier_curve_add()\n"
        "trace = bpy.context.object\n"
        "trace.name = 'trajectory_trace'\n"
        "trace.bevel_factor_end = 0.0\n"
        "trace.keyframe_insert('bevel_factor_end', frame=1)\n"
        "trace.bevel_factor_end = 1.0\n"
        "trace.keyframe_insert('bevel_factor_end', frame=120)\n"
    )
    import ast
    sched = _keyframe_schedule(ast.parse(code))
    assert "trajectory_trace" in sched
    assert ("bevel_factor_end", 1, 0.0) in sched["trajectory_trace"]


def test_ramp_value_at_step_interpolates():
    events = [
        ("bevel_factor_end", 1, 0.0),
        ("bevel_factor_end", 72, 0.0),
        ("bevel_factor_end", 432, 1.0),
    ]
    assert _ramp_value_at(events, "bevel_factor_end", 1) == 0.0
    assert _ramp_value_at(events, "bevel_factor_end", 36) == 0.0
    assert _ramp_value_at(events, "bevel_factor_end", 200) == 0.0
    assert _ramp_value_at(events, "bevel_factor_end", 432) == 1.0
    assert _ramp_value_at(events, "missing_path", 50) is None


# ---------------- Rules ----------------


def test_missing_object_creation_when_critic_and_ast_agree():
    code = (
        "import bpy\n"
        "tip = bpy.data.objects.new('tip', None)\n"
        "j1 = bpy.data.objects.new('J1', None)\n"
        "j2 = bpy.data.objects.new('J2', None)\n"
    )
    issue = _issue("The label 'E' is absent from the frame; only J1 and J2 are visible.")
    sb = _storyboard(object_names=["tip", "J1", "J2"])
    contracts = {"node_01": _contract(required_labels=["E", "J1", "J2"])}
    report = cross_reference_critic_findings(
        concept_id="forward_kinematics_chain",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    rule_ids = {f.rule_id for f in report.findings}
    assert "missing_object_creation" in rule_ids
    e_finding = next(
        f for f in report.findings
        if f.rule_id == "missing_object_creation" and "'E'" in f.diagnosis
    )
    assert e_finding.severity == "actionable"
    assert "E" in e_finding.suggested_fix


def test_missing_object_creation_skipped_when_object_exists_in_ast():
    code = (
        "import bpy\n"
        "e = bpy.data.objects.new('E', None)\n"
    )
    issue = _issue("The label 'E' is absent from the frame.")
    sb = _storyboard(object_names=["E"])
    contracts = {"node_01": _contract(required_labels=["E"])}
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    rule_ids = {f.rule_id for f in report.findings}
    assert "missing_object_creation" not in rule_ids


def test_missing_object_creation_skipped_for_existing_label_prefixed_object():
    code = (
        "import bpy\n"
        "label = bpy.data.objects.new('label_lod1', None)\n"
    )
    issue = _issue("The required label 'LOD1' is missing or unreadable.")
    sb = _storyboard(object_names=["label_lod1"])
    contracts = {"node_01": _contract(required_labels=["LOD1"])}
    report = cross_reference_critic_findings(
        concept_id="texture_mipmap_lod",
        iteration=1,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    rule_ids = {f.rule_id for f in report.findings}
    assert "missing_object_creation" not in rule_ids


def test_misnamed_object_when_contract_E_but_ast_tip():
    code = (
        "import bpy\n"
        "tip = bpy.data.objects.new('tip', None)\n"
        "j1 = bpy.data.objects.new('J1', None)\n"
    )
    issue = _issue("The end-effector is labeled 'tip' instead of the required 'E'.")
    sb = _storyboard(object_names=["tip", "J1"])
    contracts = {"node_01": _contract(required_labels=["E"])}
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    misnamed = [f for f in report.findings if f.rule_id == "misnamed_object"]
    assert len(misnamed) >= 1
    f = misnamed[0]
    assert "E" in f.diagnosis
    assert "tip" in f.diagnosis
    assert "rename" in f.suggested_fix.lower()


def test_keyframe_ramp_too_late_when_frame_in_zero_region():
    code = (
        "import bpy\n"
        "tc = bpy.data.objects.new('trace_curve', None)\n"
        "tc.bevel_factor_end = 0.0\n"
        "tc.keyframe_insert('bevel_factor_end', frame=1)\n"
        "tc.bevel_factor_end = 0.0\n"
        "tc.keyframe_insert('bevel_factor_end', frame=72)\n"
        "tc.bevel_factor_end = 1.0\n"
        "tc.keyframe_insert('bevel_factor_end', frame=432)\n"
    )
    issue = _issue(
        "No end-effector trajectory trace is visible in the frame.",
        frame_idx=36,
    )
    sb = _storyboard(object_names=["trace_curve"])
    contracts = {"node_01": _contract()}
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    ramp_findings = [f for f in report.findings if f.rule_id == "keyframe_ramp_too_late"]
    assert len(ramp_findings) == 1
    assert "trace_curve" in ramp_findings[0].diagnosis
    assert "bevel_factor_end" in ramp_findings[0].ast_evidence


def test_keyframe_ramp_passes_when_frame_in_nonzero_region():
    code = (
        "import bpy\n"
        "tc = bpy.data.objects.new('trace_curve', None)\n"
        "tc.bevel_factor_end = 0.0\n"
        "tc.keyframe_insert('bevel_factor_end', frame=1)\n"
        "tc.bevel_factor_end = 0.5\n"
        "tc.keyframe_insert('bevel_factor_end', frame=60)\n"
        "tc.bevel_factor_end = 1.0\n"
        "tc.keyframe_insert('bevel_factor_end', frame=200)\n"
    )
    # Critic claims trace missing at frame 100 — but the ramp is already at
    # 0.5 there, so AST does NOT corroborate. No finding should fire.
    issue = _issue("No trajectory trace visible at this frame.", frame_idx=100)
    sb = _storyboard(object_names=["trace_curve"])
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts={"node_01": _contract()},
    )
    assert not any(f.rule_id == "keyframe_ramp_too_late" for f in report.findings)


def test_object_hidden_at_frame_when_hide_render_true():
    code = (
        "import bpy\n"
        "axes = bpy.data.objects.new('axes_z', None)\n"
        "axes.hide_render = True\n"
        "axes.keyframe_insert('hide_render', frame=1)\n"
        "axes.hide_render = False\n"
        "axes.keyframe_insert('hide_render', frame=120)\n"
    )
    issue = _issue("The 'axes_z' arrow is not visible in the frame.", frame_idx=48)
    sb = _storyboard(object_names=["axes_z"])
    contracts = {"node_01": _contract(required_labels=["axes_z"])}
    report = cross_reference_critic_findings(
        concept_id="slerp",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts=contracts,
    )
    hidden = [f for f in report.findings if f.rule_id == "object_hidden_at_frame"]
    assert len(hidden) == 1
    assert "axes_z" in hidden[0].diagnosis
    assert "48" in hidden[0].diagnosis


def test_no_finding_when_critic_lacks_corresponding_ast_token():
    code = (
        "import bpy\n"
        "e = bpy.data.objects.new('E', None)\n"
    )
    issue = _issue("The lighting is too dim and there is general murkiness.")
    sb = _storyboard(object_names=["E"])
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[issue],
        scene_code=code,
        storyboard=sb,
        visual_contracts={"node_01": _contract(required_labels=["E"])},
    )
    assert report.findings == []


def test_empty_report_when_critic_issues_empty():
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[],
        scene_code="import bpy\n",
        storyboard=_storyboard(object_names=[]),
        visual_contracts={},
    )
    assert report.findings == []
    assert report.actionable_count == 0


def test_empty_report_when_scene_code_is_blank():
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[_issue("The label 'E' is absent.")],
        scene_code="",
        storyboard=_storyboard(object_names=[]),
        visual_contracts={},
    )
    assert report.findings == []


def test_syntax_error_in_scene_code_returns_empty_report():
    report = cross_reference_critic_findings(
        concept_id="fk",
        iteration=0,
        critic_issues=[_issue("The label 'E' is absent.")],
        scene_code="def bad(\n",  # syntax error
        storyboard=_storyboard(object_names=[]),
        visual_contracts={},
    )
    assert report.findings == []


def test_format_for_coder_empty_when_no_findings():
    report = CrossReferenceReport(concept_id="fk", iteration=0)
    assert format_cross_reference_for_coder(report) == ""


def test_format_for_coder_includes_diagnosis_and_evidence():
    report = CrossReferenceReport(
        concept_id="fk",
        iteration=0,
        findings=[CrossReferenceFinding(
            rule_id="missing_object_creation",
            severity="actionable",
            diagnosis="Critic reports 'E' missing; AST confirms not created.",
            critic_source="iter00 shot=node_01 frame=36 block/concept_mismatch",
            ast_evidence="scene.py created: tip, J1, J2",
            suggested_fix="Add bpy.data.objects.new('E', ...).",
        )],
    )
    text = format_cross_reference_for_coder(report)
    assert "CRITIC × AST CROSS-REFERENCE" in text
    assert "missing_object_creation" in text
    assert "Add bpy.data.objects.new('E', ...)" in text
    assert "fix:" in text


def test_report_serialization_is_json_safe():
    report = CrossReferenceReport(
        concept_id="fk",
        iteration=0,
        findings=[CrossReferenceFinding(
            rule_id="missing_object_creation",
            severity="actionable",
            diagnosis="d",
            critic_source="c",
            ast_evidence="a",
            suggested_fix="s",
        )],
    )
    import json
    data = json.loads(cross_reference_report_to_json(report))
    assert data["concept_id"] == "fk"
    assert data["iteration"] == 0
    assert data["actionable_count"] == 1
    assert isinstance(data["findings"], list)
    assert data["findings"][0]["rule_id"] == "missing_object_creation"
