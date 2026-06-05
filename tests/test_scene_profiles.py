from cg_tutor.agents import profile_generator
from cg_tutor.scene_profiles import (
    base_profile,
    choose_base_profile,
    format_scene_profile_for_prompt,
    profile_requires_llm,
    profile_seed,
    sanitize_profile,
)
from cg_tutor.schemas import Narrative, NarrativeNode


def test_application_scene_spec_disables_video_formula_expectations():
    spec = {
        "concept_id": "cinematic_ray_demo",
        "presentation_mode": "application_scene",
        "graphics_principles": ["Rays bounce recursively through the scene."],
        "visual_style_constraints": [
            "Do not use formula overlays or contribution breakdown tiles."
        ],
    }

    assert spec["presentation_mode"] == "application_scene"
    assert "must_show_formulas" not in spec
    assert "graphics_principles" in spec
    constraints = " ".join(spec["visual_style_constraints"]).lower()
    assert "formula overlays" in constraints
    assert "contribution breakdown tiles" in constraints


def test_cinematic_application_disables_formula_overlay():
    profile = base_profile("cinematic_application")

    assert profile.overlay_policy["formula_style"] == "none"
    assert profile.overlay_policy["disable_formula_overlay"] is True
    assert "diegetic_hud" in profile.allowed_helpers
    assert "formula_overlay" not in profile.allowed_helpers
    assert "formula_overlay" in profile.forbidden_helpers
    assert "contribution_breakdown_tiles" in profile.forbidden_helpers
    assert profile.persistent_anchors
    assert "black void with isolated primitives" in profile.forbidden_abstractions


def test_teaching_profile_keeps_formula_overlay_available():
    profile = base_profile("vector_teaching")

    assert profile.overlay_policy["formula_style"] == "teaching_overlay"
    assert "formula_overlay" in profile.allowed_helpers
    assert "formula_overlay" not in profile.forbidden_helpers


def test_formatted_profile_includes_adaptive_critic_rubric():
    text = format_scene_profile_for_prompt(base_profile("cinematic_application"))

    assert "adaptive_critic_rubric" in text
    assert "formula overlays" in text
    assert "persistent_anchors" in text
    assert "forbidden_abstractions" in text


def test_auto_profile_requires_llm_and_chooses_cinematic_for_showcase():
    spec = {
        "concept_id": "whitted_ray_tracing",
        "scene_profile": "auto",
        "scene_profile_hint": "cinematic showcase with no classroom arrows",
    }

    assert profile_requires_llm(spec)
    assert choose_base_profile(spec) == "cinematic_application"
    assert profile_seed(spec).base_profile == "cinematic_application"


def test_application_scene_presentation_mode_chooses_cinematic_profile():
    spec = {
        "concept_id": "custom_scene",
        "presentation_mode": "application_scene",
        "key_points": ["A practical visual scene with no classroom panels."],
    }

    assert choose_base_profile(spec) == "cinematic_application"


def test_application_scene_seed_adds_rainy_window_grounding():
    spec = {
        "concept_id": "rainy_window_ray_tracing",
        "presentation_mode": "application_scene",
        "scene_profile": "cinematic_application",
        "key_points": ["rainy window with an OPEN neon sign"],
    }

    profile = profile_seed(spec)

    assert "window_frame" in profile.persistent_anchors
    assert "neon_sign_OPEN" in profile.persistent_anchors
    assert any("blank glowing rectangle" in x for x in profile.forbidden_abstractions)


def test_sanitizer_keeps_grounding_fields_and_clips_lists():
    resolved = sanitize_profile({
        "profile_id": "custom",
        "base_profile": "cinematic_application",
        "persistent_anchors": [
            "window_frame",
            "window_frame",
            "x" * 260,
        ],
        "spatial_relationships": ["raindrops are in front of neon_sign_OPEN"],
        "forbidden_abstractions": ["blank glowing sign rectangle"],
    })

    profile = resolved.profile
    assert profile.persistent_anchors[0] == "window_frame"
    assert profile.persistent_anchors[1].endswith("...")
    assert len(profile.persistent_anchors) == 2
    assert profile.spatial_relationships == [
        "raindrops are in front of neon_sign_OPEN"
    ]
    assert profile.forbidden_abstractions == ["blank glowing sign rectangle"]


def test_known_builtin_profile_still_requires_llm_review():
    spec = {"concept_id": "phong_lighting", "scene_profile": "vector_teaching"}

    assert profile_requires_llm(spec)
    assert profile_seed(spec).profile_id == "vector_teaching"


def test_explicit_grounding_seed_applies_to_teaching_profiles():
    spec = {
        "concept_id": "dolly_zoom",
        "scene_profile": "transformation_demo",
        "persistent_anchors": ["hero_lighthouse", "marker_post_1"],
        "spatial_relationships": ["marker_post_1 sits behind hero_lighthouse"],
        "forbidden_abstractions": ["camera icon floating above the hero"],
    }

    profile = profile_seed(spec)

    assert profile.base_profile == "transformation_demo"
    assert "hero_lighthouse" in profile.persistent_anchors
    assert "marker_post_1" in profile.persistent_anchors
    assert "marker_post_1 sits behind hero_lighthouse" in profile.spatial_relationships
    assert "camera icon floating above the hero" in profile.forbidden_abstractions


def test_inline_incomplete_profile_requires_completion():
    spec = {
        "concept_id": "demo",
        "scene_profile": {
            "profile_id": "custom_demo",
            "base_profile": "cinematic_application",
        },
    }

    assert profile_requires_llm(spec)


def test_sanitizer_forbidden_helpers_win_conflicts():
    resolved = sanitize_profile({
        "profile_id": "custom",
        "base_profile": "cinematic_application",
        "allowed_helpers": ["thin tracer", "thick arrows"],
        "forbidden_helpers": ["thick arrows"],
    })

    assert "thick_arrows" in resolved.profile.forbidden_helpers
    assert "thick_arrows" not in resolved.profile.allowed_helpers
    assert any("forbidden wins" in w for w in resolved.validation["warnings"])


def test_sanitizer_keeps_adaptive_critic_rubric():
    resolved = sanitize_profile({
        "profile_id": "custom",
        "base_profile": "cinematic_application",
        "critic_rubric": {
            "must_have": ["glass refraction visible"],
            "must_avoid": ["formula overlays"],
            "blocking_conditions": ["scene looks like a classroom diagram"],
        },
    })

    assert resolved.profile.critic_rubric["must_have"] == [
        "glass refraction visible"
    ]
    assert "formula overlays" in resolved.profile.critic_rubric["must_avoid"]


def test_sanitizer_drops_unknown_rubric_keys_and_clips_lists():
    long_text = "x" * 260
    resolved = sanitize_profile({
        "profile_id": "custom",
        "base_profile": "cinematic_application",
        "critic_rubric": {
            "must_have": [
                "glass refraction visible",
                "thin laser trace visible",
                "security HUD looks diegetic",
                "mirror reflection visible",
                "shadow cue visible",
                "Fresnel rim visible",
                "extra item should be clipped",
            ],
            "must_avoid": [long_text],
            "aesthetic_goals": ["premium mood"],
            "blocking_conditions": ["classroom diagram"],
            "score_weights": {"framing": 0.4, "semantic": 0.6},
            "unexpected": ["ignored"],
        },
    })

    rubric = resolved.profile.critic_rubric
    assert set(rubric) == {
        "must_have",
        "must_avoid",
        "aesthetic_goals",
        "blocking_conditions",
    }
    assert len(rubric["must_have"]) == 6
    assert rubric["must_avoid"][0].endswith("...")
    warnings = " ".join(resolved.validation["warnings"])
    assert "score_weights" in warnings
    assert "unexpected" in warnings


def test_resolve_scene_profile_reviews_known_profile_with_llm(monkeypatch):
    calls = {"n": 0}

    class FakeClient:
        def complete_json(self, **_kw):
            calls["n"] += 1
            return {
                "profile_id": "phong_reviewed",
                "base_profile": "vector_teaching",
                "visual_goal": "Teaching scene tuned for Phong vectors.",
                "allowed_helpers": ["arrows", "formula_overlay"],
                "forbidden_helpers": ["unreadable_control_labels"],
                "preferred_geometry": ["readable_arrow_vectors"],
                "overlay_policy": {"formula_style": "teaching_overlay"},
                "critic_priorities": ["verify N/L/V/R labels"],
                "critic_rubric": {
                    "must_have": ["N/L/V/R arrows are readable"],
                    "must_avoid": ["missing specular highlight"],
                    "aesthetic_goals": ["clean teaching layout"],
                    "blocking_conditions": ["vectors are absent"],
                },
                "repair_policy": ["restore full vector arrows"],
                "persistent_anchors": [],
                "spatial_relationships": [],
                "forbidden_abstractions": [],
            }

    monkeypatch.setattr(
        profile_generator.LLMClient,
        "from_chain",
        staticmethod(lambda *_args, **_kw: FakeClient()),
    )
    narrative = Narrative(
        concept_id="phong_lighting",
        nodes=[
            NarrativeNode(
                id="node_01",
                title="Vectors",
                description="Show Phong vectors.",
                formulas=["I = I_a + I_d + I_s"],
                duration_sec=2.0,
                visual_intent="A sphere with N/L/V/R arrows.",
            )
        ],
    )

    resolved = profile_generator.resolve_scene_profile(
        {"concept_id": "phong_lighting", "scene_profile": "vector_teaching"},
        narrative,
        model="gpt/gpt-5.5",
    )

    assert calls["n"] == 1
    assert resolved.source == "llm"
    assert resolved.profile.profile_id == "phong_reviewed"
    assert "N/L/V/R arrows are readable" in resolved.profile.critic_rubric["must_have"]


def test_base_profile_is_valid_scene_profile():
    profile = base_profile("curve_construction")

    assert profile.profile_id == "curve_construction"
    assert profile.base_profile == "curve_construction"
    assert profile.critic_rubric["must_have"]
