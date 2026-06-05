"""Test the YAML model-config loader."""

from __future__ import annotations

from pathlib import Path

from cg_tutor.config import AgentModel, get_agent_model, get_models


def test_get_models_loads_default():
    """Default `configs/models_api.yaml` should produce the 5 agent entries."""
    ms = get_models()
    for name in ("concept_decomposer", "storyboard", "blender_coder",
                 "render_critic", "latex_overlay"):
        assert name in ms


def test_agent_model_chain_skips_none(tmp_path: Path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "agents:\n"
        "  a: {primary: codex-cli/x, fallback: claude-cli/y}\n"
        "  b: {primary: codex-cli/x, fallback: null}\n"
        "  c: {primary: null, fallback: null}\n"
    )
    ms = get_models(cfg)
    assert ms["a"].chain == ("codex-cli/x", "claude-cli/y")
    assert ms["b"].chain == ("codex-cli/x",)
    assert ms["c"].chain == ()


def test_missing_agent_returns_empty(tmp_path: Path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text("agents:\n  a:\n    primary: codex-cli/x\n")
    m = get_agent_model("does_not_exist", path=cfg)
    assert m == AgentModel(primary=None, fallback=None)
    assert m.chain == ()


def test_missing_file_returns_empty_dict(tmp_path: Path):
    """A missing file should be treated as an empty config rather than
    raising — callers fall back to their own hard-coded defaults."""
    ms = get_models(tmp_path / "missing.yaml")
    assert ms == {}
