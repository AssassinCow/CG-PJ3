"""Cooldown registry + ChainedLLMClient integration tests.

The pipeline policy: when a provider hits a timeout / 504 / 524, park it
for 3 minutes; if transient errors come in frequently, escalate to 5
minutes. While parked, ALL calls to that provider block (sleep) until
the window expires — even if there is a fallback provider, we still
wait, because the relay tends to recover faster when it isn't being
hammered.
"""

from __future__ import annotations

import time

import pytest

from cg_tutor import llm_client as lc


@pytest.fixture(autouse=True)
def _clear_registry():
    lc._reset_cooldown_registry()
    yield
    lc._reset_cooldown_registry()


def test_timeout_arms_three_minute_cooldown():
    cd, reason = lc._record_transient_error("slugA", RuntimeError("request timed out"))
    assert cd == 180.0
    assert reason == "timeout"
    assert 179 < lc._cooldown_seconds_left("slugA") <= 180


def test_504_counts_as_timeout():
    cd, _ = lc._record_transient_error("slugB", RuntimeError("HTTP 504"))
    assert cd == 180.0


def test_524_counts_as_timeout():
    cd, _ = lc._record_transient_error("slugB", RuntimeError("error 524 from cloudflare"))
    assert cd == 180.0


def test_three_frequent_errors_escalate_to_five_minutes():
    # Three non-timeout transient errors in the default 300s window
    # tip the slug into the longer cooldown.
    lc._record_transient_error("slugC", RuntimeError("connection reset"))
    lc._record_transient_error("slugC", RuntimeError("rate limit hit"))
    cd, reason = lc._record_transient_error("slugC", RuntimeError("connection reset"))
    assert cd == 300.0
    assert "frequent" in reason


def test_existing_longer_cooldown_is_not_shortened():
    # Park slugD far into the future, with no recent errors so the
    # timeout case takes the 3min branch (not the "frequent" 5min one).
    state = lc._cooldown_for("slugD")
    far_future = time.monotonic() + 1_000.0
    state.ready_at = far_future
    # A fresh timeout would normally arm a 180s cooldown; since
    # ``now + 180`` is well before ``far_future``, ready_at must stay put.
    lc._record_transient_error("slugD", RuntimeError("timed out"))
    assert lc._COOLDOWN_REGISTRY["slugD"].ready_at == far_future


def test_per_slug_isolation():
    lc._record_transient_error("slugE", RuntimeError("timed out"))
    assert lc._cooldown_seconds_left("slugE") > 0
    assert lc._cooldown_seconds_left("slugF") == 0


def test_non_transient_error_is_not_recorded(monkeypatch):
    # The registry update only runs from the chain loop and only when
    # _is_transient(e) is True. We assert _is_transient gates correctly.
    assert lc._is_transient(RuntimeError("connection reset")) is True
    assert lc._is_transient(RuntimeError("invalid api key")) is False


def test_api_provider_aliases_canonicalize_to_three_families():
    assert lc._canonical_api_provider("gpt", "gpt-5.5") == "gpt"
    assert lc._canonical_api_provider("gemini", "gemini-3.1-pro-preview") == "gemini"
    assert lc._canonical_api_provider("claude", "claude-sonnet-4-6") == "claude"
    assert lc._canonical_api_provider("claude-api", "claude-sonnet-4-6") == "claude"
    assert lc._canonical_api_provider("codex-api", "gpt-5.5") == "gpt"
    assert lc._canonical_api_provider("codex-api", "gemini-3.1-pro-preview") == "gemini"


def test_provider_base_urls_default_to_official_sdk_endpoints(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    assert lc._provider_base_url("openai") is None
    assert lc._provider_base_url("anthropic") is None
    assert lc._provider_base_url("gemini") is None


def test_provider_base_urls_use_explicit_local_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/openai")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid/anthropic")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.invalid/gemini")

    assert lc._provider_base_url("openai") == "https://example.invalid/openai"
    assert lc._provider_base_url("anthropic") == "https://example.invalid/anthropic"
    assert lc._provider_base_url("gemini") == "https://example.invalid/gemini"


# ---------------------------------------------------------------------------
# ChainedLLMClient integration: cooldown always blocks, even with fallback


class _FakeClient:
    def __init__(self, slug, *, fail_with=None, succeed_with="ok"):
        self.slug = slug
        self.fail_with = fail_with
        self.succeed_with = succeed_with
        self.calls = 0

    def complete(self, **_kw):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return self.succeed_with


def test_chain_sleeps_when_primary_is_cooled_down_even_with_fallback(monkeypatch):
    """Single provider OR primary+fallback — either way, we wait."""
    sleeps: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    # Simulate a prior timeout so primary is already in cooldown.
    lc._record_transient_error("p1", RuntimeError("timed out"))

    primary = _FakeClient("p1", succeed_with="primary-ok")
    fallback = _FakeClient("p2", succeed_with="fallback-ok")
    chain = lc.ChainedLLMClient(clients=(primary, fallback))

    out = chain.complete(system="s", user="u")

    # Primary slept through its cooldown, then served the call.
    assert out == "primary-ok"
    assert primary.calls == 1
    assert fallback.calls == 0
    assert any(s >= 179 for s in sleeps), f"expected ~180s sleep, got {sleeps}"


def test_chain_sleeps_for_only_provider(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    lc._record_transient_error("solo", RuntimeError("timed out"))

    only = _FakeClient("solo", succeed_with="solo-ok")
    chain = lc.ChainedLLMClient(clients=(only,))

    out = chain.complete(system="s", user="u")
    assert out == "solo-ok"
    assert only.calls == 1
    assert any(s >= 179 for s in sleeps)


def test_chain_arms_cooldown_on_timeout_then_falls_through(monkeypatch):
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    primary = _FakeClient("p1", fail_with=RuntimeError("timed out after 90s"))
    fallback = _FakeClient("p2", succeed_with="fallback-ok")
    chain = lc.ChainedLLMClient(clients=(primary, fallback))

    out = chain.complete(system="s", user="u")
    assert out == "fallback-ok"
    # Primary now has a cooldown armed for next time.
    assert lc._cooldown_seconds_left("p1") > 0
    assert lc._cooldown_seconds_left("p2") == 0


def test_chain_no_cooldown_when_primary_succeeds(monkeypatch):
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    primary = _FakeClient("p1", succeed_with="ok")
    chain = lc.ChainedLLMClient(clients=(primary,))

    chain.complete(system="s", user="u")
    assert lc._cooldown_seconds_left("p1") == 0
