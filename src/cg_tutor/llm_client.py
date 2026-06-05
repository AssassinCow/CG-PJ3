"""Unified LLM client with provider-chain fallback.

Two public entry points:

    LLMClient.from_model("claude-cli/opus")           # single provider
    LLMClient.from_chain(["claude-cli/opus",          # primary
                          "claude-cli/sonnet"])       # fallback

Both return objects with the same ``complete()`` / ``complete_json()``
API. The chain variant tries each provider in order; on
:class:`RuntimeError` or :class:`subprocess.TimeoutExpired` it moves to
the next entry. Retries are disabled by default so a slow provider does
not burn another full request timeout before fallback.

Providers (lazy-loaded SDK imports — no extra installs unless you use
that path):

  - ``claude-cli/<model>``  subprocess of ``claude -p``
  - ``codex-cli/<model>``   subprocess of ``codex exec``
  - ``claude/<model>``      Claude models via OpenAI-compatible API
  - ``gpt/<model>``         GPT/OpenAI models via OpenAI-compatible API
  - ``gemini/<model>``      Gemini models via OpenAI-compatible API
  - ``claude-api/<model>``  legacy alias for ``claude/<model>``
  - ``codex-api/<model>``   legacy alias for ``gpt/<model>`` or
                            ``gemini/<model>`` depending on model name
  - ``openai/<model>``      ``openai`` Python SDK
  - ``anthropic/<model>``   ``anthropic`` Python SDK
  - ``google/<model>``      ``google-genai`` or ``google-generativeai`` SDK

When the caller passes ``raw_out_dir=...`` to ``complete()``, every
provider attempt's raw output (or error repr) lands as
``raw_<provider>_<model>.txt`` under that directory so failed runs are
diagnosable.
"""

from __future__ import annotations

import json
import base64
import mimetypes
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence
from cg_tutor._logging import get_logger
from cg_tutor import terminal_ui as ui


log = get_logger(__name__)

# Auto-load .env if python-dotenv is installed; quiet no-op otherwise.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

ResponseFormat = Literal["text", "json"]


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "temporarily", "rate limit",
    "5xx", "502", "503", "504", "ECONN",
)

_TIMEOUT_HINTS = ("timeout", "timed out", "504", "524")


def _is_transient(err: BaseException) -> bool:
    s = str(err).lower()
    return any(h in s for h in _TRANSIENT_HINTS)


def _is_timeout(err: BaseException) -> bool:
    s = str(err).lower()
    return any(h in s for h in _TIMEOUT_HINTS)


# ---------------------------------------------------------------------------
# Per-provider cooldown registry.
#
# When the relay times out (504/524) or just hangs past our request budget,
# repeatedly hammering it makes the symptom worse and burns the whole
# pipeline budget on errors. We track per-slug cooldown windows: a single
# timeout parks the provider for N minutes; ``frequent_count`` transient
# errors within a rolling window escalate to a longer cooldown. The
# ChainedLLMClient consults this before each call: it skips a cooled-down
# provider when a fallback exists, and sleeps through the cooldown only
# when there is no alternative.

@dataclass
class _CooldownState:
    ready_at: float = 0.0  # monotonic; calls block until then
    recent_errors: list[float] = field(default_factory=list)


_COOLDOWN_REGISTRY: dict[str, _CooldownState] = {}


def _cooldown_config() -> tuple[float, float, float, int]:
    return (
        float(os.environ.get("CG_TUTOR_COOLDOWN_TIMEOUT_SEC", "180")),
        float(os.environ.get("CG_TUTOR_COOLDOWN_FREQUENT_SEC", "300")),
        float(os.environ.get("CG_TUTOR_COOLDOWN_FREQUENT_WINDOW_SEC", "300")),
        int(os.environ.get("CG_TUTOR_COOLDOWN_FREQUENT_COUNT", "3")),
    )


def _cooldown_for(slug: str) -> _CooldownState:
    state = _COOLDOWN_REGISTRY.get(slug)
    if state is None:
        state = _CooldownState()
        _COOLDOWN_REGISTRY[slug] = state
    return state


def _cooldown_seconds_left(slug: str) -> float:
    state = _COOLDOWN_REGISTRY.get(slug)
    if state is None:
        return 0.0
    return max(0.0, state.ready_at - time.monotonic())


def _reset_cooldown_registry() -> None:
    """Test helper. Not exported."""
    _COOLDOWN_REGISTRY.clear()


def _record_transient_error(slug: str, err: BaseException) -> tuple[float, str]:
    """Record a transient error; maybe arm a cooldown.

    Returns ``(cooldown_seconds_applied, human_reason)``. ``0.0`` means the
    error was logged but did not trip the threshold.
    """
    timeout_cd, frequent_cd, window, frequent_count = _cooldown_config()
    state = _cooldown_for(slug)
    now = time.monotonic()
    state.recent_errors = [t for t in state.recent_errors if now - t < window]
    state.recent_errors.append(now)

    new_cd = 0.0
    reason = ""
    if len(state.recent_errors) >= frequent_count:
        new_cd = frequent_cd
        reason = f"frequent ({len(state.recent_errors)} errors in {int(window)}s)"
    elif _is_timeout(err):
        new_cd = timeout_cd
        reason = "timeout"

    if new_cd <= 0:
        return 0.0, ""
    candidate = now + new_cd
    if candidate > state.ready_at:
        state.ready_at = candidate
        return new_cd, reason
    # Existing cooldown is longer; keep it.
    return max(0.0, state.ready_at - now), reason


def _slug_safe(slug: str) -> str:
    return slug.replace("/", "_").replace(":", "_")


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _image_base64_source(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "base64",
        "media_type": mime,
        "data": data,
    }


def _first_env(*names: str) -> str | None:
    return next((os.environ.get(name) for name in names if os.environ.get(name)), None)


def _provider_base_url(kind: Literal["openai", "anthropic", "gemini"]) -> str | None:
    """Optional explicit SDK base URL.

    The public repository defaults to official provider endpoints by leaving
    base_url unset. These variables are only for users who intentionally
    configure their own compatible endpoint locally.
    """
    env_names = {
        "openai": ("OPENAI_BASE_URL",),
        "anthropic": ("ANTHROPIC_BASE_URL",),
        "gemini": ("GOOGLE_BASE_URL", "GEMINI_BASE_URL"),
    }[kind]
    return _first_env(*env_names)


def _ensure_json_instruction(system: str, user: str) -> tuple[str, str]:
    """Some OpenAI-compatible endpoints require 'json' in messages when
    response_format=json_object is set.
    """
    if "json" in user.lower():
        return system, user
    return system, f"{user}\n\nReturn a valid json object."


def _chat_content(resp) -> str:
    """Extract text from standard SDK responses and relay string returns."""
    if isinstance(resp, str):
        return resp
    return resp.choices[0].message.content or ""


def _text_preview(text: str, limit: int = 300) -> str:
    if not text:
        return "<empty response>"
    s = text.replace("\n", "\\n")
    return s[:limit] + ("..." if len(s) > limit else "")


def _canonical_api_provider(provider: str, model: str) -> str:
    """Map legacy provider aliases onto the API families used by configs."""
    if provider == "claude-api":
        return "claude"
    if provider == "codex-api":
        return "gemini" if "gemini" in model.lower() else "gpt"
    return provider


# ---------------------------------------------------------------------------
# Single-provider client


class LLMClient:
    """Single-provider client. Use ``from_chain`` for multi-provider."""

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    @classmethod
    def from_model(cls, slug: str) -> "LLMClient":
        if "/" not in slug:
            raise ValueError(f"model slug must be provider/model, got {slug!r}")
        provider, model = slug.split("/", 1)
        return cls(provider=provider, model=model)

    @classmethod
    def from_chain(
        cls,
        slugs: Sequence[str],
        *,
        max_retries: int = 0,
        backoff_base: float = 2.0,
    ) -> "ChainedLLMClient":
        if not slugs:
            raise ValueError("from_chain requires at least one slug")
        clients = tuple(cls.from_model(s) for s in slugs)
        return ChainedLLMClient(
            clients=clients,
            max_retries=max_retries,
            backoff_base=backoff_base,
        )

    @property
    def slug(self) -> str:
        return f"{self.provider}/{self.model}"

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        api_provider = _canonical_api_provider(self.provider, self.model)
        if self.provider == "openai":
            return self._call_openai(system, user, response_format, temperature, max_tokens)
        if api_provider == "claude":
            return self._call_named_openai_compatible(
                system, user, response_format, temperature, max_tokens,
                key_env="CLAUDE_API_KEY",
                base_url_env="CLAUDE_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if api_provider == "gpt":
            return self._call_named_openai_compatible(
                system, user, response_format, temperature, max_tokens,
                key_env="GPT_API_KEY",
                base_url_env="GPT_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if api_provider == "gemini":
            return self._call_named_openai_compatible(
                system, user, response_format, temperature, max_tokens,
                key_env="GEMINI_API_KEY",
                base_url_env="GEMINI_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if self.provider == "anthropic":
            return self._call_anthropic(system, user, response_format, temperature, max_tokens)
        if self.provider == "google":
            return self._call_google(system, user, response_format, temperature, max_tokens)
        if self.provider == "claude-cli":
            return self._call_claude_cli(system, user)
        if self.provider == "codex-cli":
            return self._call_codex_cli(system, user)
        raise NotImplementedError(f"provider {self.provider!r} not wired yet")

    def complete_json(self, *, system: str, user: str, **kw) -> dict:
        text = self.complete(system=system, user=user, response_format="json", **kw)
        return _extract_json(text)

    def complete_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: Sequence[Path],
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        api_provider = _canonical_api_provider(self.provider, self.model)
        if self.provider == "openai":
            api_key = _first_env("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Need OPENAI_API_KEY in env")
            return self._call_openai_compatible_with_images(
                system=system,
                user=user,
                image_paths=image_paths,
                fmt=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=_provider_base_url("openai"),
            )
        if api_provider == "claude":
            return self._call_named_openai_compatible_with_images(
                system, user, image_paths, response_format, temperature, max_tokens,
                key_env="CLAUDE_API_KEY",
                base_url_env="CLAUDE_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if api_provider == "gpt":
            return self._call_named_openai_compatible_with_images(
                system, user, image_paths, response_format, temperature, max_tokens,
                key_env="GPT_API_KEY",
                base_url_env="GPT_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if api_provider == "gemini":
            return self._call_named_openai_compatible_with_images(
                system, user, image_paths, response_format, temperature, max_tokens,
                key_env="GEMINI_API_KEY",
                base_url_env="GEMINI_BASE_URL",
                fallback_key_envs=("OPENAI_API_KEY",),
            )
        if self.provider == "anthropic":
            return self._call_anthropic_with_images(
                system, user, image_paths, response_format, temperature, max_tokens
            )
        if self.provider == "google":
            return self._call_google_with_images(
                system, user, image_paths, response_format, temperature, max_tokens
            )
        raise NotImplementedError(
            f"provider {self.provider!r} does not support image input"
        )

    # ----- providers ----------------------------------------------------

    def _call_openai(self, system, user, fmt, temperature, max_tokens) -> str:
        api_key = _first_env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Need OPENAI_API_KEY in env")
        return self._call_openai_compatible(
            system=system,
            user=user,
            fmt=fmt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=_provider_base_url("openai"),
        )

    def _call_named_openai_compatible(
        self,
        system,
        user,
        fmt,
        temperature,
        max_tokens,
        *,
        key_env: str,
        base_url_env: str,
        fallback_key_envs: tuple[str, ...] = (),
        fallback_base_url_envs: tuple[str, ...] = (),
    ) -> str:
        api_key = os.environ.get(key_env)
        if not api_key:
            api_key = next((os.environ.get(k) for k in fallback_key_envs
                            if os.environ.get(k)), None)
        if not api_key:
            names = ", ".join((key_env, *fallback_key_envs))
            raise RuntimeError(f"Need one of {names} in env for {self.provider}/<model>")
        return self._call_openai_compatible(
            system=system,
            user=user,
            fmt=fmt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=os.environ.get(base_url_env)
            or next((os.environ.get(k) for k in fallback_base_url_envs
                     if os.environ.get(k)), None)
        )

    def _call_named_openai_compatible_with_images(
        self,
        system,
        user,
        image_paths,
        fmt,
        temperature,
        max_tokens,
        *,
        key_env: str,
        base_url_env: str,
        fallback_key_envs: tuple[str, ...] = (),
        fallback_base_url_envs: tuple[str, ...] = (),
    ) -> str:
        api_key = os.environ.get(key_env)
        if not api_key:
            api_key = next((os.environ.get(k) for k in fallback_key_envs
                            if os.environ.get(k)), None)
        if not api_key:
            names = ", ".join((key_env, *fallback_key_envs))
            raise RuntimeError(f"Need one of {names} in env for {self.provider}/<model>")
        return self._call_openai_compatible_with_images(
            system=system,
            user=user,
            image_paths=image_paths,
            fmt=fmt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=os.environ.get(base_url_env)
            or next((os.environ.get(k) for k in fallback_base_url_envs
                     if os.environ.get(k)), None)
        )

    def _call_openai_compatible(
        self,
        *,
        system: str,
        user: str,
        fmt: ResponseFormat,
        temperature: float,
        max_tokens: int | None,
        api_key: str,
        base_url: str | None,
        model: str | None = None,
    ) -> str:
        from openai import OpenAI  # lazy

        if fmt == "json":
            system, user = _ensure_json_instruction(system, user)
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.environ.get("CG_TUTOR_API_TIMEOUT", "300")),
        )
        kw = dict(
            model=model or self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
        )
        if max_tokens and max_tokens > 0:
            kw["max_tokens"] = max_tokens
        if fmt == "json":
            kw["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            # Some reasoning models reject explicit temperature. Retry once
            # without it instead of burning the whole provider fallback chain.
            if "temperature" not in str(e).lower():
                raise
            kw.pop("temperature", None)
            resp = client.chat.completions.create(**kw)
        return _chat_content(resp)

    def _call_openai_compatible_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: Sequence[Path],
        fmt: ResponseFormat,
        temperature: float,
        max_tokens: int | None,
        api_key: str,
        base_url: str | None,
        model: str | None = None,
    ) -> str:
        from openai import OpenAI  # lazy

        if fmt == "json":
            system, user = _ensure_json_instruction(system, user)
        content: list[dict] = [{"type": "text", "text": user}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            })
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.environ.get("CG_TUTOR_API_TIMEOUT", "300")),
        )
        kw = dict(
            model=model or self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            temperature=temperature,
        )
        if max_tokens and max_tokens > 0:
            kw["max_tokens"] = max_tokens
        if fmt == "json":
            kw["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if "temperature" not in str(e).lower():
                raise
            kw.pop("temperature", None)
            resp = client.chat.completions.create(**kw)
        return _chat_content(resp)

    def _call_anthropic(self, system, user, fmt, temperature, max_tokens) -> str:
        from anthropic import Anthropic  # lazy

        api_key = _first_env("ANTHROPIC_API_KEY")
        base_url = _provider_base_url("anthropic")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        model = self.model
        if auth_token:
            client = Anthropic(auth_token=auth_token, base_url=base_url)
        elif api_key:
            client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            raise RuntimeError(
                "Need ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in env"
            )
        if fmt == "json":
            system, user = _ensure_json_instruction(system, user)
        msg_kw = dict(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        msg_kw["max_tokens"] = max_tokens or int(
            os.environ.get("CG_TUTOR_ANTHROPIC_MAX_TOKENS", "4096")
        )
        msg = client.messages.create(**msg_kw)
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    def _call_anthropic_with_images(
        self,
        system,
        user,
        image_paths,
        fmt,
        temperature,
        max_tokens,
    ) -> str:
        from anthropic import Anthropic  # lazy

        api_key = _first_env("ANTHROPIC_API_KEY")
        base_url = _provider_base_url("anthropic")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        model = self.model
        if auth_token:
            client = Anthropic(auth_token=auth_token, base_url=base_url)
        elif api_key:
            client = Anthropic(api_key=api_key, base_url=base_url)
        else:
            raise RuntimeError(
                "Need ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in env"
            )
        if fmt == "json":
            system, user = _ensure_json_instruction(system, user)
        content: list[dict] = [{"type": "text", "text": user}]
        for path in image_paths:
            content.append({
                "type": "image",
                "source": _image_base64_source(path),
            })
        msg_kw = dict(
            model=model,
            system=system,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            max_tokens=max_tokens or int(
                os.environ.get("CG_TUTOR_ANTHROPIC_MAX_TOKENS", "4096")
            ),
        )
        msg = client.messages.create(**msg_kw)
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    def _call_claude_cli(self, system: str, user: str) -> str:
        if not shutil.which("claude"):
            raise RuntimeError("`claude` CLI not on PATH (Claude Code not installed)")
        cmd = ["claude", "-p", "--output-format", "json"]
        if self.model and self.model not in ("default", "auto"):
            cmd += ["--model", self.model]
        if system:
            cmd += ["--system-prompt", system]
        try:
            proc = subprocess.run(
                cmd, input=user, capture_output=True, text=True,
                timeout=int(os.environ.get("CG_TUTOR_LLM_TIMEOUT", "600")),
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude CLI timed out: {e}") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exit={proc.returncode}: "
                f"{proc.stderr.strip()[:400]}"
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"claude CLI did not return JSON envelope: {proc.stdout[:400]}"
            ) from e
        if data.get("is_error"):
            raise RuntimeError(f"claude CLI reported error: {data.get('result', '')[:400]}")
        return data.get("result", "")

    def _call_codex_cli(self, system: str, user: str) -> str:
        if not shutil.which("codex"):
            raise RuntimeError("`codex` CLI not on PATH")
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if self.model and self.model not in ("default", "auto"):
            cmd += ["--model", self.model]
        prompt = f"{system}\n\n{user}" if system else user
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=int(os.environ.get("CG_TUTOR_LLM_TIMEOUT", "300")),
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"codex CLI timed out: {e}") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"codex CLI exit={proc.returncode}: "
                f"{proc.stderr.strip()[:400]}"
            )
        return _extract_codex_block(proc.stdout)

    def _call_google(self, system, user, fmt, temperature, max_tokens) -> str:
        api_key = _first_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Need GOOGLE_API_KEY or GEMINI_API_KEY in env")
        base_url = _provider_base_url("gemini")
        model = self.model
        if base_url:
            return self._call_google_genai(
                system=system,
                user=user,
                image_paths=(),
                fmt=fmt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

        import google.generativeai as genai  # lazy

        genai.configure(api_key=api_key)
        gen_cfg = {
            "temperature": temperature,
            "response_mime_type": "application/json" if fmt == "json" else "text/plain",
        }
        if max_tokens and max_tokens > 0:
            gen_cfg["max_output_tokens"] = max_tokens
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config=genai.types.GenerationConfig(**gen_cfg),
        )
        resp = model.generate_content(user)
        return resp.text or ""

    def _call_google_with_images(
        self,
        system,
        user,
        image_paths,
        fmt,
        temperature,
        max_tokens,
    ) -> str:
        api_key = _first_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Need GOOGLE_API_KEY or GEMINI_API_KEY in env")
        base_url = _provider_base_url("gemini")
        model = self.model
        if not base_url:
            raise RuntimeError(
                "google/<model> image input requires google-genai endpoint; "
                "set GOOGLE_BASE_URL/GEMINI_BASE_URL"
            )
        return self._call_google_genai(
            system=system,
            user=user,
            image_paths=image_paths,
            fmt=fmt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    def _call_google_genai(
        self,
        *,
        system: str,
        user: str,
        image_paths: Sequence[Path],
        fmt: ResponseFormat,
        temperature: float,
        max_tokens: int | None,
        api_key: str,
        base_url: str,
        model: str,
    ) -> str:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google/<model> with a Gemini SDK endpoint requires "
                "`pip install google-genai`"
            ) from e

        if fmt == "json":
            system, user = _ensure_json_instruction(system, user)
        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta", "base_url": base_url},
        )
        contents: list = [f"{system}\n\n{user}" if system else user]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            contents.append(types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=mime,
            ))
        config: dict = {"temperature": temperature}
        if fmt == "json":
            config["response_mime_type"] = "application/json"
        if max_tokens and max_tokens > 0:
            config["max_output_tokens"] = max_tokens
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except TypeError:
            resp = client.models.generate_content(model=model, contents=contents)
        return getattr(resp, "text", "") or ""


# ---------------------------------------------------------------------------
# Multi-provider chained client


class ChainedLLMClient:
    """Wraps a list of providers. On exception, falls through to the
    next entry. Per-provider retries can be enabled with ``max_retries``,
    but default to 0 so timeout fallback is fast."""

    def __init__(
        self,
        *,
        clients: tuple[LLMClient, ...],
        max_retries: int = 0,
        backoff_base: float = 2.0,
    ) -> None:
        self.clients = clients
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @property
    def primary(self) -> LLMClient:
        return self.clients[0]

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        raw_out_dir: Path | None = None,
    ) -> str:
        kw = dict(system=system, user=user, response_format=response_format,
                  temperature=temperature, max_tokens=max_tokens)
        last_err: Exception | None = None
        for idx, c in enumerate(self.clients):
            self._honor_cooldown(c.slug)
            for attempt in range(self.max_retries + 1):
                log.info(ui.task(
                    f"LLM {c.slug} attempt={attempt} format={response_format}"
                ))
                try:
                    out = c.complete(**kw)
                    if response_format == "json":
                        try:
                            _extract_json(out)
                        except Exception as e:  # noqa: BLE001
                            self._dump(
                                raw_out_dir, c.slug, attempt, ok=False,
                                body=(
                                    f"invalid JSON response: {e!r}\n\n"
                                    f"RAW RESPONSE:\n{out[:200_000]}"
                                ),
                            )
                            raise RuntimeError(
                                f"{c.slug} returned invalid JSON: "
                                f"{_text_preview(out)}"
                            ) from e
                    self._dump(raw_out_dir, c.slug, attempt, ok=True, body=out)
                    if idx > 0:
                        log.info(ui.ok(f"fallback succeeded with {c.slug}"))
                    return out
                except Exception as e:  # noqa: BLE001
                    self._dump(raw_out_dir, c.slug, attempt, ok=False, body=repr(e))
                    log.info(ui.warn(f"LLM {c.slug} failed: {e!s}"))
                    last_err = e
                    transient = _is_transient(e)
                    if transient:
                        self._note_cooldown(c.slug, e)
                    will_retry = transient and attempt < self.max_retries
                    if not will_retry:
                        break
                    sleep = self.backoff_base ** attempt
                    time.sleep(sleep)
            # outer loop: advance to next provider on any failure
        raise last_err or RuntimeError("no providers in chain produced a result")

    def complete_json(self, *, system: str, user: str, **kw) -> dict:
        text = self.complete(system=system, user=user, response_format="json", **kw)
        return _extract_json(text)

    def complete_with_images(
        self,
        *,
        system: str,
        user: str,
        image_paths: Sequence[Path],
        response_format: ResponseFormat = "text",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        raw_out_dir: Path | None = None,
    ) -> str:
        kw = dict(system=system, user=user, image_paths=image_paths,
                  response_format=response_format, temperature=temperature,
                  max_tokens=max_tokens)
        last_err: Exception | None = None
        for idx, c in enumerate(self.clients):
            self._honor_cooldown(c.slug)
            for attempt in range(self.max_retries + 1):
                log.info(ui.task(
                    f"LLM {c.slug} attempt={attempt} "
                    f"format={response_format} images={len(image_paths)}"
                ))
                try:
                    out = c.complete_with_images(**kw)
                    self._dump(raw_out_dir, c.slug, attempt, ok=True, body=out)
                    if idx > 0:
                        log.info(ui.ok(f"fallback succeeded with {c.slug}"))
                    return out
                except Exception as e:  # noqa: BLE001
                    self._dump(raw_out_dir, c.slug, attempt, ok=False, body=repr(e))
                    log.info(ui.warn(f"LLM {c.slug} failed: {e!s}"))
                    last_err = e
                    transient = _is_transient(e)
                    if transient:
                        self._note_cooldown(c.slug, e)
                    will_retry = transient and attempt < self.max_retries
                    if not will_retry:
                        break
                    sleep = self.backoff_base ** attempt
                    time.sleep(sleep)
        raise last_err or RuntimeError("no providers in chain produced a result")

    @staticmethod
    def _honor_cooldown(slug: str) -> None:
        """Sleep until the provider's cooldown window expires. Always blocks
        when one is active, regardless of whether a fallback exists."""
        wait = _cooldown_seconds_left(slug)
        if wait <= 0:
            return
        log.info(ui.detail("LLM cooldown", f"{slug}: sleeping {wait:.0f}s"))
        time.sleep(wait)

    @staticmethod
    def _note_cooldown(slug: str, err: BaseException) -> None:
        cd_sec, reason = _record_transient_error(slug, err)
        if cd_sec > 0:
            log.info(ui.warn(
                f"LLM cooldown armed for {slug}: {cd_sec:.0f}s ({reason})"
            ))

    @staticmethod
    def _dump(out_dir: Path | None, slug: str, attempt: int, *,
              ok: bool, body: str) -> None:
        if not out_dir:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "ok" if ok else "err"
        path = out_dir / f"raw_{_slug_safe(slug)}_attempt{attempt:02d}_{tag}.txt"
        path.write_text(body[:200_000])  # cap so a runaway model doesn't blow up the dir


# ---------------------------------------------------------------------------
# helpers


def _extract_codex_block(stdout: str) -> str:
    """Pull the model's reply out of `codex exec` mixed output.

    Layout (observed on codex-cli 0.118):
        OpenAI Codex v0.118.0 ...
        --------
        workdir: ...
        ...
        --------
        user
        <echoed prompt>
        codex
        <model reply, possibly multi-line>
        tokens used
        <n>
    """
    lines = stdout.splitlines()
    try:
        start = max(i for i, ln in enumerate(lines) if ln.strip() == "codex")
    except ValueError:
        return stdout.strip()
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("tokens used"):
            end = i
            break
    return "\n".join(lines[start + 1:end]).strip()


def _extract_json(text: str) -> dict:
    """Tolerate fenced ```json blocks and stray prose around a JSON object."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise
