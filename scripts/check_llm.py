"""Tiny LLM smoke test.

Usage examples:
    python scripts/check_llm.py
    python scripts/check_llm.py --model codex-cli/gpt-5.5
    python scripts/check_llm.py --model openai/gpt-5.5
    python scripts/check_llm.py --model anthropic/claude-sonnet-4.6
    python scripts/check_llm.py --model codex-cli/gpt-5.5 --model claude-cli/sonnet

The script sends one tiny JSON request and prints whether the configured
provider can return parseable JSON. It intentionally does not print secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cg_tutor.llm_client import LLMClient  # noqa: E402


DEFAULT_CHAIN = ("openai/gpt-5.5", "anthropic/claude-sonnet-4.6")


def _env_state() -> str:
    names = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_BASE_URL",
        "GEMINI_BASE_URL",
    )
    return "\n".join(f"  {name}: {'set' if os.environ.get(name) else 'unset'}" for name in names)


def _tool_state() -> str:
    names = ("claude", "codex", "blender", "ffmpeg")
    return "\n".join(
        f"  {name}: {shutil.which(name) or 'not found'}"
        for name in names
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Check whether an LLM provider can answer.")
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Provider/model slug. Repeat to test a fallback chain. "
            "Default: openai/gpt-5.5 then anthropic/claude-sonnet-4.6."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for CLI and API calls during this smoke test.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum output tokens for the tiny JSON response (default 256).",
    )
    ap.add_argument(
        "--show-env",
        action="store_true",
        help="Show relevant env var presence and tool paths before calling.",
    )
    args = ap.parse_args()

    chain = tuple(args.models or DEFAULT_CHAIN)
    os.environ["CG_TUTOR_LLM_TIMEOUT"] = str(args.timeout)
    os.environ["CG_TUTOR_API_TIMEOUT"] = str(args.timeout)

    print(f"[check] python: {sys.executable}")
    print(f"[check] chain: {', '.join(chain)}")
    print(f"[check] timeout: {args.timeout}s")
    if args.show_env:
        print("[check] env:")
        print(_env_state())
        print("[check] tools:")
        print(_tool_state())

    client = LLMClient.from_chain(chain)
    t0 = time.time()
    try:
        result = client.complete_json(
            system=(
                "You are a strict JSON API. Return only a JSON object. "
                "No markdown, no prose."
            ),
            user=(
                'Return exactly this JSON object with the same values: '
                '{"ok": true, "message": "pong"}'
            ),
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    except json.JSONDecodeError as e:
        raise SystemExit(
            "[check] provider returned text, but it was not parseable JSON. "
            f"{e}"
        ) from e
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"[check] FAILED: {type(e).__name__}: {e}") from e
    elapsed = time.time() - t0

    if result.get("ok") is True:
        print(f"[check] OK in {elapsed:.2f}s")
        print(f"[check] response: {result}")
        return

    raise SystemExit(f"[check] provider returned JSON, but not the expected shape: {result}")


if __name__ == "__main__":
    main()
