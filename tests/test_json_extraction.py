"""Tests for tolerant JSON extraction in llm_client._extract_json."""

from __future__ import annotations

import pytest

from cg_tutor.llm_client import _extract_codex_block, _extract_json


def test_plain_json_object():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    txt = "```json\n{\"a\": 1, \"b\": [1, 2]}\n```"
    assert _extract_json(txt) == {"a": 1, "b": [1, 2]}


def test_fenced_without_language_tag():
    txt = "```\n{\"x\": 1}\n```"
    assert _extract_json(txt) == {"x": 1}


def test_prose_around_json():
    txt = "Here is the result:\n\n{\"answer\": 42}\n\nThat's all."
    assert _extract_json(txt) == {"answer": 42}


def test_extracts_first_complete_object_when_multiple_objects_present():
    txt = "note {\"answer\": 42}\n{\"debug\": true}"
    assert _extract_json(txt) == {"answer": 42}


def test_invalid_json_raises():
    with pytest.raises(Exception):
        _extract_json("not json at all")


def test_extract_codex_block_strips_banner():
    stdout = """OpenAI Codex v0.118.0
--------
workdir: /x
--------
user
hi
codex
the model output
maybe multiple lines
tokens used
9999
"""
    assert _extract_codex_block(stdout) == "the model output\nmaybe multiple lines"


def test_extract_codex_block_no_banner_returns_trimmed():
    assert _extract_codex_block("just text\n") == "just text"


def test_extract_codex_block_no_tokens_used_line():
    """If tokens-used footer is missing, take everything after `codex`."""
    stdout = "codex\nfinal answer\nmore text\n"
    assert _extract_codex_block(stdout) == "final answer\nmore text"
