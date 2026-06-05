"""Tests for tolerant code-fence stripping."""

from __future__ import annotations

from cg_tutor.agents.base import strip_code_fence


def test_strip_complete_python_fence():
    text = "```python\nprint('ok')\n```"
    assert strip_code_fence(text) == "print('ok')"


def test_strip_unclosed_python_fence():
    text = "```python\nprint('still ok')\n"
    assert strip_code_fence(text) == "print('still ok')"


def test_strip_fence_with_extra_metadata():
    text = "```python copy\nprint('ok')\n```"
    assert strip_code_fence(text) == "print('ok')"
