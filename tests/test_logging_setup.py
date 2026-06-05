"""Smoke tests for the central logging setup."""

from __future__ import annotations

import io
import logging

import pytest

from cg_tutor import _logging as logsetup


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Pull the module out of the configured state between tests so each
    one exercises a fresh ``configure_logging`` call."""
    logsetup._configured = False
    root = logging.getLogger("cg_tutor")
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    logsetup._configured = False
    for h in list(root.handlers):
        root.removeHandler(h)


def test_get_logger_returns_cg_tutor_namespaced_logger():
    log = logsetup.get_logger("foo")
    assert log.name == "cg_tutor.foo"


def test_get_logger_passes_already_namespaced_name_through():
    log = logsetup.get_logger("cg_tutor.bar")
    assert log.name == "cg_tutor.bar"


def test_configure_logging_installs_handler_with_short_name_prefix():
    stream = io.StringIO()
    logsetup.configure_logging("INFO", stream=stream)
    log = logsetup.get_logger("pipeline")
    log.info("hello")
    # The handler is on the root cg_tutor logger; propagate the message.
    assert "[pipeline] hello" in stream.getvalue()


def test_configure_logging_is_idempotent_no_duplicate_handlers():
    stream = io.StringIO()
    logsetup.configure_logging("INFO", stream=stream)
    logsetup.configure_logging("DEBUG", stream=stream)
    root = logging.getLogger("cg_tutor")
    # Only one handler regardless of how many times we call.
    assert len(root.handlers) == 1


def test_configure_logging_respects_explicit_level():
    stream = io.StringIO()
    logsetup.configure_logging("WARNING", stream=stream)
    log = logsetup.get_logger("pipeline")
    log.info("info-level-should-not-appear")
    log.warning("warning-level-should-appear")
    captured = stream.getvalue()
    assert "info-level-should-not-appear" not in captured
    assert "warning-level-should-appear" in captured
