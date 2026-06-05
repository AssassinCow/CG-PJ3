"""Project-wide logging setup.

Existing code prints lines like ``"[pipeline] iter00 begin"`` directly
to stdout. This module routes those messages through ``logging`` so the
CLI can filter by level (``--log-level DEBUG`` for diagnostics, ``WARNING``
for CI), tests can capture output via ``caplog``, and the bracketed
prefix is supplied automatically from the module name rather than
hard-coded into every f-string.

To avoid disturbing scripts that don't call ``configure_logging``, the
loggers default to a NullHandler. The first ``configure_logging()`` call
installs a stderr StreamHandler with the legacy ``[<short>] <message>``
format. Subsequent calls are idempotent — they update the level without
duplicating handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal


_BRACKET_FORMAT = "[%(short_name)s] %(message)s"
_ROOT_NAME = "cg_tutor"
_configured = False


class _ShortNameFilter(logging.Filter):
    """Inject ``short_name`` = trailing segment of logger name into records.

    ``cg_tutor.pipeline`` becomes ``pipeline``, ``cg_tutor.agents.storyboard``
    becomes ``storyboard``. Keeps the legacy ``[pipeline]`` prefix style.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.short_name = record.name.rsplit(".", 1)[-1]
        return True


def get_logger(module_name: str) -> logging.Logger:
    """Return a logger keyed by module name. Falls back to ``cg_tutor.<name>``
    when callers pass a bare suffix."""
    if not module_name.startswith(_ROOT_NAME):
        full = f"{_ROOT_NAME}.{module_name}"
    else:
        full = module_name
    return logging.getLogger(full)


_LevelLike = int | str | Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def configure_logging(
    level: _LevelLike | None = None,
    *,
    stream=None,
) -> None:
    """Install a single stderr handler at the requested level.

    The level resolution order is: explicit ``level`` argument →
    ``CG_TUTOR_LOG_LEVEL`` env var → ``INFO``. Idempotent: the second
    call updates the level but never installs a duplicate handler.
    """
    global _configured
    root = logging.getLogger(_ROOT_NAME)
    if level is None:
        level = os.environ.get("CG_TUTOR_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()
    root.setLevel(level)
    if _configured:
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_BRACKET_FORMAT))
    handler.addFilter(_ShortNameFilter())
    root.addHandler(handler)
    # Prevent the root logger above us from also emitting these (we have
    # our own handler; double-printing would clutter the terminal).
    root.propagate = False
    _configured = True
