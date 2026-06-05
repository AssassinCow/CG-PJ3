"""Small terminal formatting helpers for pipeline progress logs."""

from __future__ import annotations


RULE = "=" * 72


def rule() -> str:
    return RULE


def banner(title: str) -> str:
    return f"\n{RULE}\n{title}\n{RULE}"


def kv(label: str, value: object) -> str:
    return f"  {label:<16} {value}"


def step(index: int, total: int, title: str, status: str | None = None) -> str:
    suffix = f"  [{status}]" if status else ""
    return f"\n{RULE}\n[{index}/{total}] {title}{suffix}"


def iter_header(iteration: int, max_iteration: int) -> str:
    return f"\n{RULE}\niter{iteration:02d} / iter{max_iteration:02d}"


def task(title: str) -> str:
    return f"  > {title}"


def detail(label: str, value: object) -> str:
    return f"    - {label:<18} {value}"


def ok(message: str) -> str:
    return f"    OK  {message}"


def warn(message: str) -> str:
    return f"    WARN {message}"


def fail(message: str) -> str:
    return f"    FAIL {message}"


def done(message: str) -> str:
    return f"\n{RULE}\nDONE {message}"
