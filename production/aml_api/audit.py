"""Opt-in per-request provider audit for offline pipeline artifacts (never HTTP output)."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

JSON_AUDIT: ContextVar[list | None] = ContextVar("json_audit", default=None)


@contextmanager
def capture_provider_calls(records):
    token = JSON_AUDIT.set(records)
    try:
        yield
    finally:
        JSON_AUDIT.reset(token)
