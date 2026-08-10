"""Typed provider exceptions. Design: doc 01 §6 (cross-cutting: errors).

Every network adapter normalizes transport/HTTP failures into these types so
pipelines can degrade per-asset instead of all-or-nothing. Hitting a provider
429 is a logged incident, not a retry storm (doc 02 §1).
"""

from __future__ import annotations


class DataProviderError(Exception):
    """Base class for all provider-adapter failures."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class RateLimited(DataProviderError):
    """Provider returned 429 or the client-side budget was exhausted."""

    def __init__(self, provider: str, message: str, retry_after_s: float | None = None) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(provider, message)


class ProviderDown(DataProviderError):
    """Provider unreachable or persistently failing (5xx, transport errors)."""


class DataGap(DataProviderError):
    """Provider responded but the payload is missing expected data."""
