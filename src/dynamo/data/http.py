"""Shared HTTP plumbing for network adapters. Design: doc 01 §6, doc 02 §1.

Every adapter GET goes through :func:`get_json`:

- token-bucket pacing *before* the request (client-side rate limiting),
- tenacity retry: by default exactly 1 retry with jitter on transient
  failures (transport errors, 5xx), then a typed exception. ``attempts``
  raises that for BACKFILLS, which have the opposite requirement from the
  nightly path: a scheduled job must fail fast and page, whereas a
  multi-hour sweep should out-wait a flaky endpoint. EDGAR's full-text
  search 500s under sustained load -- measured 17 of 55 day-sweeps lost
  to it at the default policy, because two attempts a fraction of a
  second apart both land inside the same throttle window,
- 429 → :class:`RateLimited` (logged incident, no retry storm),
- structlog logging with provider/url context.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
    wait_random_exponential,
)

from dynamo.data.errors import ProviderDown, RateLimited
from dynamo.data.ratelimit import TokenBucket

log = structlog.get_logger(__name__)


class _Transient(Exception):
    """Internal marker for retryable failures (never escapes this module)."""


def get_json(
    client: httpx.Client,
    url: str,
    *,
    provider: str,
    bucket: TokenBucket | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 2,
    max_backoff_s: float = 30.0,
) -> Any:
    """GET ``url`` and decode JSON with pacing, jittered retry, typed errors.

    ``attempts`` defaults to 2 (one retry, doc 01 §6) with sub-second jitter —
    the nightly contract. Above 2 it switches to exponential backoff capped at
    ``max_backoff_s``, because more attempts are only useful if they are
    spread across the throttle window rather than crowded into it.
    """
    wait = (
        wait_random(0.2, 1.0)
        if attempts <= 2
        else wait_random_exponential(multiplier=1.0, max=max_backoff_s)
    )

    @retry(
        retry=retry_if_exception_type(_Transient),
        stop=stop_after_attempt(max(1, attempts)),
        wait=wait,
        reraise=True,
    )
    def _do() -> httpx.Response:
        if bucket is not None:
            bucket.acquire()
        try:
            resp = client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:  # transport-level: retryable once
            log.warning("provider_transport_error", provider=provider, url=url, error=str(exc))
            raise _Transient(str(exc)) from exc
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            log.warning("provider_rate_limited", provider=provider, url=url)
            raise RateLimited(
                provider,
                f"429 from {url}",
                retry_after_s=float(retry_after) if retry_after else None,
            )
        if resp.status_code >= 500:
            log.warning("provider_5xx", provider=provider, url=url, status=resp.status_code)
            raise _Transient(f"{resp.status_code} from {url}")
        if resp.status_code >= 400:
            raise ProviderDown(provider, f"{resp.status_code} from {url}")
        return resp

    try:
        response = _do()
    except _Transient as exc:
        raise ProviderDown(provider, str(exc)) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderDown(provider, f"invalid JSON from {url}: {exc}") from exc
