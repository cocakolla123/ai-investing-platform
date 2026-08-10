"""Shared HTTP plumbing for network adapters. Design: doc 01 §6, doc 02 §1.

Every adapter GET goes through :func:`get_json`:

- token-bucket pacing *before* the request (client-side rate limiting),
- tenacity retry: exactly 1 retry with jitter on transient failures
  (transport errors, 5xx), then a typed exception,
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
) -> Any:
    """GET ``url`` and decode JSON with pacing, one jittered retry, typed errors."""

    @retry(
        retry=retry_if_exception_type(_Transient),
        stop=stop_after_attempt(2),  # 1 retry (doc 01 §6)
        wait=wait_random(0.2, 1.0),  # jitter
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
