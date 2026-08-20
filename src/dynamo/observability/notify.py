"""Operator paging (doc 12 §7): the channel that actually reaches a human.

Before this module the entire alerting system was ``add_alert()`` inserting a
row into the ``alerts`` table. That is durable, auditable and completely
silent: nothing on the box could wake anybody up.

Design rules, in priority order:

1. **A notifier never crashes its caller.** Every transport error, timeout and
   bad status is swallowed and logged. A pipeline that is reporting a failure
   must not fail *because* it reported.
2. **Unconfigured degrades once, loudly.** No token/chat id is the normal
   development state; it logs one WARNING per process (``DisabledNotifier``)
   and is silent afterwards. It is never an exception and never a per-alert
   log flood.
3. **The transport is a seam.** ``TelegramNotifier`` takes a ``post``
   callable; tests inject a fake and no test ever touches the network.
   Swapping Telegram for e-mail/ntfy/Pushover means writing another class
   with a ``send`` method — the :class:`dynamo.domain.ports.Notifier` port.
4. **Severity floor, not severity filter.** Everything still goes to the
   ``alerts`` table. ``min_severity`` (default ``page``) only decides what is
   worth an out-of-band interruption.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

#: Ordered severity ladder. Anything unrecognized sorts as ``warn`` — an
#: unknown severity is never silently treated as debug noise.
SEVERITY_ORDER: dict[str, int] = {"debug": 0, "info": 1, "warn": 2, "page": 3}
_UNKNOWN_SEVERITY_RANK = SEVERITY_ORDER["warn"]

TELEGRAM_API = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 3500  # Telegram's hard limit is 4096; leave room for the header

#: ``(url, payload, timeout) -> (status_code, body)``. The one seam tests replace.
PostFn = Callable[[str, dict[str, Any], float], tuple[int, str]]


def severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(severity.strip().lower(), _UNKNOWN_SEVERITY_RANK)


def _urllib_post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
    """Default transport. stdlib only — no new dependency for one POST."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # url is a fixed https literal + token
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


@dataclass
class DisabledNotifier:
    """No channel configured. Logs the reason ONCE, then stays quiet.

    Returning False (rather than raising) is deliberate: an unconfigured
    platform must behave exactly like a configured one whose network is down.
    """

    reason: str = "no notification channel configured"
    _warned: bool = field(default=False, init=False, repr=False)

    def send(self, *, severity: str, name: str, message: str) -> bool:
        if not self._warned:
            self._warned = True
            log.warning(
                "notifier_disabled",
                reason=self.reason,
                hint="set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see .env.example)",
                first_suppressed_alert=name,
            )
        log.debug("notifier_suppressed", severity=severity, name=name, message=message)
        return False


@dataclass
class RecordingNotifier:
    """In-memory notifier for tests and dry runs; records instead of sending."""

    sent: list[tuple[str, str, str]] = field(default_factory=list)

    def send(self, *, severity: str, name: str, message: str) -> bool:
        self.sent.append((severity, name, message))
        return True


@dataclass
class TelegramNotifier:
    """Telegram Bot API ``sendMessage``.

    Chosen because it needs no inbound port, no SMTP relay and no paid tier,
    and the credentials are already in ``Settings``. It is an *adapter*: the
    platform depends on the ``Notifier`` port, not on Telegram.
    """

    token: str
    chat_id: str
    min_severity: str = "page"
    timeout: float = 10.0
    post: PostFn = _urllib_post
    api_base: str = TELEGRAM_API

    def _url(self) -> str:
        return f"{self.api_base}/bot{self.token}/sendMessage"

    def send(self, *, severity: str, name: str, message: str) -> bool:
        if severity_rank(severity) < severity_rank(self.min_severity):
            return False
        text = f"[{severity.upper()}] {name}\n{message}"[:MAX_MESSAGE_CHARS]
        payload = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        try:
            status, body = self.post(self._url(), payload, self.timeout)
        except Exception as exc:  # a notifier NEVER propagates
            log.warning("notify_failed", name=name, severity=severity, error=repr(exc))
            return False
        if 200 <= status < 300:
            log.info("notify_sent", name=name, severity=severity)
            return True
        # Never log the token: _url() contains it, body may echo the request.
        log.warning("notify_rejected", name=name, severity=severity, status=status, body=body[:200])
        return False


@dataclass
class AlertDispatcher:
    """Row first, page second — the DB write is the record of what happened.

    ``record`` is the existing ``MarketDataRepository.add_alert``-shaped
    callable. It is called first and its failure propagates (losing the audit
    row IS a real error); the notification that follows never does.
    """

    record: Callable[..., None]
    notifier: Any  # dynamo.domain.ports.Notifier

    def __call__(self, severity: str, name: str, message: str, *, ackable: bool = True) -> bool:
        self.record(severity=severity, name=name, message=message, ackable=ackable)
        try:
            return bool(self.notifier.send(severity=severity, name=name, message=message))
        except Exception as exc:  # belt and braces around third-party notifiers
            log.warning("notify_dispatch_failed", name=name, error=repr(exc))
            return False


def build_notifier(settings: Any) -> Any:
    """Notifier for ``settings``; :class:`DisabledNotifier` when unconfigured.

    Typed loosely on purpose: the only thing this module needs from
    ``Settings`` is four attributes, and importing it would drag config into
    every consumer of the port.
    """
    token = str(getattr(settings, "telegram_bot_token", "") or "")
    chat_id = str(getattr(settings, "telegram_chat_id", "") or "")
    if not token or not chat_id:
        missing = [
            n for n, v in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)) if not v
        ]
        return DisabledNotifier(reason=f"unset: {', '.join(missing)}")
    return TelegramNotifier(
        token=token,
        chat_id=chat_id,
        min_severity=str(getattr(settings, "notify_min_severity", "page") or "page"),
        timeout=float(getattr(settings, "notify_timeout_seconds", 10.0)),
    )
