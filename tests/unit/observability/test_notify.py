"""Something can page the operator (doc 12 §7).

No test here touches the network: ``TelegramNotifier.post`` is the seam.
"""

from __future__ import annotations

from typing import Any

import pytest
from structlog.testing import capture_logs

from dynamo.config.settings import Settings
from dynamo.observability.notify import (
    AlertDispatcher,
    DisabledNotifier,
    RecordingNotifier,
    TelegramNotifier,
    build_notifier,
    severity_rank,
)


class FakePost:
    """Records calls; configurable status / exception."""

    def __init__(self, status: int = 200, raises: Exception | None = None) -> None:
        self.status = status
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> tuple[int, str]:
        self.calls.append((url, payload, timeout))
        if self.raises is not None:
            raise self.raises
        return self.status, "{}"


def _telegram(post: FakePost, **kwargs: Any) -> TelegramNotifier:
    return TelegramNotifier(token="tok", chat_id="42", post=post, **kwargs)


# --- severity ------------------------------------------------------------


def test_severity_ladder_orders_page_highest() -> None:
    assert severity_rank("debug") < severity_rank("info") < severity_rank("warn")
    assert severity_rank("warn") < severity_rank("page")


def test_unknown_severity_sorts_as_warn_not_debug() -> None:
    """An unrecognized severity must never be silently demoted to noise."""
    assert severity_rank("catastrophe") == severity_rank("warn")


# --- the page path -------------------------------------------------------


def test_page_severity_triggers_a_send() -> None:
    post = FakePost()
    assert _telegram(post).send(severity="page", name="dd20", message="full stop") is True
    assert len(post.calls) == 1
    url, payload, _ = post.calls[0]
    assert url == "https://api.telegram.org/bottok/sendMessage"
    assert payload["chat_id"] == "42"
    assert "dd20" in payload["text"] and "full stop" in payload["text"]
    assert payload["text"].startswith("[PAGE]")


def test_below_the_floor_is_not_transmitted() -> None:
    post = FakePost()
    notifier = _telegram(post)  # min_severity defaults to "page"
    assert notifier.send(severity="info", name="x", message="y") is False
    assert notifier.send(severity="warn", name="x", message="y") is False
    assert post.calls == []


def test_lowering_the_floor_transmits_warnings() -> None:
    post = FakePost()
    assert _telegram(post, min_severity="warn").send(severity="warn", name="x", message="y") is True
    assert len(post.calls) == 1


def test_long_messages_are_truncated_not_rejected() -> None:
    post = FakePost()
    _telegram(post).send(severity="page", name="n", message="x" * 10_000)
    assert len(post.calls[0][1]["text"]) <= 3500


# --- never crash the caller ---------------------------------------------


def test_transport_exception_never_propagates() -> None:
    post = FakePost(raises=OSError("connection reset"))
    assert _telegram(post).send(severity="page", name="n", message="m") is False


def test_non_2xx_is_a_logged_failure_not_a_raise() -> None:
    assert _telegram(FakePost(status=429)).send(severity="page", name="n", message="m") is False


def test_rejection_log_does_not_leak_the_bot_token() -> None:
    with capture_logs() as entries:
        _telegram(FakePost(status=401)).send(severity="page", name="n", message="m")
    assert entries and all("bottok" not in repr(e) for e in entries)


# --- unconfigured degrades once, loudly ---------------------------------


def test_unconfigured_settings_build_a_disabled_notifier() -> None:
    notifier = build_notifier(Settings(TELEGRAM_BOT_TOKEN="", TELEGRAM_CHAT_ID=""))
    assert isinstance(notifier, DisabledNotifier)
    assert "TELEGRAM_BOT_TOKEN" in notifier.reason


def test_half_configured_is_still_disabled() -> None:
    notifier = build_notifier(Settings(TELEGRAM_BOT_TOKEN="tok", TELEGRAM_CHAT_ID=""))
    assert isinstance(notifier, DisabledNotifier)
    assert "TELEGRAM_CHAT_ID" in notifier.reason


def test_configured_settings_build_a_telegram_notifier() -> None:
    notifier = build_notifier(
        Settings(TELEGRAM_BOT_TOKEN="tok", TELEGRAM_CHAT_ID="42", DYNAMO_NOTIFY_MIN_SEVERITY="warn")
    )
    assert isinstance(notifier, TelegramNotifier)
    assert notifier.min_severity == "warn"


def test_disabled_notifier_warns_exactly_once_and_never_raises() -> None:
    notifier = DisabledNotifier(reason="unset: TELEGRAM_BOT_TOKEN")
    with capture_logs() as entries:
        for i in range(50):
            assert notifier.send(severity="page", name=f"alert-{i}", message="m") is False
    warnings = [e for e in entries if e["event"] == "notifier_disabled"]
    assert len(warnings) == 1
    assert "TELEGRAM_BOT_TOKEN" in warnings[0]["hint"]


# --- the dispatcher: row first, page second -----------------------------


class FakeRepo:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add_alert(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def test_dispatcher_writes_the_row_and_pages() -> None:
    repo, notifier = FakeRepo(), RecordingNotifier()
    assert AlertDispatcher(repo.add_alert, notifier)("page", "dd20", "stop") is True
    assert repo.rows == [{"severity": "page", "name": "dd20", "message": "stop", "ackable": True}]
    assert notifier.sent == [("page", "dd20", "stop")]


def test_dispatcher_still_writes_the_row_when_the_channel_is_dead() -> None:
    """The audit row is the record of what happened; paging is best-effort."""

    class Exploding:
        def send(self, **_: Any) -> bool:
            raise RuntimeError("channel down")

    repo = FakeRepo()
    assert AlertDispatcher(repo.add_alert, Exploding())("page", "n", "m") is False
    assert len(repo.rows) == 1


def test_dispatcher_propagates_a_failed_row_write() -> None:
    """Losing the audit row IS a real error and must not be swallowed."""

    def boom(**_: Any) -> None:
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError, match="db down"):
        AlertDispatcher(boom, RecordingNotifier())("page", "n", "m")
