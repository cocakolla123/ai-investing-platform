"""The DD20 rung really writes the kill-switch file (doc 06 §3).

``_apply_ladder_transition`` is the composition-root half of the mechanism:
the pure decision lives in ``risk/ladder_response.py`` and is tested there.
This exercises the side effects — the switch file, the alert, the event row —
against fakes, with no database and no broker.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import orchestration.assets.portfolio_exec as pexec
import pytest
from dagster import build_asset_context

from dynamo.config.settings import Settings
from dynamo.domain.models.risk import LadderRung
from dynamo.execution.gate import kill_switch_engaged

AS_OF = date(2026, 8, 20)


class MemoryLadderEvents:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> int:
        self.rows.append(kwargs)
        return len(self.rows)


class FakeDynamo:
    """Only what ``_apply_ladder_transition`` touches."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.alerts: list[tuple[str, str, str, bool]] = []

    def settings(self) -> Settings:
        return self._settings

    def engine(self) -> Any:
        return "engine-sentinel"  # every repository is faked

    def alert(self, severity: str, name: str, message: str, *, ackable: bool = True) -> None:
        self.alerts.append((severity, name, message, ackable))


class MockBroker:
    def __init__(self, open_ids: list[str] | None = None) -> None:
        self.open_ids = open_ids or []
        self.cancelled: list[str] = []

    def open_orders(self) -> list[dict[str, object]]:
        return [{"client_order_id": c} for c in self.open_ids]

    def cancel(self, client_order_id: str) -> None:
        self.cancelled.append(client_order_id)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import dynamo.persistence.repositories.risk as risk_repos

    settings = Settings(
        DYNAMO_ENV="PAPER",
        DYNAMO_KILL_SWITCH_PATH=str(tmp_path / "KILL_SWITCH"),
        _env_file=None,  # type: ignore[call-arg]
    )
    events = MemoryLadderEvents()
    dynamo = FakeDynamo(settings)
    broker = MockBroker(["coid-1", "coid-2"])
    monkeypatch.setattr(risk_repos, "LadderEventRepository", lambda _e: events)
    monkeypatch.setattr(pexec, "_broker", lambda _d: broker)

    class Rig:
        pass

    rig = Rig()
    rig.settings, rig.events, rig.dynamo, rig.broker = settings, events, dynamo, broker  # type: ignore[attr-defined]
    rig.switch = tmp_path / "KILL_SWITCH"  # type: ignore[attr-defined]
    return rig


def _apply(rig: Any, frm: LadderRung, to: LadderRung, dd: float) -> int | None:
    return pexec._apply_ladder_transition(
        build_asset_context(), rig.dynamo, frm, to, dd, AS_OF
    )


def test_dd20_writes_the_kill_switch_file_and_cancels_open_orders(rig: Any) -> None:
    event_id = _apply(rig, LadderRung.DD15, LadderRung.DD20, -0.21)

    assert event_id == 1
    assert kill_switch_engaged(rig.settings.kill_switch_file) is True
    assert "drawdown ladder dd20" in rig.switch.read_text()
    assert rig.broker.cancelled == ["coid-1", "coid-2"]


def test_dd20_pages_the_operator(rig: Any) -> None:
    _apply(rig, LadderRung.NORMAL, LadderRung.DD20, -0.25)
    severities = {a[0] for a in rig.dynamo.alerts}
    names = {a[1] for a in rig.dynamo.alerts}
    assert severities == {"page"}
    assert "kill_switch_engaged" in names and "ladder_transition" in names


def test_dd20_records_the_event_as_ack_requiring(rig: Any) -> None:
    _apply(rig, LadderRung.NORMAL, LadderRung.DD20, -0.25)
    row = rig.events.rows[0]
    assert row["to_rung"] == "dd20"
    assert row["requires_ack"] is True
    assert row["kill_switch_engaged"] is True


def test_dd8_records_an_ack_requiring_event_without_freezing(rig: Any) -> None:
    _apply(rig, LadderRung.DD5, LadderRung.DD8, -0.09)
    row = rig.events.rows[0]
    assert (row["requires_ack"], row["kill_switch_engaged"]) == (True, False)
    assert kill_switch_engaged(rig.settings.kill_switch_file) is False
    # The alert must not be alert-ackable: the LADDER ack is what clears it.
    transition = next(a for a in rig.dynamo.alerts if a[1] == "ladder_transition")
    assert transition[0] == "warn" and transition[3] is False


def test_dd5_warns_and_leaves_no_blocking_row(rig: Any) -> None:
    _apply(rig, LadderRung.NORMAL, LadderRung.DD5, -0.06)
    assert rig.events.rows[0]["requires_ack"] is False
    assert kill_switch_engaged(rig.settings.kill_switch_file) is False


def test_an_unchanged_rung_writes_nothing_at_all(rig: Any) -> None:
    assert _apply(rig, LadderRung.DD5, LadderRung.DD5, -0.06) is None
    assert rig.events.rows == []
    assert rig.dynamo.alerts == []


def test_recovery_alerts_but_never_writes_the_switch_file(rig: Any) -> None:
    _apply(rig, LadderRung.DD20, LadderRung.DD5, -0.06)
    assert kill_switch_engaged(rig.settings.kill_switch_file) is False
    assert rig.broker.cancelled == []
    assert [a[0] for a in rig.dynamo.alerts] == ["info"]


def test_a_failed_kill_switch_write_still_pages(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the freeze cannot be performed, the operator must be told to do it."""
    import dynamo.execution.gate as gate

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(gate, "engage_kill_switch", boom)
    _apply(rig, LadderRung.NORMAL, LadderRung.DD20, -0.25)
    names = {a[1] for a in rig.dynamo.alerts}
    assert "kill_switch_engage_failed" in names
    assert rig.events.rows  # the event is recorded even when the freeze fails
