"""Three-tier universe manager tests (doc 02 §2) with an in-memory store."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from dynamo.data.universe import UniverseManager, default_sector_etf_map
from dynamo.domain.models.market import Tier, UniverseMembership

AS_OF = date(2026, 8, 3)
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


class FakeStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, date], UniverseMembership] = {}
        self.alerts: list[dict[str, str]] = []

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        return [
            m
            for m in self.rows.values()
            if m.valid_from <= as_of
            and (m.valid_to is None or m.valid_to >= as_of)
            and (tier is None or m.tier == tier)
        ]

    def upsert_universe_membership(self, memberships: list[UniverseMembership]) -> int:
        for m in memberships:
            self.rows[(m.symbol, m.valid_from)] = m
        return len(memberships)

    def close_membership(
        self, symbol: str, valid_from: date, valid_to: date, *, approved_by: str | None = None
    ) -> int:
        key = (symbol, valid_from)
        if key not in self.rows:
            return 0
        current = self.rows[key]
        self.rows[key] = current.model_copy(
            update={"valid_to": valid_to, "approved_by": approved_by or current.approved_by}
        )
        return 1

    def add_alert(
        self,
        *,
        severity: str,
        name: str,
        message: str,
        ackable: bool = True,
        created_at: datetime | None = None,
    ) -> None:
        self.alerts.append({"severity": severity, "name": name, "message": message})


def make_manager(store: FakeStore, seed_path: str = "/nonexistent/seed.txt") -> UniverseManager:
    return UniverseManager(store, seed_path=seed_path)


def test_seed_core_builtin_list_is_idempotent() -> None:
    store = FakeStore()
    manager = make_manager(store)
    count = manager.seed_core(AS_OF)
    assert count == 50  # ~40 stocks + ~10 ETFs
    assert "AAPL" in {m.symbol for m in store.universe_at(AS_OF, Tier.CORE)}
    assert manager.seed_core(AS_OF) == 0  # second run is a no-op


def test_seed_core_from_file(tmp_path: Path) -> None:
    seed = tmp_path / "universe_seed.txt"
    seed.write_text("# core\nAAPL\nmsft\n\nSPY\n")
    store = FakeStore()
    make_manager(store, seed_path=str(seed)).seed_core(AS_OF)
    assert {m.symbol for m in store.universe_at(AS_OF, Tier.CORE)} == {"AAPL", "MSFT", "SPY"}


def test_promote_writes_window_and_extends() -> None:
    store = FakeStore()
    manager = make_manager(store)
    m = manager.promote("NEWCO", as_of=AS_OF, reason="haiku triage: material 8-K")
    assert m.tier == Tier.PROMOTED
    assert m.valid_to == date(2026, 9, 2)  # default 30-day window
    assert m.approved_by is None  # temporary — no approval needed

    extended = manager.promote(
        "NEWCO", as_of=date(2026, 8, 10), reason="continued flags", window_days=30
    )
    assert extended.valid_to == date(2026, 9, 9)
    assert "extended" in extended.reason


def test_expire_promotions_lapse_quietly() -> None:
    store = FakeStore()
    manager = make_manager(store)
    manager.promote("NEWCO", as_of=AS_OF, reason="flag", window_days=5)
    # After the window the row no longer matches point-in-time queries.
    later = date(2026, 8, 20)
    assert manager.members_at(later, Tier.PROMOTED) == []
    assert manager.expire_promotions(later) == []  # nothing dangling


def test_recommendations_are_ackable_alert_rows_not_writes() -> None:
    store = FakeStore()
    manager = make_manager(store)
    manager.recommend_addition("NEWCO", "sustained committee conviction", now=NOW)
    manager.recommend_demotion("OLDCO", "60d below liquidity floor", now=NOW,
                               suggested_replacement="NEWCO")
    assert len(store.alerts) == 2
    assert all(a["name"] == "universe_change_recommendation" for a in store.alerts)
    assert all(a["severity"] == "info" for a in store.alerts)
    assert "ADD NEWCO" in store.alerts[0]["message"]
    assert "DEMOTE OLDCO" in store.alerts[1]["message"]
    assert store.rows == {}  # no membership writes without approval


def test_approve_core_addition_requires_approver() -> None:
    store = FakeStore()
    manager = make_manager(store)
    with pytest.raises(ValueError):
        manager.approve_core_addition("NEWCO", as_of=AS_OF, reason="r", approved_by="")
    m = manager.approve_core_addition("NEWCO", as_of=AS_OF, reason="queue ack", approved_by="me")
    assert m.tier == Tier.CORE
    assert m.approved_by == "me"


def test_approve_core_removal_closes_interval() -> None:
    store = FakeStore()
    manager = make_manager(store)
    manager.approve_core_addition("OLDCO", as_of=AS_OF, reason="seed", approved_by="me")
    later = date(2026, 9, 1)
    closed = manager.approve_core_removal("OLDCO", as_of=later, reason="demoted", approved_by="me")
    assert closed == 1
    assert manager.members_at(date(2026, 9, 2), Tier.CORE) == []
    assert manager.members_at(date(2026, 8, 15), Tier.CORE)  # point-in-time history kept


def test_default_sector_etf_map() -> None:
    mapping = default_sector_etf_map(
        {"AAPL": "Information Technology", "JPM": "Financials", "ODD": "Unknown Sector"}
    )
    assert mapping == {"AAPL": "XLK", "JPM": "XLF"}
