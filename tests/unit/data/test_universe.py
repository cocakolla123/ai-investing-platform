"""Three-tier universe manager tests (doc 02 §2) with an in-memory store."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from dynamo.data.universe import (
    UniverseManager,
    backdated_memberships,
    default_sector_etf_map,
    plan_listing_date_backdate,
)
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


def make_manager(
    store: FakeStore,
    seed_path: str = "/nonexistent/seed.txt",
    include_crypto: bool = True,
) -> UniverseManager:
    return UniverseManager(store, seed_path=seed_path, include_crypto=include_crypto)


def test_seed_core_builtin_list_is_idempotent() -> None:
    store = FakeStore()
    manager = make_manager(store)
    count = manager.seed_core(AS_OF)
    assert count == 55  # ~40 stocks + ~10 ETFs + 5 crypto pairs (ADR-0013)
    symbols = {m.symbol for m in store.universe_at(AS_OF, Tier.CORE)}
    assert "AAPL" in symbols
    assert "BTC/USD" in symbols
    assert manager.seed_core(AS_OF) == 0  # second run is a no-op


def test_seed_core_from_file_still_appends_crypto(tmp_path: Path) -> None:
    """ADR-0013: the equity seed file does not govern the crypto sleeve —
    crypto starters join in both branches, with their own origin string."""
    seed = tmp_path / "universe_seed.txt"
    seed.write_text("# core\nAAPL\nmsft\n\nSPY\n")
    store = FakeStore()
    make_manager(store, seed_path=str(seed)).seed_core(AS_OF)
    members = store.universe_at(AS_OF, Tier.CORE)
    assert {m.symbol for m in members} == {
        "AAPL",
        "MSFT",
        "SPY",
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "AVAX/USD",
        "LINK/USD",
    }
    reasons = {m.symbol: m.reason for m in members}
    assert reasons["BTC/USD"] == "core seed: crypto starter list"


def test_seed_core_crypto_disabled_seeds_equities_only(tmp_path: Path) -> None:
    seed = tmp_path / "universe_seed.txt"
    seed.write_text("AAPL\nSPY\n")
    store = FakeStore()
    make_manager(store, seed_path=str(seed), include_crypto=False).seed_core(AS_OF)
    assert {m.symbol for m in store.universe_at(AS_OF, Tier.CORE)} == {"AAPL", "SPY"}


def test_crypto_instruments_rows_carry_the_crypto_sector() -> None:
    from dynamo.data.universe import CRYPTO_STARTERS, crypto_instruments

    rows = crypto_instruments(CRYPTO_STARTERS)
    assert [r.symbol for r in rows] == list(CRYPTO_STARTERS)
    assert all(r.sector == "crypto" for r in rows)
    assert all(not r.is_etf for r in rows)


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
    manager.recommend_demotion(
        "OLDCO", "60d below liquidity floor", now=NOW, suggested_replacement="NEWCO"
    )
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


def test_starter_sectors_cover_every_equity_starter() -> None:
    """Every equity starter has a GICS sector that resolves to a sector ETF."""
    from dynamo.data.universe import (
        STARTER_SECTORS,
        STARTER_STOCKS,
        default_sector_etf_map,
    )

    assert set(STARTER_SECTORS) == set(STARTER_STOCKS)
    etf_map = default_sector_etf_map(STARTER_SECTORS)
    assert set(etf_map) == set(STARTER_STOCKS)  # every sector name recognized


# --- Listing-date backdating (D3) ------------------------------------------

BUILD_DAY = date(2026, 8, 17)  # the day universe-wide actually ran


def member(
    symbol: str, tier: Tier, valid_from: date, *, valid_to: date | None = None
) -> UniverseMembership:
    return UniverseMembership(
        symbol=symbol,
        tier=tier,
        valid_from=valid_from,
        valid_to=valid_to,
        reason="alpaca tradable universe (NASDAQ)",
        approved_by="universe-wide",
    )


def visible_at(members: list[UniverseMembership], as_of: date) -> set[str]:
    """The point-in-time read `universe_at` performs, in memory."""
    return {
        m.symbol
        for m in members
        if m.valid_from <= as_of and (m.valid_to is None or m.valid_to >= as_of)
    }


def test_backdate_moves_valid_from_to_the_first_bar() -> None:
    members = [member("OLD", Tier.SURVEILLANCE, BUILD_DAY)]
    plan = plan_listing_date_backdate(members, {"OLD": date(2024, 1, 2)})
    assert len(plan.rebases) == 1
    rebase = plan.rebases[0]
    assert (rebase.symbol, rebase.old_valid_from, rebase.new_valid_from) == (
        "OLD",
        BUILD_DAY,
        date(2024, 1, 2),
    )
    rebased = backdated_memberships(members, plan)
    assert rebased[0].valid_from == date(2024, 1, 2)
    # Everything else rides along unchanged.
    assert rebased[0].reason == members[0].reason
    assert rebased[0].approved_by == members[0].approved_by
    assert rebased[0].tier == Tier.SURVEILLANCE


def test_backdate_is_idempotent() -> None:
    """Re-running against the already-rebased rows plans nothing."""
    members = [member("OLD", Tier.SURVEILLANCE, BUILD_DAY)]
    first_bar = {"OLD": date(2024, 1, 2)}
    once = backdated_memberships(members, plan_listing_date_backdate(members, first_bar))
    twice = plan_listing_date_backdate(once, first_bar)
    assert twice.rebases == ()
    assert twice.already_backdated == ("OLD",)
    assert backdated_memberships(once, twice) == []


def test_backdate_never_touches_tier_1_core() -> None:
    """The 55 hand-approved CORE rows are the thing that must not move."""
    core = UniverseMembership(
        symbol="AAPL",
        tier=Tier.CORE,
        valid_from=date(2024, 12, 1),
        valid_to=None,
        reason="starter",
        approved_by="seed",
    )
    members = [core, member("WIDE", Tier.SURVEILLANCE, BUILD_DAY)]
    first_bar = {"AAPL": date(2024, 1, 2), "WIDE": date(2025, 6, 3)}
    plan = plan_listing_date_backdate(members, first_bar)
    assert [r.symbol for r in plan.rebases] == ["WIDE"]
    assert "AAPL" not in plan.already_backdated + plan.no_bars
    assert [m.symbol for m in backdated_memberships(members, plan)] == ["WIDE"]


def test_a_symbol_whose_first_bar_is_after_as_of_stays_invisible() -> None:
    """The look-ahead guard: a 2025-06 IPO must not appear in a 2024 universe.

    A blanket backdate would put IPO in the 2024-01 surveillance universe and
    the replay would "discover" a company that did not yet exist."""
    members = [
        member("OLD", Tier.SURVEILLANCE, BUILD_DAY),
        member("IPO", Tier.SURVEILLANCE, BUILD_DAY),
    ]
    first_bar = {"OLD": date(2024, 1, 2), "IPO": date(2025, 6, 16)}
    rebased = backdated_memberships(members, plan_listing_date_backdate(members, first_bar))
    assert visible_at(rebased, date(2024, 3, 1)) == {"OLD"}
    assert visible_at(rebased, date(2025, 6, 15)) == {"OLD"}
    assert visible_at(rebased, date(2025, 6, 16)) == {"OLD", "IPO"}


def test_a_symbol_with_no_bars_keeps_its_late_valid_from() -> None:
    """No bars means no listing-date evidence; guessing would be look-ahead."""
    members = [member("GHOST", Tier.SURVEILLANCE, BUILD_DAY)]
    plan = plan_listing_date_backdate(members, {})
    assert plan.rebases == ()
    assert plan.no_bars == ("GHOST",)
    assert visible_at(members, date(2026, 3, 3)) == set()


def test_backdate_never_moves_a_date_forward() -> None:
    """A row already earlier than its first bar is left alone: moving it
    forward would silently REMOVE the symbol from a universe it was in."""
    members = [member("EARLY", Tier.SURVEILLANCE, date(2024, 1, 1))]
    plan = plan_listing_date_backdate(members, {"EARLY": date(2025, 5, 5)})
    assert plan.rebases == ()
    assert plan.first_bar_after_valid_from == ("EARLY",)


def test_backdate_plan_summary_counts_every_bucket() -> None:
    members = [
        member("MOVE", Tier.SURVEILLANCE, BUILD_DAY),
        member("SAME", Tier.SURVEILLANCE, date(2024, 1, 2)),
        member("GHOST", Tier.SURVEILLANCE, BUILD_DAY),
        member("EARLY", Tier.SURVEILLANCE, date(2023, 1, 1)),
    ]
    first_bar = {
        "MOVE": date(2024, 1, 2),
        "SAME": date(2024, 1, 2),
        "EARLY": date(2025, 5, 5),
    }
    plan = plan_listing_date_backdate(members, first_bar)
    assert plan.summary() == (
        "1 rows to backdate, 1 already at their first bar, "
        "1 with no bars (left late), 1 already earlier than their first bar"
    )
