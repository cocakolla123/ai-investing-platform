"""pm_favorites_sim asset: the study gate (flag AND assumed edge), favsim
ledger fills, resolution-time settlement, idempotent re-runs — and PROOF that
this asset never constructs a broker class, even in LIVE with the gate open."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import pytest
from dagster import MaterializeResult, build_asset_context
from orchestration.assets import pm_trading
from orchestration.assets.pm_trading import PM_FAV_LEDGER_VENUE, pm_favorites_sim_asset

from dynamo.config.settings import Settings
from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import PMSimLedgerEntry
from tests.unit.orchestration.conftest import FakeDynamo
from tests.unit.pm.helpers import NOW

MARKET = "KXFAV-26SEP-A"


class FakeLedger:
    """PMSimLedgerRepository's surface over a class-level store (persists
    across constructions, like the database does across asset runs)."""

    store: ClassVar[dict[tuple[str, str, str], PMSimLedgerEntry]] = {}

    def __init__(self, engine: Any, venue: str = "polymarket") -> None:
        self.venue = venue

    def ensure_seeded(self, bankroll_cents: float, now: Any) -> None:
        self.append(
            PMSimLedgerEntry(
                venue=self.venue,
                kind="deposit",
                ref="initial",
                market_id=None,
                amount_cents=bankroll_cents,
                note="seed",
                ts=now,
            )
        )

    def append(self, entry: PMSimLedgerEntry) -> bool:
        key = (entry.venue, entry.kind, entry.ref)
        if key in FakeLedger.store:
            return False
        FakeLedger.store[key] = entry
        return True

    def cash_cents(self) -> float:
        return sum(e.amount_cents for e in FakeLedger.store.values() if e.venue == self.venue)

    def entries(self) -> list[dict[str, Any]]:
        return [
            {
                "venue": e.venue,
                "kind": e.kind,
                "ref": e.ref,
                "market_id": e.market_id,
                "amount_cents": e.amount_cents,
                "note": e.note,
                "ts": e.ts,
            }
            for e in FakeLedger.store.values()
            if e.venue == self.venue
        ]


def fav_entries() -> list[PMSimLedgerEntry]:
    return [
        e
        for e in FakeLedger.store.values()
        if e.venue == PM_FAV_LEDGER_VENUE and e.kind in ("fav_fill", "fav_settle")
    ]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)  # type: ignore[call-arg]


def enabled_settings(**overrides: Any) -> Settings:
    """The post-study configuration: flag on AND a demonstrated edge supplied."""
    values: dict[str, Any] = {"pm_favorites_enabled": True, "pm_fav_assumed_net_edge": 0.008}
    values.update(overrides)
    return make_settings(**values)


class FavDynamo(FakeDynamo):
    current_settings: ClassVar[Settings | None] = None
    resolved: ClassVar[bool | None] = None  # stored market's resolved_outcome

    def settings(self) -> Settings:
        assert FavDynamo.current_settings is not None
        return FavDynamo.current_settings

    def engine(self) -> Any:
        return object()


def stored_frames() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """One trusted 95c Kalshi favorite (fee 1c -> $0.96/contract all-in)."""
    markets = {
        MARKET: PredictionMarket(
            venue="kalshi",
            market_id=MARKET,
            question=f"{MARKET}?",
            resolved_outcome=FavDynamo.resolved,
        )
    }
    prices = {
        MARKET: PredictionMarketPrice(
            venue="kalshi", market_id=MARKET, ts=NOW, implied_prob=0.95, trusted=True
        )
    }
    return {}, prices, markets


@pytest.fixture
def fav_env(monkeypatch: pytest.MonkeyPatch) -> FavDynamo:
    FakeLedger.store = {}
    FavDynamo.current_settings = make_settings()
    FavDynamo.resolved = None

    def fake_frames(dynamo: Any, venue: str = "kalshi") -> Any:
        return stored_frames()

    monkeypatch.setattr(pm_trading, "_latest_pm_frames", fake_frames)
    monkeypatch.setattr("dynamo.persistence.repositories.pm.PMSimLedgerRepository", FakeLedger)
    return FavDynamo(state_key="pm-fav-test")


def run_asset(dynamo: FavDynamo) -> MaterializeResult:
    return cast(MaterializeResult, pm_favorites_sim_asset(build_asset_context(), dynamo))


def test_default_flag_is_off_and_skips_everything(fav_env: FavDynamo) -> None:
    result = run_asset(fav_env)  # default settings: pm_favorites_enabled=False
    assert result.metadata is not None
    assert "pm_calibration" in str(result.metadata["skipped"])
    assert FakeLedger.store == {}  # not even the seed deposit


def test_enabled_without_demonstrated_edge_still_selects_nothing(fav_env: FavDynamo) -> None:
    # Flipping the flag without copying a demonstrated edge from the study
    # must stay inert: the selector's <= 0 gate is the second fail-closed.
    FavDynamo.current_settings = make_settings(pm_favorites_enabled=True)
    result = run_asset(fav_env)
    assert result.metadata is not None
    assert result.metadata["candidates"] == 0
    assert result.metadata["recorded"] == 0
    assert fav_entries() == []


def test_enabled_with_edge_records_one_fill_per_market_per_day(fav_env: FavDynamo) -> None:
    FavDynamo.current_settings = enabled_settings()
    result = run_asset(fav_env)
    assert result.metadata is not None
    assert result.metadata["candidates"] == 1
    assert result.metadata["recorded"] == 1
    assert result.metadata["settlements"] == 0
    (fill,) = fav_entries()
    assert fill.kind == "fav_fill"
    assert fill.note.startswith("fav:harvest contracts=26 ")
    # $25 stake cap / $0.96 all-in -> 26 contracts, debited at cost.
    assert math.isclose(fill.amount_cents, -(26 * 96.0))


def test_rerun_same_day_is_idempotent(fav_env: FavDynamo) -> None:
    FavDynamo.current_settings = enabled_settings()
    first = run_asset(fav_env)
    entries_after_first = dict(FakeLedger.store)
    second = run_asset(fav_env)
    assert first.metadata is not None and second.metadata is not None
    assert first.metadata["recorded"] == 1
    assert second.metadata["recorded"] == 0  # same ref -> ledger dedupes
    assert FakeLedger.store == entries_after_first


@pytest.mark.parametrize(("outcome", "credit_cents"), [(True, 26 * 100.0), (False, 0.0)])
def test_settles_resolved_fills_at_resolution(
    fav_env: FavDynamo, outcome: bool, credit_cents: float
) -> None:
    FavDynamo.current_settings = enabled_settings()
    run_asset(fav_env)  # day 1: buy 26 contracts
    FavDynamo.resolved = outcome
    result = run_asset(fav_env)
    assert result.metadata is not None
    assert result.metadata["settlements"] == 1
    settles = [e for e in fav_entries() if e.kind == "fav_settle"]
    assert len(settles) == 1
    assert math.isclose(settles[0].amount_cents, credit_cents)
    expected_pnl = (credit_cents - 26 * 96.0) / 100.0
    assert math.isclose(cast(float, result.metadata["realized_pnl_usd"]), expected_pnl)
    # Settling again is a no-op: the zero/full credit row marks it settled.
    again = run_asset(fav_env)
    assert again.metadata is not None
    assert again.metadata["settlements"] == 0


def test_live_broker_is_never_constructed(
    fav_env: FavDynamo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sim path is hard-coded: even in LIVE with live trading enabled and
    credentials present, materializing the favorites asset must not construct
    ANY broker class — the harvester never routes to a broker."""
    import dynamo.execution.kalshi_broker as kalshi_broker_mod
    import dynamo.execution.polymarket_broker as polymarket_broker_mod
    import dynamo.execution.polymarket_sim as polymarket_sim_mod

    def forbidden(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("broker constructed by the favorites asset")

    monkeypatch.setattr(kalshi_broker_mod.KalshiBroker, "__init__", forbidden)
    monkeypatch.setattr(polymarket_broker_mod.PolymarketBroker, "__init__", forbidden)
    monkeypatch.setattr(polymarket_sim_mod.PolymarketSimBroker, "__init__", forbidden)

    FavDynamo.current_settings = enabled_settings(
        DYNAMO_ENV="LIVE",
        polymarket_live_enabled=True,
        KALSHI_API_KEY_ID="key",
        KALSHI_PRIVATE_KEY_PATH="/tmp/kalshi.pem",
    )
    result = run_asset(fav_env)  # would raise if any broker were constructed
    assert result.metadata is not None
    assert result.metadata["recorded"] == 1  # the asset really ran its full path
    assert len(fav_entries()) == 1
