"""pm_arbitrage_sim asset: flag gate, strategy-tagged arbsim ledger rows,
idempotent re-runs, the unverified-pair guard — and PROOF that this asset
never constructs a live broker class, even in LIVE with the gate wide open."""

from __future__ import annotations

import math
from typing import Any, ClassVar, cast

import pytest
from dagster import MaterializeResult, build_asset_context
from orchestration.assets import pm_trading
from orchestration.assets.pm_trading import PM_ARB_LEDGER_VENUE, pm_arbitrage_sim_asset

from dynamo.config.settings import Settings
from dynamo.domain.models.market import PredictionMarket, PredictionMarketPrice
from dynamo.domain.models.pm import CrossVenueSpread, PMSimLedgerEntry
from tests.unit.orchestration.conftest import FakeDynamo
from tests.unit.pm.helpers import NOW

EVENT = "KXEV-26SEP"
PAIR_ID = "KXPAIR-26SEP-A|poly-a"


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


def arb_entries() -> list[PMSimLedgerEntry]:
    return [
        e
        for e in FakeLedger.store.values()
        if e.venue == PM_ARB_LEDGER_VENUE and e.kind in ("arb_fill", "arb_lock")
    ]


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"_env_file": None}
    values.update(overrides)
    return Settings(**values)  # type: ignore[call-arg]


class ArbDynamo(FakeDynamo):
    """FakeDynamo with injectable settings and an opaque engine sentinel
    (only ever handed to the patched FakeLedger, never dereferenced)."""

    current_settings: ClassVar[Settings | None] = None

    def settings(self) -> Settings:
        assert ArbDynamo.current_settings is not None
        return ArbDynamo.current_settings

    def engine(self) -> Any:
        return object()


def stored_frames() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A profitable 2-outcome Kalshi dutch book (asks 2 x 0.30 + 0.04 fees)."""
    markets = {
        f"{EVENT}-{s}": PredictionMarket(
            venue="kalshi", market_id=f"{EVENT}-{s}", question=f"{EVENT}-{s}?"
        )
        for s in ("A", "B")
    }
    prices = {
        market_id: PredictionMarketPrice(
            venue="kalshi", market_id=market_id, ts=NOW, implied_prob=0.30, trusted=True
        )
        for market_id in markets
    }
    return {}, prices, markets


def stored_pairs() -> list[CrossVenueSpread]:
    """A profitable cross-venue pair (YES kalshi 0.30 / NO polymarket 0.55)."""
    return [
        CrossVenueSpread(
            question="pair?",
            kalshi_market_id="KXPAIR-26SEP-A",
            kalshi_yes=0.30,
            polymarket_id="poly-a",
            polymarket_yes=0.45,
            yes_diff=-0.15,
            as_of=NOW,
        )
    ]


@pytest.fixture
def arb_env(monkeypatch: pytest.MonkeyPatch) -> ArbDynamo:
    """Fresh ledger store, stored data patched in, DB repo swapped for the fake."""
    FakeLedger.store = {}
    ArbDynamo.current_settings = make_settings()

    def fake_frames(dynamo: Any, venue: str = "kalshi") -> Any:
        return stored_frames()

    monkeypatch.setattr(pm_trading, "_latest_pm_frames", fake_frames)
    monkeypatch.setattr(pm_trading, "_latest_spread_pairs", lambda dynamo: stored_pairs())
    monkeypatch.setattr("dynamo.persistence.repositories.pm.PMSimLedgerRepository", FakeLedger)
    return ArbDynamo(state_key="pm-arb-test")


def run_asset(dynamo: ArbDynamo) -> MaterializeResult:
    return cast(MaterializeResult, pm_arbitrage_sim_asset(build_asset_context(), dynamo))


def test_disabled_flag_skips_everything(arb_env: ArbDynamo) -> None:
    ArbDynamo.current_settings = make_settings(pm_arbitrage_enabled=False)
    result = run_asset(arb_env)
    assert result.metadata is not None
    assert result.metadata["skipped"] == "pm arbitrage disabled"
    assert FakeLedger.store == {}  # not even the seed deposit


def test_records_dutch_trades_with_strategy_tag(arb_env: ArbDynamo) -> None:
    # Default settings: pm_arb_verified_pairs is EMPTY, so the profitable
    # cross-venue pair stays research-only and ONLY the dutch book trades.
    result = run_asset(arb_env)
    assert result.metadata is not None
    assert result.metadata["dutch_candidates"] == 1
    assert result.metadata["xvenue_candidates"] == 0
    assert result.metadata["xvenue_unverified"] == 1
    assert result.metadata["recorded"] == 1

    entries = arb_entries()
    fills = [e for e in entries if e.kind == "arb_fill"]
    locks = [e for e in entries if e.kind == "arb_lock"]
    assert len(fills) == 1 and len(locks) == 1
    assert fills[0].note.startswith("arb:dutch")
    assert not any(e.note.startswith("arb:xvenue") for e in entries)
    # Sized inside the $100 stake cap: asks 2 x (0.30 + 0.01 slippage) plus
    # 2 x 2c Kalshi fees = $0.66/book -> floor(100 / 0.66) = 151 books.
    assert math.isclose(fills[0].amount_cents, -(151 * 66.0))
    assert math.isclose(locks[0].amount_cents, 151 * 100.0)


def test_verified_pair_trades_with_xvenue_tag(arb_env: ArbDynamo) -> None:
    ArbDynamo.current_settings = make_settings(pm_arb_verified_pairs=(PAIR_ID,))
    result = run_asset(arb_env)
    assert result.metadata is not None
    assert result.metadata["xvenue_candidates"] == 1
    assert result.metadata["xvenue_unverified"] == 0
    assert result.metadata["recorded"] == 2
    xvenue_fills = [
        e for e in arb_entries() if e.kind == "arb_fill" and e.note.startswith("arb:xvenue")
    ]
    assert len(xvenue_fills) == 1
    assert PAIR_ID in xvenue_fills[0].note


def test_rerun_same_day_is_idempotent(arb_env: ArbDynamo) -> None:
    first = run_asset(arb_env)
    entries_after_first = dict(FakeLedger.store)
    second = run_asset(arb_env)
    assert first.metadata is not None and second.metadata is not None
    assert first.metadata["recorded"] == 1
    assert second.metadata["recorded"] == 0  # same refs -> ledger dedupes
    assert FakeLedger.store == entries_after_first


def test_live_broker_is_never_constructed(
    arb_env: ArbDynamo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sim path is hard-coded: even in LIVE with live trading enabled and
    credentials present, materializing the arbitrage asset must not construct
    ANY broker class — arbitrage never routes to the live broker."""
    import dynamo.execution.kalshi_broker as kalshi_broker_mod
    import dynamo.execution.polymarket_broker as polymarket_broker_mod
    import dynamo.execution.polymarket_sim as polymarket_sim_mod

    def forbidden(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("broker constructed by the arbitrage asset")

    monkeypatch.setattr(kalshi_broker_mod.KalshiBroker, "__init__", forbidden)
    monkeypatch.setattr(polymarket_broker_mod.PolymarketBroker, "__init__", forbidden)
    monkeypatch.setattr(polymarket_sim_mod.PolymarketSimBroker, "__init__", forbidden)

    ArbDynamo.current_settings = make_settings(
        DYNAMO_ENV="LIVE",
        polymarket_live_enabled=True,
        KALSHI_API_KEY_ID="key",
        KALSHI_PRIVATE_KEY_PATH="/tmp/kalshi.pem",
        pm_arb_verified_pairs=(PAIR_ID,),
    )
    result = run_asset(arb_env)  # would raise if any broker were constructed
    assert result.metadata is not None
    assert result.metadata["recorded"] == 2  # the asset really ran its full path
    assert len(arb_entries()) == 4
