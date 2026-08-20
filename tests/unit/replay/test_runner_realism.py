"""Bankroll-realism knobs in the replay runner (backtest-campaign Phase 3).

whole_shares floors equity share quantities (crypto exempt); min_order_notional
skips dust orders; commission_per_order reduces equity by count x fee; the
defaults (False/0/0) are bit-identical to the pre-change fill path.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from dynamo.replay.runner import ReplayConfig, ReplayRunner
from tests.unit.replay.test_runner import (
    RUN_SESSIONS,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    fixture_frames,  # noqa: F401  (module-scoped fixture reused here)
)


def _run(fixture: Any, **overrides: Any) -> Any:
    sessions, bars, scores = fixture
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(
        start=window[0],
        end=window[-1],
        label="realism-test",
        horizon_sessions=3,
        **overrides,
    )
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=config,
    )
    return runner.run()


# -- bit-parity: defaults change nothing -------------------------------------


def test_explicit_default_knobs_are_bit_identical_to_unset(fixture_frames: Any) -> None:  # noqa: F811
    baseline = _run(fixture_frames)
    explicit = _run(
        fixture_frames,
        whole_shares=False,
        min_order_notional=0.0,
        commission_per_order=0.0,
    )
    # Same run id (defaults ARE the unset values) and same everything else.
    assert explicit.run_id == baseline.run_id
    assert explicit.fills == baseline.fills
    assert explicit.equity_curve == baseline.equity_curve
    assert explicit.decisions == baseline.decisions


# -- whole shares ($100k) ----------------------------------------------------


def test_whole_shares_floors_equity_buys_and_divergence_is_bounded(
    fixture_frames: Any,  # noqa: F811
) -> None:
    fractional = _run(fixture_frames, initial_equity=100_000.0)
    whole = _run(fixture_frames, initial_equity=100_000.0, whole_shares=True)
    assert whole.run_id != fractional.run_id  # knob is in config_hash
    whole_buys = [f for f in whole.fills if f.side == "buy"]
    assert whole_buys
    for f in whole_buys:
        assert f.qty == math.floor(f.qty), f  # every buy is a whole number
    # Full exits sell the exact held (whole) quantity.
    for f in whole.fills:
        if f.side == "sell":
            assert f.qty == math.floor(f.qty) or f.qty < 1.0, f
    # At $100k the flooring residue is pennies-per-share: terminal equity
    # diverges from the fractional run by a bounded, sensible amount.
    frac_end = fractional.equity_curve[-1][1]
    whole_end = whole.equity_curve[-1][1]
    assert abs(whole_end - frac_end) / frac_end < 0.01
    assert whole_end != frac_end  # but it DOES diverge — the knob is real


# -- min order notional ($1k) ------------------------------------------------


def test_min_order_notional_skips_dust_orders_at_1k(fixture_frames: Any) -> None:  # noqa: F811
    baseline = _run(fixture_frames, initial_equity=1_000.0)
    constrained = _run(fixture_frames, initial_equity=1_000.0, min_order_notional=25.0)
    # The $1k fixture produces real ~$80 entries AND sub-dollar rebalance
    # trims — the floor must bite on the trims only.
    dust = [f for f in baseline.fills if f.qty * f.price < 25.0]
    assert dust, "fixture must produce sub-minimum orders for the test to bite"
    for f in constrained.fills:
        assert f.qty * f.price >= 25.0 - 0.01, f
    assert len(constrained.fills) < len(baseline.fills)
    assert any(f.side == "buy" for f in constrained.fills)  # real entries kept


# -- commission --------------------------------------------------------------


def test_commission_reduces_equity_by_count_times_fee(fixture_frames: Any) -> None:  # noqa: F811
    fee = 1.0
    baseline = _run(fixture_frames, initial_equity=100_000.0)
    with_fee = _run(fixture_frames, initial_equity=100_000.0, commission_per_order=fee)
    # First fill day: decisions were made from identical day-0 state, so the
    # fills match 1:1 and equity drops by exactly count x fee.
    first_fill_day = min(f.session for f in baseline.fills)
    n_orders = sum(1 for f in baseline.fills if f.session == first_fill_day)
    assert n_orders > 0
    base_curve = dict(baseline.equity_curve)
    fee_curve = dict(with_fee.equity_curve)
    assert fee_curve[first_fill_day] == pytest.approx(
        base_curve[first_fill_day] - n_orders * fee, abs=0.011
    )
    # Per-order cost carries the commission.
    base_costs = {
        (f.session, f.symbol, f.side): f.cost for f in baseline.fills if f.session == first_fill_day
    }
    for f in with_fee.fills:
        if f.session == first_fill_day:
            assert f.cost == pytest.approx(
                base_costs[(f.session, f.symbol, f.side)] + fee, abs=0.011
            )


# -- crypto exemption --------------------------------------------------------


def test_crypto_is_exempt_from_whole_share_flooring() -> None:
    from tests.unit.replay.test_runner_crypto import (
        END,
        HISTORY_SESSIONS,
        make_bars,
        make_scores,
        nyse_sessions,
    )
    from tests.unit.replay.test_runner_crypto import (
        FakeExperienceRepo as CryptoExpRepo,
    )
    from tests.unit.replay.test_runner_crypto import (
        FakeGateway as CryptoGateway,
    )
    from tests.unit.replay.test_runner_crypto import (
        FakeRunRepo as CryptoRunRepo,
    )

    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    bars, scores = make_bars(sessions), make_scores(sessions)
    window = sessions[-RUN_SESSIONS:]

    def run(whole_shares: bool) -> Any:
        config = ReplayConfig(
            start=window[0],
            end=window[-1],
            label="crypto-whole-shares",
            horizon_sessions=30,
            whole_shares=whole_shares,
        )
        runner = ReplayRunner(
            gateway_factory=lambda d: CryptoGateway(d, bars, scores),
            experience_repo=CryptoExpRepo(),
            run_repo=CryptoRunRepo(),
            config=config,
        )
        return runner.run()

    fractional = run(whole_shares=False)
    whole = run(whole_shares=True)
    btc = [f for f in whole.fills if f.symbol == "BTC/USD"]
    assert btc, "the crypto pair must trade"
    assert any(f.qty != math.floor(f.qty) for f in btc)  # fractional native
    # Crypto-only universe: flooring exempts everything, so the runs are
    # identical except for the (knob-carrying) run id.
    assert whole.fills == fractional.fills
    assert whole.equity_curve == fractional.equity_curve
    assert whole.run_id != fractional.run_id
