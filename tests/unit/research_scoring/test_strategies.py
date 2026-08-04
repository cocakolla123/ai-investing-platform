"""Strategy golden fixtures: tiny frames -> exact Signals; universe filter; dedup."""

from __future__ import annotations

from datetime import date
from typing import cast

import pandas as pd
import pytest

from dynamo.domain.models.research import SignalStatus
from dynamo.research.strategies import (
    DEFAULT_STRATEGIES,
    Rule,
    Strategy,
    StrategyRiskParams,
    UniverseFilter,
    evaluate_exit,
    evaluate_strategies,
    meanrev_v1,
    momentum_rs_v1,
)

SESSIONS = pd.bdate_range("2026-07-20", periods=6)
AS_OF = cast(pd.Timestamp, SESSIONS[-1])
PREV_SESSION = cast(pd.Timestamp, SESSIONS[-2])


def _history(per_symbol: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Build a (ts, symbol) feature-history frame. Scalars broadcast across sessions."""
    rows = []
    for sym, cols in per_symbol.items():
        for i, ts in enumerate(SESSIONS):
            row: dict[str, object] = {"ts": ts, "symbol": sym}
            for col, val in cols.items():
                row[col] = val[i] if isinstance(val, list) else val
            rows.append(row)
    return pd.DataFrame(rows).set_index(["ts", "symbol"]).sort_index()


BASE = {
    "tier": 1.0,
    "dollar_vol_20d": 2e7,
    "rs_20d_sector": 0.05,
    "dist_ma50": 0.02,
    "dist_ma200": 0.10,
    "close": 100.0,
    "atr_14": 2.0,
    "rsi_14": 50.0,
    "ret_5d_z": 0.0,
}


def test_default_registry_has_doc_05_strategies() -> None:
    assert set(DEFAULT_STRATEGIES) == {
        "momentum-rs-v1",
        "breakout-v1",
        "meanrev-v1",
        "earnings-react-v1",
    }
    mom = DEFAULT_STRATEGIES["momentum-rs-v1"]
    assert mom.holding_period_days == 40
    assert mom.risk_params.stop_atr_multiple == 2.5
    assert mom.universe_filter.min_dollar_volume == 10_000_000.0
    assert DEFAULT_STRATEGIES["breakout-v1"].holding_period_days == 30
    assert DEFAULT_STRATEGIES["meanrev-v1"].holding_period_days == 10
    assert DEFAULT_STRATEGIES["meanrev-v1"].risk_params.stop_atr_multiple == 1.5
    assert DEFAULT_STRATEGIES["earnings-react-v1"].holding_period_days == 20


def test_momentum_golden_signal() -> None:
    history = _history(
        {
            "AAA": dict(BASE, mom_12_1=0.50),  # top quintile of 3 -> pctile 1.0
            "BBB": dict(BASE, mom_12_1=0.20),
            "CCC": dict(BASE, mom_12_1=0.10),
        }
    )
    signals = evaluate_strategies([momentum_rs_v1()], history, AS_OF)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.strategy_id == "momentum-rs-v1"
    assert sig.symbol == "AAA"
    assert sig.as_of == date(2026, 7, 27)
    assert sig.status == SignalStatus.ACTIVE
    assert sig.horizon_days == 40
    assert sig.entry_ref_price == 100.0
    assert sig.stop_hint == pytest.approx(100.0 - 2.5 * 2.0)  # close - 2.5 x ATR(14)
    # strength = 0.5 + 0.5 x mean(quantile depth 1.0, 1.0, 1.0) = 1.0
    assert sig.strength == pytest.approx(1.0)
    rules = cast(list[dict[str, object]], sig.rule_snapshot["rules"])
    assert [r["feature"] for r in rules] == ["mom_12_1", "rs_20d_sector", "dist_ma50"]
    assert all(r["passed"] for r in rules)
    assert rules[0]["observed"] == pytest.approx(0.50)


def test_meanrev_golden_signal_and_uptrend_filter() -> None:
    history = _history(
        {
            # oversold in an uptrend -> fires
            "AAA": dict(BASE, rsi_14=25.0, ret_5d_z=-2.5),
            # oversold but in a downtrend (below 200d MA) -> universe filter blocks
            "BBB": dict(BASE, rsi_14=25.0, ret_5d_z=-2.5, dist_ma200=-0.05),
            "CCC": dict(BASE),
        },
    )
    # give everyone positive 12-1 momentum for the uptrend rule
    history["mom_12_1"] = 0.1
    signals = evaluate_strategies([meanrev_v1()], history, AS_OF)
    assert [s.symbol for s in signals] == ["AAA"]
    sig = signals[0]
    assert sig.strategy_id == "meanrev-v1"
    assert sig.horizon_days == 10
    assert sig.stop_hint == pytest.approx(100.0 - 1.5 * 2.0)
    assert sig.strength == pytest.approx(1.0)  # comparison rules score full depth


def test_universe_filter_liquidity_tier_and_exclusions() -> None:
    history = _history(
        {
            "ILLIQ": dict(BASE, mom_12_1=0.9, dollar_vol_20d=1e6),  # below $10M floor
            "TIER3": dict(BASE, mom_12_1=0.8, tier=3.0),  # not tier 1
            "EXCL": dict(BASE, mom_12_1=0.7),
            "OK": dict(BASE, mom_12_1=0.95),  # top of the cross-section
        }
    )
    strategy = momentum_rs_v1().model_copy(
        update={
            "universe_filter": momentum_rs_v1().universe_filter.model_copy(
                update={"exclusions": ("EXCL",)}
            )
        }
    )
    signals = evaluate_strategies([strategy], history, AS_OF)
    assert [s.symbol for s in signals] == ["OK"]


def test_overlap_dedup_winner_by_strength() -> None:
    # Ten symbols so the momentum quantile depth is graded: AAA sits at the
    # 90th percentile (depth 0.5, strength ~0.917) and also fires meanrev
    # (strength 1.0) -> meanrev wins, momentum is suppressed.
    others: dict[str, dict[str, object]] = {
        f"S{i}": dict(BASE, mom_12_1=0.1 * i) for i in range(9)
    }
    history = _history(
        {**others, "AAA": dict(BASE, mom_12_1=0.75, rsi_14=25.0, ret_5d_z=-2.5)}
    )
    signals = evaluate_strategies([momentum_rs_v1(), meanrev_v1()], history, AS_OF)
    aaa = [s for s in signals if s.symbol == "AAA"]
    assert len(aaa) == 2
    winner = next(s for s in aaa if s.status == SignalStatus.ACTIVE)
    loser = next(s for s in aaa if s.status == SignalStatus.SUPPRESSED)
    assert winner.strategy_id == "meanrev-v1"
    assert winner.strength > loser.strength
    assert loser.rule_snapshot["suppressed_by"] == "meanrev-v1"


def test_window_sessions_rule_must_hold_consecutively() -> None:
    strategy = Strategy(
        strategy_id="window-test",
        universe_filter=UniverseFilter(tiers=()),
        entry_rules=(Rule(feature="flag", op="gt", value=0.0, window_sessions=3),),
        exit_rules=(),
        holding_period_days=5,
        risk_params=StrategyRiskParams(),
    )
    holds = _history({"AAA": {"flag": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}})
    broken = _history({"AAA": {"flag": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0]}})
    assert len(evaluate_strategies([strategy], holds, AS_OF)) == 1
    assert len(evaluate_strategies([strategy], broken, AS_OF)) == 0


def test_crossed_below_against_feature_column_and_exit() -> None:
    # close crosses below the post-earnings low on the final session
    history = _history(
        {
            "AAA": dict(
                BASE,
                close=[100.0, 100.0, 100.0, 100.0, 96.0, 94.0],
                post_earnings_low=95.0,
            )
        }
    )
    strategy = DEFAULT_STRATEGIES["earnings-react-v1"]
    assert evaluate_exit(strategy, history, "AAA", AS_OF) is True
    # a day earlier close (96) was still above the level -> no exit
    assert evaluate_exit(strategy, history, "AAA", PREV_SESSION) is False


def test_exit_window_rule_rs_negative_five_sessions() -> None:
    strategy = momentum_rs_v1()
    rs_5neg = _history({"AAA": dict(BASE, rs_20d_sector=[0.1, -0.1, -0.1, -0.1, -0.1, -0.1])})
    rs_4neg = _history({"AAA": dict(BASE, rs_20d_sector=[0.1, 0.1, -0.1, -0.1, -0.1, -0.1])})
    assert evaluate_exit(strategy, rs_5neg, "AAA", AS_OF) is True
    assert evaluate_exit(strategy, rs_4neg, "AAA", AS_OF) is False


def test_no_signal_for_symbol_outside_universe_property() -> None:
    """No strategy ever emits a signal for a symbol failing its universe filter."""
    history = _history(
        {
            "GOOD": dict(BASE, mom_12_1=0.9, rsi_14=25.0, ret_5d_z=-2.5),
            "BAD": dict(
                BASE, mom_12_1=0.95, rsi_14=20.0, ret_5d_z=-3.0, dollar_vol_20d=1.0, tier=2.0
            ),
        }
    )
    for strategy in DEFAULT_STRATEGIES.values():
        for sig in evaluate_strategies([strategy], history, AS_OF):
            assert sig.symbol != "BAD"


def test_deterministic_signal_emission() -> None:
    history = _history(
        {
            "AAA": dict(BASE, mom_12_1=0.5, rsi_14=25.0, ret_5d_z=-2.5),
            "BBB": dict(BASE, mom_12_1=0.2),
            "CCC": dict(BASE, mom_12_1=0.1),
        }
    )
    strategies = list(DEFAULT_STRATEGIES.values())
    once = [s.model_dump() for s in evaluate_strategies(strategies, history, AS_OF)]
    twice = [s.model_dump() for s in evaluate_strategies(strategies, history, AS_OF)]
    assert once == twice
