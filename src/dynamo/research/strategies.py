"""Tactical strategy registry and Rule language — doc 05 section 3.

Strategies are parameterized rule objects, not code forks. ``Rule`` is a small
typed predicate language evaluated over nightly feature history frames (raw
feature values from ``research/quant``, plus detector-flag columns) —
declarative enough to render on the dashboard and diff in PRs.

Signal emission is deterministic: same inputs, same ``Signal`` rows.
Overlapping signals on one symbol dedup to the strongest; the losers are
recorded as SUPPRESSED with the winner noted in their rule snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dynamo.domain.models.research import Signal, SignalStatus

Op = Literal[
    "gt",
    "lt",
    "gte",
    "lte",
    "eq",
    "in_top_quantile",
    "in_bottom_quantile",
    "crossed_above",
    "crossed_below",
]

__all__ = [
    "DEFAULT_STRATEGIES",
    "Rule",
    "Strategy",
    "StrategyRiskParams",
    "UniverseFilter",
    "breakout_v1",
    "earnings_react_v1",
    "evaluate_exit",
    "evaluate_strategies",
    "meanrev_v1",
    "momentum_rs_v1",
]


class Rule(BaseModel):
    """Typed predicate over one feature column (doc 05 section 3.2).

    ``value`` may name another feature column for the crossed_* ops (e.g.
    close crossed_below "post_earnings_low"). The rule must hold for
    ``window_sessions`` consecutive sessions ending at ``as_of``.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    op: Op
    value: float | str
    window_sessions: int = Field(ge=1, default=1)


class UniverseFilter(BaseModel):
    """Tiers, liquidity floor, exclusions, plus extra state rules (uptrend-only etc.)."""

    model_config = ConfigDict(frozen=True)

    tiers: tuple[int, ...] = (1,)
    min_dollar_volume: float | None = None
    dollar_volume_feature: str = "dollar_vol_20d"
    exclusions: tuple[str, ...] = ()
    extra_rules: tuple[Rule, ...] = ()


class StrategyRiskParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    stop_atr_multiple: float = 2.0
    trailing_stop: bool = True
    max_concurrent_signals: int = 10
    size_hint_pct: float = 2.0


class Strategy(BaseModel):
    """Registry entry (doc 05 section 3). ``status`` written only by the promotion tool."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    universe_filter: UniverseFilter
    entry_rules: tuple[Rule, ...]
    exit_rules: tuple[Rule, ...]
    holding_period_days: int
    risk_params: StrategyRiskParams
    status: Literal["candidate", "paper", "production", "retired"] = "candidate"


# ---------------------------------------------------------------------------
# Rule evaluation over feature history frames
# ---------------------------------------------------------------------------


def _to_ts(value: object) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))  # type: ignore[arg-type]


def _session_at(sessions: pd.DatetimeIndex, pos: int) -> pd.Timestamp:
    return cast(pd.Timestamp, sessions[pos])


def _sessions(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DatetimeIndex:
    ts = pd.DatetimeIndex(history.index.get_level_values(0).unique()).sort_values()
    return cast(pd.DatetimeIndex, ts[ts <= as_of])


def _value_at(history: pd.DataFrame, ts: pd.Timestamp, symbol: str, column: str) -> float:
    try:
        v = history.loc[(ts, symbol), column]
    except KeyError:
        return float("nan")
    return float(v) if pd.notna(v) else float("nan")


def _threshold_at(
    history: pd.DataFrame, ts: pd.Timestamp, symbol: str, value: float | str
) -> float:
    if isinstance(value, str):
        return _value_at(history, ts, symbol, value)
    return float(value)


def _cross_section(history: pd.DataFrame, ts: pd.Timestamp, column: str) -> pd.Series:
    try:
        xs = history.xs(ts, level=0)[column]
    except KeyError:
        return pd.Series(dtype=float)
    return cast(pd.Series, xs).astype(float).dropna()


def _pctile(history: pd.DataFrame, ts: pd.Timestamp, symbol: str, column: str) -> float:
    xs = _cross_section(history, ts, column)
    if symbol not in xs.index or len(xs) == 0:
        return float("nan")
    return float(xs.rank(method="average", pct=True)[symbol])


def _holds_at(
    history: pd.DataFrame, rule: Rule, ts: pd.Timestamp, prev_ts: pd.Timestamp | None, symbol: str
) -> tuple[bool, float]:
    """(rule holds at ts, observed feature value)."""
    x = _value_at(history, ts, symbol, rule.feature)
    if rule.op in ("in_top_quantile", "in_bottom_quantile"):
        pct = _pctile(history, ts, symbol, rule.feature)
        if pd.isna(pct):
            return False, x
        q = float(rule.value)  # type: ignore[arg-type]
        held = pct > 1.0 - q if rule.op == "in_top_quantile" else pct <= q
        return held, x
    if pd.isna(x):
        return False, x
    if rule.op in ("crossed_above", "crossed_below"):
        if prev_ts is None:
            return False, x
        prev_x = _value_at(history, prev_ts, symbol, rule.feature)
        thr = _threshold_at(history, ts, symbol, rule.value)
        prev_thr = _threshold_at(history, prev_ts, symbol, rule.value)
        if pd.isna(prev_x) or pd.isna(thr) or pd.isna(prev_thr):
            return False, x
        if rule.op == "crossed_above":
            return prev_x <= prev_thr and x > thr, x
        return prev_x >= prev_thr and x < thr, x
    thr = _threshold_at(history, ts, symbol, rule.value)
    if pd.isna(thr):
        return False, x
    if rule.op == "gt":
        return x > thr, x
    if rule.op == "lt":
        return x < thr, x
    if rule.op == "gte":
        return x >= thr, x
    if rule.op == "lte":
        return x <= thr, x
    return x == thr, x  # eq


def _rule_holds(
    history: pd.DataFrame, rule: Rule, symbol: str, sessions: pd.DatetimeIndex
) -> tuple[bool, float]:
    """Rule holds for its full window of consecutive sessions ending at the last one."""
    if len(sessions) < rule.window_sessions:
        return False, float("nan")
    observed = float("nan")
    for i in range(rule.window_sessions):
        pos = len(sessions) - 1 - i
        prev_ts = _session_at(sessions, pos - 1) if pos >= 1 else None
        held, x = _holds_at(history, rule, _session_at(sessions, pos), prev_ts, symbol)
        if i == 0:
            observed = x
        if not held:
            return False, observed
    return True, observed


def _rule_depth(history: pd.DataFrame, rule: Rule, symbol: str, ts: pd.Timestamp) -> float:
    """Satisfaction depth in [0, 1] for strength. Quantile rules grade by how
    deep into the quantile the symbol sits; other satisfied rules score 1."""
    if rule.op in ("in_top_quantile", "in_bottom_quantile"):
        pct = _pctile(history, ts, symbol, rule.feature)
        q = float(rule.value)  # type: ignore[arg-type]
        if pd.isna(pct) or q <= 0:
            return 0.0
        depth = (pct - (1.0 - q)) / q if rule.op == "in_top_quantile" else (q - pct) / q
        return float(min(1.0, max(0.0, depth)))
    return 1.0


def _strength(history: pd.DataFrame, rules: Iterable[Rule], symbol: str, ts: pd.Timestamp) -> float:
    depths = [_rule_depth(history, r, symbol, ts) for r in rules]
    mean_depth = sum(depths) / len(depths) if depths else 0.0
    return float(min(1.0, max(0.0, 0.5 + 0.5 * mean_depth)))


def _passes_universe(
    history: pd.DataFrame,
    flt: UniverseFilter,
    symbol: str,
    sessions: pd.DatetimeIndex,
) -> bool:
    if symbol in flt.exclusions:
        return False
    ts = _session_at(sessions, -1)
    if flt.tiers and "tier" in history.columns:
        tier = _value_at(history, ts, symbol, "tier")
        if pd.isna(tier) or int(tier) not in flt.tiers:
            return False
    if flt.min_dollar_volume is not None:
        dv = _value_at(history, ts, symbol, flt.dollar_volume_feature)
        if pd.isna(dv) or dv < flt.min_dollar_volume:
            return False
    return all(_rule_holds(history, rule, symbol, sessions)[0] for rule in flt.extra_rules)


def _snapshot(
    history: pd.DataFrame, rules: Iterable[Rule], symbol: str, sessions: pd.DatetimeIndex
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rule in rules:
        held, observed = _rule_holds(history, rule, symbol, sessions)
        rows.append(
            {
                "feature": rule.feature,
                "op": rule.op,
                "value": rule.value,
                "window_sessions": rule.window_sessions,
                "observed": None if pd.isna(observed) else round(observed, 10),
                "passed": held,
            }
        )
    return rows


def evaluate_strategies(
    strategies: Sequence[Strategy],
    feature_history: pd.DataFrame,
    as_of: date | pd.Timestamp,
) -> list[Signal]:
    """Emit Signal rows for every strategy whose entry rules all hold at ``as_of``.

    ``feature_history`` has a (ts, symbol) MultiIndex and raw feature /
    detector-flag columns. Entry fires when **all** entry rules hold.
    Overlapping signals on one symbol dedup: strongest wins (ties break by
    strategy_id), losers are recorded SUPPRESSED.
    """
    ts_as_of = _to_ts(as_of)
    if feature_history.empty or not isinstance(feature_history.index, pd.MultiIndex):
        return []
    sessions = _sessions(feature_history, ts_as_of)
    if len(sessions) == 0:
        return []
    ts = _session_at(sessions, -1)
    symbols = sorted({str(s) for s in feature_history.index.get_level_values(1)})

    raw: list[Signal] = []
    for strategy in strategies:
        for symbol in symbols:
            if not _passes_universe(feature_history, strategy.universe_filter, symbol, sessions):
                continue
            results = [
                _rule_holds(feature_history, rule, symbol, sessions)
                for rule in strategy.entry_rules
            ]
            if not results or not all(held for held, _ in results):
                continue
            close = _value_at(feature_history, ts, symbol, "close")
            atr = _value_at(feature_history, ts, symbol, "atr_14")
            stop_hint = None
            if not pd.isna(close) and not pd.isna(atr):
                stop_hint = close - strategy.risk_params.stop_atr_multiple * atr
            raw.append(
                Signal(
                    strategy_id=strategy.strategy_id,
                    symbol=symbol,
                    as_of=ts_as_of.date(),
                    entry_ref_price=None if pd.isna(close) else float(close),
                    stop_hint=stop_hint,
                    horizon_days=strategy.holding_period_days,
                    strength=_strength(feature_history, strategy.entry_rules, symbol, ts),
                    rule_snapshot={
                        "rules": _snapshot(feature_history, strategy.entry_rules, symbol, sessions)
                    },
                    status=SignalStatus.ACTIVE,
                )
            )

    # Overlap dedup: winner by strength, losers SUPPRESSED with reason.
    out: list[Signal] = []
    by_symbol: dict[str, list[Signal]] = {}
    for sig in raw:
        by_symbol.setdefault(sig.symbol, []).append(sig)
    for symbol in sorted(by_symbol):
        group = sorted(by_symbol[symbol], key=lambda s: (-s.strength, s.strategy_id))
        winner = group[0]
        out.append(winner)
        for loser in group[1:]:
            out.append(
                loser.model_copy(
                    update={
                        "status": SignalStatus.SUPPRESSED,
                        "rule_snapshot": {
                            **loser.rule_snapshot,
                            "suppressed_by": winner.strategy_id,
                        },
                    }
                )
            )
    return out


def evaluate_exit(
    strategy: Strategy,
    feature_history: pd.DataFrame,
    symbol: str,
    as_of: date | pd.Timestamp,
) -> bool:
    """Exit fires when **any** exit rule holds (the risk engine's stop always wins)."""
    ts_as_of = _to_ts(as_of)
    sessions = _sessions(feature_history, ts_as_of)
    if len(sessions) == 0:
        return False
    return any(
        _rule_holds(feature_history, rule, symbol, sessions)[0] for rule in strategy.exit_rules
    )


# ---------------------------------------------------------------------------
# Initial strategy set — doc 05 section 3.1 exact defaults
# ---------------------------------------------------------------------------


def momentum_rs_v1() -> Strategy:
    return Strategy(
        strategy_id="momentum-rs-v1",
        universe_filter=UniverseFilter(tiers=(1,), min_dollar_volume=10_000_000.0),
        entry_rules=(
            Rule(feature="mom_12_1", op="in_top_quantile", value=0.2),
            Rule(feature="rs_20d_sector", op="gt", value=0.0),
            Rule(feature="dist_ma50", op="gt", value=0.0),  # price above 50d MA
        ),
        exit_rules=(
            Rule(feature="rs_20d_sector", op="lt", value=0.0, window_sessions=5),
            Rule(feature="mom_12_1", op="in_bottom_quantile", value=0.6),  # exits top 2 quintiles
        ),
        holding_period_days=40,
        risk_params=StrategyRiskParams(
            stop_atr_multiple=2.5, trailing_stop=True, max_concurrent_signals=10
        ),
        status="candidate",
    )


def breakout_v1() -> Strategy:
    return Strategy(
        strategy_id="breakout-v1",
        universe_filter=UniverseFilter(tiers=(1, 3)),
        entry_rules=(
            Rule(feature="range_break", op="eq", value=1.0),  # new 52w high detector flag
            Rule(feature="volume_spike", op="eq", value=1.0),  # volume >= 2x 60d median
            Rule(feature="base_sessions", op="gte", value=30.0),  # base below prior high
        ),
        exit_rules=(Rule(feature="close", op="crossed_below", value="breakout_level"),),
        holding_period_days=30,
        risk_params=StrategyRiskParams(
            stop_atr_multiple=2.0, trailing_stop=False, max_concurrent_signals=10
        ),
        status="candidate",
    )


def meanrev_v1() -> Strategy:
    return Strategy(
        strategy_id="meanrev-v1",
        universe_filter=UniverseFilter(
            tiers=(1,),
            extra_rules=(  # uptrend only: price > 200d MA, positive 12-1 momentum
                Rule(feature="dist_ma200", op="gt", value=0.0),
                Rule(feature="mom_12_1", op="gt", value=0.0),
            ),
        ),
        entry_rules=(
            Rule(feature="rsi_14", op="lt", value=30.0),
            Rule(feature="ret_5d_z", op="lt", value=-2.0),
        ),
        exit_rules=(
            Rule(feature="rsi_14", op="gt", value=55.0),
            Rule(feature="ret_5d_z", op="gt", value=0.0),
        ),
        holding_period_days=10,  # 10-session time stop
        risk_params=StrategyRiskParams(
            stop_atr_multiple=1.5, trailing_stop=False, max_concurrent_signals=10
        ),
        status="candidate",
    )


def earnings_react_v1() -> Strategy:
    return Strategy(
        strategy_id="earnings-react-v1",
        universe_filter=UniverseFilter(
            tiers=(1, 3),
            extra_rules=(Rule(feature="days_since_last_earnings", op="lte", value=3.0),),
        ),
        entry_rules=(
            Rule(feature="eps_surprise_pct", op="gt", value=5.0),
            Rule(feature="revenue_beat", op="eq", value=1.0),
            Rule(feature="guidance_positive", op="eq", value=1.0),
            Rule(feature="gap_held", op="eq", value=1.0),  # day-1 gap held, no full fade
        ),
        exit_rules=(Rule(feature="close", op="crossed_below", value="post_earnings_low"),),
        holding_period_days=20,
        risk_params=StrategyRiskParams(
            stop_atr_multiple=0.5, trailing_stop=False, max_concurrent_signals=10
        ),
        status="candidate",
    )


DEFAULT_STRATEGIES: dict[str, Strategy] = {
    s.strategy_id: s for s in (momentum_rs_v1(), breakout_v1(), meanrev_v1(), earnings_react_v1())
}
