"""Crypto in the replay runner (ADR-0013): 7-day/week bars, equity sessions.

v1 convention (runner module docstring): crypto trades AND marks on NYSE
sessions only. Weekend bars sit in the bars frame harmlessly, and the first
session after a weekend marks positions at THAT session's close — the whole
weekend move surfaces on Monday.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.replay.runner import ReplayConfig, ReplayRunner

END = date(2026, 6, 30)
HISTORY_SESSIONS = 280  # regime warmup (200) + slack
RUN_SESSIONS = 8

CRYPTO = "BTC/USD"
UNIVERSE = [CRYPTO]
EQUITIES = ["SPY", "TLT"]  # benchmark panel names (not tradeable universe)

BTC_BASE = 500.0
BTC_SLOPE = 3.0  # dollars per CALENDAR day — weekend moves are real


def nyse_sessions(end: date, count: int) -> list[date]:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    start = end - timedelta(days=int(count * 1.9) + 40)
    sessions = [
        cast(date, pd.Timestamp(s).date())
        for s in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    ]
    return sessions[-count:]


def btc_close(day: date, first: date) -> float:
    return BTC_BASE + BTC_SLOPE * (day - first).days


def make_bars(sessions: list[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sym in EQUITIES:
        base = 400.0 if sym == "SPY" else 90.0
        for i, day in enumerate(sessions):
            close = base * (1.0 + 0.0012 * i)
            rows.append(
                {
                    "symbol": sym,
                    "ts": datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC),
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000,
                    "source": "tiingo",
                }
            )
    # Crypto: a bar for EVERY calendar day (24/7 market), weekends included.
    first = sessions[0]
    day = first
    while day <= sessions[-1]:
        close = btc_close(day, first)
        rows.append(
            {
                "symbol": CRYPTO,
                "ts": datetime(day.year, day.month, day.day, 5, 0, tzinfo=UTC),
                "open": close - 1.0,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000,
                "source": "alpaca_crypto",
            }
        )
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def make_scores(sessions: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": CRYPTO,
                "as_of": day,
                "model_version": "test-v1",
                "expected_excess_return": 0.06,
                "downside_prob": 0.1,
                "vol_forecast": 0.6,
                "rank": None,
                "confidence": 0.6,
                "feature_coverage": 0.6,
            }
            for day in sessions
        ]
    )


class FakeGateway:
    def __init__(self, as_of: date, bars: pd.DataFrame, scores: pd.DataFrame) -> None:
        self.as_of = as_of
        self._bars = bars
        self._scores = scores

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]:
        return list(UNIVERSE)

    def bars(self, symbols: Any, lookback_days: int) -> pd.DataFrame:
        frame = self._bars
        frame = cast(pd.DataFrame, frame[pd.to_datetime(frame["ts"]).dt.date <= self.as_of])
        if symbols is not None:
            frame = cast(pd.DataFrame, frame[frame["symbol"].isin(list(symbols))])
        return frame.reset_index(drop=True)

    def feature_inputs(self, symbols: Any) -> Any:
        from dynamo.research.quant.catalog import FeatureInputs

        return FeatureInputs()

    def quant_scores(self, symbols: Any = None) -> pd.DataFrame:
        frame = cast(pd.DataFrame, self._scores[self._scores["as_of"] <= self.as_of])
        if symbols is not None:
            frame = cast(pd.DataFrame, frame[frame["symbol"].isin(list(symbols))])
        return frame.reset_index(drop=True)

    def macro_series(self, lookback_days: int) -> pd.DataFrame:
        return pd.DataFrame({"series_id": [], "ts": [], "value": [], "vintage_ts": []})

    def sectors(self, symbols: Any = None) -> dict[str, str]:
        return {CRYPTO: "crypto", "SPY": "etf", "TLT": "etf"}

    def regime_state(self) -> Any:
        return None


class FakeExperienceRepo:
    def __init__(self) -> None:
        self.inserted: list[DecisionExperience] = []

    def insert(self, exp: DecisionExperience) -> bool:
        self.inserted.append(exp)
        return True


class FakeRunRepo:
    def __init__(self) -> None:
        self.decisions: list[dict[str, Any]] = []

    def create_run(self, **kwargs: Any) -> None:
        pass

    def update_status(self, run_id: str, status: str) -> None:
        pass

    def insert_decisions(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.decisions.extend(rows)
        return len(rows)


@pytest.fixture(scope="module")
def result() -> tuple[Any, list[date]]:
    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    bars = make_bars(sessions)
    scores = make_scores(sessions)
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(
        start=window[0],
        end=window[-1],
        label="crypto-test",
        # Longer than the window: the position is held across weekends.
        horizon_sessions=30,
    )
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=FakeRunRepo(),
        config=config,
    )
    return runner.run(), sessions


def test_crypto_trades_on_equity_sessions_only(result: Any) -> None:
    res, sessions = result
    assert res.status == "completed"
    session_set = set(res.sessions)
    assert all(d.weekday() < 5 for d in session_set)  # NYSE sessions only
    btc_fills = [f for f in res.fills if f.symbol == CRYPTO]
    assert btc_fills, "the crypto pair should be bought through the full pipeline"
    assert all(f.session in session_set for f in btc_fills)
    # Fills land at that session's OPEN from the 7-day bars frame — weekend
    # bars never leak into fill pricing.
    first = sessions[0]
    for f in btc_fills:
        assert f.price == pytest.approx(btc_close(f.session, first) - 1.0)


def test_crypto_sleeve_cap_binds_in_replay(result: Any) -> None:
    res, _ = result
    btc_targets = [
        d["target_weight"] for d in res.decisions if d["symbol"] == CRYPTO
    ]
    assert btc_targets
    assert all(w <= 0.05 + 1e-9 for w in btc_targets)  # crypto_max_position
    assert any(w > 0.0 for w in btc_targets)


def test_weekend_moves_mark_at_the_monday_session(result: Any) -> None:
    """Equity at every session must equal cash + shares x THAT session's
    crypto close — in particular the first session after a weekend marks at
    its own close, so the full weekend move (3 calendar days of drift)
    appears in the Monday equity point."""
    res, sessions = result
    first = sessions[0]
    curve = dict(res.equity_curve)

    cash = 100_000.0
    shares = 0.0
    fills_by_session: dict[date, list[Any]] = {}
    for f in res.fills:
        fills_by_session.setdefault(f.session, []).append(f)
    held_over_weekend = False
    prev: date | None = None
    for day in res.sessions:
        for f in fills_by_session.get(day, ()):
            if f.side == "buy":
                cash -= f.qty * f.price + f.cost
                shares += f.qty
            else:
                cash += f.qty * f.price - f.cost
                shares -= f.qty
        expected = cash + shares * btc_close(day, first)
        assert curve[day] == pytest.approx(expected, abs=1.0), day
        if prev is not None and (day - prev).days >= 3 and shares > 0:
            # Crossing a weekend while holding: the pointwise assertion above
            # just proved the mark uses MONDAY's close (btc_close(day) sits
            # 3 calendar days of drift above Friday's), not a stale Friday
            # price — the weekend move surfaces here.
            held_over_weekend = True
        prev = day
    assert held_over_weekend, "fixture must hold the pair across a weekend"
