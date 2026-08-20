"""ReplayRunner over an in-memory fake gateway (plan Phase 4).

Covers: determinism (same config twice ⇒ identical outputs), the REAL
RiskEngine in the loop (a limit-violating candidate is vetoed), horizon
exits resolving into ``DecisionExperience`` rows with ``source='replay'``,
and the write firewall at the runner level (only replay sinks are written).
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import pandas as pd
import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.replay.runner import ReplayConfig, ReplayRunner

END = date(2026, 6, 30)
HISTORY_SESSIONS = 280  # regime warmup (200) + slack
RUN_SESSIONS = 8

UNIVERSE = ["AAA", "BBB", "CCC", "ILLQ", "LIQ"]
ALL_SYMBOLS = [*UNIVERSE, "SPY", "TLT"]
# eer ranks (desc): ILLQ 1.0, LIQ 0.8 -> both BUY; CCC bottom -> AVOID.
EER = {"ILLQ": 0.06, "LIQ": 0.05, "AAA": 0.01, "BBB": 0.005, "CCC": -0.01}
BASE = {
    "AAA": 90.0,
    "BBB": 80.0,
    "CCC": 70.0,
    "ILLQ": 50.0,
    "LIQ": 100.0,
    "SPY": 400.0,
    "TLT": 90.0,
}
VOLUME = {s: 1_000_000 for s in ALL_SYMBOLS}
VOLUME["ILLQ"] = 10  # ~ $500/day ADV -> real liquidity veto from the RiskEngine


def nyse_sessions(end: date, count: int) -> list[date]:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    start = end - timedelta(days=int(count * 1.9) + 40)
    sessions = [
        cast(date, pd.Timestamp(s).date())
        for s in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    ]
    return sessions[-count:]


def make_bars(sessions: list[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sym in ALL_SYMBOLS:
        base = BASE[sym]
        phase = sum(ord(c) for c in sym) % 7
        for i, day in enumerate(sessions):
            close = base * (1.0 + 0.0012 * i) + 0.4 * math.sin(i / 5.0 + phase)
            rows.append(
                {
                    "symbol": sym,
                    "ts": datetime(day.year, day.month, day.day, 20, 0, tzinfo=UTC),
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_close": close,
                    "volume": VOLUME[sym],
                    "source": "tiingo",
                }
            )
    return pd.DataFrame(rows)


def make_scores(sessions: list[date]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in sessions:
        for sym in UNIVERSE:
            rows.append(
                {
                    "symbol": sym,
                    "as_of": day,
                    "model_version": "test-v1",
                    "expected_excess_return": EER[sym],
                    "downside_prob": 0.1,
                    "vol_forecast": 0.2,
                    "rank": None,
                    "confidence": 0.6,
                    "feature_coverage": 1.0,
                }
            )
    return pd.DataFrame(rows)


class FakeGateway:
    """Duck-typed ReplayGateway over fixture frames, sliced to <= as_of."""

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

        return FeatureInputs()  # baseline path scores nothing (abstention)

    def quant_scores(self, symbols: Any = None) -> pd.DataFrame:
        frame = cast(pd.DataFrame, self._scores[self._scores["as_of"] <= self.as_of])
        if symbols is not None:
            frame = cast(pd.DataFrame, frame[frame["symbol"].isin(list(symbols))])
        return frame.reset_index(drop=True)

    def macro_series(self, lookback_days: int) -> pd.DataFrame:
        return pd.DataFrame({"series_id": [], "ts": [], "value": [], "vintage_ts": []})

    def sectors(self, symbols: Any = None) -> dict[str, str]:
        return {s: "tech" for s in ALL_SYMBOLS}

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
        self.runs: list[dict[str, Any]] = []
        self.statuses: list[tuple[str, str]] = []
        self.decisions: list[dict[str, Any]] = []
        self.equity_rows: list[dict[str, Any]] = []
        self.fill_rows: list[dict[str, Any]] = []
        self.equity_flushes: list[int] = []  # rows per insert_equity call

    def create_run(self, **kwargs: Any) -> None:
        self.runs.append(kwargs)

    def update_status(self, run_id: str, status: str) -> None:
        self.statuses.append((run_id, status))

    def insert_decisions(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.decisions.extend(rows)
        return len(rows)

    def insert_equity(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.equity_rows.extend(rows)
        self.equity_flushes.append(len(rows))
        return len(rows)

    def insert_fills(self, rows: Any) -> int:
        rows = [dict(r) for r in rows]
        self.fill_rows.extend(rows)
        return len(rows)


@pytest.fixture(scope="module")
def fixture_frames() -> tuple[list[date], pd.DataFrame, pd.DataFrame]:
    sessions = nyse_sessions(END, HISTORY_SESSIONS)
    return sessions, make_bars(sessions), make_scores(sessions)


def _run(fixture_frames: Any) -> tuple[Any, FakeExperienceRepo, FakeRunRepo]:
    sessions, bars, scores = fixture_frames
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(start=window[0], end=window[-1], label="test", horizon_sessions=3)
    exp_repo = FakeExperienceRepo()
    run_repo = FakeRunRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: FakeGateway(d, bars, scores),
        experience_repo=exp_repo,
        run_repo=run_repo,
        config=config,
    )
    return runner.run(), exp_repo, run_repo


@pytest.fixture(scope="module")
def result(fixture_frames: Any) -> tuple[Any, FakeExperienceRepo, FakeRunRepo]:
    return _run(fixture_frames)


def test_runner_is_quant_only_committee_goes_through_committee_replay() -> None:
    with pytest.raises(ValueError, match="quant/portfolio"):
        ReplayRunner(
            gateway_factory=lambda d: None,  # type: ignore[arg-type,return-value]
            experience_repo=FakeExperienceRepo(),
            run_repo=FakeRunRepo(),
            config=ReplayConfig(start=date(2026, 6, 1), end=date(2026, 6, 5), mode="committee"),
        )


def test_run_completes_and_uses_real_sessions(result: Any) -> None:
    res, _, run_repo = result
    assert res.status == "completed"
    assert len(res.sessions) == RUN_SESSIONS
    assert run_repo.statuses[-1] == (res.run_id, "completed")
    assert run_repo.runs[0]["mode"] == "quant"
    assert run_repo.runs[0]["budget_usd"] is None  # quant replay spends $0


def test_real_risk_engine_vetoes_illiquid_candidate(result: Any) -> None:
    """ILLQ has the TOP quant score but a ~$500/day ADV: the REAL RiskEngine
    emits a liquidity veto and the production constructor refuses the entry."""
    res, _, _ = result
    all_vetoes = {(s, r) for rec in res.session_records for s, r in rec.vetoes}
    assert ("ILLQ", "liquidity") in all_vetoes
    assert not any(f.symbol == "ILLQ" for f in res.fills)
    illq_weights = [d["target_weight"] for d in res.decisions if d["symbol"] == "ILLQ"]
    assert illq_weights and all(w == 0.0 for w in illq_weights)
    # The equally-scored liquid name IS bought through the same pipeline.
    assert any(f.symbol == "LIQ" and f.side == "buy" for f in res.fills)


def test_verdicts_never_degrade_and_new_risk_allowed(result: Any) -> None:
    res, _, _ = result
    assert all(not rec.degraded for rec in res.session_records)
    assert any(rec.new_risk_allowed for rec in res.session_records)


def test_horizon_exit_resolves_replay_experiences(result: Any) -> None:
    res, exp_repo, _ = result
    assert res.experiences, "horizon exits should resolve at least one experience"
    for exp in res.experiences:
        assert exp.source == "replay"
        assert exp.replay_run_id == res.run_id
        assert exp.resolved is True
        assert exp.id == f"exp:{res.run_id}:{exp.trade_id}"
        assert exp.pnl_pct is not None
    liq = [e for e in res.experiences if e.symbol == "LIQ"]
    assert liq and liq[0].action == "buy"
    assert liq[0].conviction == 0.8  # synthesized entry conviction
    assert liq[0].sector == "tech"
    # Everything the runner "persisted" went through the experience sink only.
    assert [e.id for e in exp_repo.inserted] == [e.id for e in res.experiences]


def test_decisions_recorded_with_fill_prices(result: Any) -> None:
    res, _, run_repo = result
    assert res.decisions == tuple(run_repo.decisions)
    liq_buys = [d for d in res.decisions if d["symbol"] == "LIQ" and d["target_weight"] > 0.0]
    assert liq_buys
    assert any(d["fill_price"] is not None for d in liq_buys)
    for d in res.decisions:
        assert d["run_id"] == res.run_id
        assert d["id"] == f"{res.run_id}:{d['as_of'].isoformat()}:{d['symbol']}"


def test_same_config_twice_is_bit_identical(fixture_frames: Any) -> None:
    res_a, exp_a, _ = _run(fixture_frames)
    res_b, exp_b, _ = _run(fixture_frames)
    assert res_a.run_id == res_b.run_id
    assert res_a.decisions == res_b.decisions
    assert res_a.equity_curve == res_b.equity_curve
    assert res_a.fills == res_b.fills
    assert [e.model_dump() for e in res_a.experiences] == [
        e.model_dump() for e in res_b.experiences
    ]
    assert [e.model_dump() for e in exp_a.inserted] == [e.model_dump() for e in exp_b.inserted]


def test_aligned_macro_handles_tz_aware_session_index() -> None:
    """Regression: bars ts is timestamptz (UTC-aware) while macro ts is a DATE
    column (naive). _aligned_macro must align them instead of raising
    ``TypeError: Cannot compare dtypes`` (2026-08 campaign dress-rehearsal)."""
    import numpy as np

    from dynamo.replay.runner import _aligned_macro

    macro = pd.DataFrame(
        {
            "series_id": ["BAMLH0A0HYM2"] * 3,
            "ts": pd.to_datetime(["2026-02-02", "2026-02-03", "2026-02-04"]),
            "vintage_ts": pd.to_datetime(["2026-02-03", "2026-02-04", "2026-02-05"]),
            "value": [3.0, 3.5, 4.5],
        }
    )
    aware_idx = pd.DatetimeIndex(
        pd.to_datetime(["2026-02-02", "2026-02-04", "2026-02-05"]), tz="UTC"
    )
    out = _aligned_macro(macro, "BAMLH0A0HYM2", aware_idx, default=4.0)
    assert np.allclose(out, [3.0, 4.5, 4.5])
    # naive index keeps working identically
    naive_idx = pd.DatetimeIndex(pd.to_datetime(["2026-02-02", "2026-02-04"]))
    assert np.allclose(_aligned_macro(macro, "BAMLH0A0HYM2", naive_idx, default=4.0), [3.0, 4.5])
