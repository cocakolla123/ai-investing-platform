"""Runtime leakage tripwires (backtest-campaign Phase 4) — plain unit tests.

Every gateway read path is fed a FUTURE-dated row through a stubbed repo /
engine / read layer and must raise :class:`ReplayLeakageError`: even if a
query's WHERE clause regresses, no future-dated data leaves the gateway.
These deliberately complement (never replace) the integration leakage suite
in ``test_gateway_leakage.py``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from dynamo.replay.gateway import HistoricalDataGateway, ReplayLeakageError

AS_OF = date(2026, 6, 15)
BEFORE = date(2026, 6, 10)
AFTER = date(2026, 6, 20)


def _dt(d: date, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


class _FakeConnection:
    def __init__(self, result: Any = None) -> None:
        self._result = result

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def execute(self, stmt: Any) -> Any:
        return self._result


class _FakeEngine:
    """Just enough Engine for the gateway constructor + connect()."""

    def __init__(self, result: Any = None) -> None:
        self._result = result

    def connect(self) -> _FakeConnection:
        return _FakeConnection(self._result)


def _gateway(result: Any = None) -> HistoricalDataGateway:
    return HistoricalDataGateway(_FakeEngine(result), AS_OF)  # type: ignore[arg-type]


def _patch_read_sql(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    monkeypatch.setattr(pd, "read_sql", lambda stmt, cx: frame)


# -- the _read_df funnel -----------------------------------------------------


def test_read_df_trips_on_future_ts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame(
            {"series_id": ["VIX"], "ts": [AFTER], "value": [1.0], "vintage_ts": [_dt(BEFORE)]}
        ),
    )
    with pytest.raises(ReplayLeakageError, match="ts"):
        _gateway().macro_series(90)


def test_read_df_trips_on_future_vintage(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame(
            {"series_id": ["VIX"], "ts": [BEFORE], "value": [1.0], "vintage_ts": [_dt(AFTER)]}
        ),
    )
    with pytest.raises(ReplayLeakageError, match="vintage_ts"):
        _gateway().macro_series(90)


def test_read_df_passes_on_knowable_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame(
            {"series_id": ["VIX"], "ts": [BEFORE], "value": [1.0], "vintage_ts": [_dt(BEFORE)]}
        ),
    )
    assert len(_gateway().macro_series(90)) == 1


def test_news_exempts_backfill_fetched_at_but_trips_on_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backfilled item (fetched TODAY, published in-window) is fine; a
    future published_at trips."""
    ok = pd.DataFrame(
        {
            "id": ["n-1"],
            "symbols": [["AAPL"]],
            "headline": ["h"],
            "summary": [""],
            "source": ["test"],
            "published_at": [_dt(BEFORE)],
            "url": [""],
            "fetched_at": [_dt(AFTER)],  # ingestion wall clock
        }
    )
    _patch_read_sql(monkeypatch, ok)
    assert _gateway().news(["AAPL"])  # no raise

    bad = ok.assign(published_at=[_dt(AFTER)])
    _patch_read_sql(monkeypatch, bad)
    with pytest.raises(ReplayLeakageError, match="published_at"):
        _gateway().news(["AAPL"])


def test_transcripts_trip_on_future_event_date(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "event_date": [AFTER],
                "source": ["defeatbeta"],
                "content": ["call text"],
            }
        ),
    )
    with pytest.raises(ReplayLeakageError, match="event_date"):
        _gateway().transcripts(["AAPL"])


# -- repo-delegating methods -------------------------------------------------


def test_bars_trip_on_future_ts() -> None:
    gw = _gateway()
    frame = pd.DataFrame({"symbol": ["AAPL"], "ts": [pd.Timestamp(_dt(AFTER))], "close": [1.0]})
    gw._repo = SimpleNamespace(bars_frame=lambda *a, **k: frame)  # type: ignore[assignment]
    with pytest.raises(ReplayLeakageError, match=r"bars\.ts"):
        gw.bars(["AAPL"], 30)


def test_quant_scores_trip_on_future_as_of() -> None:
    gw = _gateway()
    frame = pd.DataFrame({"symbol": ["AAPL"], "as_of": [AFTER], "expected_excess_return": [0.9]})
    gw._repo = SimpleNamespace(latest_quant_scores=lambda *a, **k: frame)  # type: ignore[assignment]
    with pytest.raises(ReplayLeakageError, match=r"quant_scores\.as_of"):
        gw.quant_scores(["AAPL"])


def test_verdicts_trip_on_future_as_of() -> None:
    gw = _gateway()
    frame = pd.DataFrame({"id": ["v-1"], "symbol": ["AAPL"], "as_of": [AFTER]})
    gw._repo = SimpleNamespace(latest_verdicts=lambda *a, **k: frame)  # type: ignore[assignment]
    with pytest.raises(ReplayLeakageError, match=r"verdicts\.as_of"):
        gw.verdicts(["AAPL"])


def test_macro_digest_trips_on_future_payload_as_of() -> None:
    gw = _gateway()
    gw._repo = SimpleNamespace(  # type: ignore[assignment]
        latest_macro_digest=lambda *a, **k: {"as_of": AFTER.isoformat(), "narrative": "x"}
    )
    with pytest.raises(ReplayLeakageError, match=r"macro_digest\.as_of"):
        gw.macro_digest()


def test_universe_trips_on_future_membership() -> None:
    from dynamo.domain.models.market import Tier, UniverseMembership

    gw = _gateway()
    member = UniverseMembership(symbol="NEW", tier=Tier.CORE, valid_from=AFTER, reason="seed")
    gw._repo = SimpleNamespace(universe_at=lambda *a, **k: [member])  # type: ignore[assignment]
    with pytest.raises(ReplayLeakageError, match=r"universe\.valid_from"):
        gw.universe()


def test_regime_state_trips_on_future_row() -> None:
    row = SimpleNamespace(
        _mapping={
            "as_of": AFTER,
            "regime": "neutral",
            "composite_score": 0.0,
            "inputs": {},
            "sessions_in_state": 1,
        }
    )
    result = SimpleNamespace(first=lambda: row)
    gw = _gateway(result)
    with pytest.raises(ReplayLeakageError, match=r"regime_state\.as_of"):
        gw.regime_state()


def test_journal_search_trips_on_future_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    from dynamo.domain.models.learning import JournalEntry, JournalKind

    future_entry = JournalEntry(
        id="j-future",
        kind=JournalKind.THESIS,
        symbols=("AAPL",),
        text="future thesis",
        created_at=_dt(AFTER),
    )

    class _Repo:
        def __init__(self, engine: Any) -> None: ...
        def search(self, **kwargs: Any) -> list[tuple[Any, float]]:
            return [(future_entry, 0.9)]

    monkeypatch.setattr("dynamo.replay.gateway.JournalRepository", _Repo)
    with pytest.raises(ReplayLeakageError, match=r"journal_search\.created_at"):
        _gateway().journal_search(text_query="thesis")


def test_similar_experiences_trip_on_same_day_or_future_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dynamo.domain.models.learning import (
        DecisionExperience,
        DecisionQuality,
        ForecastQuality,
        OutcomeQuality,
    )

    def exp(day: date) -> DecisionExperience:
        return DecisionExperience.model_validate(
            {
                "id": f"exp-{day.isoformat()}",
                "decision_context_id": "dc",
                "trade_id": "t",
                "source": "live",
                "created_at": _dt(day),
                "symbol": "AAPL",
                "as_of": day,
                "resolved": True,
                "pnl_pct": 1.0,
                "forecast_quality": ForecastQuality.ACCURATE,
                "decision_quality": DecisionQuality.SOUND,
                "outcome_quality": OutcomeQuality.WIN,
                "label": "win",
            }
        )

    class _Repo:
        rows: list[Any] = []

        def __init__(self, engine: Any) -> None: ...
        def similar(self, **kwargs: Any) -> list[Any]:
            return list(_Repo.rows)

    monkeypatch.setattr("dynamo.replay.gateway.ExperienceRepository", _Repo)
    # The contract is STRICTLY before as_of: a same-day row already trips.
    _Repo.rows = [exp(AS_OF)]
    with pytest.raises(ReplayLeakageError, match="strictly before"):
        _gateway().similar_experiences(symbol="AAPL")
    _Repo.rows = [exp(AFTER)]
    with pytest.raises(ReplayLeakageError, match="strictly before"):
        _gateway().similar_experiences(symbol="AAPL")
    _Repo.rows = [exp(BEFORE)]
    assert [e.id for e in _gateway().similar_experiences(symbol="AAPL")] == [
        f"exp-{BEFORE.isoformat()}"
    ]


# -- surveillance reads (wide-campaign discovery layer) ----------------------


def test_surveillance_universe_trips_on_a_future_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``valid_from`` is a knowledge time: a tier-2 membership that begins
    after ``as_of`` did not exist yet, and reading it would hand discovery a
    survivorship-tainted universe. This is the tripwire that fires if the
    WHERE clause on the membership query is ever dropped."""
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame({"symbol": ["NEW"], "valid_from": [AFTER], "valid_to": [None]}),
    )
    with pytest.raises(ReplayLeakageError, match="valid_from"):
        _gateway().surveillance_universe()


def test_surveillance_universe_allows_an_open_ended_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``valid_to`` is an END bound, not a knowledge time — a membership that
    runs until 2030 is perfectly knowable today, so it must not trip."""
    _patch_read_sql(
        monkeypatch,
        pd.DataFrame(
            {
                "symbol": ["OLD", "OLD"],
                "valid_from": [BEFORE, BEFORE],
                "valid_to": [None, date(2030, 1, 1)],
            }
        ),
    )
    assert _gateway().surveillance_universe() == ["OLD"]


def _bar_frame(ts: date, fetched: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "ts": [pd.Timestamp(_dt(ts))],
            "source": ["tiingo"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [10],
            "adj_close": [1.0],
            "fetched_at": [_dt(fetched)],
        }
    )


def test_surveillance_bars_trip_on_a_future_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_read_sql(monkeypatch, _bar_frame(AFTER, BEFORE))
    with pytest.raises(ReplayLeakageError, match="ts"):
        _gateway().surveillance_bars(["AAPL"], 30)


def test_surveillance_bars_exempt_the_backfill_ingestion_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A historical backfill ingests old bars TODAY, so ``fetched_at``
    legitimately post-dates ``as_of``. ``ts`` carries the contract."""
    _patch_read_sql(monkeypatch, _bar_frame(BEFORE, AFTER))
    assert len(_gateway().surveillance_bars(["AAPL"], 30)) == 1


def test_surveillance_bars_skip_the_query_for_an_empty_symbol_list() -> None:
    """The wide-universe read is ``symbol = ANY(:array)``; an empty array must
    short-circuit rather than degenerate into a whole-table scan."""
    frame = _gateway().surveillance_bars([], 30)
    assert frame.empty
    assert "close" in frame.columns


def test_the_gateway_exposes_no_surveillance_flag_reader() -> None:
    """``surveillance_flags`` is a LIVE table carrying triage verdicts formed
    in production. Replay discovery computes its own flags from bars and
    writes them to ``replay_flags``; a reader here would let a replay inherit
    — and act on — verdicts written after its ``as_of``."""
    readers = [
        name
        for name in dir(HistoricalDataGateway)
        if "flag" in name.lower() and not name.startswith("_")
    ]
    assert readers == []


# -- documented exemptions ---------------------------------------------------


def test_earnings_event_date_exemption_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exempt seam used by feature_inputs: a future event_date passes
    WITH the exemption (future events are legitimately knowable) and trips
    WITHOUT it."""
    frame = pd.DataFrame(
        {"symbol": ["AAPL"], "event_date": [AS_OF + timedelta(days=3)], "fetched_at": [_dt(BEFORE)]}
    )
    _patch_read_sql(monkeypatch, frame)
    gw = _gateway()
    assert len(gw._read_df(None, exempt=("event_date",))) == 1
    with pytest.raises(ReplayLeakageError, match="event_date"):
        gw._read_df(None)
