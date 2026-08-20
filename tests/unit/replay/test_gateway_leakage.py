"""Gateway leakage suite — THE point-in-time contract (plan Phase 4).

For EVERY ``HistoricalDataGateway`` read method: seed one row knowable at the
gateway's ``as_of`` and one row from AFTER it, then assert the future row is
invisible. Integration-marked (real Postgres + pgvector), following the
repo's INTEGRATION_DB_URL pattern.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest

INTEGRATION_URL = os.environ.get("INTEGRATION_DB_URL", "")
AS_OF = date(2026, 6, 15)
BEFORE = date(2026, 6, 10)
AFTER = date(2026, 6, 20)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not INTEGRATION_URL, reason="INTEGRATION_DB_URL not set"),
]


def _dt(d: date, hour: int = 12) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=UTC)


@pytest.fixture()
def engine() -> Iterator[object]:
    from sqlalchemy import create_engine, text

    from dynamo.persistence.schema import metadata

    eng = create_engine(INTEGRATION_URL, future=True)
    with eng.begin() as cx:
        cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    metadata.create_all(eng)
    yield eng
    metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def gateway(engine):  # type: ignore[no-untyped-def]
    from dynamo.replay.gateway import HistoricalDataGateway

    return HistoricalDataGateway(engine, AS_OF)  # type: ignore[arg-type]


def _repo(engine):  # type: ignore[no-untyped-def]
    from dynamo.persistence.repositories.market import MarketDataRepository

    return MarketDataRepository(engine)


def test_universe_hides_future_memberships(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import Tier, UniverseMembership

    _repo(engine).upsert_universe_membership(
        [
            UniverseMembership(
                symbol="OLD", tier=Tier.CORE, valid_from=BEFORE, reason="seed"
            ),
            UniverseMembership(
                symbol="NEW", tier=Tier.CORE, valid_from=AFTER, reason="seed"
            ),
            UniverseMembership(
                symbol="GONE",
                tier=Tier.CORE,
                valid_from=BEFORE,
                valid_to=BEFORE,
                reason="seed",
            ),
        ]
    )
    assert gateway.universe() == ["OLD"]
    assert gateway.universe(tiers=(1,)) == ["OLD"]


def test_bars_hide_future_sessions(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import Bar

    def bar(d: date) -> Bar:
        return Bar(
            symbol="AAPL",
            ts=_dt(d, 20),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
            source="tiingo",
        )

    _repo(engine).upsert_bars([bar(BEFORE), bar(AS_OF), bar(AFTER)])
    frame = gateway.bars(["AAPL"], lookback_days=30)
    dates = sorted(t.date() for t in frame["ts"])
    assert dates == [BEFORE, AS_OF]


def test_feature_inputs_bound_every_frame(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import (
        EarningsEvent,
        InsiderTx,
        Instrument,
        MacroPoint,
        NewsItem,
        XbrlFact,
    )
    from dynamo.persistence import schema

    repo = _repo(engine)
    repo.upsert_instruments(
        [Instrument(symbol="AAPL", name="Apple", sector="tech", cik="0000320193")]
    )
    repo.upsert_xbrl_facts(
        [
            XbrlFact(
                cik="0000320193", tag="Revenues", period_end=BEFORE,
                value=1.0, unit="USD", filed_at=_dt(BEFORE),
            ),
            XbrlFact(
                cik="0000320193", tag="Revenues", period_end=BEFORE,
                value=2.0, unit="USD", filed_at=_dt(AFTER),  # future filing
            ),
        ]
    )
    repo.upsert_insider_tx(
        [
            InsiderTx(
                accession_no="a-1", symbol="AAPL", insider_name="x", tx_type="P",
                role="officer", shares=1.0, price=1.0, filed_at=_dt(BEFORE),
            ),
            InsiderTx(
                accession_no="a-2", symbol="AAPL", insider_name="x", tx_type="P",
                role="officer", shares=1.0, price=1.0, filed_at=_dt(AFTER),
            ),
        ]
    )
    repo.upsert_macro(
        [
            MacroPoint(series_id="VIX", ts=BEFORE, value=15.0, vintage_ts=_dt(BEFORE)),
            # Observation ts after as_of.
            MacroPoint(series_id="VIX", ts=AFTER, value=99.0, vintage_ts=_dt(AFTER)),
            # Observation BEFORE as_of but published (vintaged) after: invisible.
            MacroPoint(series_id="VIX", ts=BEFORE, value=42.0, vintage_ts=_dt(AFTER)),
        ]
    )
    repo.upsert_earnings(
        [
            EarningsEvent(symbol="AAPL", event_date=AS_OF + timedelta(days=3)),
            EarningsEvent(symbol="MSFT", event_date=AS_OF + timedelta(days=3)),
        ],
        fetched_at=_dt(BEFORE),
    )
    # Re-fetch of the MSFT row AFTER as_of: knowledge-time excludes it.
    repo.upsert_earnings(
        [EarningsEvent(symbol="MSFT", event_date=AS_OF + timedelta(days=3))],
        fetched_at=_dt(AFTER),
    )
    # Estimate vintages (migration 0019): the knowledge axis is fetched_at —
    # a revision fetched after as_of is invisible, earlier vintages are not.
    event_date = AS_OF + timedelta(days=3)
    repo.append_estimate_vintages(
        [EarningsEvent(symbol="AAPL", event_date=event_date, eps_est=1.10)],
        fetched_at=_dt(BEFORE),
    )
    repo.append_estimate_vintages(
        [EarningsEvent(symbol="AAPL", event_date=event_date, eps_est=9.99)],
        fetched_at=_dt(AFTER),
    )

    # News: the attention features read published_at, exploded per symbol.
    repo.upsert_news(
        [
            NewsItem(
                id="n-1", symbols=("AAPL",), headline="early", source="test",
                published_at=_dt(BEFORE),
            ),
            NewsItem(
                id="n-2", symbols=("AAPL",), headline="future", source="test",
                published_at=_dt(AFTER),
            ),
        ],
        fetched_at=_dt(BEFORE),
    )
    # Short interest: the knowledge axis is publication_date, NOT the
    # settlement date. The second cycle settled BEFORE as_of and is still
    # invisible, because FINRA had not disseminated it yet — the whole point
    # of storing both dates (dynamo.data.finra).
    repo.upsert_rows(
        schema.short_interest,
        [
            {
                "symbol": "AAPL", "settlement_date": BEFORE - timedelta(days=15),
                "publication_date": BEFORE, "shares_short": 1_000.0,
            },
            {
                "symbol": "AAPL", "settlement_date": BEFORE,
                "publication_date": AFTER, "shares_short": 9_999.0,
            },
        ],
    )

    inputs = gateway.feature_inputs(["AAPL", "MSFT"])
    assert list(inputs.fundamentals["value"]) == [1.0]
    assert list(inputs.insider["accession_no"]) == ["a-1"]
    assert list(inputs.macro["value"]) == [15.0]
    assert list(inputs.earnings_calendar["symbol"]) == ["AAPL"]
    assert list(inputs.estimates["eps_est"]) == [1.10]  # PIT: 9.99 invisible
    assert list(inputs.news["symbol"]) == ["AAPL"]
    assert len(inputs.news) == 1  # the AFTER story is invisible
    # A cycle that settled before as_of but publishes after it stays hidden.
    assert list(inputs.short_interest["shares_short"]) == [1_000.0]


def test_fundamentals_read_hides_future_filings(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """The dossier's fundamentals read (longer window, symbol-filtered) is
    bounded exactly like the feature one: a fact FILED after as_of — a
    restatement of an old period — must be invisible."""
    from dynamo.domain.models.market import Instrument, XbrlFact

    repo = _repo(engine)
    repo.upsert_instruments(
        [
            Instrument(symbol="AAPL", name="Apple", sector="tech", cik="0000320193"),
            Instrument(symbol="MSFT", name="Microsoft", sector="tech", cik="0000789019"),
        ]
    )
    old_period = AS_OF - timedelta(days=1200)  # inside the 1500-day window
    repo.upsert_xbrl_facts(
        [
            XbrlFact(
                cik="0000320193", tag="Revenues", period_end=old_period,
                value=1.0, unit="USD", filed_at=_dt(BEFORE),
            ),
            XbrlFact(
                cik="0000320193", tag="Revenues", period_end=old_period,
                value=2.0, unit="USD", filed_at=_dt(AFTER),  # future restatement
            ),
            XbrlFact(
                cik="0000789019", tag="Revenues", period_end=BEFORE,
                value=3.0, unit="USD", filed_at=_dt(BEFORE),
            ),
        ]
    )

    frame = gateway.fundamentals(["AAPL"])
    assert list(frame["value"]) == [1.0]  # future filing invisible
    assert list(frame["symbol"]) == ["AAPL"]  # symbol filter honored
    assert list(frame.columns) == ["symbol", "tag", "period_end", "value", "filed_at"]

    both = gateway.fundamentals(["AAPL", "MSFT"])
    assert sorted(both["value"]) == [1.0, 3.0]
    assert gateway.fundamentals(["NOSUCH"]).empty


def test_quant_scores_hide_future_rows(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence import schema

    repo = _repo(engine)
    repo.upsert_rows(
        schema.quant_scores,
        [
            {
                "symbol": "AAPL", "as_of": d, "model_version": "m",
                "expected_excess_return": v, "confidence": 0.5, "feature_coverage": 1.0,
            }
            for d, v in ((BEFORE, 0.01), (AFTER, 0.99))
        ],
    )
    frame = gateway.quant_scores(["AAPL"])
    assert len(frame) == 1
    assert float(frame.iloc[0]["expected_excess_return"]) == 0.01
    # And the repository default (as_of=None) still sees the latest row.
    latest = repo.latest_quant_scores(["AAPL"])
    assert float(latest.iloc[0]["expected_excess_return"]) == 0.99


def test_verdicts_hide_future_rows(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence import schema

    repo = _repo(engine)
    repo.upsert_rows(
        schema.committee_verdicts,
        [
            {
                "id": f"v-{d.isoformat()}", "symbol": "AAPL", "as_of": d,
                "action": "buy", "conviction": c, "disagreement_score": 0.0,
                "payload": {"id": f"v-{d.isoformat()}"},
            }
            for d, c in ((BEFORE, 0.6), (AFTER, 0.9))
        ],
    )
    frame = gateway.verdicts(["AAPL"])
    assert len(frame) == 1
    assert str(frame.iloc[0]["id"]) == f"v-{BEFORE.isoformat()}"
    # Repository default (as_of=None) keeps live behavior: latest wins.
    assert str(repo.latest_verdicts(["AAPL"]).iloc[0]["id"]) == f"v-{AFTER.isoformat()}"


def test_macro_digest_hides_future_rows(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence import schema

    repo = _repo(engine)
    repo.upsert_rows(
        schema.macro_digests,
        [
            {"as_of": BEFORE, "payload": {"narrative": "knowable"}},
            {"as_of": AFTER, "payload": {"narrative": "future"}},
        ],
    )
    digest = gateway.macro_digest()
    assert digest is not None and digest["narrative"] == "knowable"
    # Live default reads the latest.
    live = repo.latest_macro_digest()
    assert live is not None and live["narrative"] == "future"


def test_macro_series_hides_future_ts_and_vintages(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import MacroPoint

    _repo(engine).upsert_macro(
        [
            MacroPoint(series_id="DFF", ts=BEFORE, value=5.0, vintage_ts=_dt(BEFORE)),
            MacroPoint(series_id="DFF", ts=AFTER, value=6.0, vintage_ts=_dt(AFTER)),
            MacroPoint(series_id="DFF", ts=BEFORE, value=7.0, vintage_ts=_dt(AFTER)),
        ]
    )
    frame = gateway.macro_series(90)
    assert list(frame["value"]) == [5.0]


def test_regime_state_hides_future_rows(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.persistence import schema

    _repo(engine).upsert_rows(
        schema.regime_states,
        [
            {
                "as_of": d, "regime": r, "composite_score": 0.0,
                "inputs": {}, "sessions_in_state": 1,
            }
            for d, r in ((BEFORE, "neutral"), (AFTER, "crisis"))
        ],
    )
    state = gateway.regime_state()
    assert state is not None
    assert state.as_of == BEFORE
    assert state.regime.value == "neutral"


def test_journal_search_hides_future_entries(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.learning import JournalEntry, JournalKind
    from dynamo.persistence.repositories.learning import JournalRepository

    repo = JournalRepository(engine)
    for name, day in (("past", BEFORE), ("future", AFTER)):
        repo.insert(
            JournalEntry(
                id=f"j-{name}",
                kind=JournalKind.THESIS,
                symbols=("AAPL",),
                text=f"{name} thesis for AAPL",
                created_at=_dt(day),
            ),
            embedding=[0.1] * 384,
            embedding_version="test",
        )
    hits = gateway.journal_search(embedding=[0.1] * 384, symbols=["AAPL"], top_k=5)
    assert [e.id for e, _ in hits] == ["j-past"]
    text_hits = gateway.journal_search(text_query="thesis", symbols=["AAPL"], top_k=5)
    assert [e.id for e, _ in text_hits] == ["j-past"]


def test_similar_experiences_hides_future_and_replay_rows(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.learning import (
        DecisionExperience,
        DecisionQuality,
        ForecastQuality,
        OutcomeQuality,
    )
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)

    def exp(i: int, day: date, source: str = "live") -> DecisionExperience:
        return DecisionExperience.model_validate(
            {
                "id": f"exp-{i}",
                "decision_context_id": f"dc-{i}",
                "trade_id": f"t-{i}",
                "source": source,
                "replay_run_id": None if source == "live" else "run-x",
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

    assert repo.insert(exp(1, BEFORE))
    assert repo.insert(exp(2, AFTER))  # future live row: invisible
    assert repo.insert(exp(3, BEFORE, source="replay"))  # replay row: invisible
    hits = gateway.similar_experiences(symbol="AAPL", limit=5)
    assert [e.id for e in hits] == ["exp-1"]


def test_sectors_reads_instruments_only(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """``sectors`` has no time axis (documented fidelity caveat: instruments
    are not vintaged) — assert it is scoped to the requested symbols."""
    from dynamo.domain.models.market import Instrument

    _repo(engine).upsert_instruments(
        [
            Instrument(symbol="AAPL", name="Apple", sector="tech"),
            Instrument(symbol="XOM", name="Exxon", sector="energy"),
        ]
    )
    assert gateway.sectors(["AAPL"]) == {"AAPL": "tech"}


def test_news_hides_future_items(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import NewsItem

    _repo(engine).upsert_news(
        [
            NewsItem(
                id="n-before",
                symbols=("AAPL",),
                headline="knowable headline",
                source="test",
                published_at=_dt(BEFORE),
            ),
            NewsItem(
                id="n-after",
                symbols=("AAPL",),
                headline="FUTURE headline",
                source="test",
                published_at=_dt(AFTER),
            ),
        ]
    )
    by_symbol = gateway.news(["AAPL"], lookback_days=30)
    assert [n.id for n in by_symbol["AAPL"]] == ["n-before"]


def test_instruments_scoped_to_requested_symbols(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """``instruments`` shares the sectors caveat (not vintaged) — assert
    symbol scoping, the only axis it has."""
    from dynamo.domain.models.market import Instrument

    _repo(engine).upsert_instruments(
        [
            Instrument(symbol="AAPL", name="Apple", sector="tech"),
            Instrument(symbol="XOM", name="Exxon", sector="energy"),
        ]
    )
    out = gateway.instruments(["AAPL"])
    assert set(out) == {"AAPL"}
    assert out["AAPL"].sector == "tech"


def test_gateway_has_no_write_methods() -> None:
    """The gateway is read-only by construction: no public method name that
    suggests a write path may ever appear on it."""
    from dynamo.replay.gateway import HistoricalDataGateway

    banned = ("insert", "upsert", "update", "delete", "write", "add", "create")
    public = [n for n in dir(HistoricalDataGateway) if not n.startswith("_")]
    offenders = [n for n in public if any(n.startswith(b) for b in banned)]
    assert offenders == []


def test_transcripts_hide_future_calls(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """Phase 4: gateway.transcripts mirrors the live 30-day dossier window
    and hides calls after as_of (and stale ones before the window)."""
    from dynamo.persistence import schema

    _repo(engine).upsert_rows(
        schema.transcripts,
        [
            {
                "symbol": "AAPL", "event_date": BEFORE, "source": "defeatbeta",
                "period": "Q2", "content": "knowable call", "speakers": [],
                "fetched_at": _dt(AFTER),  # backfilled TODAY — still visible
            },
            {
                "symbol": "AAPL", "event_date": AFTER, "source": "defeatbeta",
                "period": "Q3", "content": "FUTURE call", "speakers": [],
                "fetched_at": _dt(AFTER),
            },
            {
                "symbol": "AAPL", "event_date": AS_OF - timedelta(days=45),
                "source": "defeatbeta", "period": "Q1", "content": "stale call",
                "speakers": [], "fetched_at": _dt(BEFORE),
            },
        ],
    )
    frame = gateway.transcripts(["AAPL"])
    assert list(frame["content"]) == ["knowable call"]


def test_similar_experiences_within_run_learning_scoped_to_one_run(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """Phase 4 within-run learning at the SQL layer: include_replay_run_id
    admits exactly that run's resolved replay rows (still strictly before
    as_of); other runs' rows and future rows of the included run stay out."""
    from dynamo.domain.models.learning import (
        DecisionExperience,
        DecisionQuality,
        ForecastQuality,
        OutcomeQuality,
    )
    from dynamo.persistence.repositories.experience import ExperienceRepository

    repo = ExperienceRepository(engine)

    def exp(i: int, day: date, source: str, run_id: str | None) -> DecisionExperience:
        return DecisionExperience.model_validate(
            {
                "id": f"wrl-{i}",
                "decision_context_id": f"wrl-dc-{i}",
                "trade_id": f"wrl-t-{i}",
                "source": source,
                "replay_run_id": run_id,
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

    assert repo.insert(exp(1, BEFORE, "live", None))
    assert repo.insert(exp(2, BEFORE, "replay", "run-this"))
    assert repo.insert(exp(3, BEFORE, "replay", "run-other"))
    assert repo.insert(exp(4, AFTER, "replay", "run-this"))  # future: never
    # Default: live-only firewall unchanged.
    assert [e.id for e in gateway.similar_experiences(symbol="AAPL", limit=5)] == ["wrl-1"]
    # Within-run learning: live + THIS run's past rows, nothing else.
    hits = gateway.similar_experiences(
        symbol="AAPL", limit=5, include_replay_run_id="run-this"
    )
    assert sorted(e.id for e in hits) == ["wrl-1", "wrl-2"]


# --------------------------------------------------------------------------
# surveillance reads (wide-campaign discovery layer)
# --------------------------------------------------------------------------
#
# Three seeded rows, one per way the discovery layer could see the future:
# a tier-2 BAR dated as_of + 1, a MEMBERSHIP starting at as_of + 1, and a
# live SURVEILLANCE FLAG written after as_of. All three must be invisible,
# and the third must be invisible because there is no reader at all — a
# replay that inherited production's triage verdicts would be scoring
# 2026-03 against a decision made in 2026-08.


def test_surveillance_universe_hides_future_tier2_memberships(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import Tier, UniverseMembership

    _repo(engine).upsert_universe_membership(
        [
            UniverseMembership(
                symbol="WATCHED", tier=Tier.SURVEILLANCE, valid_from=BEFORE, reason="seed"
            ),
            UniverseMembership(
                symbol="TOMORROW", tier=Tier.SURVEILLANCE, valid_from=AFTER, reason="seed"
            ),
            UniverseMembership(
                symbol="DROPPED",
                tier=Tier.SURVEILLANCE,
                valid_from=BEFORE,
                valid_to=BEFORE,
                reason="seed",
            ),
            # The traded core is a DIFFERENT question: surveillance_universe
            # must not quietly return it, or "what discovery promoted" stops
            # being measurable.
            UniverseMembership(
                symbol="CORE", tier=Tier.CORE, valid_from=BEFORE, reason="seed"
            ),
        ]
    )
    assert gateway.surveillance_universe() == ["WATCHED"]
    assert "CORE" in gateway.universe()


def test_surveillance_bars_hide_future_sessions(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    from dynamo.domain.models.market import Bar

    def bar(d: date) -> Bar:
        return Bar(
            symbol="WATCHED",
            ts=_dt(d, 20),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
            source="tiingo",
        )

    _repo(engine).upsert_bars([bar(BEFORE), bar(AS_OF), bar(AFTER)])
    frame = gateway.surveillance_bars(["WATCHED"], lookback_days=30)
    assert sorted(t.date() for t in frame["ts"]) == [BEFORE, AS_OF]


def test_surveillance_flags_written_after_as_of_are_unreachable(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """Not "filtered out" — UNREACHABLE. The gateway offers no reader for the
    live table, so a triage verdict production formed after ``as_of`` cannot
    enter a replay by any route, and the discovery layer computes its own
    flags from bars instead."""
    from sqlalchemy import func, select

    from dynamo.persistence import schema
    from dynamo.replay.gateway import HistoricalDataGateway

    with engine.begin() as cx:
        cx.execute(
            schema.surveillance_flags.insert(),
            [
                {
                    "symbol": "WATCHED",
                    "detector": "price_zscore",
                    "severity": 0.9,
                    "details": {},
                    "ts": _dt(AFTER),
                    "triage_verdict": "material",
                    "triaged_at": _dt(AFTER),
                }
            ],
        )
    with engine.connect() as cx:
        seeded = int(
            cx.execute(
                select(func.count()).select_from(schema.surveillance_flags)
            ).scalar()
            or 0
        )
    assert seeded == 1, "the row must really be there for this test to mean anything"
    assert not [
        name
        for name in dir(HistoricalDataGateway)
        if "flag" in name.lower() and not name.startswith("_")
    ]
    assert not any("flag" in str(v).lower() for v in gateway.universe())


def test_dropping_the_where_clause_raises(engine, gateway) -> None:  # type: ignore[no-untyped-def]
    """The runtime tripwire behind the two reads above: both go through
    ``_read_df``, so an unbounded query raises rather than returning future
    rows. This is what makes a WHERE-clause regression loud."""
    from sqlalchemy import select

    from dynamo.domain.models.market import Bar, Tier, UniverseMembership
    from dynamo.persistence import schema
    from dynamo.replay.gateway import ReplayLeakageError

    repo = _repo(engine)
    repo.upsert_universe_membership(
        [UniverseMembership(symbol="TOMORROW", tier=Tier.SURVEILLANCE, valid_from=AFTER, reason="s")]
    )
    repo.upsert_bars(
        [
            Bar(
                symbol="TOMORROW", ts=_dt(AFTER, 20), open=1.0, high=1.0, low=1.0,
                close=1.0, volume=1, source="tiingo",
            )
        ]
    )
    m, b = schema.universe_membership, schema.bars_1d
    with pytest.raises(ReplayLeakageError, match="valid_from"):
        gateway._read_df(select(m.c.symbol, m.c.valid_from), exempt=("valid_to",))
    with pytest.raises(ReplayLeakageError, match="ts"):
        gateway._read_df(select(b.c.symbol, b.c.ts), exempt=("fetched_at",))
