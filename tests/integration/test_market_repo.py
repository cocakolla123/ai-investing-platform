"""MarketDataRepository integration tests (idempotent upserts, PIT reads).

Skipped unless ``INTEGRATION_DB_URL`` points at a Postgres instance
(e.g. the dockerized TimescaleDB from docker-compose.yml).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import Engine, create_engine, select

from dynamo.domain.models.market import Bar, Tier, UniverseMembership
from dynamo.persistence import schema
from dynamo.persistence.repositories.market import MarketDataRepository

pytestmark = pytest.mark.integration

TABLES = [
    schema.bars_1d,
    schema.universe_membership,
    schema.quarantine,
    schema.alerts,
    schema.quant_scores,
    schema.instruments,
    schema.xbrl_facts,
    schema.transcripts,
]


@pytest.fixture()
def engine(integration_db_url: str) -> Iterator[Engine]:
    eng = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(eng, tables=TABLES)
    yield eng
    with eng.begin() as conn:
        for table in TABLES:
            conn.execute(table.delete())
    eng.dispose()


@pytest.fixture()
def repo(engine: Engine) -> MarketDataRepository:
    return MarketDataRepository(engine)


def bar(symbol: str, day: int, source: str, close: float, adj: float | None) -> Bar:
    return Bar(
        symbol=symbol,
        ts=datetime(2026, 7, day, tzinfo=UTC),
        open=close, high=close, low=close, close=close,
        volume=1000,
        adj_close=adj,
        source=source,
    )


def test_upsert_bars_is_idempotent(repo: MarketDataRepository, engine: Engine) -> None:
    bars = [bar("ACME", 30, "alpaca", 10.0, None), bar("ACME", 31, "alpaca", 11.0, None)]
    repo.upsert_bars(bars)
    repo.upsert_bars(bars)  # re-poll must not duplicate (doc 01 §6)
    with engine.connect() as conn:
        count = len(conn.execute(select(schema.bars_1d)).all())
    assert count == 2

    # Same natural key with a corrected close → updated, not duplicated.
    repo.upsert_bars([bar("ACME", 31, "alpaca", 11.5, None)])
    with engine.connect() as conn:
        rows = conn.execute(select(schema.bars_1d).order_by(schema.bars_1d.c.ts)).all()
    assert len(rows) == 2
    assert rows[-1].close == 11.5


def test_bars_frame_prefers_tiingo_adjusted(repo: MarketDataRepository) -> None:
    repo.upsert_bars(
        [
            bar("ACME", 31, "alpaca", 100.0, None),
            bar("ACME", 31, "tiingo", 100.05, 50.0),  # post-split adjusted
            bar("ONLYALP", 31, "alpaca", 20.0, None),
        ]
    )
    frame = repo.bars_frame(["ACME", "ONLYALP"], date(2026, 7, 30), date(2026, 7, 31))
    by_symbol = {r["symbol"]: r for _, r in frame.iterrows()}
    assert by_symbol["ACME"]["source"] == "tiingo"
    assert by_symbol["ACME"]["close"] == 50.0  # canonical = tiingo adj_close
    assert by_symbol["ONLYALP"]["source"] == "alpaca"
    assert by_symbol["ONLYALP"]["close"] == 20.0


def test_universe_at_point_in_time(repo: MarketDataRepository) -> None:
    repo.upsert_universe_membership(
        [
            UniverseMembership(
                symbol="AAPL", tier=Tier.CORE, valid_from=date(2026, 1, 1),
                valid_to=None, reason="seed", approved_by="seed",
            ),
            UniverseMembership(
                symbol="NEWCO", tier=Tier.PROMOTED, valid_from=date(2026, 7, 1),
                valid_to=date(2026, 7, 31), reason="triage", approved_by=None,
            ),
        ]
    )
    july = repo.universe_at(date(2026, 7, 15))
    assert {m.symbol for m in july} == {"AAPL", "NEWCO"}
    august = repo.universe_at(date(2026, 8, 15))
    assert {m.symbol for m in august} == {"AAPL"}  # promotion lapsed
    core_only = repo.universe_at(date(2026, 7, 15), Tier.CORE)
    assert {m.symbol for m in core_only} == {"AAPL"}


def _core(symbol: str, valid_from: date, approved_by: str | None) -> UniverseMembership:
    return UniverseMembership(
        symbol=symbol, tier=Tier.CORE, valid_from=valid_from,
        valid_to=None, reason="core seed: built-in starter list", approved_by=approved_by,
    )


def test_backdate_core_universe_replaces_seed_in_one_shot(
    repo: MarketDataRepository, engine: Engine
) -> None:
    """Phase 1 backdate: seed rows deleted + backdated rows inserted; each
    symbol ends with exactly ONE open interval; human-approved rows survive."""
    repo.upsert_universe_membership(
        [
            _core("AAPL", date(2026, 8, 1), "seed"),
            _core("SPY", date(2026, 8, 1), "seed"),
            _core("HUMAN", date(2026, 8, 5), "someone@example.com"),
        ]
    )
    backdated = [
        UniverseMembership(
            symbol=s, tier=Tier.CORE, valid_from=date(2024, 12, 1), valid_to=None,
            reason="replay backfill campaign backdate", approved_by="replay-backdate",
        )
        for s in ("AAPL", "SPY")
    ]
    deleted, inserted = repo.backdate_core_universe(backdated)
    assert (deleted, inserted) == (2, 2)

    with engine.connect() as conn:
        rows = conn.execute(
            select(schema.universe_membership).order_by(
                schema.universe_membership.c.symbol
            )
        ).all()
    by_symbol: dict[str, list] = {}
    for r in rows:
        by_symbol.setdefault(r.symbol, []).append(r)
    # Single interval per backdated symbol, starting at the backdate.
    for sym in ("AAPL", "SPY"):
        assert len(by_symbol[sym]) == 1
        assert by_symbol[sym][0].valid_from == date(2024, 12, 1)
        assert by_symbol[sym][0].valid_to is None
        assert by_symbol[sym][0].approved_by == "replay-backdate"
    # Human-approved row untouched.
    assert by_symbol["HUMAN"][0].valid_from == date(2026, 8, 5)

    # Survivorship view: active both at the backdate and today.
    assert {m.symbol for m in repo.universe_at(date(2024, 12, 1))} == {"AAPL", "SPY"}
    assert {"AAPL", "SPY", "HUMAN"} <= {
        m.symbol for m in repo.universe_at(max(date.today(), date(2026, 8, 6)))
    }


def test_backdate_core_universe_is_idempotent_and_rebases(
    repo: MarketDataRepository, engine: Engine
) -> None:
    repo.upsert_universe_membership([_core("AAPL", date(2026, 8, 1), "seed")])

    def backdate(as_of: date) -> None:
        repo.backdate_core_universe(
            [
                UniverseMembership(
                    symbol="AAPL", tier=Tier.CORE, valid_from=as_of, valid_to=None,
                    reason="replay backfill campaign backdate",
                    approved_by="replay-backdate",
                )
            ]
        )

    def intervals() -> list[date]:
        with engine.connect() as conn:
            return [
                r.valid_from
                for r in conn.execute(
                    select(schema.universe_membership).where(
                        schema.universe_membership.c.symbol == "AAPL"
                    )
                )
            ]

    backdate(date(2024, 12, 1))
    backdate(date(2024, 12, 1))  # same as-of twice -> no dupes
    assert intervals() == [date(2024, 12, 1)]
    backdate(date(2025, 6, 1))  # different as-of -> replaced again
    assert intervals() == [date(2025, 6, 1)]


def test_seed_instrument_ciks_preserves_existing_profiles(
    repo: MarketDataRepository, engine: Engine
) -> None:
    from dynamo.domain.models.market import Instrument

    repo.upsert_instruments(
        [Instrument(symbol="AAPL", name="Apple Inc.", sector="Information Technology")]
    )
    repo.seed_instrument_ciks({"AAPL": "0000320193", "NEWCO": "0000000042"})
    with engine.connect() as conn:
        rows = {r.symbol: r for r in conn.execute(select(schema.instruments))}
    assert rows["AAPL"].cik == "0000320193"
    assert rows["AAPL"].name == "Apple Inc."  # profile untouched
    assert rows["AAPL"].sector == "Information Technology"
    assert rows["NEWCO"].cik == "0000000042"  # minimal new row
    assert rows["NEWCO"].name == "NEWCO"


def test_quarantine_and_alert_writes(repo: MarketDataRepository, engine: Engine) -> None:
    repo.add_quarantine(scope="symbol", key="ACME", reason="gap: 1 missing day")
    repo.add_alert(severity="warn", name="data_quality_systemic", message="21 symbols flagged")
    with engine.connect() as conn:
        q = conn.execute(select(schema.quarantine)).all()
        a = conn.execute(select(schema.alerts)).all()
    assert len(q) == 1 and q[0].key == "ACME"
    assert len(a) == 1 and a[0].ackable is True


def test_generic_upsert_on_quant_scores(repo: MarketDataRepository, engine: Engine) -> None:
    row = {
        "symbol": "ACME", "as_of": date(2026, 7, 31), "model_version": "v1",
        "expected_excess_return": 0.02, "confidence": 0.5, "feature_coverage": 1.0,
    }
    repo.upsert_rows(schema.quant_scores, [row])
    repo.upsert_rows(schema.quant_scores, [{**row, "expected_excess_return": 0.03}])
    with engine.connect() as conn:
        rows = conn.execute(select(schema.quant_scores)).all()
    assert len(rows) == 1
    assert rows[0].expected_excess_return == 0.03


def test_upsert_rows_chunks_below_bind_param_limit(
    repo: MarketDataRepository,
    engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batches above the wire-protocol bind-param cap split into chunks.

    Regression: EDGAR companyfacts backfills (>20k rows x 6 params) blew the
    65,535-parameter Postgres limit in one statement. Shrink the cap so a
    100-row batch must chunk, and verify every row lands and the count is
    honest (psycopg reports rowcount=-1 for multi-VALUES upserts).
    """
    from dynamo.persistence.repositories import market as market_module

    monkeypatch.setattr(market_module, "_PG_MAX_BIND_PARAMS", 100)  # ~9 rows/chunk
    bars = [bar("ACME", 1 + (i % 28), "alpaca", float(i), None) for i in range(28)]
    written = repo.upsert_bars(bars)
    assert written == 28
    with engine.connect() as conn:
        rows = conn.execute(select(schema.bars_1d)).all()
    assert len(rows) == 28


def test_upsert_xbrl_facts_dedupes_in_batch_keep_last(
    repo: MarketDataRepository, engine: Engine
) -> None:
    """EDGAR repeats (cik, tag, period_end, filed_at) within one payload
    (e.g. Q4 and FY durations sharing end+filed): the batch must dedupe
    keep-last instead of raising CardinalityViolation."""
    from dynamo.domain.models.market import XbrlFact

    filed = datetime(2026, 2, 1, tzinfo=UTC)
    facts = [
        XbrlFact(cik="0000000001", tag="Revenues", period_end=date(2025, 12, 31),
                 filed_at=filed, value=100.0, unit="USD"),
        XbrlFact(cik="0000000001", tag="Revenues", period_end=date(2025, 12, 31),
                 filed_at=filed, value=400.0, unit="USD"),
    ]
    assert repo.upsert_xbrl_facts(facts) == 1
    with engine.connect() as conn:
        rows = conn.execute(select(schema.xbrl_facts)).all()
    assert len(rows) == 1
    assert rows[0].value == 400.0  # keep-last: the row-by-row upsert end state


def test_upsert_transcripts_dedupes_in_batch_keep_last(
    repo: MarketDataRepository, engine: Engine
) -> None:
    """defeatbeta can emit the same (symbol, event_date, source) twice in one
    bulk payload; the batch must dedupe keep-last, not CardinalityViolation."""
    from dynamo.domain.models.market import Transcript

    transcripts = [
        Transcript(symbol="ACME", event_date=date(2026, 1, 15), source="defeatbeta",
                   period="Q4-2025", content="first draft", speakers=None),
        Transcript(symbol="ACME", event_date=date(2026, 1, 15), source="defeatbeta",
                   period="Q4-2025", content="revised", speakers=None),
    ]
    assert repo.upsert_transcripts(transcripts) == 1
    with engine.connect() as conn:
        rows = conn.execute(select(schema.transcripts)).all()
    assert len(rows) == 1
    assert rows[0].content == "revised"
