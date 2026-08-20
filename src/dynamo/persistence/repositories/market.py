"""MarketDataRepository — the shared write path for the nightly pipeline.

Design: doc 02 §3 (storage), doc 01 §6 (idempotency: all ingestion upserts are
natural-key idempotent via ``ON CONFLICT DO UPDATE``).

One repository per aggregate *cluster* (doc 01 §2): this one owns market data,
universe membership, quality/quarantine/alerts, and the generic upsert path
used by research/quant writers (dossiers, verdicts, scores, watchlist, ...).

Point-in-time reads:

- :meth:`bars_frame` — the canonical multi-source view. Prices prefer Tiingo's
  adjusted close, VOLUME prefers Alpaca so the whole panel sits on one tape;
  the rule and the reasoning live in :func:`canonical_bars`.
- :meth:`universe_at` — survivorship-safe membership as of a date (doc 02 §2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, cast

import pandas as pd
import structlog
from sqlalchemy import Engine, Table, delete, func, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.market import (
    Bar,
    CorporateAction,
    EarningsEvent,
    Filing,
    InsiderTx,
    Instrument,
    MacroPoint,
    NewsItem,
    PredictionMarket,
    PredictionMarketPrice,
    SurveillanceFlag,
    Tier,
    Transcript,
    UniverseMembership,
    XbrlFact,
)
from dynamo.domain.models.pm import CrossVenueSpread
from dynamo.persistence import schema

log = structlog.get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


# PostgreSQL extended-protocol hard cap on bind parameters per statement
# (uint16). psycopg raises OperationalError above this; upsert_rows chunks
# batches to stay below it.
_PG_MAX_BIND_PARAMS = 65535


def estimate_vintage_rows(
    events: Sequence[EarningsEvent],
    latest_by_key: Mapping[tuple[str, date], tuple[float | None, float | None]],
    fetched_at: datetime,
) -> list[dict[str, Any]]:
    """Rows to APPEND to ``earnings_estimate_vintages`` — pure core.

    One row per (symbol, event_date) whose ``(eps_est, revenue_est)`` differs
    from the latest stored vintage (``latest_by_key``); unchanged estimates
    append NOTHING (no daily duplicates). Events carrying no estimate at all
    are skipped. In-batch duplicates keep the last occurrence.
    """
    by_key: dict[tuple[str, date], EarningsEvent] = {}
    for event in events:
        if event.eps_est is None and event.revenue_est is None:
            continue
        by_key[(event.symbol, event.event_date)] = event
    rows: list[dict[str, Any]] = []
    for key, event in by_key.items():
        if latest_by_key.get(key) == (event.eps_est, event.revenue_est):
            continue
        rows.append(
            {
                "symbol": event.symbol,
                "event_date": event.event_date,
                "fetched_at": fetched_at,
                "eps_est": event.eps_est,
                "revenue_est": event.revenue_est,
            }
        )
    return rows


# --- The canonical-view rule (D2) -----------------------------------------

#: Feed preference for the PRICE columns (open/high/low/close/adj_close), best
#: first. Tiingo's ``adj_close`` is split- AND dividend-adjusted consolidated
#: EOD; the Alpaca free feed is split-adjusted IEX.
CANONICAL_PRICE_SOURCES: tuple[str, ...] = ("tiingo", "alpaca")
#: Feed preference for the VOLUME column, best first — deliberately the
#: REVERSE of the price preference. See :func:`canonical_bars`.
CANONICAL_VOLUME_SOURCES: tuple[str, ...] = ("alpaca", "tiingo")

CANONICAL_BAR_COLUMNS: tuple[str, ...] = (
    "symbol", "ts", "open", "high", "low",
    "close", "adj_close", "volume", "source", "volume_source",
)


def _source_rank(sources: pd.Series, preference: Sequence[str]) -> pd.Series:
    """Position of each row's source in ``preference`` (unlisted sorts last)."""
    order = {name: i for i, name in enumerate(preference)}
    return sources.map(lambda s: order.get(str(s), len(order))).astype(int)


def canonical_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a multi-source ``bars_1d`` frame to one row per (symbol, ts).

    THE RULE, stated once so it stops being implicit:

    ==========  ============================  ====================
    Column      Source, best first            Reported in
    ==========  ============================  ====================
    price*      tiingo, then alpaca           ``source``
    volume      **alpaca**, then tiingo       ``volume_source``
    ==========  ============================  ====================

    (*price = open/high/low/close/adj_close. When the winning row is Tiingo
    and carries an ``adj_close``, that adjusted value BECOMES ``close`` — the
    canonical close is an adjusted close, doc 02 §5.)

    WHY PRICE AND VOLUME COME FROM DIFFERENT FEEDS. They are not the same
    measurement problem.

    *Price* wants the most-adjusted, whole-market print, which is Tiingo.

    *Volume* wants the SAME TAPE for every symbol, which is Alpaca. Tiingo
    reports consolidated volume — every share on every venue. Alpaca's free
    feed reports IEX only, measured at ~6.6% of that (``dynamo.domain
    .volume_source``). Only 55 names carry Tiingo; the other ~11,200 are
    Alpaca-only. Taking "the best available volume per row" therefore puts 55
    mega-caps on a ~15x larger scale than the rest of the panel, and every
    cross-sectional volume ranking — ``dollar_vol_20d``, ``dollar_vol_z_60d``,
    ``amihud_20d``, ``volume_trend_20_60`` — silently sorts by which vendor
    supplied the row. Measured on comparable mega-caps: 17.4M shares/day on
    Tiingo vs 1.46M on Alpaca IEX.

    This is the crypto-calendar bug's exact shape: a data-source discontinuity
    corrupting a cross-sectional ranking without anything failing. The fix is
    not a better estimate, it is refusing to mix — one tape for the whole
    panel, even though it is the *narrower* tape.

    Commit b294501 solved the other half of the same problem: absolute
    dollar-volume THRESHOLDS are expressed in consolidated-equivalent dollars
    and scaled at the point of comparison via ``volume_source_share``. That
    makes a threshold portable across feeds; it does not make two feeds
    comparable to each other within one ranking. Both are needed, and this
    frame carries ``volume_source`` precisely so that the scaling function is
    handed the feed that produced ``volume`` rather than the feed that
    produced ``close``.

    KNOWN RESIDUAL. Alpaca history in this database starts 2024-01-02 while
    Tiingo reaches back to 2018-06-01, so a Tier-1 name's volume series
    switches tape at that boundary. The break is not removed here — it is
    LABELLED, in ``volume_source``, so a consumer can scale across it or
    refuse to span it. Removing it needs Alpaca bars before 2024, which the
    free feed does not serve.
    """
    if frame.empty:
        return pd.DataFrame({c: [] for c in CANONICAL_BAR_COLUMNS})

    frame = frame.copy()
    source = cast(pd.Series, frame["source"])
    frame["_price_rank"] = _source_rank(source, CANONICAL_PRICE_SOURCES)
    frame["_volume_rank"] = _source_rank(source, CANONICAL_VOLUME_SOURCES)

    ordered = frame.sort_values(by=["symbol", "ts", "_price_rank"], kind="mergesort")
    price = cast(
        pd.DataFrame, ordered.drop_duplicates(subset=["symbol", "ts"], keep="first")
    ).copy()

    # Volume is chosen independently, then grafted on by (symbol, ts).
    by_volume = frame.sort_values(by=["symbol", "ts", "_volume_rank"], kind="mergesort")
    chosen = cast(
        pd.DataFrame, by_volume.drop_duplicates(subset=["symbol", "ts"], keep="first")
    )
    volume = cast(pd.DataFrame, chosen[["symbol", "ts", "volume", "source"]]).copy()
    volume["volume_source"] = volume.pop("source")

    price = price.drop(columns=["volume"])
    merged = price.merge(volume, on=["symbol", "ts"], how="left")

    # The canonical close is the ADJUSTED close wherever one exists.
    adj_close = cast(pd.Series, merged["adj_close"])
    merged["close"] = adj_close.where(adj_close.notna(), merged["close"])
    return cast(pd.DataFrame, merged[list(CANONICAL_BAR_COLUMNS)]).reset_index(drop=True)


def _reported_rowcount(rowcount: int | None, fallback: int) -> int:
    """Driver-honest affected-row count for multi-VALUES statements.

    psycopg3 reports ``rowcount = -1`` for multi-VALUES ``INSERT .. ON
    CONFLICT`` statements (observed during the replay-campaign backfills, where
    every count printed as ``-1``). When the driver cannot report, fall back to
    the number of rows submitted — for our idempotent upserts every submitted
    row is inserted or updated, so the fallback is exact, not an estimate.
    """
    if rowcount is None or rowcount < 0:
        return fallback
    return int(rowcount)


class MarketDataRepository:
    """Idempotent upserts + point-in-time reads over the market-data tables."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    # -- generic helpers ---------------------------------------------------

    def upsert_rows(self, table: Table, rows: Sequence[Mapping[str, Any]]) -> int:
        """Natural-key idempotent upsert: ``ON CONFLICT (pk) DO UPDATE`` (doc 01 §6).

        Shared write path for every keyed table, including the research tables
        (dossiers, macro_digests, committee_verdicts, quant_scores,
        quality_scores, watchlist, prediction_market_views, batch_jobs, ...).

        Large batches are chunked below the PostgreSQL wire-protocol limit of
        65,535 bind parameters per statement (hit in practice by EDGAR
        companyfacts backfills: one mega-cap symbol yields >20k fact rows x
        6 columns). Each chunk commits independently — a re-run repairs any
        partial write (idempotent upserts).
        """
        if not rows:
            return 0
        params_per_row = max(1, len(rows[0]))
        chunk_size = max(1, _PG_MAX_BIND_PARAMS // params_per_row)
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = list(rows[start : start + chunk_size])
            stmt = pg_insert(table).values(chunk)
            pk = [c.name for c in table.primary_key.columns]
            updatable = {
                c.name: stmt.excluded[c.name]
                for c in table.columns
                if c.name not in pk and c.name in chunk[0]
            }
            if updatable:
                stmt = stmt.on_conflict_do_update(index_elements=pk, set_=updatable)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk)
            with self._engine.begin() as conn:
                result = conn.execute(stmt)
            total += _reported_rowcount(result.rowcount, len(chunk))
        return total

    def insert_rows(self, table: Table, rows: Sequence[Mapping[str, Any]]) -> int:
        """Plain insert for append-only tables with surrogate ids (flags, alerts)."""
        if not rows:
            return 0
        with self._engine.begin() as conn:
            result = conn.execute(table.insert(), list(rows))
        return _reported_rowcount(result.rowcount, len(rows))

    # -- ingestion upserts (provenance carried on every record) ------------

    def upsert_bars(
        self,
        bars: Iterable[Bar],
        *,
        timeframe: str = "1d",
        fetched_at: datetime | None = None,
    ) -> int:
        table = schema.bars_1d if timeframe == "1d" else schema.bars_1m
        ts = fetched_at or _now()
        rows = [
            {
                "symbol": b.symbol,
                "ts": b.ts,
                "source": b.source,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "adj_close": b.adj_close,
                "fetched_at": ts,
            }
            for b in bars
        ]
        return self.upsert_rows(table, rows)

    def upsert_instruments(self, instruments: Iterable[Instrument]) -> int:
        rows = [
            {
                "symbol": i.symbol,
                "name": i.name,
                "exchange": i.exchange,
                "sector": i.sector,
                "industry": i.industry,
                "cik": i.cik,
                "figi": i.figi,
                "is_etf": i.is_etf,
                "status": str(i.status),
                "first_seen": i.first_seen,
                "delisted_at": i.delisted_at,
            }
            for i in instruments
        ]
        return self.upsert_rows(schema.instruments, rows)

    def upsert_filings(
        self, filings: Iterable[Filing], *, fetched_at: datetime | None = None
    ) -> int:
        ts = fetched_at or _now()
        rows = [
            {
                "accession_no": f.accession_no,
                "cik": f.cik,
                "symbol": f.symbol,
                "form_type": f.form_type,
                "filed_at": f.filed_at,
                "period": f.period,
                "url": f.url,
                "items": list(f.items),
                "fetched_at": ts,
            }
            for f in filings
        ]
        return self.upsert_rows(schema.filings, rows)

    def upsert_xbrl_facts(self, facts: Iterable[XbrlFact]) -> int:
        """Point-in-time fundamentals keyed by ``filed_at`` (doc 02 §3).

        EDGAR companyfacts legitimately repeats the natural key: one filing
        can report the same ``(cik, tag, period_end, filed_at)`` for several
        durations (e.g. Q4 and full-year values both ending 12-31, filed the
        same day). A multi-VALUES ``ON CONFLICT DO UPDATE`` refuses in-batch
        duplicates (CardinalityViolation), so dedupe keep-last first — the
        same end state row-by-row upserts would produce.
        """
        deduped: dict[tuple[str, str, date, datetime], dict[str, Any]] = {
            (f.cik, f.tag, f.period_end, f.filed_at): {
                "cik": f.cik,
                "tag": f.tag,
                "period_end": f.period_end,
                "filed_at": f.filed_at,
                "value": f.value,
                "unit": f.unit,
            }
            for f in facts
        }
        return self.upsert_rows(schema.xbrl_facts, list(deduped.values()))

    def upsert_insider_tx(self, txs: Iterable[InsiderTx]) -> int:
        rows = [
            {
                "accession_no": t.accession_no,
                "symbol": t.symbol,
                "insider_name": t.insider_name,
                "tx_type": t.tx_type,
                "role": t.role,
                "shares": t.shares,
                "price": t.price,
                "filed_at": t.filed_at,
            }
            for t in txs
        ]
        return self.upsert_rows(schema.insider_tx, rows)

    def upsert_news(self, items: Iterable[NewsItem], *, fetched_at: datetime | None = None) -> int:
        ts = fetched_at or _now()
        rows = [
            {
                "id": n.id,
                "symbols": list(n.symbols),
                "headline": n.headline,
                "summary": n.summary,
                "source": n.source,
                "published_at": n.published_at,
                "url": n.url,
                "fetched_at": ts,
            }
            for n in items
        ]
        return self.upsert_rows(schema.news, rows)

    def upsert_macro(self, points: Iterable[MacroPoint]) -> int:
        """FRED vintages: (series_id, ts, vintage_ts) key keeps every revision."""
        rows = [
            {
                "series_id": p.series_id,
                "ts": p.ts,
                "vintage_ts": p.vintage_ts,
                "value": p.value,
            }
            for p in points
        ]
        return self.upsert_rows(schema.macro_series, rows)

    def upsert_earnings(
        self, events: Iterable[EarningsEvent], *, fetched_at: datetime | None = None
    ) -> int:
        ts = fetched_at or _now()
        rows = [
            {
                "symbol": e.symbol,
                "event_date": e.event_date,
                "period": e.period,
                "eps_est": e.eps_est,
                "eps_act": e.eps_act,
                "revenue_est": e.revenue_est,
                "revenue_act": e.revenue_act,
                "call_transcript_ref": e.call_transcript_ref,
                "fetched_at": ts,
            }
            for e in events
        ]
        return self.upsert_rows(schema.earnings_events, rows)

    def latest_estimate_vintages(
        self, keys: Iterable[tuple[str, date]]
    ) -> dict[tuple[str, date], tuple[float | None, float | None]]:
        """(symbol, event_date) -> latest stored (eps_est, revenue_est)."""
        keys = list(keys)
        if not keys:
            return {}
        from sqlalchemy import tuple_ as sa_tuple

        t = schema.earnings_estimate_vintages
        stmt = (
            select(t.c.symbol, t.c.event_date, t.c.fetched_at, t.c.eps_est, t.c.revenue_est)
            .where(sa_tuple(t.c.symbol, t.c.event_date).in_(keys))
            .order_by(t.c.symbol, t.c.event_date, t.c.fetched_at)
        )
        latest: dict[tuple[str, date], tuple[float | None, float | None]] = {}
        with self._engine.connect() as cx:
            for r in cx.execute(stmt):  # ordered by fetched_at: last write wins
                latest[(r.symbol, r.event_date)] = (r.eps_est, r.revenue_est)
        return latest

    def append_estimate_vintages(
        self, events: Iterable[EarningsEvent], *, fetched_at: datetime | None = None
    ) -> int:
        """Append-only estimate history (migration 0019): write one vintage
        row per (symbol, event_date) whose estimate CHANGED vs the latest
        stored vintage — the estimates-vintage fix for the destructive
        ``earnings_events`` upsert. Unchanged estimates write nothing, so
        the daily calendar refresh never duplicates rows."""
        ts = fetched_at or _now()
        events = list(events)
        latest = self.latest_estimate_vintages(
            {(e.symbol, e.event_date) for e in events}
        )
        rows = estimate_vintage_rows(events, latest, ts)
        if not rows:
            return 0
        t = schema.earnings_estimate_vintages
        stmt = pg_insert(t).values(rows).on_conflict_do_nothing(
            index_elements=[c.name for c in t.primary_key.columns]
        )
        with self._engine.begin() as cx:
            result = cx.execute(stmt)
        return _reported_rowcount(result.rowcount, len(rows))

    def upsert_transcripts(
        self, transcripts: Iterable[Transcript], *, fetched_at: datetime | None = None
    ) -> int:
        """Earnings-call transcripts, keyed by (symbol, event_date, source).

        Bulk providers (defeatbeta) can emit the same (symbol, event_date,
        source) more than once in one payload (e.g. re-published revisions);
        a multi-VALUES ``ON CONFLICT DO UPDATE`` refuses in-batch duplicates,
        so dedupe keep-last first (the row-by-row upsert end state).
        """
        ts = fetched_at or _now()
        deduped: dict[tuple[str, date, str], dict[str, Any]] = {
            (t.symbol, t.event_date, t.source): {
                "symbol": t.symbol,
                "event_date": t.event_date,
                "source": t.source,
                "period": t.period,
                "content": t.content,
                "speakers": [dict(s) for s in t.speakers] if t.speakers is not None else None,
                "fetched_at": ts,
            }
            for t in transcripts
        }
        return self.upsert_rows(schema.transcripts, list(deduped.values()))

    def link_transcript_ref(self, symbol: str, event_date: date, ref: str) -> int:
        """Point the matching earnings_events row at a stored transcript."""
        stmt = (
            update(schema.earnings_events)
            .where(
                schema.earnings_events.c.symbol == symbol,
                schema.earnings_events.c.event_date == event_date,
            )
            .values(call_transcript_ref=ref)
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return int(result.rowcount or 0)

    def upsert_corporate_actions(
        self, actions: Iterable[CorporateAction], *, fetched_at: datetime | None = None
    ) -> int:
        """Splits/dividends/mergers/spinoffs, keyed
        ``(symbol, ex_date, action_type, source)``.

        Alpaca's announcements endpoint legitimately repeats that key inside
        one window: an announcement can be amended, and a corporate event
        involving two tickers is returned once per side, so the same
        ``(symbol, ex_date, "merger")`` arrives more than once. A multi-VALUES
        ``ON CONFLICT DO UPDATE`` refuses in-batch duplicates
        (``CardinalityViolation``, hit on the first real 2024 load), so dedupe
        keep-last first — the same end state row-by-row upserts would produce.
        """
        ts = fetched_at or _now()
        deduped: dict[tuple[str, date, str, str], dict[str, Any]] = {
            (a.symbol, a.ex_date, a.action_type, a.source): {
                "symbol": a.symbol,
                "ex_date": a.ex_date,
                "action_type": a.action_type,
                "source": a.source,
                "ratio": a.ratio,
                "amount": a.amount,
                "fetched_at": ts,
            }
            for a in actions
        }
        return self.upsert_rows(schema.corporate_actions, list(deduped.values()))

    def upsert_prediction_markets(self, markets: Iterable[PredictionMarket]) -> int:
        rows = [
            {
                "venue": m.venue,
                "market_id": m.market_id,
                "question": m.question,
                "category": m.category,
                "close_ts": m.close_ts,
                "resolved_outcome": m.resolved_outcome,
            }
            for m in markets
        ]
        return self.upsert_rows(schema.prediction_markets, rows)

    def upsert_prediction_market_prices(self, prices: Iterable[PredictionMarketPrice]) -> int:
        rows = [
            {
                "venue": p.venue,
                "market_id": p.market_id,
                "ts": p.ts,
                "implied_prob": p.implied_prob,
                "volume": p.volume,
                "liquidity": p.liquidity,
                "trusted": p.trusted,
            }
            for p in prices
        ]
        return self.upsert_rows(schema.prediction_market_prices, rows)

    def upsert_cross_venue_spreads(self, spreads: Iterable[CrossVenueSpread]) -> int:
        """OddsPipe matched pairs (enrichment); keyed by (kalshi_market_id, ts)."""
        rows = [
            {
                "question": s.question,
                "kalshi_market_id": s.kalshi_market_id,
                "polymarket_id": s.polymarket_id,
                "kalshi_yes": s.kalshi_yes,
                "polymarket_yes": s.polymarket_yes,
                "yes_diff": s.yes_diff,
                "ts": s.as_of,
            }
            for s in spreads
        ]
        return self.upsert_rows(schema.pm_cross_venue_spreads, rows)

    def insert_surveillance_flags(self, flags: Iterable[SurveillanceFlag]) -> int:
        """Append-only (surrogate id); nightly runs write one batch per date."""
        rows = [
            {
                "symbol": f.symbol,
                "detector": f.detector,
                "severity": f.severity,
                "details": dict(f.details),
                "ts": f.ts,
            }
            for f in flags
        ]
        return self.insert_rows(schema.surveillance_flags, rows)

    def untriaged_flags(
        self,
        since: datetime,
        *,
        exclude_detectors: Sequence[str] = (),
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Un-triaged ``surveillance_flags`` rows written on or after ``since``.

        This is the read that turns the flag table into a queue. Triage used
        to look only at the flags computed in its own process, so a row
        written by any other path — the ``surveillance-sweep`` CLI, an earlier
        run that died before triage, the nightly asset on a night the model
        was unavailable — was never looked at again by anything.

        ``since`` bounds the queue: a flag old enough to have gone stale
        should expire unread rather than be triaged against today's prices.

        Returns columns ``id, symbol, detector, severity, details, ts``,
        oldest first so a backlog drains in the order it accumulated. The
        ``id`` is what a caller must use to write a verdict back — see
        :meth:`set_triage_verdicts`.
        """
        t = schema.surveillance_flags
        conditions = [t.c.triage_verdict.is_(None), t.c.ts >= since]
        if exclude_detectors:
            conditions.append(t.c.detector.notin_(list(exclude_detectors)))
        stmt = (
            select(t.c.id, t.c.symbol, t.c.detector, t.c.severity, t.c.details, t.c.ts)
            .where(*conditions)
            .order_by(t.c.ts, t.c.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._engine.connect() as conn:
            frame = pd.read_sql(stmt, conn)
        if frame.empty:
            return pd.DataFrame(
                {c: [] for c in ("id", "symbol", "detector", "severity", "details", "ts")}
            )
        return frame

    def set_triage_verdicts(
        self,
        flag_ids: Sequence[int],
        verdict: str,
        *,
        triaged_at: datetime | None = None,
    ) -> int:
        """Stamp a verdict on EXACTLY the given flag rows.

        Bounded by primary key on purpose. The previous write matched on
        ``symbol AND triage_verdict IS NULL`` with no time bound, so triaging
        one night's flag for a symbol also stamped every historical un-triaged
        row that symbol had ever produced with tonight's verdict — silently
        rewriting the record of what was decided, and when, about flags nobody
        had ever looked at. Matching on id makes the write say what it means.
        """
        ids = [int(i) for i in flag_ids]
        if not ids:
            return 0
        t = schema.surveillance_flags
        with self._engine.begin() as conn:
            result = conn.execute(
                update(t)
                .where(t.c.id.in_(ids))
                .values(triage_verdict=verdict, triaged_at=triaged_at or _now())
            )
        return int(result.rowcount or 0)

    def upsert_universe_membership(self, memberships: Iterable[UniverseMembership]) -> int:
        rows = [
            {
                "symbol": m.symbol,
                "valid_from": m.valid_from,
                "tier": int(m.tier),
                "valid_to": m.valid_to,
                "reason": m.reason,
                "approved_by": m.approved_by,
            }
            for m in memberships
        ]
        return self.upsert_rows(schema.universe_membership, rows)

    def backdate_core_universe(
        self,
        memberships: Sequence[UniverseMembership],
        *,
        delete_approved_by: Sequence[str] = ("seed", "replay-backdate"),
    ) -> tuple[int, int]:
        """Atomically replace the CORE seed intervals for these symbols.

        ONE transaction (replay backfill campaign): delete every existing
        CORE row for the given symbols whose ``approved_by`` is in
        ``delete_approved_by`` (the original ``'seed'`` rows plus any prior
        ``'replay-backdate'`` rows — that is what makes a re-run with a
        different ``as_of`` replace rather than accumulate intervals), then
        re-insert the new memberships with the same upsert semantics as
        :meth:`upsert_universe_membership`. Returns ``(deleted, inserted)``.

        Human-approved CORE additions (any other ``approved_by``) are never
        touched.
        """
        rows = [
            {
                "symbol": m.symbol,
                "valid_from": m.valid_from,
                "tier": int(m.tier),
                "valid_to": m.valid_to,
                "reason": m.reason,
                "approved_by": m.approved_by,
            }
            for m in memberships
        ]
        if not rows:
            return (0, 0)
        t = schema.universe_membership
        symbols = [r["symbol"] for r in rows]
        stmt = pg_insert(t).values(rows)
        pk = [c.name for c in t.primary_key.columns]
        stmt = stmt.on_conflict_do_update(
            index_elements=pk,
            set_={
                c.name: stmt.excluded[c.name] for c in t.columns if c.name not in pk
            },
        ).returning(t.c.symbol)
        # RETURNING gives a reliable insert count (psycopg3 reports rowcount=-1
        # for multi-VALUES upserts) — same pattern as ReplayRepository.
        with self._engine.begin() as conn:
            deleted = conn.execute(
                delete(t).where(
                    t.c.symbol.in_(symbols),
                    t.c.tier == int(Tier.CORE),
                    t.c.approved_by.in_(list(delete_approved_by)),
                )
            ).rowcount
            inserted = len(conn.execute(stmt).all())
        return (int(deleted or 0), int(inserted))

    def first_bar_dates(self, *, source: str | None = None) -> dict[str, date]:
        """symbol -> date of its earliest ``bars_1d`` row.

        The listing-date proxy behind the Tier-2 backdate (D3): a vendor has
        no history before a symbol starts trading, so the first bar bounds the
        listing date from above using nothing but data already in the table.
        """
        t = schema.bars_1d
        stmt = select(t.c.symbol, func.min(t.c.ts).label("first_ts")).group_by(t.c.symbol)
        if source is not None:
            stmt = stmt.where(t.c.source == source)
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()
        return {str(r[0]): cast(datetime, r[1]).date() for r in rows}

    def bars_close_frame(
        self,
        *,
        source: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """``symbol, ts, close`` straight out of ``bars_1d``, per SOURCE.

        Deliberately NOT the canonical view: the split-artifact census asks
        "are the rows this feed stored adjusted?", which is a question about
        one feed's rows and would be answered wrong by a frame that silently
        substituted another feed's close.
        """
        t = schema.bars_1d
        stmt = select(t.c.symbol, t.c.ts, t.c.close)
        if source is not None:
            stmt = stmt.where(t.c.source == source)
        if start is not None:
            stmt = stmt.where(t.c.ts >= datetime(start.year, start.month, start.day, tzinfo=UTC))
        if end is not None:
            stmt = stmt.where(
                t.c.ts <= datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC)
            )
        with self._engine.connect() as conn:
            return pd.read_sql(stmt, conn)

    def count_corporate_actions(self) -> int:
        with self._engine.connect() as conn:
            return int(
                conn.execute(select(func.count()).select_from(schema.corporate_actions)).scalar()
                or 0
            )

    def corporate_actions_by_type(self) -> dict[str, int]:
        t = schema.corporate_actions
        stmt = select(t.c.action_type, func.count()).group_by(t.c.action_type)
        with self._engine.connect() as conn:
            return {str(r[0]): int(r[1]) for r in conn.execute(stmt)}

    def split_ex_dates(self) -> set[tuple[str, date]]:
        """``(symbol, ex_date)`` for every recorded split.

        The independent witness the census cross-checks against: a large move
        landing on a ratio is a *candidate*; a large move landing on a ratio
        AND a recorded split ex-date is evidence the rows are unadjusted.
        """
        t = schema.corporate_actions
        stmt = select(t.c.symbol, t.c.ex_date).where(t.c.action_type == "split")
        with self._engine.connect() as conn:
            return {(str(r[0]), cast(date, r[1])) for r in conn.execute(stmt)}

    def rebase_universe_valid_from(
        self, rebases: Sequence[tuple[str, date, date]], *, chunk_size: int = 500
    ) -> tuple[int, int]:
        """Move membership rows from one ``valid_from`` to another, atomically.

        ``rebases``: ``(symbol, old_valid_from, new_valid_from)``. ``valid_from``
        is half the primary key, so this is a delete + insert rather than an
        update; a symbol's delete and insert always happen in ONE transaction,
        so a kill can never leave a symbol with no membership row at all.

        Chunked at ``chunk_size`` pairs: a row-values ``IN`` list of 11k tuples
        exceeds PostgreSQL's parser stack (``StatementTooComplex``, measured on
        the 11,251-row Tier-2 backdate). Chunks commit independently, so a kill
        mid-run leaves some symbols rebased and some not — which the idempotent
        re-run repairs, because an already-rebased symbol has no row at its old
        ``valid_from`` and is skipped.

        Only the exact ``(symbol, old_valid_from)`` rows named are deleted —
        rows at any other ``valid_from``, and every row of every symbol not
        named, are untouched. That is what keeps the hand-approved Tier-1 CORE
        intervals out of the blast radius. Returns ``(deleted, inserted)``.
        """
        if not rebases:
            return (0, 0)
        step = max(1, int(chunk_size))
        if len(rebases) > step:
            deleted_total, inserted_total = 0, 0
            for start in range(0, len(rebases), step):
                deleted, inserted = self.rebase_universe_valid_from(
                    rebases[start : start + step], chunk_size=step
                )
                deleted_total += deleted
                inserted_total += inserted
            return (deleted_total, inserted_total)
        t = schema.universe_membership
        keys = [(symbol, old) for symbol, old, _ in rebases]
        with self._engine.begin() as conn:
            existing = {
                (str(r.symbol), cast(date, r.valid_from)): dict(r._mapping)
                for r in conn.execute(
                    select(t).where(tuple_(t.c.symbol, t.c.valid_from).in_(keys))
                )
            }
            rows: list[dict[str, Any]] = []
            for symbol, old, new in rebases:
                row = existing.get((symbol, old))
                if row is None:  # already rebased by a previous run
                    continue
                rows.append({**row, "valid_from": new})
            if not rows:
                return (0, 0)
            deleted = conn.execute(
                delete(t).where(tuple_(t.c.symbol, t.c.valid_from).in_(keys))
            ).rowcount
            stmt = pg_insert(t).values(rows)
            pk = [c.name for c in t.primary_key.columns]
            stmt = stmt.on_conflict_do_update(
                index_elements=pk,
                set_={c.name: stmt.excluded[c.name] for c in t.columns if c.name not in pk},
            ).returning(t.c.symbol)
            inserted = len(conn.execute(stmt).all())
        return (int(deleted or 0), int(inserted))

    def seed_instrument_ciks(self, ciks: Mapping[str, str]) -> int:
        """Upsert ``instruments.cik`` for the given symbols (EDGAR mapping).

        Existing rows get ONLY their ``cik`` updated (name/sector/etc are
        preserved); missing rows are created minimally (``name=symbol``).
        """
        if not ciks:
            return 0
        t = schema.instruments
        rows = [
            {"symbol": symbol, "name": symbol, "cik": cik}
            for symbol, cik in sorted(ciks.items())
        ]
        stmt = pg_insert(t).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"], set_={"cik": stmt.excluded.cik}
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return _reported_rowcount(result.rowcount, len(rows))

    def seed_instrument_sectors(self, sectors: Mapping[str, str]) -> int:
        """Upsert ``instruments.sector`` for the given symbols.

        Existing rows get ONLY their ``sector`` updated (cik/name/etc are
        preserved — the CIK seeding and this seeding compose in either
        order); missing rows are created minimally (``name=symbol``).
        """
        if not sectors:
            return 0
        t = schema.instruments
        rows = [
            {"symbol": symbol, "name": symbol, "sector": sector}
            for symbol, sector in sorted(sectors.items())
        ]
        stmt = pg_insert(t).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"], set_={"sector": stmt.excluded.sector}
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return _reported_rowcount(result.rowcount, len(rows))

    def seed_instrument_industries(self, industries: Mapping[str, str]) -> int:
        """Upsert ``instruments.industry`` (mirrors :meth:`seed_instrument_sectors`).

        Additive in the same way: only ``industry`` is written, so the CIK,
        sector and industry seeds compose in any order.
        """
        if not industries:
            return 0
        t = schema.instruments
        rows = [
            {"symbol": symbol, "name": symbol, "industry": industry}
            for symbol, industry in sorted(industries.items())
        ]
        stmt = pg_insert(t).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"], set_={"industry": stmt.excluded.industry}
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return _reported_rowcount(result.rowcount, len(rows))

    def close_membership(
        self,
        symbol: str,
        valid_from: date,
        valid_to: date,
        *,
        approved_by: str | None = None,
    ) -> int:
        """End a membership interval (tier exit / demotion approval)."""
        stmt = (
            update(schema.universe_membership)
            .where(
                schema.universe_membership.c.symbol == symbol,
                schema.universe_membership.c.valid_from == valid_from,
            )
            .values(valid_to=valid_to, **({"approved_by": approved_by} if approved_by else {}))
        )
        with self._engine.begin() as conn:
            result = conn.execute(stmt)
        return int(result.rowcount or 0)

    def add_quarantine(
        self,
        *,
        scope: str,
        key: str,
        reason: str,
        payload: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Quarantine rows hold invalid records/symbols for inspection (doc 01 §6)."""
        self.insert_rows(
            schema.quarantine,
            [
                {
                    "scope": scope,
                    "key": key,
                    "reason": reason,
                    "payload": dict(payload) if payload else None,
                    "created_at": created_at or _now(),
                }
            ],
        )

    def add_alert(
        self,
        *,
        severity: str,
        name: str,
        message: str,
        ackable: bool = True,
        created_at: datetime | None = None,
    ) -> None:
        self.insert_rows(
            schema.alerts,
            [
                {
                    "created_at": created_at or _now(),
                    "severity": severity,
                    "name": name,
                    "message": message,
                    "ackable": ackable,
                }
            ],
        )

    # -- point-in-time reads -----------------------------------------------

    def bars_frame(
        self,
        symbols: Sequence[str] | None,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Canonical daily bars as a DataFrame (doc 02 §3 canonical view).

        The per-column source rule is :func:`canonical_bars` — read its
        docstring, not this one, for why price and volume come from different
        feeds. Columns: ``symbol, ts, open, high, low, close, adj_close,
        volume, source, volume_source``.
        """
        t = schema.bars_1d
        conditions = [
            t.c.ts >= datetime(start.year, start.month, start.day, tzinfo=UTC),
            t.c.ts <= datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC),
        ]
        if symbols is not None:
            conditions.append(t.c.symbol.in_(list(symbols)))
        stmt = select(t).where(*conditions).order_by(t.c.symbol, t.c.ts)
        with self._engine.connect() as conn:
            frame = pd.read_sql(stmt, conn)
        return canonical_bars(frame)

    def universe_at(self, as_of: date, tier: Tier | None = None) -> list[UniverseMembership]:
        """Survivorship-safe membership as of ``as_of`` (doc 02 §2)."""
        t = schema.universe_membership
        conditions = [
            t.c.valid_from <= as_of,
            (t.c.valid_to.is_(None)) | (t.c.valid_to >= as_of),
        ]
        if tier is not None:
            conditions.append(t.c.tier == int(tier))
        with self._engine.connect() as conn:
            rows = conn.execute(select(t).where(*conditions)).mappings().all()
        return [
            UniverseMembership(
                symbol=r["symbol"],
                tier=Tier(r["tier"]),
                valid_from=r["valid_from"],
                valid_to=r["valid_to"],
                reason=r["reason"],
                approved_by=r["approved_by"],
            )
            for r in rows
        ]

    def latest_verdicts(
        self,
        symbols: Sequence[str] | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Most recent committee verdict per symbol.

        ``as_of`` (optional) upper-bounds the read at ``as_of`` inclusive —
        the point-in-time contract replay's gateway depends on. ``None``
        keeps the live behavior (no bound) so call sites are unchanged.
        """
        t = schema.committee_verdicts
        stmt = select(t)
        if symbols is not None:
            stmt = stmt.where(t.c.symbol.in_(list(symbols)))
        if as_of is not None:
            stmt = stmt.where(t.c.as_of <= as_of)
        with self._engine.connect() as conn:
            frame = pd.read_sql(stmt, conn)
        if frame.empty:
            return frame
        return (
            frame.sort_values(["symbol", "as_of"])
            .drop_duplicates(subset=["symbol"], keep="last")
            .reset_index(drop=True)
        )

    def latest_quant_scores(
        self,
        symbols: Sequence[str] | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Most recent quant score per (symbol, model_version).

        ``as_of`` (optional) upper-bounds the read at ``as_of`` inclusive
        (replay point-in-time); ``None`` keeps the live behavior.
        """
        t = schema.quant_scores
        stmt = select(t)
        if symbols is not None:
            stmt = stmt.where(t.c.symbol.in_(list(symbols)))
        if as_of is not None:
            stmt = stmt.where(t.c.as_of <= as_of)
        with self._engine.connect() as conn:
            frame = pd.read_sql(stmt, conn)
        if frame.empty:
            return frame
        return (
            frame.sort_values(["symbol", "model_version", "as_of"])
            .drop_duplicates(subset=["symbol", "model_version"], keep="last")
            .reset_index(drop=True)
        )

    def latest_macro_digest(self, as_of: date | None = None) -> dict[str, Any] | None:
        """Payload of the most recent macro digest, or ``None`` when absent.

        ``as_of`` (optional) upper-bounds the read at ``as_of`` inclusive
        (replay point-in-time); ``None`` keeps the live behavior (latest row).
        """
        t = schema.macro_digests
        stmt = select(t.c.payload)
        if as_of is not None:
            stmt = stmt.where(t.c.as_of <= as_of)
        stmt = stmt.order_by(t.c.as_of.desc()).limit(1)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        if row is None or row[0] is None:
            return None
        return dict(row[0])
