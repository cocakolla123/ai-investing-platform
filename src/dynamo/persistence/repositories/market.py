"""MarketDataRepository — the shared write path for the nightly pipeline.

Design: doc 02 §3 (storage), doc 01 §6 (idempotency: all ingestion upserts are
natural-key idempotent via ``ON CONFLICT DO UPDATE``).

One repository per aggregate *cluster* (doc 01 §2): this one owns market data,
universe membership, quality/quarantine/alerts, and the generic upsert path
used by research/quant writers (dossiers, verdicts, scores, watchlist, ...).

Point-in-time reads:

- :meth:`bars_frame` — canonical daily closes (prefer Tiingo ``adj_close``,
  else Alpaca raw close; doc 02 §5).
- :meth:`universe_at` — survivorship-safe membership as of a date (doc 02 §2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import Engine, Table, delete, select, update
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
        ts = fetched_at or _now()
        rows = [
            {
                "symbol": a.symbol,
                "ex_date": a.ex_date,
                "action_type": a.action_type,
                "source": a.source,
                "ratio": a.ratio,
                "amount": a.amount,
                "fetched_at": ts,
            }
            for a in actions
        ]
        return self.upsert_rows(schema.corporate_actions, rows)

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
        )
        with self._engine.begin() as conn:
            deleted = conn.execute(
                delete(t).where(
                    t.c.symbol.in_(symbols),
                    t.c.tier == int(Tier.CORE),
                    t.c.approved_by.in_(list(delete_approved_by)),
                )
            ).rowcount
            inserted = conn.execute(stmt).rowcount
        return (int(deleted or 0), _reported_rowcount(inserted, len(rows)))

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

        Both sources are stored; the canonical close prefers the Tiingo row
        (``close`` column = its ``adj_close``, doc 02 §5 adjustment policy) and
        falls back to the Alpaca raw close. Columns:
        ``symbol, ts, open, high, low, close, adj_close, volume, source``.
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
        if frame.empty:
            empty_cols = [
                "symbol", "ts", "open", "high", "low",
                "close", "adj_close", "volume", "source",
            ]
            return pd.DataFrame({c: [] for c in empty_cols})
        # Prefer tiingo per (symbol, ts); its adj_close becomes the canonical close.
        frame["_rank"] = (frame["source"] != "tiingo").astype(int)
        frame = (
            frame.sort_values(["symbol", "ts", "_rank"])
            .drop_duplicates(subset=["symbol", "ts"], keep="first")
            .drop(columns=["_rank", "fetched_at"])
        )
        is_tiingo = (frame["source"] == "tiingo") & frame["adj_close"].notna()
        frame.loc[is_tiingo, "close"] = frame.loc[is_tiingo, "adj_close"]
        return frame.reset_index(drop=True)

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
