"""HistoricalDataGateway — THE only read path replay uses (plan Phase 4).

Every method enforces a ``<= as_of`` upper bound on the knowledge-time axis
of what it reads (bar timestamps, filing times, macro vintages, verdict /
score / digest dates, journal write times, experience dates). The gateway
leakage test suite seeds rows after ``as_of`` and asserts they are invisible
— that suite is the contract.

There are NO write methods here, by design: replay's only writes go through
``ReplayRepository`` and ``ExperienceRepository.insert`` (source='replay').

Known fidelity caveats (accepted in the plan):

- ``instruments`` (sector/cik) is not vintaged — sector drift over time is
  accepted and logged as a caveat, not reconstructed.
- earnings calendar rows are bounded by ``fetched_at`` (when the platform
  learned of the event), not ``event_date`` — future event *dates* are
  legitimately knowable in advance, but only once ingested.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select

from dynamo.domain.models.learning import DecisionExperience, JournalEntry, JournalKind
from dynamo.domain.models.market import Instrument, NewsItem
from dynamo.domain.models.risk import Regime, RegimeState
from dynamo.persistence import schema
from dynamo.persistence.repositories.experience import ExperienceRepository
from dynamo.persistence.repositories.learning import JournalRepository
from dynamo.persistence.repositories.market import MarketDataRepository

FEATURE_BARS_LOOKBACK_DAYS = 420
FUNDAMENTALS_LOOKBACK_DAYS = 500
INSIDER_LOOKBACK_DAYS = 200
MACRO_LOOKBACK_DAYS = 400
EARNINGS_LOOKAHEAD_DAYS = 30
TRANSCRIPT_LOOKBACK_DAYS = 30  # live dossier window parity (assets/research.py)

# Timestamp-like columns the runtime tripwire sweeps in every frame that
# passes through :meth:`HistoricalDataGateway._read_df` (backtest-campaign
# Phase 4). Queries whose semantics legitimately include later values exempt
# columns explicitly with a comment (future earnings event_date, backfill
# ingestion fetched_at).
_TS_LIKE_COLUMNS = (
    "ts",
    "filed_at",
    "fetched_at",
    "published_at",
    "vintage_ts",
    "as_of",
    "created_at",
    "event_date",
)


class ReplayLeakageError(RuntimeError):
    """A gateway read surfaced data dated after ``as_of`` — hard stop.

    This is the runtime tripwire behind the leakage test suite: even if a
    query's WHERE clause regresses, no future-dated row leaves the gateway.
    """


class HistoricalDataGateway:
    """Point-in-time reads at a fixed ``as_of``; one instance per session day."""

    def __init__(self, engine: Engine, as_of: date) -> None:
        self._engine = engine
        self._repo = MarketDataRepository(engine)
        self.as_of = as_of
        # Inclusive end-of-day instant for timestamp-typed bounds, matching
        # the live pipeline's whole-day semantics (bars_frame, journal reads).
        self._cutoff = datetime.combine(as_of, time.max, tzinfo=UTC)

    # -- universe / bars ---------------------------------------------------

    def universe(self, tiers: tuple[int, ...] | None = None) -> list[str]:
        """Active membership symbols as of ``as_of`` (survivorship-safe:
        ``valid_from <= as_of <= valid_to``); memberships starting after
        ``as_of`` are invisible."""
        members = self._repo.universe_at(self.as_of)
        for m in members:  # tripwire: repo must never return a future start
            self._check_value("universe.valid_from", m.valid_from)
        return sorted(
            {m.symbol for m in members if tiers is None or int(m.tier) in tiers}
        )

    def bars(self, symbols: Sequence[str] | None, lookback_days: int) -> pd.DataFrame:
        """Canonical daily bars with ``ts`` in ``[as_of - lookback, as_of]`` —
        the end bound is ``as_of``, so bars after ``as_of`` are invisible."""
        frame = self._repo.bars_frame(
            list(symbols) if symbols is not None else None,
            self.as_of - timedelta(days=lookback_days),
            self.as_of,
        )
        return self._check_frame(frame, source="bars")

    # -- research inputs ---------------------------------------------------

    def feature_inputs(self, symbols: Sequence[str]) -> Any:
        """``FeatureInputs`` mirroring the nightly ``_feature_inputs`` builder
        but with BOTH bounds on every frame:

        - bars: ``ts <= as_of`` (via :meth:`bars`);
        - fundamentals: ``as_of - 500d <= filed_at <= as_of``;
        - insider: ``as_of - 200d <= filed_at <= as_of``;
        - macro: ``as_of - 400d <= ts <= as_of`` AND ``vintage_ts <= as_of``
          (revisions published later are invisible);
        - earnings calendar: ``event_date >= as_of - 30d`` AND
          ``fetched_at <= as_of`` (knowledge-time bound).
        """
        from dynamo.data.universe import default_sector_etf_map
        from dynamo.research.quant.catalog import FeatureInputs

        symbols = list(symbols)
        bars = self.bars(symbols, FEATURE_BARS_LOOKBACK_DAYS)
        inst = schema.instruments
        inst_df = self._read_df(select(inst.c.symbol, inst.c.sector, inst.c.cik))
        sectors = {
            str(r["symbol"]): str(r["sector"])
            for r in inst_df.to_dict(orient="records")
            if r["sector"]
        }
        cik_to_symbol = {
            str(r["cik"]): str(r["symbol"])
            for r in inst_df.to_dict(orient="records")
            if r["cik"]
        }
        x = schema.xbrl_facts
        fundamentals = self._read_df(
            select(x).where(
                x.c.filed_at >= self._since(FUNDAMENTALS_LOOKBACK_DAYS),
                x.c.filed_at <= self._cutoff,
            )
        )
        if not fundamentals.empty:
            fundamentals["symbol"] = fundamentals["cik"].astype(str).map(cik_to_symbol.get)
            fundamentals = fundamentals.dropna(subset=["symbol"])
        i = schema.insider_tx
        insider = self._read_df(
            select(i).where(
                i.c.filed_at >= self._since(INSIDER_LOOKBACK_DAYS),
                i.c.filed_at <= self._cutoff,
            )
        )
        macro = self.macro_series(MACRO_LOOKBACK_DAYS)
        e = schema.earnings_events
        # event_date exempt from the tripwire: future event DATES are
        # legitimately knowable in advance — the knowledge axis here is
        # fetched_at (bounded <= cutoff above and swept by the tripwire).
        earnings = self._read_df(
            select(e.c.symbol, e.c.event_date).where(
                e.c.event_date >= self.as_of - timedelta(days=EARNINGS_LOOKAHEAD_DAYS),
                e.c.fetched_at <= self._cutoff,
            ),
            exempt=("event_date",),
        )
        return FeatureInputs(
            bars=bars,
            fundamentals=fundamentals if not fundamentals.empty else None,
            insider=insider if not insider.empty else None,
            macro=macro if not macro.empty else None,
            earnings_calendar=earnings if not earnings.empty else None,
            sector_map=default_sector_etf_map(sectors),
        )

    def quant_scores(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Latest quant score per (symbol, model_version) with
        ``as_of <= self.as_of`` — future scores are invisible."""
        frame = self._repo.latest_quant_scores(
            list(symbols) if symbols is not None else None, as_of=self.as_of
        )
        return self._check_frame(frame, source="quant_scores")

    def verdicts(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Latest committee verdict per symbol with ``as_of <= self.as_of`` —
        future verdicts are invisible."""
        frame = self._repo.latest_verdicts(
            list(symbols) if symbols is not None else None, as_of=self.as_of
        )
        return self._check_frame(frame, source="verdicts")

    def macro_digest(self) -> dict[str, Any] | None:
        """Most recent macro-digest payload with ``as_of <= self.as_of`` —
        digests written after ``as_of`` are invisible."""
        payload = self._repo.latest_macro_digest(as_of=self.as_of)
        if payload is not None and payload.get("as_of"):
            digest_as_of = payload["as_of"]
            if isinstance(digest_as_of, str):
                digest_as_of = date.fromisoformat(digest_as_of[:10])
            self._check_value("macro_digest.as_of", digest_as_of)
        return payload

    def transcripts(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Earnings-call transcripts with ``event_date`` in
        ``[as_of - 30d, as_of]`` — the live dossier window, point-in-time:
        calls after ``as_of`` are invisible. ``fetched_at`` (backfill
        ingestion wall clock) is deliberately not selected: the event date is
        the knowledge axis for a published call transcript."""
        t = schema.transcripts
        return self._read_df(
            select(t.c.symbol, t.c.event_date, t.c.source, t.c.content).where(
                t.c.symbol.in_(list(symbols)),
                t.c.event_date >= self.as_of - timedelta(days=TRANSCRIPT_LOOKBACK_DAYS),
                t.c.event_date <= self.as_of,
            )
        )

    def macro_series(self, lookback_days: int) -> pd.DataFrame:
        """Macro observations with ``ts`` in ``[as_of - lookback, as_of]`` AND
        ``vintage_ts <= as_of`` — later-published revisions are invisible."""
        m = schema.macro_series
        return self._read_df(
            select(m).where(
                m.c.ts >= self.as_of - timedelta(days=lookback_days),
                m.c.ts <= self.as_of,
                m.c.vintage_ts <= self._cutoff,
            )
        )

    def regime_state(self) -> RegimeState | None:
        """Most recent persisted regime state with ``as_of <= self.as_of`` —
        states computed for later sessions are invisible."""
        rs = schema.regime_states
        stmt = (
            select(rs)
            .where(rs.c.as_of <= self.as_of)
            .order_by(rs.c.as_of.desc())
            .limit(1)
        )
        with self._engine.connect() as cx:
            row = cx.execute(stmt).first()
        if row is None:
            return None
        m = row._mapping
        self._check_value("regime_state.as_of", m["as_of"])
        return RegimeState(
            as_of=m["as_of"],
            regime=Regime(m["regime"]),
            composite_score=float(m["composite_score"]),
            inputs=dict(m["inputs"] or {}),
            sessions_in_state=int(m["sessions_in_state"] or 0),
        )

    def instruments(self, symbols: Sequence[str]) -> dict[str, Instrument]:
        """symbol -> Instrument profile. NOT vintaged (same accepted fidelity
        caveat as :meth:`sectors`): the profile reflects the current row, not
        the classification at ``as_of``."""
        inst = schema.instruments
        stmt = select(inst).where(inst.c.symbol.in_(list(symbols)))
        frame = self._read_df(stmt)
        fields = set(Instrument.model_fields)
        out: dict[str, Instrument] = {}
        for r in frame.to_dict(orient="records"):
            payload = {k: v for k, v in r.items() if k in fields and v is not None}
            instrument = Instrument.model_validate(payload)
            out[instrument.symbol] = instrument
        return out

    def news(
        self, symbols: Sequence[str], lookback_days: int = 7
    ) -> dict[str, list[NewsItem]]:
        """symbol -> news items with ``published_at`` in
        ``[as_of - lookback, as_of]`` — items published after ``as_of`` are
        invisible. Sparse-history days simply return fewer (or no) items,
        rendering as the same absent section a thin live dossier shows."""
        n = schema.news
        # fetched_at exempt from the tripwire: historical backfills ingest
        # old items TODAY (fetched_at = ingestion wall clock, legitimately
        # after as_of) — the knowledge axis for news is published_at, which
        # is bounded above and swept by the tripwire.
        frame = self._read_df(
            select(n).where(
                n.c.published_at >= self._since(lookback_days),
                n.c.published_at <= self._cutoff,
            ),
            exempt=("fetched_at",),
        )
        wanted = set(symbols)
        out: dict[str, list[NewsItem]] = {sym: [] for sym in symbols}
        fields = set(NewsItem.model_fields)
        for r in frame.to_dict(orient="records"):
            tagged = tuple(r.get("symbols") or ())
            payload = {k: v for k, v in r.items() if k in fields and v is not None}
            payload["symbols"] = tagged
            for sym in tagged:
                if sym in wanted:
                    out[sym].append(NewsItem.model_validate(payload))
        return out

    def sectors(self, symbols: Sequence[str] | None = None) -> dict[str, str]:
        """symbol -> sector from ``instruments``. NOT vintaged (accepted
        fidelity caveat): sector reflects the current row, not the historical
        classification at ``as_of``."""
        inst = schema.instruments
        stmt = select(inst.c.symbol, inst.c.sector)
        if symbols is not None:
            stmt = stmt.where(inst.c.symbol.in_(list(symbols)))
        frame = self._read_df(stmt)
        return {
            str(r["symbol"]): str(r["sector"] or "unknown")
            for r in frame.to_dict(orient="records")
        }

    # -- learning-context reads -------------------------------------------

    def journal_search(
        self,
        *,
        embedding: Sequence[float] | None = None,
        text_query: str | None = None,
        kinds: Sequence[JournalKind] = (),
        symbols: Sequence[str] = (),
        top_k: int = 5,
    ) -> list[tuple[JournalEntry, float]]:
        """Journal search that ALWAYS passes ``as_of`` (end of the gateway
        day) — entries written after ``as_of`` are invisible."""
        hits = JournalRepository(self._engine).search(
            embedding=embedding,
            text_query=text_query,
            kinds=kinds,
            symbols=symbols,
            top_k=top_k,
            as_of=self._cutoff,
        )
        for entry, _score in hits:  # tripwire on the repo's own filter
            self._check_value("journal_search.created_at", entry.created_at)
        return hits

    def similar_experiences(
        self,
        *,
        symbol: str,
        sector: str | None = None,
        regime: str | None = None,
        sleeve: str | None = None,
        limit: int = 3,
        include_replay_run_id: str | None = None,
    ) -> list[DecisionExperience]:
        """Prior similar experiences strictly before ``as_of`` — the Phase 2
        ``similar()`` already hard-filters ``as_of <`` the query date and
        ``source='live'``; the gateway pins the query date to its own.

        ``include_replay_run_id`` (within-run learning, backtest-campaign
        Phase 4) additionally admits THIS run's own resolved replay
        experiences — still ``as_of``-bounded, still firewalled from every
        other run's rows and from live learning."""
        hits = ExperienceRepository(self._engine).similar(
            symbol=symbol,
            as_of=self.as_of,
            sector=sector,
            regime=regime,
            sleeve=sleeve,
            limit=limit,
            include_replay_run_id=include_replay_run_id,
        )
        for exp in hits:  # tripwire: the contract is STRICTLY before as_of
            if exp.as_of >= self.as_of:
                raise ReplayLeakageError(
                    f"leakage tripwire: similar_experiences returned {exp.id} "
                    f"dated {exp.as_of} not strictly before as_of {self.as_of}"
                )
        return hits

    # -- helpers -----------------------------------------------------------

    def _since(self, days: int) -> datetime:
        return datetime.combine(
            self.as_of - timedelta(days=days), time.min, tzinfo=UTC
        )

    def _read_df(self, stmt: Any, *, exempt: Sequence[str] = ()) -> pd.DataFrame:
        with self._engine.connect() as cx:
            frame = pd.read_sql(stmt, cx)
        return self._check_frame(frame, exempt=exempt, source="query")

    def _check_frame(
        self, frame: pd.DataFrame, *, exempt: Sequence[str] = (), source: str = ""
    ) -> pd.DataFrame:
        """Runtime tripwire: raise if any known ts-like column holds a value
        after the ``as_of`` cutoff. Trivial overhead — a ``max()`` per
        present column on an already-loaded frame."""
        if frame.empty:
            return frame
        for col in _TS_LIKE_COLUMNS:
            if col in exempt or col not in frame.columns:
                continue
            series = frame[col].dropna()
            if series.empty:
                continue
            self._check_value(f"{source or 'frame'}.{col}", series.max())
        return frame

    def _check_value(self, label: str, value: Any) -> None:
        """Raise :class:`ReplayLeakageError` when ``value`` post-dates the
        gateway day. Dates compare against ``as_of``; datetimes against the
        inclusive end-of-day cutoff (naive values compared naively)."""
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            cutoff = (
                self._cutoff if value.tzinfo is not None
                else self._cutoff.replace(tzinfo=None)
            )
            leaked = value > cutoff
        elif isinstance(value, date):
            leaked = value > self.as_of
        else:
            return
        if leaked:
            raise ReplayLeakageError(
                f"leakage tripwire: {label} = {value} exceeds the "
                f"as_of {self.as_of} cutoff"
            )
