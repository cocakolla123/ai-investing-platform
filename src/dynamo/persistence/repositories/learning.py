"""Learning-loop repositories (doc 10): hash-chained decision context,
reviews, journal (pgvector), proposals, calibration, metrics, model registry.

``decision_context`` is append-only INSERT with the hash chain computed here
(read last ``row_hash`` → compute); the registry's status column changes ONLY
via ``apply_promotion`` with a validated ``PromotionRecord``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.learning import (
    CalibrationPoint,
    ImprovementProposal,
    JournalEntry,
    JournalKind,
    ModelVersion,
    PostTradeReview,
    PromotionRecord,
    ProposalStatus,
    RegistryStatus,
)
from dynamo.domain.models.portfolio import DecisionContext
from dynamo.domain.models.research import Sleeve
from dynamo.learning.hash_chain import GENESIS, ChainVerification, row_hash, verify_chain
from dynamo.learning.promotion import validate_transition
from dynamo.persistence.schema import (
    calibration_points,
    decision_context,
    improvement_proposals,
    journal_entries,
    metrics_daily,
    model_versions,
    post_trade_reviews,
    promotion_records,
    trial_counter,
)


def _dc_body(dc: DecisionContext) -> dict[str, Any]:
    """The hashed portion of a decision-context row (everything but hashes)."""
    return {
        "id": dc.id,
        "created_at": dc.created_at.isoformat(),
        "env": dc.env,
        "symbol": dc.symbol,
        "sleeve": dc.sleeve.value,
        "verdict_id": dc.verdict_id,
        "target_portfolio_id": dc.target_portfolio_id,
        "payload": dc.payload,
        "expected_upside_pct": dc.expected_upside_pct,
        "expected_downside_pct": dc.expected_downside_pct,
        "horizon_days": dc.horizon_days,
    }


class DecisionContextRepository:
    """Append-only, hash-chained (doc 10 §2). No update/delete methods exist."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def append(self, dc: DecisionContext) -> str:
        """INSERT with the chain extended: prev_hash = last row_hash (or
        GENESIS), row_hash = sha256(prev ‖ canonical body). Returns row_hash."""
        body = _dc_body(dc)
        with self._engine.begin() as cx:
            last = cx.execute(
                select(decision_context.c.row_hash)
                .order_by(decision_context.c.created_at.desc(), decision_context.c.id.desc())
                .limit(1)
            ).first()
            prev = last.row_hash if last is not None else GENESIS
            digest = row_hash(prev, body)
            cx.execute(
                decision_context.insert().values(
                    id=dc.id,
                    created_at=dc.created_at,
                    env=dc.env,
                    symbol=dc.symbol,
                    sleeve=dc.sleeve.value,
                    verdict_id=dc.verdict_id,
                    target_portfolio_id=dc.target_portfolio_id,
                    payload=dc.payload,
                    expected_upside_pct=dc.expected_upside_pct,
                    expected_downside_pct=dc.expected_downside_pct,
                    horizon_days=dc.horizon_days,
                    prev_hash=prev,
                    row_hash=digest,
                )
            )
        return digest

    def get(self, dc_id: str) -> DecisionContext | None:
        with self._engine.connect() as cx:
            r = cx.execute(
                select(decision_context).where(decision_context.c.id == dc_id)
            ).first()
        if r is None:
            return None
        return DecisionContext(
            id=r.id,
            created_at=r.created_at,
            env=r.env,
            symbol=r.symbol,
            sleeve=Sleeve(r.sleeve),
            verdict_id=r.verdict_id,
            target_portfolio_id=r.target_portfolio_id,
            payload=r.payload,
            expected_upside_pct=r.expected_upside_pct,
            expected_downside_pct=r.expected_downside_pct,
            horizon_days=r.horizon_days,
        )

    def verify(self) -> ChainVerification:
        """Verify the full chain in insertion order (nightly job, doc 10 §9)."""
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(decision_context).order_by(
                    decision_context.c.created_at, decision_context.c.id
                )
            ).all()
        chain = [
            {
                **_dc_body(
                    DecisionContext(
                        id=r.id,
                        created_at=r.created_at,
                        env=r.env,
                        symbol=r.symbol,
                        sleeve=Sleeve(r.sleeve),
                        verdict_id=r.verdict_id,
                        target_portfolio_id=r.target_portfolio_id,
                        payload=r.payload,
                        expected_upside_pct=r.expected_upside_pct,
                        expected_downside_pct=r.expected_downside_pct,
                        horizon_days=r.horizon_days,
                    )
                ),
                "prev_hash": r.prev_hash,
                "row_hash": r.row_hash,
            }
            for r in rows
        ]
        return verify_chain(chain)


class ReviewRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, review: PostTradeReview) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                post_trade_reviews.insert().values(
                    trade_id=review.trade_id,
                    decision_context_id=review.decision_context_id,
                    symbol=review.symbol,
                    as_of=review.as_of,
                    payload=review.model_dump(mode="json"),
                    failure_primary=(
                        review.failure_primary.value if review.failure_primary else None
                    ),
                )
            )

    def get(self, trade_id: str) -> PostTradeReview | None:
        with self._engine.connect() as cx:
            r = cx.execute(
                select(post_trade_reviews.c.payload).where(
                    post_trade_reviews.c.trade_id == trade_id
                )
            ).first()
        return PostTradeReview.model_validate(r.payload) if r is not None else None

    def reviewed_trade_ids(self) -> frozenset[str]:
        with self._engine.connect() as cx:
            rows = cx.execute(select(post_trade_reviews.c.trade_id)).all()
        return frozenset(r.trade_id for r in rows)


class JournalRepository:
    """Journal cluster: insert with optional pgvector embedding; search by
    cosine similarity when an embedding is given, else ILIKE + filters."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(
        self,
        entry: JournalEntry,
        embedding: Sequence[float] | None = None,
        embedding_version: str = "",
    ) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                journal_entries.insert().values(
                    id=entry.id,
                    kind=entry.kind.value,
                    symbols=list(entry.symbols),
                    strategy=entry.strategy,
                    regime_at_write=entry.regime_at_write,
                    sectors=list(entry.sectors),
                    text=entry.text,
                    refs=entry.refs,
                    created_at=entry.created_at,
                    embedding=list(embedding) if embedding is not None else None,
                    embedding_version=embedding_version,
                )
            )

    def search(
        self,
        *,
        embedding: Sequence[float] | None = None,
        text_query: str | None = None,
        kinds: Sequence[JournalKind] = (),
        symbols: Sequence[str] = (),
        strategy: str | None = None,
        regime: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[JournalEntry, float]]:
        """Returns (entry, similarity). Cosine similarity in [0
        .. 1]-ish when searching by embedding; 0.0 for text-only matches."""
        stmt = select(journal_entries)
        if kinds:
            stmt = stmt.where(journal_entries.c.kind.in_([k.value for k in kinds]))
        if symbols:
            stmt = stmt.where(journal_entries.c.symbols.overlap(list(symbols)))
        if strategy is not None:
            stmt = stmt.where(journal_entries.c.strategy == strategy)
        if regime is not None:
            stmt = stmt.where(journal_entries.c.regime_at_write == regime)

        if embedding is not None:
            distance = journal_entries.c.embedding.cosine_distance(list(embedding))
            stmt = (
                stmt.add_columns(distance.label("distance"))
                .where(journal_entries.c.embedding.is_not(None))
                .order_by(distance)
                .limit(top_k)
            )
        else:
            if text_query:
                stmt = stmt.where(journal_entries.c.text.ilike(f"%{text_query}%"))
            stmt = stmt.order_by(journal_entries.c.created_at.desc()).limit(top_k)

        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()

        out: list[tuple[JournalEntry, float]] = []
        for r in rows:
            entry = JournalEntry(
                id=r.id,
                kind=JournalKind(r.kind),
                symbols=tuple(r.symbols or ()),
                strategy=r.strategy,
                regime_at_write=r.regime_at_write,
                sectors=tuple(r.sectors or ()),
                text=r.text,
                refs=r.refs or {},
                created_at=r.created_at,
            )
            similarity = 1.0 - float(r.distance) if embedding is not None else 0.0
            out.append((entry, similarity))
        return out


class ProposalRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, proposal: ImprovementProposal) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                improvement_proposals.insert().values(
                    id=proposal.id,
                    created_at=proposal.created_at,
                    kind=proposal.kind.value,
                    target=proposal.target,
                    evidence=[e.model_dump(mode="json") for e in proposal.evidence],
                    expected_effect=proposal.expected_effect,
                    proposed_change=proposal.proposed_change,
                    status=proposal.status.value,
                )
            )

    def set_status(self, proposal_id: str, status: ProposalStatus) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                update(improvement_proposals)
                .where(improvement_proposals.c.id == proposal_id)
                .values(status=status.value)
            )

    def open_proposals(self) -> list[dict[str, Any]]:
        with self._engine.connect() as cx:
            rows = cx.execute(
                select(improvement_proposals)
                .where(improvement_proposals.c.status == ProposalStatus.OPEN.value)
                .order_by(improvement_proposals.c.created_at)
            ).all()
        return [dict(r._mapping) for r in rows]


class CalibrationRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, points: Sequence[CalibrationPoint]) -> None:
        with self._engine.begin() as cx:
            for p in points:
                stmt = pg_insert(calibration_points).values(
                    scope_type=p.scope_type,
                    scope_id=p.scope_id,
                    as_of=p.as_of,
                    brier=p.brier,
                    n=p.n,
                    bucket_stats=p.bucket_stats,
                )
                cx.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[
                            calibration_points.c.scope_type,
                            calibration_points.c.scope_id,
                            calibration_points.c.as_of,
                        ],
                        set_={"brier": p.brier, "n": p.n, "bucket_stats": p.bucket_stats},
                    )
                )


class MetricsRepository:
    """Nightly metrics are recompute-and-upsert by (metric, scope, ts) so
    restatements after late data are clean (doc 10 §8)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert_metric(
        self, metric: str, scope_type: str, scope_id: str, as_of: date, value: float
    ) -> None:
        stmt = pg_insert(metrics_daily).values(
            metric=metric, scope_type=scope_type, scope_id=scope_id, as_of=as_of, value=value
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[
                metrics_daily.c.metric,
                metrics_daily.c.scope_type,
                metrics_daily.c.scope_id,
                metrics_daily.c.as_of,
            ],
            set_={"value": value},
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)


class ModelRegistryRepository:
    """Registry writes: candidates in, status changes ONLY via
    ``apply_promotion`` (doc 08 §7 — only the promotion tool writes)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert_candidate(self, mv: ModelVersion) -> None:
        with self._engine.begin() as cx:
            cx.execute(
                model_versions.insert().values(
                    id=mv.id,
                    model_type=mv.model_type,
                    algo=mv.algo,
                    features_hash=mv.features_hash,
                    train_start=mv.train_start,
                    train_end=mv.train_end,
                    hyperparams=mv.hyperparams,
                    metrics=mv.metrics,
                    artifact_path=mv.artifact_path,
                    status=RegistryStatus.CANDIDATE.value,  # always enters as candidate
                    created_at=mv.created_at or datetime.now(),
                )
            )

    def status_of(self, target_id: str) -> RegistryStatus | None:
        with self._engine.connect() as cx:
            r = cx.execute(
                select(model_versions.c.status).where(model_versions.c.id == target_id)
            ).first()
        return RegistryStatus(r.status) if r is not None else None

    def apply_promotion(self, record: PromotionRecord) -> None:
        """The ONLY path that mutates registry status. Verifies the record
        matches the stored current status and a legal transition."""
        validate_transition(record.from_status, record.to_status)
        with self._engine.begin() as cx:
            current = cx.execute(
                select(model_versions.c.status)
                .where(model_versions.c.id == record.target_id)
                .with_for_update()
            ).first()
            if current is None:
                raise LookupError(f"unknown registry target {record.target_id}")
            if RegistryStatus(current.status) != record.from_status:
                raise ValueError(
                    f"stale promotion record: registry has {current.status}, "
                    f"record expects {record.from_status.value}"
                )
            values: dict[str, Any] = {"status": record.to_status.value}
            if record.to_status == RegistryStatus.PRODUCTION:
                values["promoted_at"] = record.at
                values["promoted_by"] = record.actor
            cx.execute(
                update(model_versions)
                .where(model_versions.c.id == record.target_id)
                .values(**values)
            )
            cx.execute(
                promotion_records.insert().values(
                    id=record.id,
                    target_id=record.target_id,
                    from_status=record.from_status.value,
                    to_status=record.to_status.value,
                    actor=record.actor,
                    evidence_bundle_ref=record.evidence_bundle_ref,
                    trial_count=record.trial_count,
                    git_commit=record.git_commit,
                    config_hash=record.config_hash,
                    at=record.at,
                )
            )

    def increment_trials(self, family: str) -> int:
        """Trial counter for multiple-testing discipline (doc 08)."""
        stmt = pg_insert(trial_counter).values(family=family, trials=1)
        stmt = stmt.on_conflict_do_update(
            index_elements=[trial_counter.c.family],
            set_={"trials": trial_counter.c.trials + 1},
        )
        with self._engine.begin() as cx:
            cx.execute(stmt)
            count = cx.execute(
                select(trial_counter.c.trials).where(trial_counter.c.family == family)
            ).scalar_one()
        return int(count)
