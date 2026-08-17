"""DecisionExperience repository (plan Phase 1) — the ``ExperienceStore`` port.

FIREWALL DISCIPLINE (plan Phase 1, non-negotiable): live learning consumers
read ONLY through :meth:`ExperienceRepository.resolved_live` /
:meth:`ExperienceRepository.unresolved_live` /
:meth:`ExperienceRepository.similar`, which filter ``source='live'``
by design — replay rows can never leak into live calibration or training.
:meth:`ExperienceRepository.by_run` is the SOLE replay read path, keyed by an
explicit ``replay_run_id`` (opt-in, labeled). Any future reader added here
must pick one of those two sides; there is no unfiltered read method on
purpose.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.domain.models.learning import (
    DecisionExperience,
    DecisionQuality,
    FailureClass,
    ForecastQuality,
    OutcomeQuality,
)
from dynamo.persistence.schema import decision_experience


class ExperienceRepository:
    """See the module docstring for the live/replay firewall contract."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def insert(self, exp: DecisionExperience) -> bool:
        """Idempotent insert; returns True iff a row was actually written.

        Live rows conflict on the partial unique index
        ``decision_context_id WHERE source='live'`` (one experience per
        decision context — ON CONFLICT DO NOTHING); replay rows conflict on
        their deterministic primary key.
        """
        stmt = pg_insert(decision_experience).values(**_to_row(exp))
        if exp.source == "live" and exp.decision_context_id is not None:
            stmt = stmt.on_conflict_do_nothing(
                index_elements=[decision_experience.c.decision_context_id],
                index_where=decision_experience.c.source == "live",
            )
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=[decision_experience.c.id])
        # RETURNING yields a row ONLY when the insert actually happened —
        # driver rowcount is unreliable for ON CONFLICT DO NOTHING.
        with self._engine.begin() as cx:
            inserted = cx.execute(stmt.returning(decision_experience.c.id)).first()
        return inserted is not None

    def resolved_live(self, since: date | None = None) -> list[DecisionExperience]:
        """Resolved LIVE experiences (the learning-loop feedstock)."""
        stmt = select(decision_experience).where(
            decision_experience.c.source == "live",
            decision_experience.c.resolved.is_(True),
        )
        if since is not None:
            stmt = stmt.where(decision_experience.c.as_of >= since)
        return self._rows(stmt.order_by(decision_experience.c.created_at))

    def unresolved_live(self) -> list[DecisionExperience]:
        """LIVE experiences still awaiting resolution."""
        stmt = select(decision_experience).where(
            decision_experience.c.source == "live",
            decision_experience.c.resolved.is_(False),
        )
        return self._rows(stmt.order_by(decision_experience.c.created_at))

    def similar(
        self,
        *,
        symbol: str,
        as_of: date,
        sector: str | None = None,
        regime: str | None = None,
        sleeve: str | None = None,
        limit: int = 3,
    ) -> list[DecisionExperience]:
        """Prior similar LIVE cases for dossier evidence (plan Phase 2).

        Hard filters: ``source='live'``, resolved, and STRICTLY ``as_of <``
        the query date — future experiences are invisible by construction
        (replay-safe point-in-time). Candidates are fetched recency-first
        SQL-side, then ranked in Python by :func:`similarity_score` (recency
        decay + same-symbol/sector/regime/sleeve boosts). Experiences are
        structured rows, so this needs no embeddings — a vector rank over
        journal-style text can layer on later without changing callers.
        """
        candidate_cap = max(limit * 50, 200)
        stmt = (
            select(decision_experience)
            .where(
                decision_experience.c.source == "live",
                decision_experience.c.resolved.is_(True),
                decision_experience.c.as_of < as_of,
            )
            .order_by(decision_experience.c.as_of.desc(), decision_experience.c.id)
            .limit(candidate_cap)
        )
        candidates = self._rows(stmt)
        ranked = sorted(
            candidates,
            key=lambda e: (
                -similarity_score(
                    e, symbol=symbol, sector=sector, regime=regime, sleeve=sleeve, as_of=as_of
                ),
                e.id,
            ),
        )
        return ranked[: max(limit, 0)]

    def update_calibration(self, experience_id: str, probability: float, scope_used: str) -> None:
        """Log the advisory calibrated probability (plan Phase 2). Nothing
        reads it for sizing — surfacing/reporting only."""
        with self._engine.begin() as cx:
            cx.execute(
                update(decision_experience)
                .where(decision_experience.c.id == experience_id)
                .values(
                    calibrated_probability=float(probability),
                    calibration_scope_used=scope_used,
                )
            )

    def by_run(self, replay_run_id: str) -> list[DecisionExperience]:
        """The ONLY replay read path — explicit run id, opt-in by construction."""
        stmt = select(decision_experience).where(
            decision_experience.c.replay_run_id == replay_run_id
        )
        return self._rows(stmt.order_by(decision_experience.c.created_at))

    def count(self, source: str | None = None) -> int:
        stmt = select(func.count()).select_from(decision_experience)
        if source is not None:
            stmt = stmt.where(decision_experience.c.source == source)
        with self._engine.connect() as cx:
            return int(cx.execute(stmt).scalar() or 0)

    def live_decision_context_ids(self) -> frozenset[str]:
        """Decision contexts that already have a live experience row — the
        resolution asset's cheap idempotency pre-check."""
        stmt = select(decision_experience.c.decision_context_id).where(
            decision_experience.c.source == "live",
            decision_experience.c.decision_context_id.is_not(None),
        )
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return frozenset(str(r.decision_context_id) for r in rows)

    def _rows(self, stmt: Any) -> list[DecisionExperience]:
        with self._engine.connect() as cx:
            rows = cx.execute(stmt).all()
        return [_from_row(r) for r in rows]


RECENCY_HALF_LIFE_DAYS = 90.0
SYMBOL_BOOST = 1.0
SECTOR_BOOST = 0.5
REGIME_BOOST = 0.5
SLEEVE_BOOST = 0.25


def similarity_score(
    exp: DecisionExperience,
    *,
    symbol: str,
    as_of: date,
    sector: str | None = None,
    regime: str | None = None,
    sleeve: str | None = None,
) -> float:
    """Structured similarity for prior-case ranking: exponential recency decay
    in (0, 1] plus additive boosts for matching symbol / sector / regime /
    sleeve. Pure — unit-testable without a database. A vector rank over
    text can replace this later without changing :meth:`similar`'s shape."""
    age_days = max((as_of - exp.as_of).days, 0)
    score = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    if exp.symbol == symbol:
        score += SYMBOL_BOOST
    if sector is not None and exp.sector == sector:
        score += SECTOR_BOOST
    if regime is not None and exp.regime == regime:
        score += REGIME_BOOST
    if sleeve is not None and exp.sleeve == sleeve:
        score += SLEEVE_BOOST
    return score


def _to_row(exp: DecisionExperience) -> dict[str, Any]:
    row = exp.model_dump(mode="json")
    # Keep native datetime/date objects for the driver (model_dump(json)
    # stringifies them); enums become their .value strings above.
    row["created_at"] = exp.created_at
    row["as_of"] = exp.as_of
    row["entry_at"] = exp.entry_at
    row["exit_at"] = exp.exit_at
    return row


def _from_row(r: Any) -> DecisionExperience:
    m = r._mapping
    return DecisionExperience(
        id=m["id"],
        decision_context_id=m["decision_context_id"],
        trade_id=m["trade_id"],
        source=m["source"],
        replay_run_id=m["replay_run_id"],
        created_at=m["created_at"],
        symbol=m["symbol"],
        as_of=m["as_of"],
        sleeve=m["sleeve"],
        regime=m["regime"],
        sector=m["sector"],
        vix=m["vix"],
        quant_expected_excess_return=m["quant_expected_excess_return"],
        quant_downside_prob=m["quant_downside_prob"],
        quant_vol_forecast=m["quant_vol_forecast"],
        quant_confidence=m["quant_confidence"],
        action=m["action"],
        conviction=m["conviction"],
        disagreement_score=m["disagreement_score"],
        quant_alignment=m["quant_alignment"],
        expected_upside_pct=m["expected_upside_pct"],
        expected_downside_pct=m["expected_downside_pct"],
        horizon_days=m["horizon_days"],
        entry_at=m["entry_at"],
        exit_at=m["exit_at"],
        entry_price=m["entry_price"],
        exit_price=m["exit_price"],
        qty=m["qty"],
        holding_days=m["holding_days"],
        equity_at_decision=m["equity_at_decision"],
        position_notional=m["position_notional"],
        resolved=m["resolved"],
        pnl_pct=m["pnl_pct"],
        benchmark_excess_pct=m["benchmark_excess_pct"],
        mae_pct=m["mae_pct"],
        mfe_pct=m["mfe_pct"],
        target_achieved=m["target_achieved"],
        forecast_quality=ForecastQuality(m["forecast_quality"]),
        decision_quality=DecisionQuality(m["decision_quality"]),
        outcome_quality=OutcomeQuality(m["outcome_quality"]),
        cf_hold_pnl_pct=m["cf_hold_pnl_pct"],
        cf_half_size_pnl_pct=m["cf_half_size_pnl_pct"],
        cf_delayed_entry_pnl_pct=m["cf_delayed_entry_pnl_pct"],
        cf_hold_to_horizon_pnl_pct=m["cf_hold_to_horizon_pnl_pct"],
        cf_obey_quant_veto_pnl_pct=m["cf_obey_quant_veto_pnl_pct"],
        ev_per_decision=m["ev_per_decision"],
        calibrated_probability=m["calibrated_probability"],
        calibration_scope_used=m["calibration_scope_used"],
        failure_primary=FailureClass(m["failure_primary"]) if m["failure_primary"] else None,
        label=m["label"],
    )
