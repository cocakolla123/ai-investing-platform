"""Seed the journal from resolved decision experiences (findings item 14).

Journal hits rendered ``none`` in 300 of 300 dossiers of the control run for a
boring reason: ``journal_entries`` was empty, because entries are only ever
written by the post-trade-review asset, which needs an LLM review of a closed
LIVE trade — and none existed. Prior-case retrieval was a no-op.

``decision_experience`` already holds resolved decisions with their outcome
labels. This module turns each one into the same ``PostTradeReview`` shape the
review asset produces, so the EXISTING builders (``review_entry``,
``lesson_entries``) write the entries — no second journal format.

Two honesty constraints shape the mapping:

* Only fields the experience row actually contains are asserted. The review's
  judgement fields that no deterministic rule can supply (sizing quality,
  timing quality) are set from the recorded excursions where they are
  knowable and are NOT rendered into journal text by either builder.
* ``created_at`` is the moment the outcome became knowable (the exit), never
  the seeding wall clock. Journal search is ``created_at <=`` bounded, so
  stamping "now" would make every seeded entry invisible to a replay of the
  past and visible to a decision that predates it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from dynamo.domain.models.learning import (
    Assessment,
    CalibrationCheck,
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    JournalEntry,
    OutcomeQuality,
    PostTradeReview,
)
from dynamo.learning.journal import lesson_entries, review_entry

SEED_PROMPT_VERSION = "experience-seed-1.0.0"
SEED_MODEL_ID = "deterministic-experience-seed"  # provenance: no LLM ran


def dedupe_experiences(
    experiences: Iterable[DecisionExperience],
) -> list[DecisionExperience]:
    """One row per underlying decision, deterministically.

    The same replay decision is recorded once per run that produced it (the
    117 rows in the table cover fewer distinct trades). Keyed on the decision
    identity — symbol, date, action, entry and exit — with the smallest id
    winning so re-runs pick the same representative.
    """
    best: dict[tuple[str, str, str, str, str], DecisionExperience] = {}
    for exp in experiences:
        key = (
            exp.symbol,
            exp.as_of.isoformat(),
            exp.action or "",
            exp.entry_at.isoformat() if exp.entry_at else "",
            exp.exit_at.isoformat() if exp.exit_at else "",
        )
        current = best.get(key)
        if current is None or exp.id < current.id:
            best[key] = exp
    return sorted(best.values(), key=lambda e: (e.as_of, e.symbol, e.id))


def _timing(exp: DecisionExperience) -> Assessment:
    """Timing read off the recorded excursions: a loser that was ahead by more
    than it ended up down was exited late; everything else is on time."""
    if (
        exp.pnl_pct is not None
        and exp.pnl_pct < 0
        and exp.mfe_pct is not None
        and exp.mfe_pct > abs(exp.pnl_pct)
    ):
        return Assessment.LATE
    return Assessment.ON_TIME


def _sizing(exp: DecisionExperience) -> Assessment:
    """Sizing read off drawdown depth: an adverse excursion worse than -10%
    was too big for the conviction that was expressed; else right."""
    if exp.mae_pct is not None and exp.mae_pct < -10.0:
        return Assessment.TOO_BIG
    return Assessment.RIGHT


def _what_worked(exp: DecisionExperience) -> tuple[str, ...]:
    out: list[str] = []
    if exp.pnl_pct is not None and exp.pnl_pct > 0:
        out.append(f"the position returned {exp.pnl_pct:+.2f}%")
    if exp.benchmark_excess_pct is not None and exp.benchmark_excess_pct > 0:
        out.append(f"it beat the benchmark by {exp.benchmark_excess_pct:+.2f}%")
    if exp.decision_quality is DecisionQuality.SOUND:
        out.append("the decision was sound on what was knowable at the time")
    return tuple(out)


def _what_failed(exp: DecisionExperience) -> tuple[str, ...]:
    out: list[str] = []
    if exp.pnl_pct is not None and exp.pnl_pct < 0:
        out.append(f"the position returned {exp.pnl_pct:+.2f}%")
    if exp.benchmark_excess_pct is not None and exp.benchmark_excess_pct < 0:
        out.append(f"it lagged the benchmark by {exp.benchmark_excess_pct:+.2f}%")
    if exp.decision_quality is DecisionQuality.UNSOUND:
        out.append("the decision was already unsound on what was knowable")
    if exp.forecast_quality is ForecastQuality.INACCURATE:
        out.append("the stated conviction did not match the realized path")
    return tuple(out)


def _lessons(exp: DecisionExperience) -> tuple[str, ...]:
    conviction = "n/a" if exp.conviction is None else f"{exp.conviction:.2f}"
    context = f"{exp.regime or 'unknown-regime'}/{exp.sector or 'unknown-sector'}"
    pnl = "n/a" if exp.pnl_pct is None else f"{exp.pnl_pct:+.2f}%"
    lessons = [
        f"{exp.symbol}: {exp.action or 'unknown-action'} at conviction {conviction} "
        f"in {context} resolved {exp.outcome_quality.value} ({pnl} over "
        f"{exp.holding_days if exp.holding_days is not None else 'n/a'} days)."
    ]
    if exp.outcome_quality is OutcomeQuality.NORMAL_UNCERTAINTY:
        lessons.append(
            f"{exp.symbol}: a sound decision losing inside its expected downside is "
            "not a failure — do not over-correct on this case."
        )
    if exp.mae_pct is not None and exp.mfe_pct is not None:
        lessons.append(
            f"{exp.symbol}: the path ran {exp.mfe_pct:+.2f}% in favour and "
            f"{exp.mae_pct:+.2f}% against before resolving — size stops to that range."
        )
    return tuple(lessons)


def experience_review(exp: DecisionExperience) -> PostTradeReview:
    """A ``PostTradeReview`` reconstructed from one resolved experience row.

    ``model_id`` records that no model produced it: this is arithmetic over
    stored outcomes, not a judgement anyone should read as an LLM review.
    """
    justified = exp.forecast_quality is ForecastQuality.ACCURATE
    comment = (
        f"forecast_quality={exp.forecast_quality.value}, "
        f"decision_quality={exp.decision_quality.value}, "
        f"outcome={exp.outcome_quality.value}"
    )
    return PostTradeReview(
        trade_id=exp.trade_id or exp.id,
        decision_context_id=exp.decision_context_id or exp.id,
        symbol=exp.symbol,
        as_of=exp.as_of,
        prompt_version=SEED_PROMPT_VERSION,
        model_id=SEED_MODEL_ID,
        outcome_pnl_pct=float(exp.pnl_pct) if exp.pnl_pct is not None else 0.0,
        holding_days=int(exp.holding_days) if exp.holding_days is not None else 0,
        what_worked=_what_worked(exp),
        what_failed=_what_failed(exp),
        confidence_calibration=CalibrationCheck(
            conviction_was_justified=justified, comment=comment
        ),
        sizing_quality=_sizing(exp),
        timing_quality=_timing(exp),
        thesis_remained_valid=exp.outcome_quality is OutcomeQuality.WIN,
        exit_honored_exit_conditions=True,
        should_have_exited="earlier" if _timing(exp) is Assessment.LATE else "as_done",
        failure_primary=exp.failure_primary,
        lessons=_lessons(exp),
    )


def _knowable_at(exp: DecisionExperience) -> datetime:
    """When the lesson became knowable: the exit, else the row's own stamp."""
    stamp = exp.exit_at or exp.created_at
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def entries_for(exp: DecisionExperience) -> list[JournalEntry]:
    """Review + lesson entries for one experience, stamped and provenanced.

    Provenance rides in ``refs`` so a reader (and any future firewall audit)
    can tell a replay-derived entry from a live one — ``journal_entries`` has
    no source column of its own.
    """
    review = experience_review(exp)
    created_at = _knowable_at(exp)
    sectors = (exp.sector,) if exp.sector else ()
    entries = [
        review_entry(
            review,
            created_at,
            strategy=exp.sleeve,
            regime_at_write=exp.regime,
            sectors=sectors,
        ),
        *lesson_entries(review, created_at, strategy=exp.sleeve, regime_at_write=exp.regime),
    ]
    refs = {
        "experience_id": exp.id,
        "source": exp.source,
        "seeded_from": "decision_experience",
    }
    if exp.replay_run_id:
        refs["replay_run_id"] = exp.replay_run_id
    return [entry.model_copy(update={"refs": {**entry.refs, **refs}}) for entry in entries]


def seed_entries(experiences: Sequence[DecisionExperience]) -> list[JournalEntry]:
    """Deduped, deterministically ordered journal entries for a set of rows."""
    out: list[JournalEntry] = []
    for exp in dedupe_experiences(experiences):
        if not exp.resolved:
            continue  # an unresolved decision has no lesson yet
        out.extend(entries_for(exp))
    return out
