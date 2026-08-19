"""Seeding the journal from resolved experiences (findings item 14)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.learning import (
    Assessment,
    DecisionExperience,
    DecisionQuality,
    ForecastQuality,
    JournalKind,
    OutcomeQuality,
)
from dynamo.learning.journal_seed import (
    SEED_MODEL_ID,
    dedupe_experiences,
    entries_for,
    experience_review,
    seed_entries,
)

ENTRY = datetime(2026, 7, 2, tzinfo=UTC)
EXIT = datetime(2026, 7, 31, tzinfo=UTC)
CREATED = datetime(2026, 8, 15, tzinfo=UTC)  # seeding-time stamp, must NOT be used


def experience(**overrides: object) -> DecisionExperience:
    base: dict[str, object] = {
        "id": "exp:replay-abc:KO-2026-07-02",
        "source": "replay",
        "replay_run_id": "replay-abc",
        "created_at": CREATED,
        "symbol": "KO",
        "as_of": date(2026, 7, 1),
        "sector": "Consumer Staples",
        "regime": "risk_on",
        "action": "buy",
        "conviction": 0.85,
        "entry_at": ENTRY,
        "exit_at": EXIT,
        "holding_days": 29,
        "resolved": True,
        "pnl_pct": 7.3,
        "benchmark_excess_pct": 7.0,
        "mae_pct": -0.98,
        "mfe_pct": 11.38,
        "outcome_quality": OutcomeQuality.WIN,
        "decision_quality": DecisionQuality.SOUND,
        "forecast_quality": ForecastQuality.ACCURATE,
    }
    base.update(overrides)
    return DecisionExperience.model_validate(base)


def test_entries_come_from_the_existing_builders() -> None:
    entries = entries_for(experience())
    kinds = [e.kind for e in entries]
    assert kinds[0] is JournalKind.REVIEW
    assert set(kinds[1:]) == {JournalKind.LESSON}
    review = entries[0]
    assert review.id.startswith("review:")
    assert "Post-trade review KO: pnl +7.30% over 29d." in review.text
    assert "the position returned +7.30%" in review.text
    assert review.symbols == ("KO",)
    assert review.sectors == ("Consumer Staples",)
    assert review.regime_at_write == "risk_on"


def test_entries_are_stamped_when_the_outcome_became_knowable() -> None:
    """Journal search is created_at-bounded: stamping the seeding wall clock
    would hide every entry from a replay of the period it describes."""
    for entry in entries_for(experience()):
        assert entry.created_at == EXIT
        assert entry.created_at != CREATED


def test_provenance_is_recorded_in_refs() -> None:
    for entry in entries_for(experience()):
        assert entry.refs["experience_id"] == "exp:replay-abc:KO-2026-07-02"
        assert entry.refs["source"] == "replay"
        assert entry.refs["replay_run_id"] == "replay-abc"
        assert entry.refs["seeded_from"] == "decision_experience"


def test_review_is_marked_as_not_model_generated() -> None:
    review = experience_review(experience())
    assert review.model_id == SEED_MODEL_ID
    assert review.confidence_calibration.conviction_was_justified is True
    assert "outcome=win" in review.confidence_calibration.comment


def test_losing_trade_records_what_failed_and_late_timing() -> None:
    review = experience_review(
        experience(
            pnl_pct=-4.0,
            benchmark_excess_pct=-2.0,
            mfe_pct=6.0,
            mae_pct=-12.0,
            outcome_quality=OutcomeQuality.LOSS,
            forecast_quality=ForecastQuality.INACCURATE,
        )
    )
    assert "the position returned -4.00%" in review.what_failed
    assert "the stated conviction did not match the realized path" in review.what_failed
    assert review.timing_quality is Assessment.LATE  # gave back a +6% excursion
    assert review.sizing_quality is Assessment.TOO_BIG  # -12% drawdown
    assert review.should_have_exited == "earlier"


def test_normal_uncertainty_gets_its_own_lesson() -> None:
    review = experience_review(
        experience(pnl_pct=-0.5, outcome_quality=OutcomeQuality.NORMAL_UNCERTAINTY)
    )
    assert any("do not over-correct" in lesson for lesson in review.lessons)


def test_duplicate_rows_from_multiple_runs_collapse_to_one() -> None:
    rows = [
        experience(id="exp:replay-b:KO-2026-07-02", replay_run_id="replay-b"),
        experience(id="exp:replay-a:KO-2026-07-02", replay_run_id="replay-a"),
    ]
    assert [e.id for e in dedupe_experiences(rows)] == ["exp:replay-a:KO-2026-07-02"]
    other = experience(id="exp:replay-a:JPM-2026-07-02", symbol="JPM")
    assert len(dedupe_experiences([*rows, other])) == 2


def test_seeding_is_deterministic_and_idempotent_by_id() -> None:
    rows = [experience(), experience(id="exp:replay-abc:JPM-2026-07-02", symbol="JPM")]
    first = seed_entries(rows)
    second = seed_entries(list(reversed(rows)))
    assert [e.id for e in first] == [e.id for e in second]
    assert [e.text for e in first] == [e.text for e in second]
    # Deterministic ids are what makes the insert (conflict-do-nothing) a
    # no-op on a re-run.
    assert len({e.id for e in first}) == len(first)


def test_unresolved_experiences_are_skipped() -> None:
    assert seed_entries([experience(resolved=False)]) == []
