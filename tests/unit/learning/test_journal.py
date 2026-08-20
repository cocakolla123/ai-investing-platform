"""Journal builders + similar-situation ranking (doc 10 §6)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import (
    Assessment,
    CalibrationCheck,
    FailureClass,
    JournalEntry,
    JournalKind,
    PostTradeReview,
)
from dynamo.domain.models.research import Action, CommitteeVerdict
from dynamo.learning.journal import (
    JournalFilters,
    incident_entry,
    lesson_entries,
    rank_similar,
    review_entry,
    thesis_entry,
)

NOW = datetime(2026, 8, 3, 9, 0, tzinfo=UTC)

VERDICT = CommitteeVerdict(
    id="v-1",
    symbol="AAPL",
    as_of=date(2026, 8, 1),
    action=Action.BUY,
    conviction=0.7,
    thesis="Services growth is underpriced.",
    bull_case="bull",
    bear_case="bear",
)

REVIEW = PostTradeReview(
    trade_id="t-1",
    decision_context_id="dc-1",
    symbol="AAPL",
    as_of=date(2026, 8, 3),
    prompt_version="v1",
    model_id="opus",
    outcome_pnl_pct=-3.0,
    holding_days=20,
    what_worked=("entry timing",),
    what_failed=("ignored margin dissent",),
    confidence_calibration=CalibrationCheck(conviction_was_justified=False, comment="high"),
    sizing_quality=Assessment.TOO_BIG,
    timing_quality=Assessment.LATE,
    thesis_remained_valid=False,
    exit_honored_exit_conditions=True,
    should_have_exited="earlier",
    failure_primary=FailureClass.CONFIDENCE_CALIBRATION,
    lessons=("Weight dissents on margins more.", "Cap adds into earnings."),
)


def _entry(eid: str, kind: JournalKind = JournalKind.LESSON, **kw: object) -> JournalEntry:
    defaults: dict[str, object] = {
        "id": eid,
        "kind": kind,
        "text": f"text {eid}",
        "created_at": NOW,
    }
    defaults.update(kw)
    return JournalEntry.model_validate(defaults)


def test_thesis_entry_from_verdict() -> None:
    entry = thesis_entry(VERDICT, NOW, regime_at_write="risk_on", sectors=("tech",))
    assert entry.kind == JournalKind.THESIS
    assert entry.id == "thesis:v-1"
    assert entry.symbols == ("AAPL",)
    assert "underpriced" in entry.text
    assert entry.refs["verdict_id"] == "v-1"


def test_review_entry_carries_failure_class_and_refs() -> None:
    entry = review_entry(REVIEW, NOW)
    assert entry.kind == JournalKind.REVIEW
    assert "confidence_calibration" in entry.text
    assert entry.refs == {"trade_id": "t-1", "decision_context_id": "dc-1"}


def test_lesson_entries_one_per_lesson() -> None:
    lessons = lesson_entries(REVIEW, NOW, strategy="earnings-momo")
    assert [e.id for e in lessons] == ["lesson:t-1:0", "lesson:t-1:1"]
    assert all(e.kind == JournalKind.LESSON for e in lessons)
    assert lessons[0].text == "Weight dissents on margins more."


def test_incident_entry() -> None:
    entry = incident_entry("inc-9", "Tiingo served stale closes.", NOW, symbols=("AAPL",))
    assert entry.kind == JournalKind.INCIDENT
    assert entry.id == "incident:inc-9"


def test_rank_similar_orders_by_boosted_score() -> None:
    same_sector = _entry("e1", sectors=("tech",), symbols=("MSFT",))
    plain = _entry("e2")
    ranked = rank_similar(
        [(plain, 0.80), (same_sector, 0.78)],
        JournalFilters(sectors=frozenset({"tech"})),
    )
    # 0.78 + sector boost (0.05) beats 0.80.
    assert [r.entry.id for r in ranked] == ["e1", "e2"]
    assert ranked[0].similarity == 0.78
    assert ranked[0].score > ranked[1].score


def test_rank_similar_kind_is_hard_filter() -> None:
    lesson = _entry("e1", kind=JournalKind.LESSON)
    incident = _entry("e2", kind=JournalKind.INCIDENT)
    ranked = rank_similar(
        [(lesson, 0.5), (incident, 0.99)],
        JournalFilters(kinds=frozenset({JournalKind.LESSON})),
    )
    assert [r.entry.id for r in ranked] == ["e1"]


def test_rank_similar_dedupes_and_truncates() -> None:
    entries = [(_entry(f"e{i}"), 0.1 * i) for i in range(8)]
    entries.append((_entry("e7"), 0.05))  # duplicate id, worse score
    ranked = rank_similar(entries, top_k=3)
    assert len(ranked) == 3
    assert [r.entry.id for r in ranked] == ["e7", "e6", "e5"]
    assert ranked[0].similarity == pytest.approx(0.7)  # best-scoring duplicate kept


def test_rank_similar_no_filters_pure_similarity() -> None:
    ranked = rank_similar([(_entry("a"), 0.2), (_entry("b"), 0.9)])
    assert [r.entry.id for r in ranked] == ["b", "a"]
