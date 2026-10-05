"""Journal metadata derivation (docs/journal_metadata.md): point in time."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.learning import PostTradeReview
from dynamo.learning.journal import lesson_entries
from dynamo.learning.journal_meta import entry_sectors, regime_on_or_before

HISTORY = [(date(2026, 3, 2), "risk_on"), (date(2026, 3, 9), "neutral")]


def test_regime_is_the_last_session_on_or_before_the_date() -> None:
    assert regime_on_or_before(HISTORY, date(2026, 3, 2)) == "risk_on"
    assert regime_on_or_before(HISTORY, date(2026, 3, 8)) == "risk_on"
    assert regime_on_or_before(HISTORY, date(2026, 3, 9)) == "neutral"


def test_regime_never_uses_a_later_session() -> None:
    assert regime_on_or_before(HISTORY, date(2026, 3, 1)) is None
    assert regime_on_or_before([], date(2026, 3, 1)) is None


def test_sectors_are_distinct_sorted_and_skip_unclassified() -> None:
    m = {"A": "Tech", "B": "Energy", "C": None, "D": "unknown", "E": "Tech"}
    assert entry_sectors(["A", "B", "C", "D", "E", "Z"], m) == ("Energy", "Tech")


def test_lesson_entries_carry_sectors() -> None:
    review = PostTradeReview.model_construct(
        trade_id="t1", symbol="KO", lessons=("a", "b"), decision_context_id="dc1"
    )
    entries = lesson_entries(
        review,
        datetime(2026, 3, 9, tzinfo=UTC),
        strategy="long_term",
        regime_at_write="neutral",
        sectors=("Consumer Staples",),
    )
    assert [e.sectors for e in entries] == [("Consumer Staples",)] * 2
    assert {(e.strategy, e.regime_at_write) for e in entries} == {("long_term", "neutral")}
