"""estimate_vintage_rows — the pure append-on-change core of the estimates
vintage fix (migration 0019). No database: latest-vintage state is injected."""

from __future__ import annotations

from datetime import UTC, date, datetime

from dynamo.domain.models.market import EarningsEvent
from dynamo.persistence.repositories.market import estimate_vintage_rows

FETCHED = datetime(2026, 8, 18, 7, 0, tzinfo=UTC)
EVENT_DATE = date(2026, 9, 1)


def _event(
    symbol: str = "AAPL",
    eps: float | None = 1.10,
    revenue: float | None = None,
    event_date: date = EVENT_DATE,
) -> EarningsEvent:
    return EarningsEvent(
        symbol=symbol, event_date=event_date, eps_est=eps, revenue_est=revenue
    )


def test_first_sighting_appends_a_vintage() -> None:
    rows = estimate_vintage_rows([_event()], {}, FETCHED)
    assert rows == [
        {
            "symbol": "AAPL",
            "event_date": EVENT_DATE,
            "fetched_at": FETCHED,
            "eps_est": 1.10,
            "revenue_est": None,
        }
    ]


def test_unchanged_estimate_appends_nothing() -> None:
    latest = {("AAPL", EVENT_DATE): (1.10, None)}
    assert estimate_vintage_rows([_event()], latest, FETCHED) == []


def test_changed_eps_or_revenue_appends_new_vintage() -> None:
    latest = {("AAPL", EVENT_DATE): (1.10, 5e9)}
    eps_change = estimate_vintage_rows([_event(eps=1.25, revenue=5e9)], latest, FETCHED)
    assert [r["eps_est"] for r in eps_change] == [1.25]
    rev_change = estimate_vintage_rows([_event(eps=1.10, revenue=6e9)], latest, FETCHED)
    assert [r["revenue_est"] for r in rev_change] == [6e9]


def test_events_without_any_estimate_are_skipped() -> None:
    assert estimate_vintage_rows([_event(eps=None, revenue=None)], {}, FETCHED) == []


def test_in_batch_duplicates_keep_last() -> None:
    rows = estimate_vintage_rows([_event(eps=1.0), _event(eps=2.0)], {}, FETCHED)
    assert [r["eps_est"] for r in rows] == [2.0]


def test_distinct_events_each_get_a_row() -> None:
    rows = estimate_vintage_rows(
        [
            _event(symbol="AAPL"),
            _event(symbol="MSFT", eps=3.3),
            _event(symbol="AAPL", event_date=date(2026, 12, 1), eps=1.5),
        ],
        {},
        FETCHED,
    )
    assert len(rows) == 3
