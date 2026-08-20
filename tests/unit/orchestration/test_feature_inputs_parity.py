"""The nightly ``FeatureInputs`` builder and the replay gateway agree.

Four features (``news_count_60d``, ``unusual_volume_no_news``,
``short_interest_pct_float``, ``days_to_cover``, ``short_interest_change``)
trained and backtested fine and returned NaN every night, because
``orchestration/assets/research.py::_feature_inputs`` builds its own
``FeatureInputs`` and simply never read ``news`` or ``short_interest``. That
is a train/serve skew a feature test cannot see: both sides are individually
correct.

These tests make the skew impossible to reintroduce silently:

* the two builders must construct ``FeatureInputs`` with the SAME keyword set
  (AST comparison — no database, no network);
* the lookback windows must be numerically equal;
* the short-interest read must bound on ``publication_date``, never
  ``settlement_date`` (behavioural, on an in-memory SQLite table).

``tests/integration/test_feature_inputs_live_parity.py`` carries the
end-to-end version against real Postgres.
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import orchestration.assets.research as research_mod
import pandas as pd
import pytest

AS_OF = date(2026, 6, 16)


def _feature_inputs_kwargs(func: Any) -> set[str]:
    """Keyword names of the ``FeatureInputs(...)`` call inside ``func``."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FeatureInputs"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no FeatureInputs(...) call found in {func!r}")


def test_live_builder_and_gateway_populate_the_same_fields() -> None:
    from dynamo.replay.gateway import HistoricalDataGateway

    live = _feature_inputs_kwargs(research_mod._feature_inputs)
    gateway = _feature_inputs_kwargs(HistoricalDataGateway.feature_inputs)
    assert live == gateway, (
        "the nightly feature builder and the replay gateway disagree on which "
        f"FeatureInputs fields they populate: live-only={sorted(live - gateway)}, "
        f"gateway-only={sorted(gateway - live)}. A field only the gateway fills "
        "is a feature that works in training and is NaN in production."
    )
    # Belt and braces: the fields those four dark features actually need.
    assert {"news", "short_interest", "estimates"} <= live


def test_lookback_windows_match_the_gateway() -> None:
    """Same catalog, same windows — a differing window is train/serve skew."""
    from dynamo.replay import gateway as gw

    assert research_mod.FEATURE_BARS_LOOKBACK_DAYS == gw.FEATURE_BARS_LOOKBACK_DAYS
    assert research_mod.FEATURE_FUNDAMENTALS_LOOKBACK_DAYS == gw.FUNDAMENTALS_LOOKBACK_DAYS
    assert research_mod.FEATURE_INSIDER_LOOKBACK_DAYS == gw.INSIDER_LOOKBACK_DAYS
    assert research_mod.FEATURE_MACRO_LOOKBACK_DAYS == gw.MACRO_LOOKBACK_DAYS
    assert research_mod.FEATURE_NEWS_LOOKBACK_DAYS == gw.FEATURE_NEWS_LOOKBACK_DAYS
    assert (
        research_mod.FEATURE_SHORT_INTEREST_LOOKBACK_DAYS
        == gw.FEATURE_SHORT_INTEREST_LOOKBACK_DAYS
    )
    assert research_mod.FEATURE_EARNINGS_LOOKAHEAD_DAYS == gw.EARNINGS_LOOKAHEAD_DAYS


# --- the short-interest bound, behaviourally -------------------------------


@pytest.fixture()
def short_interest_engine() -> Any:
    """SQLite carrying ONLY ``short_interest`` — enough to execute the real
    statement the live builder emits. Three cycles: one published well before
    ``as_of``, one settled before but published AFTER (the look-ahead trap),
    one settled and published after."""
    from sqlalchemy import create_engine

    from dynamo.persistence import schema

    engine = create_engine("sqlite://")
    schema.short_interest.create(engine)
    rows = [
        # settled 8 days before publication, both before as_of — visible.
        {
            "symbol": "ACME",
            "settlement_date": date(2026, 5, 29),
            "publication_date": date(2026, 6, 8),
            "shares_short": 1_000.0,
        },
        # THE TRAP: settled BEFORE as_of, disseminated by FINRA AFTER it. A
        # read bounded on settlement_date sees this; the publication bound
        # must not.
        {
            "symbol": "ACME",
            "settlement_date": date(2026, 6, 15),
            "publication_date": date(2026, 6, 24),
            "shares_short": 2_000.0,
        },
        # Comfortably in the future on both axes.
        {
            "symbol": "ACME",
            "settlement_date": date(2026, 6, 30),
            "publication_date": date(2026, 7, 9),
            "shares_short": 3_000.0,
        },
    ]
    with engine.begin() as cx:
        cx.execute(schema.short_interest.insert(), rows)
    return engine


def _capture_statements(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every statement ``_feature_inputs`` emits; return empty frames."""
    seen: list[Any] = []

    def fake_read_df(dynamo: Any, stmt: Any) -> pd.DataFrame:
        seen.append(stmt)
        return pd.DataFrame({c.key: pd.Series(dtype="object") for c in stmt.selected_columns})

    monkeypatch.setattr(research_mod, "_read_df", fake_read_df)
    return seen


class _Dynamo:
    def bars_window(self, symbols: Any, as_of: date, days: int) -> pd.DataFrame:
        return pd.DataFrame(columns=pd.Index(["symbol", "ts", "close", "volume"]))


def _short_interest_statement(monkeypatch: pytest.MonkeyPatch) -> Any:
    seen = _capture_statements(monkeypatch)
    research_mod._feature_inputs(_Dynamo(), ["ACME"], AS_OF)  # type: ignore[arg-type]
    matches = [
        s
        for s in seen
        if {"shares_short", "publication_date"} <= {c.key for c in s.selected_columns}
    ]
    assert len(matches) == 1, "expected exactly one short-interest read"
    return matches[0]


def test_short_interest_is_bounded_on_publication_date(
    monkeypatch: pytest.MonkeyPatch, short_interest_engine: Any
) -> None:
    """A cycle FINRA published after ``as_of`` is invisible, even though its
    settlement date precedes ``as_of`` by a day."""
    stmt = _short_interest_statement(monkeypatch)
    with short_interest_engine.connect() as cx:
        frame = pd.read_sql(stmt, cx)
    assert sorted(frame["publication_date"].astype(str)) == ["2026-06-08"]
    assert 2_000.0 not in set(frame["shares_short"])


def test_short_interest_read_does_not_filter_on_settlement_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settlement date is a description, not a knowledge time. It is
    selected (the catalog dedupes cycles with it) but never bounded on."""
    from sqlalchemy.dialects import postgresql

    stmt = _short_interest_statement(monkeypatch)
    where = str(
        stmt.whereclause.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "publication_date" in where
    assert "settlement_date" not in where


def test_news_read_uses_the_full_year_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """``unusual_volume_no_news`` needs a year of headlines to tell "quiet"
    from "not covered by the feed at all"."""
    from sqlalchemy.dialects import postgresql

    seen = _capture_statements(monkeypatch)
    research_mod._feature_inputs(_Dynamo(), ["ACME"], AS_OF)  # type: ignore[arg-type]
    news = [s for s in seen if {c.key for c in s.selected_columns} == {"symbols", "published_at"}]
    assert len(news) == 1
    where = str(
        news[0].whereclause.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    floor = datetime.combine(
        AS_OF - timedelta(days=research_mod.FEATURE_NEWS_LOOKBACK_DAYS), time.min, tzinfo=UTC
    )
    assert floor.date().isoformat() in where
