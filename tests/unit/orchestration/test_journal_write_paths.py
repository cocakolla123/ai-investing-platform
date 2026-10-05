"""Nightly journal write paths fill metadata (docs/journal_metadata.md)."""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import orchestration.assets.learning_ops as learning_ops
import orchestration.assets.research as research
import pytest

WRITTEN = datetime(2026, 10, 5, 3, 30, tzinfo=UTC)


def test_regime_lookup_is_bounded_by_the_write_date(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_read_df(dynamo: Any, stmt: Any) -> Any:
        import pandas as pd

        seen.append(str(stmt.compile(compile_kwargs={"literal_binds": True})))
        return pd.DataFrame({"regime": ["neutral"]})

    monkeypatch.setattr(research, "_read_df", fake_read_df)
    assert research._regime_at(SimpleNamespace(), date(2026, 10, 5))  # type: ignore[arg-type] == "neutral"
    sql = seen[0].replace("\n", " ")
    assert "regime_states.as_of <= '2026-10-05'" in sql
    assert "ORDER BY regime_states.as_of DESC" in sql


def test_journal_meta_fills_all_three_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        research, "_regime_at", lambda d, day: calls.setdefault("day", day) and "risk_on"
    )
    monkeypatch.setattr(research, "_sectors_for", lambda d, syms: {"KO": "Consumer Staples"})
    meta = research._journal_meta(SimpleNamespace(), "KO", WRITTEN, strategy="long_term")  # type: ignore[arg-type]
    assert meta == {
        "strategy": "long_term",
        "regime_at_write": "risk_on",
        "sectors": ("Consumer Staples",),
    }
    assert calls["day"] == date(2026, 10, 5)  # the write date, not later


def test_thesis_path_writes_regime_and_sectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from dynamo.domain.models.research import Action, CommitteeVerdict

    written: list[Any] = []
    monkeypatch.setattr(research, "utcnow", lambda: WRITTEN)
    monkeypatch.setattr(research, "_regime_at", lambda d, day: "risk_on")
    monkeypatch.setattr(research, "_sectors_for", lambda d, syms: {"KO": "Consumer Staples"})
    monkeypatch.setattr(research, "_write_journal_entries", lambda d, es: written.extend(es))
    dynamo = SimpleNamespace(market_repo=lambda: SimpleNamespace(upsert_rows=lambda *a: 1))
    verdict = CommitteeVerdict(
        id="v1",
        symbol="KO",
        as_of=date(2026, 10, 5),
        action=Action.BUY,
        conviction=0.6,
        thesis="t",
        bull_case="b",
        bear_case="r",
        expected_upside_pct=5.0,
        expected_downside_pct=3.0,
        horizon_days=20,
        prompt_version="p",
        model_id="m",
    )
    research._persist_verdict(dynamo, verdict)  # type: ignore[arg-type]
    (entry,) = written
    assert entry.regime_at_write == "risk_on"
    assert entry.sectors == ("Consumer Staples",)
    assert entry.strategy is None  # sleeve not known when the thesis is written


def test_review_path_routes_both_builders_through_journal_meta() -> None:
    """post_trade_reviews needs an LLM to run end to end; this checks its
    source passes the shared metadata (with the trade's sleeve) to both the
    review and the lesson builder."""
    module = ast.parse(inspect.getsource(learning_ops))
    tree = next(
        n
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef) and n.name == "post_trade_reviews"
    )
    meta_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_journal_meta"
    ]
    assert meta_calls and any(k.arg == "strategy" for k in meta_calls[0].keywords)
    builders = {
        getattr(n.func, "id", ""): n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") in {"review_entry", "lesson_entries"}
    }
    for name in ("review_entry", "lesson_entries"):
        assert any(k.arg is None for k in builders[name].keywords), f"{name} lacks **meta"
