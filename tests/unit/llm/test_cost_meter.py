"""Cost meter: ledger rows, month-to-date math, budget kill-switch."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dynamo.llm.cost_meter import BudgetExceededError, CostMeter, InMemoryLedgerWriter

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)


def make_meter(budget: float = 90.0) -> tuple[CostMeter, InMemoryLedgerWriter]:
    writer = InMemoryLedgerWriter()
    return CostMeter(writer, budget, now_fn=lambda: NOW), writer


class TestLedger:
    def test_record_writes_row(self) -> None:
        meter, writer = make_meter()
        meter.record(
            model_id="claude-sonnet-4-6",
            call_kind="dossier",
            input_tokens=14_000,
            output_tokens=3_000,
            cache_read_tokens=2_000,
            cost_usd=0.05,
            batch=True,
        )
        assert len(writer.entries) == 1
        entry = writer.entries[0]
        assert entry.model_id == "claude-sonnet-4-6"
        assert entry.call_kind == "dossier"
        assert entry.input_tokens == 14_000
        assert entry.cache_read_tokens == 2_000
        assert entry.batch is True
        assert entry.ts == NOW

    def test_record_usage_computes_cost(self) -> None:
        meter, writer = make_meter()
        usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=0)
        meter.record_usage(model_id="claude-haiku-4-5", call_kind="triage", usage=usage, batch=True)
        assert writer.entries[0].cost_usd == pytest.approx(0.5)  # $1/MTok halved

    def test_month_to_date_excludes_prior_months(self) -> None:
        meter, writer = make_meter()
        meter.record(model_id="m", call_kind="k", input_tokens=1, output_tokens=1, cost_usd=2.0)
        # simulate a July row
        july = writer.entries[0]
        writer.entries.append(
            type(july)(
                ts=datetime(2026, 7, 10, tzinfo=UTC),
                model_id="m",
                call_kind="k",
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cost_usd=50.0,
                batch=False,
            )
        )
        assert meter.month_to_date_usd() == pytest.approx(2.0)


class TestKillSwitch:
    def test_under_budget_passes(self) -> None:
        meter, _ = make_meter(budget=10.0)
        meter.record(model_id="m", call_kind="k", input_tokens=1, output_tokens=1, cost_usd=9.99)
        meter.check_budget()  # no raise

    def test_at_or_over_budget_raises(self) -> None:
        meter, _ = make_meter(budget=10.0)
        meter.record(model_id="m", call_kind="k", input_tokens=1, output_tokens=1, cost_usd=10.0)
        with pytest.raises(BudgetExceededError):
            meter.check_budget()

    def test_budget_utilization(self) -> None:
        meter, _ = make_meter(budget=90.0)
        meter.record(model_id="m", call_kind="k", input_tokens=1, output_tokens=1, cost_usd=45.0)
        assert meter.budget_utilization() == pytest.approx(0.5)
