"""Cost meter + monthly kill-switch. Design: doc 03 section 5.

Every Claude call is recorded to the cost_ledger via a pluggable writer.
`check_budget()` refuses new submissions once month-to-date spend reaches
`CLAUDE_MONTHLY_BUDGET_USD`; alert (log) at 70% utilization.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from prometheus_client import Counter, Gauge
from sqlalchemy import text
from sqlalchemy.engine import Engine

from dynamo.llm.pricing import cost_from_usage

log = logging.getLogger(__name__)

# Exposed as claude_cost_usd_total (prometheus_client appends _total to counters).
CLAUDE_COST_USD = Counter("claude_cost_usd", "Cumulative Claude API spend in USD")
CLAUDE_BUDGET_UTILIZATION = Gauge(
    "claude_budget_utilization", "Month-to-date Claude spend / monthly budget"
)

_ALERT_UTILIZATION = 0.7


class BudgetExceededError(RuntimeError):
    """Monthly Claude budget reached — no new submissions allowed."""


@dataclass(frozen=True)
class LedgerEntry:
    ts: datetime
    model_id: str
    call_kind: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    batch: bool


class LedgerWriter(Protocol):
    def write(self, entry: LedgerEntry) -> None: ...

    def month_to_date_usd(self, month_start: datetime) -> float: ...


class InMemoryLedgerWriter:
    """Test double: keeps ledger rows in a list."""

    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def write(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def month_to_date_usd(self, month_start: datetime) -> float:
        return sum(e.cost_usd for e in self.entries if e.ts >= month_start)


class SqlAlchemyLedgerWriter:
    """Writes to the cost_ledger table (schema.py owns the DDL)."""

    _INSERT = text(
        "INSERT INTO cost_ledger "
        "(ts, model_id, call_kind, input_tokens, output_tokens, cache_read_tokens, "
        "cost_usd, batch) "
        "VALUES (:ts, :model_id, :call_kind, :input_tokens, :output_tokens, "
        ":cache_read_tokens, :cost_usd, :batch)"
    )
    _MTD = text("SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE ts >= :month_start")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def write(self, entry: LedgerEntry) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                self._INSERT,
                {
                    "ts": entry.ts,
                    "model_id": entry.model_id,
                    "call_kind": entry.call_kind,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "cache_read_tokens": entry.cache_read_tokens,
                    "cost_usd": entry.cost_usd,
                    "batch": entry.batch,
                },
            )

    def month_to_date_usd(self, month_start: datetime) -> float:
        with self._engine.connect() as conn:
            value = conn.execute(self._MTD, {"month_start": month_start}).scalar()
        return float(value or 0.0)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CostMeter:
    """Records every call and enforces the monthly kill-switch."""

    def __init__(
        self,
        writer: LedgerWriter,
        monthly_budget_usd: float,
        *,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._writer = writer
        self._budget = monthly_budget_usd
        self._now = now_fn
        self._alerted = False

    def _month_start(self) -> datetime:
        now = self._now()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def record(
        self,
        *,
        model_id: str,
        call_kind: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cost_usd: float,
        batch: bool = False,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            ts=self._now(),
            model_id=model_id,
            call_kind=call_kind,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost_usd,
            batch=batch,
        )
        self._writer.write(entry)
        CLAUDE_COST_USD.inc(cost_usd)
        CLAUDE_BUDGET_UTILIZATION.set(self.budget_utilization())
        return entry

    def record_usage(
        self, *, model_id: str, call_kind: str, usage: Any, batch: bool = False
    ) -> LedgerEntry:
        """Convenience: compute dollars from a response.usage object, then record."""
        uc = cost_from_usage(model_id, usage, batch=batch)
        return self.record(
            model_id=model_id,
            call_kind=call_kind,
            input_tokens=uc.input_tokens + uc.cache_write_tokens,
            output_tokens=uc.output_tokens,
            cache_read_tokens=uc.cache_read_tokens,
            cost_usd=uc.cost_usd,
            batch=batch,
        )

    def month_to_date_usd(self) -> float:
        return self._writer.month_to_date_usd(self._month_start())

    def budget_utilization(self) -> float:
        if self._budget <= 0:
            return 1.0
        return self.month_to_date_usd() / self._budget

    def check_budget(self) -> None:
        """Raise BudgetExceededError once spend >= the monthly budget."""
        mtd = self.month_to_date_usd()
        utilization = mtd / self._budget if self._budget > 0 else 1.0
        CLAUDE_BUDGET_UTILIZATION.set(utilization)
        if mtd >= self._budget:
            raise BudgetExceededError(
                f"Claude monthly budget reached: ${mtd:.2f} of ${self._budget:.2f}"
            )
        if utilization >= _ALERT_UTILIZATION and not self._alerted:
            self._alerted = True
            log.warning(
                "claude budget utilization at %.0f%% ($%.2f of $%.2f)",
                utilization * 100,
                mtd,
                self._budget,
            )
