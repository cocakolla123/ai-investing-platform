"""Cost meter + monthly kill-switch. Design: doc 03 section 5.

Every Claude call is recorded to the cost_ledger via a pluggable writer.
`check_budget()` refuses new submissions once month-to-date spend reaches
`CLAUDE_MONTHLY_BUDGET_USD`; alert (log) at 70% utilization.

A row may also carry a :class:`CallTrace` (eval Phase 4): which committee
role made the call, the request it served, the prompt version, the journal
entries retrieved into its context, wall-clock latency, which attempt it was
and whether its output passed the client's schema check. Rows written
without one keep NULL trace columns.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
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
class CallTrace:
    """Per-call tracing fields; every one optional (NULL when unknown)."""

    role: str | None = None  # dossier | bull | bear | arbiter | triage | ...
    request_id: str | None = None  # batch custom_id, when the call served one
    prompt_hash: str | None = None  # sha256[:16] of the system prompt (its version)
    retrieved_ids: tuple[str, ...] | None = None  # journal/experience ids in context
    latency_ms: float | None = None  # wall clock of the call, retries excluded
    attempt: int | None = None  # 1 = first try; >1 = a retry
    validation_ok: bool | None = None  # output parsed with its required keys
    equiv_cost_usd: float | None = None  # API-equivalent cost reported by the CLI

    @staticmethod
    def ids(values: Sequence[str] | None) -> tuple[str, ...] | None:
        return None if values is None else tuple(str(v) for v in values)

    @staticmethod
    def hash_prompt(system: str) -> str:
        """The prompt's version id: the system prompt is the versioned part
        (contexts and per-symbol instructions are hashed in replay_contexts)."""
        return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


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
    trace: CallTrace | None = None


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
        "cost_usd, batch, role, request_id, prompt_hash, retrieved_ids, latency_ms, "
        "attempt, validation_ok, equiv_cost_usd) "
        "VALUES (:ts, :model_id, :call_kind, :input_tokens, :output_tokens, "
        ":cache_read_tokens, :cost_usd, :batch, :role, :request_id, :prompt_hash, "
        "CAST(:retrieved_ids AS JSONB), :latency_ms, :attempt, :validation_ok, "
        ":equiv_cost_usd)"
    )
    _MTD = text("SELECT COALESCE(SUM(cost_usd), 0) FROM cost_ledger WHERE ts >= :month_start")

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def write(self, entry: LedgerEntry) -> None:
        tr = entry.trace or CallTrace()
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
                    "role": tr.role,
                    "request_id": tr.request_id,
                    "prompt_hash": tr.prompt_hash,
                    "retrieved_ids": (
                        None if tr.retrieved_ids is None else json.dumps(list(tr.retrieved_ids))
                    ),
                    "latency_ms": tr.latency_ms,
                    "attempt": tr.attempt,
                    "validation_ok": tr.validation_ok,
                    "equiv_cost_usd": tr.equiv_cost_usd,
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
        trace: CallTrace | None = None,
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
            trace=trace,
        )
        self._writer.write(entry)
        CLAUDE_COST_USD.inc(cost_usd)
        CLAUDE_BUDGET_UTILIZATION.set(self.budget_utilization())
        return entry

    def record_usage(
        self,
        *,
        model_id: str,
        call_kind: str,
        usage: Any,
        batch: bool = False,
        trace: CallTrace | None = None,
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
            trace=trace,
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
