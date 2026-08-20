"""Periodic reconciliation against broker truth (doc 09 §5).

Any divergence halts trading: a symbol mismatch halts that symbol, a cash or
equity mismatch beyond tolerance halts globally. Halts are sticky — they clear
only on explicit human acknowledgment, never by "re-checking and quietly
resuming".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from dynamo.domain.models.portfolio import OrderState, Position
from dynamo.domain.ports import Broker

GLOBAL_SCOPE = "global"
QTY_TOLERANCE = 1e-6

# Run-scope label written to ``reconciliation_runs.scope`` by the equity/crypto
# reconciler. The PM and options reconcilers write ``pm*`` / ``options*``; the
# equity sleeve's sticky-halt query keys off this prefix, so the three sleeves
# never inherit each other's halts.
EQUITY_RUN_SCOPE = "equities"


@dataclass(frozen=True)
class Divergence:
    scope: str  # symbol, or "global"
    kind: str  # position_qty | unknown_position | unknown_order | order_state | cash | equity
    internal: float | str | None
    broker: float | str | None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "kind": self.kind,
            "internal": self.internal,
            "broker": self.broker,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReconciliationResult:
    run_id: int
    clean: bool
    divergences: tuple[Divergence, ...] = ()

    @property
    def halted_scopes(self) -> frozenset[str]:
        return frozenset(d.scope for d in self.divergences)


class ReconciliationStore(Protocol):
    def record_run(
        self, ts: datetime, scope: str, result: str, divergences: Sequence[dict[str, Any]]
    ) -> int: ...

    def record_ack(self, run_id: int, acked_by: str, acked_at: datetime, note: str) -> None: ...


class InternalStateReader(Protocol):
    """Internal truth the broker is compared against."""

    def positions(self) -> Sequence[Position]: ...

    def open_order_ids(self) -> Sequence[tuple[str, OrderState, str]]: ...  # (coid, state, symbol)

    def cash(self) -> float: ...

    def equity(self) -> float: ...


AlertCallback = Callable[[str, str], None]

_OPEN_BROKER_STATUSES = frozenset({"new", "accepted", "partially_filled", "pending_new"})


@dataclass
class HaltRegistry:
    """Sticky halt scopes. Cleared only via explicit acknowledgment."""

    _halts: dict[str, int] = field(default_factory=dict)  # scope -> run_id

    def engage(self, scope: str, run_id: int) -> None:
        self._halts[scope] = run_id

    def is_halted(self, symbol: str | None = None) -> bool:
        if GLOBAL_SCOPE in self._halts:
            return True
        return symbol is not None and symbol in self._halts

    @property
    def active(self) -> dict[str, int]:
        return dict(self._halts)

    def clear(self, scope: str) -> int | None:
        return self._halts.pop(scope, None)


def acknowledge_reconciliation(
    store: ReconciliationStore, run_id: int, acked_by: str, note: str, now: datetime
) -> None:
    """Record the one deliberate act that clears a sticky halt (doc 09 §5).

    The halt lives in the database — an unacked ``divergent`` run — so the ack
    row *is* the clearance; no in-memory registry has to survive for it to
    work. A blank note is refused: "who/when/why" is the whole point of the
    audit row, and nothing ever clears a halt automatically.
    """
    if not note.strip():
        raise ValueError("reconciliation ack requires a note")
    store.record_ack(run_id, acked_by, now, note)


class Reconciler:
    def __init__(
        self,
        broker: Broker,
        internal: InternalStateReader,
        store: ReconciliationStore,
        alert: AlertCallback,
        *,
        cash_tolerance: float = 1.00,
        halts: HaltRegistry | None = None,
        run_scope: str = EQUITY_RUN_SCOPE,
    ) -> None:
        self._broker = broker
        self._internal = internal
        self._store = store
        self._alert = alert
        self._cash_tolerance = cash_tolerance
        self._run_scope = run_scope
        self.halts = halts if halts is not None else HaltRegistry()

    def reconcile(self, now: datetime) -> ReconciliationResult:
        """Diff broker truth vs internal state; divergence → sticky halt + alert."""
        divergences: list[Divergence] = []
        divergences.extend(self._diff_positions())
        divergences.extend(self._diff_orders())
        divergences.extend(self._diff_cash())

        clean = not divergences
        # The run scope is sleeve-prefixed so the equity sticky-halt query can
        # tell an equity halt from a pm/options one (they share the table).
        if clean:
            scope = self._run_scope
        elif any(d.scope == GLOBAL_SCOPE for d in divergences):
            scope = f"{self._run_scope}:{GLOBAL_SCOPE}"
        else:
            scope = f"{self._run_scope}:symbols"
        run_id = self._store.record_run(
            now,
            scope,
            "clean" if clean else "divergent",
            [d.as_dict() for d in divergences],
        )
        if not clean:
            for d in divergences:
                self.halts.engage(d.scope, run_id)
            self._alert(
                "reconciliation_divergence",
                f"run {run_id}: {len(divergences)} divergence(s); "
                f"halted scopes: {sorted({d.scope for d in divergences})}",
            )
        return ReconciliationResult(run_id, clean, tuple(divergences))

    def acknowledge(self, run_id: int, acked_by: str, note: str, now: datetime) -> list[str]:
        """Explicit human ack clears the halts engaged by ``run_id`` (sticky
        otherwise) and records who/when/why in ``reconciliation_acks``."""
        acknowledge_reconciliation(self._store, run_id, acked_by, note, now)
        cleared = [s for s, rid in self.halts.active.items() if rid == run_id]
        for scope in cleared:
            self.halts.clear(scope)
        return cleared

    # --- diffs ---

    def _diff_positions(self) -> list[Divergence]:
        out: list[Divergence] = []
        internal = {p.symbol: p.qty for p in self._internal.positions() if abs(p.qty) > 0}
        broker = {
            str(p["symbol"]): float(p["qty"])
            for p in self._broker.positions()
            if abs(float(p["qty"])) > 0
        }
        for symbol in sorted(set(internal) | set(broker)):
            iq, bq = internal.get(symbol, 0.0), broker.get(symbol, 0.0)
            if abs(iq - bq) > QTY_TOLERANCE:
                kind = (
                    "position_qty"
                    if symbol in internal and symbol in broker
                    else ("unknown_position")
                )
                out.append(Divergence(symbol, kind, iq, bq, "position qty mismatch"))
        return out

    def _diff_orders(self) -> list[Divergence]:
        out: list[Divergence] = []
        internal_open = {
            coid: (state, symbol) for coid, state, symbol in self._internal.open_order_ids()
        }
        broker_open = {
            str(o["client_order_id"]): o
            for o in self._broker.open_orders()
            if str(o.get("status", "")) in _OPEN_BROKER_STATUSES
        }
        for coid, o in broker_open.items():
            if coid not in internal_open:
                out.append(
                    Divergence(
                        str(o.get("symbol", GLOBAL_SCOPE)),
                        "unknown_order",
                        None,
                        coid,
                        "open at broker, unknown internally",
                    )
                )
        for coid, (state, symbol) in internal_open.items():
            if state in (OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED) and (
                coid not in broker_open
            ):
                # Might have terminally resolved at the broker; check directly.
                try:
                    remote = self._broker.get_order(coid)
                    remote_status = str(remote.get("status", ""))
                except Exception:
                    remote_status = "unknown"
                if remote_status in _OPEN_BROKER_STATUSES or remote_status == "unknown":
                    out.append(
                        Divergence(
                            symbol,
                            "order_state",
                            state.value,
                            remote_status,
                            f"order {coid} open internally, not open at broker",
                        )
                    )
        return out

    def _diff_cash(self) -> list[Divergence]:
        out: list[Divergence] = []
        account = self._broker.account()
        broker_cash = float(account.get("cash", 0.0))
        broker_equity = float(account.get("equity", 0.0))
        if abs(broker_cash - self._internal.cash()) > self._cash_tolerance:
            out.append(
                Divergence(
                    GLOBAL_SCOPE, "cash", self._internal.cash(), broker_cash, "cash mismatch"
                )
            )
        if abs(broker_equity - self._internal.equity()) > self._cash_tolerance:
            out.append(
                Divergence(
                    GLOBAL_SCOPE,
                    "equity",
                    self._internal.equity(),
                    broker_equity,
                    "equity mismatch",
                )
            )
        return out
