"""Typed execution errors. Design: doc 09 sections 4, 8.

Rejects are terminal and never auto-retried; transient broker faults get one
jittered retry and then escalate to ``HaltSignal`` (fail-closed).
"""

from __future__ import annotations


class ExecutionError(Exception):
    """Base class for execution-layer failures."""


class BrokerRejection(ExecutionError):
    """Broker rejected the order. Terminal — never auto-retried (doc 09 §8)."""

    def __init__(self, client_order_id: str, reason: str) -> None:
        super().__init__(f"broker rejected {client_order_id}: {reason}")
        self.client_order_id = client_order_id
        self.reason = reason


class DuplicateClientOrderId(ExecutionError):
    """Broker already knows this client_order_id — fetch state, do not resubmit."""

    def __init__(self, client_order_id: str) -> None:
        super().__init__(f"duplicate client_order_id {client_order_id}")
        self.client_order_id = client_order_id


class BrokerUnavailable(ExecutionError):
    """Transient broker fault (5xx / connection). Retried once with jitter."""


class BrokerDisconnect(BrokerUnavailable):
    """Connection lost mid-order: state ambiguous until reconciliation (doc 09 §8)."""


class RateLimited(ExecutionError):
    """429 from the broker — a logged incident, never a retry storm."""


class HaltSignal(ExecutionError):
    """Fail-closed escalation: stop submitting, hold positions, alert a human."""


class IllegalTransition(ExecutionError):
    """Order state machine violation. Illegal transitions raise, never coerce."""

    def __init__(self, client_order_id: str, from_state: str, to_state: str) -> None:
        super().__init__(f"illegal transition {from_state} -> {to_state} for {client_order_id}")
        self.client_order_id = client_order_id
        self.from_state = from_state
        self.to_state = to_state


class MissingDecisionContext(ExecutionError):
    """No order without a snapshot (doc 09 §6) — enforced in code, not convention."""

    def __init__(self, client_order_id: str) -> None:
        super().__init__(f"order {client_order_id} has no decision_context_id")
        self.client_order_id = client_order_id


class ReconciliationPending(ExecutionError):
    """Startup reconciliation has not completed; new submissions are blocked."""


class ReconciliationHalt(ExecutionError):
    """Sticky reconciliation halt is active for this scope; ack required to clear."""

    def __init__(self, scope: str) -> None:
        super().__init__(f"reconciliation halt active for scope {scope!r}")
        self.scope = scope
