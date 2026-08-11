"""Counterfactual path-walkers for decision experiences (plan Phase 1).

Every function is pure and total over bar series that are PASSED IN — this
module never touches persistence (import-linter: learning imports domain
only). Session arrays are ordered oldest → newest with index 0 = the entry
session; each walker returns a pnl percentage on the original position's
notional, or ``None`` when the data is insufficient to answer honestly.

Conventions:

- ``pnl_pct`` values are percentages (5.0 = +5%), matching
  ``ClosedTrade.pnl_pct`` upstream.
- "Insufficient data" is always ``None``, never 0.0 — zero is a real answer
  (the hold counterfactual), absence is not.
"""

from __future__ import annotations

from collections.abc import Sequence


def hold_pnl_pct() -> float:
    """Hold-instead-of-entering: no position was taken, so the counterfactual
    pnl is exactly 0.0 by definition (cash earns nothing in this model)."""
    return 0.0


def half_size_pnl_pct(realized_pnl_pct: float | None) -> float | None:
    """Half the position size ⇒ half the pnl on the original notional.

    Per-share return is unchanged; expressing the counterfactual against the
    ORIGINAL position's notional keeps all cf_* columns on one scale.
    """
    if realized_pnl_pct is None:
        return None
    return realized_pnl_pct / 2.0


def delayed_entry_pnl_pct(
    opens: Sequence[float],
    exit_session_index: int,
    exit_price: float | None,
) -> float | None:
    """Enter one session later (at that session's open), exit as realized.

    ``opens`` starts at the actual entry session (index 0). When the trade
    exited on its entry session the delayed entry never happens → 0.0 (a real
    answer: no position). Missing next-session open or exit price → ``None``.
    """
    if exit_session_index <= 0:
        return 0.0
    if exit_price is None or len(opens) < 2:
        return None
    delayed_open = opens[1]
    if delayed_open <= 0:
        return None
    return (exit_price / delayed_open - 1.0) * 100.0


def hold_to_horizon_pnl_pct(
    entry_price: float | None,
    closes: Sequence[float],
    horizon_days: int | None,
) -> float | None:
    """Ignore the actual exit; hold to the verdict's original horizon.

    ``closes`` starts at the entry session (index 0), so ``closes[horizon]``
    is the close ``horizon_days`` sessions after entry. ``None`` when the
    horizon close is not yet in the path (or inputs are missing/degenerate).
    """
    if entry_price is None or entry_price <= 0:
        return None
    if horizon_days is None or horizon_days <= 0:
        return None
    if len(closes) <= horizon_days:
        return None
    horizon_close = closes[horizon_days]
    if horizon_close <= 0:
        return None
    return (horizon_close / entry_price - 1.0) * 100.0


def obey_quant_veto_pnl_pct(
    quant_vetoed: bool | None,
    realized_pnl_pct: float | None,
) -> float | None:
    """What obeying the quant veto would have earned.

    A vetoed-but-taken trade obeyed means no position → 0.0. When quant did
    not veto, obeying it is exactly what happened → the realized pnl. Unknown
    veto status → ``None``.
    """
    if quant_vetoed is None:
        return None
    if quant_vetoed:
        return 0.0
    return realized_pnl_pct


def ev_per_decision(realized_pnl_pct: float | None) -> float | None:
    """Expected value per decision — v1 definition: the realized pnl_pct.

    Deliberately simple: with one resolution per decision there is no
    defensible per-row expectation beyond the realization itself. Phase 6
    (metrics) refines this into a population statistic (mean pnl per decision
    by cohort, counterfactual-adjusted); do NOT invent per-row math here.
    """
    return realized_pnl_pct
