"""Same-day tactical entries + PDT guard — pure selection logic (ADR-0010).

The 15-minute intraday loop evaluates two triggers during market hours:

(a) newly promoted Tier-3 names with a MATERIAL triage verdict today and a
    fresh committee ``buy``/``add`` verdict, and
(b) breakout-strategy signals for watchlist tacticals whose live quote has
    crossed the signal's entry reference price.

Triggered names run the SAME sizing cascade as the nightly path
(:func:`dynamo.risk.sizing.size_positions`) — no intraday shortcut exists. The
orchestration op turns the resulting weights into marketable-limit equity
orders through the standard OrderManager (decision context, gate, kill-switch).

Hard limits: at most ``cap_per_day`` intraday entries per calendar day, and a
PDT guard — a name entered same-day is NEVER sold same-day automatically;
stops are exempted, but if a same-day stop exit would trip the pattern
day-trader rule under $25k equity it is logged + alerted instead of executed,
UNLESS it is a disaster stop.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from dynamo.domain.models.research import Action, CommitteeVerdict, Signal
from dynamo.domain.models.risk import RiskConfig
from dynamo.risk.sizing import SizingCandidate, SizingContext, size_positions

PDT_EQUITY_FLOOR = 25_000.0
# Executing a 4th day trade within 5 business days flags PDT; block at >= 3.
PDT_MAX_DAY_TRADES = 3

TRIGGER_PROMOTION = "tier3_promotion"
TRIGGER_BREAKOUT = "breakout"

SKIP_CAP_REACHED = "intraday_cap_reached"
SKIP_ALREADY_ENTERED_TODAY = "already_entered_today"
SKIP_NO_VERDICT = "no_buy_verdict"
SKIP_NO_QUOTE = "no_live_quote"
SKIP_NOT_TRIGGERED = "breakout_not_triggered"
SKIP_NO_SIZING_INPUTS = "no_sizing_inputs"
SKIP_ZERO_WEIGHT = "sized_to_zero"


@dataclass(frozen=True)
class IntradayEntryParams:
    cap_per_day: int = 3


@dataclass(frozen=True)
class IntradayEntry:
    symbol: str
    trigger: str  # tier3_promotion | breakout
    weight: float  # cascade output, fraction of equity
    quote: float  # live quote the order prices against
    conviction: float
    signal_ref: str = ""  # strategy id for breakout entries
    rationale: str = ""


@dataclass(frozen=True)
class IntradayEntriesResult:
    entries: tuple[IntradayEntry, ...]
    skipped: tuple[tuple[str, str], ...]


def select_intraday_entries(
    *,
    promoted_material: Sequence[str],
    breakout_signals: Sequence[Signal],
    live_quotes: Mapping[str, float],
    verdicts: Mapping[str, CommitteeVerdict],
    sizing_candidates: Mapping[str, SizingCandidate],
    config: RiskConfig,
    context: SizingContext,
    entries_today: int,
    entered_today: frozenset[str],
    params: IntradayEntryParams | None = None,
) -> IntradayEntriesResult:
    """Deterministic same-day entry selection under the daily cap.

    Candidates are ranked by conviction descending (breakouts use signal
    strength when no verdict exists), ties broken by symbol. Every selected
    name must survive the full sizing cascade with a positive weight.
    """
    params = params if params is not None else IntradayEntryParams()
    skipped: list[tuple[str, str]] = []

    triggered: dict[str, tuple[str, float, str]] = {}  # symbol -> (trigger, rank, ref)
    for symbol in sorted(set(promoted_material)):
        verdict = verdicts.get(symbol)
        if verdict is None or verdict.action not in (Action.BUY, Action.ADD):
            skipped.append((symbol, SKIP_NO_VERDICT))
            continue
        triggered[symbol] = (TRIGGER_PROMOTION, verdict.conviction, "")
    for signal in sorted(breakout_signals, key=lambda s: s.symbol):
        symbol = signal.symbol
        if symbol in triggered:
            continue
        quote = live_quotes.get(symbol)
        if quote is None or not math.isfinite(quote) or quote <= 0.0:
            skipped.append((symbol, SKIP_NO_QUOTE))
            continue
        ref = signal.entry_ref_price
        if ref is None or not math.isfinite(ref) or quote < ref:
            skipped.append((symbol, SKIP_NOT_TRIGGERED))
            continue
        verdict = verdicts.get(symbol)
        rank = verdict.conviction if verdict is not None else signal.strength
        triggered[symbol] = (TRIGGER_BREAKOUT, rank, signal.strategy_id)

    budget = max(params.cap_per_day - entries_today, 0)
    ordered = sorted(triggered.items(), key=lambda kv: (-kv[1][1], kv[0]))

    eligible: list[tuple[str, str, float, str, float, SizingCandidate]] = []
    for symbol, (trigger, rank, signal_ref) in ordered:
        if symbol in entered_today:
            skipped.append((symbol, SKIP_ALREADY_ENTERED_TODAY))
            continue
        quote = live_quotes.get(symbol)
        if quote is None or not math.isfinite(quote) or quote <= 0.0:
            skipped.append((symbol, SKIP_NO_QUOTE))
            continue
        candidate = sizing_candidates.get(symbol)
        if candidate is None:
            skipped.append((symbol, SKIP_NO_SIZING_INPUTS))
            continue
        eligible.append((symbol, trigger, rank, signal_ref, quote, candidate))

    # One cascade run over the ranked list, so sector/cluster/gross budgets
    # are consumed cumulatively exactly as in the nightly path.
    sized = size_positions([row[5] for row in eligible], config, context)
    entries: list[IntradayEntry] = []
    for (symbol, trigger, rank, signal_ref, quote, _), position in zip(
        eligible, sized, strict=True
    ):
        if len(entries) >= budget:
            skipped.append((symbol, SKIP_CAP_REACHED))
            continue
        if position.weight <= 0.0:
            skipped.append((symbol, SKIP_ZERO_WEIGHT))
            continue
        entries.append(
            IntradayEntry(
                symbol=symbol,
                trigger=trigger,
                weight=position.weight,
                quote=quote,
                conviction=rank,
                signal_ref=signal_ref,
                rationale=(
                    f"{trigger} same-day entry at {quote:.2f}: sized "
                    f"{position.weight:.4f} of equity by the standard cascade "
                    f"(clamps: {', '.join(position.clamps_applied) or 'none'})"
                ),
            )
        )
    return IntradayEntriesResult(entries=tuple(entries), skipped=tuple(skipped))


# --- PDT guard (doc 13 §6: the execution layer counts day trades) ----------


def count_day_trades(fills: Iterable[tuple[str, str, date]]) -> int:
    """Day trades in a window of ``(symbol, side, fill_date)`` fills.

    A day trade is a buy and a sell of the same symbol on the same day; each
    (symbol, day) contributes ``min(#buys, #sells)`` day trades.
    """
    buys: dict[tuple[str, date], int] = {}
    sells: dict[tuple[str, date], int] = {}
    for symbol, side, day in fills:
        key = (symbol, day)
        if side == "buy":
            buys[key] = buys.get(key, 0) + 1
        elif side == "sell":
            sells[key] = sells.get(key, 0) + 1
    return sum(min(n, sells.get(key, 0)) for key, n in buys.items())


@dataclass(frozen=True)
class PdtDecision:
    allowed: bool
    reason: str


def pdt_exit_allowed(
    *,
    equity: float,
    day_trades_in_window: int,
    entered_today: bool,
    is_disaster_stop: bool,
) -> PdtDecision:
    """May an automatic same-day exit run? (Stops only — nothing else may
    auto-exit a name entered the same day.)

    Disaster stops always run (capital protection outranks PDT bookkeeping).
    Above the $25k floor PDT does not apply. Below it, an exit that would be
    the 4th day trade in 5 business days is refused — the caller logs + alerts
    instead of executing.
    """
    if not entered_today:
        return PdtDecision(True, "not_a_day_trade")
    if is_disaster_stop:
        return PdtDecision(True, "disaster_stop_overrides_pdt")
    if not math.isfinite(equity):
        return PdtDecision(False, "invalid_equity_fail_closed")
    if equity >= PDT_EQUITY_FLOOR:
        return PdtDecision(True, "equity_above_pdt_floor")
    if day_trades_in_window >= PDT_MAX_DAY_TRADES:
        return PdtDecision(False, "would_trip_pdt")
    return PdtDecision(True, "day_trade_within_pdt_allowance")
