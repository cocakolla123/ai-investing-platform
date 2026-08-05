"""Options sleeve selection & sizing — defined-risk only (ADR-0010).

Pure code with the doc 06 veto-contract philosophy: the risk engine — not the
research view — decides what trades, and every rule here fails CLOSED. An
underlying with any missing, non-finite, or out-of-range input is skipped with
a recorded reason, never sized.

Structure selection (one decision per underlying per run):

- LONG CALL:  verdict ``buy`` ∧ conviction >= ``min_conviction`` ∧ quant
  agrees (``quant_alignment == agree``) ∧ IV rank <= ``iv_rank_max``.
- CASH-SECURED PUT: the same ``buy`` conviction on a name passing long-term
  eligibility, but IV rank ABOVE ``iv_rank_max`` (premium too rich to buy) and
  at/above ``short_leg_iv_rank_min`` — the committee wants a lower entry, so
  get paid to wait; full assignment cash must be reservable.
- LONG PUT: verdict ``sell``/``avoid`` with conviction >= floor on a HELD name
  (protective), or on an unheld name when quant agrees (bearish expression);
  IV rank <= ``iv_rank_max`` either way.
- COVERED CALL: a long-term holding with a ``hold``/``trim`` verdict, >= 100
  shares per contract, and elevated IV (rank >= ``short_leg_iv_rank_min``).

Contract choice: the nearest MONTHLY expiry (third Friday) with DTE in
``[min_dte, max_dte]``; strike by delta target (~0.6 long legs, ~0.3 short
legs) from chain greeks when every candidate row carries a finite delta, else
a documented moneyness heuristic (long call ~5% ITM, long put ~5% ITM, short
legs ~5% OTM). Liquidity floors (open interest, relative spread) filter rows
BEFORE selection. Buys price at the ask, sells at the bid — limit orders only.

Sizing: per-position premium at risk (long premium; CSP assignment
obligation; covered calls contribute zero — share-covered) <=
``per_position_cap x equity``; sleeve total (existing positions + new
decisions) <= ``premium_at_risk_cap x equity``; at most
``max_open_positions`` distinct contracts. Greedy allocation by conviction
descending with a deterministic underlying tiebreak.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from dynamo.domain.models.options import (
    CONTRACT_MULTIPLIER,
    STRUCTURE_LEGS,
    OptionChainRow,
    OptionPosition,
    OptionRight,
    OptionStructure,
    OptionTradeDecision,
    occ_symbol,
)
from dynamo.domain.models.research import Action, CommitteeVerdict, QuantAlignment

SKIP_INVALID_EQUITY = "invalid_equity"
SKIP_NO_VERDICT_MATCH = "no_structure_matches_verdict"
SKIP_NO_SPOT = "no_spot_price"
SKIP_NO_IV_RANK = "no_iv_rank"
SKIP_NO_CHAIN = "no_chain"
SKIP_NO_EXPIRY_IN_WINDOW = "no_monthly_expiry_in_dte_window"
SKIP_NO_LIQUID_STRIKE = "no_strike_passes_liquidity_floors"
SKIP_BELOW_ONE_CONTRACT = "below_one_contract"
SKIP_SLEEVE_CAP_EXHAUSTED = "sleeve_cap_exhausted"
SKIP_MAX_OPEN_POSITIONS = "max_open_positions"
SKIP_ALREADY_HELD = "contract_already_held"
SKIP_INSUFFICIENT_SHARES = "insufficient_share_collateral"
SKIP_INSUFFICIENT_CASH = "insufficient_cash_collateral"

IV_RANK_MIN_OBS = 10


@dataclass(frozen=True)
class OptionsSizingParams:
    """Mirrors the ``options_*`` settings; plain floats so this stays pure."""

    premium_at_risk_cap: float = 0.05
    per_position_cap: float = 0.01
    min_dte: int = 21
    max_dte: int = 90
    min_open_interest: int = 100
    max_spread_pct: float = 0.10
    min_conviction: float = 0.7
    iv_rank_max: float = 0.8
    max_open_positions: int = 8
    short_leg_iv_rank_min: float = 0.5  # "elevated IV" floor for premium selling
    long_delta_target: float = 0.6
    short_delta_target: float = 0.3


@dataclass(frozen=True)
class OptionsSizingInputs:
    """Everything the sizer reads — domain models and plain values only."""

    verdicts: Mapping[str, CommitteeVerdict]
    chains: Mapping[str, Sequence[OptionChainRow]]  # underlying -> rows
    spot_prices: Mapping[str, float]
    iv_ranks: Mapping[str, float]  # underlying -> IV rank proxy in [0, 1]
    equity: float
    cash_available: float
    shares_held: Mapping[str, float]  # underlying -> share qty held
    long_term_holdings: frozenset[str]  # held names in the long-term sleeve
    long_term_eligible: frozenset[str]  # names passing long-term eligibility
    open_positions: Sequence[OptionPosition]
    now: datetime


@dataclass(frozen=True)
class OptionsSizingResult:
    decisions: tuple[OptionTradeDecision, ...]
    skipped: tuple[tuple[str, str], ...]  # (underlying-or-occ, reason)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def iv_rank(current_iv: float, history: Sequence[float]) -> float | None:
    """Min-max IV rank proxy from stored ATM-IV history; None fails closed.

    Requires >= ``IV_RANK_MIN_OBS`` finite observations; a flat history maps
    to 0.5 (no information either way).
    """
    clean = [v for v in history if math.isfinite(v)]
    if not math.isfinite(current_iv) or len(clean) < IV_RANK_MIN_OBS:
        return None
    lo, hi = min(clean), max(clean)
    if hi <= lo:
        return 0.5
    return min(max((current_iv - lo) / (hi - lo), 0.0), 1.0)


def is_monthly_expiry(d: date) -> bool:
    """Standard monthly expiration: the third Friday of the month."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def _liquid(row: OptionChainRow, params: OptionsSizingParams) -> bool:
    """Liquidity floors — fail closed on any missing/non-finite field."""
    if row.bid is None or row.ask is None or row.open_interest is None:
        return False
    bid, ask = row.bid, row.ask
    if not _finite(bid, ask) or bid <= 0.0 or ask <= bid:
        return False
    if row.open_interest < params.min_open_interest:
        return False
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid <= params.max_spread_pct


def _pick_expiry(
    rows: Sequence[OptionChainRow], today: date, params: OptionsSizingParams
) -> date | None:
    """Nearest monthly expiry with DTE inside ``[min_dte, max_dte]``."""
    candidates = sorted(
        {
            r.expiry
            for r in rows
            if is_monthly_expiry(r.expiry)
            and params.min_dte <= (r.expiry - today).days <= params.max_dte
        }
    )
    return candidates[0] if candidates else None


def _pick_strike(
    rows: Sequence[OptionChainRow],
    right: OptionRight,
    spot: float,
    delta_target: float,
    params: OptionsSizingParams,
) -> OptionChainRow | None:
    """Delta-targeted strike; moneyness heuristic when greeks are missing.

    When EVERY liquid candidate carries a finite delta, pick the row whose
    |delta| is closest to the target. Otherwise (documented heuristic): a
    ~0.6-delta long leg is approximated by a strike ~5% in the money and a
    ~0.3-delta short leg by a strike ~5% out of the money.
    """
    candidates = [r for r in rows if r.right is right and _liquid(r, params)]
    if not candidates:
        return None
    if all(r.delta is not None and math.isfinite(r.delta) for r in candidates):
        return min(
            candidates,
            key=lambda r: (abs(abs(r.delta or 0.0) - delta_target), r.strike),
        )
    long_leg = delta_target >= 0.5
    itm = long_leg  # long legs target in-the-money, short legs out-of-the-money
    if right is OptionRight.CALL:
        target_strike = spot * (0.95 if itm else 1.05)
    else:
        target_strike = spot * (1.05 if itm else 0.95)
    return min(candidates, key=lambda r: (abs(r.strike - target_strike), r.strike))


@dataclass(frozen=True)
class _Candidate:
    underlying: str
    structure: OptionStructure
    row: OptionChainRow
    limit_price: float  # ask for buys, bid for sells
    contracts: int
    premium_at_risk: float  # dollars counted against the sleeve
    collateral_shares: float
    collateral_cash: float
    conviction: float
    iv_rank: float
    dte: int
    verdict: CommitteeVerdict


def _structure_for(
    verdict: CommitteeVerdict,
    *,
    held_shares: float,
    long_term_holding: bool,
    long_term_eligible: bool,
    iv_rank_value: float,
    params: OptionsSizingParams,
) -> OptionStructure | None:
    """The doc-approved structure ladder; None when nothing matches."""
    action, conviction = verdict.action, verdict.conviction
    quant_agrees = verdict.quant_alignment == QuantAlignment.AGREE
    if action == Action.BUY and conviction >= params.min_conviction:
        if quant_agrees and iv_rank_value <= params.iv_rank_max:
            return OptionStructure.LONG_CALL
        if (
            long_term_eligible
            and iv_rank_value > params.iv_rank_max
            and iv_rank_value >= params.short_leg_iv_rank_min
        ):
            return OptionStructure.CASH_SECURED_PUT
        return None
    if action in (Action.SELL, Action.AVOID) and conviction >= params.min_conviction:
        if iv_rank_value > params.iv_rank_max:
            return None  # never buy premium at extreme IV
        if held_shares > 0.0:
            return OptionStructure.LONG_PUT  # protective
        if quant_agrees:
            return OptionStructure.LONG_PUT  # bearish expression
        return None
    if (
        action in (Action.HOLD, Action.TRIM)
        and long_term_holding
        and held_shares >= CONTRACT_MULTIPLIER
        and iv_rank_value >= params.short_leg_iv_rank_min
    ):
        return OptionStructure.COVERED_CALL
    return None


def _size_candidate(
    underlying: str,
    structure: OptionStructure,
    verdict: CommitteeVerdict,
    inputs: OptionsSizingInputs,
    iv_rank_value: float,
    params: OptionsSizingParams,
) -> _Candidate | str:
    """Contract choice + per-position sizing for one underlying (fail closed)."""
    spot = inputs.spot_prices.get(underlying)
    if spot is None or not _finite(spot) or spot <= 0.0:
        return SKIP_NO_SPOT
    rows = inputs.chains.get(underlying) or ()
    if not rows:
        return SKIP_NO_CHAIN
    expiry = _pick_expiry(rows, inputs.now.date(), params)
    if expiry is None:
        return SKIP_NO_EXPIRY_IN_WINDOW

    _, right = STRUCTURE_LEGS[structure]
    long_leg = structure in (OptionStructure.LONG_CALL, OptionStructure.LONG_PUT)
    delta_target = params.long_delta_target if long_leg else params.short_delta_target
    expiry_rows = [r for r in rows if r.expiry == expiry]
    row = _pick_strike(expiry_rows, right, spot, delta_target, params)
    if row is None:
        return SKIP_NO_LIQUID_STRIKE
    assert row.bid is not None and row.ask is not None  # _liquid guaranteed

    limit_price = row.ask if long_leg else row.bid  # buys at ask, sells at bid
    per_position_budget = params.per_position_cap * inputs.equity
    shares = inputs.shares_held.get(underlying, 0.0)
    if not math.isfinite(shares):
        shares = 0.0

    if long_leg:
        per_contract_risk = limit_price * CONTRACT_MULTIPLIER
        contracts = math.floor(per_position_budget / per_contract_risk)
        collateral_shares, collateral_cash = 0.0, 0.0
    elif structure is OptionStructure.CASH_SECURED_PUT:
        per_contract_risk = row.strike * CONTRACT_MULTIPLIER
        contracts = math.floor(per_position_budget / per_contract_risk)
        affordable = (
            math.floor(inputs.cash_available / per_contract_risk)
            if _finite(inputs.cash_available) and inputs.cash_available > 0.0
            else 0
        )
        if affordable < 1:
            return SKIP_INSUFFICIENT_CASH
        contracts = min(contracts, affordable)
        collateral_shares = 0.0
        collateral_cash = contracts * per_contract_risk
    else:  # covered call — capped by round lots held; zero sleeve risk
        per_contract_risk = 0.0
        contracts = math.floor(shares / CONTRACT_MULTIPLIER)
        if contracts < 1:
            return SKIP_INSUFFICIENT_SHARES
        collateral_shares = shares
        collateral_cash = 0.0

    if contracts < 1:
        return SKIP_BELOW_ONE_CONTRACT
    return _Candidate(
        underlying=underlying,
        structure=structure,
        row=row,
        limit_price=limit_price,
        contracts=contracts,
        premium_at_risk=contracts * per_contract_risk,
        collateral_shares=collateral_shares,
        collateral_cash=collateral_cash,
        conviction=verdict.conviction,
        iv_rank=iv_rank_value,
        dte=(expiry - inputs.now.date()).days,
        verdict=verdict,
    )


def _decision(candidate: _Candidate, contracts: int, premium_at_risk: float) -> OptionTradeDecision:
    row, verdict = candidate.row, candidate.verdict
    side, right = STRUCTURE_LEGS[candidate.structure]
    occ = occ_symbol(candidate.underlying, row.expiry, right, row.strike)
    collateral_cash = (
        contracts * row.strike * CONTRACT_MULTIPLIER
        if candidate.structure is OptionStructure.CASH_SECURED_PUT
        else 0.0
    )
    rationale = (
        f"{candidate.structure.value} {contracts}x {occ} @ {candidate.limit_price:.2f} "
        f"limit: committee {verdict.action.value} conviction {verdict.conviction:.2f} "
        f"({verdict.quant_alignment.value} quant), IV rank {candidate.iv_rank:.2f}, "
        f"{candidate.dte}d to expiry, strike {row.strike:.2f}"
        + (f", delta {row.delta:.2f}" if row.delta is not None else ", moneyness heuristic")
        + f"; premium at risk ${premium_at_risk:,.2f} within the per-position and sleeve caps"
    )
    return OptionTradeDecision(
        structure=candidate.structure,
        side=side,
        right=right,
        underlying=candidate.underlying,
        occ_symbol=occ,
        expiry=row.expiry,
        strike=row.strike,
        contracts=contracts,
        limit_price=candidate.limit_price,
        premium_at_risk=premium_at_risk,
        collateral_shares=candidate.collateral_shares,
        collateral_cash=collateral_cash,
        conviction=verdict.conviction,
        iv_rank=candidate.iv_rank,
        dte=candidate.dte,
        rationale=rationale,
        verdict_ref=verdict.id,
    )


def select_and_size_option_trades(
    inputs: OptionsSizingInputs, params: OptionsSizingParams | None = None
) -> OptionsSizingResult:
    """Filter, size, and greedily allocate the options sleeve (module docstring).

    Deterministic: candidates ranked by conviction descending, ties broken by
    underlying ascending. The sleeve total (existing premium at risk +
    assignment obligations + new decisions) never exceeds
    ``premium_at_risk_cap x equity``; no position exceeds
    ``per_position_cap x equity``; at most ``max_open_positions`` distinct
    contracts are held.
    """
    params = params if params is not None else OptionsSizingParams()
    if not _finite(inputs.equity) or inputs.equity <= 0.0:
        return OptionsSizingResult(decisions=(), skipped=(("*", SKIP_INVALID_EQUITY),))

    held_occ = {p.occ_symbol for p in inputs.open_positions if p.contracts > 0}
    sleeve_used = sum(p.premium_at_risk for p in inputs.open_positions)
    if not math.isfinite(sleeve_used):
        return OptionsSizingResult(decisions=(), skipped=(("*", SKIP_INVALID_EQUITY),))

    skipped: list[tuple[str, str]] = []
    candidates: list[_Candidate] = []
    for underlying in sorted(inputs.verdicts):
        verdict = inputs.verdicts[underlying]
        rank = inputs.iv_ranks.get(underlying)
        if rank is None or not _finite(rank) or not 0.0 <= rank <= 1.0:
            skipped.append((underlying, SKIP_NO_IV_RANK))
            continue
        shares = inputs.shares_held.get(underlying, 0.0)
        if not math.isfinite(shares):
            shares = 0.0
        conviction = verdict.conviction
        if not _finite(conviction):
            skipped.append((underlying, SKIP_NO_VERDICT_MATCH))
            continue
        structure = _structure_for(
            verdict,
            held_shares=shares,
            long_term_holding=underlying in inputs.long_term_holdings,
            long_term_eligible=underlying in inputs.long_term_eligible,
            iv_rank_value=rank,
            params=params,
        )
        if structure is None:
            skipped.append((underlying, SKIP_NO_VERDICT_MATCH))
            continue
        outcome = _size_candidate(underlying, structure, verdict, inputs, rank, params)
        if isinstance(outcome, str):
            skipped.append((underlying, outcome))
        elif outcome.row.occ in held_occ:
            skipped.append((outcome.row.occ, SKIP_ALREADY_HELD))
        else:
            candidates.append(outcome)

    candidates.sort(key=lambda c: (-c.conviction, c.underlying))

    sleeve_remaining = params.premium_at_risk_cap * inputs.equity - sleeve_used
    open_slots = params.max_open_positions - len(held_occ)
    cash_remaining = inputs.cash_available if _finite(inputs.cash_available) else 0.0
    decisions: list[OptionTradeDecision] = []
    for candidate in candidates:
        if open_slots <= 0:
            skipped.append((candidate.underlying, SKIP_MAX_OPEN_POSITIONS))
            continue
        contracts = candidate.contracts
        premium = candidate.premium_at_risk
        if premium > 0.0 and premium > sleeve_remaining:
            # Shrink to the sleeve remainder; below one contract → skip.
            per_contract = premium / candidate.contracts
            contracts = math.floor(sleeve_remaining / per_contract)
            if contracts < 1:
                skipped.append((candidate.underlying, SKIP_SLEEVE_CAP_EXHAUSTED))
                continue
            premium = contracts * per_contract
        if candidate.structure is OptionStructure.CASH_SECURED_PUT:
            required = contracts * candidate.row.strike * CONTRACT_MULTIPLIER
            if required > cash_remaining:
                skipped.append((candidate.underlying, SKIP_INSUFFICIENT_CASH))
                continue
            cash_remaining -= required
        decisions.append(_decision(candidate, contracts, premium))
        sleeve_remaining -= premium
        open_slots -= 1
    return OptionsSizingResult(decisions=tuple(decisions), skipped=tuple(skipped))
