"""Rule-based portfolio construction (doc 07 section 2).

Four pure steps: sleeve split -> candidate selection -> sizing (DELEGATED to
the risk engine's cascade, never duplicated here) -> cash residual with a
floor. An optional fifth step (``ConstructionConfig.ballast_symbol``) parks
the residual above the cash floor in a ballast ETF instead of dollars — see
``_ballast_lot`` for the full semantics; empty symbol keeps construction
bit-identical to the four-step path. Every field of ``RiskVerdict`` is an
inviolable bound for ACTIVE lots: a final greedy
compliance pass allocates weights in priority order against the verdict's
caps, so the constructed portfolio can never violate a constraint — if the
desired set is infeasible, lower-ranked names get less or nothing and the
residual stays in cash (never an exception).

Deterministic: sorted iteration, stable tiebreaks (symbol asc); identical
inputs produce byte-identical ``TargetPortfolio`` JSON. No I/O, no clock.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from dynamo.domain.models.portfolio import (
    PortfolioState,
    ProposedChange,
    RebalanceTrigger,
    TargetPortfolio,
    TargetWeight,
)
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    QualityScore,
    QuantScore,
    Signal,
    SignalStatus,
    Sleeve,
    WatchlistRow,
)
from dynamo.domain.models.risk import Regime, RiskConfig, RiskVerdict
from dynamo.domain.symbols import is_crypto_symbol
from dynamo.portfolio.config import ConstructionConfig
from dynamo.risk.ladder import RungEffects, ballast_multiplier_for, effects_for
from dynamo.risk.sizing import SizingCandidate, SizingContext, size_positions

# Below this weight a position is dropped rather than opened (doc 06 section 4).
MIN_POSITION_WEIGHT = 0.005

_EXIT_ACTIONS = frozenset({Action.SELL})
_TRIM_ACTIONS = frozenset({Action.TRIM})
_RETAIN_ACTIONS = frozenset({Action.BUY, Action.ADD, Action.HOLD})
_ENTRY_ACTIONS = frozenset({Action.BUY, Action.ADD})
_VETO_ACTIONS = frozenset({Action.AVOID, Action.SELL})

# Regime multiplier on the tactical sleeve; combined with the ladder's
# tactical multiplier via min() — the split honors whichever is tighter.
_TRIM_FRACTION = 0.5


@dataclass(frozen=True)
class AdvisoryInputs:
    """All advisory research inputs for one construction run (doc 07 section 1)."""

    as_of: datetime
    watchlist: tuple[WatchlistRow, ...] = ()
    verdicts: Mapping[str, CommitteeVerdict] = field(default_factory=dict)
    quant_scores: Mapping[str, QuantScore] = field(default_factory=dict)
    quality_scores: Mapping[str, QualityScore] = field(default_factory=dict)
    signals: tuple[Signal, ...] = ()
    sigma_annual: Mapping[str, float] = field(default_factory=dict)
    adv_dollar_20d: Mapping[str, float] = field(default_factory=dict)
    sector_by_symbol: Mapping[str, str] = field(default_factory=dict)
    # Meta-model calibrated P(win) per symbol (plan Phase 3). DARK by
    # default: empty unless meta_sizing_enabled is True post-promotion; an
    # empty mapping leaves construction bit-identical to the pre-meta path.
    effective_probability: Mapping[str, float] = field(default_factory=dict)
    # Higher-risk names (crypto, names beyond the core universe) that share
    # ConstructionConfig.risk_sleeve_cap. Empty, or a None cap, = no sleeve.
    risk_sleeve_symbols: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Exclusion:
    """A candidate excluded before/at selection — always logged with a reason."""

    symbol: str
    sleeve: Sleeve | None
    reason: str


@dataclass(frozen=True)
class ConstructionOutput:
    """Full construction result: the target plus its audit trail."""

    target: TargetPortfolio
    exclusions: tuple[Exclusion, ...]
    deferred: tuple[ProposedChange, ...]  # discretionary trades queued (turnover)
    skipped: tuple[ProposedChange, ...]  # below the min-trade noise floor


@dataclass(frozen=True)
class _Lot:
    """Internal selected lot flowing into the compliance pass, in priority order."""

    symbol: str
    sleeve: Sleeve
    desired: float  # pre-compliance desired weight
    is_new: bool
    mandatory_cap: float | None  # e.g. trim to current/2
    strategy_id: str | None
    verdict_id: str | None
    rationale: str
    clamps: tuple[str, ...] = ()
    verdict_clamped: bool = False  # a verdict cap bound in the compliance pass


def _finite_pos(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _core_sleeve(symbol: str) -> Sleeve:
    """The nightly-watchlist sleeve a symbol belongs to (ADR-0013).

    Crypto pairs ride the long-term selection path (ranked watchlist, entry
    verdicts) but are tagged ``Sleeve.CRYPTO`` so lots, orders, and decision
    contexts carry the sleeve and the crypto caps bind in sizing/compliance.
    """
    return Sleeve.CRYPTO if is_crypto_symbol(symbol) else Sleeve.LONG_TERM


def _held_sleeves(state: PortfolioState) -> dict[str, set[Sleeve]]:
    out: dict[str, set[Sleeve]] = {}
    for position in state.positions:
        if position.qty == 0.0:
            continue
        sleeves = {lot.sleeve for lot in position.lots} or {_core_sleeve(position.symbol)}
        out.setdefault(position.symbol, set()).update(sleeves)
    return out


def _regime_tactical_multiplier(regime: Regime, config: ConstructionConfig) -> float:
    if regime is Regime.CRISIS:
        return 0.0
    if regime is Regime.RISK_OFF:
        return config.risk_off_tactical_multiplier
    return 1.0


class RuleBasedConstructor:
    """Phase-1 deterministic constructor behind the ``PortfolioOptimizer`` port."""

    def construct(
        self,
        candidates: AdvisoryInputs,
        verdict: RiskVerdict,
        current: PortfolioState,
        config: ConstructionConfig,
        *,
        risk_config: RiskConfig | None = None,
        current_weights: Mapping[str, float] | None = None,
        turnover_used_mtd: float = 0.0,
        triggers: frozenset[RebalanceTrigger] = frozenset(),
        risk_verdict_ref: str | None = None,
    ) -> TargetPortfolio:
        return self.construct_detailed(
            candidates,
            verdict,
            current,
            config,
            risk_config=risk_config,
            current_weights=current_weights,
            turnover_used_mtd=turnover_used_mtd,
            triggers=triggers,
            risk_verdict_ref=risk_verdict_ref,
        ).target

    def construct_detailed(
        self,
        candidates: AdvisoryInputs,
        verdict: RiskVerdict,
        current: PortfolioState,
        config: ConstructionConfig,
        *,
        risk_config: RiskConfig | None = None,
        current_weights: Mapping[str, float] | None = None,
        turnover_used_mtd: float = 0.0,
        triggers: frozenset[RebalanceTrigger] = frozenset(),
        risk_verdict_ref: str | None = None,
    ) -> ConstructionOutput:
        rcfg = risk_config if risk_config is not None else RiskConfig()
        cur_w = dict(current_weights) if current_weights is not None else current.weights
        cur_w = {s: w for s, w in cur_w.items() if _finite_pos(w)}
        exclusions: list[Exclusion] = []

        vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}
        new_risk = verdict.new_risk_allowed
        effects = effects_for(verdict.ladder)
        regime = verdict.regime.regime

        # ------------------------------------------------ step 1: sleeve split
        investable = max(0.0, min(verdict.gross_exposure_cap, 1.0 - verdict.cash_floor))
        tactical_frac = min(config.tactical_fraction, config.tactical_max)
        tactical_frac *= min(
            effects.tactical_multiplier, _regime_tactical_multiplier(regime, config)
        )
        if not effects.tactical_allowed:
            tactical_frac = 0.0
        tactical_budget = investable * tactical_frac
        long_term_budget = investable - tactical_budget

        # ------------------------------------- step 2: exits / trims / retains
        held = _held_sleeves(current)
        exits: dict[str, str] = {}
        trims: dict[str, str] = {}
        for symbol in sorted(held):
            v = candidates.verdicts.get(symbol)
            if symbol in vetoed:
                exits[symbol] = "risk veto on held symbol"
            elif v is not None and v.action in _EXIT_ACTIONS:
                exits[symbol] = f"committee verdict {v.action.value}"
            elif v is not None and v.action is Action.AVOID:
                exits[symbol] = "committee verdict avoid on held symbol"
            elif v is not None and v.action in _TRIM_ACTIONS:
                trims[symbol] = f"committee verdict {v.action.value}"

        lt_lots, tac_lots = self._select(
            candidates, config, verdict, held, exits, trims, cur_w, new_risk, exclusions
        )

        # -------------------------------- step 3: sizing (delegated cascade)
        lt_lots = self._size_sleeve(
            lt_lots,
            Sleeve.LONG_TERM,
            long_term_budget,
            {},
            candidates,
            verdict,
            rcfg,
            effects.risk_multiplier,
            effects.tactical_multiplier,
            vetoed,
            current.equity,
        )
        lt_weights = {}
        for lot in lt_lots:
            lt_weights[lot.symbol] = lt_weights.get(lot.symbol, 0.0) + lot.desired
        tac_lots = self._size_sleeve(
            tac_lots,
            Sleeve.TACTICAL,
            tactical_budget,
            lt_weights,
            candidates,
            verdict,
            rcfg,
            effects.risk_multiplier,
            effects.tactical_multiplier,
            vetoed,
            current.equity,
        )

        # ------------------------- turnover budget (phase B: defer/skip diffs)
        ordered = [*lt_lots, *tac_lots]
        mandatory = (
            set(exits)
            | set(trims)
            | self._current_breaches(cur_w, verdict, candidates.sector_by_symbol, config)
        )
        ordered, deferred, skipped = self._apply_turnover(
            ordered, cur_w, config, mandatory, turnover_used_mtd, triggers
        )

        # ------------------- step 4 + hard guarantee: verdict compliance pass
        final_lots = self._clamp_to_verdict(
            ordered,
            verdict,
            rcfg,
            candidates.sector_by_symbol,
            risk_sleeve_symbols=candidates.risk_sleeve_symbols,
            risk_sleeve_cap=config.risk_sleeve_cap,
        )

        # ------------------------------ step 5: idle-cash ballast overlay
        if config.ballast_symbol:
            ballast = self._ballast_lot(config, verdict, effects, final_lots, cur_w)
            if ballast is not None:
                final_lots.append(ballast)

        weights = tuple(
            TargetWeight(
                symbol=lot.symbol,
                weight=lot.desired,
                sleeve=lot.sleeve,
                strategy_id=lot.strategy_id,
                verdict_id=lot.verdict_id,
                rationale=lot.rationale,
            )
            for lot in final_lots
            if lot.desired > 0.0
        )
        # The compliance pass caps the total at min(gross cap, 1 - cash floor),
        # so the residual is >= the floor by construction (weights + cash = 1).
        total = math.fsum(w.weight for w in weights)
        cash = 1.0 - total

        changes, executed_turnover = self._diff(
            weights, cur_w, exits, trims, mandatory, final_lots, triggers, config
        )
        attribution = self._cash_attribution(cash, verdict, investable, total)

        ref = risk_verdict_ref or f"rv:{verdict.as_of.isoformat()}"
        target = TargetPortfolio(
            id="pending",
            as_of=candidates.as_of,
            risk_verdict_ref=ref,
            weights=weights,
            cash=cash,
            cash_attribution=attribution,
            changes=changes,
            turnover_used_mtd=turnover_used_mtd + executed_turnover,
        )
        target = target.model_copy(update={"id": _target_id(target)})
        return ConstructionOutput(
            target=target,
            exclusions=tuple(exclusions),
            deferred=tuple(deferred),
            skipped=tuple(skipped),
        )

    # ------------------------------------------------------------- selection
    def _select(
        self,
        inputs: AdvisoryInputs,
        config: ConstructionConfig,
        verdict: RiskVerdict,
        held: Mapping[str, set[Sleeve]],
        exits: Mapping[str, str],
        trims: Mapping[str, str],
        cur_w: Mapping[str, float],
        new_risk: bool,
        exclusions: list[Exclusion],
    ) -> tuple[list[_Lot], list[_Lot]]:
        vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}
        regime = verdict.regime.regime

        # ---- long-term sleeve: ranked watchlist ---------------------------
        lt_rows = sorted(
            (r for r in inputs.watchlist if r.sleeve is Sleeve.LONG_TERM),
            key=lambda r: (r.rank, r.symbol),
        )
        rank_cutoff = math.ceil(config.long_term_top_fraction * len(lt_rows))
        lt_lots: list[_Lot] = []
        lt_symbols: set[str] = set()

        # Retained holdings first (incumbents; doc 07: holds are retained).
        # Crypto lots (Sleeve.CRYPTO) live inside the long-term selection
        # path — same retention/entry rules, crypto caps applied in sizing.
        # The ballast symbol is NEVER auto-retained as an active incumbent:
        # its weight is recomputed by the overlay every run (it may still
        # become an ACTIVE pick via the normal entry path — watchlist rank +
        # entry verdict — in which case the active weight wins and ballast
        # is the residual on top).
        for symbol in sorted(held):
            if symbol in exits or symbol not in cur_w:
                continue
            if config.ballast_symbol and symbol == config.ballast_symbol:
                continue
            if not held[symbol] & {Sleeve.LONG_TERM, Sleeve.CRYPTO}:
                continue
            v = inputs.verdicts.get(symbol)
            if symbol in trims:
                lt_lots.append(
                    _Lot(
                        symbol=symbol,
                        sleeve=_core_sleeve(symbol),
                        desired=cur_w[symbol] * _TRIM_FRACTION,
                        is_new=False,
                        mandatory_cap=cur_w[symbol] * _TRIM_FRACTION,
                        strategy_id=None,
                        verdict_id=v.id if v else None,
                        rationale=f"trim to {_TRIM_FRACTION:.0%} of current ({trims[symbol]})",
                    )
                )
                lt_symbols.add(symbol)
            elif v is None or v.action in _RETAIN_ACTIONS or v.action is Action.NO_ACTION:
                reason = (
                    "retained; no fresh advisory input"
                    if v is None
                    else f"retained on {v.action.value} verdict"
                )
                lt_lots.append(
                    _Lot(
                        symbol=symbol,
                        sleeve=_core_sleeve(symbol),
                        desired=cur_w[symbol],
                        is_new=False,
                        mandatory_cap=None,
                        strategy_id=None,
                        verdict_id=v.id if v else None,
                        rationale=reason,
                    )
                )
                lt_symbols.add(symbol)

        # New long-term entries in rank order.
        for order, row in enumerate(lt_rows):
            symbol = row.symbol
            if symbol in lt_symbols or symbol in exits or symbol in trims:
                continue
            v = inputs.verdicts.get(symbol)
            if symbol in vetoed:
                exclusions.append(Exclusion(symbol, Sleeve.LONG_TERM, "risk veto"))
                continue
            if v is None or v.action not in _ENTRY_ACTIONS:
                action = v.action.value if v else "none"
                exclusions.append(
                    Exclusion(symbol, Sleeve.LONG_TERM, f"no entry verdict (action={action})")
                )
                continue
            if not new_risk:
                exclusions.append(Exclusion(symbol, Sleeve.LONG_TERM, "new_risk_allowed is false"))
                continue
            quality = self._quality_of(symbol, row, inputs)
            if quality is None or quality < config.quality_floor:
                exclusions.append(
                    Exclusion(
                        symbol,
                        Sleeve.LONG_TERM,
                        f"quality {quality if quality is not None else 'missing'} "
                        f"below floor {config.quality_floor:g}",
                    )
                )
                continue
            if order >= rank_cutoff:
                exclusions.append(
                    Exclusion(
                        symbol,
                        Sleeve.LONG_TERM,
                        f"rank outside top {config.long_term_top_fraction:.0%}",
                    )
                )
                continue
            lt_lots.append(
                _Lot(
                    symbol=symbol,
                    sleeve=_core_sleeve(symbol),
                    desired=0.0,  # sized by the cascade
                    is_new=True,
                    mandatory_cap=None,
                    strategy_id=None,
                    verdict_id=v.id,
                    rationale=f"rank {row.rank} {v.action.value} verdict",
                )
            )
            lt_symbols.add(symbol)

        # ---- tactical sleeve: active signals ------------------------------
        active = sorted(
            (s for s in inputs.signals if s.status is SignalStatus.ACTIVE),
            key=lambda s: (-s.strength, s.symbol, s.strategy_id),
        )
        tac_lots: list[_Lot] = []
        tac_symbols: set[str] = set()

        for symbol in sorted(held):
            if symbol in exits or symbol in trims or symbol not in cur_w:
                continue
            if config.ballast_symbol and symbol == config.ballast_symbol:
                continue  # ballast is overlay-governed, never a tactical incumbent
            if Sleeve.TACTICAL not in held[symbol]:
                continue
            v = inputs.verdicts.get(symbol)
            tac_lots.append(
                _Lot(
                    symbol=symbol,
                    sleeve=Sleeve.TACTICAL,
                    desired=cur_w[symbol],
                    is_new=False,
                    mandatory_cap=None,
                    strategy_id=None,
                    verdict_id=v.id if v else None,
                    rationale="retained tactical holding",
                )
            )
            tac_symbols.add(symbol)

        for signal in active:
            symbol = signal.symbol
            if symbol in tac_symbols or symbol in exits or symbol in trims:
                continue
            v = inputs.verdicts.get(symbol)
            if symbol in vetoed:
                exclusions.append(Exclusion(symbol, Sleeve.TACTICAL, "risk veto"))
                continue
            if v is not None and v.action in _VETO_ACTIONS:
                exclusions.append(
                    Exclusion(symbol, Sleeve.TACTICAL, f"committee {v.action.value} vetoes signal")
                )
                continue
            if regime not in config.tactical_entry_regimes:
                exclusions.append(
                    Exclusion(
                        symbol, Sleeve.TACTICAL, f"regime {regime.value} blocks tactical entry"
                    )
                )
                continue
            if not new_risk:
                exclusions.append(Exclusion(symbol, Sleeve.TACTICAL, "new_risk_allowed is false"))
                continue
            tac_lots.append(
                _Lot(
                    symbol=symbol,
                    sleeve=Sleeve.TACTICAL,
                    desired=0.0,
                    is_new=True,
                    mandatory_cap=None,
                    strategy_id=signal.strategy_id,
                    verdict_id=v.id if v else None,
                    rationale=f"signal {signal.strategy_id} strength {signal.strength:.2f}",
                )
            )
            tac_symbols.add(symbol)

        # ---- position-count caps (two lots of one symbol count twice) -----
        tac_cap = min(config.max_tactical_positions, config.max_positions)
        dropped_tac = tac_lots[tac_cap:]
        tac_lots = tac_lots[:tac_cap]
        lt_cap = max(config.max_positions - len(tac_lots), 0)
        dropped_lt = lt_lots[lt_cap:]
        lt_lots = lt_lots[:lt_cap]
        for lot in dropped_lt:
            exclusions.append(Exclusion(lot.symbol, lot.sleeve, "max position count reached"))
        for lot in dropped_tac:
            exclusions.append(
                Exclusion(lot.symbol, lot.sleeve, "max tactical position count reached")
            )
        return lt_lots, tac_lots

    @staticmethod
    def _quality_of(symbol: str, row: WatchlistRow, inputs: AdvisoryInputs) -> float | None:
        score = inputs.quality_scores.get(symbol)
        if score is not None and score.composite is not None:
            return score.composite
        return row.quality_component

    # ---------------------------------------------------------------- sizing
    def _size_sleeve(
        self,
        lots: list[_Lot],
        sleeve: Sleeve,
        sleeve_budget: float,
        other_sleeve_weights: Mapping[str, float],
        inputs: AdvisoryInputs,
        verdict: RiskVerdict,
        rcfg: RiskConfig,
        risk_multiplier: float,
        tactical_multiplier: float,
        vetoed: set[str],
        equity: float,
    ) -> list[_Lot]:
        """Delegate sizing of new/resizable lots to the risk cascade.

        Retained and trim lots keep their carried weights (turnover-friendly);
        only new entries are sized fresh. The final compliance pass is what
        guarantees verdict constraints for every lot either way.
        """
        fixed = [lot for lot in lots if not lot.is_new]
        to_size = [lot for lot in lots if lot.is_new]
        if not to_size:
            return lots

        fixed_total = math.fsum(lot.desired for lot in fixed)
        remaining = max(sleeve_budget - fixed_total, 0.0)
        existing: dict[str, float] = dict(other_sleeve_weights)
        for lot in fixed:
            existing[lot.symbol] = existing.get(lot.symbol, 0.0) + lot.desired

        sizing_candidates: list[SizingCandidate] = []
        for lot in to_size:
            v = inputs.verdicts.get(lot.symbol)
            quant = inputs.quant_scores.get(lot.symbol)
            conviction = v.conviction if v is not None else 0.5
            if lot.sleeve is Sleeve.TACTICAL and v is None:
                strengths = [
                    s.strength
                    for s in inputs.signals
                    if s.symbol == lot.symbol and s.status is SignalStatus.ACTIVE
                ]
                if strengths:
                    conviction = max(strengths)
            sizing_candidates.append(
                SizingCandidate(
                    symbol=lot.symbol,
                    # Crypto lots ride the long-term sleeve budget but carry
                    # Sleeve.CRYPTO (crypto caps bind inside the cascade).
                    sleeve=lot.sleeve,
                    conviction=conviction,
                    sigma_annual=inputs.sigma_annual.get(lot.symbol, float("nan")),
                    adv_dollar_20d=inputs.adv_dollar_20d.get(lot.symbol, float("nan")),
                    expected_excess_return=(
                        quant.expected_excess_return if quant is not None else None
                    ),
                    vol_forecast_annual=quant.vol_forecast if quant is not None else None,
                    sector=inputs.sector_by_symbol.get(lot.symbol),
                    cluster_id=verdict.clusters.get(lot.symbol),
                    # Dark meta path (plan Phase 3): absent from the mapping
                    # -> None -> the tilt uses raw conviction, unchanged.
                    effective_probability=inputs.effective_probability.get(lot.symbol),
                )
            )
        ctx = SizingContext(
            equity=equity,
            existing_weights=existing,
            sector_by_symbol=inputs.sector_by_symbol,
            cluster_by_symbol=verdict.clusters,
            vetoed=frozenset(vetoed),
            sleeve_budget=remaining,
            risk_multiplier=risk_multiplier,
            tactical_multiplier=tactical_multiplier,
        )
        sized = size_positions(sizing_candidates, rcfg, ctx)
        out: list[_Lot] = list(fixed)
        for lot, sp in zip(to_size, sized, strict=True):
            if sp.weight <= 0.0:
                continue
            out.append(
                _Lot(
                    symbol=lot.symbol,
                    sleeve=lot.sleeve,
                    desired=sp.weight,
                    is_new=True,
                    mandatory_cap=None,
                    strategy_id=lot.strategy_id,
                    verdict_id=lot.verdict_id,
                    rationale=lot.rationale,
                    clamps=sp.clamps_applied,
                )
            )
        return out

    # ------------------------------------------------------ turnover budget
    @staticmethod
    def _current_breaches(
        cur_w: Mapping[str, float],
        verdict: RiskVerdict,
        sector_by_symbol: Mapping[str, str],
        config: ConstructionConfig,
    ) -> set[str]:
        """Symbols whose CURRENT weight violates the verdict -> mandatory trades."""
        breaches: set[str] = set()
        vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}
        gross = math.fsum(w for w in cur_w.values() if w > 0.0)
        # With ballast enabled the portfolio deliberately sits AT the invested
        # ceiling, so sub-noise mark-to-market drift would flag every symbol
        # mandatory every other day; the noise floor is the drift band there
        # (the overlay re-trims the residual anyway). Ballast disabled keeps
        # the exact pre-ballast semantics bit-for-bit.
        tol = config.min_trade_weight if config.ballast_symbol else 1e-12
        portfolio_breach = (
            gross > verdict.gross_exposure_cap + tol or 1.0 - gross < verdict.cash_floor - tol
        )
        sector_tot: dict[str, float] = {}
        cluster_tot: dict[str, float] = {}
        for symbol, w in cur_w.items():
            if w <= 0.0:
                continue
            sector = sector_by_symbol.get(symbol)
            if sector is not None:
                sector_tot[sector] = sector_tot.get(sector, 0.0) + w
            cluster = verdict.clusters.get(symbol)
            if cluster is not None:
                cluster_tot[cluster] = cluster_tot.get(cluster, 0.0) + w
        hot_sectors = {
            s
            for s, tot in sector_tot.items()
            if s in verdict.sector_caps and tot > verdict.sector_caps[s] + 1e-12
        }
        hot_clusters = {
            c
            for c, tot in cluster_tot.items()
            if c in verdict.cluster_caps and tot > verdict.cluster_caps[c] + 1e-12
        }
        for symbol, w in cur_w.items():
            cap = verdict.per_position_cap.get(symbol)
            if (
                symbol in vetoed
                or (cap is not None and w > cap + 1e-12)
                or portfolio_breach
                or sector_by_symbol.get(symbol) in hot_sectors
                or verdict.clusters.get(symbol) in hot_clusters
            ):
                breaches.add(symbol)
        return breaches

    def _apply_turnover(
        self,
        lots: list[_Lot],
        cur_w: Mapping[str, float],
        config: ConstructionConfig,
        mandatory: set[str],
        turnover_used_mtd: float,
        triggers: frozenset[RebalanceTrigger],
    ) -> tuple[list[_Lot], list[ProposedChange], list[ProposedChange]]:
        """Defer discretionary trades past the monthly budget; skip noise trades.

        Deferral/skip pins the symbol's target at its current weight; the
        compliance pass afterwards still guarantees every verdict constraint.
        """
        by_symbol: dict[str, float] = {}
        for lot in lots:
            by_symbol[lot.symbol] = by_symbol.get(lot.symbol, 0.0) + lot.desired
        symbols = sorted(set(by_symbol) | set(cur_w))

        deltas = {s: by_symbol.get(s, 0.0) - cur_w.get(s, 0.0) for s in symbols}
        remaining = max(config.turnover_budget_monthly - turnover_used_mtd, 0.0)
        pinned: dict[str, float] = {}  # symbol -> pinned target (current weight)
        deferred: list[ProposedChange] = []
        skipped: list[ProposedChange] = []
        trigger = _primary_trigger(triggers)

        discretionary = sorted(
            (
                s
                for s in symbols
                if s not in mandatory
                and abs(deltas[s]) > 0.0
                # The ballast symbol is overlay-governed and budget-exempt: a
                # held ballast position must never look like a discretionary
                # full exit here (the overlay re-adds its weight afterwards),
                # and funding trims for active picks must never be deferred.
                and not (config.ballast_symbol and s == config.ballast_symbol)
            ),
            key=lambda s: (-abs(deltas[s]), s),
        )
        for symbol in discretionary:
            delta = deltas[symbol]
            change = ProposedChange(
                symbol=symbol,
                current_weight=cur_w.get(symbol, 0.0),
                target_weight=by_symbol.get(symbol, 0.0),
                source="construction",
                trigger=trigger,
                rationale="",
                constraints_applied=(),
            )
            if abs(delta) < config.min_trade_weight:
                pinned[symbol] = cur_w.get(symbol, 0.0)
                skipped.append(change.model_copy(update={"rationale": "below min trade size"}))
                continue
            cost = abs(delta) / 2.0  # one-way turnover
            if cost > remaining:
                pinned[symbol] = cur_w.get(symbol, 0.0)
                deferred.append(
                    change.model_copy(update={"rationale": "turnover budget exhausted"})
                )
                continue
            remaining -= cost

        if not pinned:
            return lots, deferred, skipped

        out: list[_Lot] = []
        seen_pinned: set[str] = set()
        for lot in lots:
            if lot.symbol not in pinned:
                out.append(lot)
                continue
            if lot.symbol in seen_pinned:
                continue  # collapse multi-lot symbol into one pinned lot
            seen_pinned.add(lot.symbol)
            out.append(
                _Lot(
                    symbol=lot.symbol,
                    sleeve=lot.sleeve,
                    desired=pinned[lot.symbol],
                    is_new=False,
                    mandatory_cap=None,
                    strategy_id=lot.strategy_id,
                    verdict_id=lot.verdict_id,
                    rationale="pinned at current weight (deferred/skipped)",
                )
            )
        # Pinned held symbols with no lot at all (e.g. deferred full exit is
        # impossible — exits are mandatory — but a deferred new buy needs no lot).
        return out, deferred, skipped

    # --------------------------------------------- verdict compliance pass
    @staticmethod
    def _clamp_to_verdict(
        lots: list[_Lot],
        verdict: RiskVerdict,
        rcfg: RiskConfig,
        sector_by_symbol: Mapping[str, str],
        *,
        risk_sleeve_symbols: frozenset[str] = frozenset(),
        risk_sleeve_cap: float | None = None,
    ) -> list[_Lot]:
        """Greedy, priority-ordered allocation under every verdict constraint.

        Guarantees: combined per-symbol weight <= verdict cap, sector and
        cluster totals <= caps, gross <= gross cap, total <= 1 - cash floor.
        Lower-priority lots absorb the infeasibility (fewer names, more cash).
        """
        symbol_alloc: dict[str, float] = {}
        sector_alloc: dict[str, float] = {}
        cluster_alloc: dict[str, float] = {}
        crypto_alloc = 0.0  # crypto sleeve total (ADR-0013)
        risk_alloc = 0.0  # bankroll risk sleeve total (dynamo.portfolio.bankroll)
        total = 0.0
        out: list[_Lot] = []
        investable = max(0.0, min(verdict.gross_exposure_cap, 1.0 - verdict.cash_floor))
        vetoed = {v.symbol for v in verdict.vetoes if v.symbol != "*"}

        for lot in lots:
            clamps: list[str] = list(lot.clamps)
            w = lot.desired
            verdict_clamped = False
            if lot.symbol in vetoed:
                w = 0.0
                verdict_clamped = True
                clamps.append("veto")
            if not math.isfinite(w) or w < 0.0:
                w = 0.0
                clamps.append("invalid_input")

            symbol_cap = verdict.per_position_cap.get(lot.symbol, rcfg.per_position_hard)
            lot_is_crypto = is_crypto_symbol(lot.symbol)
            if lot_is_crypto:
                # Crypto caps are hard bounds even for retained lots
                # (ADR-0013): per-pair, then the collective sleeve below.
                symbol_cap = min(symbol_cap, rcfg.crypto_max_position)
            avail = symbol_cap - symbol_alloc.get(lot.symbol, 0.0)
            if w > avail:
                w = max(avail, 0.0)
                verdict_clamped = True
                clamps.append("per_position_cap")

            if lot_is_crypto:
                avail = rcfg.crypto_sleeve_cap - crypto_alloc
                if w > avail:
                    w = max(avail, 0.0)
                    verdict_clamped = True
                    clamps.append("crypto_sleeve_cap")

            lot_in_risk_sleeve = risk_sleeve_cap is not None and lot.symbol in risk_sleeve_symbols
            if lot_in_risk_sleeve:
                avail = (risk_sleeve_cap or 0.0) - risk_alloc
                if w > avail:
                    w = max(avail, 0.0)
                    verdict_clamped = True
                    clamps.append("risk_sleeve_cap")

            sector = sector_by_symbol.get(lot.symbol)
            if sector is not None and sector in verdict.sector_caps:
                avail = verdict.sector_caps[sector] - sector_alloc.get(sector, 0.0)
                if w > avail:
                    w = max(avail, 0.0)
                    verdict_clamped = True
                    clamps.append("sector_cap")

            cluster = verdict.clusters.get(lot.symbol)
            if cluster is not None and cluster in verdict.cluster_caps:
                avail = verdict.cluster_caps[cluster] - cluster_alloc.get(cluster, 0.0)
                if w > avail:
                    w = max(avail, 0.0)
                    verdict_clamped = True
                    clamps.append("cluster_cap")

            avail = investable - total
            if w > avail:
                w = max(avail, 0.0)
                verdict_clamped = True
                clamps.append("gross_cap")

            if lot.mandatory_cap is not None and w > lot.mandatory_cap:
                w = lot.mandatory_cap
                clamps.append("mandatory_trim")

            if 0.0 < w < MIN_POSITION_WEIGHT:
                w = 0.0
                clamps.append("min_weight_floor")

            if w > 0.0:
                symbol_alloc[lot.symbol] = symbol_alloc.get(lot.symbol, 0.0) + w
                if sector is not None:
                    sector_alloc[sector] = sector_alloc.get(sector, 0.0) + w
                if cluster is not None:
                    cluster_alloc[cluster] = cluster_alloc.get(cluster, 0.0) + w
                if lot_is_crypto:
                    crypto_alloc += w
                if lot_in_risk_sleeve:
                    risk_alloc += w
                total += w
            out.append(
                _Lot(
                    symbol=lot.symbol,
                    sleeve=lot.sleeve,
                    desired=w,
                    is_new=lot.is_new,
                    mandatory_cap=lot.mandatory_cap,
                    strategy_id=lot.strategy_id,
                    verdict_id=lot.verdict_id,
                    rationale=lot.rationale,
                    clamps=tuple(clamps),
                    verdict_clamped=verdict_clamped,
                )
            )
        return out

    # ------------------------------------------------ idle-cash ballast
    @staticmethod
    def _ballast_lot(
        config: ConstructionConfig,
        verdict: RiskVerdict,
        effects: RungEffects,
        final_lots: list[_Lot],
        cur_w: Mapping[str, float],
    ) -> _Lot | None:
        """Idle-cash ballast overlay: the post-compliance residual, governed
        by the same risk posture as everything else.

        Semantics (deliberate decisions, tested):

        - Ballast is a CASH SUBSTITUTE, not an active bet. It is computed
          AFTER the compliance pass as the residual between the invested
          total and ``1 - max(verdict.cash_floor, config.ballast_cash_floor)``
          (bounded by the gross cap), so by construction it can never crowd
          out an active pick, breach the cash floor, or breach the gross cap.
        - It deliberately bypasses per-position/sector/cluster caps: those
          govern active bets, and the residual would otherwise be
          un-investable. This is exactly why the production default is the
          diversified index (SPY): a single name carrying uncapped residual
          weight is not ballast, it is a concentrated bet.
        - Ladder governance uses ballast's OWN, much deeper schedule
          (:data:`dynamo.risk.ladder.BALLAST_RUNG_MULTIPLIERS`, overridable
          per run via ``config.ballast_ladder_multipliers``) and NOT the
          active rung's ``risk_multiplier`` — campaign findings item 20. A
          -5% portfolio drawdown is noise for a passive index floor: the
          control run cut invested exposure 94.9% -> 47.7% at DD5, rebought
          higher two weeks later, and gave back most of its benchmark gap.
          Ballast is therefore untouched at DD5/DD8/DD10, trimmed modestly at
          DD15 (0.75, on top of that rung's 0.5 gross override) and fully
          exited at DD20. ACTIVE positions' ladder behavior is unchanged —
          this function is the only consumer of the ballast table.
        - Every hard protection still binds: the residual is bounded by
          ``verdict.gross_exposure_cap`` and the cash floor BY CONSTRUCTION
          (so DD15's 0.5 gross override and DD20's 0.0 de-risk regardless of
          the table); when increases are blocked (``new_risk_allowed`` false,
          a rung with ``new_positions_allowed`` false, or a degraded verdict)
          existing ballast is HELD but never increased — mirroring how active
          positions are retained while no new risk opens; and a risk veto on
          the ballast symbol zeroes it (exit).
        - If the ballast symbol also earns an ACTIVE weight, the active
          weight wins and ballast is the residual on top (merged per-symbol
          at execution).
        - Churn guard: when the whole-symbol move is below
          ``min_trade_weight``, ballast pins at its current weight (bounded
          by the residual so compliance trims still happen).
        """
        symbol = config.ballast_symbol
        active_total = math.fsum(lot.desired for lot in final_lots if lot.desired > 0.0)
        floor = max(verdict.cash_floor, config.ballast_cash_floor)
        investable = max(0.0, min(verdict.gross_exposure_cap, 1.0 - floor))
        residual = max(0.0, investable - active_total)
        active_sym = math.fsum(
            lot.desired for lot in final_lots if lot.symbol == symbol and lot.desired > 0.0
        )
        held_ballast = max(0.0, cur_w.get(symbol, 0.0) - active_sym)
        vetoed = any(v.symbol == symbol for v in verdict.vetoes)
        increase_ok = (
            verdict.new_risk_allowed and effects.new_positions_allowed and not verdict.degraded
        )
        # Ballast's own ladder schedule, not the active risk multiplier: a
        # CEILING on the ballast weight, so it binds whether ballast is being
        # sized up or merely held (a deep rung must be able to TRIM a held
        # floor, not just refuse to add to it).
        ballast_multiplier = ballast_multiplier_for(
            verdict.ladder, config.ballast_ladder_multipliers
        )
        ceiling = residual * ballast_multiplier
        if vetoed:
            desired = 0.0
            rationale = "ballast exited: risk veto on ballast symbol"
        elif increase_ok:
            desired = ceiling
            rationale = (
                f"idle-cash ballast: residual {residual:.4f} above the "
                f"{floor:.0%} cash floor x ballast ladder multiplier "
                f"{ballast_multiplier:g} ({verdict.ladder.value})"
            )
        else:
            desired = min(held_ballast, ceiling)
            rationale = "ballast held, not increased (new risk blocked)"
        if not vetoed:
            current_total = cur_w.get(symbol, 0.0)
            if abs((active_sym + desired) - current_total) < config.min_trade_weight:
                desired = min(held_ballast, ceiling)
        if desired < MIN_POSITION_WEIGHT:
            return None  # dust never opens; a held dust position exits via diff
        return _Lot(
            symbol=symbol,
            sleeve=Sleeve.BALLAST,
            desired=desired,
            is_new=held_ballast <= 0.0,
            mandatory_cap=None,
            strategy_id="ballast",
            verdict_id=None,
            rationale=rationale,
        )

    # ------------------------------------------------------------------ diff
    @staticmethod
    def _diff(
        weights: tuple[TargetWeight, ...],
        cur_w: Mapping[str, float],
        exits: Mapping[str, str],
        trims: Mapping[str, str],
        mandatory: set[str],
        final_lots: list[_Lot],
        triggers: frozenset[RebalanceTrigger],
        config: ConstructionConfig,
    ) -> tuple[tuple[ProposedChange, ...], float]:
        target_by_symbol: dict[str, float] = {}
        for w in weights:
            target_by_symbol[w.symbol] = target_by_symbol.get(w.symbol, 0.0) + w.weight
        clamps_by_symbol: dict[str, tuple[str, ...]] = {}
        source_by_symbol: dict[str, str] = {}
        verdict_clamped: set[str] = set()
        for lot in final_lots:
            clamps_by_symbol.setdefault(lot.symbol, lot.clamps)
            source_by_symbol.setdefault(
                lot.symbol,
                lot.strategy_id or lot.verdict_id or lot.sleeve.value,
            )
            if lot.verdict_clamped:
                verdict_clamped.add(lot.symbol)
        default_trigger = _primary_trigger(triggers)

        changes: list[ProposedChange] = []
        executed_turnover = 0.0
        for symbol in sorted(set(target_by_symbol) | set(cur_w)):
            current = cur_w.get(symbol, 0.0)
            target = target_by_symbol.get(symbol, 0.0)
            delta = target - current
            # A reduction forced by a binding verdict cap is a compliance
            # clamp: mandatory and exempt from the turnover budget.
            is_mandatory = symbol in mandatory or (delta < 0.0 and symbol in verdict_clamped)
            if abs(delta) < config.min_trade_weight and not is_mandatory:
                continue
            if delta == 0.0:
                continue
            if symbol in exits:
                trigger = RebalanceTrigger.VERDICT_CHANGE
                rationale = exits[symbol]
            elif symbol in trims:
                trigger = RebalanceTrigger.VERDICT_CHANGE
                rationale = trims[symbol]
            elif is_mandatory:
                trigger = RebalanceTrigger.RISK_BREACH
                rationale = "clamp to risk compliance"
            else:
                trigger = default_trigger
                rationale = "rebalance toward constructed target"
            changes.append(
                ProposedChange(
                    symbol=symbol,
                    current_weight=current,
                    target_weight=target,
                    source=source_by_symbol.get(symbol, "construction"),
                    trigger=trigger,
                    rationale=rationale,
                    constraints_applied=clamps_by_symbol.get(symbol, ()),
                )
            )
            # Ballast trades never charge the monthly turnover budget: the
            # ballast symbol is a cash substitute, and charging its funding
            # trims would starve the active picks they exist to fund.
            if not (config.ballast_symbol and symbol == config.ballast_symbol):
                executed_turnover += abs(delta) / 2.0
        return tuple(changes), executed_turnover

    # ------------------------------------------------------ cash attribution
    @staticmethod
    def _cash_attribution(
        cash: float, verdict: RiskVerdict, investable: float, total: float
    ) -> dict[str, float]:
        """Decompose cash into floor / vol_target / regime / scarcity (doc 07 s4)."""
        floor = verdict.cash_floor
        gap = max(0.0, (1.0 - floor) - investable)
        regime_part = gap if verdict.regime.regime in (Regime.RISK_OFF, Regime.CRISIS) else 0.0
        vol_part = gap - regime_part
        scarcity = max(cash - floor - gap, 0.0)
        return {
            "floor": floor,
            "vol_target": vol_part,
            "regime": regime_part,
            "scarcity": scarcity,
        }


def _primary_trigger(triggers: frozenset[RebalanceTrigger]) -> RebalanceTrigger:
    priority = (
        RebalanceTrigger.RISK_BREACH,
        RebalanceTrigger.VERDICT_CHANGE,
        RebalanceTrigger.REGIME_CHANGE,
        RebalanceTrigger.VOL_TARGET,
        RebalanceTrigger.DRIFT,
        RebalanceTrigger.SCHEDULED,
    )
    for t in priority:
        if t in triggers:
            return t
    return RebalanceTrigger.SCHEDULED


def _target_id(target: TargetPortfolio) -> str:
    payload = target.model_dump_json(exclude={"id"})
    return "tp-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
