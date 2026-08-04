"""Shared builders for portfolio-construction tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

from dynamo.domain.models.portfolio import Lot, PortfolioState, Position
from dynamo.domain.models.research import (
    Action,
    CommitteeVerdict,
    QualityScore,
    Signal,
    SignalStatus,
    Sleeve,
    WatchlistRow,
)
from dynamo.domain.models.risk import (
    LadderRung,
    Regime,
    RegimeState,
    RiskMetrics,
    RiskVerdict,
    Veto,
)
from dynamo.portfolio.constructor import AdvisoryInputs

AS_OF = datetime(2026, 8, 3)
D = date(2026, 8, 3)


def regime_state(regime: Regime = Regime.NEUTRAL) -> RegimeState:
    return RegimeState(as_of=D, regime=regime, composite_score=0.0)


def make_verdict(
    *,
    regime: Regime = Regime.NEUTRAL,
    ladder: LadderRung = LadderRung.NORMAL,
    new_risk: bool = True,
    gross_cap: float = 0.95,
    cash_floor: float = 0.05,
    per_position_cap: Mapping[str, float] | None = None,
    sector_caps: Mapping[str, float] | None = None,
    cluster_caps: Mapping[str, float] | None = None,
    clusters: Mapping[str, str] | None = None,
    vetoed: Sequence[str] = (),
) -> RiskVerdict:
    return RiskVerdict(
        as_of=AS_OF,
        regime=regime_state(regime),
        ladder=ladder,
        new_risk_allowed=new_risk,
        gross_exposure_cap=gross_cap,
        net_exposure_cap=gross_cap,
        cash_floor=cash_floor,
        per_position_cap=dict(per_position_cap or {}),
        sector_caps=dict(sector_caps or {}),
        cluster_caps=dict(cluster_caps or {}),
        clusters=dict(clusters or {}),
        vetoes=tuple(
            Veto(symbol=s, rule="test", reason="test veto") for s in vetoed
        ),
        metrics=RiskMetrics(as_of=AS_OF),
    )


def make_state(
    weights: Mapping[str, float],
    equity: float = 100_000.0,
    sleeves: Mapping[str, Sleeve] | None = None,
) -> PortfolioState:
    """State whose book weights equal ``weights`` (price 100, qty = w*E/100)."""
    positions = []
    invested = 0.0
    for symbol in sorted(weights):
        w = weights[symbol]
        qty = w * equity / 100.0
        sleeve = (sleeves or {}).get(symbol, Sleeve.LONG_TERM)
        invested += w * equity
        positions.append(
            Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=100.0,
                lots=(
                    Lot(
                        symbol=symbol,
                        sleeve=sleeve,
                        qty=qty,
                        entry_price=100.0,
                        entry_date=D,
                    ),
                ),
            )
        )
    return PortfolioState(
        as_of=AS_OF,
        equity=equity,
        cash=equity - invested,
        positions=tuple(positions),
        high_water_mark=equity,
    )


def make_row(
    symbol: str,
    rank: int,
    sleeve: Sleeve = Sleeve.LONG_TERM,
    quality: float | None = 80.0,
) -> WatchlistRow:
    return WatchlistRow(
        as_of=D,
        sleeve=sleeve,
        symbol=symbol,
        rank=rank,
        blend_score=1.0 / (1 + rank),
        quality_component=quality,
    )


def make_cverdict(
    symbol: str, action: Action, conviction: float = 0.6
) -> CommitteeVerdict:
    return CommitteeVerdict(
        id=f"cv-{symbol}",
        symbol=symbol,
        as_of=D,
        action=action,
        conviction=conviction,
        thesis="t",
        bull_case="b",
        bear_case="b",
    )


def make_signal(symbol: str, strength: float = 0.6, strategy: str = "strat1") -> Signal:
    return Signal(
        strategy_id=strategy,
        symbol=symbol,
        as_of=D,
        strength=strength,
        status=SignalStatus.ACTIVE,
    )


def make_quality(symbol: str, composite: float | None) -> QualityScore:
    return QualityScore(symbol=symbol, as_of=D, composite=composite)


def advisory(
    *,
    watchlist: Sequence[WatchlistRow] = (),
    verdicts: Sequence[CommitteeVerdict] = (),
    signals: Sequence[Signal] = (),
    quality: Sequence[QualityScore] = (),
    symbols: Sequence[str] = (),
    sigma: float = 0.20,
    adv: float = 1e9,
    sectors: Mapping[str, str] | None = None,
) -> AdvisoryInputs:
    universe = set(symbols)
    universe |= {r.symbol for r in watchlist}
    universe |= {v.symbol for v in verdicts}
    universe |= {s.symbol for s in signals}
    return AdvisoryInputs(
        as_of=AS_OF,
        watchlist=tuple(watchlist),
        verdicts={v.symbol: v for v in verdicts},
        quant_scores={},
        quality_scores={q.symbol: q for q in quality},
        signals=tuple(signals),
        sigma_annual=dict.fromkeys(sorted(universe), sigma),
        adv_dollar_20d=dict.fromkeys(sorted(universe), adv),
        sector_by_symbol=dict(sectors or {}),
    )
