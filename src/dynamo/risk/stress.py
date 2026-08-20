"""Deterministic stress suite: historical window replays + synthetic shocks.

Design: docs/design/06-risk-engine.md section 6. Results are in equity terms
with drawdown-ladder implications; a window absent from the provided history
yields a StressResult noting it was skipped (never silently dropped).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np

from dynamo.domain.models.risk import LadderRung, StressResult
from dynamo.risk.ladder import rung_for_drawdown
from dynamo.risk.types import FloatArray, ReturnsPanel

DEFENSIVE_FLOOR_BREACH = -0.15  # doc 06 section 6: breach -> sizing warning
RATE_SHOCK_BPS = 200.0
TLT_RETURN_PER_100BPS = -0.17  # long-duration proxy: ~17y duration
SINGLE_NAME_SHOCK = -0.40
MARKET_GAP_SHOCK = -0.10


@dataclass(frozen=True)
class ScenarioWindow:
    name: str
    start: date
    end: date


HISTORICAL_WINDOWS: tuple[ScenarioWindow, ...] = (
    ScenarioWindow("2008_q4", date(2008, 9, 1), date(2008, 11, 30)),
    ScenarioWindow("2010_flash_crash_week", date(2010, 5, 3), date(2010, 5, 10)),
    ScenarioWindow("2011_us_downgrade", date(2011, 8, 1), date(2011, 8, 31)),
    ScenarioWindow("2015_aug_selloff", date(2015, 8, 17), date(2015, 8, 28)),
    ScenarioWindow("2018_q4", date(2018, 10, 1), date(2018, 12, 24)),
    ScenarioWindow("2020_covid_crash", date(2020, 2, 19), date(2020, 3, 23)),
    ScenarioWindow("2022_rate_shock", date(2022, 1, 3), date(2022, 6, 16)),
)


def _ladder_implication(
    impact_pct: float, equity: float, high_water_mark: float
) -> LadderRung | None:
    if not (np.isfinite(equity) and equity > 0.0):
        return None
    hwm = high_water_mark if np.isfinite(high_water_mark) and high_water_mark > 0.0 else equity
    hwm = max(hwm, equity)
    stressed = equity * (1.0 + impact_pct)
    return rung_for_drawdown(stressed / hwm - 1.0)


def _compound(col: FloatArray) -> float:
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return 0.0
    return float(np.prod(1.0 + finite) - 1.0)


def replay_window(
    window: ScenarioWindow,
    panel: ReturnsPanel,
    weights: Mapping[str, float],
    equity: float,
    high_water_mark: float,
    market_proxy: str = "SPY",
) -> StressResult:
    """Replay one historical window across current holdings."""
    sliced = panel.window(window.start, window.end)
    if sliced.n_sessions == 0:
        return StressResult(
            scenario=window.name,
            equity_impact_pct=0.0,
            ladder_implication=None,
            details={"skipped": 1.0},
        )

    proxy_return: float | None = None
    proxy_col = sliced.column(market_proxy)
    if proxy_col is not None:
        proxy_return = _compound(proxy_col)

    impact = 0.0
    proxied = 0
    details: dict[str, float] = {}
    for symbol, weight in weights.items():
        if weight == 0.0 or not np.isfinite(weight):
            continue
        col = sliced.column(symbol)
        if col is not None and np.isfinite(col).any():
            sym_return = _compound(col)
        elif proxy_return is not None:
            sym_return = proxy_return
            proxied += 1
        else:
            sym_return = 0.0
            proxied += 1
        details[symbol] = sym_return
        impact += float(weight) * sym_return

    details["sessions"] = float(sliced.n_sessions)
    if proxied:
        details["proxied_symbols"] = float(proxied)
    if impact <= DEFENSIVE_FLOOR_BREACH:
        details["breaches_defensive_floor"] = 1.0
    return StressResult(
        scenario=window.name,
        equity_impact_pct=impact,
        ladder_implication=_ladder_implication(impact, equity, high_water_mark),
        details=details,
    )


def _tlt_beta(panel: ReturnsPanel, symbol: str, tlt_symbol: str) -> float | None:
    sym_col = panel.column(symbol)
    tlt_col = panel.column(tlt_symbol)
    if sym_col is None or tlt_col is None:
        return None
    mask = np.isfinite(sym_col) & np.isfinite(tlt_col)
    if int(mask.sum()) < 60:
        return None
    x = tlt_col[mask]
    y = sym_col[mask]
    var = float(np.var(x))
    if var <= 0.0:
        return None
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return cov / var


def synthetic_scenarios(
    panel: ReturnsPanel,
    weights: Mapping[str, float],
    equity: float,
    high_water_mark: float,
    tlt_symbol: str = "TLT",
) -> list[StressResult]:
    """Synthetic shocks: -10% gap with corr -> 1, +200bps rates, single name -40%."""
    results: list[StressResult] = []
    finite_weights = {s: float(w) for s, w in weights.items() if np.isfinite(w) and w != 0.0}
    gross = float(sum(abs(w) for w in finite_weights.values()))

    gap_impact = MARKET_GAP_SHOCK * gross
    results.append(
        StressResult(
            scenario="synthetic_gap_down_10pct_corr1",
            equity_impact_pct=gap_impact,
            ladder_implication=_ladder_implication(gap_impact, equity, high_water_mark),
            details={"gross_exposure": gross},
        )
    )

    tlt_shock = TLT_RETURN_PER_100BPS * (RATE_SHOCK_BPS / 100.0)
    rate_details: dict[str, float] = {"tlt_shock_return": tlt_shock}
    rate_impact = 0.0
    betas_found = 0
    for symbol, weight in finite_weights.items():
        beta = _tlt_beta(panel, symbol, tlt_symbol)
        if beta is None:
            continue
        betas_found += 1
        rate_details[f"beta_{symbol}"] = beta
        rate_impact += weight * beta * tlt_shock
    if betas_found == 0:
        results.append(
            StressResult(
                scenario="synthetic_rates_up_200bps",
                equity_impact_pct=0.0,
                ladder_implication=None,
                details={"skipped": 1.0},
            )
        )
    else:
        if rate_impact <= DEFENSIVE_FLOOR_BREACH:
            rate_details["breaches_defensive_floor"] = 1.0
        results.append(
            StressResult(
                scenario="synthetic_rates_up_200bps",
                equity_impact_pct=rate_impact,
                ladder_implication=_ladder_implication(rate_impact, equity, high_water_mark),
                details=rate_details,
            )
        )

    largest = max(finite_weights.values(), default=0.0)
    single_impact = SINGLE_NAME_SHOCK * max(largest, 0.0)
    results.append(
        StressResult(
            scenario="synthetic_single_name_down_40pct",
            equity_impact_pct=single_impact,
            ladder_implication=_ladder_implication(single_impact, equity, high_water_mark),
            details={"largest_weight": max(largest, 0.0)},
        )
    )
    return results


def run_stress_suite(
    panel: ReturnsPanel,
    weights: Mapping[str, float],
    equity: float,
    high_water_mark: float,
    tlt_symbol: str = "TLT",
    market_proxy: str = "SPY",
) -> list[StressResult]:
    """Full deterministic stress suite: all historical replays + synthetics."""
    results = [
        replay_window(w, panel, weights, equity, high_water_mark, market_proxy)
        for w in HISTORICAL_WINDOWS
    ]
    results.extend(synthetic_scenarios(panel, weights, equity, high_water_mark, tlt_symbol))
    return results
