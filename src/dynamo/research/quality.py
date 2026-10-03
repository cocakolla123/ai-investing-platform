"""Long-term quality score — doc 05 section 2.

Seven sub-scores (0-100 each) from point-in-time XBRL facts (``filed_at`` <=
as_of; TTM assembled from the four most recent *filed* quarters), expressed as
sector-relative percentiles. ``QualityScore = sum(weight_i x subscore_i)``.

Null handling: a sub-score with insufficient data is ``None``; the remaining
weights renormalize. More than ``max_null_subscores`` (default 2) nulls means
no composite — flagged, never silently defaulted.

Optional qualitative inputs (committee dossier reads) blend 50/50 into the
business-quality (``moat``) and management (``capital_allocation``) sub-scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from dynamo.domain.models.research import QualityScore
from dynamo.research.quant.catalog import (
    _fcf_ttm,  # point-in-time helpers shared with the feature catalog
    _rows,
    dt_naive_utc,
    gross_margin_stability,
    insider_net_buy_signed,
    latest_value,
    n_filed_quarters,
    naive_utc_ts,
    pit_close,
    pit_facts,
    quarterly_series,
    revenue_tag,
    roic_ttm,
    share_count_yoy,
    shares_outstanding,
    to_ts,
    ttm_sum,
)

SUBSCORE_NAMES = (
    "business_quality",
    "revenue_growth",
    "profitability",
    "cash_flow",
    "valuation",
    "balance_sheet",
    "management",
)

# metric -> (subscore, higher_is_better)
_HIGH = True
_LOW = False

__all__ = ["QualityConfig", "compute_quality_scores"]


class QualityConfig(BaseModel):
    """Weights 20/15/15/15/15/10/10 (doc 05 section 2.1); promotion-gated."""

    model_config = ConfigDict(frozen=True)

    weights: dict[str, float] = {
        "business_quality": 0.20,
        "revenue_growth": 0.15,
        "profitability": 0.15,
        "cash_flow": 0.15,
        "valuation": 0.15,
        "balance_sheet": 0.10,
        "management": 0.10,
    }
    min_quarters: int = 8
    max_null_subscores: int = 2

    @model_validator(mode="after")
    def _check(self) -> QualityConfig:
        if set(self.weights) != set(SUBSCORE_NAMES):
            raise ValueError(f"weights must cover exactly {SUBSCORE_NAMES}")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1")
        return self

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nan() -> float:
    return float("nan")


def _price_panel(bars: pd.DataFrame | None, as_of: pd.Timestamp) -> pd.DataFrame:
    if bars is None or bars.empty or not {"symbol", "ts", "close"} <= set(bars.columns):
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    df = bars.copy()
    # Repository bars are tz-aware; as_of is naive UTC (see naive_utc_ts).
    df["ts"] = dt_naive_utc(df["ts"])
    df = df[df["ts"] <= as_of]
    if df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    return df.pivot_table(index="ts", columns="symbol", values="close", aggfunc="last").sort_index()


def _enterprise_value(
    facts: pd.DataFrame, price: pd.DataFrame, sym: str, at: pd.Timestamp
) -> float:
    sh = shares_outstanding(facts, sym, at)
    px = pit_close(price, sym, at)
    if math.isnan(sh) or math.isnan(px):
        return _nan()
    ltd = latest_value(facts, sym, "LongTermDebt", at)
    cash = latest_value(facts, sym, "CashAndCashEquivalentsAtCarryingValue", at)
    return sh * px + (0.0 if math.isnan(ltd) else ltd) - (0.0 if math.isnan(cash) else cash)


def _valuation_history(
    facts: pd.DataFrame, price: pd.DataFrame, sym: str, as_of: pd.Timestamp, n_quarters: int = 12
) -> tuple[list[float], list[float]]:
    """(ev_ebit history, fcf_yield history) replayed at past quarter filings."""
    rev_tag = revenue_tag(facts, sym, as_of)
    filings = (
        _rows(facts, (facts["symbol"] == sym) & (facts["tag"] == rev_tag))
        .sort_values(["period_end", "filed_at"])
        .groupby("period_end", sort=True)
        .last()["filed_at"]
    )
    ev_ebit_hist: list[float] = []
    fcf_yield_hist: list[float] = []
    for filed_at in filings.iloc[-n_quarters:]:
        at = to_ts(filed_at)
        sub = _rows(facts, facts["filed_at"] <= at)
        ev = _enterprise_value(sub, price, sym, at)
        if math.isnan(ev) or ev <= 0:
            continue
        ebit = ttm_sum(sub, sym, "OperatingIncomeLoss", at)
        if not math.isnan(ebit) and ebit > 0:
            ev_ebit_hist.append(ev / ebit)
        fcf = _fcf_ttm(sub, sym, at)
        if not math.isnan(fcf):
            fcf_yield_hist.append(fcf / ev)
    return ev_ebit_hist, fcf_yield_hist


def _own_history_pctile(current: float, history: Sequence[float], higher_is_better: bool) -> float:
    if math.isnan(current) or len(history) < 4:
        return _nan()
    if higher_is_better:
        return 100.0 * float(np.mean([current >= h for h in history]))
    return 100.0 * float(np.mean([current <= h for h in history]))


def _symbol_metrics(
    facts: pd.DataFrame,
    price: pd.DataFrame,
    insider: pd.DataFrame | None,
    sym: str,
    as_of: pd.Timestamp,
    min_quarters: int,
) -> dict[str, float]:
    """Raw metric values (NaN = unavailable). Sign conventions noted per metric."""
    m: dict[str, float] = {}
    quarters = n_filed_quarters(facts, sym, as_of)
    rev_tag = revenue_tag(facts, sym, as_of)
    rev = ttm_sum(facts, sym, rev_tag, as_of)
    ebit = ttm_sum(facts, sym, "OperatingIncomeLoss", as_of)
    ni = ttm_sum(facts, sym, "NetIncomeLoss", as_of)
    fcf = _fcf_ttm(facts, sym, as_of)
    roic = roic_ttm(facts, sym, as_of)

    # business quality / moat (needs >= min_quarters filed quarters)
    if quarters >= min_quarters:
        m["gm_stability"] = gross_margin_stability(facts, sym, as_of)  # higher (less neg) better
        gp = quarterly_series(facts, sym, "GrossProfit", as_of).iloc[-12:]
        rv = quarterly_series(facts, sym, rev_tag, as_of).iloc[-12:]
        gm_q = (gp / rv.where(rv > 0)).dropna()
        m["roic_persistence"] = (
            float((gm_q > gm_q.median()).mean()) if len(gm_q) >= min_quarters else _nan()
        )
    else:
        m["gm_stability"] = _nan()
        m["roic_persistence"] = _nan()

    # revenue growth
    if quarters >= min_quarters and not math.isnan(rev):
        prev = ttm_sum(facts, sym, rev_tag, as_of, offset_quarters=4)
        m["rev_yoy"] = rev / prev - 1.0 if prev > 0 else _nan()
        prev2 = ttm_sum(facts, sym, rev_tag, as_of, offset_quarters=8)
        if prev > 0 and prev2 > 0:
            m["rev_accel"] = (rev / prev - 1.0) - (prev / prev2 - 1.0)
        else:
            m["rev_accel"] = _nan()
        old = ttm_sum(facts, sym, rev_tag, as_of, offset_quarters=12)
        m["rev_cagr_3y"] = (rev / old) ** (1.0 / 3.0) - 1.0 if old > 0 and rev > 0 else _nan()
    else:
        m["rev_yoy"] = m["rev_accel"] = m["rev_cagr_3y"] = _nan()

    # profitability
    m["op_margin"] = ebit / rev if rev > 0 and not math.isnan(ebit) else _nan()
    m["net_margin"] = ni / rev if rev > 0 and not math.isnan(ni) else _nan()
    m["roic"] = roic

    # cash flow
    m["fcf_conversion"] = fcf / ni if ni > 0 and not math.isnan(fcf) else _nan()
    m["fcf_margin"] = fcf / rev if rev > 0 and not math.isnan(fcf) else _nan()
    ev = _enterprise_value(facts, price, sym, as_of)
    m["fcf_yield"] = fcf / ev if ev > 0 and not math.isnan(fcf) else _nan()

    # valuation (sector part; own-history part handled separately)
    m["ev_ebit"] = ev / ebit if ev > 0 and not math.isnan(ebit) and ebit > 0 else _nan()

    # balance sheet
    ltd = latest_value(facts, sym, "LongTermDebt", as_of)
    cash = latest_value(facts, sym, "CashAndCashEquivalentsAtCarryingValue", as_of)
    net_debt = (0.0 if math.isnan(ltd) else ltd) - (0.0 if math.isnan(cash) else cash)
    if math.isnan(ltd) and math.isnan(cash):
        m["net_debt_ebitda"] = _nan()
    elif net_debt <= 0:
        m["net_debt_ebitda"] = 0.0  # net cash: best possible (scored 100 below)
    elif not math.isnan(ebit) and ebit > 0:
        m["net_debt_ebitda"] = net_debt / ebit  # EBIT as EBITDA proxy (no D&A tag)
    else:
        m["net_debt_ebitda"] = _nan()
    m["_net_cash"] = 1.0 if net_debt <= 0 and not (math.isnan(ltd) and math.isnan(cash)) else 0.0
    interest = ttm_sum(facts, sym, "InterestExpense", as_of)
    if math.isnan(ebit):
        m["interest_coverage"] = _nan()
    elif math.isnan(interest) or interest <= 0:
        m["interest_coverage"] = 1e6 if ebit > 0 else 0.0  # no interest burden
    else:
        m["interest_coverage"] = ebit / interest

    # management: dilution + insider record
    # Same share tag, two reporting periods a year apart. This compared the
    # latest filing with itself (latest_value ignored its as_of), so dilution
    # read 0.0 for every company and the management sub-score never saw it.
    shares = share_count_yoy(facts, sym, to_ts(as_of))
    m["dilution_yoy"] = _nan() if shares is None else shares[0] / shares[1] - 1.0
    m["insider_net_buy"] = insider_net_buy_signed(insider, sym, as_of, window_days=180)
    return m


_METRIC_ORIENTATION: dict[str, tuple[str, bool]] = {
    "gm_stability": ("business_quality", _HIGH),
    "roic_persistence": ("business_quality", _HIGH),
    "rev_yoy": ("revenue_growth", _HIGH),
    "rev_accel": ("revenue_growth", _HIGH),
    "rev_cagr_3y": ("revenue_growth", _HIGH),
    "op_margin": ("profitability", _HIGH),
    "net_margin": ("profitability", _HIGH),
    "roic": ("profitability", _HIGH),
    "fcf_conversion": ("cash_flow", _HIGH),
    "fcf_margin": ("cash_flow", _HIGH),
    "fcf_yield": ("cash_flow", _HIGH),
    "net_debt_ebitda": ("balance_sheet", _LOW),
    "interest_coverage": ("balance_sheet", _HIGH),
    "dilution_yoy": ("management", _LOW),
    "insider_net_buy": ("management", _HIGH),
}


def _sector_percentiles(values: pd.Series, sectors: pd.Series, higher_is_better: bool) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in sectors.groupby(sectors).groups.items():
        grp = values.loc[list(idx)].dropna()
        if grp.empty:
            continue
        if len(grp) == 1:
            out.loc[grp.index] = 50.0
            continue
        pct = grp.rank(method="average", pct=True) * 100.0
        out.loc[grp.index] = pct if higher_is_better else 100.0 - pct + 100.0 / len(grp)
    return out.clip(0.0, 100.0)


def compute_quality_scores(
    *,
    fundamentals: pd.DataFrame | None,
    sector_map: Mapping[str, str],
    as_of: date | pd.Timestamp,
    config: QualityConfig | None = None,
    bars: pd.DataFrame | None = None,
    insider: pd.DataFrame | None = None,
    qualitative: Mapping[str, Mapping[str, float]] | None = None,
) -> list[QualityScore]:
    """Compute QualityScore rows for every symbol in ``sector_map`` at ``as_of``.

    Point-in-time by construction: only ``filed_at`` <= as_of facts are read,
    so a historical ``as_of`` replayed today reproduces the historical scores.
    """
    cfg = config or QualityConfig()
    ts_as_of = naive_utc_ts(as_of)
    facts = pit_facts(fundamentals, ts_as_of)
    price = _price_panel(bars, ts_as_of)
    symbols = sorted(sector_map)
    sectors = pd.Series({s: str(sector_map[s]) for s in symbols})
    qual = qualitative or {}

    metrics = pd.DataFrame(
        {
            sym: _symbol_metrics(facts, price, insider, sym, ts_as_of, cfg.min_quarters)
            for sym in symbols
        }
    ).T
    metrics.index.name = "symbol"

    # Sector-relative percentiles per metric.
    pctiles = pd.DataFrame(index=metrics.index, dtype=float)
    for metric, (_, higher) in _METRIC_ORIENTATION.items():
        pctiles[metric] = _sector_percentiles(cast(pd.Series, metrics[metric]), sectors, higher)
    # Net-cash names score 100 on leverage regardless of peers (doc 05 s2.1).
    pctiles.loc[metrics["_net_cash"] >= 1.0, "net_debt_ebitda"] = 100.0

    # Valuation: 50% own 3y history, 50% sector percentile (doc 05 s2.1).
    val_sector = cast(
        pd.Series,
        pd.concat(
            [
                _sector_percentiles(cast(pd.Series, metrics["ev_ebit"]), sectors, _LOW),
                _sector_percentiles(cast(pd.Series, metrics["fcf_yield"]), sectors, _HIGH),
            ],
            axis=1,
        ).mean(axis=1),
    )

    scores: list[QualityScore] = []
    for sym in symbols:
        n_q = n_filed_quarters(facts, sym, ts_as_of)
        subs: dict[str, float | None] = {}
        for name in SUBSCORE_NAMES:
            if name == "valuation":
                continue
            cols = [mc for mc, (sub, _) in _METRIC_ORIENTATION.items() if sub == name]
            vals = cast(pd.Series, pctiles.loc[sym, cols]).dropna()
            subs[name] = float(vals.mean()) if len(vals) else None

        # valuation
        if n_q >= cfg.min_quarters and not price.empty:
            ev_hist, fcf_hist = _valuation_history(facts, price, sym, ts_as_of)
            own_parts = [
                p
                for p in (
                    _own_history_pctile(float(metrics.loc[sym, "ev_ebit"]), ev_hist, _LOW),
                    _own_history_pctile(float(metrics.loc[sym, "fcf_yield"]), fcf_hist, _HIGH),
                )
                if not math.isnan(p)
            ]
            sector_part = val_sector.loc[sym]
            parts = []
            if own_parts:
                parts.append(float(np.mean(own_parts)))
            if not math.isnan(sector_part):
                parts.append(float(sector_part))
            subs["valuation"] = float(np.mean(parts)) if parts else None
        else:
            subs["valuation"] = None

        # History-gated sub-scores are null below min_quarters (recent IPOs).
        if n_q < cfg.min_quarters:
            subs["business_quality"] = None
            subs["revenue_growth"] = None

        # Qualitative dossier blends (evidence-referenced or absent).
        q = qual.get(sym, {})
        moat = q.get("moat")
        if moat is not None:
            base = subs["business_quality"]
            subs["business_quality"] = float(moat) if base is None else (base + float(moat)) / 2.0
        cap = q.get("capital_allocation")
        if cap is not None:
            base = subs["management"]
            subs["management"] = float(cap) if base is None else (base + float(cap)) / 2.0

        null_count = sum(1 for v in subs.values() if v is None)
        if null_count > cfg.max_null_subscores:
            composite = None  # ineligible for the long-term sleeve — flagged
        else:
            live = {k: v for k, v in subs.items() if v is not None}
            total_w = sum(cfg.weights[k] for k in live)
            composite = (
                float(sum(cfg.weights[k] / total_w * v for k, v in live.items()))
                if total_w > 0
                else None
            )
        scores.append(
            QualityScore(
                symbol=sym,
                as_of=ts_as_of.date(),
                composite=composite,
                subscores=subs,
                config_hash=cfg.config_hash,
            )
        )
    return scores
