"""Watchlist blend — doc 05 section 4.

Deterministic nightly blend of ``CommitteeVerdict``, ``QuantScore``, and
``QualityScore`` into ranked long-term and tactical lists:

``blend = 100 x (w_c x committee + w_q x quant + w_s x quality)``

* committee: action map {buy 1.0, add 0.8, hold 0.5, trim/sell 0.1,
  avoid/no_action 0.0} x conviction, decayed x0.9 per week of age past 7d.
* quant: cross-sectional percentile of expected excess return x confidence
  minus ``0.5 x downside_prob``, floor-clamped at 0.
* quality: composite/100; for the tactical list this weight redistributes to
  the other two components.

Disagreement handling: ``quant_alignment == conflict`` caps the blend at the
40th percentile of the night's candidate list and badges the row; a verdict
with no fresh QuantScore gets a zero quant component and a ``no-quant`` badge
— never an imputed score.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date

import numpy as np

from dynamo.domain.models.research import (
    Action,
    BlendConfig,
    CommitteeVerdict,
    QualityScore,
    QuantAlignment,
    QuantScore,
    Signal,
    SignalStatus,
    Sleeve,
    WatchlistRow,
)

ACTION_COMPONENT: dict[Action, float] = {
    Action.BUY: 1.0,
    Action.ADD: 0.8,
    Action.HOLD: 0.5,
    Action.TRIM: 0.1,
    Action.SELL: 0.1,
    Action.AVOID: 0.0,
    Action.NO_ACTION: 0.0,
}

STALE_VERDICT_DAYS = 28
LONG_TERM_MIN_QUALITY = 60.0
LONG_TERM_MIN_CONVICTION = 0.6
LONG_TERM_ACTIONS = frozenset({Action.BUY, Action.ADD})

__all__ = ["blend_config_hash", "build_watchlist"]


def blend_config_hash(config: BlendConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _committee_component(verdict: CommitteeVerdict | None, as_of: date) -> float | None:
    if verdict is None:
        return None
    base = ACTION_COMPONENT[verdict.action] * verdict.conviction
    age_days = (as_of - verdict.as_of).days
    weeks_past = max(0.0, (age_days - 7) / 7.0)
    return base * (0.9**weeks_past)


def _quant_component(
    score: QuantScore | None,
    pctile: float | None,
    config: BlendConfig,
) -> float | None:
    """Percentile x confidence minus downside penalty, clamped at 0. None = no score."""
    if score is None or pctile is None:
        return None
    downside = score.downside_prob or 0.0
    return max(0.0, pctile * score.confidence - config.downside_penalty * downside)


def _expected_return_pctiles(scores: dict[str, QuantScore]) -> dict[str, float]:
    valid = {
        s: sc.expected_excess_return
        for s, sc in scores.items()
        if sc.expected_excess_return is not None
    }
    if not valid:
        return {}
    ordered = sorted(valid, key=lambda s: (valid[s], s))
    n = len(ordered)
    return {s: (i + 1) / n for i, s in enumerate(ordered)}


def _apply_conflict_cap(
    blends: dict[str, float], conflicted: set[str], cap_pctile: float
) -> dict[str, float]:
    """Conflicted symbols cannot rank above the cap percentile of the list."""
    if not blends or not conflicted:
        return blends
    cap_value = float(np.quantile(list(blends.values()), cap_pctile))
    return {s: min(v, cap_value) if s in conflicted else v for s, v in blends.items()}


def build_watchlist(
    *,
    as_of: date,
    verdicts: Sequence[CommitteeVerdict] = (),
    quant_scores: Sequence[QuantScore] = (),
    quality_scores: Sequence[QualityScore] = (),
    signals: Sequence[Signal] = (),
    config: BlendConfig | None = None,
    long_term_top_n: int = 25,
    tactical_top_k: int = 10,
) -> list[WatchlistRow]:
    """Assemble both sleeves' ranked lists (doc 05 sections 4.1-4.3).

    ``signal_ref`` on tactical rows is the index of the signal in ``signals``.
    Deterministic: ties break by symbol. Returns long-term rows then tactical.
    """
    cfg = config or BlendConfig()
    cfg_hash = blend_config_hash(cfg)

    latest_verdict: dict[str, CommitteeVerdict] = {}
    for v in verdicts:
        if v.as_of > as_of:
            continue
        cur = latest_verdict.get(v.symbol)
        if cur is None or v.as_of > cur.as_of:
            latest_verdict[v.symbol] = v
    quant: dict[str, QuantScore] = {s.symbol: s for s in quant_scores if s.as_of == as_of}
    quality: dict[str, QualityScore] = {}
    for q in quality_scores:
        cur_q = quality.get(q.symbol)
        if q.as_of <= as_of and (cur_q is None or q.as_of > cur_q.as_of):
            quality[q.symbol] = q
    pctiles = _expected_return_pctiles(quant)

    symbols = sorted(set(latest_verdict) | set(quant) | set(quality))

    committee: dict[str, float | None] = {}
    quant_comp: dict[str, float | None] = {}
    quality_comp: dict[str, float | None] = {}
    badges: dict[str, list[str]] = {}
    for sym in symbols:
        verdict = latest_verdict.get(sym)
        committee[sym] = _committee_component(verdict, as_of)
        quant_comp[sym] = _quant_component(quant.get(sym), pctiles.get(sym), cfg)
        qs = quality.get(sym)
        quality_comp[sym] = None if qs is None or qs.composite is None else qs.composite / 100.0
        b: list[str] = []
        if verdict is not None and quant.get(sym) is None:
            b.append("no-quant")
        if verdict is not None and verdict.quant_alignment == QuantAlignment.CONFLICT:
            b.append("conflict")
        if verdict is not None and (as_of - verdict.as_of).days > STALE_VERDICT_DAYS:
            b.append("stale-verdict")
        badges[sym] = b

    conflicted = {s for s in symbols if "conflict" in badges[s]}

    def _blend(sym: str, tactical: bool) -> float:
        w_c, w_q, w_s = cfg.w_committee, cfg.w_quant, cfg.w_quality
        if tactical:
            # Quality is slow-moving context, not a timing signal: its weight
            # redistributes proportionally to the other two (doc 05 s4.1).
            total = w_c + w_q
            w_c, w_q, w_s = w_c + w_s * w_c / total, w_q + w_s * w_q / total, 0.0
        c = committee.get(sym) or 0.0
        q = quant_comp.get(sym) or 0.0
        s = quality_comp.get(sym) or 0.0
        return float(min(100.0, max(0.0, 100.0 * (w_c * c + w_q * q + w_s * s))))

    rows: list[WatchlistRow] = []

    # Long-term sleeve: state-driven candidate set (doc 05 s4.3).
    lt_candidates = [
        sym
        for sym in symbols
        if (qs := quality.get(sym)) is not None
        and qs.composite is not None
        and qs.composite >= LONG_TERM_MIN_QUALITY
        and (v := latest_verdict.get(sym)) is not None
        and v.action in LONG_TERM_ACTIONS
        and v.conviction >= LONG_TERM_MIN_CONVICTION
    ]
    lt_blends = _apply_conflict_cap(
        {sym: _blend(sym, tactical=False) for sym in lt_candidates},
        conflicted,
        cfg.conflict_rank_cap_pctile,
    )
    lt_order = sorted(lt_blends, key=lambda s: (-lt_blends[s], s))[:long_term_top_n]
    for rank, sym in enumerate(lt_order, start=1):
        v = latest_verdict.get(sym)
        rows.append(
            WatchlistRow(
                as_of=as_of,
                sleeve=Sleeve.LONG_TERM,
                symbol=sym,
                rank=rank,
                blend_score=lt_blends[sym],
                committee_component=committee[sym],
                quant_component=quant_comp[sym],
                quality_component=quality_comp[sym],
                badges=tuple(badges[sym]),
                blend_config_hash=cfg_hash,
                verdict_ref=v.id if v is not None else None,
            )
        )

    # Tactical sleeve: signal-driven — a strategy must fire (doc 05 s4.3).
    active = [
        (i, sig)
        for i, sig in enumerate(signals)
        if sig.status == SignalStatus.ACTIVE and sig.as_of <= as_of
    ]
    tac_blends = _apply_conflict_cap(
        {sig.symbol: _blend(sig.symbol, tactical=True) for _, sig in active},
        conflicted,
        cfg.conflict_rank_cap_pctile,
    )
    ranked = sorted(
        active,
        key=lambda item: (-item[1].strength * tac_blends[item[1].symbol], item[1].symbol),
    )[:tactical_top_k]
    for rank, (idx, sig) in enumerate(ranked, start=1):
        sym = sig.symbol
        v = latest_verdict.get(sym)
        rows.append(
            WatchlistRow(
                as_of=as_of,
                sleeve=Sleeve.TACTICAL,
                symbol=sym,
                rank=rank,
                blend_score=tac_blends[sym],
                committee_component=committee.get(sym),
                quant_component=quant_comp.get(sym),
                quality_component=quality_comp.get(sym),
                badges=tuple(badges.get(sym, [])),
                blend_config_hash=cfg_hash,
                verdict_ref=v.id if v is not None else None,
                signal_ref=idx,
            )
        )
    return rows
