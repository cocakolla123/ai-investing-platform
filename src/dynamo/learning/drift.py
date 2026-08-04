"""Nightly drift monitors (doc 10 §7). Pure functions, never apply anything.

Each monitor compares a signal against its explicit expectation and, when
tripped, returns a typed ``ImprovementProposal`` with evidence. Accepted
proposals enter the doc 08 promotion pipeline — there is no code path from a
monitor to production config.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime

from dynamo.domain.models.learning import ImprovementProposal, ProposalKind
from dynamo.domain.models.research import Evidence, SourceType

IC_DECAY_WINDOW = 30  # sessions
IC_DECAY_FRACTION = 0.5
PSI_THRESHOLD = 0.25
EXPECTATION_BAND_WINDOW = 20  # sessions
SLIPPAGE_WINDOW = 30  # trades
SLIPPAGE_RATIO = 1.5
MK_Z_THRESHOLD = 1.645  # one-sided 95%


def _proposal(
    kind: ProposalKind,
    target: str,
    as_of: date,
    now: datetime,
    evidence: Sequence[Evidence],
    expected_effect: str,
    proposed_change: Mapping[str, object],
) -> ImprovementProposal:
    return ImprovementProposal(
        id=f"{kind.value}:{target}:{as_of.isoformat()}",
        created_at=now,
        kind=kind,
        target=target,
        evidence=tuple(evidence),
        expected_effect=expected_effect,
        proposed_change=dict(proposed_change),
    )


def _evidence(claim: str, ref: str, as_of: date, source: SourceType) -> Evidence:
    return Evidence(claim=claim, source_type=source, source_ref=ref, as_of=as_of)


def ic_decay_monitor(
    rolling_ic: Sequence[float],
    promotion_ic: float,
    model_id: str,
    as_of: date,
    now: datetime,
    *,
    window: int = IC_DECAY_WINDOW,
    fraction: float = IC_DECAY_FRACTION,
) -> ImprovementProposal | None:
    """Rolling IC below ``fraction`` of the promotion IC for ``window``
    consecutive sessions → retrain proposal."""
    if promotion_ic <= 0 or len(rolling_ic) < window:
        return None
    recent = rolling_ic[-window:]
    threshold = fraction * promotion_ic
    if all(ic < threshold for ic in recent):
        return _proposal(
            ProposalKind.RETRAIN,
            model_id,
            as_of,
            now,
            [
                _evidence(
                    f"rolling IC < {threshold:.4f} ({fraction:.0%} of promotion IC "
                    f"{promotion_ic:.4f}) for {window} sessions; "
                    f"latest {recent[-1]:.4f}",
                    f"metrics_daily:rolling_ic:{model_id}",
                    as_of,
                    SourceType.FUNDAMENTAL,
                )
            ],
            "restore IC toward promotion-time level via retrain on recent data",
            {"model_id": model_id, "action": "retrain"},
        )
    return None


def feature_psi_monitor(
    psi_by_feature: Mapping[str, float],
    top_features: Sequence[str],
    model_id: str,
    as_of: date,
    now: datetime,
    *,
    threshold: float = PSI_THRESHOLD,
) -> ImprovementProposal | None:
    """PSI above threshold on any top-importance feature → retrain proposal."""
    drifted = {
        f: psi_by_feature[f]
        for f in top_features
        if psi_by_feature.get(f, 0.0) > threshold
    }
    if not drifted:
        return None
    evidence = [
        _evidence(
            f"feature {name!r} PSI {value:.3f} > {threshold} vs training distribution",
            f"metrics_daily:psi:{model_id}:{name}",
            as_of,
            SourceType.FUNDAMENTAL,
        )
        for name, value in sorted(drifted.items())
    ]
    return _proposal(
        ProposalKind.RETRAIN,
        model_id,
        as_of,
        now,
        evidence,
        "refresh model on the drifted feature distribution",
        {"model_id": model_id, "action": "retrain", "drifted_features": sorted(drifted)},
    )


def strategy_expectation_monitor(
    rolling_metric: Sequence[float],
    expectation_band: tuple[float, float],
    strategy_id: str,
    as_of: date,
    now: datetime,
    *,
    window: int = EXPECTATION_BAND_WINDOW,
) -> ImprovementProposal | None:
    """Performance outside the 90% backtest expectation band for ``window``
    consecutive sessions → strategy-retirement proposal."""
    low, high = expectation_band
    if len(rolling_metric) < window:
        return None
    recent = rolling_metric[-window:]
    if all(v < low or v > high for v in recent):
        return _proposal(
            ProposalKind.STRATEGY_RETIREMENT,
            strategy_id,
            as_of,
            now,
            [
                _evidence(
                    f"rolling performance outside 90% expectation band "
                    f"[{low:.3f}, {high:.3f}] for {window} sessions; "
                    f"latest {recent[-1]:.3f}",
                    f"metrics_daily:strategy_perf:{strategy_id}",
                    as_of,
                    SourceType.FUNDAMENTAL,
                )
            ],
            "retire or re-validate the strategy against current conditions",
            {"strategy_id": strategy_id, "action": "review_for_retirement"},
        )
    return None


def slippage_monitor(
    realized_bps: Sequence[float],
    modeled_bps: Sequence[float],
    as_of: date,
    now: datetime,
    *,
    window: int = SLIPPAGE_WINDOW,
    ratio: float = SLIPPAGE_RATIO,
    target: str = "execution.cost_model",
) -> ImprovementProposal | None:
    """Mean realized slippage > ``ratio`` x modeled over the last ``window``
    trades → threshold-change proposal for the cost model."""
    if len(realized_bps) < window or len(modeled_bps) < window:
        return None
    realized = sum(realized_bps[-window:]) / window
    modeled = sum(modeled_bps[-window:]) / window
    if modeled <= 0 or realized <= ratio * modeled:
        return None
    return _proposal(
        ProposalKind.THRESHOLD_CHANGE,
        target,
        as_of,
        now,
        [
            _evidence(
                f"realized slippage {realized:.2f} bps > {ratio}x modeled "
                f"{modeled:.2f} bps over last {window} trades",
                "metrics_daily:slippage_realized_vs_model",
                as_of,
                SourceType.PRICE,
            )
        ],
        "recalibrate transaction-cost assumptions to realized fills",
        {"target": target, "observed_bps": realized, "modeled_bps": modeled},
    )


def mann_kendall_z(series: Sequence[float]) -> float:
    """Mann-Kendall trend statistic (normal approximation, no tie correction
    beyond sign zeros). Positive z = upward trend."""
    n = len(series)
    if n < 3:
        return 0.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            s += (diff > 0) - (diff < 0)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if var <= 0 or s == 0:
        return 0.0
    adjust = -1 if s > 0 else 1
    return (s + adjust) / math.sqrt(var)


def incident_trend_monitor(
    daily_incident_counts: Sequence[float],
    provider: str,
    as_of: date,
    now: datetime,
    *,
    z_threshold: float = MK_Z_THRESHOLD,
) -> ImprovementProposal | None:
    """Sustained upward Mann-Kendall trend in data-quality incidents for a
    provider → new-data-source proposal."""
    z = mann_kendall_z(daily_incident_counts)
    if z <= z_threshold:
        return None
    return _proposal(
        ProposalKind.NEW_DATA_SOURCE,
        provider,
        as_of,
        now,
        [
            _evidence(
                f"data-quality incidents trending up for {provider!r} "
                f"(Mann-Kendall z={z:.2f} > {z_threshold})",
                f"metrics_daily:incidents:{provider}",
                as_of,
                SourceType.JOURNAL,
            )
        ],
        "evaluate a replacement or supplementary provider",
        {"provider": provider, "mann_kendall_z": z},
    )
