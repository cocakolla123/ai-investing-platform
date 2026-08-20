"""Composite screens over the feature catalog. Pure: frames in, frames out.

Currently one screen, the one the owner actually asked for.

THE SLEEPER SCREEN
==================

The thesis in one sentence: *an under-followed company that is quietly getting
better, being accumulated, and going up without anybody noticing.* Each clause
is a component, each component is a documented weight, and the whole thing is
a ranking — not a recommendation.

**IT IS UNVALIDATED.** Nothing here has been auditioned, backtested, or
measured against a forward return. The weights below are priors written down
before looking at any outcome, which is the only honest way to write them, and
which is also exactly why they should not be trusted yet. It is pre-registered
through the alpha-research rails (``dynamo.cli.alpha``) as hypothesis
``sleeper-screen-v1`` (trial family ``sleeper-composite``); until that audition
runs and passes, this module produces an opinion, not evidence.

Composition
-----------

Every component is a CROSS-SECTIONAL PERCENTILE RANK inside the scored
universe, so no component's units or scale can dominate another's, and the
composite is a weighted mean of the components present:

======================== ====== ==============================================
Component                Weight Inputs (higher composite = better)
======================== ====== ==============================================
``improving``             0.25  ``rev_growth_yoy``, ``gross_margin_ttm``
``accumulation``          0.25  ``obv_slope_60d``, ``sustained_volume_expansion``
``low_attention``         0.20  ``news_count_60d`` (inverted: fewer stories)
``quiet_momentum``        0.20  ``ret_60d`` via a band (see below),
                                ``max_session_share_60d`` (inverted)
``liquidity``             0.10  ``days_to_exit`` (inverted: fewer days)
======================== ====== ==============================================

Two components are not plain ranks and are worth stating explicitly:

*Quiet momentum* deliberately does NOT reward the biggest gainer. ``ret_60d``
is scored through a band — full marks inside ``[momentum_floor,
momentum_ceiling]``, tapering to zero below 0% and above roughly twice the
ceiling. A name up 80% in three months is not a sleeper; it is awake, and
everyone can see it. The second half of the component uses
``max_session_share_60d`` inverted, so a move earned across many sessions
beats the same move delivered by one gap.

*Low attention* is the load-bearing assumption of the whole screen: news
coverage is a usable proxy for how closely a name is followed. It is a proxy,
not a measurement — a name can be quiet because it is genuinely undiscovered
or because our news collection simply missed it. ``news_count_60d`` is NaN
(rather than 0) when no news was collected at all, which is what keeps the
second case from masquerading as the first.

Missing inputs
--------------

A component whose inputs are all NaN for a symbol is DROPPED for that symbol
and the remaining weights are renormalized — a name with no XBRL does not get
a zero for fundamentals, which would rank it below a genuinely deteriorating
company. But degradation has a floor: below ``min_components`` present
components, or missing either of the two components that ARE the thesis
(accumulation and low attention), the symbol scores NaN and drops out of the
screen entirely. Partial evidence is discounted; absent evidence is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

__all__ = [
    "SLEEPER_COMPONENT_WEIGHTS",
    "SleeperConfig",
    "sleeper_scores",
]

# Component weights. They sum to 1.0 and are renormalized over whichever
# components a given symbol actually has (see the module docstring).
SLEEPER_COMPONENT_WEIGHTS: Mapping[str, float] = {
    "improving": 0.25,
    "accumulation": 0.25,
    "low_attention": 0.20,
    "quiet_momentum": 0.20,
    "liquidity": 0.10,
}

# Components that ARE the thesis: without them there is no sleeper claim to
# make, only a fundamentals-and-liquidity screen wearing its name.
REQUIRED_COMPONENTS: tuple[str, ...] = ("accumulation", "low_attention")

COMPONENT_INPUTS: Mapping[str, tuple[tuple[str, bool], ...]] = {
    # (feature name, higher_is_better)
    "improving": (("rev_growth_yoy", True), ("gross_margin_ttm", True)),
    "accumulation": (("obv_slope_60d", True), ("sustained_volume_expansion", True)),
    "low_attention": (("news_count_60d", False),),
    "liquidity": (("days_to_exit", False),),
}


@dataclass(frozen=True)
class SleeperConfig:
    """Tunables, all with the priors that were written down before testing."""

    weights: Mapping[str, float] = field(
        default_factory=lambda: dict(SLEEPER_COMPONENT_WEIGHTS)
    )
    # "Positive but not extreme": the 60-session return band that scores full
    # marks. Below 0% scores nothing; above ~2x the ceiling scores nothing.
    momentum_floor: float = 0.02
    momentum_ceiling: float = 0.25
    # A symbol needs this many of the five components to be scored at all.
    min_components: int = 4
    required_components: Sequence[str] = REQUIRED_COMPONENTS


def _pct_rank(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Cross-sectional percentile rank in ``[0, 1]``; NaN stays NaN.

    ``method="average"`` makes ties deterministic (order of the index cannot
    move a score), which is what the determinism test pins.
    """
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce")).replace(
        [np.inf, -np.inf], np.nan
    )
    ranked = cast(pd.Series, numeric.rank(method="average", pct=True))
    return ranked if higher_is_better else 1.0 - ranked


def _band_score(values: pd.Series, floor: float, ceiling: float) -> pd.Series:
    """1.0 inside ``[floor, ceiling]``, tapering to 0 outside it.

    Below the floor the taper runs down to 0 at a return of 0% (a falling name
    is not a quiet riser). Above the ceiling it runs down to 0 at
    ``2 x ceiling`` — by then the market has plainly noticed.
    """
    r = cast(
        pd.Series,
        cast(pd.Series, pd.to_numeric(values, errors="coerce")).replace(
            [np.inf, -np.inf], np.nan
        ),
    )
    below = (r / floor).clip(lower=0.0, upper=1.0) if floor > 0 else r * 0.0 + 1.0
    above = (1.0 - (r - ceiling) / ceiling).clip(lower=0.0, upper=1.0)
    inside = (r >= floor) & (r <= ceiling)
    scored = cast(pd.Series, below.where(r < floor, above.where(r > ceiling, 1.0)))
    return cast(pd.Series, scored.where(~inside, 1.0).where(r.notna()))


def _mean_of_available(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    """Row-mean over the parts that are present; all-NaN rows stay NaN."""
    if not parts:
        return pd.Series(np.nan, index=index, dtype=float)
    frame = pd.concat(parts, axis=1)
    return cast(pd.Series, frame.mean(axis=1, skipna=True))


def _component_scores(features: pd.DataFrame, config: SleeperConfig) -> pd.DataFrame:
    """The five component scores, each in ``[0, 1]``, NaN where unknowable."""
    index = features.index
    out: dict[str, pd.Series] = {}
    for name, inputs in COMPONENT_INPUTS.items():
        parts = [
            _pct_rank(cast(pd.Series, features[col]), higher_is_better=higher)
            for col, higher in inputs
            if col in features.columns
        ]
        out[name] = _mean_of_available(parts, index)

    momentum_parts: list[pd.Series] = []
    if "ret_60d" in features.columns:
        momentum_parts.append(
            _band_score(
                cast(pd.Series, features["ret_60d"]),
                config.momentum_floor,
                config.momentum_ceiling,
            )
        )
    if "max_session_share_60d" in features.columns:
        momentum_parts.append(
            _pct_rank(
                cast(pd.Series, features["max_session_share_60d"]), higher_is_better=False
            )
        )
    out["quiet_momentum"] = _mean_of_available(momentum_parts, index)
    return pd.DataFrame(out, index=index)


def sleeper_scores(
    features: pd.DataFrame, config: SleeperConfig | None = None
) -> pd.DataFrame:
    """Score a RAW feature frame for the sleeper thesis. UNVALIDATED.

    ``features`` is a ``build_raw_features``-shaped frame indexed by symbol
    (raw, not cross-sectionally normalized — this function does its own
    ranking, and double-ranking would only lose information).

    Returns one row per input symbol with the five component scores, the
    number of components present, and ``sleeper_score`` in ``[0, 1]``: the
    weight-renormalized mean of the components that are present, or NaN when
    too few are (see the module docstring). Deterministic: the same frame
    always gives the same numbers, and re-ordering the rows cannot change any
    symbol's score.

    The score is CROSS-SECTIONAL — it says where a name sits among the names
    scored beside it, and means nothing about a single symbol in isolation.
    """
    cfg = config or SleeperConfig()
    index = pd.Index(features.index, name=features.index.name or "symbol")
    if features.empty:
        return pd.DataFrame(
            {
                **{name: pd.Series(dtype=float) for name in cfg.weights},
                "components_present": pd.Series(dtype=float),
                "sleeper_score": pd.Series(dtype=float),
            },
            index=index,
        )

    components = _component_scores(features, cfg)
    weights = pd.Series({name: float(cfg.weights.get(name, 0.0)) for name in components})
    present = components.notna()

    weighted = (components.fillna(0.0) * weights).sum(axis=1)
    available_weight = present.astype(float).mul(weights, axis=1).sum(axis=1)
    score = cast(pd.Series, weighted / available_weight.where(available_weight > 0))

    n_present = cast(pd.Series, present.sum(axis=1))
    usable = n_present >= cfg.min_components
    for name in cfg.required_components:
        if name in present.columns:
            usable &= cast(pd.Series, present[name])

    out = components.copy()
    out["components_present"] = n_present.astype(float)
    out["sleeper_score"] = score.where(usable)
    out.index = index
    return out
