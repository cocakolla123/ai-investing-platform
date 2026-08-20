"""CANDIDATE: ``range_reversion`` — how range-bound, and how far from centre.

Auditioned via the ADR-0015 alpha-research rails::

    python -m dynamo.cli.alpha audition \
        --hypothesis-id range-reversion-meanrev \
        --candidate-file research/candidates/range_reversion.py \
        --train-start 2023-01-01 --train-end 2025-12-31

Contract (``dynamo.research.discovery.audition.load_candidate``): a
module-level ``CANDIDATE`` that is a
:class:`dynamo.research.quant.catalog.FeatureSpec`, whose ``name`` is a new
python identifier absent from ``FEATURE_CATALOG``, and whose ``fn`` maps a
``FeatureContext`` to a Series over the universe, NaN for missing inputs and
never raising. ``group="meanrev"`` puts it outside ``RAW_GROUPS``, so the
audition winsorizes and cross-sectionally z-scores it per session exactly
like catalog groups 1-8.

The score
---------

The arithmetic lives in :mod:`dynamo.research.quant.range_stats` — one
definition, shared with the tests and covered by the same gates as the rest
of ``src`` — and is built entirely from the catalog's per-symbol window
helpers (``_apply_per_symbol`` / ``_roll_per_symbol`` / ``_shift_per_symbol``)
so a mixed equity+crypto universe cannot turn "N sessions" into "N calendar
days". Read that module's docstring for the full derivation and caveats;
the formula in brief::

    VR_bar     = mean over k in {5, 10} of
                     Var_120(k-session log return) / (k * Var_120(1-session))
    vr_term    = clip(1 - VR_bar, 0, 1)                # Lo-MacKinlay reversion
    up, dn     = distinct excursions beyond the upper / lower 20-session
                 2-sigma Bollinger band in the trailing 120 sessions
    touch_term = clip(min(up, dn) / 2, 0, 1)           # two-sided oscillation
    rangedness = sqrt(vr_term * touch_term)            # [0, 1] conjunction
    position   = 0.5 * clip(-(P - mu20) / sd20 / 2, -1, 1)
               + 0.5 * clip(1 - 2 * (P - low60) / (high60 - low60), -1, 1)
    SCORE      = rangedness * position                 # [-1, +1]

SIGN CONVENTION: **positive = attractive long.** Positive means an
established range AND price currently near the bottom of it — the pattern
this candidate exists to find. Negative means an established range with price
near the top. Approximately zero means there is no established range at all,
whatever the price level: measured on synthetic drifting random walks a
downtrending name sitting at its 60-session low scores 0.000, because
``rangedness`` is a conjunction and the variance-ratio term vetoes it. That
refusal to buy dips in downtrends is the point of the construction, not a
side effect.

Predicted sign, pre-registered before any test (hypothesis
``range-reversion-meanrev``, family ``mean_reversion``): POSITIVE rank IC
against the 20-session excess-return label, small — order 0.01 to 0.03 — and
expected to be WEAK or absent on today's 55-name mega-cap panel, which is
the wrong universe for this idea. See the hypothesis YAML for that scope
argument in full.
"""

from __future__ import annotations

from dynamo.research.quant.catalog import FeatureSpec
from dynamo.research.quant.range_stats import (
    BAND_WINDOW,
    RANGE_WINDOW,
    TOUCH_WINDOW,
    VR_LAGS,
    VR_WINDOW,
    range_reversion_score,
)

# Sessions of a symbol's OWN history the score needs before it reports a
# number. The excursion counter binds: 120 counted sessions on top of the
# 20-session band that defines a touch.
LOOKBACK_SESSIONS = max(
    VR_WINDOW + max(VR_LAGS),  # variance ratio: 130
    TOUCH_WINDOW + BAND_WINDOW,  # two-sided excursions: 140
    RANGE_WINDOW,  # position percentile: 60
)

CANDIDATE = FeatureSpec(
    name="range_reversion",
    group="meanrev",
    lookback_days=LOOKBACK_SESSIONS,
    fn=range_reversion_score,
)
