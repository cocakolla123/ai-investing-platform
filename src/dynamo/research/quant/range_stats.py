"""Range-boundness / mean-reversion statistics (ADR-0015 candidate support).

This module is **library math, not a catalog feature**. Nothing here is wired
into ``FEATURE_CATALOG`` or into production scoring; it exists so the
pre-registered ``range_reversion`` candidate (``research/candidates/``) and
its tests can share one definition of the arithmetic. Adding it changes no
trained model and no panel-cache fingerprint.

Everything is computed with the per-symbol window helpers from
:mod:`dynamo.research.quant.catalog` (``_apply_per_symbol`` /
``_roll_per_symbol`` / ``_shift_per_symbol``). That is not stylistic: the
price panel is indexed by the UNION of every symbol's bar timestamps, so a
mixed equity+crypto universe makes any positional operation on the shared
index measure CALENDAR days rather than sessions. Reusing the helpers keeps
"N periods" meaning "N sessions of THIS symbol's own history" and keeps every
number here invariant to who else is in the universe — the property
:mod:`tests.unit.quant.test_calendar_invariance` pins down.

The three components
--------------------

**1. Variance ratio (Lo-MacKinlay style).** For lag ``k``,

    VR_k = Var_W(r_k) / (k * Var_W(r_1))

where ``r_1`` are one-session log returns, ``r_k`` are OVERLAPPING k-session
log returns (a rolling k-sum of ``r_1``), and both variances are sample
variances over the trailing ``W = 120`` sessions. Under a random walk
VR_k = 1; VR_k < 1 means negatively autocorrelated increments (mean
reversion), VR_k > 1 means positive autocorrelation (trending). We average
VR_5 and VR_10.

Honest caveats, stated rather than buried:

* We use the plain ratio of sample variances. Lo & MacKinlay's finite-sample
  unbiasing correction (and their heteroskedasticity-robust test statistic)
  are NOT applied. The overlapping estimator carries a small downward bias of
  order k/W; with a common ``k`` and ``W`` across names it is close to a
  constant shift, and the score is used cross-sectionally, so the ranking is
  affected far less than the level. No p-value is claimed from VR here.
* VR at k=5/10 measures SHORT-horizon reversion. A slow, smooth oscillator
  (a 40-session cycle, say) is locally monotone over 5-10 sessions and can
  read VR > 1 — it looks like a trend at these lags. That blind spot is why
  the band-touch term below is a co-equal component and not a tiebreaker.

**2. Band touches (the oscillation counter).** 20-session Bollinger bands at
2 sigma. A session "touches" a band when the adjusted close is at or beyond
it. Touches are collapsed into EXCURSIONS (a touch whose previous session was
not touching), so a name pinned outside its band for ten sessions counts once,
not ten times. We count excursions to each band separately and take the
MINIMUM of the two counts: a genuine oscillator visits both ends roughly
evenly, while a trending name rides one band and sees the other rarely.

That is an ASYMMETRY, not a veto, and the difference matters: measured on a
drifting random walk over 120 sessions the opposite band still gets poked
once or twice, so a touch count alone would not reject trends. The variance
ratio is the primary trend veto and the touch count is the primary veto for
"statistically reverting but never actually visits its edges" — which is why
they are combined as a conjunction below rather than added.

The counting window is **120 sessions, not 60**, and that is a deliberate
departure from the obvious choice. Measured on synthetic range-bound series:
at the exact moment price sits at the low of its range — the moment the
score exists to find — the trailing 60 sessions are by construction biased
toward the LOWER band, so a 60-session two-sided requirement frequently sees
zero upper excursions and vetoes precisely the setup we want. Widening the
oscillation evidence to ~6 months while leaving ``position`` on the 20/60
windows separates the two questions cleanly: "has this thing oscillated?"
is structural and wants a long window; "where is it now?" is current and
wants a short one. ``count_window`` remains an argument for anyone who wants
the tighter reading.

Bands and touches both use the ADJUSTED close. Mixing raw intraday
high/low against adjusted-close bands would mis-scale every split-adjusted
name, which is a worse error than losing the intraday reach.

**3. Position in range (the signed part).** Two views, averaged:

* ``-(P - mean_20) / sd_20 / 2`` clipped to [-1, +1]: +1 at the lower 2-sigma
  band, -1 at the upper one;
* ``1 - 2 * (P - low_60) / (high_60 - low_60)`` clipped to [-1, +1]: +1 at the
  60-session low, -1 at the 60-session high.

The score and its sign convention
---------------------------------

    vr_term    = clip(1 - VR_bar, 0, 1)                  # statistical reversion
    touch_term = clip(min(up_120, dn_120) / 2, 0, 1)     # observed oscillation
    rangedness = sqrt(vr_term * touch_term)              # [0, 1], a CONJUNCTION
    position   = 0.5 * clip(-(P - mu20) / sd20 / 2, -1, 1)
               + 0.5 * clip(1 - 2 * pctile_60, -1, 1)    # [-1, +1]
    SCORE      = rangedness * position                   # [-1, +1]

``rangedness`` is non-negative and unsigned — it answers "is there an
established range here?". ``position`` carries the sign — "where in it are we
now?". The geometric mean makes ``rangedness`` a conjunction: a name must be
mean-reverting by the variance ratio AND observed to visit BOTH ends of its
band, or it is not a range.

**Sign convention: POSITIVE = attractive long.** A positive score means an
established range AND price currently near the bottom of it. A negative score
means an established range with price near the top (attractive to exit or
short). A score near zero means no established range — a trending or drifting
name — REGARDLESS of where price sits, which is deliberate: buying the dip in
a downtrend is the failure mode this construction is built to refuse.

NaN discipline matches the catalog: insufficient own-history, a zero
volatility window, or a degenerate 60-session range yields NaN, never a
fabricated zero, and no function here raises.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from dynamo.research.quant.catalog import (
    FeatureContext,
    _finite,
    _log_returns,
    _nan_series,
    _roll_per_symbol,
    _shift_per_symbol,
)

__all__ = [
    "BAND_SIGMA",
    "BAND_WINDOW",
    "POSITION_SIGMA",
    "RANGE_WINDOW",
    "TOUCH_TARGET",
    "TOUCH_WINDOW",
    "VR_LAGS",
    "VR_WINDOW",
    "band_excursions",
    "range_position",
    "range_reversion_score",
    "rangedness",
    "score_components",
    "variance_ratio",
    "variance_ratio_mean",
]

VR_WINDOW = 120  # sessions of own history the variance estimates use
VR_LAGS: tuple[int, ...] = (5, 10)
BAND_WINDOW = 20  # Bollinger lookback
BAND_SIGMA = 2.0
TOUCH_WINDOW = 120  # sessions the excursion counter spans (see docstring §2)
TOUCH_TARGET = 2.0  # excursions PER BAND that saturate the oscillation term
RANGE_WINDOW = 60  # high-low range for the percentile view
POSITION_SIGMA = 2.0  # sigma at which the z-view saturates (the band itself)


def _safe_ratio(num: pd.DataFrame, den: pd.DataFrame) -> pd.DataFrame:
    """Elementwise ratio with non-positive denominators sent to NaN."""
    positive = cast(pd.DataFrame, den.where(den > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / positive
    return cast(pd.DataFrame, out.replace([np.inf, -np.inf], np.nan))


def variance_ratio(price: pd.DataFrame, k: int, *, window: int = VR_WINDOW) -> pd.DataFrame:
    """Panel of ``VR_k`` (module docstring section 1), aligned to ``price``.

    ``window`` sample variances are strict (``min_periods == window``): a
    symbol needs ``window`` usable k-session returns of its OWN history before
    any value appears, so short listings stay NaN rather than reporting a
    ratio estimated off a handful of points.
    """
    if price.empty or len(price.columns) == 0 or k < 1:
        return pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    log_ret = _log_returns(price)
    var_1 = _roll_per_symbol(log_ret, window, "var")
    ret_k = _roll_per_symbol(log_ret, k, "sum")
    var_k = _roll_per_symbol(ret_k, window, "var")
    return _safe_ratio(var_k, cast(pd.DataFrame, var_1 * float(k)))


def variance_ratio_mean(
    price: pd.DataFrame, *, lags: tuple[int, ...] = VR_LAGS, window: int = VR_WINDOW
) -> pd.DataFrame:
    """Mean of ``variance_ratio`` over ``lags``; NaN if any lag is NaN."""
    if price.empty or not lags:
        return pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    total = pd.DataFrame(0.0, index=price.index, columns=price.columns, dtype=float)
    for k in lags:
        total = cast(pd.DataFrame, total + variance_ratio(price, k, window=window))
    return cast(pd.DataFrame, total / float(len(lags)))


def band_excursions(
    price: pd.DataFrame,
    *,
    window: int = BAND_WINDOW,
    sigma: float = BAND_SIGMA,
    count_window: int = TOUCH_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing counts of upper- and lower-band EXCURSIONS (docstring §2).

    Returns ``(upper_counts, lower_counts)`` panels: for each session, how many
    distinct excursions beyond the upper / lower Bollinger band occurred in the
    trailing ``count_window`` sessions of that symbol's own history.
    """
    empty = pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    if price.empty or len(price.columns) == 0:
        return empty, empty.copy()
    mid = _roll_per_symbol(price, window, "mean")
    sd = _roll_per_symbol(price, window, "std")
    usable = cast(pd.DataFrame, mid.notna() & (sd > 0))
    upper = cast(pd.DataFrame, mid + sigma * sd)
    lower = cast(pd.DataFrame, mid - sigma * sd)

    def _count(touching: pd.DataFrame) -> pd.DataFrame:
        # NaN where the band itself is undefined, so leading sessions never
        # masquerade as "not touching".
        flag = cast(pd.DataFrame, touching.astype(float).where(usable))
        prev = _shift_per_symbol(flag, 1)
        started = cast(pd.DataFrame, (flag > 0) & ~(prev > 0)).astype(float)
        started = cast(pd.DataFrame, started.where(flag.notna()))
        return _roll_per_symbol(started, count_window, "sum")

    return _count(cast(pd.DataFrame, price >= upper)), _count(cast(pd.DataFrame, price <= lower))


def range_position(
    price: pd.DataFrame,
    *,
    band_window: int = BAND_WINDOW,
    range_window: int = RANGE_WINDOW,
    position_sigma: float = POSITION_SIGMA,
) -> pd.DataFrame:
    """Signed position in the range: +1 near the low, -1 near the high (§3)."""
    if price.empty or len(price.columns) == 0:
        return pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    mid = _roll_per_symbol(price, band_window, "mean")
    sd = _roll_per_symbol(price, band_window, "std")
    z = _safe_ratio(cast(pd.DataFrame, price - mid), sd)
    pos_z = cast(pd.DataFrame, (-z / position_sigma).clip(-1.0, 1.0))

    high = _roll_per_symbol(price, range_window, "max")
    low = _roll_per_symbol(price, range_window, "min")
    pctile = _safe_ratio(cast(pd.DataFrame, price - low), cast(pd.DataFrame, high - low))
    pos_pct = cast(pd.DataFrame, cast(pd.DataFrame, 1.0 - 2.0 * pctile).clip(-1.0, 1.0))
    return cast(pd.DataFrame, 0.5 * (pos_z + pos_pct))


def rangedness(
    price: pd.DataFrame,
    *,
    lags: tuple[int, ...] = VR_LAGS,
    vr_window: int = VR_WINDOW,
    touch_target: float = TOUCH_TARGET,
) -> pd.DataFrame:
    """Unsigned [0, 1] "is there an established range here?" panel.

    GEOMETRIC mean of the variance-ratio term and the two-sided excursion
    term, i.e. a conjunction: an established range needs BOTH statistical
    mean reversion AND observed two-sided oscillation, and either term at
    zero zeroes the whole thing. That is the property that makes a trending
    name which happens to sit near its low score ~0 rather than looking like
    a bargain — measured on a drifting random walk, the additive version
    still scored ~0.25 rangedness because a long enough drift eventually
    pokes both bands once.
    """
    if price.empty or len(price.columns) == 0:
        return pd.DataFrame(index=price.index, columns=price.columns, dtype=float)
    vr = variance_ratio_mean(price, lags=lags, window=vr_window)
    vr_term = cast(pd.DataFrame, (1.0 - vr).clip(0.0, 1.0))
    up, dn = band_excursions(price)
    two_sided = cast(pd.DataFrame, np.minimum(up, dn))
    touch_term = cast(pd.DataFrame, (two_sided / float(touch_target)).clip(0.0, 1.0))
    return cast(pd.DataFrame, np.sqrt(vr_term * touch_term))


def range_reversion_score(ctx: FeatureContext) -> pd.Series:
    """The candidate feature function: ``rangedness * position``.

    Positive = established range, price near the low (attractive long).
    Negative = established range, price near the high.
    ~0 = no established range, whatever the price level.
    """
    price = ctx.price
    if price.empty or len(price) < 2:
        return _nan_series(ctx)
    score = cast(pd.DataFrame, rangedness(price) * range_position(price))
    if score.empty:
        return _nan_series(ctx)
    last = cast(pd.Series, score.iloc[-1])
    return _finite(last.astype(float)).reindex(list(ctx.universe))


def _last_valid(panel: pd.DataFrame, universe: list[str]) -> pd.Series:
    """Last row of a per-symbol panel, reindexed onto ``universe``."""
    if panel.empty:
        return pd.Series(np.nan, index=pd.Index(universe), dtype=float)
    return _finite(cast(pd.Series, panel.iloc[-1]).astype(float)).reindex(pd.Index(universe))


def score_components(ctx: FeatureContext) -> pd.DataFrame:
    """Diagnostics: the score alongside every input that produced it.

    Not used by the candidate itself — this exists so a human (or a notebook)
    can see WHY a name scored what it scored instead of trusting one number.
    """
    universe = list(ctx.universe)
    price = ctx.price
    if price.empty:
        nan = pd.Series(np.nan, index=pd.Index(universe), dtype=float)
        columns = [
            "variance_ratio",
            "upper_excursions",
            "lower_excursions",
            "rangedness",
            "position",
            "score",
        ]
        return pd.DataFrame({c: nan for c in columns})
    up, dn = band_excursions(price)
    rng = rangedness(price)
    pos = range_position(price)
    return pd.DataFrame(
        {
            "variance_ratio": _last_valid(variance_ratio_mean(price), universe),
            "upper_excursions": _last_valid(up, universe),
            "lower_excursions": _last_valid(dn, universe),
            "rangedness": _last_valid(rng, universe),
            "position": _last_valid(pos, universe),
            "score": _last_valid(cast(pd.DataFrame, rng * pos), universe),
        }
    )
