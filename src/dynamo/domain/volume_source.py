"""Which tape a bar's ``volume`` was measured on, and how to compare across them.

The platform stores bars from feeds that measure *different things* under the
same column name. Tiingo reports **consolidated** volume — every share printed
on every US venue, the number "average daily volume" normally means. Alpaca's
free feed reports **IEX only**: one venue's share of that tape. Both land in
``bars_1d.volume`` and both get multiplied by a close to make "dollar volume",
so a threshold written against one silently means something else against the
other.

That is not a rounding difference. On the measured sample IEX prints roughly
1/15th of the consolidated tape, so a $10M/day floor keyed off Alpaca volume
is really asking for ~$150M/day consolidated — it rejects most of the market.
Read the other way, a floor *calibrated* on Tiingo and then pointed at Alpaca
rows admits names ~15x thinner than intended. Whole-market surveillance runs
on the Alpaca rows, so this is the common case, not the exotic one.

The convention this module establishes:

    **Every dollar-volume threshold in the codebase is expressed in
    consolidated-equivalent dollars.** Observed volume is scaled UP to
    consolidated-equivalent at the point of comparison, via
    :func:`consolidated_equivalent_dollar_volume`, using the bar's ``source``.

Thresholds then mean the same thing regardless of which feed supplied the row,
and a feed change moves the scaling factor rather than silently re-calibrating
every gate in the system.

Domain-pure on purpose: ``dynamo.risk`` may not import ``dynamo.data`` (the
import-linter "core logic does not import adapters" contract), but both need
this, and everything may import domain.
"""

from __future__ import annotations

from collections.abc import Mapping

# --- The measurement ------------------------------------------------------
#
# IEX's share of consolidated volume, as measured against this repo's own
# database rather than taken from a vendor factsheet.
#
# PROVENANCE (read this before trusting the number):
#
#   Sample:   the ONLY 4 symbols carrying both an Alpaca (IEX) and a Tiingo
#             (consolidated) bar for the same session: XLB, XLC, XLP, XLY.
#             620 symbol-days over 2026-01-01..2026-08-14.
#   Method:   per (symbol, session), ratio = alpaca.volume / tiingo.volume;
#             median per symbol, then median across symbols.
#   Result:   XLB 6.4%, XLC 6.3%, XLP 7.2%, XLY 5.7%  ->  6.6% (~15.2x).
#
# THIS IS AN ESTIMATE, NOT A CALIBRATION. Three caveats, all material:
#
#   1. n = 4 symbols, and all four are large sector ETFs. Single-stock and
#      small-cap routing differs from ETF routing; the ETF-only sample almost
#      certainly does not generalise cleanly to the 11k-name tail that the
#      whole-market sweep actually scans.
#   2. The ratio is NOT stable over time. Measured on the same 4 symbols over
#      longer windows: 3.2% across all of 2024-01..2026-08, 6.4% over 2026 YTD,
#      8.1% over the most recent 60 sessions. It has roughly doubled in 2.5
#      years and is still drifting up. A constant is the wrong shape for a
#      moving quantity; it is used here because a per-symbol, per-period
#      estimate needs dual-source coverage the database does not have.
#   3. Consequently, treat any threshold derived from this as accurate to
#      within a factor of ~2, not to two significant figures. It is a unit
#      conversion that stops errors of 15x, not a precision instrument.
#
# TO RE-MEASURE: join bars_1d to itself on (symbol, ts::date) with
# a.source='alpaca' and t.source='tiingo', both volumes > 0, and take the
# median of per-symbol median ratios over a recent window. Widening the sample
# requires backfilling Tiingo bars for more symbols — that, not better
# arithmetic, is what would turn this estimate into a calibration.
IEX_CONSOLIDATED_VOLUME_SHARE: float = 0.066

# --- The crypto venue share -----------------------------------------------
#
# Alpaca's crypto feed reports ITS OWN venue's volume — there is no
# consolidated crypto tape to read, but there IS a whole market this venue is
# a sliver of, and dollar-volume thresholds in this codebase are calibrated
# against whole-market liquidity. Left unscaled, the venue number walks into
# the same $1M ``min_adv_dollar`` floor the equities use and vetoes the
# entire crypto sleeve: the first live risk campaign zeroed every crypto buy,
# reading ~$4k/day of "dollar volume" for AVAX.
#
# PROVENANCE (an ESTIMATE, NOT A CALIBRATION — same caveat as the IEX share):
#
#   Own venue (this repo's bars_1d, source='alpaca_crypto', median daily
#   close x volume over 2026-07-01..2026-08-16, 47 sessions/pair):
#     BTC/USD ~$64k   ETH/USD ~$37k   SOL/USD ~$10k
#     AVAX/USD ~$4.9k LINK/USD ~$3.6k
#   Whole market (public aggregator figures — CoinGecko/CoinMarketCap global
#   spot volume, order of magnitude, 2024-2025):
#     BTC ~$10-30B/day, ETH ~$5-15B/day, SOL ~$1-3B/day,
#     AVAX and LINK each ~$100-500M/day.
#   Implied venue share: ~1e-6 (BTC) to ~2e-5 (AVAX/LINK).
#
# The constant is 1e-3 — at least ~50x LARGER than any implied share above,
# i.e. it deliberately under-scales. Scaling by it still under-states every
# pair's whole-market liquidity by >= 50x, so it fails toward rejecting a
# trade, while restoring the sleeve's floors and entry caps to roughly the
# right order of magnitude (AVAX: $4.9k observed -> ~$4.9M equivalent vs
# >= $250M reported globally). Sizing participation caps remain bounded by
# the far tighter crypto sleeve caps (settings.crypto_max_position).
#
# TO RE-MEASURE: median close x volume per pair from bars_1d
# (source='alpaca_crypto') over a recent window, against a public global
# spot-volume aggregate for the same pairs and window; take the max implied
# share and keep a >= 10x conservative margin above it.
ALPACA_CRYPTO_CONSOLIDATED_VOLUME_SHARE: float = 0.001

# Fraction of consolidated volume each bar source observes. A source absent
# from this map is assumed consolidated (share 1.0) — the conservative choice,
# since it under-states rather than over-states a name's liquidity and so
# fails toward rejecting a trade rather than admitting one.
VOLUME_SOURCE_SHARE: Mapping[str, float] = {
    # Free Alpaca market data is the IEX feed only (doc 02 §1).
    "alpaca": IEX_CONSOLIDATED_VOLUME_SHARE,
    # Tiingo EOD is consolidated-tape volume — the reference this is measured
    # against, so 1.0 by construction.
    "tiingo": 1.0,
    # Alpaca's own crypto venue vs the global spot market (see above).
    "alpaca_crypto": ALPACA_CRYPTO_CONSOLIDATED_VOLUME_SHARE,
}

DEFAULT_VOLUME_SOURCE_SHARE: float = 1.0


def volume_source_share(source: str | None) -> float:
    """Fraction of the consolidated tape that ``source`` observes.

    Unknown or missing sources return 1.0 (assume consolidated). See the
    module docstring for why that default is the conservative one.
    """
    if not source:
        return DEFAULT_VOLUME_SOURCE_SHARE
    return VOLUME_SOURCE_SHARE.get(str(source), DEFAULT_VOLUME_SOURCE_SHARE)


def consolidated_equivalent_dollar_volume(
    observed_dollar_volume: float, source: str | None
) -> float:
    """Scale a feed's observed dollar volume up to consolidated-equivalent.

    This is the function that makes a threshold portable. Given $66k/day of
    observed IEX dollar volume it returns ~$1M/day of consolidated-equivalent
    dollar volume, which is the number a liquidity floor should be compared
    against.

    Non-finite or non-positive input returns 0.0 — an unknown volume is not
    evidence of liquidity.
    """
    value = float(observed_dollar_volume)
    if not (value > 0.0) or value != value or value in (float("inf"), float("-inf")):
        return 0.0
    share = volume_source_share(source)
    if not (share > 0.0):
        return 0.0
    return value / share


def observed_dollar_volume_threshold(consolidated_threshold: float, source: str | None) -> float:
    """Inverse of :func:`consolidated_equivalent_dollar_volume`.

    Converts a consolidated-equivalent threshold into the observed number a
    given feed would have to print to clear it. Useful when the comparison has
    to happen in observed units (e.g. a vectorised filter over a frame, or an
    operator-facing message that quotes the raw figure a feed will show).
    """
    value = float(consolidated_threshold)
    if not (value > 0.0):
        return 0.0
    return value * volume_source_share(source)


def dominant_source(sources: object) -> str | None:
    """Best-effort single source label for a group of bars.

    Bars for one symbol can in principle mix feeds. Rather than average two
    incompatible scales, pick the source observing the SMALLEST share of the
    tape — the most conservative reading of the group's liquidity. Returns
    ``None`` for an empty or unusable input, which callers treat as
    "assume consolidated".
    """
    try:
        labels = {str(s) for s in sources if s is not None and str(s) != "nan"}  # type: ignore[union-attr]
    except TypeError:
        return None
    if not labels:
        return None
    return min(labels, key=volume_source_share)
