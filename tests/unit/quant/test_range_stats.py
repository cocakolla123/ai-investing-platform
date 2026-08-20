"""Range/mean-reversion statistics behind the ``range_reversion`` candidate.

Four things are pinned here:

1. the SIGN CONVENTION — strongly positive at the low of an established
   range, strongly negative at its high, ~0 for a pure trend whatever the
   price level (the "do not buy the dip in a downtrend" property);
2. the variance-ratio ARITHMETIC, checked against a hand computation on a
   known series rather than against the implementation restating itself;
3. NaN-safety on short history — never a raise, never a fabricated number;
4. CALENDAR INVARIANCE — the score for an equity is identical whether or not
   crypto shares the universe. This is the property
   ``tests/unit/quant/test_calendar_invariance.py`` pins for the catalog, and
   the reason the score is built from the per-symbol window helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import FeatureContext, FeatureInputs, build_context, to_ts
from dynamo.research.quant.range_stats import (
    VR_WINDOW,
    band_excursions,
    range_position,
    range_reversion_score,
    rangedness,
    score_components,
    variance_ratio,
)

AS_OF = to_ts("2026-03-03")
SESSIONS = 420  # comfortably past the 120-session variance window


# --------------------------------------------------------------------------
# synthetic series
# --------------------------------------------------------------------------


def oscillator(
    n: int = SESSIONS,
    low: float = 4.30,
    high: float = 5.30,
    theta: float = 0.10,
    sigma: float = 0.10,
    seed: int = 4,
) -> np.ndarray:
    """The user's father's stock, in a test: an Ornstein-Uhlenbeck price that
    reverts to the middle of a $4.30-$5.30 band and is clipped to it."""
    rng = np.random.default_rng(seed)
    mid = (low + high) / 2.0
    x = mid
    out = []
    for _ in range(n):
        x += theta * (mid - x) + rng.normal(0.0, sigma)
        out.append(float(min(max(x, low), high)))
    return np.asarray(out)


def trend(n: int = SESSIONS, drift: float = 0.0015, vol: float = 0.012, seed: int = 5):
    """Drifting geometric random walk: variance ratio ~1, no true range."""
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def bars_from(series: dict[str, np.ndarray], freq: str = "B") -> pd.DataFrame:
    length = max(len(v) for v in series.values())
    ts = (
        pd.bdate_range(end=AS_OF, periods=length)
        if freq == "B"
        else pd.date_range(end=AS_OF, periods=length, freq="D")
    )
    rows = []
    for symbol, prices in series.items():
        for t, p in zip(ts[-len(prices) :], prices, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "ts": t,
                    "open": p,
                    "high": p * 1.01,
                    "low": p * 0.99,
                    "close": p,
                    "adj_close": p,
                    "volume": 500_000,
                }
            )
    return pd.DataFrame(rows)


def context_for(series: dict[str, np.ndarray], freq: str = "B") -> FeatureContext:
    bars = bars_from(series, freq)
    as_of = to_ts(bars["ts"].max())
    return build_context(FeatureInputs(bars=bars), as_of, list(series))


def _truncate_at_extreme(
    prices: np.ndarray, *, kind: str, window: int = 60, min_length: int = 220
) -> np.ndarray:
    """Cut the series so its LAST session IS the low (or high) of its own
    trailing ``window`` sessions — the moment the score should be extreme.

    The cut point has to be a running extreme of the TRUNCATED series, not of
    the original tail, or the truncation moves the window out from under it.
    """
    series = pd.Series(prices)
    rolling = series.rolling(window, min_periods=window)
    extreme = (rolling.min() if kind == "low" else rolling.max()).to_numpy()
    is_extreme = np.isclose(series.to_numpy(), extreme)
    candidates = np.flatnonzero(is_extreme & (np.arange(len(series)) >= min_length))
    assert candidates.size, f"no running {kind} after session {min_length}"
    return prices[: int(candidates[-1]) + 1]


# --------------------------------------------------------------------------
# 1. sign convention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [3, 4, 7, 11])
def test_score_is_strongly_positive_at_the_low_of_an_established_range(seed: int) -> None:
    prices = _truncate_at_extreme(oscillator(seed=seed), kind="low")
    score = range_reversion_score(context_for({"OSC": prices}))
    assert float(score["OSC"]) > 0.20


@pytest.mark.parametrize("seed", [3, 4, 7, 11])
def test_score_is_strongly_negative_at_the_high_of_an_established_range(seed: int) -> None:
    prices = _truncate_at_extreme(oscillator(seed=seed), kind="high")
    score = range_reversion_score(context_for({"OSC": prices}))
    assert float(score["OSC"]) < -0.20


def test_score_is_about_zero_for_a_pure_trend_in_either_direction() -> None:
    """The construction's whole point: a trending name is not a range, so it
    scores ~0 EVEN WHEN it is sitting at its 60-session low. Buying the dip
    in a downtrend is the failure mode this refuses."""
    up = trend(drift=0.0015, seed=5)
    down = trend(drift=-0.0025, seed=17)  # ends near its 60-session low
    ctx = context_for({"UP": up, "DOWN": down})
    components = score_components(ctx)
    assert abs(float(components.loc["UP", "score"])) < 0.05
    assert abs(float(components.loc["DOWN", "score"])) < 0.05
    # The downtrend really is near its low — position is bullish and the
    # rangedness gate is what zeroes it, not a happy accident of placement.
    assert float(components.loc["DOWN", "position"]) > 0.2
    assert float(components.loc["DOWN", "rangedness"]) == pytest.approx(0.0, abs=1e-12)


def test_score_is_bounded_and_the_range_low_beats_the_range_high() -> None:
    prices = oscillator(seed=4)
    low = range_reversion_score(context_for({"OSC": _truncate_at_extreme(prices, kind="low")}))
    high = range_reversion_score(context_for({"OSC": _truncate_at_extreme(prices, kind="high")}))
    assert -1.0 <= float(high["OSC"]) < float(low["OSC"]) <= 1.0


def test_rangedness_is_unsigned_and_position_carries_the_sign() -> None:
    prices = oscillator(seed=4)
    ctx_low = context_for({"OSC": _truncate_at_extreme(prices, kind="low")})
    ctx_high = context_for({"OSC": _truncate_at_extreme(prices, kind="high")})
    for ctx in (ctx_low, ctx_high):
        panel = rangedness(ctx.price)
        values = panel.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0
    assert float(range_position(ctx_low.price).iloc[-1]["OSC"]) > 0
    assert float(range_position(ctx_high.price).iloc[-1]["OSC"]) < 0


def test_consecutive_sessions_outside_a_band_count_as_one_excursion() -> None:
    """Fifteen sessions pinned above the band is ONE excursion, not fifteen —
    otherwise the counter measures how long a breakout lasted rather than how
    often the name oscillated."""
    quiet = np.array([100.0 * (1 + (0.001 if i % 2 else -0.001)) for i in range(200)])
    prices = np.concatenate([quiet, np.full(15, 130.0)])
    ctx = context_for({"STEP": prices})
    up, dn = band_excursions(ctx.price)
    assert float(up.iloc[-1]["STEP"]) == 1.0
    assert float(dn.iloc[-1]["STEP"]) == 0.0


def test_a_trend_visits_its_bands_lopsidedly_while_an_oscillator_does_not() -> None:
    """The asymmetry the two-sided ``min(up, dn)`` exploits. Note it is only
    an ASYMMETRY, not a veto: over 120 sessions a drifting random walk still
    pokes its opposite band once or twice, which is exactly why ``rangedness``
    is a conjunction with the variance ratio rather than a touch count alone.
    """
    ctx = context_for({"OSC": oscillator(seed=4), "UP": trend(seed=5)})
    up, dn = band_excursions(ctx.price)
    osc_up, osc_dn = float(up.iloc[-1]["OSC"]), float(dn.iloc[-1]["OSC"])
    trend_up, trend_dn = float(up.iloc[-1]["UP"]), float(dn.iloc[-1]["UP"])
    assert min(osc_up, osc_dn) >= 2.0
    assert abs(osc_up - osc_dn) <= max(osc_up, osc_dn) / 2.0  # balanced
    assert trend_up >= 2.0 * trend_dn  # lopsided: rides the upper band
    # An excursion count can never exceed the number of sessions counted.
    assert max(osc_up, osc_dn) <= 120.0


# --------------------------------------------------------------------------
# 2. variance-ratio arithmetic on a known series
# --------------------------------------------------------------------------


def test_variance_ratio_matches_a_hand_computation() -> None:
    """VR_k against numpy on the same numbers — the definition, not a
    restatement of the implementation."""
    rng = np.random.default_rng(21)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
    ctx = context_for({"AAA": prices})
    k, window = 5, VR_WINDOW

    log_returns = np.diff(np.log(prices))
    k_returns = np.convolve(log_returns, np.ones(k), mode="valid")  # overlapping k-sums
    expected = float(np.var(k_returns[-window:], ddof=1)) / (
        k * float(np.var(log_returns[-window:], ddof=1))
    )
    actual = float(variance_ratio(ctx.price, k).iloc[-1]["AAA"])
    assert actual == pytest.approx(expected, rel=1e-10)


def test_variance_ratio_is_about_one_for_a_random_walk() -> None:
    """The null the statistic is read against: iid increments give VR ~ 1."""
    rng = np.random.default_rng(99)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 1200)))
    ctx = context_for({"RW": prices})
    assert float(variance_ratio(ctx.price, 5).iloc[-1]["RW"]) == pytest.approx(1.0, abs=0.25)


def test_variance_ratio_is_below_one_for_a_mean_reverting_series() -> None:
    ctx = context_for({"OSC": oscillator(seed=4)})
    assert float(variance_ratio(ctx.price, 10).iloc[-1]["OSC"]) < 0.9


def test_variance_ratio_is_nan_when_the_price_never_moves() -> None:
    """Zero one-period variance is a division by zero, which must be NaN and
    not an infinity that survives into a score."""
    ctx = context_for({"FLAT": np.full(SESSIONS, 7.0)})
    assert np.isnan(float(variance_ratio(ctx.price, 5).iloc[-1]["FLAT"]))
    assert np.isnan(float(range_reversion_score(ctx)["FLAT"]))


# --------------------------------------------------------------------------
# 3. NaN safety on short history
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sessions", [1, 2, 25, 80, 129, 138])
def test_short_history_is_nan_and_never_raises(n_sessions: int) -> None:
    """139 sessions of own history is the floor (120 counted excursion
    sessions on top of the 20-session band; the variance ratio needs 130).
    Below it there is no number to report — NaN, not a value estimated off a
    handful of points."""
    ctx = context_for({"NEW": oscillator(n=n_sessions, seed=4)})
    score = range_reversion_score(ctx)
    assert list(score.index) == ["NEW"]
    assert np.isnan(float(score["NEW"]))


def test_empty_price_panel_is_nan_for_every_symbol() -> None:
    ctx = build_context(FeatureInputs(bars=None), AS_OF, ["AAA", "BBB"])
    score = range_reversion_score(ctx)
    assert list(score.index) == ["AAA", "BBB"]
    assert score.isna().all()


def test_symbols_absent_from_the_bars_are_nan_not_missing() -> None:
    bars = bars_from({"OSC": oscillator(seed=4)})
    ctx = build_context(FeatureInputs(bars=bars), AS_OF, ["OSC", "GHOST"])
    score = range_reversion_score(ctx)
    assert list(score.index) == ["OSC", "GHOST"]
    assert np.isnan(float(score["GHOST"]))
    assert np.isfinite(float(score["OSC"]))


def test_a_short_history_symbol_does_not_poison_its_peers() -> None:
    ctx = context_for({"OSC": oscillator(seed=4), "NEW": oscillator(n=40, seed=6)})
    score = range_reversion_score(ctx)
    assert np.isnan(float(score["NEW"]))
    assert np.isfinite(float(score["OSC"]))


# --------------------------------------------------------------------------
# 4. calendar invariance — the regression this whole design exists to avoid
# --------------------------------------------------------------------------


def test_score_is_invariant_to_crypto_sharing_the_universe() -> None:
    """Same equities, same bars, same as-of date — with and without three
    7-day-a-week crypto pairs in the panel. Identical values.

    ``catalog._pivot`` indexes the panel by the UNION of every symbol's
    timestamps, so adding crypto inflates a 5-day equity panel with ~2 empty
    rows in every 7. Any positional operation on that shared index would
    silently measure calendar days: a 120-session variance window would need
    168 calendar rows and never fill. Reusing the per-symbol helpers is what
    makes this pass.
    """
    equities = {
        "AAA": oscillator(seed=4),
        "BBB": oscillator(seed=7, low=12.0, high=15.0, sigma=0.30),
        "CCC": trend(seed=5),
    }
    crypto = {
        "BTC/USD": trend(n=588, seed=31),  # same span, 7 days a week
        "ETH/USD": oscillator(n=588, seed=32),
        "SOL/USD": trend(n=588, seed=33),
    }
    equity_bars = bars_from(equities, freq="B")
    crypto_bars = bars_from(crypto, freq="D")

    without = range_reversion_score(
        build_context(FeatureInputs(bars=equity_bars), AS_OF, list(equities))
    )
    mixed = build_context(
        FeatureInputs(bars=pd.concat([equity_bars, crypto_bars], ignore_index=True)),
        AS_OF,
        [*equities, *crypto],
    )
    with_crypto = range_reversion_score(mixed)

    pd.testing.assert_series_equal(with_crypto.loc[list(equities)], without)
    # Not a vacuous pass: the equities must have real values on both sides.
    assert without.notna().all()


def test_every_component_is_invariant_to_crypto_sharing_the_universe() -> None:
    """The score is a product, so component-level invariance is the stronger
    statement: no single term may leak the panel's calendar."""
    equities = {"AAA": oscillator(seed=4), "BBB": trend(seed=5)}
    crypto = {"BTC/USD": oscillator(n=588, seed=32)}
    equity_bars = bars_from(equities, freq="B")
    mixed_bars = pd.concat([equity_bars, bars_from(crypto, freq="D")], ignore_index=True)

    without = score_components(
        build_context(FeatureInputs(bars=equity_bars), AS_OF, list(equities))
    )
    with_crypto = score_components(
        build_context(FeatureInputs(bars=mixed_bars), AS_OF, [*equities, *crypto])
    )
    pd.testing.assert_frame_equal(with_crypto.loc[list(equities)], without)
    assert bool(without["variance_ratio"].notna().all())
