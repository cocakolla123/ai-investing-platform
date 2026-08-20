"""Accumulation/distribution (group 12), momentum quality (13), attention (14).

What is worth pinning:

* OBV is arithmetic, so it is checked against a hand-computed series — not
  against itself;
* ``sustained_volume_expansion`` must react to twenty elevated sessions and
  NOT to one 10x print (the existing spike detector already covers pops, and
  the pilot scored that shape at -3.03% forward excess return);
* ``unusual_volume_no_news`` degrades to NaN for an uncovered symbol rather
  than flagging every heavy day as unexplained;
* a +30% earned over 60 sessions must score materially better on every
  momentum-quality feature than the same +30% delivered by one gap.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.research.quant.catalog import (
    MOMENTUM_QUALITY_WINDOW,
    FeatureInputs,
    to_ts,
)
from dynamo.research.quant.features import build_raw_features

AS_OF = to_ts("2026-06-26")
QUALITY_FEATURES = (
    "up_session_fraction_60d",
    "max_session_share_60d",
    "drawdown_adj_momentum_60d",
)


def _bars(
    symbol: str,
    closes: np.ndarray,
    volumes: np.ndarray | None = None,
    end: str = "2026-06-26",
) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=len(closes))
    vol = np.full(len(closes), 1_000_000.0) if volumes is None else volumes.astype(float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": idx,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "adj_close": closes,
            "volume": vol,
        }
    )


def _raw(bars: pd.DataFrame, universe: list[str], news: pd.DataFrame | None = None):
    return build_raw_features(FeatureInputs(bars=bars, news=news), AS_OF, universe)


# --------------------------------------------------------------------------
# on-balance volume
# --------------------------------------------------------------------------


def _obv_bars(last_20_steps: list[float]) -> pd.DataFrame:
    """41 closes on a flat 1,000-share tape: 20 warm-up steps (needed because
    a symbol's first session has no prior close and so no OBV), then the 20
    steps under test."""
    steps = np.array([1.0, -1.0] * 10 + last_20_steps)
    closes = np.concatenate([[100.0], 100.0 + np.cumsum(steps)])
    return _bars("AAA", closes, np.full(len(closes), 1_000.0))


def test_obv_slope_matches_a_hand_computed_signed_volume_run() -> None:
    """The 20 sessions under test are 15 up and 5 down, each on 1,000 shares.

    OBV change is ``(15 - 5) x 1,000 = +10,000`` out of ``20 x 1,000 =
    20,000`` traded, so the slope is exactly ``0.5``.
    """
    steps = [1.0] * 15 + [-1.0] * 5
    rng = np.random.default_rng(0)
    rng.shuffle(steps)
    raw = _raw(_obv_bars(steps), ["AAA"])
    assert float(raw.loc["AAA", "obv_slope_20d"]) == 0.5


def test_obv_slope_is_zero_when_buying_and_selling_volume_cancel() -> None:
    raw = _raw(_obv_bars([1.0, -1.0] * 10), ["AAA"])
    assert float(raw.loc["AAA", "obv_slope_20d"]) == 0.0


def test_obv_slope_separates_accumulation_from_distribution() -> None:
    """Same price path, opposite volume placement: the name whose volume
    lands on up sessions accumulates, the other distributes."""
    n = 80
    steps = np.array([1.0, -1.0] * (n // 2))
    closes = np.concatenate([[100.0], 100.0 + np.cumsum(steps)])
    heavy_up = np.concatenate([[1.0], np.where(steps > 0, 5_000.0, 1_000.0)])
    heavy_down = np.concatenate([[1.0], np.where(steps > 0, 1_000.0, 5_000.0)])
    bars = pd.concat(
        [_bars("BUY", closes, heavy_up), _bars("SELL", closes, heavy_down)],
        ignore_index=True,
    )
    raw = _raw(bars, ["BUY", "SELL"])
    assert float(raw.loc["BUY", "obv_slope_60d"]) > 0.5
    assert float(raw.loc["SELL", "obv_slope_60d"]) < -0.5


# --------------------------------------------------------------------------
# sustained expansion is not a spike
# --------------------------------------------------------------------------


def _volume_expansion(volumes: np.ndarray) -> float:
    closes = 100.0 * 1.0005 ** np.arange(len(volumes))
    raw = _raw(_bars("AAA", closes, volumes), ["AAA"])
    return float(raw.loc["AAA", "sustained_volume_expansion"])


def test_sustained_expansion_sees_weeks_of_elevation_and_ignores_one_pop() -> None:
    n = 200
    flat = np.full(n, 1_000_000.0)

    # One enormous day: exactly the shape the existing spike detector already
    # fires on, and the shape the pilot measured at -3.03%. A 20-session
    # MEDIAN barely notices it.
    spike = flat.copy()
    spike[-1] = 10_000_000.0
    assert abs(_volume_expansion(spike)) < 0.05

    # Twenty sessions at 1.5x: no single day is remarkable, the regime is.
    sustained = flat.copy()
    sustained[-20:] = 1_500_000.0
    assert _volume_expansion(sustained) == 0.5

    # A quiet name reads negative — this is signed, not an absolute alarm.
    drying_up = flat.copy()
    drying_up[-20:] = 600_000.0
    assert _volume_expansion(drying_up) < -0.3


# --------------------------------------------------------------------------
# unusual volume WITHOUT news
# --------------------------------------------------------------------------


def _heavy_volume_bars(symbol: str, n: int = 200, heavy_every: int = 10) -> pd.DataFrame:
    closes = 100.0 * 1.0005 ** np.arange(n)
    volumes = np.full(n, 1_000_000.0)
    volumes[np.arange(n) % heavy_every == 0] = 5_000_000.0
    return _bars(symbol, closes, volumes)


def _news_rows(symbol: str, days_ago: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "published_at": [AS_OF - pd.Timedelta(days=d) for d in days_ago],
        }
    )


def test_unexplained_volume_is_nan_for_a_symbol_the_feed_never_covered() -> None:
    """The false-positive guard. COVERED and UNCOVERED have IDENTICAL bars —
    heavy days with no story attached. Only the covered one may score; the
    uncovered one is missing data, not a silent accumulator."""
    bars = pd.concat(
        [_heavy_volume_bars("COVERED"), _heavy_volume_bars("UNCOVERED")],
        ignore_index=True,
    )
    # One story about COVERED, long ago and far from any heavy day: enough to
    # establish the feed sees this symbol, not enough to explain anything.
    news = _news_rows("COVERED", [300])
    raw = _raw(bars, ["COVERED", "UNCOVERED"], news=news)
    assert np.isnan(float(raw.loc["UNCOVERED", "unusual_volume_no_news"]))
    assert float(raw.loc["COVERED", "unusual_volume_no_news"]) > 0.0


def test_unexplained_volume_is_nan_when_no_news_was_collected_at_all() -> None:
    raw = _raw(_heavy_volume_bars("AAA"), ["AAA"], news=None)
    assert np.isnan(float(raw.loc["AAA", "unusual_volume_no_news"]))


def test_explained_heavy_days_do_not_count() -> None:
    """Two names, identical heavy days. One has a story on each heavy day and
    scores 0.0; the other has only the coverage-establishing story and scores
    the full heavy-day rate."""
    bars = pd.concat(
        [_heavy_volume_bars("QUIET"), _heavy_volume_bars("EXPLAINED")],
        ignore_index=True,
    )
    heavy_days = sorted(
        pd.bdate_range(end="2026-06-26", periods=200)[np.arange(200) % 10 == 0]
    )
    explained = pd.DataFrame(
        {"symbol": "EXPLAINED", "published_at": heavy_days}
    )
    news = pd.concat([_news_rows("QUIET", [300]), explained], ignore_index=True)
    raw = _raw(bars, ["QUIET", "EXPLAINED"], news=news)
    assert float(raw.loc["EXPLAINED", "unusual_volume_no_news"]) == 0.0
    assert float(raw.loc["QUIET", "unusual_volume_no_news"]) > 0.05


def test_news_count_reads_the_trailing_60_days_only() -> None:
    bars = pd.concat(
        [_heavy_volume_bars("LOUD"), _heavy_volume_bars("QUIET")], ignore_index=True
    )
    news = pd.concat(
        [
            _news_rows("LOUD", [1, 5, 12, 30, 55]),
            _news_rows("LOUD", [90, 200]),  # outside the window
            _news_rows("QUIET", [300]),
        ],
        ignore_index=True,
    )
    raw = _raw(bars, ["LOUD", "QUIET"], news=news)
    assert float(raw.loc["LOUD", "news_count_60d"]) == 5.0
    # A covered-but-unwritten-about name is a legitimate 0, not NaN: that is
    # the low-attention reading the sleeper screen wants.
    assert float(raw.loc["QUIET", "news_count_60d"]) == 0.0


def test_news_count_is_nan_when_no_news_was_collected() -> None:
    raw = _raw(_heavy_volume_bars("AAA"), ["AAA"], news=None)
    assert np.isnan(float(raw.loc["AAA", "news_count_60d"]))


# --------------------------------------------------------------------------
# momentum quality: a gap is not a climb
# --------------------------------------------------------------------------


def _same_total_return_paths(total: float = 0.30, n: int = 200) -> pd.DataFrame:
    """Two names up exactly ``total`` over the last 60 sessions: one by a
    single overnight gap, one by 60 equal steps."""
    base = np.full(n, 100.0)
    smooth = base.copy()
    step = (1.0 + total) ** (1.0 / MOMENTUM_QUALITY_WINDOW)
    smooth[-MOMENTUM_QUALITY_WINDOW:] = 100.0 * step ** np.arange(
        1, MOMENTUM_QUALITY_WINDOW + 1
    )
    gap = base.copy()
    gap[-MOMENTUM_QUALITY_WINDOW:] = 100.0 * (1.0 + total)
    return pd.concat([_bars("SMOOTH", smooth), _bars("GAP", gap)], ignore_index=True)


def test_one_gap_scores_materially_worse_than_the_same_move_earned() -> None:
    raw = _raw(_same_total_return_paths(), ["SMOOTH", "GAP"])

    # The premise: ret_60d cannot tell them apart at all.
    assert float(raw.loc["SMOOTH", "ret_60d"]) == pytest.approx(
        float(raw.loc["GAP", "ret_60d"]), rel=1e-9
    )

    # Every session participated vs one did.
    assert float(raw.loc["SMOOTH", "up_session_fraction_60d"]) == 1.0
    assert float(raw.loc["GAP", "up_session_fraction_60d"]) < 0.05

    # The single biggest session carried 1/60th of the move vs all of it.
    assert float(raw.loc["SMOOTH", "max_session_share_60d"]) < 0.05
    assert float(raw.loc["GAP", "max_session_share_60d"]) == 1.0


def test_drawdown_adjusted_momentum_penalizes_a_violent_path() -> None:
    """Same +30% over 60 sessions; the second one gave back 30% en route."""
    n = 200
    smooth = np.full(n, 100.0)
    step = 1.30 ** (1.0 / MOMENTUM_QUALITY_WINDOW)
    smooth[-MOMENTUM_QUALITY_WINDOW:] = 100.0 * step ** np.arange(
        1, MOMENTUM_QUALITY_WINDOW + 1
    )
    violent = np.full(n, 100.0)
    half = MOMENTUM_QUALITY_WINDOW // 2
    violent[-MOMENTUM_QUALITY_WINDOW:-half] = np.linspace(98.0, 70.0, half)
    violent[-half:] = np.linspace(75.0, 130.0, half)
    bars = pd.concat(
        [_bars("SMOOTH", smooth), _bars("VIOLENT", violent)], ignore_index=True
    )
    raw = _raw(bars, ["SMOOTH", "VIOLENT"])

    assert float(raw.loc["SMOOTH", "ret_60d"]) == pytest.approx(
        float(raw.loc["VIOLENT", "ret_60d"]), rel=1e-9
    )
    smooth_score = float(raw.loc["SMOOTH", "drawdown_adj_momentum_60d"])
    violent_score = float(raw.loc["VIOLENT", "drawdown_adj_momentum_60d"])
    assert smooth_score > violent_score
    # ~30% drawdown divides the score by ~1.3.
    assert violent_score < smooth_score / 1.25


def test_momentum_quality_is_nan_without_enough_history() -> None:
    short = 100.0 * 1.001 ** np.arange(MOMENTUM_QUALITY_WINDOW - 5)
    raw = _raw(_bars("AAA", short), ["AAA"])
    for col in QUALITY_FEATURES:
        assert np.isnan(float(raw.loc["AAA", col]))


# --------------------------------------------------------------------------
# calendar invariance (the property the whole catalog is held to)
# --------------------------------------------------------------------------


NEW_FEATURES = (
    "obv_slope_20d",
    "obv_slope_60d",
    "sustained_volume_expansion",
    *QUALITY_FEATURES,
)


def test_new_features_are_invariant_to_crypto_in_the_universe() -> None:
    rng = np.random.default_rng(4)
    b_idx = pd.bdate_range(end="2026-06-26", periods=400)
    d_idx = pd.date_range(end="2026-06-26", periods=560, freq="D")
    equity = pd.DataFrame(
        {
            "symbol": "AAA",
            "ts": b_idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(b_idx)))),
            "volume": rng.integers(100_000, 900_000, len(b_idx)).astype(float),
        }
    )
    equity["adj_close"] = equity["close"]
    crypto = pd.DataFrame(
        {
            "symbol": "BTC/USD",
            "ts": d_idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 500.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(d_idx)))),
            "volume": rng.integers(100_000, 900_000, len(d_idx)).astype(float),
        }
    )
    crypto["adj_close"] = crypto["close"]

    alone = _raw(equity, ["AAA"])
    mixed = _raw(pd.concat([equity, crypto], ignore_index=True), ["AAA", "BTC/USD"])
    pd.testing.assert_series_equal(
        cast(pd.Series, mixed.loc["AAA", list(NEW_FEATURES)]),
        cast(pd.Series, alone.loc["AAA", list(NEW_FEATURES)]),
    )
    assert bool(mixed.loc["AAA", list(NEW_FEATURES)].notna().all())
