"""The sleeper screen: composite arithmetic, determinism, honest degradation.

The screen is unvalidated by construction, so these tests do not (and cannot)
assert that it picks winners. They assert that it computes what its docstring
says it computes, that it is deterministic, and — the one that actually
protects anybody — that a name missing its evidence drops out instead of
scoring on the components it happens to have.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.research.screens import (
    SLEEPER_COMPONENT_WEIGHTS,
    SleeperConfig,
    sleeper_scores,
)

COLUMNS = (
    "rev_growth_yoy",
    "gross_margin_ttm",
    "obv_slope_60d",
    "sustained_volume_expansion",
    "news_count_60d",
    "ret_60d",
    "max_session_share_60d",
    "days_to_exit",
)


def _frame(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame.reindex(columns=pd.Index(COLUMNS))
    frame.index.name = "symbol"
    return frame


def _sleeper(**overrides: float) -> dict[str, float]:
    """A textbook sleeper: growing, accumulated, unwritten-about, up quietly."""
    base = {
        "rev_growth_yoy": 0.30,
        "gross_margin_ttm": 0.55,
        "obv_slope_60d": 0.40,
        "sustained_volume_expansion": 0.60,
        "news_count_60d": 1.0,
        "ret_60d": 0.12,
        "max_session_share_60d": 0.04,
        "days_to_exit": 0.5,
    }
    base.update(overrides)
    return base


def _dud(**overrides: float) -> dict[str, float]:
    """Its opposite on every axis."""
    base = {
        "rev_growth_yoy": -0.15,
        "gross_margin_ttm": 0.10,
        "obv_slope_60d": -0.40,
        "sustained_volume_expansion": -0.50,
        "news_count_60d": 400.0,
        "ret_60d": -0.20,
        "max_session_share_60d": 0.70,
        "days_to_exit": 40.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------


def test_weights_are_documented_and_sum_to_one() -> None:
    assert sum(SLEEPER_COMPONENT_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(SLEEPER_COMPONENT_WEIGHTS) == {
        "improving",
        "accumulation",
        "low_attention",
        "quiet_momentum",
        "liquidity",
    }


def test_the_composite_is_the_weighted_mean_of_its_components() -> None:
    scored = sleeper_scores(_frame({"BEST": _sleeper(), "WORST": _dud()}))
    for symbol in ("BEST", "WORST"):
        row = scored.loc[symbol]
        expected = sum(
            float(row[name]) * weight for name, weight in SLEEPER_COMPONENT_WEIGHTS.items()
        )
        assert float(row["sleeper_score"]) == pytest.approx(expected)
    assert float(scored.loc["BEST", "sleeper_score"]) > float(scored.loc["WORST", "sleeper_score"])
    assert float(scored.loc["BEST", "components_present"]) == 5.0


def test_low_attention_rewards_the_name_nobody_is_writing_about() -> None:
    """Everything else identical; only the story count differs."""
    scored = sleeper_scores(
        _frame(
            {
                "QUIET": _sleeper(news_count_60d=0.0),
                "LOUD": _sleeper(news_count_60d=250.0),
            }
        )
    )
    assert float(scored.loc["QUIET", "low_attention"]) > float(scored.loc["LOUD", "low_attention"])
    assert float(scored.loc["QUIET", "sleeper_score"]) > float(scored.loc["LOUD", "sleeper_score"])


def test_momentum_is_banded_so_the_biggest_gainer_does_not_win() -> None:
    """The whole point of "positive but not extreme": a name up 12% scores
    full marks, a name up 90% scores nothing, and a faller scores nothing."""
    scored = sleeper_scores(
        _frame(
            {
                "QUIET": _sleeper(ret_60d=0.12),
                "PARABOLIC": _sleeper(ret_60d=0.90),
                "FALLING": _sleeper(ret_60d=-0.10),
                "FLAT": _sleeper(ret_60d=0.0),
            }
        )
    )
    momentum = cast(pd.Series, scored["quiet_momentum"])
    assert float(momentum["QUIET"]) > float(momentum["PARABOLIC"])
    assert float(momentum["QUIET"]) > float(momentum["FALLING"])
    assert float(scored.loc["QUIET", "sleeper_score"]) == scored["sleeper_score"].max()
    # Nothing above ~2x the ceiling gets any momentum credit at all.
    config = SleeperConfig()
    assert 2 * config.momentum_ceiling < 0.90


def test_a_move_earned_over_many_sessions_beats_the_same_move_gapped() -> None:
    scored = sleeper_scores(
        _frame(
            {
                "EARNED": _sleeper(ret_60d=0.12, max_session_share_60d=0.03),
                "GAPPED": _sleeper(ret_60d=0.12, max_session_share_60d=0.95),
            }
        )
    )
    assert float(scored.loc["EARNED", "quiet_momentum"]) > float(
        scored.loc["GAPPED", "quiet_momentum"]
    )


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_scores_are_deterministic_and_order_independent() -> None:
    rows = {
        f"S{i:02d}": _sleeper(ret_60d=0.05 + i / 100.0, news_count_60d=float(i)) for i in range(12)
    }
    frame = _frame(rows)

    first = sleeper_scores(frame)
    second = sleeper_scores(frame)
    pd.testing.assert_frame_equal(first, second)

    shuffled = frame.loc[list(reversed(frame.index))]
    reordered = sleeper_scores(shuffled).reindex(frame.index)
    pd.testing.assert_frame_equal(reordered, first)


def test_tied_inputs_score_identically() -> None:
    scored = sleeper_scores(_frame({"A": _sleeper(), "B": _sleeper(), "C": _dud()}))
    assert float(scored.loc["A", "sleeper_score"]) == float(scored.loc["B", "sleeper_score"])


# --------------------------------------------------------------------------
# degradation — the test that actually protects somebody
# --------------------------------------------------------------------------


def test_a_name_with_no_fundamentals_is_discounted_not_zeroed() -> None:
    """No XBRL is not the same as deteriorating. The fundamentals component is
    dropped and the remaining weights renormalize, so the name still competes
    on the evidence it does have."""
    rows = {
        "NO_XBRL": _sleeper(),
        "PEER": _sleeper(),
        "DUD": _dud(),
    }
    frame = _frame(rows)
    frame.loc["NO_XBRL", ["rev_growth_yoy", "gross_margin_ttm"]] = np.nan
    scored = sleeper_scores(frame)

    assert np.isnan(float(scored.loc["NO_XBRL", "improving"]))
    assert float(scored.loc["NO_XBRL", "components_present"]) == 4.0
    assert not np.isnan(float(scored.loc["NO_XBRL", "sleeper_score"]))
    # Scored on four components, it must still beat the name that is bad on
    # everything — a zero for "unknown" would have buried it instead.
    assert float(scored.loc["NO_XBRL", "sleeper_score"]) > float(scored.loc["DUD", "sleeper_score"])


def test_a_name_without_the_thesis_components_drops_out_entirely() -> None:
    """Accumulation and low attention ARE the sleeper claim. Missing either
    one, there is nothing to say and the screen says nothing — rather than
    quietly scoring the name on fundamentals and liquidity and calling the
    result a sleeper."""
    frame = _frame({"NO_NEWS": _sleeper(), "NO_VOLUME": _sleeper(), "FINE": _sleeper()})
    frame.loc["NO_NEWS", "news_count_60d"] = np.nan
    frame.loc["NO_VOLUME", ["obv_slope_60d", "sustained_volume_expansion"]] = np.nan
    scored = sleeper_scores(frame)

    assert np.isnan(float(scored.loc["NO_NEWS", "sleeper_score"]))
    assert np.isnan(float(scored.loc["NO_VOLUME", "sleeper_score"]))
    assert not np.isnan(float(scored.loc["FINE", "sleeper_score"]))


def test_too_few_components_scores_nan() -> None:
    frame = _frame({"THIN": _sleeper(), "FINE": _sleeper()})
    frame.loc["THIN", ["rev_growth_yoy", "gross_margin_ttm", "days_to_exit"]] = np.nan
    scored = sleeper_scores(frame)
    assert float(scored.loc["THIN", "components_present"]) == 3.0
    assert np.isnan(float(scored.loc["THIN", "sleeper_score"]))


def test_a_frame_missing_whole_columns_does_not_raise() -> None:
    """Feature frames from an older catalog, or a universe with no short
    interest wired up yet, must degrade rather than explode."""
    frame = _frame({"AAA": _sleeper(), "BBB": _dud()}).drop(columns=["days_to_exit"])
    scored = sleeper_scores(frame)
    assert np.isnan(float(scored.loc["AAA", "liquidity"]))
    assert float(scored.loc["AAA", "components_present"]) == 4.0


def test_an_empty_universe_returns_an_empty_frame() -> None:
    scored = sleeper_scores(_frame({}))
    assert scored.empty
    assert "sleeper_score" in scored.columns
