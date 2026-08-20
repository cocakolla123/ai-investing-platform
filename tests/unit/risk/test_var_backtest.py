"""VaR exceptions are counted from realized returns (doc 06 §6).

``var_exceptions_250d`` defaulted to 0, no production ``RiskEngineInputs``
passed it, and ``kupiec_traffic_light(0)`` is always GREEN — so "too many VaR
exceptions -> caps tighten 25%" was unreachable code.
"""

from __future__ import annotations

import math

from dynamo.risk.var import (
    MIN_BACKTEST_OBSERVATIONS,
    VarTrafficLight,
    backtest_var,
    count_var_exceptions,
)


def _clean(n: int, var: float = 0.02, ret: float = 0.001) -> list[tuple[float, float]]:
    return [(var, ret)] * n


def test_a_loss_beyond_the_forecast_is_an_exception() -> None:
    exceptions, worst = count_var_exceptions([(0.02, -0.03)])
    assert exceptions == 1
    assert math.isclose(worst, 0.01)


def test_a_loss_inside_the_forecast_is_not() -> None:
    assert count_var_exceptions([(0.02, -0.015)]) == (0, 0.0)


def test_a_loss_exactly_at_the_forecast_is_not_an_exception() -> None:
    assert count_var_exceptions([(0.02, -0.02)])[0] == 0


def test_gains_are_never_exceptions() -> None:
    assert count_var_exceptions([(0.01, 0.25)])[0] == 0


def test_days_without_a_forecast_are_skipped_not_counted() -> None:
    """A day the model did not speak on is evidence neither way."""
    observations = [(0.0, -0.5), (float("nan"), -0.5), (-1.0, -0.5)]
    assert count_var_exceptions(observations) == (0, 0.0)


def test_a_non_finite_return_is_skipped() -> None:
    assert count_var_exceptions([(0.02, float("nan"))]) == (0, 0.0)


def test_worst_excess_is_the_largest_overshoot() -> None:
    _, worst = count_var_exceptions([(0.02, -0.03), (0.02, -0.09), (0.02, -0.025)])
    assert math.isclose(worst, 0.07)


# --- the traffic light and the cap multiplier ----------------------------


def test_a_short_history_never_tightens_caps() -> None:
    """Do not tighten on noise: below 60 observations the light stays green."""
    result = backtest_var([(0.02, -0.5)] * (MIN_BACKTEST_OBSERVATIONS - 1))
    assert result.light is VarTrafficLight.GREEN
    assert result.tighten_factor == 1.0
    assert result.caps_tightened is False


def test_a_well_behaved_model_stays_green() -> None:
    result = backtest_var(_clean(250))
    assert (result.exceptions, result.observations) == (0, 250)
    assert result.light is VarTrafficLight.GREEN
    assert result.tighten_factor == 1.0


def test_a_cluster_of_exceptions_tightens_caps_25_percent() -> None:
    """doc 06 §6: the red zone multiplies caps by 0.75."""
    observations = _clean(220) + [(0.02, -0.09)] * 30
    result = backtest_var(observations)
    assert result.exceptions == 30
    assert result.light is VarTrafficLight.RED
    assert result.tighten_factor == 0.75
    assert result.caps_tightened is True


def test_a_moderate_excess_leaves_green_before_it_goes_red() -> None:
    observations = _clean(232) + [(0.02, -0.09)] * 18
    result = backtest_var(observations)
    assert result.light is VarTrafficLight.YELLOW
    assert result.tighten_factor == 1.0  # yellow warns; only red tightens


def test_only_the_trailing_window_counts() -> None:
    """Ancient exceptions must age out — the model is judged on 250 sessions."""
    stale_breaches = [(0.02, -0.5)] * 100
    result = backtest_var(stale_breaches + _clean(250), window=250)
    assert result.exceptions == 0
    assert result.observations == 250


def test_unusable_pairs_do_not_pad_the_window() -> None:
    """Skipped days must not count toward the 250 either."""
    result = backtest_var([(0.0, -0.5)] * 200 + _clean(70))
    assert result.observations == 70
