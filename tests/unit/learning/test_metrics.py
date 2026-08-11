"""Learning-metrics suite tests (plan Phase 6): hand-computed Brier/ECE/
Sharpe, empty-input totality, cut_by determinism, and the metrics_daily row
builder. Pure — no database, no network, no LLM."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.learning.metrics import (
    abstention_rate,
    benchmark_excess,
    brier_score,
    calibration_summary,
    cut_by,
    daily_pnl_series,
    ev_per_decision,
    expected_calibration_error,
    failure_class_frequency,
    latest_value_of_learning,
    learning_metric_rows,
    max_drawdown_pct,
    performance_by_quant_alignment,
    risk_summary,
    scored_pairs,
    sharpe_ratio,
    sortino_ratio,
    success_rate_by_conviction_bucket,
)


def _exp(
    i: int,
    *,
    conviction: float | None = 0.6,
    label: str = "win",
    pnl: float | None = 1.0,
    as_of: date = date(2026, 8, 3),
    exit_day: date | None = None,
    resolved: bool = True,
    target_achieved: bool | None = None,
    quant_alignment: str | None = "agree",
    regime: str | None = "neutral",
    sector: str | None = "tech",
    sleeve: str | None = "long_term",
    benchmark_excess_pct: float | None = None,
    failure_primary: str | None = None,
) -> DecisionExperience:
    exit_at = None
    if exit_day is not None:
        exit_at = datetime(exit_day.year, exit_day.month, exit_day.day, 20, tzinfo=UTC)
    return DecisionExperience.model_validate(
        {
            "id": f"exp:test:{i}",
            "created_at": datetime(2026, 8, 3, 12, tzinfo=UTC),
            "symbol": f"SYM{i}",
            "as_of": as_of,
            "sleeve": sleeve,
            "regime": regime,
            "sector": sector,
            "conviction": conviction,
            "quant_alignment": quant_alignment,
            "resolved": resolved,
            "pnl_pct": pnl,
            "benchmark_excess_pct": benchmark_excess_pct,
            "target_achieved": target_achieved,
            "exit_at": exit_at,
            "failure_primary": failure_primary,
            "label": label,
        }
    )


# --- Brier + ECE (hand-computed) -------------------------------------------


def test_brier_hand_computed() -> None:
    rows = [_exp(1, conviction=0.8, label="win"), _exp(2, conviction=0.4, label="loss")]
    # ((0.8-1)^2 + (0.4-0)^2) / 2 = (0.04 + 0.16) / 2
    assert brier_score(rows) == pytest.approx(0.1)


def test_ece_hand_computed() -> None:
    rows = [_exp(1, conviction=0.85, label="win"), _exp(2, conviction=0.45, label="loss")]
    # bucket 0.8-0.9: |0.85 - 1| = 0.15; bucket 0.4-0.5: |0.45 - 0| = 0.45
    # each holds half the mass -> ECE = 0.5*0.15 + 0.5*0.45 = 0.30
    assert expected_calibration_error(rows) == pytest.approx(0.30)


def test_scored_pairs_prefer_target_achieved_over_label() -> None:
    # target_achieved=False overrides the win label (path-based outcome first).
    rows = [_exp(1, conviction=0.9, label="win", target_achieved=False)]
    assert scored_pairs(rows) == [(0.9, False)]


def test_scored_pairs_skip_unresolved_and_unscored() -> None:
    rows = [
        _exp(1, resolved=False),
        _exp(2, conviction=None),
        _exp(3, conviction=0.7, label="win"),
    ]
    assert scored_pairs(rows) == [(0.7, True)]


def test_success_rate_buckets_omit_empty_and_count_hits() -> None:
    rows = [
        _exp(1, conviction=0.65, label="win"),
        _exp(2, conviction=0.68, label="loss"),
        _exp(3, conviction=0.95, label="win"),
    ]
    buckets = success_rate_by_conviction_bucket(rows)
    assert set(buckets) == {"0.6-0.7", "0.9-1.0"}
    assert buckets["0.6-0.7"].n == 2
    assert buckets["0.6-0.7"].success_rate == pytest.approx(0.5)
    assert buckets["0.6-0.7"].mean_conviction == pytest.approx(0.665)
    assert buckets["0.9-1.0"].success_rate == pytest.approx(1.0)


# --- pnl series + risk stats (documented assumptions) ------------------------


def test_daily_pnl_series_sums_per_exit_day_and_zero_fills_weekdays() -> None:
    rows = [
        _exp(1, pnl=2.0, exit_day=date(2026, 8, 3)),  # Monday
        _exp(2, pnl=-1.0, exit_day=date(2026, 8, 3)),
        _exp(3, pnl=3.0, exit_day=date(2026, 8, 5)),  # Wednesday
    ]
    series = daily_pnl_series(rows)
    assert series == [
        (date(2026, 8, 3), 1.0),
        (date(2026, 8, 4), 0.0),  # Tuesday zero-filled
        (date(2026, 8, 5), 3.0),
    ]


def test_sharpe_hand_computed() -> None:
    # mean=2, sample std=1 -> 2 * sqrt(252)
    assert sharpe_ratio([1.0, 2.0, 3.0]) == pytest.approx(2.0 * math.sqrt(252))


def test_sortino_hand_computed() -> None:
    # mean=0.5; downside dev = sqrt(((-1)^2 + (-1)^2)/4) = sqrt(0.5)
    expected = 0.5 / math.sqrt(0.5) * math.sqrt(252)
    assert sortino_ratio([2.0, -1.0, 2.0, -1.0]) == pytest.approx(expected)


def test_sortino_none_without_downside() -> None:
    assert sortino_ratio([1.0, 2.0]) is None


def test_max_drawdown_hand_computed() -> None:
    # cumulative: 1, -1, 0 -> peak 1, trough -1 -> drawdown 2
    assert max_drawdown_pct([1.0, -2.0, 1.0]) == pytest.approx(2.0)
    assert max_drawdown_pct([1.0, 2.0]) == pytest.approx(0.0)


def test_risk_summary_totality_on_flat_series() -> None:
    rows = [_exp(1, pnl=1.0, exit_day=date(2026, 8, 3))]
    summary = risk_summary(rows)
    assert summary.n_days == 1
    assert summary.sharpe is None  # a single day has no variance
    assert summary.sortino is None
    assert summary.max_drawdown_pct == pytest.approx(0.0)


# --- distributions, EV, abstention -------------------------------------------


def test_benchmark_excess_mean_and_median() -> None:
    rows = [
        _exp(1, benchmark_excess_pct=1.0),
        _exp(2, benchmark_excess_pct=2.0),
        _exp(3, benchmark_excess_pct=6.0),
    ]
    summary = benchmark_excess(rows)
    assert (summary.n, summary.mean, summary.median) == (3, pytest.approx(3.0), 2.0)


def test_ev_per_decision_is_mean_pnl() -> None:
    rows = [_exp(1, pnl=2.0), _exp(2, pnl=-1.0)]
    assert ev_per_decision(rows) == pytest.approx(0.5)


def test_abstention_rate_from_verdict_counts() -> None:
    assert abstention_rate({"buy": 3, "no_action": 1}) == pytest.approx(0.25)
    assert abstention_rate({}) is None


# --- cuts ---------------------------------------------------------------------


def test_cut_by_deterministic_sorted_keys_and_stats() -> None:
    rows = [
        _exp(1, quant_alignment="agree", label="win", pnl=2.0),
        _exp(2, quant_alignment="agree", label="loss", pnl=-1.0),
        _exp(3, quant_alignment="conflict", label="loss", pnl=-3.0),
        _exp(4, quant_alignment=None, label="win", pnl=1.0),  # keyless: skipped
    ]
    cuts = performance_by_quant_alignment(rows)
    assert list(cuts) == ["agree", "conflict"]  # sorted, deterministic
    assert cuts["agree"].n == 2
    assert cuts["agree"].win_rate == pytest.approx(0.5)
    assert cuts["agree"].mean_pnl_pct == pytest.approx(0.5)
    assert cuts["conflict"].win_rate == pytest.approx(0.0)
    # identical input (shuffled) -> identical output
    assert cut_by(list(reversed(rows)), lambda e: e.quant_alignment) == cuts


def test_failure_class_frequency_counts_primary() -> None:
    rows = [
        _exp(1, label="loss", failure_primary="claude_reasoning"),
        _exp(2, label="loss", failure_primary="claude_reasoning"),
        _exp(3, label="loss", failure_primary="data_quality"),
        _exp(4, label="win"),
    ]
    assert failure_class_frequency(rows) == {"claude_reasoning": 2, "data_quality": 1}


# --- value-of-learning passthrough -------------------------------------------


def test_latest_value_of_learning_picks_highest_generation() -> None:
    rows = [
        {"generation": 1, "value_of_learning": 0.01, "n_test": 30},
        {"generation": 3, "value_of_learning": -0.02, "n_test": 40},
        {"generation": 2, "value_of_learning": 0.05, "n_test": 35},
        {"generation": 4, "value_of_learning": None, "n_test": 10},  # skipped fold
    ]
    assert latest_value_of_learning(rows) == (pytest.approx(-0.02), 40)
    assert latest_value_of_learning([]) is None


# --- empty-input totality -----------------------------------------------------


def test_empty_inputs_are_total() -> None:
    assert brier_score([]) is None
    assert expected_calibration_error([]) is None
    assert calibration_summary([]).n == 0
    assert success_rate_by_conviction_bucket([]) == {}
    assert benchmark_excess([]).n == 0
    assert ev_per_decision([]) is None
    assert daily_pnl_series([]) == []
    assert sharpe_ratio([]) is None
    assert sortino_ratio([]) is None
    assert max_drawdown_pct([]) is None
    assert risk_summary([]).n_days == 0
    assert cut_by([], lambda e: e.sector) == {}
    assert failure_class_frequency([]) == {}
    assert learning_metric_rows([], {}, []) == [
        ("learning.n_resolved", "platform", "all", 0.0),
        ("learning.n_scored", "platform", "all", 0.0),
    ]


# --- metrics_daily row builder ------------------------------------------------


def test_learning_metric_rows_namespaced_and_scoped() -> None:
    rows = [
        _exp(1, conviction=0.8, label="win", pnl=2.0, exit_day=date(2026, 8, 3),
             benchmark_excess_pct=1.0),
        _exp(2, conviction=0.4, label="loss", pnl=-1.0, exit_day=date(2026, 8, 4),
             failure_primary="quant_model"),
    ]
    generation_rows = [{"generation": 1, "value_of_learning": 0.02, "n_test": 31}]
    metric_rows = learning_metric_rows(rows, {"buy": 8, "no_action": 2}, generation_rows)
    assert all(metric.startswith("learning.") for metric, _, _, _ in metric_rows)
    by_key = {(m, st, sid): v for m, st, sid, v in metric_rows}
    assert by_key[("learning.brier", "platform", "all")] == pytest.approx(0.1)
    assert by_key[("learning.success_rate", "platform", "all")] == pytest.approx(0.5)
    assert by_key[("learning.ev_per_decision", "platform", "all")] == pytest.approx(0.5)
    assert by_key[("learning.abstention_rate", "platform", "all")] == pytest.approx(0.2)
    assert by_key[("learning.value_of_learning", "platform", "all")] == pytest.approx(0.02)
    assert by_key[("learning.win_rate", "quant_alignment", "agree")] == pytest.approx(0.5)
    assert by_key[("learning.win_rate", "regime", "neutral")] == pytest.approx(0.5)
    assert by_key[("learning.failure_count", "failure_class", "quant_model")] == 1.0
    assert by_key[("learning.bucket_n", "conviction_bucket", "0.8-0.9")] == 1.0
    # deterministic: same inputs, identical rows in identical order
    assert metric_rows == learning_metric_rows(rows, {"buy": 8, "no_action": 2},
                                               generation_rows)
