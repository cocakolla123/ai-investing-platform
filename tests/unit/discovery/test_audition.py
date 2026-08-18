"""Audition harness: candidate loading contract, panel build, permutation
null math, incremental walk-forward value, verdict logic, research log."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from dynamo.backtest.walkforward import WalkForwardConfig
from dynamo.research.discovery.audition import (
    LOG_HEADER,
    CandidateLoadError,
    append_research_log,
    build_audition_panel,
    decide_verdict,
    format_log_row,
    incremental_value_test,
    load_candidate,
    permutation_null_test,
)
from dynamo.research.quant.catalog import FeatureSpec
from dynamo.research.quant.models import LightGBMModel
from tests.unit.discovery.conftest import planted_audition_panel
from tests.unit.quant.conftest import make_bars


def _session_dates(bars: pd.DataFrame) -> list[date]:
    return sorted({cast(date, pd.Timestamp(t).date()) for t in bars["ts"]})


SMALL_WF = WalkForwardConfig(
    train_sessions=60, test_sessions=20, purge_sessions=10, embargo_sessions=5, roll_sessions=20
)


def _fast_model() -> LightGBMModel:
    return LightGBMModel("regression", n_estimators=20, params={"min_child_samples": 5})


# --- candidate-file contract ----------------------------------------------


def test_load_candidate_good_file(candidate_file: Path) -> None:
    spec = load_candidate(candidate_file)
    assert isinstance(spec, FeatureSpec)
    assert spec.name == "ret_5d_test_candidate"
    assert callable(spec.fn)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("x = 1\n", "no module-level CANDIDATE"),
        ("CANDIDATE = {'name': 'x'}\n", "must be a dynamo.research.quant.catalog.FeatureSpec"),
        ("def broken(:\n", "failed to execute"),
        ("raise RuntimeError('boom')\n", "failed to execute"),
        (
            "from dynamo.research.quant.catalog import FeatureSpec\n"
            "CANDIDATE = FeatureSpec(name='ret_20d', group='momentum', "
            "lookback_days=20, fn=lambda ctx: None)\n",
            "collides with the existing feature catalog",
        ),
        (
            "from dynamo.research.quant.catalog import FeatureSpec\n"
            "CANDIDATE = FeatureSpec(name='not an identifier', group='momentum', "
            "lookback_days=20, fn=lambda ctx: None)\n",
            "valid identifier",
        ),
    ],
)
def test_load_candidate_refuses_malformed(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad_candidate.py"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CandidateLoadError, match=match):
        load_candidate(path)


def test_load_candidate_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CandidateLoadError, match="not found"):
        load_candidate(tmp_path / "nope.py")


# --- panel build -----------------------------------------------------------


def test_build_audition_panel_builds_normalized_candidate(candidate_file: Path) -> None:
    from dynamo.research.quant.catalog import FeatureInputs

    bars = make_bars(["AAA", "BBB", "CCC", "DDD", "XLK"], n_sessions=80)
    sector_map = {s: "XLK" for s in ["AAA", "BBB", "CCC", "DDD"]}
    universe = ["AAA", "BBB", "CCC", "DDD"]
    sessions = _session_dates(bars)[40:70]
    candidate = load_candidate(candidate_file)
    calls: list[date] = []

    def inputs_for(day: date) -> FeatureInputs:
        calls.append(day)
        return FeatureInputs(bars=bars, sector_map=sector_map)

    panel = build_audition_panel(
        candidate, inputs_for, bars, sector_map, universe, sessions, horizon_sessions=5
    )
    assert len(panel.sessions) == 25  # trailing horizon excluded
    assert panel.candidate.name == "ret_5d_test_candidate"
    assert panel.base_features.index.equals(panel.candidate.index)
    assert panel.labels.index.equals(panel.candidate.index)
    # Cross-sectional z-score: each session's candidate values sum to ~0.
    per_session_mean = panel.candidate.groupby(level=0).mean().abs()
    assert float(per_session_mean.max()) < 1e-8
    assert not panel.prices.empty
    assert len(calls) == 25


def test_build_audition_panel_cache_hits_still_compute_candidate(
    candidate_file: Path,
) -> None:
    """A base-features cache hit must NOT skip the candidate's point-in-time
    build — candidates are never cached."""
    from dynamo.research.quant.catalog import FeatureInputs
    from dynamo.research.quant.features import build_features

    bars = make_bars(["AAA", "BBB", "CCC", "XLK"], n_sessions=60)
    sector_map = {s: "XLK" for s in ["AAA", "BBB", "CCC"]}
    universe = ["AAA", "BBB", "CCC"]
    sessions = _session_dates(bars)[40:50]
    inputs = FeatureInputs(bars=bars, sector_map=sector_map)

    class WarmCache:
        def __init__(self) -> None:
            self.puts = 0

        def get(self, day: date) -> pd.DataFrame:
            return build_features(inputs, day, universe)

        def put(self, day: date, frame: pd.DataFrame) -> None:
            self.puts += 1

    calls: list[date] = []

    def inputs_for(day: date) -> FeatureInputs:
        calls.append(day)
        return inputs

    cache = WarmCache()
    panel = build_audition_panel(
        load_candidate(candidate_file),
        inputs_for,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=3,
        cache=cache,
    )
    assert len(panel.sessions) == 7
    assert cache.puts == 0  # every base frame served from cache
    assert len(calls) == 7  # candidate still built per session
    assert panel.candidate.notna().any()


def test_candidate_fn_raise_becomes_nan(tmp_path: Path) -> None:
    path = tmp_path / "raiser.py"
    path.write_text(
        "from dynamo.research.quant.catalog import FeatureSpec\n"
        "def _fn(ctx):\n    raise RuntimeError('boom')\n"
        "CANDIDATE = FeatureSpec(name='raiser_candidate', group='momentum', "
        "lookback_days=1, fn=_fn)\n",
        encoding="utf-8",
    )
    from dynamo.research.quant.catalog import FeatureInputs

    bars = make_bars(["AAA", "BBB", "CCC", "XLK"], n_sessions=40)
    sector_map = {s: "XLK" for s in ["AAA", "BBB", "CCC"]}
    sessions = _session_dates(bars)[30:36]
    panel = build_audition_panel(
        load_candidate(path),
        lambda d: FeatureInputs(bars=bars, sector_map=sector_map),
        bars,
        sector_map,
        ["AAA", "BBB", "CCC"],
        sessions,
        horizon_sessions=2,
    )
    assert panel.candidate.isna().all()  # never raises, NaN parity


# --- permutation null ------------------------------------------------------


def test_permutation_null_planted_signal_scores_high() -> None:
    panel = planted_audition_panel(n_sessions=60, signal_strength=2.0)
    res = permutation_null_test(
        panel.candidate, cast(pd.Series, panel.labels["excess_return_20d"]), n_permutations=200
    )
    assert res["real_ic"] is not None and res["real_ic"] > 0.3
    assert res["null_percentile"] >= 99.0
    assert res["n_sessions_used"] == 60
    assert res["n_permutations"] == 200


def test_permutation_null_noise_is_unremarkable() -> None:
    # Noise candidates over several seeds: never extreme, roughly centered.
    percentiles = []
    for seed in (1, 2, 3, 4, 5):
        panel = planted_audition_panel(n_sessions=60, signal_strength=0.0, seed=seed)
        res = permutation_null_test(
            panel.candidate,
            cast(pd.Series, panel.labels["excess_return_20d"]),
            n_permutations=200,
            seed=seed,
        )
        assert abs(res["real_ic"]) < 0.2
        percentiles.append(res["null_percentile"])
    assert min(percentiles) < 95.0  # not systematically significant
    assert 5.0 < float(np.mean(percentiles)) < 95.0  # ~uniform, not degenerate


def test_permutation_null_empty_cross_section() -> None:
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-02", periods=3), ["A", "B"]], names=["ts", "symbol"]
    )
    res = permutation_null_test(
        pd.Series(np.nan, index=idx), pd.Series(1.0, index=idx), n_permutations=10
    )
    assert res["real_ic"] is None
    assert res["null_percentile"] is None
    assert res["n_sessions_used"] == 0


# --- incremental walk-forward value ----------------------------------------


def test_incremental_value_planted_candidate_lifts_pooled_ic() -> None:
    panel = planted_audition_panel(n_sessions=140, signal_strength=2.0)
    res = incremental_value_test(panel, wf_config=SMALL_WF, model_factory=_fast_model)
    assert res["n_folds"] >= 2
    assert len(res["fold_ics_with"]) == len(res["fold_ics_without"])  # same folds
    assert res["ic_with"] > 0.5  # the candidate drives the label
    assert res["ic_delta"] > 0.3  # base features are pure noise


def test_incremental_value_noise_candidate_adds_nothing() -> None:
    panel = planted_audition_panel(n_sessions=140, signal_strength=0.0)
    res = incremental_value_test(panel, wf_config=SMALL_WF, model_factory=_fast_model)
    assert abs(res["ic_delta"]) < 0.15


# --- verdict ---------------------------------------------------------------


def _passing_inputs() -> tuple[dict, dict]:
    incremental = {"ic_delta": 0.03}
    null = {"real_ic": 0.06, "null_percentile": 99.5, "null_ic_std": 0.01}
    return incremental, null


def test_verdict_passes_when_all_bars_clear() -> None:
    incremental, null = _passing_inputs()
    v = decide_verdict(incremental=incremental, null=null, family_trials=3)
    assert v.passed and v.reasons == ()
    assert v.family_trials == 3
    assert v.expected_max_null_ic > 0.0  # >1 trial raises the bar above zero


def test_verdict_fails_on_non_positive_delta() -> None:
    incremental, null = _passing_inputs()
    incremental["ic_delta"] = 0.0
    v = decide_verdict(incremental=incremental, null=null, family_trials=1)
    assert not v.passed and any("adds nothing" in r for r in v.reasons)


def test_verdict_fails_below_null_percentile() -> None:
    incremental, null = _passing_inputs()
    null["null_percentile"] = 90.0
    v = decide_verdict(incremental=incremental, null=null, family_trials=1)
    assert not v.passed and any("noise" in r for r in v.reasons)


def test_verdict_trials_raise_the_expected_max_bar() -> None:
    incremental, null = _passing_inputs()
    few = decide_verdict(incremental=incremental, null=null, family_trials=2)
    many = decide_verdict(incremental=incremental, null=null, family_trials=10_000)
    assert many.expected_max_null_ic > few.expected_max_null_ic
    # A wide null (small universe) with many trials eventually rejects the
    # same IC that passed with few trials.
    wide_null = dict(null, null_ic_std=0.05)
    v = decide_verdict(incremental=incremental, null=wide_null, family_trials=10_000)
    assert not v.passed and any("expected-max-under-null" in r for r in v.reasons)


def test_verdict_missing_stats_fail_closed() -> None:
    v = decide_verdict(
        incremental={"ic_delta": None},
        null={"real_ic": None, "null_percentile": None, "null_ic_std": 0.0},
        family_trials=1,
    )
    assert not v.passed and len(v.reasons) == 3


# --- research log ----------------------------------------------------------


def test_research_log_created_with_header_then_appended(tmp_path: Path) -> None:
    incremental, null = _passing_inputs()
    verdict = decide_verdict(incremental=incremental, null=null, family_trials=2)
    row = format_log_row(
        day=date(2026, 8, 18),
        hypothesis_id="vol-skew",
        family="price-vol",
        family_trials=2,
        verdict=verdict,
    )
    log = tmp_path / "docs" / "alpha_research_log.md"
    append_research_log(log, row)
    text = log.read_text(encoding="utf-8")
    assert text.startswith(LOG_HEADER)
    assert "| 2026-08-18 | vol-skew | price-vol | 2 | +0.0300 | 99.5 | PASS |" in text
    append_research_log(log, row)
    assert log.read_text(encoding="utf-8").count(LOG_HEADER.splitlines()[0]) == 1
    assert log.read_text(encoding="utf-8").count("| 2026-08-18 |") == 2


def test_log_row_fail_verdict_and_nan_formatting() -> None:
    verdict = decide_verdict(
        incremental={"ic_delta": None},
        null={"real_ic": None, "null_percentile": None, "null_ic_std": 0.0},
        family_trials=1,
    )
    row = format_log_row(
        day=date(2026, 8, 18),
        hypothesis_id="h",
        family="f",
        family_trials=1,
        verdict=verdict,
    )
    assert "| n/a | n/a | FAIL |" in row
