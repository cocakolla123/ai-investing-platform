"""Quant training CLI internals (backtest-campaign Phase 2): the synthetic
separable-panel smoke (artifacts + candidate row + sensible metrics, fast),
the degenerate-panel gate, the scaled walk-forward geometry, and the mixed
equity+crypto panel path. NO database — panels and loaders are injected."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from dynamo.backtest.walkforward import WalkForwardConfig
from dynamo.cli import quant_train as qt
from dynamo.research.quant.bundle import bundle_from_version
from dynamo.research.quant.catalog import FeatureInputs
from dynamo.research.quant.scoring import ModelBundle, score_universe
from tests.unit.quant.conftest import make_bars

TRAIN_START = date(2024, 1, 2)
TRAIN_END = date(2025, 3, 1)


def separable_panel(
    n_symbols: int = 30, n_sessions: int = 300, seed: int = 11
) -> qt.TrainingPanel:
    """~30 symbols x 300 sessions with one strongly predictive feature across
    all three labels — the model should crush the featureless baseline."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2024-01-02", periods=n_sessions)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([sessions, symbols], names=["ts", "symbol"])
    signal = rng.normal(size=len(idx))
    noise = rng.normal(size=len(idx))
    features = pd.DataFrame({"signal": signal, "noise": noise}, index=idx)
    excess = 0.05 * signal + 0.005 * rng.normal(size=len(idx))
    labels = pd.DataFrame(
        {
            "excess_return_20d": excess,
            "downside_20d": (excess < np.quantile(excess, 0.2)).astype(float),
            "realized_vol_20d": np.exp(0.5 * signal) * 0.2,
            # Parallel residual label (features.build_labels emits it): keep
            # it NOISELESS so a residual-kind run is sharply distinguishable.
            "residual_return_20d": 0.05 * signal,
        },
        index=idx,
    )
    return qt.TrainingPanel(
        features=features,
        labels=labels,
        sessions=tuple(cast(pd.Timestamp, t).date() for t in sessions),
    )


@pytest.fixture(scope="module")
def smoke(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    artifact_dir = tmp_path_factory.mktemp("quant-train")
    panel = separable_panel()
    report = qt.train_quant_bundle(
        panel,
        version="smoke-v1",
        artifact_dir=artifact_dir,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
    )
    return {"dir": artifact_dir, "panel": panel, "report": report}


def test_smoke_produces_artifacts_and_sensible_metrics(smoke: dict[str, Any]) -> None:
    report: qt.TrainReport = smoke["report"]
    assert report.bundle_path.is_file()
    for task in ("ranker", "downside", "vol"):
        assert (smoke["dir"] / f"smoke-v1-{task}.joblib").is_file()
    m = report.registry_metrics
    assert m["n_folds"] >= 2
    assert m["rank_ic_pooled"] is not None and m["rank_ic_pooled"] > 0.5
    assert m["rank_ic_t_stat"] is not None and m["rank_ic_t_stat"] > 0.0
    assert m["rank_win_share"] == 1.0
    assert m["downside_brier_pooled"] < m["downside_base_rate_brier_pooled"]
    assert m["downside_auc_pooled"] > 0.7
    assert m["vol_rmse_log_pooled"] is not None
    assert m["vol_qlike_pooled"] is not None
    assert m["fold_ic_dispersion"] >= 0.0
    assert m["n_samples"] == len(smoke["panel"].features)


def test_smoke_bundle_loads_and_scores(smoke: dict[str, Any]) -> None:
    report: qt.TrainReport = smoke["report"]
    panel: qt.TrainingPanel = smoke["panel"]
    bundle = bundle_from_version(
        "smoke-v1", smoke["dir"], current_feature_names=report.feature_names
    )
    last_ts = panel.features.index.get_level_values(0)[-1]
    day = panel.features.xs(last_ts, level="ts")
    scores = score_universe(bundle, day, panel.sessions[-1])
    assert len(scores) == len(day)
    assert all(s.expected_excess_return is not None for s in scores)
    assert all(s.model_version == "smoke-v1" for s in scores)


def test_smoke_registers_one_quant_bundle_candidate(smoke: dict[str, Any]) -> None:
    report: qt.TrainReport = smoke["report"]
    mv = qt.registry_model_version(report, TRAIN_START, TRAIN_END)
    assert mv.id == "smoke-v1"
    assert mv.model_type == "quant_bundle"
    assert mv.algo == "lightgbm"
    assert mv.train_start == TRAIN_START and mv.train_end == TRAIN_END
    assert mv.artifact_path == str(report.bundle_path)
    assert mv.metrics["rank_ic_pooled"] == report.registry_metrics["rank_ic_pooled"]
    assert mv.features_hash  # the ordered feature-name contract travels along


def test_go_no_go_table_prints_folds_gate_and_dispersion(
    smoke: dict[str, Any],
) -> None:
    lines: list[str] = []
    qt.print_go_no_go(smoke["report"], out=lines.append)
    text = "\n".join(lines)
    assert "GO/NO-GO" in text
    assert "pooled rank IC" in text
    assert "t-stat" in text and "ship gate >= 2.00" in text
    assert "dispersion" in text
    assert "VERDICT:" in text


# -- gates -------------------------------------------------------------------


def test_gate_refuses_too_few_sessions(tmp_path: Path) -> None:
    panel = separable_panel(n_symbols=30, n_sessions=40)
    with pytest.raises(qt.PanelTooSmallError, match="feature sessions"):
        qt.train_quant_bundle(
            panel,
            version="tiny",
            artifact_dir=tmp_path,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
        )


def test_gate_refuses_too_few_symbols(tmp_path: Path) -> None:
    panel = separable_panel(n_symbols=5, n_sessions=120)
    with pytest.raises(qt.PanelTooSmallError, match="usable symbols"):
        qt.train_quant_bundle(
            panel,
            version="tiny",
            artifact_dir=tmp_path,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
        )


def test_gate_maps_to_exit_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() turns PanelTooSmallError into exit 2 (the refuse contract)."""
    calls: dict[str, Any] = {}

    class FakeRegistry:
        def status_of(self, target_id: str) -> None:
            return None

        def insert_candidate(self, mv: Any) -> None:  # pragma: no cover
            calls["registered"] = mv

    class FakeGateway:
        def __init__(self, engine: Any, as_of: date) -> None:
            self.as_of = as_of

        def universe(self, tiers: Any = None) -> list[str]:
            return ["AAA", "BBB"]

        def sectors(self, symbols: Any = None) -> dict[str, str]:
            return {}

        def bars(self, symbols: Any, lookback_days: int) -> pd.DataFrame:
            return pd.DataFrame()

        def feature_inputs(self, symbols: Any) -> FeatureInputs:
            return FeatureInputs()

    monkeypatch.setattr("dynamo.persistence.db.make_engine", lambda url: object())
    monkeypatch.setattr(
        "dynamo.persistence.repositories.learning.ModelRegistryRepository",
        lambda engine: FakeRegistry(),
    )
    monkeypatch.setattr("dynamo.replay.gateway.HistoricalDataGateway", FakeGateway)
    rc = qt.main(
        [
            "--train-start", "2026-01-05",
            "--train-end", "2026-02-27",
            "--label", "gate-test",
            "--database-url", "postgresql://unused/unused",
            "--no-cache",  # unit test: never write the real artifacts/ cache
        ]
    )
    assert rc == 2
    assert "registered" not in calls


def test_existing_registry_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class TakenRegistry:
        def status_of(self, target_id: str) -> str:
            return "candidate"

    monkeypatch.setattr("dynamo.persistence.db.make_engine", lambda url: object())
    monkeypatch.setattr(
        "dynamo.persistence.repositories.learning.ModelRegistryRepository",
        lambda engine: TakenRegistry(),
    )
    rc = qt.main(
        [
            "--train-start", "2026-01-05",
            "--train-end", "2026-02-27",
            "--label", "campaign-v1",
            "--database-url", "postgresql://unused/unused",
        ]
    )
    assert rc == 1


# -- fold geometry -----------------------------------------------------------


def test_walkforward_uses_doc08_defaults_when_window_is_long() -> None:
    assert qt.walkforward_for(1000) == WalkForwardConfig()


def test_walkforward_scales_down_honestly_for_short_windows() -> None:
    from dynamo.backtest.walkforward import generate_folds

    for n in (60, 120, 280, 480):
        cfg = qt.walkforward_for(n)
        assert cfg.purge_sessions == 20  # never below the label horizon
        folds = generate_folds(pd.bdate_range("2024-01-02", periods=n), cfg)
        assert folds, f"{n} sessions must yield at least one fold"
        # Purge discipline: >= 20 sessions between train end and test start.
        for fold in folds:
            gap = list(
                pd.bdate_range(
                    fold.train_sessions[-1], fold.test_sessions[0]
                )
            )
            assert len(gap) - 2 >= 20


def test_walkforward_refuses_impossible_windows() -> None:
    with pytest.raises(qt.PanelTooSmallError):
        qt.walkforward_for(25)


# -- panel assembly (real feature/label builders, small frames) --------------


def test_build_panel_mixed_equity_crypto_no_crash_and_nan_coverage() -> None:
    """Crypto rides along with NaN fundamentals (ADR-0013): the panel builds
    without raising, crypto keeps bars-driven features + SPY-benchmark labels,
    and scoring reflects the holes in feature_coverage."""
    universe = ["AAA", "BBB", "BTC/USD"]
    bars = make_bars([*universe, "XLK", "SPY"], n_sessions=90)
    inputs = FeatureInputs(bars=bars, sector_map={"AAA": "XLK", "BBB": "XLK"})
    sector_map = {"AAA": "XLK", "BBB": "XLK", "BTC/USD": "SPY"}
    sessions = [cast(pd.Timestamp, t).date() for t in pd.bdate_range("2024-01-02", periods=90)]

    panel = qt.build_panel(
        lambda d: inputs, bars, sector_map, universe, sessions, horizon_sessions=20
    )
    assert len(panel.sessions) == 70  # trailing horizon excluded
    assert set(panel.features.index.get_level_values("symbol")) == set(universe)

    crypto_feats = panel.features.xs("BTC/USD", level="symbol")
    assert crypto_feats["earnings_yield"].isna().all()  # fundamentals NaN by design
    assert crypto_feats["ret_20d"].notna().any()  # bars-driven features live
    crypto_labels = panel.labels.xs("BTC/USD", level="symbol")
    assert crypto_labels["excess_return_20d"].notna().any()  # SPY benchmark
    assert crypto_labels["realized_vol_20d"].notna().any()

    # Scoring a mixed day reflects the NaN groups in coverage, never crashes.
    last_ts = panel.features.index.get_level_values(0)[-1]
    day = panel.features.xs(last_ts, level="ts")
    scores = {
        s.symbol: s
        for s in score_universe(
            ModelBundle(model_version="baseline-v0"), day, panel.sessions[-1]
        )
    }
    assert 0.0 < scores["BTC/USD"].feature_coverage < 1.0
    assert scores["BTC/USD"].feature_coverage < scores["AAA"].feature_coverage

    # 3 symbols is below the symbol gate — the trainer refuses this panel.
    with pytest.raises(qt.PanelTooSmallError, match="usable symbols"):
        qt.check_panel(panel)


def test_label_sector_map_falls_back_to_spy_for_crypto() -> None:
    sectors = {"AAA": "Information Technology", "BTC/USD": "crypto"}
    mapped = qt.label_sector_map(sectors, ["AAA", "BTC/USD", "ZZZ"])
    assert mapped == {"AAA": "XLK", "BTC/USD": "SPY", "ZZZ": "SPY"}


def test_build_panel_empty_sessions_yields_empty_panel() -> None:
    panel = qt.build_panel(
        lambda d: FeatureInputs(),
        pd.DataFrame(),
        {},
        ["AAA"],
        [date(2026, 1, 5)],  # horizon >= sessions -> nothing usable
        horizon_sessions=20,
    )
    assert panel.sessions == ()
    with pytest.raises(qt.PanelTooSmallError, match="empty"):
        qt.check_panel(panel)


# -- retune prep: tuning flags threaded into models / feature subset ---------


def test_tuning_flags_parse_and_default_off() -> None:
    base = ["--train-start", "2024-01-02", "--train-end", "2025-01-02", "--label", "x"]
    args = qt._build_parser().parse_args(base)
    assert args.lgbm_max_depth is None
    assert args.lgbm_min_data_in_leaf is None
    assert args.lgbm_lambda_l2 is None
    assert args.feature_groups is None
    tuned = qt._build_parser().parse_args(
        [
            *base,
            "--lgbm-max-depth", "4",
            "--lgbm-min-data-in-leaf", "40",
            "--lgbm-lambda-l2", "0.5",
            "--feature-groups", "momentum", "volatility",
        ]
    )
    assert tuned.lgbm_max_depth == 4
    assert tuned.lgbm_min_data_in_leaf == 40
    assert tuned.lgbm_lambda_l2 == 0.5
    assert tuned.feature_groups == ["momentum", "volatility"]


def test_lgbm_params_from_flags_maps_to_canonical_names() -> None:
    assert qt.lgbm_params_from_flags(None, None, None) == {}
    assert qt.lgbm_params_from_flags(4, 40, 0.5) == {
        "max_depth": 4,
        "min_child_samples": 40,
        "reg_lambda": 0.5,
    }
    assert qt.lgbm_params_from_flags(4, None, None) == {"max_depth": 4}


def test_select_feature_groups_filters_by_catalog_group() -> None:
    from dynamo.research.quant.catalog import FEATURE_CATALOG

    frame = pd.DataFrame(
        np.zeros((2, len(FEATURE_CATALOG))),
        index=pd.Index(["AAA", "BBB"], name="symbol"),
        columns=pd.Index(list(FEATURE_CATALOG)),
    )
    momentum = qt.select_feature_groups(frame, ["momentum"])
    assert len(momentum.columns) > 0
    assert all(FEATURE_CATALOG[c].group == "momentum" for c in momentum.columns)
    both = qt.select_feature_groups(frame, ["momentum", "volatility"])
    assert set(momentum.columns) < set(both.columns)


def test_select_feature_groups_refuses_typos_and_empty_selection() -> None:
    frame = pd.DataFrame({"ret_20d": [0.0]}, index=pd.Index(["AAA"], name="symbol"))
    with pytest.raises(ValueError, match="unknown feature group"):
        qt.select_feature_groups(frame, ["momentumm"])
    with pytest.raises(ValueError, match="no columns"):
        qt.select_feature_groups(frame, ["macro"])  # valid group, absent here


def test_lgbm_params_reach_every_task_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retune knobs must reach all three task factories AND the config
    hash / registry hyperparams, so a tuned run is distinguishable."""
    from dynamo.research.quant import models as models_mod

    real = models_mod.LightGBMModel
    captured: list[dict[str, Any]] = []
    original_init = real.__init__

    def recording_init(self: Any, objective: Any = "regression", **kwargs: Any) -> None:
        captured.append(dict(kwargs.get("params") or {}))
        original_init(self, objective, **kwargs)

    # Patch __init__ (not the class): saved models must stay picklable.
    monkeypatch.setattr(real, "__init__", recording_init)
    knobs = {"max_depth": 3, "min_child_samples": 30, "reg_lambda": 1.0}
    panel = separable_panel(n_symbols=25, n_sessions=150)
    report = qt.train_quant_bundle(
        panel,
        version="tuned-v1",
        artifact_dir=tmp_path,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        lgbm_params=knobs,
    )
    assert captured and all(p == knobs for p in captured)
    assert report.lgbm_params == knobs
    baseline = qt.train_quant_bundle(
        panel,
        version="plain-v1",
        artifact_dir=tmp_path,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
    )
    assert report.config_hash != baseline.config_hash  # knobs move the hash
    mv = qt.registry_model_version(report, TRAIN_START, TRAIN_END)
    assert mv.hyperparams["lgbm_params"] == knobs
    mv_plain = qt.registry_model_version(baseline, TRAIN_START, TRAIN_END)
    assert "lgbm_params" not in mv_plain.hyperparams  # default runs unchanged


def test_feature_groups_subset_the_trained_bundle(tmp_path: Path) -> None:
    """Group selection drops the other groups' columns before the gate and
    records the subset in the report + registry hyperparams."""
    from dynamo.research.quant.catalog import FEATURE_CATALOG

    panel = separable_panel(n_symbols=25, n_sessions=150)
    # Rename the two synthetic features to real catalog names from two groups.
    momentum_col = next(n for n, s in FEATURE_CATALOG.items() if s.group == "momentum")
    vol_col = next(n for n, s in FEATURE_CATALOG.items() if s.group == "volatility")
    renamed = {"signal": momentum_col, "noise": vol_col}
    panel = qt.TrainingPanel(
        features=panel.features.rename(columns=renamed),
        labels=panel.labels,
        sessions=panel.sessions,
    )
    report = qt.train_quant_bundle(
        panel,
        version="momentum-only-v1",
        artifact_dir=tmp_path,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        feature_groups=["momentum"],
    )
    assert report.feature_names == (momentum_col,)  # volatility column dropped
    assert report.feature_groups == ("momentum",)
    mv = qt.registry_model_version(report, TRAIN_START, TRAIN_END)
    assert mv.hyperparams["feature_groups"] == ["momentum"]


# -- factor-residualized rank labels (--label-kind) ---------------------------


def test_label_kind_flag_parses_and_defaults_to_sector_excess() -> None:
    base = ["--train-start", "2024-01-02", "--train-end", "2025-01-02", "--label", "x"]
    assert qt._build_parser().parse_args(base).label_kind == "sector_excess"
    args = qt._build_parser().parse_args([*base, "--label-kind", "residual"])
    assert args.label_kind == "residual"
    with pytest.raises(SystemExit):
        qt._build_parser().parse_args([*base, "--label-kind", "bogus"])


def test_residual_label_kind_trains_rank_on_residual_column(tmp_path: Path) -> None:
    """--label-kind residual: the rank task targets residual_return_20d, the
    config hash moves, and the registry hyperparams record the kind; the
    default run stays bit-compatible (no label_kind key, unchanged target)."""
    panel = separable_panel(n_symbols=25, n_sessions=150)
    residual = qt.train_quant_bundle(
        panel,
        version="residual-v1",
        artifact_dir=tmp_path,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        label_kind="residual",
    )
    default = qt.train_quant_bundle(
        panel,
        version="default-v1",
        artifact_dir=tmp_path,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
    )
    assert residual.task_reports["rank"]["target"] == "residual_return_20d"
    assert default.task_reports["rank"]["target"] == "excess_return_20d"
    assert residual.config_hash != default.config_hash
    # The noiseless residual label is easier to rank than the noisy excess one.
    assert residual.registry_metrics["rank_ic_pooled"] > 0.9
    mv = qt.registry_model_version(residual, TRAIN_START, TRAIN_END)
    assert mv.hyperparams["label_kind"] == "residual"
    mv_default = qt.registry_model_version(default, TRAIN_START, TRAIN_END)
    assert "label_kind" not in mv_default.hyperparams


def test_train_quant_bundle_refuses_unknown_label_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown label kind"):
        qt.train_quant_bundle(
            separable_panel(n_symbols=25, n_sessions=150),
            version="bad-kind",
            artifact_dir=tmp_path,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            label_kind="bogus",
        )


def test_main_refuses_unknown_feature_group_before_any_db_work() -> None:
    rc = qt.main(
        [
            "--train-start", "2026-01-05",
            "--train-end", "2026-02-27",
            "--label", "typo-test",
            "--feature-groups", "momentumm",
            "--no-cache",
        ]
    )
    assert rc == 1  # refused before touching engine/registry/gateway


# -- progress observability (the invisible-2h-run fix) -----------------------


def _panel_fixtures() -> tuple[Any, Any, dict[str, str], list[str], list[date]]:
    universe = ["AAA", "BBB"]
    bars = make_bars([*universe, "XLK", "SPY"], n_sessions=60)
    inputs = FeatureInputs(bars=bars, sector_map={"AAA": "XLK"})
    sector_map = {"AAA": "XLK", "BBB": "SPY"}
    sessions = [
        cast(pd.Timestamp, t).date() for t in pd.bdate_range("2024-01-02", periods=60)
    ]
    return inputs, bars, sector_map, universe, sessions


def test_progress_file_is_written_and_parseable(tmp_path: Path) -> None:
    import json

    inputs, bars, sector_map, universe, sessions = _panel_fixtures()
    progress_path = tmp_path / "progress.jsonl"
    panel = qt.build_panel(
        lambda d: inputs, bars, sector_map, universe, sessions,
        horizon_sessions=20,
        on_session=qt.progress_file_writer(progress_path),
    )
    lines = progress_path.read_text().splitlines()
    assert len(lines) == len(panel.sessions)
    records = [json.loads(line) for line in lines]  # every line parses
    assert [r["session_i"] for r in records] == list(range(1, len(lines) + 1))
    assert {r["n_sessions"] for r in records} == {len(panel.sessions)}
    elapsed = [r["elapsed_s"] for r in records]
    assert all(isinstance(e, float | int) and e >= 0 for e in elapsed)
    assert elapsed == sorted(elapsed)  # monotonically non-decreasing


def test_progress_lines_carry_elapsed_and_eta() -> None:
    inputs, bars, sector_map, universe, sessions = _panel_fixtures()
    lines: list[str] = []
    qt.build_panel(
        lambda d: inputs, bars, sector_map, universe, sessions,
        horizon_sessions=20, progress=lines.append,
    )
    assert lines, "40 feature sessions must produce a 25-session progress line"
    assert all("elapsed " in line and "ETA " in line for line in lines)


def test_hms_formatting_is_clamped_and_exact() -> None:
    assert qt._hms(3661.9) == "1:01:01"
    assert qt._hms(59.0) == "0:00:59"
    assert qt._hms(0.0) == "0:00:00"
    assert qt._hms(-5.0) == "0:00:00"
    assert qt._hms(float("nan")) == "0:00:00"


def test_progress_file_flag_parses() -> None:
    base = ["--train-start", "2024-01-02", "--train-end", "2025-01-02", "--label", "x"]
    assert qt._build_parser().parse_args(base).progress_file is None
    args = qt._build_parser().parse_args([*base, "--progress-file", "/tmp/p.jsonl"])
    assert args.progress_file == Path("/tmp/p.jsonl")


def test_registry_metrics_are_json_safe(smoke: dict[str, Any]) -> None:
    import json

    payload = json.dumps(smoke["report"].registry_metrics)
    assert "NaN" not in payload  # NaN flattened to null for the registry row
    assert not any(
        isinstance(v, float) and math.isnan(v)
        for v in smoke["report"].registry_metrics.values()
        if isinstance(v, float)
    )
