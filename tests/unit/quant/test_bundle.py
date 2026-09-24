"""Quant bundle artifacts (backtest-campaign Phase 2): save -> load -> score
bit-identical, and every failure mode LOUD — unknown version, sha mismatch,
feature-name drift vs the current catalog. Filesystem only, no database."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from dynamo.learning.artifacts import ArtifactIntegrityError
from dynamo.research.quant.bundle import (
    QuantBundleError,
    bundle_from_version,
    feature_names_hash,
    read_bundle_manifest,
    save_bundle,
)
from dynamo.research.quant.catalog import FEATURE_CATALOG
from dynamo.research.quant.models import LightGBMModel
from dynamo.research.quant.scoring import score_universe
from dynamo.research.quant.training import feature_quantiles, fit_final_model

VERSION = "test-bundle-v1"
N_SESSIONS = 40
N_SYMBOLS = 12
TRAIN_START = date(2024, 1, 2)
TRAIN_END = date(2024, 2, 27)


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tiny (ts, symbol) panel over the FULL feature catalog (drift check is
    against catalog names by default) with a planted signal column."""
    rng = np.random.default_rng(5)
    sessions = pd.bdate_range("2024-01-02", periods=N_SESSIONS)
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    idx = pd.MultiIndex.from_product([sessions, symbols], names=["ts", "symbol"])
    names = list(FEATURE_CATALOG)
    features = pd.DataFrame(
        rng.normal(size=(len(idx), len(names))), index=idx, columns=pd.Index(names)
    )
    signal = features[names[0]]
    labels = pd.DataFrame(
        {
            "excess_return_20d": 0.05 * signal + 0.002 * rng.normal(size=len(idx)),
            "downside_20d": (signal < signal.quantile(0.25)).astype(float),
            "realized_vol_20d": np.exp(0.3 * signal) * 0.2,
        },
        index=idx,
    )
    return features, labels


def _factory(objective: str):
    return lambda: LightGBMModel(
        objective,  # type: ignore[arg-type]
        n_estimators=15,
        params={"min_child_samples": 5},
    )


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("bundles")
    features, labels = _panel()
    ranker = fit_final_model(features, labels, task="rank", model_factory=_factory("regression"))
    downside = fit_final_model(features, labels, task="downside", model_factory=_factory("binary"))
    vol = fit_final_model(features, labels, task="vol", model_factory=_factory("regression"))
    path = save_bundle(
        directory,
        VERSION,
        ranker=ranker,
        downside=downside,
        vol=vol,
        feature_names=list(features.columns),
        train_feature_quantiles=feature_quantiles(features),
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        n_samples=len(features),
        config_hash="cfg-abc",
        metrics={"rank_ic_pooled": 0.9},
        fold_ic_dispersion=0.05,
    )
    return {
        "dir": directory,
        "path": path,
        "features": features,
        "ranker": ranker,
        "downside": downside,
        "vol": vol,
    }


def test_save_writes_bundle_manifest_and_task_artifacts(saved: dict) -> None:
    directory: Path = saved["dir"]
    manifest = read_bundle_manifest(directory, VERSION)
    assert manifest["bundle_version"] == VERSION
    assert manifest["train_start"] == TRAIN_START.isoformat()
    assert manifest["config_hash"] == "cfg-abc"
    assert manifest["metrics"]["rank_ic_pooled"] == 0.9
    assert manifest["feature_names"] == list(FEATURE_CATALOG)
    assert manifest["features_hash"] == feature_names_hash(list(FEATURE_CATALOG))
    for task in ("ranker", "downside", "vol"):
        spec = manifest["tasks"][task]
        assert (directory / spec["artifact_file"]).is_file()
        assert (directory / f"{spec['model_id']}.manifest.json").is_file()


def test_save_load_score_bit_identical(saved: dict) -> None:
    """The loaded bundle scores the fixture features EXACTLY like the
    in-memory models it was saved from — same predictions, ranks, coverage."""
    from typing import Any, cast

    from dynamo.research.quant.scoring import ModelBundle

    features: pd.DataFrame = saved["features"]
    last_ts = cast(Any, features.index.get_level_values(0)[-1])
    day_features = features.xs(last_ts, level="ts")
    as_of = date(2024, 2, 27)

    in_memory = ModelBundle(
        model_version=VERSION,
        ranker=saved["ranker"],
        downside=saved["downside"],
        vol=saved["vol"],
        train_feature_quantiles=feature_quantiles(features),
        fold_ic_dispersion=0.05,
    )
    loaded = bundle_from_version(VERSION, saved["dir"])
    assert loaded.model_version == VERSION
    assert loaded.fold_ic_dispersion == 0.05
    assert loaded.vol_is_log is True
    a = score_universe(in_memory, day_features, as_of)
    b = score_universe(loaded, day_features, as_of)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]
    assert all(s.expected_excess_return is not None for s in b)
    # And loading twice is itself bit-identical (deterministic artifacts).
    c = score_universe(bundle_from_version(VERSION, saved["dir"]), day_features, as_of)
    assert [s.model_dump() for s in b] == [s.model_dump() for s in c]


def test_unknown_version_is_loud(saved: dict) -> None:
    with pytest.raises(QuantBundleError, match="unknown quant model version 'nope'"):
        bundle_from_version("nope", saved["dir"])


def test_sha_mismatch_is_loud(saved: dict, tmp_path: Path) -> None:
    import shutil

    corrupt = tmp_path / "corrupt"
    shutil.copytree(saved["dir"], corrupt)
    artifact = corrupt / f"{VERSION}-ranker.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="does not match"):
        bundle_from_version(VERSION, corrupt)


def test_feature_drift_is_loud(saved: dict, tmp_path: Path) -> None:
    # A trained feature the catalog no longer computes -> refuse to score.
    with pytest.raises(QuantBundleError, match="no longer computes"):
        bundle_from_version(
            VERSION,
            saved["dir"],
            current_feature_names=[*list(FEATURE_CATALOG)[:-1], "brand_new_feature"],
        )


def _day(saved: dict) -> pd.DataFrame:
    features: pd.DataFrame = saved["features"]
    return cast(
        pd.DataFrame, features.xs(cast(Any, features.index.get_level_values(0)[-1]), level="ts")
    )


def test_a_grown_catalog_still_loads_and_scores_identically(saved: dict) -> None:
    """Adding features to the catalog used to make EVERY existing bundle
    unloadable (exact-equality drift check) — campaign-v8 included. The bundle
    now selects its own trained columns by name, so the new column is ignored
    and the scores are bit-identical."""
    as_of = date(2024, 2, 27)
    base = _day(saved)
    grown_names = [*list(FEATURE_CATALOG), "brand_new_feature"]
    loaded = bundle_from_version(VERSION, saved["dir"], current_feature_names=grown_names)
    assert loaded.feature_names == tuple(FEATURE_CATALOG)

    grown = base.assign(brand_new_feature=123.0)
    a = score_universe(loaded, base, as_of)
    b = score_universe(loaded, grown, as_of)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_a_reordered_frame_is_realigned_not_misfed(saved: dict) -> None:
    """Column order feeds a tree model positionally; selecting by name in the
    TRAINING order makes a reordered frame score exactly like the original."""
    as_of = date(2024, 2, 27)
    base = _day(saved)
    loaded = bundle_from_version(VERSION, saved["dir"])
    shuffled = cast(pd.DataFrame, base[list(reversed(base.columns))])
    a = score_universe(loaded, base, as_of)
    b = score_universe(loaded, shuffled, as_of)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_missing_task_entry_is_loud(saved: dict, tmp_path: Path) -> None:
    import json
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(saved["dir"], broken)
    manifest_path = broken / f"{VERSION}.bundle.json"
    payload = json.loads(manifest_path.read_text())
    del payload["tasks"]["vol"]
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(QuantBundleError, match="no 'vol' task entry"):
        bundle_from_version(VERSION, broken)


def test_loader_is_reexported_for_the_preflight_hook() -> None:
    """The Phase-1 preflight hook resolves
    ``dynamo.research.quant.training.bundle_from_version`` — keep it there."""
    from dynamo.research.quant import training

    assert training.bundle_from_version is bundle_from_version


def test_fit_final_model_learns_the_planted_signal() -> None:
    features, labels = _panel()
    ranker = fit_final_model(features, labels, task="rank", model_factory=_factory("regression"))
    from typing import cast

    pred = pd.Series(np.asarray(ranker.predict(features)), index=features.index)
    ic = pred.corr(cast(pd.Series, labels["excess_return_20d"]), method="spearman")
    assert ic > 0.8

    downside = fit_final_model(features, labels, task="downside", model_factory=_factory("binary"))
    probs = np.asarray(downside.predict(features))
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    vol_model = fit_final_model(features, labels, task="vol", model_factory=_factory("regression"))
    # Vol models emit LOG vol (scoring exponentiates when vol_is_log).
    log_pred = np.asarray(vol_model.predict(features))
    assert np.corrcoef(log_pred, np.log(labels["realized_vol_20d"]))[0, 1] > 0.8


def test_fit_final_model_drops_nan_labels_and_requires_multiindex() -> None:
    features, labels = _panel()
    labels = labels.copy()
    labels.iloc[: len(labels) // 2, 0] = np.nan  # half the rank labels gone
    model = fit_final_model(features, labels, task="rank", model_factory=_factory("regression"))
    assert model is not None
    flat = features.reset_index(drop=True)
    with pytest.raises(ValueError, match="MultiIndex"):
        fit_final_model(
            flat,
            labels["excess_return_20d"],
            task="rank",
            model_factory=_factory("regression"),
        )
    all_nan = labels.copy()
    all_nan["realized_vol_20d"] = np.nan
    with pytest.raises(ValueError, match="no rows"):
        fit_final_model(features, all_nan, task="vol", model_factory=_factory("regression"))


def test_feature_quantiles_shape_and_nan_columns() -> None:
    features, _ = _panel()
    features = features.copy()
    features["all_nan"] = np.nan
    q = feature_quantiles(features)
    assert list(q.index) == ["q01", "q99"]
    assert list(q.columns) == list(features.columns)
    assert q.loc["q01", features.columns[0]] < q.loc["q99", features.columns[0]]
    assert np.isnan(q.loc["q01", "all_nan"])


# -- seed-ensemble bundles (--ensemble-seeds; N artifacts per task) -----------


def _ensemble_factory(objective: str, n: int = 2):
    from dynamo.research.quant.scoring import AveragingEnsemble

    def make() -> AveragingEnsemble:
        return AveragingEnsemble(
            [
                LightGBMModel(
                    objective,  # type: ignore[arg-type]
                    seed=42 + i,
                    n_estimators=15,
                    # Subsample diversity: LightGBM's seed only changes the
                    # fit under row/column subsampling (the trainer applies
                    # the same recipe for --ensemble-seeds > 1).
                    params={
                        "min_child_samples": 5,
                        "subsample": 0.7,
                        "subsample_freq": 1,
                        "colsample_bytree": 0.7,
                    },
                )
                for i in range(n)
            ]
        )

    return make


def test_ensemble_bundle_saves_members_and_loads_averaging_wrapper(
    tmp_path: Path,
) -> None:
    """An AveragingEnsemble task model persists one artifact PER member and
    loads back as the same wrapper — predictions bit-identical to in-memory."""
    from dynamo.research.quant.scoring import AveragingEnsemble

    features, labels = _panel()
    ranker = fit_final_model(
        features, labels, task="rank", model_factory=_ensemble_factory("regression")
    )
    downside = fit_final_model(
        features, labels, task="downside", model_factory=_ensemble_factory("binary")
    )
    vol = fit_final_model(features, labels, task="vol", model_factory=_factory("regression"))
    save_bundle(
        tmp_path,
        "ens-v1",
        ranker=ranker,
        downside=downside,
        vol=vol,
        feature_names=list(features.columns),
        train_feature_quantiles=feature_quantiles(features),
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        n_samples=len(features),
        config_hash="cfg-ens",
        metrics={},
        fold_ic_dispersion=0.02,
    )
    manifest = read_bundle_manifest(tmp_path, "ens-v1")
    members = manifest["tasks"]["ranker"]["members"]
    assert [m["model_id"] for m in members] == ["ens-v1-ranker-s00", "ens-v1-ranker-s01"]
    for m in members:
        assert (tmp_path / m["artifact_file"]).is_file()
    # Mixed bundle: the single-model vol task keeps the original shape.
    assert "members" not in manifest["tasks"]["vol"]
    assert manifest["tasks"]["vol"]["model_id"] == "ens-v1-vol"

    loaded = bundle_from_version("ens-v1", tmp_path)
    assert isinstance(loaded.ranker, AveragingEnsemble)
    assert len(loaded.ranker.members) == 2
    assert not isinstance(loaded.vol, AveragingEnsemble)
    from typing import Any, cast

    import numpy as _np

    last_ts = cast(Any, features.index.get_level_values(0)[-1])
    day = features.xs(last_ts, level="ts")
    _np.testing.assert_array_equal(loaded.ranker.predict(day), ranker.predict(day))
    # The wrapper truly averages: member predictions differ (different seeds)
    # and their mean equals the ensemble output.
    member_preds = [_np.asarray(m.predict(day), dtype=float) for m in loaded.ranker.members]
    assert not _np.array_equal(member_preds[0], member_preds[1])
    _np.testing.assert_allclose(
        loaded.ranker.predict(day), _np.mean(_np.stack(member_preds), axis=0)
    )


def test_single_model_bundles_keep_the_pre_ensemble_manifest_shape(saved: dict) -> None:
    manifest = read_bundle_manifest(saved["dir"], VERSION)
    for task in ("ranker", "downside", "vol"):
        assert "members" not in manifest["tasks"][task]
        assert "model_id" in manifest["tasks"][task]


def test_ensemble_member_sha_mismatch_is_loud(tmp_path: Path) -> None:
    features, labels = _panel()
    ranker = fit_final_model(
        features, labels, task="rank", model_factory=_ensemble_factory("regression")
    )
    single = fit_final_model(features, labels, task="vol", model_factory=_factory("regression"))
    save_bundle(
        tmp_path,
        "ens-bad",
        ranker=ranker,
        downside=single,
        vol=single,
        feature_names=list(features.columns),
        train_feature_quantiles=None,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        n_samples=len(features),
        config_hash="cfg",
        metrics={},
        fold_ic_dispersion=0.0,
    )
    artifact = tmp_path / "ens-bad-ranker-s01.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ArtifactIntegrityError):
        bundle_from_version("ens-bad", tmp_path)


def test_existing_campaign_v3_bundle_on_disk_still_loads() -> None:
    """Backward compatibility against the REAL registered bundle (when the
    working copy has it): the ensemble-aware loader must read the
    pre-ensemble manifest unchanged."""
    root = Path(__file__).resolve().parents[3]
    directory = root / "artifacts" / "models" / "quant"
    if not (directory / "campaign-v3.bundle.json").is_file():
        pytest.skip("campaign-v3 bundle not present on this machine")
    manifest = read_bundle_manifest(directory, "campaign-v3")
    bundle = bundle_from_version(
        "campaign-v3", directory, current_feature_names=manifest["feature_names"]
    )
    assert bundle.ranker is not None and bundle.downside is not None
    assert bundle.vol is not None
