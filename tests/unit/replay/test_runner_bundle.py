"""Trained-bundle loading in the replay runner (backtest-campaign Phase 2):
``quant_model_version`` set -> the injected bundle scores the universe
(non-None expected excess returns, real fills); unset -> baseline behavior
bit-identical to before, factory never touched; the version participates in
config_hash; the default factory is the artifact loader and its failures are
loud BEFORE any run row is written."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

from dynamo.replay.runner import ReplayConfig, ReplayRunner, quant_scores_for
from dynamo.research.quant.bundle import QuantBundleError
from dynamo.research.quant.scoring import ModelBundle
from tests.unit.replay.test_runner import (
    EER,
    RUN_SESSIONS,
    UNIVERSE,
    FakeExperienceRepo,
    FakeGateway,
    FakeRunRepo,
    fixture_frames,  # noqa: F401  (module-scoped fixture reused here)
)


class StaticModel:
    """ForecastModel stand-in: fixed prediction per symbol (features row
    index), NaN for unknown symbols — deterministic, fit-free."""

    def __init__(self, by_symbol: dict[str, float]) -> None:
        self._by_symbol = by_symbol

    def fit(self, x: Any, y: Any, cfg: dict[str, Any] | None = None) -> StaticModel:
        return self

    def predict(self, x: Any) -> np.ndarray:
        return np.asarray([self._by_symbol.get(str(s), float("nan")) for s in x.index], dtype=float)

    def feature_importances(self) -> dict[str, float]:
        return {}


def _stub_bundle(version: str) -> ModelBundle:
    return ModelBundle(
        model_version=version,
        ranker=StaticModel(dict(EER)),
        downside=StaticModel(dict.fromkeys(UNIVERSE, 0.1)),
        vol=StaticModel(dict.fromkeys(UNIVERSE, float(np.log(0.2)))),
    )


class NoPersistedScoresGateway(FakeGateway):
    """The campaign shape: NO persisted historical scores anywhere — every
    score must come from the bundle (or baseline abstention)."""

    def quant_scores(self, symbols: Any = None) -> Any:
        import pandas as pd

        return pd.DataFrame()


def _run(fixture, *, version: str, factory, gateway_cls=NoPersistedScoresGateway):
    sessions, bars, scores = fixture
    window = sessions[-RUN_SESSIONS:]
    config = ReplayConfig(
        start=window[0],
        end=window[-1],
        label="bundle-test",
        horizon_sessions=3,
        quant_model_version=version,
        # The fake gateway returns an EMPTY FeatureInputs on purpose (the stub
        # bundle predicts from the row index, not from feature values), so
        # every row's coverage is ~1/51. This test is about bundle wiring, not
        # the coverage gate — the gate has its own tests.
        min_feature_coverage=0.0,
    )
    exp_repo = FakeExperienceRepo()
    run_repo = FakeRunRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: gateway_cls(d, bars, scores),
        experience_repo=exp_repo,
        run_repo=run_repo,
        config=config,
        bundle_factory=factory,
    )
    return runner.run(), exp_repo, run_repo


def test_version_set_scores_through_injected_bundle(fixture_frames: Any) -> None:  # noqa: F811
    factory_calls: list[str] = []

    def factory(version: str) -> ModelBundle:
        factory_calls.append(version)
        return _stub_bundle(version)

    res, _, _ = _run(fixture_frames, version="campaign-v1", factory=factory)
    assert res.status == "completed"
    assert factory_calls == ["campaign-v1"]  # loaded ONCE at run start
    # The bundle produced real expected excess returns -> BUY verdicts,
    # fills, and experiences stamped with the bundle's model version.
    verdict_rows = [d for d in res.decisions if d["payload"]["verdict"] is not None]
    assert verdict_rows, "bundle scores must synthesize verdicts"
    assert any(f.side == "buy" for f in res.fills)
    assert res.experiences
    for exp in res.experiences:
        assert exp.quant_expected_excess_return is not None
    liq = [e for e in res.experiences if e.symbol == "LIQ"]
    assert liq and liq[0].quant_expected_excess_return == pytest.approx(EER["LIQ"])


def test_version_unset_keeps_baseline_and_never_touches_the_factory(
    fixture_frames: Any,  # noqa: F811
) -> None:
    def exploding_factory(version: str) -> ModelBundle:  # pragma: no cover
        raise AssertionError("factory must not be called when version is unset")

    res, _, _ = _run(fixture_frames, version="", factory=exploding_factory)
    assert res.status == "completed"
    # No persisted scores + baseline-v0 abstention -> zero verdicts, no fills.
    assert all(d["payload"]["verdict"] is None for d in res.decisions)
    assert not res.fills


def test_config_hash_and_run_id_differ_with_model_version() -> None:
    base = dict(start=date(2026, 6, 1), end=date(2026, 6, 5), label="x")
    without = ReplayConfig(**base)  # type: ignore[arg-type]
    with_v1 = ReplayConfig(**base, quant_model_version="campaign-v1")  # type: ignore[arg-type]
    with_v2 = ReplayConfig(**base, quant_model_version="campaign-v2")  # type: ignore[arg-type]
    assert without.quant_model_version == ""  # default preserves old behavior
    assert len({without.config_hash, with_v1.config_hash, with_v2.config_hash}) == 3
    assert len({without.run_id, with_v1.run_id, with_v2.run_id}) == 3


def test_default_factory_is_the_artifact_loader_and_fails_loud(
    fixture_frames: Any,  # noqa: F811
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No injected factory -> ``bundle_from_version`` resolves the version;
    an unknown version dies BEFORE create_run writes anything."""
    import dynamo.research.quant.bundle as bundle_mod

    monkeypatch.setattr(
        bundle_mod,
        "bundle_from_version",
        lambda version, artifact_dir=None, **kw: bundle_mod.read_bundle_manifest(tmp_path, version),
    )
    sessions, bars, scores = fixture_frames
    window = sessions[-RUN_SESSIONS:]
    run_repo = FakeRunRepo()
    runner = ReplayRunner(
        gateway_factory=lambda d: NoPersistedScoresGateway(d, bars, scores),
        experience_repo=FakeExperienceRepo(),
        run_repo=run_repo,
        config=ReplayConfig(
            start=window[0],
            end=window[-1],
            label="loud",
            quant_model_version="missing-version",
        ),
    )
    with pytest.raises(QuantBundleError, match="unknown quant model version"):
        runner.run()
    assert run_repo.runs == []  # nothing written: fail before create_run


def test_quant_scores_for_prefers_persisted_then_bundle(fixture_frames: Any) -> None:  # noqa: F811
    """Persisted rows (what-was-known-then) still win; the bundle only fills
    symbols without a fresh persisted score."""
    sessions, bars, scores = fixture_frames
    day = sessions[-1]
    gw = FakeGateway(day, bars, scores)  # persisted scores exist for everyone
    got = quant_scores_for(gw, UNIVERSE, day, 5, bundle=_stub_bundle("campaign-v1"))
    assert {s.model_version for s in got.values()} == {"test-v1"}
    gw_empty = NoPersistedScoresGateway(day, bars, scores)
    got_bundle = quant_scores_for(gw_empty, UNIVERSE, day, 5, bundle=_stub_bundle("campaign-v1"))
    assert {s.model_version for s in got_bundle.values()} == {"campaign-v1"}
    assert all(s.expected_excess_return is not None for s in got_bundle.values())
