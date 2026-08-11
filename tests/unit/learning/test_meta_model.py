"""Meta-model feature builder + trainer (plan Phase 3).

Feature builder: determinism, NaN-safety, FEATURE_NAMES alignment, shared
snapshot path between training (experience) and scoring (decision-time).
Trainer: learns synthetic separable data (OOS Brier beats the base rate) and
refuses below the 150-experience hard gate.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from dynamo.domain.models.learning import DecisionExperience
from dynamo.domain.models.research import Action, CommitteeVerdict, QuantAlignment
from dynamo.learning.hierarchy import CalibratedProbability
from dynamo.learning.meta_model import (
    FEATURE_NAMES,
    MIN_TRAIN_EXPERIENCES,
    DecisionSnapshot,
    MetaModel,
    build_feature_row,
    build_features,
    features_hash,
    quant_veto_flag,
    sector_code,
    snapshot_from_decision,
    snapshot_from_experience,
)

NOW = datetime(2026, 8, 10, 20, 0, tzinfo=UTC)
D = date(2026, 8, 10)


def _exp(**overrides: object) -> DecisionExperience:
    base: dict[str, object] = {
        "id": "exp:live:dc-1",
        "created_at": NOW,
        "symbol": "AAPL",
        "as_of": D,
        "sleeve": "long_term",
        "regime": "risk_off",
        "sector": "tech",
        "vix": 22.5,
        "action": "buy",
        "conviction": 0.7,
        "disagreement_score": 0.3,
        "quant_alignment": "conflict",
        "expected_upside_pct": 12.0,
        "expected_downside_pct": 6.0,
        "horizon_days": 20,
        "quant_expected_excess_return": 0.02,
        "quant_downside_prob": 0.2,
        "resolved": True,
        "pnl_pct": 3.0,
        "label": "win",
    }
    base.update(overrides)
    return DecisionExperience(**base)  # type: ignore[arg-type]


# --- feature builder --------------------------------------------------------


def test_feature_row_keys_match_feature_names_in_order() -> None:
    row = build_feature_row(_exp(), None, [])
    assert tuple(row) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in row.values())


def test_feature_row_is_deterministic() -> None:
    a = build_feature_row(_exp(), None, [_exp(id="n1", pnl_pct=5.0)])
    b = build_feature_row(_exp(), None, [_exp(id="n1", pnl_pct=5.0)])
    assert a == b


def test_nan_safety_empty_inputs_yield_finite_floats() -> None:
    row = build_features(DecisionSnapshot(), None, [])
    assert tuple(row) == FEATURE_NAMES
    assert all(math.isfinite(v) for v in row.values())
    bad = DecisionSnapshot(conviction=float("nan"), vix=float("inf"), horizon_days=None)
    row = build_features(bad, None, [])
    assert all(math.isfinite(v) for v in row.values())


def test_verdict_and_context_features_encode() -> None:
    row = build_feature_row(_exp(), None, [])
    assert row["conviction"] == 0.7
    assert (row["qa_agree"], row["qa_neutral"], row["qa_conflict"]) == (0.0, 0.0, 1.0)
    assert row["quant_veto"] == 1.0  # buy against a quant conflict
    assert row["regime_risk_off"] == 1.0 and row["regime_crisis"] == 0.0
    assert row["sleeve_long_term"] == 1.0 and row["sleeve_tactical"] == 0.0
    assert row["upside_downside_ratio"] == pytest.approx(2.0)
    assert 0.0 <= row["sector_code"] < 1.0
    assert row["sector_code"] == sector_code("tech")  # stable, not hash()


def test_calibration_features_fall_back_to_conviction() -> None:
    without = build_feature_row(_exp(), None, [])
    assert without["calibrated_probability"] == 0.7
    assert without["calibration_n"] == 0.0
    calib = CalibratedProbability(value=0.55, scope_used="sleeve=long_term", n_effective=40)
    with_calib = build_feature_row(_exp(), calib, [])
    assert with_calib["calibrated_probability"] == 0.55
    assert with_calib["calibration_n"] == 40.0


def test_neighbor_retrieval_features() -> None:
    neighbors = [
        _exp(id="n1", pnl_pct=10.0),
        _exp(id="n2", pnl_pct=-4.0),
        _exp(id="n3", pnl_pct=None),  # unresolved pnl is skipped
    ]
    row = build_feature_row(_exp(), None, neighbors)
    assert row["neighbor_count"] == 2.0
    assert row["neighbor_mean_pnl_pct"] == pytest.approx(3.0)
    assert row["neighbor_hit_rate"] == pytest.approx(0.5)


def test_quant_veto_flag_definition() -> None:
    assert quant_veto_flag("buy", "conflict") == 1.0
    assert quant_veto_flag("add", "conflict") == 1.0
    assert quant_veto_flag("sell", "conflict") == 0.0
    assert quant_veto_flag("buy", "agree") == 0.0
    assert quant_veto_flag(None, None) == 0.0


def test_snapshot_paths_agree_on_shared_fields() -> None:
    """Training (experience) and scoring (verdict) snapshots must produce the
    same values for every field both sides can populate."""
    verdict = CommitteeVerdict(
        id="cv-AAPL",
        symbol="AAPL",
        as_of=D,
        action=Action.BUY,
        conviction=0.7,
        thesis="t",
        bull_case="b",
        bear_case="b",
        expected_upside_pct=12.0,
        expected_downside_pct=6.0,
        horizon_days=20,
        disagreement_score=0.3,
        quant_alignment=QuantAlignment.CONFLICT,
        self_critique="might be wrong",
    )
    from_decision = snapshot_from_decision(
        verdict, sector="tech", sleeve="long_term", vix=22.5
    )
    from_experience = snapshot_from_experience(_exp())
    row_d = build_features(from_decision, None, [])
    row_e = build_features(from_experience, None, [])
    experience_only_defaults = {
        "dissent_count",
        "has_self_critique",
        "journal_context_used",
        "regime_confidence",
        "spy_ret_20d",
        "quant_score",
        "quant_prob",
        "regime_risk_on",
        "regime_neutral",
        "regime_risk_off",
        "regime_crisis",
    }
    for name in FEATURE_NAMES:
        if name in experience_only_defaults:
            continue
        assert row_d[name] == row_e[name], name
    assert from_decision.dissent_count == 0
    assert from_decision.has_self_critique is True


def test_features_hash_is_stable() -> None:
    assert features_hash() == features_hash(FEATURE_NAMES)
    assert features_hash(("a", "b")) != features_hash(("b", "a"))


# --- trainer ----------------------------------------------------------------


def _synthetic(n: int, seed: int = 7) -> tuple[pd.DataFrame, list[bool], list[int]]:
    """Separable data: P(win) tracks conviction; everything else neutral."""
    rng = np.random.default_rng(seed)
    conviction = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(size=n) < conviction).tolist()
    rows = []
    for c in conviction:
        row = build_features(DecisionSnapshot(conviction=float(c)), None, [])
        rows.append(row)
    return pd.DataFrame(rows), labels, list(range(n))


def test_trainer_refuses_below_hard_gate() -> None:
    frame, labels, order = _synthetic(MIN_TRAIN_EXPERIENCES - 1)
    with pytest.raises(ValueError, match="hard gate"):
        MetaModel().train(frame, labels, order)


@pytest.mark.parametrize("algo", ["gradient_boosting", "logistic"])
def test_trainer_beats_base_rate_on_separable_data(algo: str) -> None:
    frame, labels, order = _synthetic(400)
    model = MetaModel(algo=algo)
    metrics = model.train(frame, labels, order)
    assert metrics["oos_brier"] < metrics["base_rate_brier"]
    assert metrics["train_n"] + metrics["oos_n"] == 400
    lo = build_features(DecisionSnapshot(conviction=0.1), None, [])
    hi = build_features(DecisionSnapshot(conviction=0.9), None, [])
    p_lo, p_hi = model.predict_proba([lo, hi])
    assert p_hi > p_lo
    assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0


def test_trainer_is_deterministic() -> None:
    frame, labels, order = _synthetic(300)
    row = build_features(DecisionSnapshot(conviction=0.6), None, [])
    m1, m2 = MetaModel(), MetaModel()
    metrics1 = m1.train(frame, labels, order)
    metrics2 = m2.train(frame, labels, order)
    assert metrics1 == metrics2
    assert m1.predict_proba([row]) == m2.predict_proba([row])


def test_trainer_degenerate_labels_fall_back_to_base_rate() -> None:
    frame, _, order = _synthetic(200)
    labels = [True] * 200  # single class — no classifier can fit
    model = MetaModel()
    metrics = model.train(frame, labels, order)
    row = build_features(DecisionSnapshot(conviction=0.5), None, [])
    assert model.predict_proba([row]) == [metrics["base_rate"]]


def test_trainer_alignment_mismatch_raises() -> None:
    frame, labels, order = _synthetic(200)
    with pytest.raises(ValueError, match="align"):
        MetaModel().train(frame, labels[:-1], order)
