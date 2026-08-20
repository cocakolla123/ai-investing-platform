"""Property tests for the canonical walk-forward splitter (doc 08 section 4)."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dynamo.backtest.walkforward import (
    ConfigFrozenError,
    EvaluationRegistry,
    WalkForwardConfig,
    compute_config_hash,
    generate_folds,
)

SMALL = WalkForwardConfig(
    train_sessions=120, test_sessions=40, purge_sessions=20, embargo_sessions=10, roll_sessions=40
)


def _sessions(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-05", periods=n)


@st.composite
def _configs(draw: st.DrawFn) -> WalkForwardConfig:
    return WalkForwardConfig(
        train_sessions=draw(st.integers(30, 150)),
        test_sessions=draw(st.integers(10, 60)),
        purge_sessions=draw(st.integers(0, 25)),
        embargo_sessions=draw(st.integers(0, 15)),
        roll_sessions=draw(st.integers(10, 60)),
    )


@settings(max_examples=40, deadline=None)
@given(n=st.integers(150, 900), config=_configs())
def test_walkforward_properties(n: int, config: WalkForwardConfig) -> None:
    sessions = _sessions(n)
    pos = {ts: i for i, ts in enumerate(sessions)}
    folds = generate_folds(sessions, config)

    prev_test_start = None
    for fold in folds:
        assert len(fold.test_sessions) == config.test_sessions
        # Folds chronological, train strictly before test.
        if prev_test_start is not None:
            assert fold.test_start > prev_test_start
        prev_test_start = fold.test_start
        train_max = cast(pd.Timestamp, fold.train_sessions.max())
        assert train_max < fold.test_start

        # Purge: no train session within `purge` sessions of test start.
        gap = pos[fold.test_start] - pos[train_max]
        assert gap > config.purge_sessions

    # Embargo: sessions just after an earlier fold's test window never re-enter
    # a later fold's training set.
    for i, earlier in enumerate(folds):
        end_pos = pos[earlier.test_end]
        embargo_zone = set(sessions[end_pos + 1 : end_pos + 1 + config.embargo_sessions])
        for later in folds[i + 1 :]:
            assert embargo_zone.isdisjoint(set(later.train_sessions))


def test_fold_windows_match_config() -> None:
    sessions = _sessions(400)
    folds = generate_folds(sessions, SMALL)
    assert len(folds) >= 2
    f0 = folds[0]
    assert len(f0.train_sessions) == SMALL.train_sessions
    assert f0.train_sessions[0] == sessions[0]
    # test starts exactly train + purge sessions in
    assert f0.test_start == sessions[SMALL.train_sessions + SMALL.purge_sessions]
    # roll: next fold's test starts roll sessions later
    assert (
        folds[1].test_start
        == sessions[SMALL.train_sessions + SMALL.purge_sessions + SMALL.roll_sessions]
    )


def test_too_few_sessions_yields_no_folds() -> None:
    assert generate_folds(_sessions(100), SMALL) == []


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(train_sessions=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(purge_sessions=-1)


def test_freeze_guard_refuses_repeat_window() -> None:
    sessions = _sessions(400)
    folds = generate_folds(sessions, SMALL)
    registry = EvaluationRegistry()
    h = compute_config_hash({"model": "lgbm", "features": ["a", "b"]})
    registry.claim_folds(h, folds)
    with pytest.raises(ConfigFrozenError):
        registry.claim_folds(h, folds)
    # A different config hash may see the same windows.
    registry.claim_folds(compute_config_hash({"model": "lgbm", "features": ["a"]}), folds)
    records = registry.records()
    assert len(records) == 2 * len(folds)
    assert {"config_hash", "test_start", "test_end", "claim_order"} <= set(records[0])


def test_freeze_guard_is_all_or_nothing() -> None:
    sessions = _sessions(400)
    folds = generate_folds(sessions, SMALL)
    registry = EvaluationRegistry()
    h = compute_config_hash({"v": 1})
    registry.claim(h, folds[-1].test_start, folds[-1].test_end)
    with pytest.raises(ConfigFrozenError):
        registry.claim_folds(h, folds)
    # The earlier fold windows must not have been partially recorded.
    assert not registry.is_claimed(h, folds[0].test_start, folds[0].test_end)


def test_config_hash_stable_and_order_independent() -> None:
    a = compute_config_hash({"x": 1, "y": [1, 2]})
    b = compute_config_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != compute_config_hash({"x": 2, "y": [1, 2]})
