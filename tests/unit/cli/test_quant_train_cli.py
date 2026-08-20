def test_session_stride_keeps_every_nth_session_and_the_last_one() -> None:
    """Overlapping 20-day labels make adjacent sessions ~95% redundant, so
    striding buys build time AND sample independence. The window must still
    END at train_end, or the panel silently stops short of the requested range."""
    from datetime import date as _date
    from itertools import pairwise

    sessions = [_date(2026, 1, d) for d in range(1, 21)]
    stride = 5
    strided = sessions[::-1][::stride][::-1]
    assert strided[-1] == sessions[-1]
    assert len(strided) == 4
    gaps = {(b - a).days for a, b in pairwise(strided)}
    assert gaps == {stride}


# -- fold geometry and the ship gate -----------------------------------------
#
# campaign-v5 shipped on a rank t-statistic of 54.05 computed from TWO fold
# ICs, [0.0397, 0.0412]. With n=2 the sample standard deviation is just the
# gap between two numbers, so any two folds that agree produce an arbitrarily
# large t. These tests pin both halves of the fix: the geometry has to yield
# enough folds for the statistic to mean anything, and the gate has to refuse
# it when it does not.


def _folds_for(n_sessions: int, stride: int = 1) -> int:
    import pandas as pd

    from dynamo.backtest.walkforward import generate_folds
    from dynamo.cli.quant_train import walkforward_for

    cfg = walkforward_for(n_sessions, session_stride=stride)
    return len(generate_folds(pd.bdate_range("2024-06-01", periods=n_sessions), cfg))


def test_a_window_too_short_for_the_fold_size_is_refused_not_resliced() -> None:
    """The policy, stated as a test. The 89-row panel that shipped
    campaign-v5 genuinely cannot support fixed-size folds, and the honest
    answer is to fail the gate -- NOT to slice the same data thinner until
    four folds appear. Re-slicing adds no information but multiplies the
    t-statistic by sqrt(n), which is how a bad model passes."""
    from dynamo.research.quant.training import SHIP_MIN_FOLDS

    assert _folds_for(89, stride=4) < SHIP_MIN_FOLDS


def test_purge_covers_the_label_horizon_in_calendar_terms_not_strided_rows() -> None:
    """Purge exists so a fold never trains on rows whose 20-session forward
    windows overlap its test set. Counted in strided ROWS it meant 80 calendar
    sessions at stride 4 -- safe, but it spent a fifth of the panel buying 4x
    more separation than the labels need."""
    from dynamo.cli.quant_train import HORIZON_SESSIONS, walkforward_for

    for stride in (1, 2, 4, 5):
        cfg = walkforward_for(200, session_stride=stride)
        covered = cfg.purge_sessions * stride
        assert covered >= HORIZON_SESSIONS, f"stride {stride} under-purges: {covered} sessions"
        # ...and does not wildly over-purge either.
        assert covered < HORIZON_SESSIONS + stride, f"stride {stride} over-purges: {covered}"


def test_a_two_fold_run_cannot_ship_however_large_its_t_stat() -> None:
    """The gate hole, directly. A t-stat of 54 from 2 folds must not ship."""
    from dynamo.research.quant.training import SHIP_MIN_FOLDS, SHIP_T_STAT, _t_stat

    campaign_v5_folds = [0.0397, 0.0412]
    t = _t_stat(campaign_v5_folds)
    assert t > 50.0, "the historical number reproduces"
    assert t >= SHIP_T_STAT, "and it clears the old gate comfortably"
    assert len(campaign_v5_folds) < SHIP_MIN_FOLDS, "which is why fold count is now a gate too"


def test_a_window_too_small_for_one_fold_refuses_loudly() -> None:
    import pytest

    from dynamo.cli.quant_train import PanelTooSmallError, walkforward_for

    with pytest.raises(PanelTooSmallError, match="cannot form a single"):
        walkforward_for(20, session_stride=1)


def test_fold_count_follows_the_data_and_nothing_else() -> None:
    """Fold count must be an OUTPUT of how much history exists.

    More years -> strictly more folds. This is the property that makes
    sqrt(n_folds) legitimate: each added fold is a genuinely new exam on data
    the model never saw, not a re-cut of the same cake.
    """
    counts = [_folds_for(n) for n in (600, 1042, 1500, 2000)]
    assert counts == sorted(counts), f"fold count must not shrink with data: {counts}"
    assert counts[-1] > counts[0] * 2, f"more data must buy meaningfully more folds: {counts}"


def test_the_real_window_clears_the_fold_gate() -> None:
    """1,042 sessions is the measured 2022-01..2026-02 window."""
    from dynamo.research.quant.training import SHIP_MIN_FOLDS

    assert _folds_for(1042) >= SHIP_MIN_FOLDS


def test_fold_count_is_invariant_to_stride() -> None:
    """Stride is a compute knob, not a statistics knob. Train and test both
    scale with it, so striding must not change how many exams the model
    sits -- otherwise --session-stride would quietly be a gate lever."""
    full = _folds_for(1042, stride=1)
    for stride in (2, 4):
        strided = _folds_for(1042 // stride, stride=stride)
        assert abs(strided - full) <= 1, f"stride {stride}: {strided} vs {full} folds"


def test_fold_size_does_not_shrink_to_manufacture_folds() -> None:
    """The doc-08 defaults (train 3y / test 6mo) are deliberately unused: on
    the real window they yield TWO folds, the exact geometry that let
    campaign-v5 ship on a t computed from two numbers. But the replacement
    must not err the other way -- test windows stay a fixed size as the
    window grows, so extra folds come from extra data."""
    from dynamo.cli.quant_train import TEST_SESSIONS, TRAIN_SESSIONS, walkforward_for

    for n in (1042, 1500, 2000):
        cfg = walkforward_for(n)
        assert cfg.test_sessions == TEST_SESSIONS
        assert cfg.train_sessions == TRAIN_SESSIONS
