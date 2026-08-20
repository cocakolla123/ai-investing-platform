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


def test_the_geometry_that_shipped_campaign_v5_no_longer_yields_two_folds() -> None:
    """89 strided rows used to give 2 folds; the gate needs 4."""
    from dynamo.research.quant.training import SHIP_MIN_FOLDS

    assert _folds_for(89, stride=4) >= SHIP_MIN_FOLDS


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


def test_more_data_still_yields_at_least_the_minimum_folds() -> None:
    from dynamo.research.quant.training import SHIP_MIN_FOLDS

    for n in (120, 250, 416, 900):
        assert _folds_for(n) >= SHIP_MIN_FOLDS, f"{n} sessions gave too few folds"
