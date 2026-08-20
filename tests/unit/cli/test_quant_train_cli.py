

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
