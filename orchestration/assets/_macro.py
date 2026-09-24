"""Shared macro-series helpers for the live nightly assets.

``macro_series`` keeps every FRED revision — the primary key is
``(series_id, ts, vintage_ts)`` — so one observation date appears once per
revision. Three live readers selected ``(series_id, ts, value)`` and then
indexed on ``ts`` as if it were unique:

* ``regime_state`` and ``risk_verdict`` reindexed on it and raised "cannot
  reindex on an axis with duplicate labels" on every live night, which is why
  the scores and risk/portfolio jobs never completed and no paper order was
  ever sized;
* ``_macro_context`` (the committee's macro digest) took ``iloc[-1]`` of a
  ts-sorted group, i.e. an arbitrary revision of the latest date.

The live assets classify *tonight*, so the latest revision of each
observation is the value actually known now. Point-in-time vintage selection
for historical dates is the replay gateway's job, not this module's.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd


def latest_revisions(macro: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(series_id, ts)``: the most recent ``vintage_ts``.

    Frames without a ``vintage_ts`` column are returned de-duplicated on
    ``(series_id, ts)`` keeping the last row, so a caller that forgot to
    select the vintage still cannot produce duplicate labels.
    """
    if macro.empty:
        return macro
    keys = ["series_id", "ts"]
    order = [*keys, "vintage_ts"] if "vintage_ts" in macro.columns else keys
    return (
        macro.sort_values(order, kind="mergesort")
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )


def aligned_series(
    macro: pd.DataFrame,
    series_id: str,
    dates: Sequence[date] | pd.Index,
    default: float,
) -> np.ndarray:
    """``series_id`` forward-filled onto ``dates``, ``default`` where unknown.

    Both sides are compared as tz-naive calendar dates: macro ``ts`` is a
    ``Date`` column while the bar-derived ``dates`` may carry a timestamptz,
    and a naive/aware mismatch in ``reindex`` would raise (or, on some pandas
    paths, silently align nothing and hand back all-``default``).
    """
    n = len(dates)
    if macro.empty:
        return np.full(n, default, dtype=float)
    sub = latest_revisions(cast(pd.DataFrame, macro[macro["series_id"] == series_id]))
    if sub.empty:
        return np.full(n, default, dtype=float)
    index = _naive_days(sub["ts"])
    s = pd.Series(sub["value"].to_numpy(dtype=float), index=index).sort_index()
    target = _naive_days(dates)
    aligned = s.reindex(target, method="ffill").fillna(default)
    return np.asarray(aligned, dtype=float)


def _naive_days(values: object) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(list(values), utc=True))  # type: ignore[arg-type]
    # pandas-stubs omits DatetimeIndex.floor; the runtime method exists.
    return cast(pd.DatetimeIndex, cast(Any, idx.tz_localize(None)).floor("D"))
