"""``macro_series`` keeps every FRED revision; live readers must not trip on it.

The primary key is ``(series_id, ts, vintage_ts)``, so a revised observation
appears once per revision. ``regime_state`` and ``risk_verdict`` reindexed on
``ts`` as if it were unique and raised "cannot reindex on an axis with
duplicate labels" every live night — which is why the scores and
risk/portfolio jobs never completed and no paper order was ever sized.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from orchestration.assets._macro import aligned_series, latest_revisions


def _macro() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # 09-18 was published, then revised twice
            {
                "series_id": "T10Y3M",
                "ts": date(2026, 9, 18),
                "vintage_ts": datetime(2026, 9, 19, tzinfo=UTC),
                "value": 0.50,
            },
            {
                "series_id": "T10Y3M",
                "ts": date(2026, 9, 18),
                "vintage_ts": datetime(2026, 9, 22, tzinfo=UTC),
                "value": 0.58,
            },
            {
                "series_id": "T10Y3M",
                "ts": date(2026, 9, 18),
                "vintage_ts": datetime(2026, 9, 20, tzinfo=UTC),
                "value": 0.55,
            },
            {
                "series_id": "T10Y3M",
                "ts": date(2026, 9, 21),
                "vintage_ts": datetime(2026, 9, 22, tzinfo=UTC),
                "value": 0.61,
            },
            {
                "series_id": "BAMLH0A0HYM2",
                "ts": date(2026, 9, 18),
                "vintage_ts": datetime(2026, 9, 19, tzinfo=UTC),
                "value": 3.10,
            },
        ]
    )


def test_latest_revision_wins_regardless_of_row_order() -> None:
    out = latest_revisions(_macro())
    t = out[out["series_id"] == "T10Y3M"].set_index("ts")["value"]
    assert t[date(2026, 9, 18)] == 0.58
    assert not out.duplicated(["series_id", "ts"]).any()


def test_frame_without_vintage_column_still_cannot_duplicate() -> None:
    out = latest_revisions(_macro().drop(columns="vintage_ts"))
    assert not out.duplicated(["series_id", "ts"]).any()


def test_revisions_align_without_raising_and_forward_fill() -> None:
    dates = [date(2026, 9, 18), date(2026, 9, 19), date(2026, 9, 21), date(2026, 9, 22)]
    got = aligned_series(_macro(), "T10Y3M", dates, default=1.0)
    np.testing.assert_allclose(got, [0.58, 0.58, 0.61, 0.61])


def test_tz_aware_bar_dates_align_to_date_keyed_macro() -> None:
    """risk_verdict passes bar-derived (timestamptz) dates; they must align
    with the Date-keyed macro rather than silently fall back to the default."""
    aware = pd.DatetimeIndex(pd.to_datetime(["2026-09-18", "2026-09-21"]).tz_localize("UTC"))
    got = aligned_series(_macro(), "T10Y3M", aware, default=1.0)
    np.testing.assert_allclose(got, [0.58, 0.61])


def test_before_first_observation_and_missing_series_use_default() -> None:
    got = aligned_series(_macro(), "T10Y3M", [date(2026, 9, 17)], default=1.0)
    np.testing.assert_allclose(got, [1.0])
    missing = aligned_series(_macro(), "VIXCLS", [date(2026, 9, 18)], default=20.0)
    np.testing.assert_allclose(missing, [20.0])
    empty = aligned_series(_macro().iloc[0:0], "T10Y3M", [date(2026, 9, 18)], default=4.0)
    np.testing.assert_allclose(empty, [4.0])
