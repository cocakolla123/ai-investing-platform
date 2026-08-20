"""Golden-output regression for the ``_per_symbol`` hoist.

All five bar detectors used to group + sort the shared bars frame themselves,
so ``run_detectors`` paid that cost FIVE times per session — the dominant
per-day cost of a wide-universe discovery sweep. The grouping is now computed
once and threaded through.

The refactor is only safe if it is invisible, so this module asserts exactly
that on a multi-symbol panel that fires every bar detector: the flags from
the hoisted path and from the per-detector path are equal field for field,
in the same order.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from tests.unit.data.test_detectors import (
    _truncate_at_extreme,
    liquid,
    make_bars,
    noisy_flat,
    oscillator,
)

from dynamo.data.detectors import (
    DetectorInputs,
    _per_symbol,
    price_zscore,
    range_bound,
    range_break,
    rel_strength,
    run_detectors,
    volume_spike,
)
from dynamo.domain.models.market import SurveillanceFlag

AS_OF = datetime(2026, 7, 31, 22, 0, tzinfo=UTC)


def _panel() -> pd.DataFrame:
    """One frame holding every shape the bar detectors react to.

    - ``OSC``: an oscillator sitting near its low (``range_bound``);
    - ``ACME``: a flat series with a +10% closing session on 10x volume
      (``price_zscore``, ``volume_spike``, ``range_break``);
    - ``LAG`` / ``XLK``: a symbol and its sector ETF diverging over 20
      sessions (``rel_strength``);
    - ``SPLIT``: a session whose price ratio is outside the sanity band
      (``data_quality``), so the branch that ``continue``s is covered too.
    """
    osc = liquid(_truncate_at_extreme(oscillator(), kind="low"))

    closes = noisy_flat(70)
    closes.append(closes[-1] * 1.10)
    acme = make_bars("ACME", closes, [100_000] * 70 + [1_000_000])

    flat = noisy_flat(60)
    lag = make_bars("LAG", [*flat, *[flat[-1] * (1.0 + 0.01 * i) for i in range(1, 22)]])
    etf = make_bars("XLK", noisy_flat(81))

    split_closes = noisy_flat(70)
    split_closes.append(split_closes[-1] * 8.0)  # reverse-split shaped
    split = make_bars("SPLIT", split_closes)

    return pd.concat([osc, acme, lag, etf, split], ignore_index=True)


SECTOR_ETFS = {"LAG": "XLK"}


def _key(flag: SurveillanceFlag) -> tuple[object, ...]:
    return (
        flag.symbol,
        flag.detector,
        flag.severity,
        tuple(sorted(flag.details.items())),
        flag.ts,
    )


def test_panel_fires_every_bar_detector() -> None:
    """Guard on the fixture itself: a golden test over a panel that fires
    nothing would pass forever while proving nothing."""
    flags = run_detectors(DetectorInputs(bars=_panel(), as_of=AS_OF, sector_etf_map=SECTOR_ETFS))
    assert {f.detector for f in flags} == {
        "range_bound",
        "price_zscore",
        "volume_spike",
        "range_break",
        "rel_strength",
        "data_quality",
    }


def test_hoisted_grouping_reproduces_every_detector_exactly() -> None:
    bars = _panel()
    frames = _per_symbol(bars)
    empty = pd.DataFrame()
    for detector, kwargs in (
        (price_zscore, {}),
        (volume_spike, {}),
        (range_break, {}),
        (range_bound, {}),
        (rel_strength, {"sector_etf_map": SECTOR_ETFS}),
    ):
        golden = detector(bars, as_of=AS_OF, **kwargs)  # type: ignore[operator]
        hoisted = detector(bars, as_of=AS_OF, per_symbol=frames, **kwargs)  # type: ignore[operator]
        assert [_key(f) for f in hoisted] == [_key(f) for f in golden], detector
        # ``bars`` is never read once the grouping is supplied — an empty
        # frame must give the same answer, which is what lets the replay
        # panel skip materializing a combined frame at all.
        detached = detector(empty, as_of=AS_OF, per_symbol=frames, **kwargs)  # type: ignore[operator]
        assert [_key(f) for f in detached] == [_key(f) for f in golden], detector


def test_run_detectors_is_identical_with_a_supplied_grouping() -> None:
    bars = _panel()
    golden = run_detectors(DetectorInputs(bars=bars, as_of=AS_OF, sector_etf_map=SECTOR_ETFS))
    hoisted = run_detectors(
        DetectorInputs(
            bars=pd.DataFrame(),
            as_of=AS_OF,
            per_symbol=_per_symbol(bars),
            sector_etf_map=SECTOR_ETFS,
        )
    )
    assert [_key(f) for f in hoisted] == [_key(f) for f in golden]


def test_empty_grouping_runs_no_bar_detector() -> None:
    assert run_detectors(DetectorInputs(bars=pd.DataFrame(), as_of=AS_OF, per_symbol={})) == []
