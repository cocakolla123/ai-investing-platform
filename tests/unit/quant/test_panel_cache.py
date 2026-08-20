"""Panel feature cache (training re-run speedup): bit-identical round trips,
stale-key (universe / catalog fingerprint) misses, corruption tolerance, and
the ``build_panel`` wiring incl. the ``--no-cache`` CLI bypass."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from dynamo.cli import quant_train as qt
from dynamo.research.quant.catalog import FeatureInputs
from dynamo.research.quant.panel_cache import (
    PanelFeatureCache,
    catalog_fingerprint,
    universe_hash,
)
from tests.unit.quant.conftest import make_bars

DAY = date(2024, 6, 3)
UNIVERSE = ["AAA", "BBB", "CCC"]


def frame_with_nans(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(3, 4))
    values[0, 1] = np.nan
    values[2, 3] = np.nan
    out = pd.DataFrame(
        values,
        index=pd.Index(UNIVERSE, name="symbol"),
        columns=pd.Index(["ret_20d", "rsi_14", "earnings_yield", "vol_cc_20d"]),
    )
    return out


# -- round trip --------------------------------------------------------------


def test_round_trip_is_bit_identical(tmp_path: Path) -> None:
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    frame = frame_with_nans()
    cache.put(DAY, frame)
    got = cache.get(DAY)
    assert got is not None
    pd.testing.assert_frame_equal(got, frame, check_exact=True)


def test_real_build_features_round_trip(tmp_path: Path) -> None:
    """The actual normalized feature frame survives parquet exactly."""
    from dynamo.research.quant.features import build_features

    bars = make_bars(["AAA", "BBB", "CCC", "XLK"], n_sessions=120)
    inputs = FeatureInputs(bars=bars, sector_map={"AAA": "XLK", "BBB": "XLK"})
    frame = build_features(inputs, DAY, UNIVERSE)
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    cache.put(DAY, frame)
    got = cache.get(DAY)
    assert got is not None
    pd.testing.assert_frame_equal(got, frame, check_exact=True)


def test_miss_on_unknown_session(tmp_path: Path) -> None:
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    assert cache.get(DAY) is None


# -- key discipline ----------------------------------------------------------


def test_universe_hash_is_order_insensitive_and_set_sensitive() -> None:
    assert universe_hash(["B", "A"]) == universe_hash(["A", "B"])
    assert universe_hash(["A", "B"]) != universe_hash(["A", "B", "C"])


def test_different_universe_misses(tmp_path: Path) -> None:
    PanelFeatureCache(tmp_path, UNIVERSE).put(DAY, frame_with_nans())
    other = PanelFeatureCache(tmp_path, [*UNIVERSE, "DDD"])
    assert other.get(DAY) is None


def test_same_universe_different_order_hits(tmp_path: Path) -> None:
    PanelFeatureCache(tmp_path, UNIVERSE).put(DAY, frame_with_nans())
    reordered = PanelFeatureCache(tmp_path, list(reversed(UNIVERSE)))
    assert reordered.get(DAY) is not None


def test_different_catalog_fingerprint_misses(tmp_path: Path) -> None:
    PanelFeatureCache(tmp_path, UNIVERSE).put(DAY, frame_with_nans())
    drifted = PanelFeatureCache(tmp_path, UNIVERSE, fingerprint="beef" * 16)
    assert drifted.get(DAY) is None


def test_default_fingerprint_is_stable_and_used(tmp_path: Path) -> None:
    fp = catalog_fingerprint()
    assert fp == catalog_fingerprint()  # deterministic within a code version
    assert len(fp) == 64
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    assert cache.directory.name.startswith(fp[:16])


# -- corruption --------------------------------------------------------------


def test_corrupted_file_is_a_miss_then_rebuildable(tmp_path: Path) -> None:
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    cache.put(DAY, frame_with_nans())
    cache.path_for(DAY).write_bytes(b"not a parquet file")
    assert cache.get(DAY) is None  # miss, not a crash
    assert not cache.path_for(DAY).exists()  # cleared for rebuild
    cache.put(DAY, frame_with_nans())  # rebuild path works
    assert cache.get(DAY) is not None


def test_put_never_leaves_tmp_litter(tmp_path: Path) -> None:
    cache = PanelFeatureCache(tmp_path, UNIVERSE)
    cache.put(DAY, frame_with_nans())
    leftovers = [p for p in cache.directory.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# -- build_panel wiring ------------------------------------------------------


def panel_fixtures() -> tuple[Any, Any, dict[str, str], list[str], list[date]]:
    universe = ["AAA", "BBB", "CCC"]
    bars = make_bars([*universe, "XLK", "SPY"], n_sessions=60)
    inputs = FeatureInputs(bars=bars, sector_map={"AAA": "XLK", "BBB": "XLK"})
    sector_map = {"AAA": "XLK", "BBB": "XLK", "CCC": "SPY"}
    sessions = [cast(pd.Timestamp, t).date() for t in pd.bdate_range("2024-01-02", periods=60)]
    return inputs, bars, sector_map, universe, sessions


def test_build_panel_warm_cache_skips_the_loader(tmp_path: Path) -> None:
    inputs, bars, sector_map, universe, sessions = panel_fixtures()
    calls: list[date] = []

    def loader(d: date) -> FeatureInputs:
        calls.append(d)
        return inputs

    cache = PanelFeatureCache(tmp_path, universe)
    cold = qt.build_panel(
        loader,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=cache,
    )
    n_feature_sessions = len(cold.sessions)
    assert len(calls) == n_feature_sessions  # cold: every session built

    calls.clear()
    warm = qt.build_panel(
        loader,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=cache,
    )
    assert calls == []  # warm: zero gateway loads
    pd.testing.assert_frame_equal(
        cast(pd.DataFrame, warm.features),
        cast(pd.DataFrame, cold.features),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(cast(pd.DataFrame, warm.labels), cast(pd.DataFrame, cold.labels))
    assert warm.sessions == cold.sessions


def test_build_panel_without_cache_always_builds(tmp_path: Path) -> None:
    inputs, bars, sector_map, universe, sessions = panel_fixtures()
    calls: list[date] = []

    def loader(d: date) -> FeatureInputs:
        calls.append(d)
        return inputs

    first = qt.build_panel(loader, bars, sector_map, universe, sessions, horizon_sessions=20)
    n = len(first.sessions)
    qt.build_panel(loader, bars, sector_map, universe, sessions, horizon_sessions=20)
    assert len(calls) == 2 * n  # no cache object -> no reuse


def test_build_panel_writes_incrementally_and_reports_hits(tmp_path: Path) -> None:
    """Each session's frame lands on disk as soon as it is built (crash-safe
    warm-up), and the progress line reports hit counts on the warm run."""
    inputs, bars, sector_map, universe, sessions = panel_fixtures()
    cache = PanelFeatureCache(tmp_path, universe)
    seen_on_disk: list[int] = []

    original_put = cache.put

    def counting_put(day: date, frame: pd.DataFrame) -> None:
        original_put(day, frame)
        seen_on_disk.append(len(list(cache.directory.glob("*.parquet"))))

    cache.put = counting_put  # type: ignore[method-assign]
    lines: list[str] = []
    panel = qt.build_panel(
        lambda d: inputs,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=cache,
        progress=lines.append,
    )
    assert seen_on_disk == list(range(1, len(panel.sessions) + 1))
    assert any("0 cache hits" in line for line in lines)

    lines.clear()
    qt.build_panel(
        lambda d: pytest.fail("warm run must not load"),  # type: ignore[arg-type]
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=cache,
        progress=lines.append,
    )
    n = len(panel.sessions)
    assert f"feature cache served {n}/{n} sessions" in "\n".join(lines)


def test_stale_cache_entries_are_rebuilt_not_reused(tmp_path: Path) -> None:
    """A frame cached under a different catalog fingerprint is invisible —
    the loader runs again instead of silently reusing stale features."""
    inputs, bars, sector_map, universe, sessions = panel_fixtures()
    old = PanelFeatureCache(tmp_path, universe, fingerprint="0ld" + "0" * 61)
    qt.build_panel(
        lambda d: inputs,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=old,
    )
    calls: list[date] = []

    def loader(d: date) -> FeatureInputs:
        calls.append(d)
        return inputs

    fresh = PanelFeatureCache(tmp_path, universe)  # current fingerprint
    panel = qt.build_panel(
        loader,
        bars,
        sector_map,
        universe,
        sessions,
        horizon_sessions=20,
        cache=fresh,
    )
    assert len(calls) == len(panel.sessions)  # zero stale hits


# -- CLI flag ----------------------------------------------------------------


def test_no_cache_flag_parses() -> None:
    base = ["--train-start", "2024-01-02", "--train-end", "2025-01-02", "--label", "x"]
    assert qt._build_parser().parse_args(base).no_cache is False
    assert qt._build_parser().parse_args([*base, "--no-cache"]).no_cache is True
