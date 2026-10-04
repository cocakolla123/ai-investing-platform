"""Warm campaign-v10's panel cache in parallel shards.

``build_panel`` builds one session at a time. After a change that cools the
cache (the canonical_bars fix, 1c6c04d), rebuilding 680 sessions x 1,315
symbols serially takes over a day. Each shard here builds every Nth session
into the same ``PanelFeatureCache``. Writes are atomic per session, so
shards never collide, and a later quant_train / ml_baselines run is served
entirely from cache.

    python scripts/eval/warm_panel_cache.py --shard 0 --of 6
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

TRAIN_START, TRAIN_END = date(2021, 3, 1), date(2026, 9, 23)  # campaign-v10
STRIDE = 2
SYMBOLS_FILE = "config/training_universe_wide.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="build at most N sessions (timing)")
    args = ap.parse_args()

    from dynamo.cli.backfill import read_symbols_file
    from dynamo.config.settings import Settings
    from dynamo.data.universe import feature_input_symbols
    from dynamo.persistence.db import make_engine
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import _sessions_between
    from dynamo.research.quant.features import build_features
    from dynamo.research.quant.panel_cache import PanelFeatureCache

    engine = make_engine(Settings().database_url)
    universe = read_symbols_file(Path(SYMBOLS_FILE))
    feature_symbols = feature_input_symbols(universe)
    sessions = _sessions_between(TRAIN_START, TRAIN_END)[::-1][::STRIDE][::-1]
    cache = PanelFeatureCache(Path(Settings().artifact_dir) / "panel_cache", feature_symbols)
    mine = sessions[args.shard :: args.of]
    built = skipped = 0
    started = time.monotonic()
    for day in mine:
        if cache.get(day) is not None:
            skipped += 1
            continue
        inputs = HistoricalDataGateway(engine, day).feature_inputs(feature_symbols)
        cache.put(day, build_features(inputs, day, [str(s) for s in universe]))
        built += 1
        rate = (time.monotonic() - started) / built
        print(
            f"shard {args.shard}/{args.of} {day} built={built} skipped={skipped} "
            f"of {len(mine)} ({rate:.0f}s/session)",
            flush=True,
        )
        if args.limit and built >= args.limit:
            break
    print(f"shard {args.shard} done: built={built} skipped={skipped} cache={cache.directory}")


if __name__ == "__main__":
    main()
