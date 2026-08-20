"""Data-integrity repairs for the wide campaign (docs/wide_campaign_design.md).

One-shot, idempotent, re-runnable commands that fix defects measured against
the live database rather than inferred from the code:

    python -m dynamo.cli.repair corporate-actions --start 2024-01-01
    python -m dynamo.cli.repair backdate-universe --tier 2
    python -m dynamo.cli.repair census --source alpaca

- ``corporate-actions`` (D4) — ``corporate_actions`` held **0 rows**, so
  ``quality.corp_action_sanity`` had nothing to check overnight moves against
  and every split looked like an unexplained 40%+ jump. The Alpaca
  announcements endpoint is universe-independent (it returns the whole market),
  so this is one pass over a date range, chunked into <=90-day windows by the
  adapter. Idempotent: the table is keyed on
  ``(symbol, ex_date, action_type, source)``.

- ``backdate-universe`` (D3) — Tier-2 membership was written with
  ``valid_from`` = the day the universe was built, making the whole wide
  universe invisible to every point-in-time read before it. Rebases each row
  onto its first bar, a listing-date proxy that cannot inject look-ahead. See
  ``dynamo.data.universe.plan_listing_date_backdate`` for the four buckets and
  why three of them are refusals to act. ``--dry-run`` prints the plan and
  writes nothing.

- ``census`` (D1) — counts single-session moves too large to be prices and how
  many land on a near-exact split ratio, i.e. how much of the panel is still
  unadjusted. Read-only; this is the before/after measurement for a
  re-backfill, and the same number the replay preflight refuses to run on.
  ``--corporate-actions`` cross-checks each candidate against a real recorded
  split, which is the difference between "a ratio that looks like a split" and
  "a split".

Separate from ``dynamo.cli.backfill`` on purpose: these are repairs, they run
once and then stop being interesting, and keeping them apart leaves the
ingestion CLI free of one-shot migration code.

Zero LLM calls. ``corporate-actions`` makes real Alpaca requests; the other
two are database-only.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from typing import Any

DEFAULT_CORPORATE_ACTION_START = date(2024, 1, 1)


# -- injectable composition seams (tests monkeypatch these) -----------------


def _settings() -> Any:
    from dynamo.config.settings import Settings

    return Settings()


def _repo(database_url: str | None) -> Any:
    from dynamo.config.settings import Settings
    from dynamo.persistence.db import make_engine
    from dynamo.persistence.repositories.market import MarketDataRepository

    return MarketDataRepository(make_engine(database_url or Settings().database_url))


def _alpaca_market(settings: Any) -> Any:
    from dynamo.data.alpaca_data import AlpacaMarketData

    return AlpacaMarketData(settings.alpaca_api_key, settings.alpaca_api_secret)


# -- D4: corporate actions --------------------------------------------------


def _cmd_corporate_actions(args: argparse.Namespace) -> int:
    """Pull splits/dividends/mergers/spinoffs into ``corporate_actions``."""
    settings = _settings()
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        print(
            "ERROR: ALPACA_API_KEY / ALPACA_API_SECRET are not set — "
            "corporate-actions cannot run (no network calls made)."
        )
        return 1
    repo = _repo(args.database_url)
    end: date = args.end or date.today()
    if args.start > end:
        print(f"ERROR: --start {args.start} is after --end {end}")
        return 1

    before = repo.count_corporate_actions()
    adapter = _alpaca_market(settings)
    started = time.monotonic()
    # The endpoint is universe-independent: it returns announcements for the
    # whole market, so this is one pass over the range, not one per symbol.
    # Written in yearly slabs so a kill costs at most one slab, and each slab
    # is committed before the next is fetched.
    total = 0
    window_start = args.start
    while window_start <= end:
        window_end = min(date(window_start.year, 12, 31), end)
        actions = list(adapter.corporate_actions(window_start, window_end))
        written = repo.upsert_corporate_actions(actions)
        total += written
        print(
            f"corporate-actions {window_start} -> {window_end}: "
            f"{len(actions)} announcements, {written} rows upserted "
            f"({time.monotonic() - started:.0f}s)"
        )
        window_start = window_end + timedelta(days=1)

    after = repo.count_corporate_actions()
    print(
        f"corporate_actions: {before} rows before, {after} after "
        f"(+{after - before} new, {total - (after - before)} refreshed in place)"
    )
    by_type = repo.corporate_actions_by_type()
    for action_type, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {action_type}: {count}")
    if after == 0:
        print(
            "WARNING: the table is still empty. The Alpaca announcements "
            "endpoint returned nothing for this range — corp_action_sanity "
            "remains uncheckable."
        )
        return 1
    return 0


# -- D3: backdate the wide universe ----------------------------------------


def _cmd_backdate_universe(args: argparse.Namespace) -> int:
    """Rebase Tier-N ``valid_from`` onto each symbol's first bar."""
    from dynamo.data.universe import backdated_memberships, plan_listing_date_backdate
    from dynamo.domain.models.market import Tier

    repo = _repo(args.database_url)
    tier = Tier(args.tier)
    # Read membership from far enough in the future to see every row,
    # including the ones dated "today" that are the whole problem.
    horizon = date.today() + timedelta(days=365)
    members = repo.universe_at(horizon)
    target = [m for m in members if int(m.tier) == int(tier)]
    other = [m for m in members if int(m.tier) != int(tier)]
    other_counts: dict[int, int] = {}
    for m in other:
        other_counts[int(m.tier)] = other_counts.get(int(m.tier), 0) + 1
    print(
        f"universe_membership: {len(target)} tier-{int(tier)} rows, "
        f"{len(other)} rows in other tiers {other_counts} (never touched)"
    )

    first_bar = repo.first_bar_dates(source=args.source)
    print(f"first-bar dates known for {len(first_bar)} symbols (source={args.source or 'any'})")

    plan = plan_listing_date_backdate(members, first_bar, tier=tier)
    print(plan.summary())

    # The projected membership: rebased rows REPLACE their originals.
    rebased = backdated_memberships(members, plan)
    moved = {(r.symbol, r.old_valid_from) for r in plan.rebases}
    projected = [m for m in target if (m.symbol, m.valid_from) not in moved] + rebased
    print(f"projected valid_from distribution across {len(projected)} tier-{int(tier)} rows:")
    by_year: dict[int, int] = {}
    for m in projected:
        by_year[m.valid_from.year] = by_year.get(m.valid_from.year, 0) + 1
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]}")
    for probe in args.report_as_of:
        visible = sum(
            1
            for m in projected
            if m.valid_from <= probe and (m.valid_to is None or m.valid_to >= probe)
        )
        print(f"  projected visible at {probe}: {visible}")

    if not plan.rebases:
        print("nothing to backdate — already at their first bars (idempotent no-op)")
        return 0
    if args.dry_run:
        print("--dry-run: no writes performed")
        return 0

    deleted, inserted = repo.rebase_universe_valid_from(
        [(r.symbol, r.old_valid_from, r.new_valid_from) for r in plan.rebases]
    )
    print(f"rebased {len(plan.rebases)} rows: {deleted} deleted, {inserted} inserted")

    # Verification: other tiers byte-identical, target tier now visible.
    after = repo.universe_at(horizon)
    after_other = sorted(
        (m.symbol, int(m.tier), m.valid_from) for m in after if int(m.tier) != int(tier)
    )
    expected_other = sorted((m.symbol, int(m.tier), m.valid_from) for m in other)
    if after_other != expected_other:
        print("ERROR: rows outside the target tier CHANGED — this is a bug")
        return 1
    print(f"verify: {len(after_other)} rows outside tier {int(tier)} UNCHANGED")
    for probe in args.report_as_of:
        visible = repo.universe_at(probe, tier)
        print(f"verify: {len(visible)} tier-{int(tier)} symbols visible at {probe}")
    return 0


# -- D1: the split-artifact census -----------------------------------------


def _cmd_census(args: argparse.Namespace) -> int:
    """Count extreme moves and split artifacts in the stored bars."""
    import pandas as pd

    from dynamo.data.quality import extreme_move_census

    repo = _repo(args.database_url)
    frame = repo.bars_close_frame(source=args.source, start=args.start, end=args.end)
    print(f"loaded {len(frame)} bars (source={args.source or 'any'})")
    census = extreme_move_census(frame, min_median_close=args.min_median_close)
    print(census.summary())

    artifacts = census.events[census.events["split_ratio"].notna()]
    if artifacts.empty:
        print("no near-exact split ratios — the panel reads as adjusted")
        return 0

    grouped = artifacts.groupby("split_ratio").agg(
        events=("symbol", "size"), symbols=("symbol", "nunique")
    )
    print("by ratio:")
    print(grouped.to_string())

    if args.corporate_actions:
        # A ratio that LOOKS like a split is not a split. corporate_actions
        # is the independent witness: only a candidate landing on a recorded
        # split ex-date is evidence of unadjusted data. Everything else is a
        # genuine price move that happens to have doubled or halved.
        splits = repo.split_ex_dates()
        if not splits:
            print(
                "corporate_actions holds no splits — cross-check unavailable "
                "(run `repair corporate-actions` first)"
            )
            return 0
        keys = {
            (str(s), pd.Timestamp(t).date())
            for s, t in zip(artifacts["symbol"], artifacts["ts"], strict=True)
        }
        confirmed = keys & splits
        print(
            f"cross-check vs corporate_actions ({len(splits)} recorded splits): "
            f"{len(confirmed)} of {len(keys)} candidates fall on a real split "
            f"ex-date — those are unadjusted rows; the other "
            f"{len(keys) - len(confirmed)} are price moves that merely look "
            "like split ratios"
        )
        for symbol, day in sorted(confirmed)[:25]:
            print(f"  UNADJUSTED {symbol} {day}")
        return 1 if confirmed else 0
    return 0


# -- parser -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynamo.cli.repair", description=__doc__)
    parser.add_argument("--database-url", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    ca = sub.add_parser(
        "corporate-actions",
        help="pull splits/dividends/mergers/spinoffs into corporate_actions (D4)",
    )
    ca.add_argument("--start", type=date.fromisoformat, default=DEFAULT_CORPORATE_ACTION_START)
    ca.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="default: today",
    )

    bd = sub.add_parser(
        "backdate-universe",
        help="rebase a tier's valid_from onto each symbol's first bar (D3)",
    )
    bd.add_argument(
        "--tier",
        type=int,
        default=2,
        help="membership tier to rebase (default 2 = SURVEILLANCE). Rows of "
        "every other tier are ignored entirely.",
    )
    bd.add_argument(
        "--source",
        default=None,
        help="restrict the first-bar lookup to one bars_1d source "
        "(default: any source)",
    )
    bd.add_argument(
        "--report-as-of",
        nargs="*",
        type=date.fromisoformat,
        default=[date(2024, 1, 2), date(2026, 3, 3)],
        help="dates to report visible membership counts for",
    )
    bd.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")

    ce = sub.add_parser("census", help="count extreme moves and split artifacts (D1)")
    ce.add_argument("--source", default="alpaca", help="bars_1d source (default alpaca)")
    ce.add_argument("--start", type=date.fromisoformat, default=None)
    ce.add_argument("--end", type=date.fromisoformat, default=None)
    ce.add_argument(
        "--min-median-close",
        type=float,
        default=0.0,
        help="drop symbols whose median close is below this before counting",
    )
    ce.add_argument(
        "--corporate-actions",
        action="store_true",
        help="cross-check candidates against recorded split ex-dates; exit 1 "
        "if any candidate lands on a real split (i.e. the rows are unadjusted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "corporate-actions": _cmd_corporate_actions,
        "backdate-universe": _cmd_backdate_universe,
        "census": _cmd_census,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
