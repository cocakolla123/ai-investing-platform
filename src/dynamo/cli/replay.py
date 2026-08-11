"""Historical replay CLI (plan Phase 4) — manual only, NEVER scheduled.

Usage:
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode quant --run-label first-window
    python -m dynamo.cli.replay list
    python -m dynamo.cli.replay show --run-id replay-abc123def456

``--mode committee`` is accepted as an argument but exits with a message —
committee replay (with its hard per-run dollar cap) ships in Phase 5. The
quant mode spends $0: zero LLM calls anywhere in the replay path.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-replay",
        description="Run/inspect historical quant-only replays (manual, never scheduled)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a replay over [start, end]")
    run.add_argument("--start", required=True, type=date.fromisoformat)
    run.add_argument("--end", required=True, type=date.fromisoformat)
    run.add_argument(
        "--mode",
        choices=("quant", "committee"),
        default="quant",
        help="committee mode ships in Phase 5 (stub here)",
    )
    run.add_argument("--run-label", required=True, help="human label for the run")
    run.add_argument(
        "--universe-tier",
        type=int,
        action="append",
        default=None,
        help="universe tier(s) to replay (repeatable; default: 1 3)",
    )
    run.add_argument("--initial-equity", type=float, default=100_000.0)
    run.add_argument("--horizon-sessions", type=int, default=20)
    run.add_argument("--database-url", default=None)

    lst = sub.add_parser("list", help="list replay runs")
    lst.add_argument("--database-url", default=None)
    lst.add_argument("--limit", type=int, default=20)

    show = sub.add_parser("show", help="show one run's decisions/experiences")
    show.add_argument("--run-id", required=True)
    show.add_argument("--database-url", default=None)
    return parser


def _engine(database_url: str | None) -> Any:
    from dynamo.config.settings import Settings
    from dynamo.persistence.db import make_engine

    return make_engine(database_url or Settings().database_url)


def _cmd_run(args: argparse.Namespace) -> int:
    if args.mode == "committee":
        print(
            "committee replay mode ships in Phase 5 (budget-capped LLM replay); "
            "only --mode quant is available today."
        )
        return 2
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import ReplayConfig, ReplayRunner

    engine = _engine(args.database_url)
    config = ReplayConfig(
        start=args.start,
        end=args.end,
        mode=args.mode,
        label=args.run_label,
        universe_tiers=tuple(args.universe_tier or (1, 3)),
        initial_equity=args.initial_equity,
        horizon_sessions=args.horizon_sessions,
    )
    runner = ReplayRunner(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        experience_repo=ExperienceRepository(engine),
        run_repo=ReplayRepository(engine),
        config=config,
    )
    print(f"replay {config.run_id} ({config.start} -> {config.end}, mode={config.mode})")
    result = runner.run(progress=print)

    equities = [e for _, e in result.equity_curve]
    summary = [
        f"run:          {result.run_id} [{result.status}]",
        f"sessions:     {len(result.sessions)}",
        f"decisions:    {len(result.decisions)}",
        f"fills:        {len(result.fills)}",
        f"experiences:  {len(result.experiences)} resolved (source='replay')",
    ]
    if equities:
        start_eq, end_eq = equities[0], equities[-1]
        peak, max_dd = start_eq, 0.0
        for e in equities:
            peak = max(peak, e)
            max_dd = min(max_dd, e / peak - 1.0)
        summary += [
            f"equity:       {start_eq:,.2f} -> {end_eq:,.2f} "
            f"({(end_eq / start_eq - 1.0) * 100.0:+.2f}%)",
            f"max drawdown: {max_dd * 100.0:.2f}%",
        ]
    print("\n".join(summary))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from dynamo.persistence.repositories.replay import ReplayRepository

    repo = ReplayRepository(_engine(args.database_url))
    runs = repo.list_runs(limit=args.limit)
    if not runs:
        print("no replay runs")
        return 0
    for r in runs:
        print(
            f"{r['id']}  {r['status']:<10} {r['mode']:<6} "
            f"{r['start_as_of']} -> {r['end_as_of']}  label={r['label']!r}"
        )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.replay import ReplayRepository

    engine = _engine(args.database_url)
    repo = ReplayRepository(engine)
    run = repo.get_run(args.run_id)
    if run is None:
        print(f"run {args.run_id!r} not found")
        return 1
    decisions = repo.decisions_for(args.run_id)
    experiences = ExperienceRepository(engine).by_run(args.run_id)
    print(
        f"{run['id']}  status={run['status']} mode={run['mode']} "
        f"{run['start_as_of']} -> {run['end_as_of']} label={run['label']!r}"
    )
    print(f"decisions: {len(decisions)}   experiences: {len(experiences)}")
    for d in decisions:
        fill = f"{d['fill_price']:.2f}" if d["fill_price"] is not None else "-"
        weight = f"{d['target_weight']:.4f}" if d["target_weight"] is not None else "-"
        print(
            f"  {d['as_of']} {d['symbol']:<6} {d['action']:<9} "
            f"w={weight} fill={fill}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "list":
        return _cmd_list(args)
    return _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
