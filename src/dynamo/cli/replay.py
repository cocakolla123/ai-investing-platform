"""Historical replay CLI (plan Phases 4+5) — manual only, NEVER scheduled.

Usage:
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode quant --run-label first-window
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode committee --run-label first-committee --max-usd 5.00
    python -m dynamo.cli.replay estimate --start 2026-08-03 --end 2026-08-14
    python -m dynamo.cli.replay list
    python -m dynamo.cli.replay show --run-id replay-abc123def456

Quant mode spends $0 (zero LLM calls anywhere in the path). Committee mode
is the ONLY token-spending code path in the platform's learning stack, and it
refuses to start without an explicit ``--max-usd`` hard cap (no default) and
an ``ANTHROPIC_API_KEY``. The ``estimate`` subcommand projects committee-mode
cost with ZERO API calls — run it first to pick a sane ``--max-usd``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-replay",
        description="Run/inspect historical replays (manual, never scheduled)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a replay over [start, end]")
    run.add_argument("--start", required=True, type=date.fromisoformat)
    run.add_argument("--end", required=True, type=date.fromisoformat)
    run.add_argument(
        "--mode",
        choices=("quant", "committee"),
        default="quant",
        help="quant = free, no LLM; committee = real deep dives, REQUIRES --max-usd",
    )
    run.add_argument("--run-label", required=True, help="human label for the run")
    run.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="HARD per-run dollar cap (required for --mode committee; no default)",
    )
    run.add_argument(
        "--deep-dives-per-day",
        type=int,
        default=4,
        help="committee mode: per-day deep-dive shortlist cap (default 4)",
    )
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

    est = sub.add_parser(
        "estimate",
        help="dry-run committee-mode cost projection — ZERO API calls, no spend",
    )
    est.add_argument("--start", required=True, type=date.fromisoformat)
    est.add_argument("--end", required=True, type=date.fromisoformat)
    est.add_argument("--deep-dives-per-day", type=int, default=4)

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
        return _cmd_run_committee(args)
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


def _cmd_run_committee(args: argparse.Namespace) -> int:
    """Committee replay: the only token-spending path, human-triggered, capped."""
    from dynamo.config.settings import Settings
    from dynamo.llm.anthropic_client import AnthropicClient
    from dynamo.llm.cost_meter import CostMeter, SqlAlchemyLedgerWriter
    from dynamo.llm.embeddings import LocalEmbedder
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.budget import BudgetLedgerTee, ReplayBudget
    from dynamo.replay.committee import CommitteeReplay
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import ReplayConfig

    settings = Settings()
    if not settings.anthropic_api_key:
        print("committee replay requires ANTHROPIC_API_KEY — refusing to start.")
        return 2

    engine = _engine(args.database_url)
    run_repo = ReplayRepository(engine)
    config = ReplayConfig(
        start=args.start,
        end=args.end,
        mode="committee",
        label=args.run_label,
        universe_tiers=tuple(args.universe_tier or (1, 3)),
        initial_equity=args.initial_equity,
        horizon_sessions=args.horizon_sessions,
        committee_deep_dives_per_day=args.deep_dives_per_day,
    )
    # Resume accounting: a re-run of the same config (same deterministic
    # run id) starts from the durable spent_usd, not from zero.
    prior = run_repo.get_run(config.run_id)
    prior_spent = float(prior["spent_usd"]) if prior is not None else 0.0
    budget = ReplayBudget(
        cap_usd=args.max_usd,
        run_repo=run_repo,
        run_id=config.run_id,
        prior_spent_usd=prior_spent,
    )
    # The live calendar-month CostMeter stays the OUTER bound, on REAL wall
    # clock; the tee feeds every actual metered dollar into the run budget.
    meter = CostMeter(
        BudgetLedgerTee(SqlAlchemyLedgerWriter(engine), budget),
        settings.claude_monthly_budget_usd,
    )
    replay = CommitteeReplay(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        research_model=AnthropicClient(settings, meter),
        budget=budget,
        run_repo=run_repo,
        experience_repo=ExperienceRepository(engine),
        config=config,
        tier_models={
            "screen": settings.model_screen,
            "analyst": settings.model_analyst,
            "arbiter": settings.model_arbiter,
        },
        embedder=LocalEmbedder(),
    )
    print(
        f"committee replay {config.run_id} ({config.start} -> {config.end}) "
        f"cap=${budget.cap_usd:.2f}"
        + (f" resuming from ${prior_spent:.2f} spent" if prior_spent else "")
    )
    result = replay.run(progress=print)
    print(
        "\n".join(
            [
                f"run:          {result.run_id} [{result.status}]",
                f"sessions:     {len(result.sessions)}",
                f"deep dives:   {len(result.deep_dives)} (-> replay_verdicts)",
                f"experiences:  {len(result.experiences)} resolved (source='replay')",
                f"spend:        ${result.spent_usd:.2f} of ${result.cap_usd:.2f} cap",
            ]
        )
    )
    return 0 if result.status == "completed" else 3


def _cmd_estimate(args: argparse.Namespace) -> int:
    """Dry-run cost projection: sessions x shortlist cap x per-deep-dive cost.

    ZERO API calls and zero spend: prompt sizes come from the on-disk prompt
    files, token counts from the conservative ~3 chars/token rule, prices
    from ``dynamo.llm.pricing``. Assumptions (all err HIGH — see
    ``replay.budget``): full shortlist every session, ~6k-char thin dossier
    context, full ``max_tokens`` output per stage, no cache and no batch
    discount (real runs batch at 50% off with 1h prompt caching).
    """
    from dynamo.config.settings import Settings
    from dynamo.replay.budget import (
        ASSUMED_CONTEXT_CHARS,
        estimate_deep_dive_cost,
        estimate_run_cost,
    )
    from dynamo.replay.runner import _sessions_between
    from dynamo.research.committee import _DEFAULT_MAX_TOKENS, STAGES
    from dynamo.research.prompt_loader import load_prompt

    settings = Settings()
    tier_models = {
        "screen": settings.model_screen,
        "analyst": settings.model_analyst,
        "arbiter": settings.model_arbiter,
    }
    stage_specs = {
        stage: (tier, len(load_prompt(prompt_name).system), _DEFAULT_MAX_TOKENS[stage])
        for stage, (prompt_name, _schema, tier) in STAGES.items()
    }
    per_dive = estimate_deep_dive_cost(
        tier_models, stage_specs, context_chars=ASSUMED_CONTEXT_CHARS
    )
    sessions = _sessions_between(args.start, args.end)
    total = estimate_run_cost(
        n_sessions=len(sessions),
        deep_dives_per_day=args.deep_dives_per_day,
        per_dive_usd=per_dive,
    )
    print(f"committee replay estimate {args.start} -> {args.end} (NO API calls made)")
    print(f"  sessions:            {len(sessions)}")
    print(f"  deep dives per day:  {args.deep_dives_per_day} (cap; real days may run fewer)")
    print(f"  per deep dive:       ${per_dive:.4f} (4 stages, conservative overestimate)")
    print(
        f"  models:              analyst={tier_models['analyst']} "
        f"arbiter={tier_models['arbiter']}"
    )
    print(
        f"  assumptions:         ~{ASSUMED_CONTEXT_CHARS} context chars, ~3 chars/token, "
        "full max_tokens output, no cache/batch discounts"
    )
    print(f"  projected total:     ${total:.2f}")
    print("  suggested --max-usd: at or below the projected total — the run-time")
    print("  projection uses the same conservative math, so real spend lands lower.")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from dynamo.persistence.repositories.replay import ReplayRepository

    repo = ReplayRepository(_engine(args.database_url))
    runs = repo.list_runs(limit=args.limit)
    if not runs:
        print("no replay runs")
        return 0
    for r in runs:
        spend = ""
        if r["budget_usd"] is not None:
            spend = f"  ${r['spent_usd']:.2f}/${r['budget_usd']:.2f}"
        print(
            f"{r['id']}  {r['status']:<16} {r['mode']:<9} "
            f"{r['start_as_of']} -> {r['end_as_of']}  label={r['label']!r}{spend}"
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
    verdicts = repo.verdicts_for(args.run_id)
    experiences = ExperienceRepository(engine).by_run(args.run_id)
    print(
        f"{run['id']}  status={run['status']} mode={run['mode']} "
        f"{run['start_as_of']} -> {run['end_as_of']} label={run['label']!r}"
    )
    if run["budget_usd"] is not None:
        print(f"budget: ${run['spent_usd']:.2f} spent of ${run['budget_usd']:.2f} cap")
    print(
        f"decisions: {len(decisions)}   verdicts: {len(verdicts)}   "
        f"experiences: {len(experiences)}"
    )
    for v in verdicts:
        print(
            f"  {v['as_of']} {v['symbol']:<6} {v['action']:<9} "
            f"conviction={v['conviction']:.2f} disagreement={v['disagreement_score']:.2f}"
        )
    for d in decisions:
        fill = f"{d['fill_price']:.2f}" if d["fill_price"] is not None else "-"
        weight = f"{d['target_weight']:.4f}" if d["target_weight"] is not None else "-"
        print(
            f"  {d['as_of']} {d['symbol']:<6} {d['action']:<9} "
            f"w={weight} fill={fill}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.mode == "committee" and args.max_usd is None:
            parser.error("--mode committee requires --max-usd (hard per-run cap, no default)")
        if args.max_usd is not None and args.max_usd <= 0:
            parser.error("--max-usd must be positive")
        return _cmd_run(args)
    if args.command == "estimate":
        return _cmd_estimate(args)
    if args.command == "list":
        return _cmd_list(args)
    return _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
