"""Historical replay CLI (plan Phases 4+5) — manual only, NEVER scheduled.

Usage:
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode quant --run-label first-window
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode committee --run-label first-committee --max-usd 5.00
    python -m dynamo.cli.replay run --start 2026-08-03 --end 2026-08-14 \
        --mode portfolio --source-run-id replay-abc123def456 \
        --run-label 1k --initial-equity 1000 --min-order-notional 25
    python -m dynamo.cli.replay estimate --start 2026-08-03 --end 2026-08-14
    python -m dynamo.cli.replay list
    python -m dynamo.cli.replay show --run-id replay-abc123def456
    python -m dynamo.cli.replay report --run-id replay-abc123def456
    python -m dynamo.cli.replay compare --run-ids replay-a replay-b \
        --png artifacts/backtests/compare.png

Quant and portfolio modes spend $0 (zero LLM calls anywhere in their path —
portfolio mode replays a committee run's PERSISTED verdicts and is routed
through ``_cmd_run``, which never imports an LLM module). Committee mode
is the ONLY token-spending code path in the platform's learning stack, and it
refuses to start without an explicit ``--max-usd`` hard cap (no default) and
an ``ANTHROPIC_API_KEY``. The ``estimate`` subcommand projects committee-mode
cost with ZERO API calls — run it first to pick a sane ``--max-usd``.
"""

from __future__ import annotations

import argparse
import os
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
        choices=("quant", "portfolio", "committee"),
        default="quant",
        help=(
            "quant = free, no LLM; portfolio = free verdict-reuse over a "
            "committee run (REQUIRES --source-run-id, no LLM); committee = "
            "real deep dives, REQUIRES --max-usd"
        ),
    )
    run.add_argument(
        "--source-run-id",
        default="",
        help=(
            "portfolio mode: the completed committee run whose persisted "
            "replay_verdicts this run replays (required for --mode portfolio)"
        ),
    )
    run.add_argument(
        "--whole-shares",
        action="store_true",
        help=(
            "floor all desired share quantities to whole shares (crypto pairs "
            "exempt — fractional native); part of config_hash/run id"
        ),
    )
    run.add_argument(
        "--min-order-notional",
        type=float,
        default=0.0,
        help="skip orders below this dollar value entirely (default 0 = off)",
    )
    run.add_argument(
        "--commission-per-order",
        type=float,
        default=0.0,
        help="flat dollar commission added per executed order (default 0)",
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
    run.add_argument(
        "--quant-model-version",
        default="",
        help=(
            "registered quant bundle (dynamo.cli.quant_train --label) to score "
            "with; unset = baseline-v0 heuristic (honest abstention). Part of "
            "config_hash, so it changes the run id."
        ),
    )
    run.add_argument(
        "--quant-score-max-age-days",
        type=int,
        default=5,
        help="max staleness of persisted quant scores before the bundle rescoring (default 5)",
    )
    run.add_argument(
        "--code-version",
        default="",
        help="code version stamp recorded in the run config (freeze across a campaign)",
    )
    run.add_argument("--database-url", default=None)
    run.add_argument(
        "--skip-preflight",
        action="store_true",
        help=(
            "skip the data-readiness preflight; the skip is recorded in the "
            "run metadata (config_json), never in ReplayConfig"
        ),
    )
    run.add_argument(
        "--acknowledge-model-knowledge",
        action="store_true",
        help=(
            "committee mode: proceed even though the window starts within "
            "the safety margin of a committee model's documented "
            "training-data cutoff (recorded in config_json; the leakage "
            "audit flags it)"
        ),
    )
    run.add_argument(
        "--no-macro-digest",
        action="store_true",
        help=(
            "committee mode: do NOT generate a replay macro digest on days "
            "without a persisted one (part of config_hash/run id)"
        ),
    )
    run.add_argument(
        "--no-within-run-learning",
        action="store_true",
        help=(
            "committee mode: exclude THIS run's own resolved replay "
            "experiences from prior-case retrieval (part of config_hash)"
        ),
    )

    est = sub.add_parser(
        "estimate",
        help="dry-run committee-mode cost projection — ZERO API calls, no spend",
    )
    est.add_argument("--start", required=True, type=date.fromisoformat)
    est.add_argument("--end", required=True, type=date.fromisoformat)
    est.add_argument("--deep-dives-per-day", type=int, default=4)
    est.add_argument(
        "--no-macro-digest",
        action="store_true",
        help="exclude the per-session replay macro-digest generation cost",
    )

    aud = sub.add_parser(
        "audit",
        help=(
            "post-run leakage audit over persisted contexts/verdicts -> "
            "artifacts/backtests/<run_id>/leakage_audit.{json,md}; exit 0 "
            "clean, 1 flagged (informational — a human reviews)"
        ),
    )
    aud.add_argument("--run-id", required=True)
    aud.add_argument("--database-url", default=None)
    aud.add_argument("--out-dir", default=None)

    rep = sub.add_parser(
        "report",
        help=(
            "build report.json + tearsheet for a finished quant/portfolio run "
            "-> artifacts/backtests/<run_id>/; exit 2 unknown run"
        ),
    )
    rep.add_argument("--run-id", required=True)
    rep.add_argument("--database-url", default=None)
    rep.add_argument(
        "--artifact-dir",
        default=None,
        help="artifact root (default: Settings().artifact_dir)",
    )

    cmp_ = sub.add_parser(
        "compare",
        help=(
            "metrics table across finished runs (+ optional overlaid "
            "equity-curve png); exit 2 on any unknown run id"
        ),
    )
    cmp_.add_argument("--run-ids", required=True, nargs="+")
    cmp_.add_argument(
        "--png",
        default=None,
        help=(
            "write an overlaid equity-curve png here (normalized to 100 at "
            "start for cross-bankroll comparability; needs matplotlib — "
            "skipped with a note when absent)"
        ),
    )
    cmp_.add_argument("--database-url", default=None)

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


def _bundle_factory(engine: Any) -> Any:
    """Composition-layer bundle loading for ``--quant-model-version``: the
    registry row's ``artifact_path`` (recorded by ``dynamo.cli.quant_train``)
    names the bundle manifest, so artifacts move with the registry; without a
    row the loader falls back to the trainer's default artifact directory.
    CANDIDATE status is fine here — replay opts into candidates explicitly;
    only the nightly production asset requires promoted status."""

    def load(version: str) -> Any:
        from pathlib import Path

        from sqlalchemy import select

        from dynamo.persistence import schema
        from dynamo.research.quant.bundle import bundle_from_version

        with engine.connect() as cx:
            row = cx.execute(
                select(schema.model_versions.c.artifact_path).where(
                    schema.model_versions.c.id == version
                )
            ).first()
        artifact_dir = (
            Path(str(row.artifact_path)).parent
            if row is not None and row.artifact_path
            else None
        )
        return bundle_from_version(version, artifact_dir)

    return load


def _preflight_extras(engine: Any, config: Any, *, skip: bool) -> dict[str, Any] | None:
    """The ``{"preflight": ...}`` config_json extras, or ``None`` on failure.

    Runs :func:`dynamo.replay.preflight.run_preflight` before EVERY run.
    ``skip=True`` records the skip in run metadata instead of checking (kept
    out of ``ReplayConfig`` so the deterministic run id never changes).
    """
    from dynamo.replay.preflight import PreflightError, run_preflight

    if skip:
        print("preflight SKIPPED (--skip-preflight) — skip recorded in run metadata")
        return {"preflight": {"skipped": True}}
    try:
        report = run_preflight(engine, config)
    except PreflightError as exc:
        print("preflight FAILED — refusing to run:")
        for check in exc.report.checks:
            print(f"  [{check.status.upper():7}] {check.name}: {check.detail}")
        print("fix the data above (backfills), or re-run with --skip-preflight.")
        return None
    for check in report.warnings:
        print(f"preflight warning: {check.name}: {check.detail}")
    print("preflight OK")
    return {"preflight": {"skipped": False, **report.as_dict()}}


def _cmd_run(args: argparse.Namespace) -> int:
    # TOKEN FIREWALL: only committee mode ever reaches the LLM wiring —
    # quant and portfolio runs go through this function, which imports no
    # LLM module and constructs no client. Portfolio mode replays the source
    # committee run's PERSISTED verdicts (ReplayRepository.verdicts_for).
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
        quant_score_max_age_days=args.quant_score_max_age_days,
        quant_model_version=args.quant_model_version,
        source_run_id=args.source_run_id,
        whole_shares=args.whole_shares,
        min_order_notional=args.min_order_notional,
        commission_per_order=args.commission_per_order,
        generate_macro_digest=not args.no_macro_digest,
        within_run_learning=not args.no_within_run_learning,
        code_version=args.code_version,
    )
    extras = _preflight_extras(engine, config, skip=args.skip_preflight)
    if extras is None:
        return 1
    replay_repo = ReplayRepository(engine)
    runner = ReplayRunner(
        gateway_factory=lambda d: HistoricalDataGateway(engine, d),
        experience_repo=ExperienceRepository(engine),
        run_repo=replay_repo,
        config=config,
        run_config_extras=extras,
        bundle_factory=_bundle_factory(engine),
        verdict_source=replay_repo.verdicts_for if args.mode == "portfolio" else None,
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
    from dynamo.replay.committee import CommitteeReplay, ModelKnowledgeError
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
        quant_score_max_age_days=args.quant_score_max_age_days,
        quant_model_version=args.quant_model_version,
        generate_macro_digest=not args.no_macro_digest,
        within_run_learning=not args.no_within_run_learning,
        code_version=args.code_version,
    )
    extras = _preflight_extras(engine, config, skip=args.skip_preflight)
    if extras is None:
        return 1
    # Network isolation (Phase 4): the embedder must never download model
    # weights mid-run — offline env vars are already set in main(); the live
    # backend actually loaded is recorded into config_json for the audit.
    embedder = LocalEmbedder()
    extras["embedder_backend"] = embedder.backend
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
    try:
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
            embedder=embedder,
            run_config_extras=extras,
            bundle_factory=_bundle_factory(engine),
            acknowledge_model_knowledge=args.acknowledge_model_knowledge,
            cutoff_margin_days=settings.replay_cutoff_margin_days,
        )
    except ModelKnowledgeError as exc:
        print(f"model knowledge gate BLOCKED the run ($0 spent):\n  {exc}")
        return 1
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
        ASSUMED_MACRO_CONTEXT_CHARS,
        estimate_deep_dive_cost,
        estimate_request_cost,
        estimate_run_cost,
    )
    from dynamo.replay.committee import (
        MACRO_DIGEST_MAX_TOKENS,
        model_knowledge_gate,
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
    per_session = 0.0
    if not args.no_macro_digest:
        per_session = estimate_request_cost(
            tier_models["analyst"],
            prompt_chars=len(load_prompt("macro_digest").system)
            + ASSUMED_MACRO_CONTEXT_CHARS,
            max_tokens=MACRO_DIGEST_MAX_TOKENS,
        )
    total = estimate_run_cost(
        n_sessions=len(sessions),
        deep_dives_per_day=args.deep_dives_per_day,
        per_dive_usd=per_dive,
        per_session_usd=per_session,
    )
    print(f"committee replay estimate {args.start} -> {args.end} (NO API calls made)")
    print(f"  sessions:            {len(sessions)}")
    print(f"  deep dives per day:  {args.deep_dives_per_day} (cap; real days may run fewer)")
    print(f"  per deep dive:       ${per_dive:.4f} (4 stages, conservative overestimate)")
    if per_session:
        print(
            f"  macro digest/day:    ${per_session:.4f} "
            "(generated on days without a persisted digest; --no-macro-digest to drop)"
        )
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
    # Model knowledge gate verdict — the same check CommitteeReplay enforces,
    # surfaced here BEFORE any dollar is at stake.
    gate = model_knowledge_gate(
        args.start, tier_models, settings.replay_cutoff_margin_days
    )
    print("  model knowledge gate (documented training-data cutoffs):")
    for model_id, cutoff in sorted(gate["cutoffs"].items()):
        marker = "BLOCKS" if model_id in gate["blocking_models"] else "ok"
        print(f"    {model_id}: cutoff {cutoff} [{marker}]")
    if gate["blocked"]:
        print(
            f"  gate verdict:        BLOCKED — window start {args.start} is within "
            f"{gate['margin_days']}d of a model cutoff; the run will refuse unless "
            "--acknowledge-model-knowledge is passed (recorded + audited)."
        )
    else:
        print(f"  gate verdict:        OK (margin {gate['margin_days']}d)")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Post-run leakage audit: exit 0 clean, 1 flagged, 2 unknown run."""
    from pathlib import Path

    from dynamo.config.settings import Settings
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.audit import audit_run, write_audit_artifacts

    engine = _engine(args.database_url)
    repo = ReplayRepository(engine)
    run = repo.get_run(args.run_id)
    if run is None:
        print(f"run {args.run_id!r} not found")
        return 2
    contexts = repo.contexts_for(args.run_id)
    verdicts = repo.verdicts_for(args.run_id)
    report = audit_run(run, contexts, verdicts)
    out_dir = Path(
        args.out_dir or Path(Settings().artifact_dir) / "backtests" / args.run_id
    )
    json_path, md_path = write_audit_artifacts(report, out_dir)
    for name, cat in sorted(report.categories.items()):
        print(
            f"  [{cat.status:7}] {name}: {cat.detail}"
            + (f" ({len(cat.findings)} finding(s))" if cat.findings else "")
        )
    print(f"audit artifacts: {json_path} {md_path}")
    if report.passed:
        print(f"leakage audit PASS for {args.run_id}")
        return 0
    print(
        f"leakage audit FLAGGED findings for {args.run_id} — informational; "
        "review the artifacts above."
    )
    return 1


def _benchmark_closes(engine: Any, run: dict[str, Any]) -> Any:
    """SPY closes over the run window, read through a gateway fixed at the
    run's END date — bounded like every replay read (nothing post-window)."""
    import pandas as pd

    from dynamo.replay.gateway import HistoricalDataGateway

    start, end = run["start_as_of"], run["end_as_of"]
    gw = HistoricalDataGateway(engine, end)
    bars = gw.bars(["SPY"], (end - start).days + 10)
    if bars.empty:
        return None
    from typing import cast

    spy = cast("pd.DataFrame", bars[bars["symbol"] == "SPY"]).sort_values("ts")
    if spy.empty:
        return None
    return pd.Series(
        spy["close"].astype(float).to_numpy(),
        index=[pd.Timestamp(t).date() for t in spy["ts"]],
        name="SPY",
    )


def _cmd_report(args: argparse.Namespace) -> int:
    """Report a finished quant/portfolio run: exit 0 ok, 1 no curve, 2 unknown."""
    from pathlib import Path

    from dynamo.config.settings import Settings
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.report import ReplayReportError, UnknownRunError, replay_report
    from dynamo.replay.report import total_costs as fills_total_costs

    engine = _engine(args.database_url)
    repo = ReplayRepository(engine)
    run = repo.get_run(args.run_id)
    if run is None:
        print(f"run {args.run_id!r} not found")
        return 2
    artifact_dir = Path(args.artifact_dir or Settings().artifact_dir)
    try:
        report = replay_report(
            args.run_id,
            repo,
            artifact_dir,
            benchmark_closes=_benchmark_closes(engine, run),
        )
    except UnknownRunError as exc:  # pragma: no cover - guarded above
        print(str(exc))
        return 2
    except ReplayReportError as exc:
        print(str(exc))
        return 1
    costs = fills_total_costs(repo.fills_for(args.run_id))
    initial = float(run["config_json"].get("initial_equity") or 0.0)
    m = report.metrics
    print(f"report for {args.run_id} -> {artifact_dir / 'backtests' / args.run_id}")
    print(f"  sessions:         {m.n_sessions} ({report.window_start} -> {report.window_end})")
    print(f"  total return:     {m.total_return * 100.0:+.2f}%")
    print(f"  sharpe:           {m.sharpe:.2f}  (deflated: {report.deflated_sharpe:.2f})")
    print(f"  max drawdown:     {m.max_drawdown * 100.0:.2f}%")
    print(f"  turnover 1-way:   {report.behavior.one_way_turnover_total:.2f}")
    drag = f" ({costs / initial * 100.0:.2f}% of initial equity)" if initial > 0 else ""
    print(f"  cost drag:        ${costs:,.2f}{drag}")
    if report.benchmark is not None:
        print(
            f"  benchmark excess: {report.benchmark.excess_total_return * 100.0:+.2f}% "
            f"vs {report.benchmark.benchmark}"
        )
    else:
        print("  benchmark excess: n/a (no SPY bars over the window)")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Cross-run metrics table (+ optional normalized-curve png); exit 2 on
    any unknown run id, 1 when a run has no persisted curve."""
    from dynamo.backtest.metrics import metrics_block
    from dynamo.persistence.repositories.replay import ReplayRepository
    from dynamo.replay.report import (
        derive_turnover_total,
        equity_series,
        realized_pnls,
        total_costs,
    )

    engine = _engine(args.database_url)
    repo = ReplayRepository(engine)
    rows: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}
    for run_id in args.run_ids:
        run = repo.get_run(run_id)
        if run is None:
            print(f"run {run_id!r} not found")
            return 2
        equity_rows = repo.equity_for(run_id)
        if not equity_rows:
            print(
                f"run {run_id!r} has no persisted equity curve "
                f"(mode={run['mode']!r}) — nothing to compare"
            )
            return 1
        fills = repo.fills_for(run_id)
        curve = equity_series(equity_rows)
        curves[run_id] = curve
        block = metrics_block(curve)
        pnls = [p for _, _, p in realized_pnls(fills)]
        rows.append(
            {
                "run_id": run_id,
                "label": str(run["label"]),
                "initial": float(run["config_json"].get("initial_equity") or curve.iloc[0]),
                "final": float(curve.iloc[-1]),
                "return_pct": block.total_return * 100.0,
                "sharpe": block.sharpe,
                "sortino": block.sortino,
                "max_dd_pct": block.max_drawdown * 100.0,
                "turnover": derive_turnover_total(equity_rows, fills),
                "costs": total_costs(fills),
                "win_rate_pct": (
                    100.0 * sum(1 for p in pnls if p > 0.0) / len(pnls) if pnls else 0.0
                ),
            }
        )
    header = (
        f"{'run':<22} {'label':<14} {'initial':>12} {'final':>12} {'return':>8} "
        f"{'sharpe':>7} {'sortino':>8} {'maxDD':>7} {'turnover':>9} "
        f"{'costs':>10} {'win%':>6}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['run_id']:<22} {r['label'][:14]:<14} {r['initial']:>12,.2f} "
            f"{r['final']:>12,.2f} {r['return_pct']:>+7.2f}% {r['sharpe']:>7.2f} "
            f"{r['sortino']:>8.2f} {r['max_dd_pct']:>6.2f}% {r['turnover']:>9.2f} "
            f"{r['costs']:>10,.2f} {r['win_rate_pct']:>5.1f}%"
        )
    if args.png:
        _write_compare_png(curves, {r["run_id"]: r["label"] for r in rows}, args.png)
    return 0


def _write_compare_png(
    curves: dict[str, Any], labels: dict[str, str], path: str
) -> None:
    """Overlaid equity curves normalized to 100 at start (cross-bankroll
    comparability). matplotlib is optional — absent is a note, not an error
    (guarded importlib load, mirroring the quantstats tearsheet pattern)."""
    import importlib

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        print(f"png skipped: matplotlib not installed (wanted {path})")
        return
    from pathlib import Path

    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, curve in curves.items():
        normalized = 100.0 * curve / float(curve.iloc[0])
        label = labels.get(run_id) or run_id
        ax.plot(list(normalized.index), normalized.to_numpy(), label=f"{label} ({run_id})")
    ax.set_title("Replay equity curves (normalized to 100)")
    ax.set_ylabel("equity (start = 100)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


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
    # Network isolation (backtest-campaign Phase 4): forbid Hugging Face hub
    # traffic BEFORE any embedder import — a cold sentence-transformers cache
    # must fail into the deterministic hash fallback (recorded in
    # config_json), never fetch weights mid-replay.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.mode == "committee" and args.max_usd is None:
            parser.error("--mode committee requires --max-usd (hard per-run cap, no default)")
        if args.max_usd is not None and args.max_usd <= 0:
            parser.error("--max-usd must be positive")
        if args.mode == "portfolio" and not args.source_run_id:
            parser.error(
                "--mode portfolio requires --source-run-id (a completed committee run)"
            )
        return _cmd_run(args)
    if args.command == "estimate":
        return _cmd_estimate(args)
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "list":
        return _cmd_list(args)
    return _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
