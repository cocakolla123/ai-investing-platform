"""Alpha-research rails CLI (ADR-0015) — pre-registered candidate auditions.

Usage:
    python -m dynamo.cli.alpha register --file hypothesis.yaml
    python -m dynamo.cli.alpha register --id vol_skew --title "..." \
        --rationale "..." --prediction "..." --trial-family price-vol
    python -m dynamo.cli.alpha audition --hypothesis-id vol_skew \
        --candidate-file candidates/vol_skew.py \
        --train-start 2024-01-01 --train-end 2025-12-31 [--symbols ...]
    python -m dynamo.cli.alpha propose --hypothesis-id vol_skew
    python -m dynamo.cli.alpha list

Manual only — NEVER scheduled, ZERO LLM calls, never auto-promotes. The
audition REFUSES candidates whose hypothesis was not registered in a prior
invocation (``registered_at`` strictly before this invocation's start).
Every audition increments the family's persistent trial counter and appends
one row to ``docs/alpha_research_log.md`` — pass or fail. ``propose`` only
emits a NEW_FEATURE ImprovementProposal (human-approved downstream) when the
stored verdict passed.

Exit codes: 0 = done (audition: verdict PASSED), 1 = refusal/error,
2 = audition completed with a FAILED verdict (recorded, counted, logged).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

DEFAULT_LOG_PATH = Path("docs/alpha_research_log.md")
HORIZON_SESSIONS = 20  # label horizon — mirrors quant_train


# --------------------------------------------------------------------------
# dependency seams (Protocols so tests inject fakes; main() wires the DB)
# --------------------------------------------------------------------------


class HypothesisStore(Protocol):
    def insert(self, h: Any) -> None: ...

    def get(self, hypothesis_id: str) -> Any: ...

    def list_all(self) -> list[Any]: ...

    def record_audit(
        self, hypothesis_id: str, status: Any, audit_results: dict[str, Any]
    ) -> None: ...

    def set_status(self, hypothesis_id: str, status: Any) -> None: ...


class TrialCounter(Protocol):
    def increment_trials(self, family: str) -> int: ...


class ProposalStore(Protocol):
    def insert(self, proposal: Any) -> None: ...


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-alpha",
        description="Alpha-research rails: register, audition, propose (manual, never scheduled)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="pre-register a hypothesis (before any testing)")
    reg.add_argument("--file", type=Path, default=None, help="hypothesis YAML/JSON file")
    reg.add_argument("--id", default=None, help="slug id (a-z0-9_-)")
    reg.add_argument("--title", default=None)
    reg.add_argument(
        "--rationale", default=None, help="WHY it should work — written before testing"
    )
    reg.add_argument("--prediction", default=None, help="expected sign/magnitude")
    reg.add_argument("--trial-family", default=None, help="trial-counter family key")
    reg.add_argument("--data-requirements", default="")
    reg.add_argument("--database-url", default=None)

    aud = sub.add_parser("audition", help="run the audition harness for a registered hypothesis")
    aud.add_argument("--hypothesis-id", required=True)
    aud.add_argument(
        "--candidate-file",
        required=True,
        type=Path,
        help="python file exposing CANDIDATE: FeatureSpec (see research/discovery/audition.py)",
    )
    aud.add_argument("--train-start", required=True, type=date.fromisoformat)
    aud.add_argument("--train-end", required=True, type=date.fromisoformat)
    aud.add_argument(
        "--symbols", nargs="+", default=None, help="explicit universe (default: tier-1/3)"
    )
    aud.add_argument("--n-permutations", type=int, default=None)
    aud.add_argument("--no-cache", action="store_true", help="bypass the panel feature cache")
    aud.add_argument("--database-url", default=None)

    prop = sub.add_parser("propose", help="emit a NEW_FEATURE proposal for a PASSED audition")
    prop.add_argument("--hypothesis-id", required=True)
    prop.add_argument("--database-url", default=None)

    lst = sub.add_parser("list", help="list registered hypotheses and their status")
    lst.add_argument("--database-url", default=None)
    return parser


def _jsonify(obj: Any) -> Any:
    """NaN/numpy-safe deep conversion so audit_results always stores clean JSON."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list | tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating | np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _load_hypothesis_file(path: Path) -> dict[str, Any]:
    """Parse a hypothesis YAML/JSON file into a flat field mapping."""
    import json

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "pyyaml is not importable — use a .json hypothesis file or CLI args"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"hypothesis file {path} must contain a mapping")
    return {str(k): v for k, v in data.items()}


# --------------------------------------------------------------------------
# commands (pure of the database given injected stores)
# --------------------------------------------------------------------------


def register_command(
    fields: dict[str, Any],
    hypotheses: HypothesisStore,
    *,
    now: datetime,
    out: Callable[[str], None] = print,
) -> int:
    from dynamo.research.discovery.registry import Hypothesis

    try:
        hypothesis = Hypothesis(
            id=str(fields.get("id") or ""),
            title=str(fields.get("title") or ""),
            rationale=str(fields.get("rationale") or ""),
            prediction=str(fields.get("prediction") or ""),
            trial_family=str(fields.get("trial_family") or ""),
            data_requirements=str(fields.get("data_requirements") or ""),
            registered_at=now,
        )
    except ValueError as exc:
        out(f"refused: invalid hypothesis — {exc}")
        return 1
    if hypotheses.get(hypothesis.id) is not None:
        out(
            f"refused: hypothesis {hypothesis.id!r} already exists — "
            "pre-registrations are immutable; register a new id"
        )
        return 1
    hypotheses.insert(hypothesis)
    out(
        f"registered hypothesis {hypothesis.id!r} (family {hypothesis.trial_family!r}) "
        f"at {now.isoformat()} — audition it in a LATER invocation"
    )
    return 0


def audition_command(
    *,
    hypothesis_id: str,
    candidate_file: Path,
    audition_start: datetime,
    hypotheses: HypothesisStore,
    trials: TrialCounter,
    panel_builder: Callable[[Any], Any],  # FeatureSpec -> AuditionPanel
    log_path: Path = DEFAULT_LOG_PATH,
    wf_config: Any = None,
    model_factory: Callable[[], Any] | None = None,
    n_permutations: int | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """The audition flow. ``audition_start`` MUST be captured at invocation
    start (main() does) — it is the instant the pre-registration rule checks
    ``registered_at`` against."""
    from dynamo.research.discovery.audition import (
        DEFAULT_N_PERMUTATIONS,
        LABEL_COLUMN,
        CandidateLoadError,
        append_research_log,
        decide_verdict,
        format_log_row,
        incremental_value_test,
        load_candidate,
        permutation_null_test,
        signal_audit,
    )
    from dynamo.research.discovery.registry import (
        HypothesisStatus,
        PreRegistrationError,
        ensure_auditionable,
    )

    hypothesis = hypotheses.get(hypothesis_id)
    if hypothesis is None:
        out(
            f"refused: no registered hypothesis {hypothesis_id!r} — pre-register "
            "first (python -m dynamo.cli.alpha register), then audition later"
        )
        return 1
    try:
        ensure_auditionable(hypothesis, audition_start)
    except PreRegistrationError as exc:
        out(f"refused: {exc}")
        return 1
    try:
        candidate = load_candidate(candidate_file)
    except CandidateLoadError as exc:
        out(f"refused: {exc}")
        return 1

    panel = panel_builder(candidate)
    if len(panel.sessions) == 0 or panel.base_features.empty:
        out("refused: audition panel is empty — check the window/universe/backfills")
        return 1
    if wf_config is None:
        from dynamo.cli.quant_train import PanelTooSmallError, walkforward_for

        try:
            wf_config = walkforward_for(len(panel.sessions))
        except PanelTooSmallError as exc:
            out(f"refused: {exc}")
            return 1
    if model_factory is None:
        from dynamo.research.quant.models import LightGBMModel

        model_factory = lambda: LightGBMModel("regression")  # noqa: E731

    out(
        f"audition {hypothesis_id!r}: candidate {candidate.name!r} "
        f"({candidate.group}), {len(panel.sessions)} sessions"
    )
    audit = signal_audit(panel)
    incremental = incremental_value_test(panel, wf_config=wf_config, model_factory=model_factory)
    # The trial is counted BEFORE the verdict: a look at OOS results is a
    # spent trial whatever the outcome — that is the discipline.
    family_trials = trials.increment_trials(hypothesis.trial_family)
    null = permutation_null_test(
        panel.candidate,
        panel.labels[LABEL_COLUMN],
        n_permutations=n_permutations or DEFAULT_N_PERMUTATIONS,
    )
    verdict = decide_verdict(incremental=incremental, null=null, family_trials=family_trials)
    status = HypothesisStatus.AUDITIONED if verdict.passed else HypothesisStatus.REJECTED
    audit_results = _jsonify(
        {
            "candidate": {
                "name": candidate.name,
                "group": candidate.group,
                "lookback_days": candidate.lookback_days,
                "file": str(candidate_file),
            },
            "window": {
                "n_sessions": len(panel.sessions),
                "start": panel.sessions[0].isoformat(),
                "end": panel.sessions[-1].isoformat(),
            },
            "signal_audit": audit,
            "incremental": incremental,
            "permutation_null": null,
            "family_trials": family_trials,
            "verdict": verdict.as_dict(),
            "audition_started_at": audition_start.isoformat(),
        }
    )
    hypotheses.record_audit(hypothesis_id, status, audit_results)
    append_research_log(
        log_path,
        format_log_row(
            day=audition_start.date(),
            hypothesis_id=hypothesis_id,
            family=hypothesis.trial_family,
            family_trials=family_trials,
            verdict=verdict,
        ),
    )

    def _fmt(v: Any, spec: str = "+.4f") -> str:
        return "n/a" if v is None else format(float(v), spec)

    out(f"  family {hypothesis.trial_family!r} trials: {family_trials}")
    out(
        f"  incremental OOS rank IC: {_fmt(incremental['ic_without'])} -> "
        f"{_fmt(incremental['ic_with'])} (delta {_fmt(incremental['ic_delta'])}, "
        f"{incremental['n_folds']} folds)"
    )
    out(
        f"  permutation null: real IC {_fmt(null['real_ic'])}, "
        f"percentile {_fmt(null['null_percentile'], '.1f')} "
        f"of {null['n_permutations']} shuffles; trials-adjusted bar "
        f"{verdict.expected_max_null_ic:+.4f}"
    )
    if audit.get("available"):
        out(f"  alphalens audit: ic_mean={audit.get('ic_mean')}")
    else:
        out(f"  alphalens audit: skipped ({audit.get('reason', 'unavailable')})")
    if verdict.passed:
        out("  VERDICT: PASSED — propose with `python -m dynamo.cli.alpha propose`")
        return 0
    out("  VERDICT: FAILED")
    for reason in verdict.reasons:
        out(f"    - {reason}")
    return 2


def propose_command(
    *,
    hypothesis_id: str,
    hypotheses: HypothesisStore,
    proposals: ProposalStore,
    now: datetime,
    out: Callable[[str], None] = print,
) -> int:
    from dynamo.domain.models.learning import ImprovementProposal, ProposalKind
    from dynamo.domain.models.research import Evidence, SourceType
    from dynamo.research.discovery.registry import HypothesisStatus

    hypothesis = hypotheses.get(hypothesis_id)
    if hypothesis is None:
        out(f"refused: no hypothesis {hypothesis_id!r}")
        return 1
    if hypothesis.status != HypothesisStatus.AUDITIONED:
        out(
            f"refused: hypothesis {hypothesis_id!r} is {hypothesis.status.value} — "
            "only a hypothesis whose audition PASSED can be proposed"
        )
        return 1
    results = hypothesis.audit_results or {}
    verdict = results.get("verdict") or {}
    if not verdict.get("passed"):
        out(
            f"refused: stored audition verdict for {hypothesis_id!r} did not pass "
            "the thresholds — no proposal"
        )
        return 1
    candidate = results.get("candidate") or {}
    feature_name = str(candidate.get("name") or hypothesis_id)
    proposal = ImprovementProposal(
        id=f"proposal-newfeature-{hypothesis_id}",
        created_at=now,
        kind=ProposalKind.NEW_FEATURE,
        target=f"research.quant.catalog:{feature_name}",
        evidence=(
            Evidence(
                claim=(
                    f"audition passed: IC delta {verdict.get('ic_delta')}, "
                    f"null percentile {verdict.get('null_percentile')}, "
                    f"family trials {verdict.get('family_trials')}"
                ),
                source_type=SourceType.JOURNAL,
                source_ref=f"hypotheses/{hypothesis_id}",
                as_of=now.date(),
            ),
        ),
        expected_effect=hypothesis.prediction,
        proposed_change={
            "add_feature": feature_name,
            "group": candidate.get("group"),
            "candidate_file": candidate.get("file"),
            "hypothesis_id": hypothesis_id,
            "audit_results": results,
        },
    )
    proposals.insert(proposal)
    hypotheses.set_status(hypothesis_id, HypothesisStatus.PROPOSED)
    out(
        f"proposed: {proposal.id} (kind=new_feature, target={proposal.target}) — "
        "OPEN for human review; nothing is applied automatically"
    )
    return 0


def list_command(hypotheses: HypothesisStore, *, out: Callable[[str], None] = print) -> int:
    rows = hypotheses.list_all()
    if not rows:
        out("no hypotheses registered")
        return 0
    out(f"{'id':<28} {'status':<12} {'family':<20} registered_at")
    for h in rows:
        out(f"{h.id:<28} {h.status.value:<12} {h.trial_family:<20} {h.registered_at.isoformat()}")
    return 0


# --------------------------------------------------------------------------
# composition (real database + gateway) — mirrors quant_train.main
# --------------------------------------------------------------------------


def _engine(database_url: str | None) -> Any:
    from dynamo.persistence.db import make_engine

    url = database_url or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("no database url: pass --database-url or set DATABASE_URL")
    return make_engine(url)


def _real_panel_builder(
    engine: Any,
    train_start: date,
    train_end: date,
    symbols: list[str] | None,
    use_cache: bool,
    out: Callable[[str], None],
) -> Callable[[Any], Any]:
    """Closure over the gateway/universe/bars wiring; runs on first call."""
    from dynamo.cli.quant_train import BENCHMARK_FALLBACK, label_sector_map
    from dynamo.replay.gateway import HistoricalDataGateway
    from dynamo.replay.runner import _sessions_between
    from dynamo.research.discovery.audition import build_audition_panel

    def build(candidate: Any) -> Any:
        gw_end = HistoricalDataGateway(engine, train_end)
        universe = [str(s) for s in symbols] if symbols else gw_end.universe((1, 3))
        if not universe:
            raise RuntimeError(f"universe is empty at {train_end}")
        sector_map = label_sector_map(gw_end.sectors(universe), universe)
        sessions = _sessions_between(train_start, train_end)
        label_bars = gw_end.bars(
            sorted({*universe, *sector_map.values(), BENCHMARK_FALLBACK}),
            (train_end - train_start).days + 60,
        )
        cache = None
        if use_cache:
            from dynamo.research.quant.panel_cache import PanelFeatureCache

            cache = PanelFeatureCache(Path("artifacts") / "panel_cache", universe)
            out(f"panel cache: {cache.directory} (bypass with --no-cache)")
        return build_audition_panel(
            candidate,
            lambda d: HistoricalDataGateway(engine, d).feature_inputs(universe),
            label_bars,
            sector_map,
            universe,
            sessions,
            horizon_sessions=HORIZON_SESSIONS,
            cache=cache,
            progress=lambda msg: out(msg),
        )

    return build


def main(argv: list[str] | None = None) -> int:
    # Captured BEFORE anything else: the instant the pre-registration rule
    # compares registered_at against — nothing registered by this invocation
    # can ever be auditioned by it.
    invocation_start = datetime.now(UTC)
    args = _build_parser().parse_args(argv)

    from dynamo.persistence.repositories.discovery import HypothesisRepository

    engine = _engine(args.database_url)
    hypotheses = HypothesisRepository(engine)

    if args.command == "register":
        if args.file is not None:
            fields = _load_hypothesis_file(args.file)
        else:
            fields = {
                "id": args.id,
                "title": args.title,
                "rationale": args.rationale,
                "prediction": args.prediction,
                "trial_family": args.trial_family,
                "data_requirements": args.data_requirements,
            }
        return register_command(fields, hypotheses, now=invocation_start)

    if args.command == "audition":
        if args.train_end <= args.train_start:
            print("--train-end must be after --train-start")
            return 1
        from dynamo.persistence.repositories.learning import ModelRegistryRepository

        return audition_command(
            hypothesis_id=args.hypothesis_id,
            candidate_file=args.candidate_file,
            audition_start=invocation_start,
            hypotheses=hypotheses,
            trials=ModelRegistryRepository(engine),
            panel_builder=_real_panel_builder(
                engine,
                args.train_start,
                args.train_end,
                args.symbols,
                not args.no_cache,
                lambda msg: print(msg, flush=True),
            ),
            n_permutations=args.n_permutations,
        )

    if args.command == "propose":
        from dynamo.persistence.repositories.learning import ProposalRepository

        return propose_command(
            hypothesis_id=args.hypothesis_id,
            hypotheses=hypotheses,
            proposals=ProposalRepository(engine),
            now=invocation_start,
        )

    return list_command(hypotheses)


if __name__ == "__main__":
    sys.exit(main())
