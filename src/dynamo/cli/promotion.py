"""Promotion CLI (doc 08 section 7): the ONLY tool that writes registry status.

Usage:
    python -m dynamo.cli.promotion stages
    python -m dynamo.cli.promotion run --target my-strategy --stage 1 \
        --evidence evidence.json --registry registry.json --actor owner \
        [--approval approvals/promotion-my-strategy.yaml] \
        [--evidence-bundle artifacts/backtests/run-123/]

Registry backends (plan Phase 3 — registry adoption):

- **Database (default when DATABASE_URL is set, or ``--registry-db``)**: the
  real ``model_versions`` / ``promotion_records`` / ``trial_counter`` tables
  via ``ModelRegistryRepository`` — the registry the meta-model trainer
  writes candidates into.
- **JSON file (``--registry <path>``)**: the original file-backed registry,
  kept for backward compatibility; an explicit ``--registry`` always wins.

The core stage logic lives in ``dynamo.backtest.promotion`` (pure); this
module wires a registry repository and the git-tree check at runtime. There
is no ``--force``: a dirty tree, a missing approval artifact, or an illegal
transition is a refusal, not a warning.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


class DbRegistry:
    """``RegistryRepository`` protocol over the real database registry.

    Thin adapter: reads via ``ModelRegistryRepository`` (unknown targets read
    as ``candidate``, mirroring the JSON registry); every status mutation
    goes through ``apply_promotion`` — the one legal write path.
    """

    def __init__(self, engine: Any) -> None:
        from dynamo.persistence.repositories.learning import ModelRegistryRepository

        self._repo = ModelRegistryRepository(engine)

    def current_status(self, target_id: str) -> Any:
        from dynamo.domain.models.learning import RegistryStatus

        return self._repo.status_of(target_id) or RegistryStatus.CANDIDATE

    def apply(self, record: Any) -> None:
        self._repo.apply_promotion(record)

    def trial_count(self, family: str) -> int:
        return self._repo.trial_count(family)

    def increment_trial(self, family: str) -> int:
        return self._repo.increment_trials(family)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-promote", description="Model/strategy promotion pipeline tool"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stages", help="list stage gates and their pass criteria")

    run = sub.add_parser("run", help="evaluate a stage gate for a target")
    run.add_argument("--target", required=True, help="registry target id")
    run.add_argument("--stage", required=True, type=int, help="stage number (1-7)")
    run.add_argument("--evidence", required=True, help="path to evidence JSON")
    run.add_argument(
        "--registry",
        default=None,
        help="path to a JSON registry file (backward compat; overrides DB mode)",
    )
    run.add_argument(
        "--registry-db",
        action="store_true",
        help="use the database registry via DATABASE_URL (default when the env var is set)",
    )
    run.add_argument("--actor", required=True, help="who is running the promotion")
    run.add_argument("--family", default=None, help="strategy family for trial counting")
    run.add_argument(
        "--approval",
        default=None,
        help=(
            "HMAC-signed approval artifact (write one with "
            "`python -m dynamo.cli.approve sign --kind promotion --target <id> ...`); "
            "verified against $DYNAMO_APPROVAL_KEY by the same routine as the live gate"
        ),
    )
    run.add_argument("--evidence-bundle", default=None, help="evidence bundle ref")
    run.add_argument(
        "--allow-dirty",
        action="store_true",
        help=argparse.SUPPRESS,  # intentionally unimplemented: there is no --force
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Lazy imports: the CLI is the composition layer; keeping them inside
    # main() keeps module import side-effect free and the core pure.
    import json
    import subprocess
    from datetime import UTC, datetime
    from pathlib import Path

    import yaml

    from dynamo.backtest.promotion import PROMOTION_STAGES, run_stage
    from dynamo.domain.models.learning import PromotionRecord, RegistryStatus
    from dynamo.execution import gate as live_gate
    from dynamo.learning.promotion import PromotionError

    args = _build_parser().parse_args(argv)

    if args.command == "stages":
        for gate in PROMOTION_STAGES:
            transition = (
                f"{gate.transition[0].value} -> {gate.transition[1].value}"
                if gate.transition
                else "(evidence only)"
            )
            print(f"stage {gate.stage}: {gate.name}  {transition}")
            for c in gate.criteria:
                print(f"    - {c.key} {c.op} {c.threshold!r}: {c.description}")
            if gate.requires_approval:
                print("    - signed human-approval artifact REQUIRED (no --force)")
                print("      write one: python -m dynamo.cli.approve sign --kind promotion \\")
                print("          --target <id> --expiry YYYY-MM-DD --approver <you>")
        return 0

    if getattr(args, "allow_dirty", False):
        print("refused: there is no dirty-tree override (doc 08 section 8)")
        return 2

    class JsonFileRegistry:
        """Minimal file-backed repository; only ``apply`` mutates status."""

        def __init__(self, path: Path) -> None:
            self.path = path
            if path.exists():
                self._data: dict[str, Any] = json.loads(path.read_text())
            else:
                self._data = {"status": {}, "trials": {}, "records": []}

        def _save(self) -> None:
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

        def current_status(self, target_id: str) -> RegistryStatus:
            raw = self._data["status"].get(target_id, RegistryStatus.CANDIDATE.value)
            return RegistryStatus(raw)

        def apply(self, record: PromotionRecord) -> None:
            self._data["status"][record.target_id] = record.to_status.value
            self._data["records"].append(json.loads(record.model_dump_json()))
            self._save()

        def trial_count(self, family: str) -> int:
            return int(self._data["trials"].get(family, 0))

        def increment_trial(self, family: str) -> int:
            n = self.trial_count(family) + 1
            self._data["trials"][family] = n
            self._save()
            return n

    def git_state() -> tuple[bool, str]:
        try:
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return not dirty, commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, ""  # not a repo / git missing -> refuse (fail closed)

    def load_promotion_approval(
        path: str | None, target_id: str
    ) -> tuple[dict[str, Any] | None, bool, str]:
        """Parse + cryptographically verify the approval artifact.

        Same artifact format, same ``hmac.compare_digest`` check and same
        refusal reasons as the live gate — ``gate.check_approval`` is the one
        verification routine (doc 08 §7 stage 7, doc 13 §4 risk #3). A file
        that merely *contains* a ``signature:`` key does not pass; the HMAC
        must match ``$DYNAMO_APPROVAL_KEY`` over the canonical body.

        Returns ``(body, valid, detail)``; ``detail`` is the operator-facing
        refusal text and is empty on success.
        """
        if path is None:
            return None, False, ""
        p = Path(path)
        if not p.exists():
            return None, False, f"no such approval artifact: {p}"
        try:
            artifact = live_gate.load_approval(p)
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            return None, False, f"approval artifact {p} is unreadable: {exc}"
        result = live_gate.check_approval(
            artifact,
            key=live_gate.approval_key_from_env(),
            today=datetime.now(UTC).date(),
            scope_target=target_id,
        )
        body = dict(artifact.body)
        if not result.permitted:
            return body, False, f"{result.reason.value}: {result.detail}"
        return body, True, ""

    evidence = json.loads(Path(args.evidence).read_text())
    clean_tree, commit = git_state()
    approval, approval_valid, approval_detail = load_promotion_approval(args.approval, args.target)
    if args.approval is not None and not approval_valid:
        # Loud and instructive: an unverifiable approval is a refusal, never a
        # downgrade to "unapproved" that a later stage might paper over.
        print(f"refused: approval artifact did not verify — {approval_detail}")
        return 2

    # Registry selection: explicit --registry <json> wins (backward compat);
    # otherwise DB mode via DATABASE_URL (--registry-db forces it and refuses
    # loudly when the URL is missing rather than silently using a file).
    import os

    repo: Any
    if args.registry is not None:
        repo = JsonFileRegistry(Path(args.registry))
    else:
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            print(
                "refused: no registry backend — pass --registry <json> "
                "or set DATABASE_URL (see --registry-db)"
            )
            return 2
        from dynamo.persistence.db import make_engine

        repo = DbRegistry(make_engine(database_url))
    try:
        result = run_stage(
            args.target,
            args.stage,
            evidence,
            repo=repo,
            actor=args.actor,
            at=datetime.now(UTC),
            clean_tree=clean_tree,
            strategy_family=args.family,
            approval=approval,
            approval_valid=approval_valid,
            evidence_bundle_ref=args.evidence_bundle,
            git_commit=commit,
            config_hash=str(evidence.get("config_hash", "")),
        )
    except (PromotionError, LookupError, ValueError) as exc:
        # LookupError/ValueError surface DB-registry refusals (unknown target,
        # stale record) the same way stage-gate refusals print.
        print(f"refused: {exc}")
        return 2

    print(
        f"stage {result.stage} ({result.stage_name}): "
        f"{'PASS' if result.passed else 'FAIL'}  trial #{result.trial_count}"
    )
    for line in result.details:
        print(f"  {line}")
    if result.record is not None:
        print(
            f"transition written: {result.record.from_status.value} -> "
            f"{result.record.to_status.value} by {result.record.actor}"
        )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
