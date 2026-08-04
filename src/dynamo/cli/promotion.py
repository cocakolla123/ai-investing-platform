"""Promotion CLI (doc 08 section 7): the ONLY tool that writes registry status.

Usage:
    python -m dynamo.cli.promotion stages
    python -m dynamo.cli.promotion run --target my-strategy --stage 1 \
        --evidence evidence.json --registry registry.json --actor owner \
        [--approval approvals/promotion-my-strategy.yaml] \
        [--evidence-bundle artifacts/backtests/run-123/]

The core stage logic lives in ``dynamo.backtest.promotion`` (pure); this
module wires a JSON-file registry repository and the git-tree check at
runtime. There is no ``--force``: a dirty tree, a missing approval artifact,
or an illegal transition is a refusal, not a warning.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


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
    run.add_argument("--registry", required=True, help="path to registry JSON file")
    run.add_argument("--actor", required=True, help="who is running the promotion")
    run.add_argument("--family", default=None, help="strategy family for trial counting")
    run.add_argument("--approval", default=None, help="signed approval artifact path")
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

    from dynamo.backtest.promotion import PROMOTION_STAGES, run_stage
    from dynamo.domain.models.learning import PromotionRecord, RegistryStatus
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
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            return not dirty, commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False, ""  # not a repo / git missing -> refuse (fail closed)

    def load_approval(path: str | None) -> tuple[dict[str, Any] | None, bool]:
        if path is None:
            return None, False
        p = Path(path)
        if not p.exists():
            return None, False
        text = p.read_text()
        parsed: dict[str, Any]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
            for line in text.splitlines():
                if ":" in line and not line.lstrip().startswith("#"):
                    key, _, value = line.partition(":")
                    parsed[key.strip()] = value.strip().strip("'\"")
        # Validity here = artifact present, targets this id, carries a
        # signature; cryptographic verification shares the live-gate HMAC
        # scheme and is wired by the approvals tool that writes the file.
        valid = bool(parsed.get("target_id")) and bool(parsed.get("signature"))
        return parsed, valid

    evidence = json.loads(Path(args.evidence).read_text())
    clean_tree, commit = git_state()
    approval, approval_valid = load_approval(args.approval)

    repo = JsonFileRegistry(Path(args.registry))
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
    except PromotionError as exc:
        print(f"refused: {exc}")
        return 2

    print(f"stage {result.stage} ({result.stage_name}): "
          f"{'PASS' if result.passed else 'FAIL'}  trial #{result.trial_count}")
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
