"""Backfill decision experiences for already-resolved reviewed trades.

Usage:
    python -m dynamo.cli.experience_backfill [--database-url URL]

Best-effort, idempotent, LLM-free: runs the same spine join as the
``experience_resolution`` asset (reviewed closed trades lacking a
``decision_experience`` row → build → insert with ``source='live'``) and
prints built/skipped counts. Trades with incomplete joins are skipped, a
single bad trade never aborts the run. Safe to re-run: live inserts are
idempotent per decision context.

Run from the repository root (the shared resolver lives in the
``orchestration`` composition package, importable from there).
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-experience-backfill",
        description="Build decision_experience rows for already-reviewed closed trades",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="override DATABASE_URL / settings (optional)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Lazy imports: the CLI is the composition layer; keeping them inside
    # main() keeps module import side-effect free and the core pure.
    import logging

    from orchestration.resources import DynamoResource

    args = _build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("experience_backfill")

    from orchestration.assets.learning_ops import resolve_experiences

    dynamo = DynamoResource(database_url=args.database_url or "")
    counts = resolve_experiences(dynamo, log)
    print(
        "experience backfill: "
        f"built={counts['built']} skipped={counts['skipped']} "
        f"existing={counts['existing']} failed={counts['failed']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
