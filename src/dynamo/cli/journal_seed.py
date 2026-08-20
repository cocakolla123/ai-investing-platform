"""Seed the journal from resolved decision experiences (findings item 14).

Usage:
    python -m dynamo.cli.journal_seed [--dry-run]
    python -m dynamo.cli.journal_seed --replay-run-id <run>   # opt-in, see below
    python -m dynamo.cli.journal_seed --reembed               # upgrade vectors

Journal hits rendered ``none`` in 300 of 300 dossiers of the control run
because ``journal_entries`` was empty, so prior-case retrieval was a no-op.
This turns resolved ``decision_experience`` rows into journal entries via the
existing builders (``review_entry`` / ``lesson_entries``). Idempotent: entry
ids are deterministic and inserts conflict-do-nothing.

TWO deliberate refusals:

* **The replay firewall.** Replay-derived experiences (``source='replay'``)
  are NOT seeded by default: ``journal_entries`` has no source column, and
  live retrieval reads the whole table, so seeding them would quietly feed
  one replay run's outcomes to live decisions and to every other run.
  ``--replay-run-id`` seeds exactly one run's rows, and every entry records
  its provenance in ``refs``.
* **The hash embedder.** ``LocalEmbedder`` silently falls back to
  sha256-seeded pseudo-vectors when the ``embeddings`` extra is absent;
  vectors that carry no meaning make similarity search theatre. This CLI
  refuses to run on the fallback unless ``--allow-hash-embedder`` is passed,
  and always reports which backend ran (also stored in
  ``journal_entries.embedding_version``). After installing the extra,
  ``--reembed`` upgrades the vectors already written, in place.

REMEDIATING THE 113 EXISTING FALLBACK ENTRIES
---------------------------------------------
Every row in ``journal_entries`` was written under the fallback, so RETRIEVAL
CURRENTLY ABSTAINS: ``dynamo.learning.journal.journal_retrieval_gate`` refuses
to put those hits into a live committee prompt, because random past trades
labelled "similar past situations" are worse than no precedent at all. The
whole repair is two commands::

    uv sync --extra embeddings
    uv run python -m dynamo.cli.journal_seed --reembed

``--reembed`` rewrites only the vector — never the text, never ``created_at``
— so no point-in-time read changes, and retrieval re-enables itself once the
entries stop being fallback-stamped (no setting to flip back). Verify with::

    SELECT embedding_version, count(*) FROM journal_entries GROUP BY 1;

Full write-up, including why this was not simply done in place:
``docs/journal_embeddings.md``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-journal-seed",
        description="Write journal entries from resolved decision experiences",
    )
    parser.add_argument("--database-url", default=None, help="override DATABASE_URL")
    parser.add_argument(
        "--replay-run-id",
        action="append",
        default=None,
        dest="replay_run_ids",
        metavar="RUN_ID",
        help="ALSO seed this replay run's experiences (firewall opt-in; repeatable)",
    )
    parser.add_argument(
        "--allow-hash-embedder",
        action="store_true",
        help="proceed even when the real embedding backend is unavailable",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="re-embed entries previously written with the hash fallback",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would be written")
    return parser


def _embedder(allow_fallback: bool, log: Any) -> Any:
    from dynamo.llm.embeddings import BACKEND_HASH_FALLBACK, LocalEmbedder

    embedder = LocalEmbedder()
    if embedder.backend == BACKEND_HASH_FALLBACK and not allow_fallback:
        log.error(
            "journal_seed: embedder resolved to %s — install the embeddings extra "
            "(`uv sync --extra embeddings`) or pass --allow-hash-embedder",
            BACKEND_HASH_FALLBACK,
        )
        return None
    log.info("journal_seed: embedding backend = %s", embedder.backend)
    return embedder


def main(argv: list[str] | None = None) -> int:
    import logging

    from orchestration.resources import DynamoResource

    from dynamo.learning.journal_seed import seed_entries
    from dynamo.llm.embeddings import BACKEND_HASH_FALLBACK
    from dynamo.persistence.repositories.experience import ExperienceRepository
    from dynamo.persistence.repositories.learning import JournalRepository

    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("journal_seed")

    dynamo = DynamoResource(database_url=args.database_url or "")
    engine = dynamo.engine()
    journal = JournalRepository(engine)
    embedder = _embedder(args.allow_hash_embedder, log)
    if embedder is None:
        return 2

    if args.reembed:
        stale = journal.entries_with_embedding_version(BACKEND_HASH_FALLBACK)
        if not stale:
            print("journal reembed: nothing to upgrade")
            return 0
        if args.dry_run:
            print(f"journal reembed: would re-embed {len(stale)} entries")
            return 0
        vectors = embedder.embed([e.text for e in stale])
        for entry, vector in zip(stale, vectors, strict=True):
            journal.update_embedding(entry.id, vector, embedder.backend)
        print(f"journal reembed: {len(stale)} entries -> backend={embedder.backend}")
        return 0

    repo = ExperienceRepository(engine)
    experiences = repo.resolved_live()
    live_count = len(experiences)
    replay_count = 0
    for run_id in args.replay_run_ids or ():
        replay = repo.by_run(run_id)
        replay_count += len(replay)
        log.warning(
            "journal_seed: seeding %d replay experiences from run %s — these are "
            "REPLAY outcomes entering the shared journal (provenance recorded in "
            "refs.replay_run_id)",
            len(replay),
            run_id,
        )
        experiences = [*experiences, *replay]

    entries = seed_entries(experiences)
    if args.dry_run:
        print(
            f"journal seed (dry run): live={live_count} replay={replay_count} "
            f"-> {len(entries)} entries, backend={embedder.backend}"
        )
        return 0
    if not entries:
        print("journal seed: no resolved experiences to seed")
        return 0
    vectors = embedder.embed([e.text for e in entries])
    for entry, vector in zip(entries, vectors, strict=True):
        journal.insert(entry, vector, embedding_version=embedder.backend)
    kinds: dict[str, int] = {}
    for entry in entries:
        kinds[entry.kind.value] = kinds.get(entry.kind.value, 0) + 1
    print(
        f"journal seed: experiences live={live_count} replay={replay_count} "
        f"-> entries={len(entries)} ({kinds}) backend={embedder.backend}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
