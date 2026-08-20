"""Backup / retention / restore-drill CLI (doc 12 §4).

    python -m dynamo.cli.backup run       # pg_dump -Fc + prune + metric
    python -m dynamo.cli.backup prune     # retention sweep only
    python -m dynamo.cli.backup verify    # restore the latest dump into a
                                          # SCRATCH database, assert row counts,
                                          # drop it, write the drill metric
    python -m dynamo.cli.backup status    # what exists, how stale, drill age

Scheduling is deliberately outside this tool: host cron or a systemd timer
(``docs/runbooks/db-restore.md`` has the crontab line). Every subcommand is
idempotent and safe to re-run.

Failure is loud: a non-zero exit AND — when a database is reachable — a
``page``-severity alert routed through the notifier from G2, because a backup
that silently stops is the classic solo-operator catastrophe.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynamo-backup", description="Postgres backup, retention and restore drill"
    )
    parser.add_argument(
        "--dir", default=None, help="backup directory (default: <DYNAMO_HOME>/backups)"
    )
    parser.add_argument(
        "--via",
        default="auto",
        choices=("auto", "docker", "local"),
        help="run pg tooling in the postgres container (docker) or off $PATH (local)",
    )
    parser.add_argument(
        "--container",
        default="ai-investing-platform-postgres-1",
        help="postgres container name for --via docker",
    )
    parser.add_argument("--database-url", default=None, help="override DATABASE_URL")
    parser.add_argument("--keep-daily", type=int, default=30)
    parser.add_argument("--keep-monthly", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="take a dump, prune, record the metric")
    sub.add_parser("prune", help="apply retention to the backup directory")
    sub.add_parser("status", help="report dump inventory and staleness")
    verify = sub.add_parser("verify", help="restore-drill into a scratch database")
    verify.add_argument("--dump", default=None, help="dump to restore (default: newest)")
    verify.add_argument(
        "--keep-scratch",
        action="store_true",
        help="leave the scratch database behind for inspection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import os
    import shutil
    import subprocess
    import time
    from datetime import UTC, datetime
    from pathlib import Path

    from dynamo.config.settings import get_settings
    from dynamo.persistence.backup import (
        CRITICAL_TABLES,
        KIND_LOGICAL,
        KIND_RESTORE_DRILL,
        BackupError,
        DockerExec,
        LocalExec,
        VerificationResult,
        backup_is_stale,
        compare_counts,
        dump_argv,
        dump_name,
        dump_timestamp,
        last_backup_at,
        parse_database_url,
        psql_argv,
        record_backup_metric,
        restore_argv,
        retention_plan,
        scratch_database_name,
    )

    args = _build_parser().parse_args(argv)
    settings = get_settings()
    now = datetime.now(UTC)

    def say(message: str) -> None:
        if not args.quiet:
            print(message)

    backup_dir = Path(args.dir) if args.dir else settings.anchored("backups")
    database_url = args.database_url or settings.database_url
    target = parse_database_url(database_url)

    if args.via == "docker":
        executor: DockerExec | LocalExec = DockerExec(container=args.container)
    elif args.via == "local":
        executor = LocalExec()
    else:  # auto: prefer matching-version tooling in the container
        executor = LocalExec() if shutil.which("pg_dump") else DockerExec(container=args.container)

    def run(
        argv_: list[str], *, stdin: Path | None = None, stdout: Path | None = None
    ) -> tuple[int, str, str]:
        fin = stdin.open("rb") if stdin else None
        fout = stdout.open("wb") if stdout else subprocess.PIPE
        try:
            proc = subprocess.run(  # argv is constructed, never a shell string
                argv_,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                env={**os.environ, "PGPASSWORD": target.password},
                check=False,
            )
        finally:
            if fin is not None:
                fin.close()
            if fout is not subprocess.PIPE:
                fout.close()  # type: ignore[union-attr]
        out = b"" if stdout else (proc.stdout or b"")
        return proc.returncode, out.decode("utf-8", "replace"), (
            proc.stderr or b""
        ).decode("utf-8", "replace")

    def counts(db: str) -> dict[str, int]:
        sql = " UNION ALL ".join(
            f"SELECT '{t}', count(*) FROM {t}" for t in CRITICAL_TABLES
        )
        code, out, err = run(psql_argv(target, executor, sql, database=db))
        if code != 0:
            raise BackupError(f"row count query failed on {db}: {err.strip()}")
        result: dict[str, int] = {}
        for line in out.splitlines():
            if "|" in line:
                name, _, value = line.partition("|")
                result[name.strip()] = int(value.strip())
        return result

    def alert(name: str, message: str) -> None:
        """Best effort: the CLI must still exit non-zero if this fails."""
        try:
            from dynamo.observability.notify import AlertDispatcher, build_notifier
            from dynamo.persistence.db import make_engine
            from dynamo.persistence.repositories.market import MarketDataRepository

            repo = MarketDataRepository(make_engine(database_url))
            AlertDispatcher(repo.add_alert, build_notifier(settings))(
                "page", name, message, ackable=False
            )
        except Exception as exc:  # the DB may be exactly what is broken
            print(f"(could not raise alert {name}: {exc!r})", file=sys.stderr)

    def engine() -> object:
        from dynamo.persistence.db import make_engine

        return make_engine(database_url)

    # ---------------------------------------------------------------- run
    if args.command == "run":
        backup_dir.mkdir(parents=True, exist_ok=True)
        out_path = backup_dir / dump_name(target.database, now)
        say(f"pg_dump -Fc {target.database} -> {out_path}")
        started = time.monotonic()
        code, _, err = run(dump_argv(target, executor), stdout=out_path)
        elapsed = time.monotonic() - started
        if code != 0:
            out_path.unlink(missing_ok=True)
            message = f"pg_dump failed (exit {code}): {err.strip()[:800]}"
            print(f"BACKUP FAILED: {message}", file=sys.stderr)
            alert("backup_failed", message)
            return 1
        size = out_path.stat().st_size
        if size == 0:
            out_path.unlink(missing_ok=True)
            alert("backup_failed", "pg_dump produced a zero-byte archive")
            print("BACKUP FAILED: zero-byte archive", file=sys.stderr)
            return 1
        say(f"  ok: {size:,} bytes in {elapsed:.1f}s")

        plan = retention_plan(
            backup_dir.glob(f"*{'.dump'}"),
            keep_daily=args.keep_daily,
            keep_monthly=args.keep_monthly,
        )
        for stale in plan.delete:
            stale.unlink(missing_ok=True)
        say(f"  retention: {len(plan.keep)} kept, {len(plan.delete)} pruned")

        try:
            record_backup_metric(engine(), KIND_LOGICAL, now)
            say(f"  metric: backup_last_success_timestamp{{kind=logical}} = {now.isoformat()}")
        except Exception as exc:
            print(f"WARNING: backup succeeded but the metric write failed: {exc!r}",
                  file=sys.stderr)
            return 1
        return 0

    # -------------------------------------------------------------- prune
    if args.command == "prune":
        plan = retention_plan(
            backup_dir.glob("*.dump"),
            keep_daily=args.keep_daily,
            keep_monthly=args.keep_monthly,
        )
        for stale in plan.delete:
            say(f"  rm {stale.name}")
            stale.unlink(missing_ok=True)
        for odd in plan.unrecognized:
            say(f"  (left alone, unrecognized: {odd.name})")
        say(f"{len(plan.keep)} kept, {len(plan.delete)} pruned")
        return 0

    # ------------------------------------------------------------- status
    if args.command == "status":
        dumps = sorted(
            (p for p in backup_dir.glob("*.dump") if dump_timestamp(p)),
            key=lambda p: dump_timestamp(p) or now,
            reverse=True,
        )
        say(f"backup dir: {backup_dir}")
        say(f"dumps: {len(dumps)}")
        for p in dumps[:5]:
            say(f"  {p.name}  {p.stat().st_size:,} bytes")
        try:
            last = last_backup_at(engine(), KIND_LOGICAL)  # type: ignore[arg-type]
            drill = last_backup_at(engine(), KIND_RESTORE_DRILL)  # type: ignore[arg-type]
        except Exception as exc:
            say(f"metric: unavailable ({exc!r})")
            return 1
        say(f"last logical backup: {last.isoformat() if last else 'never'}")
        say(f"last restore drill:  {drill.isoformat() if drill else 'never'}")
        stale = backup_is_stale(last, now)
        say(f"stale (> 26h): {stale}")
        return 1 if stale else 0

    # ------------------------------------------------------------- verify
    dumps = sorted(
        (p for p in backup_dir.glob("*.dump") if dump_timestamp(p)),
        key=lambda p: dump_timestamp(p) or now,
    )
    dump_path = Path(args.dump) if args.dump else (dumps[-1] if dumps else None)
    if dump_path is None or not dump_path.exists():
        print(f"refused: no dump to verify in {backup_dir}", file=sys.stderr)
        return 2

    scratch = scratch_database_name(now)
    if scratch == target.database:  # pragma: no cover - defensive
        print("refused: scratch name collides with the live database", file=sys.stderr)
        return 2
    say(f"restore drill: {dump_path.name} -> scratch database {scratch}")

    source = counts(target.database)
    code, _, err = run(
        psql_argv(target, executor, f'CREATE DATABASE "{scratch}"', database="postgres")
    )
    if code != 0:
        print(f"refused: could not create scratch database: {err.strip()}", file=sys.stderr)
        return 1

    result = VerificationResult(scratch_database=scratch, source_counts=source,
                                dump_bytes=dump_path.stat().st_size)
    try:
        started = time.monotonic()
        code, _, err = run(
            restore_argv(target.with_database(scratch), executor), stdin=dump_path
        )
        result.restore_seconds = time.monotonic() - started
        if code != 0:
            raise BackupError(f"pg_restore failed (exit {code}): {err.strip()[:800]}")
        result.restored_counts = counts(scratch)
        result.mismatches = compare_counts(source, result.restored_counts)
    except BackupError as exc:
        print(f"RESTORE DRILL FAILED: {exc}", file=sys.stderr)
        alert("restore_drill_failed", str(exc))
        if not args.keep_scratch:
            run(psql_argv(target, executor, f'DROP DATABASE IF EXISTS "{scratch}"',
                          database="postgres"))
        return 1
    finally:
        if not args.keep_scratch:
            run(psql_argv(target, executor, f'DROP DATABASE IF EXISTS "{scratch}"',
                          database="postgres"))

    for table in CRITICAL_TABLES:
        say(f"  {table}: {source.get(table, 0):,} -> {result.restored_counts.get(table, 0):,}")
    say(f"  restore took {result.restore_seconds:.1f}s from a "
        f"{result.dump_bytes:,}-byte archive")

    if not result.ok:
        for problem in result.mismatches:
            print(f"  MISMATCH: {problem}", file=sys.stderr)
        alert("restore_drill_failed", "; ".join(result.mismatches)[:800])
        return 1

    record_backup_metric(engine(), KIND_RESTORE_DRILL, now)
    say(f"  metric: backup_last_success_timestamp{{kind=restore_drill}} = {now.isoformat()}")
    say("restore drill PASSED — log it in docs/ops-log.md")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
