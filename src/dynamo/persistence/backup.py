"""Postgres backups, retention, restore verification (doc 12 §4).

Postgres is the single system of record — ~945k XBRL facts, 6.2M bars, 104k
corporate actions, 3.9M short-interest rows and the entire replay history —
and until this module the ``pgdata`` Docker volume was the only copy.

**Why a Python CLI rather than a compose service or a bash script**

- A compose *service* means a container resident 24/7 on a 8 GB host budget
  (doc 12 §1 leaves ~2 GB of headroom) for a job that runs for a minute a
  night. Worse, the backup you most want is the one taken while the app
  containers are stopped.
- A bash script would have to reimplement DATABASE_URL parsing, the
  ``metrics_daily`` write, retention selection and the failure alert. All four
  already exist in Python and are unit-testable there; in bash none of them
  would be.

So: pure logic here (retention, argv construction, verification assertions),
subprocess boundaries injected as a ``Runner``, and ``make backup`` /
``make backup-verify`` as the operator-facing entry points. Scheduling is
host cron or a systemd timer — see ``docs/runbooks/db-restore.md``.

**Why the tools run in the container**: ``pg_dump`` is not installed on a
typical app host, but the pinned ``timescale/timescaledb-ha:pg16`` image has
exactly the matching major version. Dumping with a mismatched ``pg_dump`` is
the classic way to produce an archive that will not restore.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

BACKUP_METRIC = "backup_last_success_timestamp"
BACKUP_SCOPE = "backup"
KIND_LOGICAL = "logical"
KIND_RESTORE_DRILL = "restore_drill"

DEFAULT_KEEP_DAILY = 30
DEFAULT_KEEP_MONTHLY = 12
#: doc 12 §7: "Backup failure — page — backup_last_success_timestamp stale > 26h".
STALE_AFTER_HOURS = 26.0
#: doc 12 §4: the restore drill is quarterly, and a live-gate dependency.
DRILL_VALID_DAYS = 100

#: ``dynamo-<db>-<YYYYmmddTHHMMSSZ>.dump``
DUMP_SUFFIX = ".dump"
_STAMP_RE = re.compile(r"-(\d{8}T\d{6})Z" + re.escape(DUMP_SUFFIX) + r"$")


class BackupError(RuntimeError):
    """A backup or restore step failed. Never swallowed — a silent backup
    failure is the classic solo-operator catastrophe (doc 12 §4)."""


@dataclass(frozen=True)
class DbTarget:
    """Connection parts, extracted from a SQLAlchemy URL."""

    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def psql_flags(self) -> tuple[str, ...]:
        return ("-h", self.host, "-p", str(self.port), "-U", self.user)

    def with_database(self, database: str) -> DbTarget:
        return DbTarget(self.host, self.port, self.user, self.password, database)


def parse_database_url(url: str) -> DbTarget:
    """``postgresql+psycopg://u:p@h:5432/db`` -> :class:`DbTarget`.

    Deliberately strict: a URL we cannot parse must raise, not silently back
    up the wrong database.
    """
    from urllib.parse import unquote, urlsplit

    parts = urlsplit(url)
    if not parts.scheme.startswith("postgresql") and not parts.scheme.startswith("postgres"):
        raise BackupError(f"not a postgres URL: {parts.scheme!r}")
    database = parts.path.lstrip("/")
    if not database:
        raise BackupError(f"no database name in URL {url!r}")
    if not parts.hostname:
        raise BackupError(f"no host in URL {url!r}")
    return DbTarget(
        host=parts.hostname,
        port=int(parts.port or 5432),
        user=unquote(parts.username or "postgres"),
        password=unquote(parts.password or ""),
        database=database,
    )


# --- where the tools run -------------------------------------------------


@dataclass(frozen=True)
class DockerExec:
    """Run pg tooling inside the postgres container (the default).

    ``container`` is resolved by the caller; inside the container the database
    is reachable at ``localhost``, which is why the host part of the target is
    overridden here rather than in the URL.
    """

    container: str = "ai-investing-platform-postgres-1"
    docker: str = "docker"

    def wrap(self, target: DbTarget, argv: Sequence[str]) -> list[str]:
        return [
            self.docker,
            "exec",
            "-i",
            "-e",
            f"PGPASSWORD={target.password}",
            "-e",
            f"PGOPTIONS={RESTORE_PGOPTIONS}",
            self.container,
            *argv,
        ]

    def rewrite(self, target: DbTarget) -> DbTarget:
        # Port-mapped from the host; inside the container it is the local socket.
        return DbTarget("localhost", 5432, target.user, target.password, target.database)


@dataclass(frozen=True)
class LocalExec:
    """Run pg tooling straight off ``$PATH`` (CI, or a host with client tools)."""

    def wrap(self, target: DbTarget, argv: Sequence[str]) -> list[str]:
        return list(argv)

    def rewrite(self, target: DbTarget) -> DbTarget:
        return target


Executor = DockerExec | LocalExec

#: ``(argv, stdin_path, stdout_path, env) -> (returncode, stderr_text)``
Runner = Callable[[Sequence[str], Path | None, Path | None, dict[str, str]], tuple[int, str]]


def dump_argv(target: DbTarget, executor: Executor) -> list[str]:
    """``pg_dump -Fc`` argv. Custom format because it is the only one
    ``pg_restore`` can restore selectively and in parallel, and it is already
    compressed."""
    inner = target if isinstance(executor, LocalExec) else executor.rewrite(target)
    return executor.wrap(
        target,
        ["pg_dump", "-Fc", "--no-owner", "--no-acl", *inner.psql_flags, "-d", inner.database],
    )


#: Session-level memory cap for restores, passed via PGOPTIONS.
#:
#: The timescaledb-ha image auto-tunes maintenance_work_mem to ~1 GB for
#: this container, whose Docker limit is 2.5 GiB with ~2 GiB already
#: resident. Building the primary key on a 21.8M-row xbrl_facts during the
#: 2026-09-14 restore drill asked for that 1 GB on top and the kernel
#: KILLED THE SERVER mid-drill (crash recovery, no data loss). A restore
#: session capped at 256 MB builds the same index slower and fits anywhere;
#: the drill is a correctness check, not a race.
RESTORE_PGOPTIONS = "-c maintenance_work_mem=256MB"


def restore_argv(target: DbTarget, executor: Executor, *, jobs: int = 1) -> list[str]:
    inner = target if isinstance(executor, LocalExec) else executor.rewrite(target)
    argv = [
        "pg_restore",
        "--no-owner",
        "--no-acl",
        "--exit-on-error",
        *inner.psql_flags,
        "-d",
        inner.database,
    ]
    if jobs > 1:
        argv += ["-j", str(jobs)]
    return executor.wrap(target, argv)


def psql_argv(
    target: DbTarget, executor: Executor, sql: str, *, database: str | None = None
) -> list[str]:
    inner = target if isinstance(executor, LocalExec) else executor.rewrite(target)
    db = database or inner.database
    return executor.wrap(
        target,
        ["psql", "-v", "ON_ERROR_STOP=1", "-tAq", *inner.psql_flags, "-d", db, "-c", sql],
    )


# --- naming + retention (pure) ------------------------------------------


def dump_name(database: str, at: datetime) -> str:
    return f"dynamo-{database}-{at.astimezone(UTC):%Y%m%dT%H%M%S}Z{DUMP_SUFFIX}"


def dump_timestamp(path: Path | str) -> datetime | None:
    """Timestamp encoded in a dump filename; None when it is not ours."""
    match = _STAMP_RE.search(Path(path).name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


@dataclass(frozen=True)
class RetentionPlan:
    keep: tuple[Path, ...] = ()
    delete: tuple[Path, ...] = ()
    unrecognized: tuple[Path, ...] = ()


def retention_plan(
    paths: Iterable[Path],
    *,
    keep_daily: int = DEFAULT_KEEP_DAILY,
    keep_monthly: int = DEFAULT_KEEP_MONTHLY,
) -> RetentionPlan:
    """Doc 12 §4: "30 daily, 12 monthly".

    Newest ``keep_daily`` dumps are kept outright. Beyond that, the newest
    dump in each of the most recent ``keep_monthly`` calendar months is kept
    and the rest are deleted. Files we do not recognize are never deleted —
    an unknown file in the backup directory is somebody else's, and a
    retention sweep is not the place to find out.
    """
    dated: list[tuple[datetime, Path]] = []
    unknown: list[Path] = []
    for p in paths:
        ts = dump_timestamp(p)
        if ts is None:
            unknown.append(p)
        else:
            dated.append((ts, p))
    dated.sort(key=lambda item: item[0], reverse=True)

    keep: list[Path] = [p for _, p in dated[:keep_daily]]
    keep_set = set(keep)

    monthly: dict[tuple[int, int], Path] = {}
    for ts, p in dated[keep_daily:]:
        monthly.setdefault((ts.year, ts.month), p)  # dated is newest-first
    for _, p in sorted(monthly.items(), reverse=True)[:keep_monthly]:
        if p not in keep_set:
            keep.append(p)
            keep_set.add(p)

    delete = [p for _, p in dated if p not in keep_set]
    return RetentionPlan(tuple(keep), tuple(delete), tuple(sorted(unknown)))


# --- verification --------------------------------------------------------

#: Tables whose emptiness after a restore means the archive is worthless.
#: Chosen because they are the expensive-to-refetch history, not because they
#: are the biggest: an empty ``bars_1d`` is years of provider budget.
CRITICAL_TABLES: tuple[str, ...] = (
    "bars_1d",
    "instruments",
    "xbrl_facts",
    "corporate_actions",
    "short_interest",
)


@dataclass
class VerificationResult:
    scratch_database: str
    source_counts: dict[str, int] = field(default_factory=dict)
    restored_counts: dict[str, int] = field(default_factory=dict)
    mismatches: tuple[str, ...] = ()
    dump_bytes: int = 0
    dump_seconds: float = 0.0
    restore_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.mismatches


def compare_counts(
    source: dict[str, int], restored: dict[str, int], *, require_nonempty: bool = True
) -> tuple[str, ...]:
    """Row-count assertions (doc 12 §4 step 1). A restore that "succeeds" and
    produces an empty database is the failure mode this catches."""
    problems: list[str] = []
    for table in sorted(set(source) | set(restored)):
        src = source.get(table)
        dst = restored.get(table)
        if src is None or dst is None:
            problems.append(f"{table}: missing from {'source' if src is None else 'restore'}")
            continue
        if src != dst:
            problems.append(f"{table}: {src} rows in source, {dst} restored")
        elif require_nonempty and src == 0:
            problems.append(f"{table}: 0 rows in both — nothing was backed up")
    return tuple(problems)


def scratch_database_name(at: datetime) -> str:
    """Never ``dynamo_dev``. The drill restores into a throwaway database and
    drops it; overwriting the working database to test the backup would be
    the single most destructive thing this module could do."""
    return f"dynamo_restore_drill_{at.astimezone(UTC):%Y%m%d%H%M%S}"


# --- the metric ----------------------------------------------------------


def record_backup_metric(engine: Any, kind: str, at: datetime) -> None:
    """``backup_last_success_timestamp{kind}`` (doc 12 §6) into ``metrics_daily``.

    A gauge in the ``metrics_daily`` table rather than a live Prometheus gauge
    because the backup runs as a cron job in its own process: an in-memory
    gauge there would be scraped by nobody. The API exports the row.
    """
    from sqlalchemy.dialects.postgresql import insert

    from dynamo.persistence import schema

    row = {
        "metric": BACKUP_METRIC,
        "scope_type": BACKUP_SCOPE,
        "scope_id": kind,
        "as_of": at.astimezone(UTC).date(),
        "value": at.timestamp(),
    }
    stmt = insert(schema.metrics_daily).values([row])
    stmt = stmt.on_conflict_do_update(
        index_elements=["metric", "scope_type", "scope_id", "as_of"],
        set_={"value": stmt.excluded.value},
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def last_backup_at(engine: Any, kind: str = KIND_LOGICAL) -> datetime | None:
    from sqlalchemy import select

    from dynamo.persistence import schema

    t = schema.metrics_daily
    stmt = (
        select(t.c.value)
        .where(t.c.metric == BACKUP_METRIC, t.c.scope_type == BACKUP_SCOPE, t.c.scope_id == kind)
        .order_by(t.c.as_of.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        value = conn.execute(stmt).scalar()
    return None if value is None else datetime.fromtimestamp(float(value), tz=UTC)


def backup_is_stale(last: datetime | None, now: datetime, hours: float = STALE_AFTER_HOURS) -> bool:
    if last is None:
        return True
    return (now - last).total_seconds() > hours * 3600.0


def backup_drill_ok(engine: Any, now: datetime, *, valid_days: int = DRILL_VALID_DAYS) -> bool:
    """``Readiness.backup_drill_ok`` (doc 00 §5.4) — quarterly drill, from the
    metric the drill itself writes. No drill on record means False: the live
    gate fails closed."""
    last = last_backup_at(engine, KIND_RESTORE_DRILL)
    if last is None:
        return False
    return (now - last).days <= valid_days


def monthly_kept(paths: Iterable[Path]) -> dict[tuple[int, int], date]:
    """Diagnostic: which months the surviving dumps cover."""
    out: dict[tuple[int, int], date] = {}
    for p in paths:
        ts = dump_timestamp(p)
        if ts is not None:
            out[(ts.year, ts.month)] = ts.date()
    return out
