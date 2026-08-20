"""Backups exist and are verifiable (doc 12 §4).

Everything here is pure or subprocess-free: argv construction, retention
selection, row-count assertions, the metric contract. The end-to-end
dump/restore is the ``make backup-verify`` drill, not a unit test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dynamo.persistence.backup import (
    CRITICAL_TABLES,
    DEFAULT_KEEP_DAILY,
    DEFAULT_KEEP_MONTHLY,
    BackupError,
    DockerExec,
    LocalExec,
    backup_is_stale,
    compare_counts,
    dump_argv,
    dump_name,
    dump_timestamp,
    monthly_kept,
    parse_database_url,
    psql_argv,
    restore_argv,
    retention_plan,
    scratch_database_name,
)

URL = "postgresql+psycopg://dynamo:s3cr3t@localhost:5432/dynamo_dev"


# --- URL parsing ---------------------------------------------------------


def test_sqlalchemy_url_parses_to_libpq_parts() -> None:
    t = parse_database_url(URL)
    assert (t.host, t.port, t.user, t.password, t.database) == (
        "localhost", 5432, "dynamo", "s3cr3t", "dynamo_dev"
    )


def test_percent_encoded_password_is_decoded() -> None:
    t = parse_database_url("postgresql://u:p%40ss%2Fword@h:5432/db")
    assert t.password == "p@ss/word"


@pytest.mark.parametrize(
    "url",
    [
        "mysql://u:p@h/db",                       # wrong engine
        "postgresql+psycopg://u:p@h:5432/",       # no database
        "postgresql+psycopg:///db",               # no host
    ],
)
def test_unparseable_urls_raise_rather_than_guess(url: str) -> None:
    """Silently backing up the wrong database is worse than not backing up."""
    with pytest.raises(BackupError):
        parse_database_url(url)


# --- argv ----------------------------------------------------------------


def test_dump_uses_custom_format() -> None:
    argv = dump_argv(parse_database_url(URL), LocalExec())
    assert argv[0] == "pg_dump"
    assert "-Fc" in argv  # the only format pg_restore can restore selectively
    assert argv[-2:] == ["-d", "dynamo_dev"]


def test_docker_exec_passes_the_password_by_env_not_argv() -> None:
    argv = dump_argv(parse_database_url(URL), DockerExec(container="pg"))
    assert argv[:3] == ["docker", "exec", "-i"]
    assert "PGPASSWORD=s3cr3t" in argv
    assert "s3cr3t" not in " ".join(a for a in argv if not a.startswith("PGPASSWORD"))


def test_docker_exec_targets_localhost_inside_the_container() -> None:
    """The host sees a mapped port; the container sees its own postgres."""
    url = "postgresql+psycopg://dynamo:p@db.internal:6543/dynamo_dev"
    argv = dump_argv(parse_database_url(url), DockerExec(container="pg"))
    assert "db.internal" not in argv
    assert "localhost" in argv and "5432" in argv


def test_restore_stops_on_the_first_error() -> None:
    argv = restore_argv(parse_database_url(URL), LocalExec())
    assert "--exit-on-error" in argv  # a half-restored database must not look OK


def test_psql_can_target_another_database() -> None:
    argv = psql_argv(parse_database_url(URL), LocalExec(), "SELECT 1", database="postgres")
    assert argv[-3:] == ["-d", "postgres", "SELECT 1"] or "postgres" in argv


# --- naming + retention --------------------------------------------------


def test_dump_name_round_trips_its_timestamp() -> None:
    at = datetime(2026, 8, 20, 3, 39, 55, tzinfo=UTC)
    name = dump_name("dynamo_dev", at)
    assert name == "dynamo-dynamo_dev-20260820T033955Z.dump"
    assert dump_timestamp(Path(name)) == at


def test_foreign_files_have_no_timestamp() -> None:
    assert dump_timestamp(Path("notes.txt")) is None
    assert dump_timestamp(Path("dynamo-dynamo_dev.dump")) is None


def _dumps(days: int, *, end: datetime | None = None) -> list[Path]:
    end = end or datetime(2026, 8, 20, 3, 0, tzinfo=UTC)
    return [Path(dump_name("dynamo_dev", end - timedelta(days=d))) for d in range(days)]


def test_thirty_daily_are_kept_outright() -> None:
    plan = retention_plan(_dumps(40))
    assert len(plan.keep) >= DEFAULT_KEEP_DAILY
    newest_30 = set(_dumps(DEFAULT_KEEP_DAILY))
    assert newest_30 <= set(plan.keep)


def test_beyond_thirty_days_one_dump_per_month_survives() -> None:
    plan = retention_plan(_dumps(700))
    months = monthly_kept(plan.keep)
    # 30 daily spans at most 2 calendar months; plus 12 monthlies.
    assert DEFAULT_KEEP_MONTHLY <= len(months) <= DEFAULT_KEEP_MONTHLY + 2
    assert plan.delete  # old dailies really are pruned
    assert set(plan.keep).isdisjoint(plan.delete)


def test_nothing_is_deleted_when_there_is_little_history() -> None:
    plan = retention_plan(_dumps(5))
    assert plan.delete == ()


def test_unrecognized_files_are_never_deleted() -> None:
    """A retention sweep is not the place to discover somebody else's file."""
    paths = [*_dumps(3), Path("README.md"), Path("wal-g-manifest.json")]
    plan = retention_plan(paths)
    assert set(plan.unrecognized) == {Path("README.md"), Path("wal-g-manifest.json")}
    assert Path("README.md") not in plan.delete


def test_retention_is_order_independent() -> None:
    paths = _dumps(90)
    assert set(retention_plan(paths).keep) == set(retention_plan(reversed(paths)).keep)


# --- verification assertions --------------------------------------------


def test_matching_nonzero_counts_pass() -> None:
    counts = {t: 100 for t in CRITICAL_TABLES}
    assert compare_counts(counts, dict(counts)) == ()


def test_a_row_count_mismatch_is_reported() -> None:
    src = {"bars_1d": 6_196_596}
    problems = compare_counts(src, {"bars_1d": 6_000_000})
    assert len(problems) == 1 and "6196596" in problems[0].replace(",", "")


def test_an_empty_restore_that_matches_an_empty_source_still_fails() -> None:
    """A "successful" restore of nothing is the failure this catches."""
    problems = compare_counts({"bars_1d": 0}, {"bars_1d": 0})
    assert problems and "nothing was backed up" in problems[0]


def test_a_missing_table_is_reported() -> None:
    assert compare_counts({"bars_1d": 5}, {}) != ()


def test_the_scratch_database_is_never_the_working_one() -> None:
    name = scratch_database_name(datetime(2026, 8, 20, 3, 40, 19, tzinfo=UTC))
    assert name == "dynamo_restore_drill_20260820034019"
    assert name not in {"dynamo_dev", "dynamo_paper", "dynamo_live"}


# --- staleness -----------------------------------------------------------


def test_no_backup_on_record_is_stale() -> None:
    assert backup_is_stale(None, datetime.now(UTC)) is True


def test_a_backup_inside_the_window_is_fresh() -> None:
    now = datetime(2026, 8, 20, 12, tzinfo=UTC)
    assert backup_is_stale(now - timedelta(hours=25), now) is False
    assert backup_is_stale(now - timedelta(hours=27), now) is True
