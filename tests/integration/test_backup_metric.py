"""``backup_last_success_timestamp`` really lands in ``metrics_daily``.

Skipped unless ``INTEGRATION_DB_URL`` points at a Postgres instance — the
metric write is an upsert on a composite primary key, which is exactly the
part a fake engine would not exercise.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, create_engine, delete

from dynamo.persistence import schema
from dynamo.persistence.backup import (
    BACKUP_METRIC,
    KIND_LOGICAL,
    KIND_RESTORE_DRILL,
    backup_drill_ok,
    backup_is_stale,
    last_backup_at,
    record_backup_metric,
)

pytestmark = pytest.mark.integration


@pytest.fixture()
def engine(integration_db_url: str) -> Iterator[Engine]:
    eng = create_engine(integration_db_url, future=True)
    schema.metadata.create_all(eng, tables=[schema.metrics_daily])
    with eng.begin() as conn:
        conn.execute(
            delete(schema.metrics_daily).where(schema.metrics_daily.c.metric == BACKUP_METRIC)
        )
    yield eng
    with eng.begin() as conn:
        conn.execute(
            delete(schema.metrics_daily).where(schema.metrics_daily.c.metric == BACKUP_METRIC)
        )
    eng.dispose()


def test_the_metric_is_written_and_read_back(engine: Engine) -> None:
    at = datetime(2026, 8, 20, 3, 39, 55, tzinfo=UTC)
    record_backup_metric(engine, KIND_LOGICAL, at)
    read = last_backup_at(engine, KIND_LOGICAL)
    assert read is not None
    assert abs((read - at).total_seconds()) < 1.0


def test_no_metric_reads_as_never_and_therefore_stale(engine: Engine) -> None:
    assert last_backup_at(engine, KIND_LOGICAL) is None
    assert backup_is_stale(None, datetime.now(UTC)) is True


def test_two_backups_the_same_day_upsert_rather_than_collide(engine: Engine) -> None:
    """The primary key includes ``as_of`` only — a re-run must not blow up."""
    day = datetime(2026, 8, 20, 3, tzinfo=UTC)
    record_backup_metric(engine, KIND_LOGICAL, day)
    record_backup_metric(engine, KIND_LOGICAL, day + timedelta(hours=6))
    read = last_backup_at(engine, KIND_LOGICAL)
    assert read is not None and read.hour == 9


def test_logical_and_drill_kinds_do_not_overwrite_each_other(engine: Engine) -> None:
    at = datetime(2026, 8, 20, 3, tzinfo=UTC)
    record_backup_metric(engine, KIND_LOGICAL, at)
    record_backup_metric(engine, KIND_RESTORE_DRILL, at - timedelta(days=10))
    assert last_backup_at(engine, KIND_LOGICAL) is not None
    assert last_backup_at(engine, KIND_RESTORE_DRILL) is not None
    assert last_backup_at(engine, KIND_LOGICAL) != last_backup_at(engine, KIND_RESTORE_DRILL)


def test_backup_drill_ok_is_false_without_a_drill(engine: Engine) -> None:
    """doc 00 §5.4 makes the drill a live-gate dependency; it fails closed."""
    assert backup_drill_ok(engine, datetime.now(UTC)) is False


def test_a_drill_inside_the_quarter_satisfies_readiness(engine: Engine) -> None:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    record_backup_metric(engine, KIND_RESTORE_DRILL, now - timedelta(days=30))
    assert backup_drill_ok(engine, now) is True


def test_a_drill_older_than_a_quarter_does_not(engine: Engine) -> None:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    record_backup_metric(engine, KIND_RESTORE_DRILL, now - timedelta(days=200))
    assert backup_drill_ok(engine, now) is False


def test_the_newest_drill_is_the_one_that_counts(engine: Engine) -> None:
    """Re-running the drill refreshes readiness; the old row does not drag it down."""
    now = datetime(2026, 8, 20, tzinfo=UTC)
    record_backup_metric(engine, KIND_RESTORE_DRILL, now - timedelta(days=200))
    record_backup_metric(engine, KIND_RESTORE_DRILL, now - timedelta(days=1))
    assert backup_drill_ok(engine, now) is True
