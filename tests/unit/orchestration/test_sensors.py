"""Sensor logic against in-memory fakes — no network, no database."""

from __future__ import annotations

from datetime import timedelta

from dagster import SensorResult, SkipReason, build_sensor_context
from orchestration.resources import utcnow
from orchestration.sensors import (
    batch_completion_sensor,
    budget_sensor,
    failure_clustering_sensor,
    quarantine_alert_sensor,
)

from tests.unit.orchestration.conftest import FakeDynamo


def _ctx(cursor: str | None = None):
    return build_sensor_context(cursor=cursor)


# --- batch_completion_sensor ---------------------------------------------


def test_batch_sensor_skips_with_no_pending(fake_dynamo: FakeDynamo) -> None:
    result = batch_completion_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SkipReason)


def test_batch_sensor_advances_finished_stage(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.batches = [{"batch_id": "b1", "stage": "dossier", "submitted_at": utcnow()}]
    fake_dynamo.st.statuses = {"b1": "ended"}
    result = batch_completion_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    [request] = result.run_requests or []
    assert request.run_key == "advance-b1"
    config = (request.run_config or {})["ops"]["advance_committee_stage"]["config"]
    assert config == {"batch_id": "b1", "stage": "dossier"}


def test_batch_sensor_advances_each_stage_with_its_own_config(
    fake_dynamo: FakeDynamo,
) -> None:
    fake_dynamo.st.batches = [
        {"batch_id": "b-d", "stage": "dossier", "submitted_at": utcnow()},
        {"batch_id": "b-a", "stage": "advocates", "submitted_at": utcnow()},
        {"batch_id": "b-r", "stage": "arbiter", "submitted_at": utcnow()},
    ]
    fake_dynamo.st.statuses = {"b-d": "ended", "b-a": "ended", "b-r": "ended"}
    result = batch_completion_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    stages = {
        (r.run_config or {})["ops"]["advance_committee_stage"]["config"]["stage"]
        for r in result.run_requests or []
    }
    assert stages == {"dossier", "advocates", "arbiter"}


def test_batch_sensor_waits_on_unfinished_batch(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.batches = [{"batch_id": "b1", "stage": "dossier", "submitted_at": utcnow()}]
    fake_dynamo.st.statuses = {"b1": "in_progress"}
    result = batch_completion_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    assert not result.run_requests
    assert fake_dynamo.st.expired == []


def test_batch_sensor_expires_stale_batch_with_warn(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.batches = [
        {"batch_id": "b1", "stage": "advocates", "submitted_at": utcnow() - timedelta(hours=25)}
    ]
    fake_dynamo.st.statuses = {"b1": "in_progress"}
    result = batch_completion_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    assert not result.run_requests
    assert fake_dynamo.st.expired == ["b1"]
    [alert] = fake_dynamo.st.alerts
    assert alert["severity"] == "warn"
    assert alert["name"] == "batch_expiry"


# --- budget_sensor --------------------------------------------------------


def _spend(fake_dynamo: FakeDynamo, usd: float) -> None:
    fake_dynamo.cost_meter().record(
        model_id="claude-test", call_kind="test", input_tokens=0, output_tokens=0, cost_usd=usd
    )


def test_budget_sensor_quiet_below_warn(fake_dynamo: FakeDynamo) -> None:
    _spend(fake_dynamo, 50.0)  # 50% of the $100 fake budget
    assert isinstance(budget_sensor(_ctx(), dynamo=fake_dynamo), SkipReason)
    assert fake_dynamo.st.alerts == []


def test_budget_sensor_warns_at_70(fake_dynamo: FakeDynamo) -> None:
    _spend(fake_dynamo, 75.0)
    budget_sensor(_ctx(), dynamo=fake_dynamo)
    [alert] = fake_dynamo.st.alerts
    assert alert["severity"] == "warn"


def test_budget_sensor_pages_at_100(fake_dynamo: FakeDynamo) -> None:
    _spend(fake_dynamo, 120.0)
    budget_sensor(_ctx(), dynamo=fake_dynamo)
    [alert] = fake_dynamo.st.alerts
    assert alert["severity"] == "page"


def test_budget_sensor_alerts_once_per_level_per_month(fake_dynamo: FakeDynamo) -> None:
    _spend(fake_dynamo, 75.0)
    month = utcnow().strftime("%Y-%m")
    result = budget_sensor(_ctx(cursor=f"{month}:warn"), dynamo=fake_dynamo)
    assert isinstance(result, SkipReason)
    assert fake_dynamo.st.alerts == []


# --- quarantine_alert_sensor ---------------------------------------------


def test_quarantine_sensor_quiet_below_threshold(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.quarantine_count = 5
    assert isinstance(quarantine_alert_sensor(_ctx(), dynamo=fake_dynamo), SkipReason)


def test_quarantine_sensor_pages_on_systemic_failure(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.quarantine_count = 25
    quarantine_alert_sensor(_ctx(), dynamo=fake_dynamo)
    [alert] = fake_dynamo.st.alerts
    assert alert["severity"] == "page"
    assert alert["name"] == "data_quality_systemic"


def test_quarantine_sensor_pages_once_per_day(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.quarantine_count = 25
    today = utcnow().date().isoformat()
    result = quarantine_alert_sensor(_ctx(cursor=today), dynamo=fake_dynamo)
    assert isinstance(result, SkipReason)
    assert fake_dynamo.st.alerts == []


# --- failure_clustering_sensor (plan Phase 6) ------------------------------


def test_clustering_sensor_quiet_below_threshold(fake_dynamo: FakeDynamo) -> None:
    fake_dynamo.st.failure_count = 149
    fake_dynamo.st.clustering_watermark = 0
    result = failure_clustering_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SkipReason)
    assert "149 new resolved failures" in str(result.skip_message)


def test_clustering_sensor_triggers_at_threshold_with_count_keyed_run(
    fake_dynamo: FakeDynamo,
) -> None:
    fake_dynamo.st.failure_count = 150
    fake_dynamo.st.clustering_watermark = 0
    result = failure_clustering_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    [request] = result.run_requests or []
    assert request.run_key == "failure-clustering-150"
    # A re-evaluation before the run completes produces the IDENTICAL run key
    # — Dagster dedupes it, so the trigger fires effectively once.
    again = failure_clustering_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(again, SensorResult)
    [repeat] = again.run_requests or []
    assert repeat.run_key == request.run_key


def test_clustering_sensor_quiet_after_watermark_advances(
    fake_dynamo: FakeDynamo,
) -> None:
    """The clustering run records watermark=count; the sensor goes quiet
    until another 150 failures accumulate."""
    fake_dynamo.st.failure_count = 150
    fake_dynamo.st.clustering_watermark = 150
    assert isinstance(failure_clustering_sensor(_ctx(), dynamo=fake_dynamo), SkipReason)
    fake_dynamo.st.failure_count = 299
    assert isinstance(failure_clustering_sensor(_ctx(), dynamo=fake_dynamo), SkipReason)
    fake_dynamo.st.failure_count = 300
    result = failure_clustering_sensor(_ctx(), dynamo=fake_dynamo)
    assert isinstance(result, SensorResult)
    [request] = result.run_requests or []
    assert request.run_key == "failure-clustering-300"
