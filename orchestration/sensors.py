"""Sensors — committee stage advancement, quality alerts, budget alerts.

The batch sensor is deliberately thin: it polls batch status and requests a
run of ``committee_advance_job``; the heavy lifting (parse + persist + next
stage submission) happens inside the op where it is logged and retryable.
All state access goes through ``DynamoResource`` methods so tests inject fakes.
"""

from datetime import timedelta

from dagster import (
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    SkipReason,
    sensor,
)

from orchestration.assets.learning_ops import failure_clustering_job
from orchestration.assets.research import committee_advance_job
from orchestration.resources import DynamoResource, utcnow

BATCH_TERMINAL = frozenset({"ended", "completed"})
BATCH_EXPIRY = timedelta(hours=24)
QUARANTINE_SYSTEMIC_THRESHOLD = 20
BUDGET_WARN = 0.7
BUDGET_PAGE = 1.0
CLUSTERING_TRIGGER = 150  # new resolved failures between clustering runs


def advance_run_request(batch_id: str, stage: str) -> RunRequest:
    """The run request that advances one finished committee batch stage."""
    return RunRequest(
        run_key=f"advance-{batch_id}",
        run_config={
            "ops": {"advance_committee_stage": {"config": {"batch_id": batch_id, "stage": stage}}}
        },
    )


@sensor(job=committee_advance_job, minimum_interval_seconds=120)
def batch_completion_sensor(
    context: SensorEvaluationContext, dynamo: DynamoResource
) -> SensorResult | SkipReason:
    """Advance committee stages as their Message Batches finish (doc 03 §3)."""
    pending = dynamo.pending_batches()
    if not pending:
        return SkipReason("no submitted batches")
    model = dynamo.research_model()
    if model is None:
        return SkipReason("no research model configured")

    run_requests: list[RunRequest] = []
    now = utcnow()
    for row in pending:
        batch_id, stage = str(row["batch_id"]), str(row["stage"])
        status = model.batch_status(batch_id)
        if status in BATCH_TERMINAL:
            run_requests.append(advance_run_request(batch_id, stage))
        elif now - row["submitted_at"] > BATCH_EXPIRY:
            # doc 12 §7: batch expiry is a warn; resubmission of failures is
            # handled via deterministic custom_ids on the next nightly run.
            dynamo.alert(
                "warn",
                "batch_expiry",
                f"committee batch {batch_id} ({stage}) unresolved > 24h",
            )
            dynamo.mark_batch_expired(batch_id)
        else:
            context.log.info("batch %s (%s) still %s", batch_id, stage, status)
    return SensorResult(run_requests=run_requests)


@sensor(minimum_interval_seconds=600)
def quarantine_alert_sensor(
    context: SensorEvaluationContext, dynamo: DynamoResource
) -> SkipReason | None:
    """Systemic data-quality page: > 20 symbols quarantined in 24h (doc 02 §6)."""
    count = dynamo.quarantine_count_24h()
    if count <= QUARANTINE_SYSTEMIC_THRESHOLD:
        return SkipReason(f"{count} quarantined symbols in 24h (threshold ok)")
    day = utcnow().date().isoformat()
    if context.cursor == day:
        return SkipReason("already paged today")
    dynamo.alert(
        "page",
        "data_quality_systemic",
        (
            f"{count} symbols quarantined in the last 24h "
            f"(> {QUARANTINE_SYSTEMIC_THRESHOLD}) — probable provider-level issue; "
            "signals for them are already suppressed (runbook: provider-outage)"
        ),
    )
    context.update_cursor(day)
    return None


@sensor(minimum_interval_seconds=900)
def budget_sensor(context: SensorEvaluationContext, dynamo: DynamoResource) -> SkipReason | None:
    """Claude budget alerts: >= 70% warn, >= 100% page (doc 03 §5, doc 12 §7)."""
    utilization = dynamo.cost_meter().budget_utilization()
    month = utcnow().strftime("%Y-%m")
    if utilization >= BUDGET_PAGE:
        level = "page"
    elif utilization >= BUDGET_WARN:
        level = "warn"
    else:
        return SkipReason(f"budget utilization {utilization:.0%}")
    cursor = f"{month}:{level}"
    if context.cursor == cursor:
        return SkipReason(f"already alerted {level} this month")
    if level == "page":
        message = (
            f"Claude monthly budget exhausted ({utilization:.0%}); the cost meter now "
            "refuses new batch submissions — research-only stop, orders and stops "
            "untouched (runbook: budget-exhaustion)"
        )
    else:
        message = (
            f"Claude budget at {utilization:.0%} of monthly cap — reduce the deep-dive "
            "cap or accept the degradation date (doc 03 §5 knobs)"
        )
    dynamo.alert(level, f"claude_budget_{level}", message)
    context.update_cursor(cursor)
    return None


@sensor(job=failure_clustering_job, minimum_interval_seconds=3600)
def failure_clustering_sensor(
    context: SensorEvaluationContext, dynamo: DynamoResource
) -> SensorResult | SkipReason:
    """Trigger failure clustering every ~150 new resolved failures (plan
    Phase 6).

    Watermark contract: the failure_clustering asset records how many failure
    experiences it saw as the durable ``metrics_daily`` row
    ``learning.cluster_watermark`` (read back via
    ``DynamoResource.cluster_watermark``, max over runs). This sensor fires
    when (current failure count - watermark) reaches the threshold; the
    count-keyed ``run_key`` makes repeated evaluations before the run
    completes idempotent (Dagster dedupes identical run keys), and once the
    asset advances the watermark the sensor goes quiet again. The triggered
    job only ever OPENS ImprovementProposals — a human approves them in the
    dashboard, nothing auto-applies.
    """
    count = dynamo.failure_experience_count()
    watermark = dynamo.cluster_watermark()
    new = count - watermark
    if new < CLUSTERING_TRIGGER:
        return SkipReason(
            f"{new} new resolved failures since last clustering "
            f"(< {CLUSTERING_TRIGGER}; total {count}, watermark {watermark})"
        )
    return SensorResult(run_requests=[RunRequest(run_key=f"failure-clustering-{count}")])


ALL_SENSORS = [
    batch_completion_sensor,
    quarantine_alert_sensor,
    budget_sensor,
    failure_clustering_sensor,
]
