"""Daily activity report asset — runs in the 02:00 ET rollup chain.

Builds the plain-language ``DailyReport`` for the day, writes the markdown to
``{artifact_dir}/reports/daily/{YYYY-MM-DD}.md`` and upserts the rendered
markdown + JSON into ``daily_reports`` (served by the API and the dashboard
Activity Log page).
"""

from pathlib import Path

from dagster import AssetExecutionContext, MaterializeResult
from dagster import asset as dagster_asset
from sqlalchemy.dialects.postgresql import insert as pg_insert

from dynamo.persistence import schema
from orchestration.resources import DynamoResource, utcnow

REPORTING_GROUP = "reporting"


@dagster_asset(name="daily_report", group_name=REPORTING_GROUP, deps=["metrics_daily"])
def daily_report_asset(context: AssetExecutionContext, dynamo: DynamoResource) -> MaterializeResult:
    """Build + render today's activity report; store the file and the row."""
    from dynamo.reporting.daily import render_markdown
    from dynamo.reporting.sql_source import build_daily_report_for_engine

    settings = dynamo.settings()
    day = utcnow().date()
    report = build_daily_report_for_engine(
        dynamo.engine(), day, settings.claude_monthly_budget_usd
    )
    markdown = render_markdown(report)

    out_dir = Path(settings.artifact_dir) / "reports" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day.isoformat()}.md"
    path.write_text(markdown, encoding="utf-8")

    t = schema.daily_reports
    values = {
        "as_of": day,
        "payload": report.model_dump(mode="json"),
        "markdown": markdown,
        "created_at": utcnow(),
    }
    stmt = pg_insert(t).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[t.c.as_of],
        set_={k: v for k, v in values.items() if k != "as_of"},
    )
    with dynamo.engine().begin() as cx:
        cx.execute(stmt)
    context.log.info("daily_report: wrote %s", path)
    return MaterializeResult(
        metadata={
            "path": str(path),
            "day": day.isoformat(),
            "decisions": len(report.decisions.buys) + len(report.decisions.sells),
            "trades": len(report.trades),
            "claude_today_usd": report.cost.total_today_usd,
        }
    )


REPORTING_ASSETS = [daily_report_asset]
