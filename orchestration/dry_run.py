"""Wiring dry-run: proves the definitions load and the composition root builds.

``make pipeline-dry`` executes this job. It needs no database and no API keys —
it reports which providers are configured and which nightly capabilities that
enables (doc 01 §6: degrade per-asset, never all-or-nothing).
"""

from dagster import OpExecutionContext, job, op

from orchestration.resources import DynamoResource


@op
def check_wiring(context: OpExecutionContext, dynamo: DynamoResource) -> None:
    s = dynamo.settings()
    configured = {
        "anthropic": bool(s.anthropic_api_key),
        "alpaca": bool(s.alpaca_api_key and s.alpaca_api_secret),
        "tiingo": bool(s.tiingo_api_key),
        "finnhub": bool(s.finnhub_api_key),
        "fred": bool(s.fred_api_key),
        "kalshi": bool(s.kalshi_api_key_id and s.kalshi_private_key_path),
        "polymarket": True,  # keyless reads
        "edgar": True,  # declared User-Agent only
    }
    for name, ok in sorted(configured.items()):
        context.log.info("provider %-11s %s", name, "configured" if ok else "MISSING (asset skips)")
    if not configured["anthropic"]:
        context.log.warning(
            "no ANTHROPIC_API_KEY: nights run quant-only (degraded mode, doc 12 §8)"
        )
    context.log.info("env=%s budget=$%.0f/mo", s.dynamo_env.value, s.claude_monthly_budget_usd)


@job
def dry_run_job() -> None:
    """`make pipeline-dry`: definitions load + composition-root smoke."""
    check_wiring()
