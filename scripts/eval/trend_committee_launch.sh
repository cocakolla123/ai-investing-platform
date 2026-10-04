#!/usr/bin/env bash
# Pre-registered trend + committee run (docs/preregistration_trend_committee.md).
# Re-running resumes (deterministic run id skips finished days).
set -euo pipefail
cd "$(dirname "$0")/../.."
CLAUDE_CODE_ARBITER_FALLBACK="" HF_HUB_OFFLINE=1 .venv/bin/python -m dynamo.cli.replay run \
  --mode committee --start 2026-08-03 --end 2026-08-21 \
  --deep-dives-per-day 4 --universe-tier 1 --discovery off \
  --quant-model-version trend-rule-v1 --no-require-shipping-bundle \
  --no-macro-digest --no-within-run-learning --backend claude_code \
  --code-version trend-committee-6f6d039 --max-usd 250 --run-label trend-committee \
  >> artifacts/logs/trend_committee.log 2>&1 < /dev/null
