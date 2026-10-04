#!/usr/bin/env bash
# The three pre-registered trend-rule replays (docs/preregistration_trend_rule.md).
# Quant-only: no model calls. Run concurrently (no latency measurement here).
set -euo pipefail
cd "$(dirname "$0")/../.."
COMMON=(--mode quant --universe-tier 1 --universe-tier 3 --discovery off
  --initial-equity 100000 --ballast SPY --no-require-shipping-bundle
  --code-version trend-5970f24)
run() {  # name model start end
  .venv/bin/python -m dynamo.cli.replay run "${COMMON[@]}" --run-label "trend-$1" \
    --quant-model-version "$2" --start "$3" --end "$4" \
    >> "artifacts/logs/trend_$1.log" 2>&1 < /dev/null &
}
run long trend-rule-v1 2025-01-02 2026-08-14
run camp trend-rule-v1 2026-03-03 2026-08-14
run v8camp campaign-v8 2026-03-03 2026-08-14
wait
