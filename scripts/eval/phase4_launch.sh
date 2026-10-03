#!/usr/bin/env bash
# Run the two pre-registered Phase 4 arms ONE AFTER THE OTHER
# (docs/preregistration_eval_tracing.md): seq to completion, then par.
# Re-running resumes (deterministic run ids skip finished days).
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMON=(--mode committee --start 2026-08-03 --end 2026-08-07
  --deep-dives-per-day 4 --universe-tier 1 --discovery off
  --quant-model-version campaign-v8 --no-require-shipping-bundle
  --no-macro-digest --no-within-run-learning --backend claude_code
  --code-version phase4-ec9b97e --max-usd 250)

run_arm() {  # name, extra flags...
  local name="$1"; shift
  CLAUDE_CODE_ARBITER_FALLBACK="" HF_HUB_OFFLINE=1 .venv/bin/python -m dynamo.cli.replay run \
    "${COMMON[@]}" --run-label "phase4-${name}" "$@" \
    >> "artifacts/logs/phase4_${name}.log" 2>&1 < /dev/null
}

run_arm seq
run_arm par --parallel-advocates
