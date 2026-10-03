#!/usr/bin/env bash
# Launch the four pre-registered Phase 3 arms concurrently
# (docs/preregistration_eval_decisions.md). Re-running this script resumes
# every arm: days with persisted verdicts are skipped (deterministic run ids).
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMON=(--mode committee --start 2026-08-03 --end 2026-08-21
  --deep-dives-per-day 4 --universe-tier 1 --discovery off
  --quant-model-version campaign-v8 --no-require-shipping-bundle
  --no-macro-digest --no-within-run-learning --backend claude_code
  --code-version phase3-806d954 --max-usd 250)

launch() {  # name retrieval replicate
  local name="$1" mode="$2" rep="$3"
  CLAUDE_CODE_ARBITER_FALLBACK="" HF_HUB_OFFLINE=1 nohup .venv/bin/python -m dynamo.cli.replay run \
    "${COMMON[@]}" --run-label "phase3-${mode}" --journal-retrieval "$mode" --replicate "$rep" \
    >> "artifacts/logs/phase3_${name}.log" 2>&1 < /dev/null &
  echo "$name pid $!"
}

launch off off 0
launch vector vector 0
launch hybrid hybrid 0
launch vector-rep vector 1
