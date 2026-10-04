#!/usr/bin/env bash
# Sequenced re-runs on the fixed bars (1c6c04d), one heavy job at a time so
# Postgres (2.5 GiB container) isn't overloaded: wait for the panel-cache
# shards, then Phase 5 / v10 from cache, then true discovery, then wide.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "[w]arm_panel_cache" >/dev/null; do sleep 60; done
echo "=== phase5 $(date)" && PYTHONPATH=. .venv/bin/python scripts/eval/ml_baselines.py --tag barfix \
  > artifacts/logs/phase5_barfix.log 2>&1 < /dev/null
CV=(--code-version rerun-52c1be7 --no-require-shipping-bundle)
W=(--start 2026-03-03 --end 2026-08-14 --ballast SPY --min-feature-coverage 0.35)
echo "=== disc-true $(date)" && .venv/bin/python -m dynamo.cli.replay run --mode quant \
  --universe-tier 1 --universe-tier 3 --discovery code \
  --discovery-min-dollar-volume 2000000 --discovery-min-price 5 \
  --discovery-promotions-per-day 15 --discovery-promotion-days 30 \
  --quant-model-version campaign-v8 --initial-equity 100000 "${W[@]}" "${CV[@]}" \
  --run-label rerun-disc-quant-100k-true >> artifacts/logs/rerun_disc_true.log 2>&1 < /dev/null
echo "=== wide $(date)" && .venv/bin/python -m dynamo.cli.replay run --mode quant \
  --universe-tier 1 --universe-tier 2 --universe-tier 3 --discovery code \
  --discovery-min-dollar-volume 2000000 --discovery-min-price 5 \
  --discovery-promotions-per-day 60 --discovery-promotion-days 30 \
  --quant-model-version campaign-v8 --initial-equity 100000 "${W[@]}" "${CV[@]}" \
  --run-label rerun-wide-quant-100k >> artifacts/logs/rerun_wide.log 2>&1 < /dev/null
echo "=== all done $(date)"
