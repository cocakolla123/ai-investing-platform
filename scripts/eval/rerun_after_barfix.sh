#!/usr/bin/env bash
# Re-run results made before the replay fixes — bar scale (1c6c04d), live-parity
# ADV and debated-name trading (8cc854a) — one heavy job at a time so Postgres
# (2.5 GiB container) isn't overloaded. Settings copied from each original
# run's stored config_json. No model calls anywhere.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=".venv/bin/python -m dynamo.cli.replay run"
CV=(--code-version rerun-5dd26dd --no-require-shipping-bundle)
W=(--start 2026-03-03 --end 2026-08-14 --ballast SPY --min-feature-coverage 0.35)

# 1. light: committee portfolio passes + core control — already run; wait if still running
while pgrep -f "[r]un-label rerun3-(committee|core)" >/dev/null; do sleep 30; done

# 2. heavy, after the panel-cache shards: Phase 5 / v10 from cache
while pgrep -f "[w]arm_panel_cache" >/dev/null; do sleep 60; done
echo "=== phase5 $(date)"
PYTHONPATH=. .venv/bin/python scripts/eval/ml_baselines.py --tag barfix \
  > artifacts/logs/phase5_barfix.log 2>&1 < /dev/null

# 3. true discovery, then wide-traded
echo "=== disc-true $(date)"
$R --mode quant --universe-tier 1 --universe-tier 3 --discovery code \
  --discovery-min-dollar-volume 2000000 --discovery-min-price 5 \
  --discovery-promotions-per-day 15 --discovery-promotion-days 30 \
  --quant-model-version campaign-v8 --initial-equity 100000 "${W[@]}" "${CV[@]}" \
  --run-label rerun3-disc-quant-100k-true >> artifacts/logs/rerun3_disc_true.log 2>&1 < /dev/null
# Wide-traded (orig replay-699b62040d45) dropped 2026-10-04 by the user: ~14h, and with a
# no-edge model it cannot answer "does trading wide help?". Re-test with a working model.
echo "=== all done $(date)"
