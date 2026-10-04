#!/usr/bin/env bash
# Committee portfolio passes on the fixed bars (reuse replay-63220a3a5e9a's verdicts).
set -euo pipefail
cd "$(dirname "$0")/../.."
for EQ in 1000 10000 100000; do
  .venv/bin/python -m dynamo.cli.replay run --mode portfolio --source-run-id replay-63220a3a5e9a \
    --universe-tier 1 --universe-tier 3 --start 2026-03-03 --end 2026-08-14 --ballast SPY \
    --min-feature-coverage 0.35 --initial-equity "$EQ" --code-version rerun-52c1be7 \
    --no-require-shipping-bundle --run-label "rerun-committee-portfolio-$EQ" \
    >> artifacts/logs/rerun_portfolio.log 2>&1 < /dev/null
done
