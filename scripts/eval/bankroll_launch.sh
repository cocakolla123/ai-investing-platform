#!/usr/bin/env bash
# Pre-registered bankroll-profile comparison (docs/preregistration_bankroll_profile.md):
# flat vs --bankroll-profile v1 at $1k / $10k / $100k, one run at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f "[r]un-label rerun3-(committee|core)" >/dev/null; do sleep 30; done
for EQ in 1000 10000 100000; do
  for ARM in flat v1; do
    EXTRA=(); [ "$ARM" = v1 ] && EXTRA=(--bankroll-profile v1)
    echo "=== $EQ $ARM $(date)"
    .venv/bin/python -m dynamo.cli.replay run --mode portfolio --source-run-id replay-63220a3a5e9a \
      --universe-tier 1 --universe-tier 3 --start 2026-03-03 --end 2026-08-14 --ballast SPY \
      --min-feature-coverage 0.35 --initial-equity "$EQ" --no-require-shipping-bundle \
      --code-version bankroll-5dd26dd --run-label "bankroll-$ARM-$EQ" ${EXTRA[@]+"${EXTRA[@]}"} \
      >> artifacts/logs/bankroll.log 2>&1 < /dev/null
  done
done
echo "=== done $(date)"
