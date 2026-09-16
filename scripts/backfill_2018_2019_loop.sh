#!/usr/bin/env bash
# Chip away at the 2018-2019 panel backfill within Tiingo's free-tier quota.
#
# Tiingo cut us off after ~261 symbols in one burst, and the live eod_bars
# asset draws on the SAME quota — so this sleeps between bursts rather than
# retrying hard, leaving headroom for production ingestion. The backfill CLI
# skips symbols that already cover the range, so every pass is a no-op for
# work already done and the loop is safe to re-run or restart.
#
# Stops early once coverage stops improving (two consecutive passes that add
# no new symbols means the remaining names have no Tiingo history, not that
# we are throttled).
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=artifacts/logs/bf_bars_2018_2019_loop.log
SLEEP_S=${SLEEP_S:-3900}   # 65 min — past a plausible hourly reset
PASSES=${PASSES:-12}
stale=0

covered() {
  docker exec ai-investing-platform-postgres-1 psql -U dynamo -d dynamo_dev -t -A -c \
    "select count(distinct symbol) from bars_1d where ts >= '2018-01-01' and ts < '2020-01-01';" 2>/dev/null | tr -d ' '
}

for i in $(seq 1 "$PASSES"); do
  before=$(covered)
  echo "=== pass $i/$PASSES starting $(date -u +%FT%TZ) — $before symbols cover 2018-2019 ===" >> "$LOG"
  .venv/bin/python -m dynamo.cli.backfill bars \
    --start 2018-01-01 --end 2019-12-31 \
    --symbols-file config/training_universe_wide.txt \
    --provider tiingo >> "$LOG" 2>&1
  after=$(covered)
  echo "=== pass $i done $(date -u +%FT%TZ) — $before -> $after symbols ===" >> "$LOG"

  if [ "$after" = "$before" ]; then
    stale=$((stale + 1))
    if [ "$stale" -ge 2 ]; then
      echo "=== no progress in 2 consecutive passes; remaining symbols likely have no Tiingo history. stopping. ===" >> "$LOG"
      break
    fi
  else
    stale=0
  fi

  [ "$i" -lt "$PASSES" ] && sleep "$SLEEP_S"
done
echo "=== loop finished $(date -u +%FT%TZ), final coverage: $(covered) symbols ===" >> "$LOG"
