#!/usr/bin/env bash
# Always-on Dagster for the paper/live host: `start | stop | status | restart`.
#
# Why not `dagster dev` (see docs/ops-log.md, 2026-09-23):
#   * without DAGSTER_HOME it silently creates a throwaway .tmp_dagster_home_*
#     instance with NO schedules enabled — the paper run then fires nothing;
#   * it always writes its workspace file into a macOS temp dir that is purged
#     after ~3 days, and the daemon dies on its next workspace refresh.
# This script runs the two production processes (dagster-daemon +
# dagster-webserver) against the repo's persistent .dagster_home and
# workspace.yaml, each under a restart loop, with size-capped logs.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

export DAGSTER_HOME="$ROOT/.dagster_home"
RUN_DIR="$DAGSTER_HOME/host"
LOG_DIR="$ROOT/artifacts/logs"
DAEMON_LOG="$LOG_DIR/dagster-daemon.log"
WEB_LOG="$LOG_DIR/dagster-webserver.log"
LOG_MAX_BYTES=$((200 * 1024 * 1024))
PORT="${DAGSTER_PORT:-3000}"
BIN="$ROOT/.venv/bin"

mkdir -p "$DAGSTER_HOME" "$RUN_DIR" "$LOG_DIR"
# An empty dagster.yaml marks the directory as the intended instance and
# silences the "no dagster.yaml" fallback warning.
[ -f "$DAGSTER_HOME/dagster.yaml" ] || : > "$DAGSTER_HOME/dagster.yaml"

rotate() {  # keep one previous generation; a crash-looping daemon once wrote 16.5 GB
  local f="$1"
  if [ -f "$f" ] && [ "$(stat -f%z "$f")" -gt "$LOG_MAX_BYTES" ]; then
    mv -f "$f" "$f.1"
  fi
}

alive() { [ -f "$RUN_DIR/$1.pid" ] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

supervise() {  # name log cmd... — restart on exit, rotating the log between runs
  local name="$1" log="$2"; shift 2
  (
    trap '' HUP  # outlive the terminal/session that launched us
    trap 'kill "$child" 2>/dev/null; exit 0' TERM INT
    while true; do
      rotate "$log"
      echo "=== $(date -u +%FT%TZ) starting $name (DAGSTER_HOME=$DAGSTER_HOME) ===" >> "$log"
      "$@" >> "$log" 2>&1 &
      child=$!
      wait "$child" || true
      echo "=== $(date -u +%FT%TZ) $name exited; restarting in 15s ===" >> "$log"
      sleep 15
    done
  ) </dev/null >/dev/null 2>&1 &
  # Fully detached: holding the caller's stdout would tie the scheduler's
  # lifetime to whatever launched it.
  echo $! > "$RUN_DIR/$name.pid"
  disown "$!" 2>/dev/null || true
}

# Real dagster executables only. A looser pattern matched any shell whose
# command line merely MENTIONED them (e.g. `tail dagster-daemon.log`) and
# made start() refuse to run beside its own caller.
stray() { pgrep -f "bin/dagster-daemon run|bin/dagster-webserver|bin/dagster dev|-m dagster[ ._]|dagster\._daemon|-m dagster_webserver" || true; }

start() {
  if alive daemon || alive webserver; then
    echo "already running"; status; return 0
  fi
  if [ -n "$(stray)" ]; then
    echo "refusing to start: unmanaged dagster processes are running (two daemons" \
         "fight over heartbeats). Stop them first:"; stray | xargs ps -o pid=,command= -p
    exit 1
  fi
  supervise daemon "$DAEMON_LOG" "$BIN/dagster-daemon" run -w "$ROOT/workspace.yaml"
  supervise webserver "$WEB_LOG" "$BIN/dagster-webserver" -w "$ROOT/workspace.yaml" \
    -h 127.0.0.1 -p "$PORT"
  echo "started (DAGSTER_HOME=$DAGSTER_HOME, UI http://127.0.0.1:$PORT)"
}

stop() {
  for name in webserver daemon; do
    if alive "$name"; then
      local pid; pid="$(cat "$RUN_DIR/$name.pid")"
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$RUN_DIR/$name.pid"
  done
  for _ in $(seq 1 30); do [ -z "$(stray)" ] && break; sleep 1; done
  [ -z "$(stray)" ] || { stray | xargs kill -9 2>/dev/null || true; }
  echo "stopped"
}

status() {
  for name in daemon webserver; do
    if alive "$name"; then echo "$name: running (supervisor pid $(cat "$RUN_DIR/$name.pid"))"
    else echo "$name: NOT running"; fi
  done
  "$BIN/python" - <<'PY'
from dagster import DagsterInstance
from dagster._core.scheduler.instigation import InstigatorStatus
i = DagsterInstance.get()
st = i.all_instigator_state()
on = sum(s.status == InstigatorStatus.RUNNING for s in st)
print(f"instance: {i.root_directory}  schedules/sensors RUNNING: {on}/{len(st)}")
hb = i.get_daemon_heartbeats()
print("daemon heartbeats:", ", ".join(sorted(hb)) or "none")
PY
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; sleep 2; start ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status|restart"; exit 2 ;;
esac
