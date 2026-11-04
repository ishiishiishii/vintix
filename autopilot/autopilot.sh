#!/usr/bin/env bash
# Scheduled headless Cursor agent for this repo (runs inside the container).
#
#   autopilot/autopilot.sh on [interval_hours]   # enable and start the scheduler
#   autopilot/autopilot.sh off [--now]           # disable (--now also stops a running tick)
#   autopilot/autopilot.sh status
#   autopilot/autopilot.sh run-once              # one tick now, in the foreground
#
# Each tick runs `agent -p` with autopilot/PROMPT.md. Secrets come from /workspace/.env
# (CURSOR_API_KEY, GH_TOKEN). Settings: autopilot/config.env.
set -uo pipefail

AP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$AP_DIR/.." && pwd)"
STATE_DIR="$AP_DIR/state"
LOG_DIR="$AP_DIR/logs"
ENABLED="$STATE_DIR/ENABLED"
DAEMON_PID="$STATE_DIR/daemon.pid"
TICK_PID="$STATE_DIR/tick.pid"
AGENT_BIN="${AGENT_BIN:-$HOME/.local/bin/agent}"
mkdir -p "$STATE_DIR" "$LOG_DIR"

# shellcheck disable=SC1091
source "$AP_DIR/config.env"
[ -f "$STATE_DIR/interval_hours" ] && INTERVAL_HOURS="$(cat "$STATE_DIR/interval_hours")"

load_secrets() {
  if [ -f "$WORKSPACE/.env" ]; then set -a; source "$WORKSPACE/.env"; set +a; fi
  export PATH="$HOME/.local/bin:$PATH"
}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

tick() {
  load_secrets
  if [ -z "${CURSOR_API_KEY:-}" ]; then
    echo "[autopilot] CURSOR_API_KEY is not set in $WORKSPACE/.env; skipping tick" >&2
    return 1
  fi
  if alive "$TICK_PID"; then
    echo "[autopilot] previous tick still running (pid $(cat "$TICK_PID")); skipping" >&2
    return 0
  fi
  local ts log
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/tick_$ts.log"
  echo "[autopilot] tick $ts -> $log"
  (
    cd "$WORKSPACE" || exit 1
    timeout "$((TICK_TIMEOUT_MIN * 60))" "$AGENT_BIN" -p --force --trust --approve-mcps \
      --workspace "$WORKSPACE" --output-format text \
      ${AGENT_MODEL:+--model "$AGENT_MODEL"} \
      "$(sed -e "s/{{MAX_NEW_GPU_HOURS}}/$MAX_NEW_GPU_HOURS/g" -e "s/{{TICK_TS}}/$ts/g" "$AP_DIR/PROMPT.md")"
  ) > "$log" 2>&1 &
  echo $! > "$TICK_PID"
  wait "$(cat "$TICK_PID")"
  local rc=$?
  rm -f "$TICK_PID"
  echo "[autopilot] tick $ts finished rc=$rc" | tee -a "$log"
  return $rc
}

daemon() {
  echo $$ > "$DAEMON_PID"
  while [ -f "$ENABLED" ]; do
    tick || true
    local slept=0 total
    total="$(awk -v h="$INTERVAL_HOURS" 'BEGIN { printf "%d", h * 3600 }')"
    while [ -f "$ENABLED" ] && [ "$slept" -lt "$total" ]; do sleep 60; slept=$((slept + 60)); done
  done
  rm -f "$DAEMON_PID"
}

case "${1:-status}" in
  on)
    [ -n "${2:-}" ] && echo "$2" > "$STATE_DIR/interval_hours" && INTERVAL_HOURS="$2"
    touch "$ENABLED"
    if alive "$DAEMON_PID"; then
      echo "autopilot already running (pid $(cat "$DAEMON_PID")), interval ${INTERVAL_HOURS}h"
    else
      setsid nohup "$0" _daemon > "$LOG_DIR/daemon.log" 2>&1 < /dev/null &
      sleep 1
      echo "autopilot ON: first tick now, then every ${INTERVAL_HOURS}h (logs: autopilot/logs/)"
    fi
    ;;
  off)
    rm -f "$ENABLED"
    alive "$DAEMON_PID" && kill "$(cat "$DAEMON_PID")" 2>/dev/null
    rm -f "$DAEMON_PID"
    if [ "${2:-}" = "--now" ] && alive "$TICK_PID"; then
      pkill -TERM -P "$(cat "$TICK_PID")" 2>/dev/null; kill "$(cat "$TICK_PID")" 2>/dev/null
      rm -f "$TICK_PID"
      echo "autopilot OFF (running tick stopped)"
    else
      echo "autopilot OFF$(alive "$TICK_PID" && echo ' (current tick will finish; use off --now to stop it)')"
    fi
    ;;
  status)
    echo "enabled : $([ -f "$ENABLED" ] && echo yes || echo no)"
    echo "daemon  : $(alive "$DAEMON_PID" && echo "running (pid $(cat "$DAEMON_PID"))" || echo stopped)"
    echo "tick    : $(alive "$TICK_PID" && echo "running (pid $(cat "$TICK_PID"))" || echo idle)"
    echo "interval: ${INTERVAL_HOURS}h, tick timeout ${TICK_TIMEOUT_MIN}min, new-experiment budget ${MAX_NEW_GPU_HOURS} GPU-h"
    last="$(ls -t "$LOG_DIR"/tick_*.log 2>/dev/null | head -1)"
    if [ -n "$last" ]; then echo "last log: ${last#"$WORKSPACE"/}"; tail -3 "$last"; fi
    ;;
  run-once) tick ;;
  _daemon) daemon ;;
  *) echo "usage: $0 on [hours] | off [--now] | status | run-once" >&2; exit 2 ;;
esac
