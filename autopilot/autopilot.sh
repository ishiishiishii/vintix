#!/usr/bin/env bash
# Scheduled headless Cursor agent for this repo (runs inside the container).
#
#   autopilot/autopilot.sh on [interval_hours]   # enable and start the scheduler (first check now)
#   autopilot/autopilot.sh off [--now]           # disable (--now also stops a running AI tick)
#   autopilot/autopilot.sh status
#   autopilot/autopilot.sh check                 # would the next check call the AI? (no AI call)
#   autopilot/autopilot.sh run-once              # one AI tick now even if nothing changed
#
# Every INTERVAL_HOURS a cheap check (fingerprint.py: issues, running jobs, pipeline progress,
# git HEADs) runs first; the AI (`agent -p` with PROMPT.md) is called only if something changed
# since the last AI tick. The scheduler turns itself off when it keeps waiting on the user.
# Secrets come from /workspace/.env (GH_TOKEN, optional CURSOR_API_KEY). Settings: config.env.
set -uo pipefail

AP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$AP_DIR/.." && pwd)"
STATE_DIR="$AP_DIR/state"
LOG_DIR="$AP_DIR/logs"
ENABLED="$STATE_DIR/ENABLED"
DAEMON_PID="$STATE_DIR/daemon.pid"
TICK_PID="$STATE_DIR/tick.pid"
LAST_FP="$STATE_DIR/last_fingerprint"
IDLE_CHECKS="$STATE_DIR/idle_checks"
WAITING_TICKS="$STATE_DIR/waiting_ticks"
OUTCOME="$STATE_DIR/last_outcome"
WAITING_ON="$STATE_DIR/waiting_on"
AUTO_OFF_REASON="$STATE_DIR/auto_off_reason"
CHECK_LOG="$LOG_DIR/checks.log"
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
now() { date '+%Y-%m-%d %H:%M:%S'; }
count() { [ -f "$1" ] && cat "$1" || echo 0; }
ai_ticks_today() { ls "$LOG_DIR"/tick_"$(date +%Y%m%d)"_*.log 2>/dev/null | wc -l; }

fingerprint() { (cd "$WORKSPACE" && python3 "$AP_DIR/fingerprint.py" 2>&1); }

auto_off() {
  local reason="$1"
  rm -f "$ENABLED"
  echo "$(now) $reason" > "$AUTO_OFF_REASON"
  echo "[$(now)] AUTO-OFF: $reason" | tee -a "$CHECK_LOG"
  if [ -n "${NOTIFY_ISSUE:-}" ]; then
    local waiting=""
    [ -f "$WAITING_ON" ] && waiting="$(tr '\n' ' ' < "$WAITING_ON")"
    "$HOME/.local/bin/gh" issue comment "$NOTIFY_ISSUE" -R ishiishiishii/vintix --body \
"**autopilot を自動停止しました（$(now)）**
- 理由: $reason
- 判断待ちの Issue: ${waiting:-（記録なし）}
- 再開: 判断をコメントするか \`autopilot\` ラベルを付けたうえで、コンテナで \`autopilot/autopilot.sh on\`" >/dev/null 2>&1 || true
  fi
}

ai_tick() {
  if [ -z "${CURSOR_API_KEY:-}" ] && "$AGENT_BIN" status 2>&1 | grep -qi "not logged in"; then
    echo "[autopilot] Cursor CLI is not logged in (run: agent login) and CURSOR_API_KEY is unset; skipping" >&2
    return 1
  fi
  local ts log rc
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/tick_$ts.log"
  rm -f "$OUTCOME" "$WAITING_ON"
  echo "[$(now)] AI tick $ts -> ${log#"$WORKSPACE"/}" | tee -a "$CHECK_LOG"
  (
    cd "$WORKSPACE" || exit 1
    timeout "$((TICK_TIMEOUT_MIN * 60))" "$AGENT_BIN" -p --force --trust --approve-mcps \
      --workspace "$WORKSPACE" --output-format text \
      ${AGENT_MODEL:+--model "$AGENT_MODEL"} \
      "$(sed -e "s/{{MAX_NEW_GPU_HOURS}}/$MAX_NEW_GPU_HOURS/g" -e "s/{{TICK_TS}}/$ts/g" "$AP_DIR/PROMPT.md")"
  ) > "$log" 2>&1 &
  echo $! > "$TICK_PID"
  wait "$(cat "$TICK_PID")"
  rc=$?
  rm -f "$TICK_PID"
  echo "[autopilot] tick $ts finished rc=$rc" >> "$log"
  # Fingerprint after the tick, so the agent's own comments/commits do not count as "new" next time.
  fingerprint | cut -d' ' -f1 > "$LAST_FP"
  echo 0 > "$IDLE_CHECKS"
  local outcome
  outcome="$(head -1 "$OUTCOME" 2>/dev/null | tr -d '[:space:]')"
  if [ "$outcome" = "waiting" ]; then
    echo $(( $(count "$WAITING_TICKS") + 1 )) > "$WAITING_TICKS"
  else
    echo 0 > "$WAITING_TICKS"
  fi
  echo "[$(now)] AI tick $ts done rc=$rc outcome=${outcome:-unknown} waiting_streak=$(count "$WAITING_TICKS")" | tee -a "$CHECK_LOG"
  if [ "$(count "$WAITING_TICKS")" -ge "$AUTO_OFF_WAITING_TICKS" ]; then
    auto_off "AI が ${AUTO_OFF_WAITING_TICKS} 回続けて「判断待ちで進められない」と報告した"
  fi
  return $rc
}

# One scheduled check: call the AI only if something changed (or when forced).
check_and_maybe_tick() {
  local force="${1:-}"
  load_secrets
  if alive "$TICK_PID"; then
    echo "[$(now)] skip: previous AI tick still running (pid $(cat "$TICK_PID"))" | tee -a "$CHECK_LOG"
    return 0
  fi
  local fp_line fp
  fp_line="$(fingerprint)"
  fp="${fp_line%% *}"
  if [ -z "$force" ] && [ -f "$LAST_FP" ] && [ "$fp" = "$(cat "$LAST_FP")" ]; then
    echo $(( $(count "$IDLE_CHECKS") + 1 )) > "$IDLE_CHECKS"
    echo "[$(now)] no change ($fp) -> AI not called (idle checks: $(count "$IDLE_CHECKS"))" | tee -a "$CHECK_LOG"
    if [ "$(count "$IDLE_CHECKS")" -ge "$AUTO_OFF_IDLE_CHECKS" ]; then
      auto_off "${AUTO_OFF_IDLE_CHECKS} 回続けて変化なし（Issue・実験・コミットに動きがない）"
    fi
    return 0
  fi
  if [ -z "$force" ] && [ "$(ai_ticks_today)" -ge "$MAX_AI_TICKS_PER_DAY" ]; then
    echo "[$(now)] changed but daily AI limit reached ($MAX_AI_TICKS_PER_DAY) -> wait for tomorrow" | tee -a "$CHECK_LOG"
    return 0
  fi
  echo "[$(now)] change detected: ${fp_line}" >> "$CHECK_LOG"
  ai_tick
}

daemon() {
  echo $$ > "$DAEMON_PID"
  while [ -f "$ENABLED" ]; do
    check_and_maybe_tick || true
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
    rm -f "$AUTO_OFF_REASON"
    echo 0 > "$IDLE_CHECKS"; echo 0 > "$WAITING_TICKS"
    if alive "$DAEMON_PID"; then
      echo "autopilot already running (pid $(cat "$DAEMON_PID")), every ${INTERVAL_HOURS}h"
    else
      setsid nohup "$0" _daemon >> "$LOG_DIR/daemon.log" 2>&1 < /dev/null &
      sleep 1
      echo "autopilot ON: check now, then every ${INTERVAL_HOURS}h; AI is called only when something changed (logs: autopilot/logs/)"
    fi
    ;;
  off)
    rm -f "$ENABLED"
    alive "$DAEMON_PID" && kill "$(cat "$DAEMON_PID")" 2>/dev/null
    rm -f "$DAEMON_PID"
    if [ "${2:-}" = "--now" ] && alive "$TICK_PID"; then
      pkill -TERM -P "$(cat "$TICK_PID")" 2>/dev/null; kill "$(cat "$TICK_PID")" 2>/dev/null
      rm -f "$TICK_PID"
      echo "autopilot OFF (running AI tick stopped)"
    else
      echo "autopilot OFF$(alive "$TICK_PID" && echo ' (current AI tick will finish; use off --now to stop it)')"
    fi
    ;;
  status)
    echo "enabled : $([ -f "$ENABLED" ] && echo yes || echo no)$([ -f "$AUTO_OFF_REASON" ] && echo " (auto-off: $(cat "$AUTO_OFF_REASON"))")"
    echo "daemon  : $(alive "$DAEMON_PID" && echo "running (pid $(cat "$DAEMON_PID"))" || echo stopped)"
    echo "AI tick : $(alive "$TICK_PID" && echo "running (pid $(cat "$TICK_PID"))" || echo idle); today $(ai_ticks_today)/${MAX_AI_TICKS_PER_DAY}"
    echo "interval: ${INTERVAL_HOURS}h; auto-off after ${AUTO_OFF_IDLE_CHECKS} unchanged checks or ${AUTO_OFF_WAITING_TICKS} 'waiting' AI ticks (now: $(count "$IDLE_CHECKS") / $(count "$WAITING_TICKS"))"
    if [ -f "$CHECK_LOG" ]; then echo "recent checks:"; tail -4 "$CHECK_LOG"; fi
    ;;
  check)
    load_secrets
    fp_line="$(fingerprint)"
    if [ -f "$LAST_FP" ] && [ "${fp_line%% *}" = "$(cat "$LAST_FP")" ]; then echo "no change -> AI would not be called"; else echo "changed -> AI would be called"; fi
    echo "$fp_line"
    ;;
  run-once) check_and_maybe_tick force ;;
  _check) check_and_maybe_tick ;;
  _daemon) daemon ;;
  *) echo "usage: $0 on [hours] | off [--now] | status | check | run-once" >&2; exit 2 ;;
esac
