#!/bin/bash
# 5-minute monitor for the resumed K1 VLN-CE training.
#
# Watches /tmp/k1_booster_train.log and writes alerts to MONITOR_LOG.
#
# Alerts:
#   (1) action_rate reward term < -1e6  — early-collapse detection
#       (legged-loco v3 diverged to ~-3e15; booster_train uses weight=-1.0
#        so a stable run stays in the -0.1..-0.5 range)
#   (2) Train mean_reward < 50% of peak  — regression detection
#   (3) tmux session 'k1_vlnce_train' gone  — completion or crash
#
# Auto-stop (matches the original 15k stall config):
#   (4) reward hasn't improved by > EPS for STALL_ITERS iters -> kill session
#
# Idempotent state is kept in /tmp/k1_vlnce_resume_state.txt

set -u

LOG=/tmp/k1_booster_train.log
MONITOR_LOG=/tmp/k1_vlnce_monitor.log
STATE=/tmp/k1_vlnce_resume_state.txt
CHECK_INTERVAL=300         # 5 min
STALL_ITERS=15000          # iters with no improvement -> auto-stop
EPS=0.5                    # min reward delta to count as "improved"
ACTION_RATE_FLOOR="-1e6"   # below this -> collapse alert
DROP_FRAC=0.5              # peak * DROP_FRAC is the regression floor

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

log() { echo "[$(date +%F\ %H:%M:%S)] $*" | tee -a "$MONITOR_LOG"; }
alert() { echo "[$(date +%F\ %H:%M:%S)] ALERT: $*" | tee -a "$MONITOR_LOG"; }

log "monitor started — interval=${CHECK_INTERVAL}s log=$LOG"

while true; do
  sleep $CHECK_INTERVAL

  # (3) crash/completion detection
  if ! "$TMUX_BIN" has-session -t k1_vlnce_train 2>/dev/null; then
    # distinguish completion vs crash by looking at end of log
    if [[ -f "$LOG" ]]; then
      LAST_ITER=$(grep -E "Learning iteration" "$LOG" | tail -1 | sed -E 's/.*iteration ([0-9]+)\/([0-9]+).*/\1 \2/')
      TAIL=$(tail -20 "$LOG" | tr '\n' ' ')
      if echo "$TAIL" | grep -qiE "traceback|error|cuda|killed|segmentation"; then
        alert "training tmux session gone — looks like a CRASH (iter=${LAST_ITER:-?}). Tail: $(echo "$TAIL" | head -c 400)"
      else
        alert "training tmux session gone — looks like COMPLETION (iter=${LAST_ITER:-?})"
      fi
    else
      alert "training tmux session gone AND log missing"
    fi
    break
  fi

  if [[ ! -f "$LOG" ]]; then
    log "log $LOG not present yet, waiting..."
    continue
  fi

  LATEST_ITER=$(grep -E "Learning iteration" "$LOG" | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
  # "Mean reward:" appears with right-padded label; value is the last whitespace token on that line
  LATEST_REWARD=$(grep -E "^\s+Mean reward:" "$LOG" | tail -1 | awk '{print $NF}')
  # Booster env reward terms have "/" in their key, so rsl_rl prints them as
  # "Episode_Reward/action_rate: -0.18" (not "Mean episode action_rate: ...")
  LATEST_AR=$(grep -E "Episode_Reward/action_rate:" "$LOG" | tail -1 | awk '{print $NF}')

  if [[ -z "${LATEST_ITER:-}" || -z "${LATEST_REWARD:-}" ]]; then
    log "no iter/reward yet (iter=${LATEST_ITER:-?} reward=${LATEST_REWARD:-?}), waiting..."
    continue
  fi

  if [[ -f "$STATE" ]]; then
    read PEAK_ITER PEAK_REWARD < "$STATE"
  else
    PEAK_ITER=$LATEST_ITER
    PEAK_REWARD=$LATEST_REWARD
    echo "$PEAK_ITER $PEAK_REWARD" > "$STATE"
  fi

  IMPROVED=$(awk -v cur="$LATEST_REWARD" -v peak="$PEAK_REWARD" -v eps="$EPS" \
              'BEGIN { print (cur - peak > eps) ? "1" : "0" }')
  if [[ "$IMPROVED" == "1" ]]; then
    PEAK_ITER=$LATEST_ITER
    PEAK_REWARD=$LATEST_REWARD
    echo "$PEAK_ITER $PEAK_REWARD" > "$STATE"
  fi

  STALL=$(( LATEST_ITER - PEAK_ITER ))
  log "iter=$LATEST_ITER reward=$LATEST_REWARD action_rate=${LATEST_AR:-NA} peak=${PEAK_REWARD}@iter${PEAK_ITER} stall=${STALL}"

  # (1) action_rate collapse
  if [[ -n "${LATEST_AR:-}" ]]; then
    AR_COLLAPSE=$(awk -v ar="$LATEST_AR" -v floor="$ACTION_RATE_FLOOR" \
                    'BEGIN { print (ar+0 < floor+0) ? "1" : "0" }')
    if [[ "$AR_COLLAPSE" == "1" ]]; then
      alert "action_rate=${LATEST_AR} fell below ${ACTION_RATE_FLOOR} at iter=${LATEST_ITER} — COLLAPSE"
    fi
  fi

  # (2) reward regression > 50% from peak
  REGRESSED=$(awk -v cur="$LATEST_REWARD" -v peak="$PEAK_REWARD" -v frac="$DROP_FRAC" \
                'BEGIN { print (peak > 0 && cur < peak * frac) ? "1" : "0" }')
  if [[ "$REGRESSED" == "1" ]]; then
    alert "reward=${LATEST_REWARD} dropped below 50% of peak=${PEAK_REWARD} (iter=${LATEST_ITER}) — REGRESSION"
  fi

  # (4) 15k stall auto-stop
  if [[ "$STALL" -gt "$STALL_ITERS" ]]; then
    alert "plateau: ${STALL} iters with no improvement > ${EPS} since peak=${PEAK_REWARD}@iter${PEAK_ITER} — auto-stopping"
    "$TMUX_BIN" kill-session -t k1_vlnce_train 2>/dev/null || true
    break
  fi
done

log "monitor exited."
