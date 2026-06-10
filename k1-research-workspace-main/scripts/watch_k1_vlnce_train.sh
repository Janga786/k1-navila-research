#!/bin/bash
# Plateau watcher for the K1 VLN-CE training.
#
# Checks the training log every 10 min. Stops training if mean reward
# hasn't improved by more than EPS in STALL_ITERS iters, or if it drops
# by more than DROP_THRESHOLD from the running best. Records the best-
# reward checkpoint name to /tmp/k1_vlnce_best_ckpt.txt.
#
# Idempotent: re-running just re-reads state from log.

set -e

LOG=/tmp/k1_booster_train.log
STATE=/tmp/k1_vlnce_watch_state.txt
BEST=/tmp/k1_vlnce_best_ckpt.txt
CHECK_INTERVAL=600         # 10 min
STALL_ITERS=15000          # iters with no improvement -> stop
EPS=0.5                    # min reward delta to count as "improved"
DROP_THRESHOLD=20.0        # if reward drops > DROP from best -> stop

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

log() { echo "[$(date +%F\ %H:%M:%S)] $*"; }

while true; do
  sleep $CHECK_INTERVAL

  if [[ ! -f "$LOG" ]]; then
    log "log $LOG not found yet, waiting..."
    continue
  fi

  if ! "$TMUX_BIN" has-session -t k1_vlnce_train 2>/dev/null; then
    log "tmux session k1_vlnce_train gone — training exited."
    break
  fi

  # Latest iteration + reward (RSL-RL prints "Learning iteration N/M" and "Mean reward: X")
  LATEST_ITER=$(grep -E "Learning iteration" "$LOG" | tail -1 | sed -E 's/.*iteration ([0-9]+)\/.*/\1/')
  LATEST_REWARD=$(grep -E "^\s+Mean reward:" "$LOG" | tail -1 | awk '{print $3}')

  if [[ -z "$LATEST_ITER" || -z "$LATEST_REWARD" ]]; then
    log "no iter/reward yet (iter=$LATEST_ITER reward=$LATEST_REWARD), waiting..."
    continue
  fi

  if [[ -f "$STATE" ]]; then
    read BEST_ITER BEST_REWARD < "$STATE"
  else
    BEST_ITER=$LATEST_ITER
    BEST_REWARD=$LATEST_REWARD
  fi

  IMPROVED=$(awk -v cur="$LATEST_REWARD" -v best="$BEST_REWARD" -v eps="$EPS" \
              'BEGIN { print (cur - best > eps) ? "1" : "0" }')
  if [[ "$IMPROVED" == "1" ]]; then
    log "improved: iter=$LATEST_ITER reward=$LATEST_REWARD (prev best $BEST_REWARD @ $BEST_ITER)"
    BEST_ITER=$LATEST_ITER
    BEST_REWARD=$LATEST_REWARD
    echo "$BEST_ITER $BEST_REWARD" > "$STATE"
    # Save best checkpoint name
    NEAREST=$(( (LATEST_ITER / 2000) * 2000 ))
    echo "model_${NEAREST}.pt at iter=$LATEST_ITER reward=$LATEST_REWARD" > "$BEST"
  fi

  STALL=$(( LATEST_ITER - BEST_ITER ))
  DROP=$(awk -v cur="$LATEST_REWARD" -v best="$BEST_REWARD" \
              'BEGIN { print (best - cur > 0) ? best - cur : 0 }')

  log "iter=$LATEST_ITER reward=$LATEST_REWARD best=$BEST_REWARD@iter$BEST_ITER stall=$STALL drop=$DROP"

  if [[ "$STALL" -gt "$STALL_ITERS" ]]; then
    log "STOPPING: plateau ($STALL iters with no improvement > $EPS, best $BEST_REWARD @ iter $BEST_ITER)"
    "$TMUX_BIN" kill-session -t k1_vlnce_train
    break
  fi

  IS_DROP=$(awk -v drop="$DROP" -v thresh="$DROP_THRESHOLD" \
              'BEGIN { print (drop > thresh) ? "1" : "0" }')
  if [[ "$IS_DROP" == "1" ]]; then
    log "STOPPING: reward dropped $DROP below best ($BEST_REWARD @ iter $BEST_ITER)"
    "$TMUX_BIN" kill-session -t k1_vlnce_train
    break
  fi
done

log "watcher exited. Best: $(cat $BEST 2>/dev/null || echo 'no record')"
