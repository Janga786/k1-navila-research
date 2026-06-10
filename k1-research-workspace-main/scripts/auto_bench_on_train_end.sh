#!/bin/bash
# Fires the full VLN-CE benchmark when the K1 training session ends.
#
# Conservative: only fires if a checkpoint of iter >= MIN_ITER_FOR_BENCH is
# present (default 15000). Without this, a transient tmux hiccup early in
# training could fire a 12-hour benchmark on a not-yet-converged policy.
#
# NO pre-bench smoke — we've already validated integration manually. Pre-smoke
# was triggering race conditions where two Isaac Sim instances briefly ran
# concurrently with the still-active training, causing GPU contention.

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

BOOSTER_LOGS=~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce
LOG=/tmp/k1_vlnce_autobench.log
MIN_ITER_FOR_BENCH=15000

log() { echo "[$(date +%F\ %H:%M:%S)] $*" >> "$LOG"; }

log "auto-bench watcher started (MIN_ITER_FOR_BENCH=$MIN_ITER_FOR_BENCH)"

# Wait for training session to come up.
for i in $(seq 1 30); do
  if "$TMUX_BIN" has-session -t k1_vlnce_train 2>/dev/null; then
    log "training session 'k1_vlnce_train' detected"
    break
  fi
  sleep 10
done

# Wait for training session to end.
while "$TMUX_BIN" has-session -t k1_vlnce_train 2>/dev/null; do
  sleep 60
done

log "training session ended."

# Identify the run dir
RUN_DIR=$(ls -1dt "$BOOSTER_LOGS"/2026-05-18_*_k1_vlnce_v3 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
  log "ERROR: no training run dir found; aborting."
  exit 1
fi
RUN_SUBDIR=$(basename "$RUN_DIR")
log "run dir: $RUN_DIR"

# Pick the highest-iter checkpoint
LATEST_PT=$(ls -1 "$RUN_DIR"/model_*.pt 2>/dev/null | grep -oP 'model_\d+\.pt' \
            | sort -t_ -k2 -n | tail -1)
LATEST_ITER=$(echo "$LATEST_PT" | grep -oP '\d+' || echo "0")
log "latest checkpoint: $LATEST_PT (iter $LATEST_ITER)"

if (( LATEST_ITER < MIN_ITER_FOR_BENCH )); then
  log "ABORT: latest iter $LATEST_ITER < $MIN_ITER_FOR_BENCH. Training likely died early."
  log "Manual review needed — not firing the 12-hour benchmark on a half-baked policy."
  exit 1
fi

# Fire the full benchmark in tmux
log "launching full benchmark (1077 episodes) on $LATEST_PT..."
bash ~/Projects/k1_research/scripts/bench_k1_vlnce.sh full "$RUN_SUBDIR" 0 1077 >> "$LOG" 2>&1

# Wait for full benchmark to finish
log "waiting for benchmark tmux session 'bench_vlnce' to exit..."
while "$TMUX_BIN" has-session -t bench_vlnce 2>/dev/null; do
  sleep 120
done

log "benchmark session ended; aggregating results."
LOAD_RUN_NAME="booster_vlnce_${RUN_SUBDIR}"
RESULTS_DIR="$HOME/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_${LOAD_RUN_NAME}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac

python ~/Projects/k1_research/aggregate_k1_vision_results.py \
    --measurements-dir "$RESULTS_DIR/measurements" >> "$LOG" 2>&1 || true

log "DONE. Aggregation in $LOG; per-episode JSONs in $RESULTS_DIR."
