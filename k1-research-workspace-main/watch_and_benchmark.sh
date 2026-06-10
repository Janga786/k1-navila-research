#!/bin/bash
# Two-stage watcher:
#   STAGE 1: wait for k1train (v1, 2000 iters) to finish.
#   STAGE 2: launch k1train2 (v2, 10000 iters, --resume from v1).
#   STAGE 3: wait for k1train2 to finish.
#   STAGE 4: pick the better policy (v2 if present, else v1).
#   STAGE 5: kill remaining training tmux, launch benchmark.
#
# The user wants to use the full overnight 8h GPU window for training.
# v1 is the "safe baseline" (2000 iters @ ~1.7h from start).
# v2 is the overnight extension (10000 iters @ another ~7.5h from v1 end).

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib"
TMUX="$HOME/miniconda3/bin/tmux"

LEGGED_LOCO=~/Projects/k1_research/legged-loco
TARGET_DIR="$LEGGED_LOCO/logs/rsl_rl/k1_vision_rough"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ----------------------------------------------------------------------------
# STAGE 1 — wait for v1 (2000 iters)
# ----------------------------------------------------------------------------

V1_DIR=""
while [[ -z "$V1_DIR" ]]; do
    V1_DIR=$(ls -td "$TARGET_DIR"/*k1_vision_v1/ 2>/dev/null | head -1 | sed 's|/$||')
    [[ -z "$V1_DIR" ]] && sleep 60
done
V1_RUN=$(basename "$V1_DIR")
log "watcher: tracking v1 run $V1_RUN"

while true; do
    LATEST=$(ls -1 "$V1_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [[ -n "$LATEST" ]]; then
        ITER=$(basename "$LATEST" .pt | sed 's/model_//')
        # rsl_rl is 0-indexed: a 2000-iter run saves model_1999.pt as final
        if [[ "$ITER" -ge 1999 ]]; then
            log "watcher: v1 reached iter $ITER. Holding for clean exit..."
            break
        fi
    fi
    sleep 60
done

# Give Isaac Sim ~60s to flush + cleanly exit so GPU is fully free
sleep 60

# Kill v1 tmux if still alive (training script may have already exited)
if "$TMUX" has-session -t k1train 2>/dev/null; then
    log "watcher: v1 tmux still alive — sending SIGTERM"
    pkill -f "scripts/train.py --task=k1_vision --run_name=k1_vision_v1" 2>/dev/null || true
    sleep 15
    "$TMUX" kill-session -t k1train 2>/dev/null || true
fi
# Confirm GPU is free
log "watcher: GPU state before v2 launch:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/    /'

# ----------------------------------------------------------------------------
# STAGE 2 — launch v2 (10000 iters, --resume from v1)
# ----------------------------------------------------------------------------

# Pass the FULL timestamped run name; legged-loco's get_checkpoint_path uses
# re.match (anchored at start) so a substring like "k1_vision_v1" alone fails.
log "watcher: launching v2 (10000 iters, --resume from $V1_RUN)"
"$TMUX" new-session -d -s k1train2 \
    "cd $LEGGED_LOCO && source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlnce-isaac && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/train.py --task=k1_vision --run_name=k1_vision_v2_10k --max_iterations=10000 --save_interval=500 --resume True --load_run=$V1_RUN --headless 2>&1 | tee /tmp/k1_train_10k.log"

# Verify v2 launched (Isaac Sim takes 30-60s; check after 90s)
sleep 90
if ! "$TMUX" has-session -t k1train2 2>/dev/null; then
    log "watcher: ERROR — v2 tmux session ended. Using v1 instead."
    BEST_RUN="$V1_RUN"
else
    log "watcher: v2 launched OK. Polling for model_10000.pt..."
    # ------------------------------------------------------------------------
    # STAGE 3 — wait for v2 (10000 iters)
    # ------------------------------------------------------------------------
    V2_DIR=""
    while [[ -z "$V2_DIR" ]]; do
        V2_DIR=$(ls -td "$TARGET_DIR"/*k1_vision_v2_10k/ 2>/dev/null | head -1 | sed 's|/$||')
        [[ -z "$V2_DIR" ]] && sleep 60
    done
    V2_RUN=$(basename "$V2_DIR")
    log "watcher: tracking v2 run $V2_RUN"

    # Wall-clock deadline: 6h of v2 training, then force stop so benchmark
    # has time to run before the user wakes (~7:55 AM if 8h from 23:55).
    V2_START=$(date +%s)
    V2_BUDGET=21600   # 6 hours

    LAST_PROGRESS_MIN=0
    LAST_ITER=0
    while true; do
        sleep 60
        # tmux ended naturally?
        if ! "$TMUX" has-session -t k1train2 2>/dev/null; then
            log "watcher: k1train2 tmux ended naturally."
            break
        fi
        LATEST=$(ls -1 "$V2_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
        if [[ -n "$LATEST" ]]; then
            ITER=$(basename "$LATEST" .pt | sed 's/model_//')
            if [[ "$ITER" -ne "$LAST_ITER" ]]; then
                log "watcher: v2 latest iter=$ITER"
                LAST_ITER=$ITER
                LAST_PROGRESS_MIN=0
            else
                LAST_PROGRESS_MIN=$((LAST_PROGRESS_MIN + 1))
            fi
            # Crash fallback: iter >= 2500 and no new checkpoint in 10 min
            if [[ "$ITER" -ge 2500 && "$LAST_PROGRESS_MIN" -ge 10 ]]; then
                log "watcher: v2 stalled at iter=$ITER for 10+ min. Stopping."
                break
            fi
        fi
        # Wall-clock budget check
        ELAPSED=$(( $(date +%s) - V2_START ))
        if [[ "$ELAPSED" -ge "$V2_BUDGET" ]]; then
            log "watcher: v2 hit 6h wall-clock budget at iter=$LAST_ITER. Stopping."
            break
        fi
    done

    # Check that v2's last checkpoint is actually better than v1's by file
    # presence alone. Default policy: prefer v2 if it has any checkpoint past
    # model_2000.pt; otherwise use v1.
    V2_LAST_ITER=$(ls -1 "$V2_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 \
                   | xargs -n1 basename 2>/dev/null \
                   | sed 's/model_//;s/.pt//')
    V2_LAST_ITER=${V2_LAST_ITER:-0}
    if [[ "$V2_LAST_ITER" -gt 2000 ]]; then
        BEST_RUN="$V2_RUN"
        log "watcher: choosing v2 ($V2_RUN, iter=$V2_LAST_ITER) for benchmark."
    else
        BEST_RUN="$V1_RUN"
        log "watcher: v2 didn't progress past v1; falling back to v1 ($V1_RUN)."
    fi
fi

# ----------------------------------------------------------------------------
# STAGE 5 — release GPU, launch benchmark
# ----------------------------------------------------------------------------

# Kill training sessions
"$TMUX" kill-session -t k1train  2>/dev/null || true
"$TMUX" kill-session -t k1train2 2>/dev/null || true
pkill -f "scripts/train.py" 2>/dev/null || true
sleep 30
log "watcher: GPU state before benchmark launch:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/    /'

log "watcher: launching benchmark pipeline for $BEST_RUN..."
bash ~/Projects/k1_research/run_k1_benchmark.sh "$BEST_RUN" 0 1077 240 \
    2>&1 | tee -a /tmp/k1_pipeline.log
log "watcher: benchmark pipeline returned."
