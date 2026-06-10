#!/bin/bash
# Mega pipeline: v2 training (resume from v1) -> benchmark.
# No tmux required — uses setsid + nohup detach via parent script.
# Detaches stdout/stderr to log files.

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib"

LEGGED_LOCO=~/Projects/k1_research/legged-loco
NAVILA_BENCH=~/Projects/k1_research/NaVILA-Bench
PIPELINE_LOG=/tmp/mega_pipeline.log
TRAIN_LOG=/tmp/k1_train_10k.log
BENCH_LOG=/tmp/k1_bench_vision.log
VLM_LOG=/tmp/vlm_server.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

V1_RUN="2026-05-15_23-17-29_k1_vision_v1"
V2_NAME="k1_vision_v2_10k"
# Use 8000 to make total iters = 2000+8000 = 10000 (user's intent).
V2_MAX_ITERS=8000
V2_SAVE_INTERVAL=500
# Wall-clock cap on v2: 6h gives benchmark ~2h before 8h overnight expires.
V2_BUDGET_S=21600

# ----------------------------------------------------------------------------
# Stage 1: ensure v1 checkpoint exists
# ----------------------------------------------------------------------------
V1_DIR="$LEGGED_LOCO/logs/rsl_rl/k1_vision_rough/$V1_RUN"
V1_FINAL=$(ls -1 "$V1_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [[ -z "$V1_FINAL" ]]; then
    log "ERROR: v1 checkpoint not found at $V1_DIR"
    exit 1
fi
V1_ITER=$(basename "$V1_FINAL" .pt | sed 's/model_//')
log "v1 final checkpoint: model_${V1_ITER}.pt"

# ----------------------------------------------------------------------------
# Stage 2: run v2 (8000 more iters from v1)
# ----------------------------------------------------------------------------
log "launching v2 training: $V2_NAME (8000 iters, total=10000)"
cd "$LEGGED_LOCO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes

# Run v2 with a wall-clock kill via timeout(1)
timeout "$V2_BUDGET_S" python scripts/train.py \
    --task=k1_vision \
    --run_name="$V2_NAME" \
    --max_iterations="$V2_MAX_ITERS" \
    --save_interval="$V2_SAVE_INTERVAL" \
    --resume True \
    --load_run="$V1_RUN" \
    --headless \
    > "$TRAIN_LOG" 2>&1
V2_EXIT=$?
log "v2 training exited with code $V2_EXIT"

# ----------------------------------------------------------------------------
# Stage 3: pick best run for benchmark
# ----------------------------------------------------------------------------
V2_DIR=$(ls -td "$LEGGED_LOCO"/logs/rsl_rl/k1_vision_rough/*${V2_NAME}/ 2>/dev/null \
         | head -1 | sed 's|/$||')
if [[ -n "$V2_DIR" ]]; then
    V2_FINAL=$(ls -1 "$V2_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [[ -n "$V2_FINAL" ]]; then
        V2_ITER=$(basename "$V2_FINAL" .pt | sed 's/model_//')
    else
        V2_ITER=0
    fi
else
    V2_ITER=0
fi
log "v2 final checkpoint iter=$V2_ITER"

# Choose v2 if it has any checkpoint past v1 (i.e., iter > V1_ITER)
if [[ "$V2_ITER" -gt "$V1_ITER" ]]; then
    BEST_RUN=$(basename "$V2_DIR")
    log "choosing v2 ($BEST_RUN, iter=$V2_ITER) for benchmark"
else
    BEST_RUN="$V1_RUN"
    log "v2 didn't progress past v1; falling back to v1 ($V1_RUN)"
fi

# ----------------------------------------------------------------------------
# Stage 4: launch benchmark
# ----------------------------------------------------------------------------
# Kill any lingering python (just in case)
pkill -f "scripts/train.py" 2>/dev/null || true
sleep 30

log "GPU state before benchmark launch:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/    /'

# Symlink trained run into NaVILA-Bench's logs
SRC="$LEGGED_LOCO/logs/rsl_rl/k1_vision_rough/$BEST_RUN"
DST="$NAVILA_BENCH/logs/rsl_rl/k1_vision_rough/$BEST_RUN"
mkdir -p "$NAVILA_BENCH/logs/rsl_rl/k1_vision_rough"
if [[ ! -e "$DST" ]]; then
    ln -s "$SRC" "$DST"
    log "symlinked $SRC -> $DST"
fi

# Start VLM bridge (detached)
log "launching VLM bridge..."
setsid bash -c "
    cd $NAVILA_BENCH
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate navila
    python scripts/vlm_server_bridge.py \
        --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
        --port 54321 > $VLM_LOG 2>&1
" < /dev/null > /dev/null 2>&1 &
disown

# Wait for VLM bridge ready (up to 10 min)
log "waiting up to 600s for VLM bridge..."
for i in $(seq 1 60); do
    if grep -q "listening on" "$VLM_LOG" 2>/dev/null; then
        log "VLM bridge ready."
        break
    fi
    sleep 10
done
if ! grep -q "listening on" "$VLM_LOG" 2>/dev/null; then
    log "WARNING: VLM bridge didn't announce listening; proceeding anyway"
    tail -10 "$VLM_LOG" 2>/dev/null | sed 's/^/    /'
fi

# Launch benchmark
log "launching benchmark (1077 episodes, 240s/ep cap)..."
cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes

python scripts/run_benchmark.py \
    --task=k1_matterport_vision \
    --low_level_policy_dir="$BEST_RUN" \
    --start-idx=0 --end-idx=1077 \
    --per-episode-timeout=240 \
    > "$BENCH_LOG" 2>&1

log "benchmark finished."
log "results:"
python ~/Projects/k1_research/aggregate_k1_vision_results.py --paper-table 2>&1 | sed 's/^/    /'
log "mega pipeline complete."
