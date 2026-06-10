#!/bin/bash
# Fix-v1 benchmark pipeline: re-runs the K1 vision benchmark with:
#   * Fix A: base_contact termination + relaxed bad_orientation (1.3 rad)
#   * Fix B: image_observations pre-warmed with 8 real frames
#
# Uses load_run=<orig>_fix_v1 so eval_results land in a DISTINCT directory
# from the broken SR=0% run for easy comparison.

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib"

NAVILA_BENCH=~/Projects/k1_research/NaVILA-Bench
LOAD_RUN="2026-05-16_01-16-18_k1_vision_v2_10k_fix_v1"  # symlink → original run
VLM_LOG=/tmp/vlm_server_fix_v1.log
BENCH_LOG=/tmp/k1_bench_fix_v1.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Fix-v1 pipeline starting. load_run=$LOAD_RUN"

# Start VLM bridge (detached child)
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

# Wait for VLM bridge ready
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
fi

log "GPU state before benchmark launch:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | sed 's/^/    /'

log "launching benchmark (1077 episodes, 240s/ep cap)..."
cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes

python scripts/run_benchmark.py \
    --task=k1_matterport_vision \
    --low_level_policy_dir="$LOAD_RUN" \
    --start-idx=0 --end-idx=1077 \
    --per-episode-timeout=240 \
    > "$BENCH_LOG" 2>&1

log "benchmark finished."
log "aggregated results:"
python ~/Projects/k1_research/aggregate_k1_vision_results.py 2>&1 | sed 's/^/    /'
log "Fix-v1 pipeline complete."
