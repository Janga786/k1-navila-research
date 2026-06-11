#!/bin/bash
# Diagnostics-enabled K1 VLN-CE v3 benchmark slice (Step 1).
# Mirrors bench_k1_vlnce_v3.sh exactly but adds --diag and a custom out_tag/range,
# so it never clobbers the production v3_model_16000 results.
#
# Usage: bench_diag.sh <start_idx> <end_idx> [out_tag] [ckpt_path] [extra_flags]
#   extra_flags e.g. "--clean_render --closed_loop"  (passed to navila_eval_v3.py)
set -e
export PATH="$HOME/miniconda3/bin:$PATH"

START_IDX="${1:-0}"
END_IDX="${2:-20}"
OUT_TAG="${3:-v3_diag}"
DEFAULT_CKPT="$HOME/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/model_16000.pt"
CKPT="${4:-$DEFAULT_CKPT}"
EXTRA_FLAGS="${5:-}"
NAVILA_BENCH="$HOME/Projects/k1_research/NaVILA-Bench"
VLM_LOG=/tmp/vlm_server_vlnce_v3.log

[[ -f "$CKPT" ]] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
echo "[diag-bench] ckpt=$CKPT  eps=[$START_IDX,$END_IDX)  out_tag=$OUT_TAG"

# --- ensure VLM bridge (navila env) is up ---
if ! pgrep -f vlm_server_bridge.py > /dev/null; then
    echo "[diag-bench] launching VLM bridge (8B model load can take a few minutes)..."
    setsid bash -c "
        cd $NAVILA_BENCH
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate navila
        python scripts/vlm_server_bridge.py \
            --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
            --port 54321 $VLM_BRIDGE_EXTRA > $VLM_LOG 2>&1
    " < /dev/null > /dev/null 2>&1 &
    disown
    echo "[diag-bench] waiting up to 600s for 'listening on'..."
    for i in $(seq 1 60); do
        grep -q "listening on" "$VLM_LOG" 2>/dev/null && { echo "[diag-bench] VLM bridge ready."; break; }
        sleep 10
    done
    grep -q "listening on" "$VLM_LOG" 2>/dev/null || { echo "[diag-bench] ERROR: bridge not ready; tail:"; tail -20 "$VLM_LOG"; exit 2; }
else
    echo "[diag-bench] VLM bridge already running (pid $(pgrep -f vlm_server_bridge.py | head -1))"
fi

# --- run the eval slice with diagnostics (vlnce-isaac env) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd "$NAVILA_BENCH"

for ((i=START_IDX; i<END_IDX; i++)); do
    echo "================ episode $i ================"
    timeout -k 30 ${EP_TIMEOUT:-900} python scripts/navila_eval_v3.py \
        --task=k1_matterport_vision --num_envs=1 \
        --checkpoint="$CKPT" \
        --episode_idx=$i \
        --gait_phase_init=0.0 \
        --out_tag=$OUT_TAG \
        --diag $EXTRA_FLAGS \
        --headless --enable_cameras || echo "[ep $i timeout/error]"
done
echo "[diag-bench] DONE eps [$START_IDX,$END_IDX) -> eval_results/k1_matterport_vision_loco_${OUT_TAG}/"