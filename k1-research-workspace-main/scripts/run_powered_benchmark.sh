#!/bin/bash
# Powered K1 VLN-CE-Isaac benchmark — full 1077-episode run with the recommended
# production navigation config. RESUMABLE (skips episodes already scored), so a
# multi-hour run survives crashes/restarts. Swap the checkpoint to benchmark the
# retrained velocity policy the moment it's exported.
#
# Usage:
#   run_powered_benchmark.sh <out_tag> [ckpt_path] [start] [end] [extra_flags]
#
# Examples:
#   # current policy, full set, production config:
#   run_powered_benchmark.sh prod_v3_model16000
#   # the new velocity policy when ready:
#   run_powered_benchmark.sh prod_newpolicy /path/to/new/model_XXXX.pt
#   # ceiling diagnostic over the full set (ground-truth stop; NOT a fair SR):
#   run_powered_benchmark.sh ceiling_v3 "" 0 1077 "--proximity_stop 3.0"
#
# Production config (baked in): --closed_loop --clean_render --bright --max_episode_s 120
# (recommended defaults from the A/F arm sweep; chunk/stop_assist intentionally OFF).
set -u
export PATH="$HOME/miniconda3/bin:$PATH"

OUT_TAG="${1:?need an out_tag}"
DEFAULT_CKPT="$HOME/Projects/k1_research/checkpoints/model_14498.pt"  # production policy (was v3 model_16000)
CKPT="${2:-$DEFAULT_CKPT}"; [ -z "$CKPT" ] && CKPT="$DEFAULT_CKPT"
START="${3:-0}"
END="${4:-1077}"
EXTRA="${5:-}"
# 2026-06-11: RAW frames (no --clean_render/--bright — the 6/8 A/B found processing
# suppressed stops; the lab 3090 renders clean). Transform override: TRANSFORM=crop.
PROD_FLAGS="--closed_loop --max_episode_s 120 --vlm_transform ${TRANSFORM:-stretch}"
NB="$HOME/Projects/k1_research/NaVILA-Bench"
RESDIR="$NB/eval_results/k1_matterport_vision_loco_${OUT_TAG}"
MEAS="$RESDIR/measurements"
VLM_LOG=/tmp/vlm_server_vlnce_v3.log
LOG=/tmp/powered_${OUT_TAG}.log

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
mkdir -p "$MEAS"
echo "[powered] ckpt=$CKPT"
echo "[powered] out_tag=$OUT_TAG  episodes=[$START,$END)  flags: $PROD_FLAGS $EXTRA"
echo "[powered] results -> $RESDIR   log -> $LOG"

# --- ensure VLM bridge (navila env) is up ---
if ! pgrep -f vlm_server_bridge.py > /dev/null; then
    echo "[powered] launching VLM bridge..."
    setsid bash -c "
        cd $NB; source ~/miniconda3/etc/profile.d/conda.sh; conda activate navila
        python scripts/vlm_server_bridge.py \
            --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
            --port 54321 ${VLM_BRIDGE_EXTRA:-} > $VLM_LOG 2>&1
    " < /dev/null > /dev/null 2>&1 &
    disown
    for i in $(seq 1 60); do
        grep -q "listening on" "$VLM_LOG" 2>/dev/null && { echo "[powered] VLM ready."; break; }
        sleep 10
    done
    grep -q "listening on" "$VLM_LOG" 2>/dev/null || { echo "[powered] ERROR: VLM not ready"; tail -20 "$VLM_LOG"; exit 2; }
else
    echo "[powered] VLM bridge already running."
fi

# --- map episode_idx -> measurement file index (episode_id-1); IDs are
#     non-contiguous, so the output file is NOT named by idx. Build it once. ---
EP_JSON="$NB/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz"
mapfile -t FILEIDX < <("$HOME/miniconda3/bin/python" - "$EP_JSON" "$START" "$END" <<'PY'
import gzip, json, sys
d = json.load(gzip.open(sys.argv[1], 'rt'))['episodes']
for i in range(int(sys.argv[2]), int(sys.argv[3])):
    print(int(d[i]['episode_id']) - 1)
PY
)
[ "${#FILEIDX[@]}" -eq "$((END-START))" ] || { echo "[powered] ERROR: idx->file map failed"; exit 3; }

# --- run (resumable) ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd "$NB"

t0=$(SECONDS)
done_n=0; ran_n=0; skip_n=0
total=$((END - START))
for ((i=START; i<END; i++)); do
    fidx="${FILEIDX[$((i-START))]}"
    if [ -f "$MEAS/${fidx}.json" ]; then skip_n=$((skip_n+1)); continue; fi   # resume
    echo "================ episode idx $i (file ${fidx}.json, $((done_n+1))/$total) ================" | tee -a "$LOG"
    timeout -k 30 ${EP_TIMEOUT:-900} python scripts/navila_eval_v3.py \
        --task=k1_matterport_vision --num_envs=1 \
        --checkpoint="$CKPT" --episode_idx=$i --gait_phase_init=0.0 \
        --out_tag=$OUT_TAG $PROD_FLAGS $EXTRA \
        --headless --enable_cameras >> "$LOG" 2>&1 || echo "[ep $i timeout/error]" | tee -a "$LOG"
    ran_n=$((ran_n+1)); done_n=$((done_n+1))
    if (( done_n % 10 == 0 )); then
        el=$((SECONDS - t0)); rate=$(( el / (ran_n>0?ran_n:1) ))
        left=$(( (total - skip_n - ran_n) * rate ))
        echo "[powered] progress: ran=$ran_n skipped=$skip_n  ~${rate}s/ep  ETA ~$((left/60)) min" | tee -a "$LOG"
    fi
done
echo "[powered] finished: ran=$ran_n skipped=$skip_n in $(((SECONDS-t0)/60)) min" | tee -a "$LOG"

# --- aggregate over the full intended set (missing = failures) ---
echo "[powered] === AGGREGATE ($OUT_TAG) ==="
python "$HOME/Projects/k1_research/aggregate_k1_vision_results.py" \
    --measurements-dir "$MEAS" --total-episodes "$total" --paper-table
