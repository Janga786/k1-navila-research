#!/bin/bash
# Camera-height experiment: K1 rgb_camera z=0.85 → 0.50 (match Go2/H1 mount).
# Runs 50 episodes with a fresh out_tag, aggregates, restores cfg.
#
# Pre-conditions verified inside the script:
#   * main bench tmux 'bench_v3' is GONE
#   * no `navila_eval_v3.py` python process holds the GPU
#
# Safety: original cfg is backed up to *.bak_cam085 and ALWAYS restored
# (trap on EXIT) so a Ctrl-C mid-run can't leave the repo in a half-edited
# state.
#
# Outputs land in:
#   NaVILA-Bench/eval_results/k1_matterport_vision_loco_v3_cam050_n50/measurements/
#
# Compare with:
#   python ~/Projects/k1_research/aggregate_k1_vision_results.py \
#          --measurements-dir <above>/measurements --paper-table

set -u

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

CFG="$HOME/Projects/k1_research/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/k1/k1_matterport_vision_cfg.py"
BAK="$CFG.bak_cam085"
CKPT="$HOME/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/model_16000.pt"
NAVILA_BENCH="$HOME/Projects/k1_research/NaVILA-Bench"
OUT_TAG="v3_cam050_n50"
LOG=/tmp/k1_v3_cam050.log
DRIVER=/tmp/k1_v3_cam050_driver.sh
N_EPS="${1:-50}"

log() { echo "[$(date +%F\ %H:%M:%S)] $*" | tee -a "$LOG"; }

# Always restore cfg on exit
restore_cfg() {
  if [[ -f "$BAK" ]]; then
    mv -f "$BAK" "$CFG"
    log "restored original cfg from $BAK"
  fi
}
trap restore_cfg EXIT

# ---- 0. pre-conditions
log "Pre-flight: main bench must be dead"
if "$TMUX_BIN" has-session -t bench_v3 2>/dev/null; then
  log "FATAL: tmux 'bench_v3' is still alive. Aborting."; exit 1
fi
if pgrep -f "navila_eval_v3.py" >/dev/null; then
  log "FATAL: navila_eval_v3.py is still running. Aborting."; exit 1
fi
if pgrep -f "k1_v3_bench_driver.sh" >/dev/null; then
  log "FATAL: bench driver still running. Aborting."; exit 1
fi
log "Pre-flight OK"

# ---- 1. backup + edit cfg
if [[ ! -f "$CFG" ]]; then
  log "FATAL: cfg not found at $CFG"; exit 1
fi
cp "$CFG" "$BAK"
log "backed up cfg to $BAK"
# rgb_camera mount: 0.85 → 0.50 (NaVILA-input)
sed -i 's|pos=(0.10, 0.0, 0.85)|pos=(0.10, 0.0, 0.50)|' "$CFG"
# Sanity-check the edit went through
if ! grep -q "pos=(0.10, 0.0, 0.50)" "$CFG"; then
  log "FATAL: cfg edit did not apply. Aborting (cfg will be restored)."; exit 2
fi
log "edited cfg: K1 rgb_camera mount 0.85 -> 0.50"

# ---- 2. start VLM bridge if needed
VLM_LOG=/tmp/vlm_server_vlnce_v3_cam050.log
if ! pgrep -f vlm_server_bridge.py >/dev/null; then
  log "launching VLM bridge…"
  setsid bash -c "
    cd $NAVILA_BENCH
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate navila
    python scripts/vlm_server_bridge.py \
      --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
      --port 54321 > $VLM_LOG 2>&1
  " </dev/null >/dev/null 2>&1 &
  disown
  log "waiting up to 10 min for VLM bridge…"
  for i in $(seq 1 60); do
    grep -q "listening on" "$VLM_LOG" 2>/dev/null && { log "VLM bridge ready"; break; }
    sleep 10
  done
else
  log "VLM bridge already running (pid $(pgrep -f vlm_server_bridge.py | head -1))"
fi

# ---- 3. driver loop (N_EPS sequential episodes starting from dataset idx 0)
cat > "$DRIVER" <<EOF
#!/bin/bash
set -e
export OMNI_KIT_ACCEPT_EULA=yes
cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
for ((i=0; i<$N_EPS; i++)); do
    echo "=== cam050 episode \$i ==="
    timeout 300 python scripts/navila_eval_v3.py \\
        --task=k1_matterport_vision --num_envs=1 \\
        --checkpoint="$CKPT" \\
        --episode_idx=\$i \\
        --gait_phase_init=0.0 \\
        --out_tag=$OUT_TAG \\
        --headless --enable_cameras || echo "[ep \$i timeout/error]"
done
EOF
chmod +x "$DRIVER"

"$TMUX_BIN" kill-session -t bench_v3_cam050 2>/dev/null || true
"$TMUX_BIN" new-session -d -s bench_v3_cam050 "bash $DRIVER 2>&1 | tee /tmp/k1_v3_cam050_bench.log"
log "tmux 'bench_v3_cam050' started for $N_EPS episodes"

# ---- 4. wait for driver to finish
log "waiting for driver to complete…"
while "$TMUX_BIN" has-session -t bench_v3_cam050 2>/dev/null; do
  sleep 30
done
log "driver done. measurements in $NAVILA_BENCH/eval_results/k1_matterport_vision_loco_${OUT_TAG}/measurements/"

# ---- 5. aggregate
log "aggregating…"
python ~/Projects/k1_research/aggregate_k1_vision_results.py \
  --measurements-dir "$NAVILA_BENCH/eval_results/k1_matterport_vision_loco_${OUT_TAG}/measurements" \
  --paper-table 2>&1 | tee -a "$LOG"

# cfg restored by trap
log "experiment complete"
