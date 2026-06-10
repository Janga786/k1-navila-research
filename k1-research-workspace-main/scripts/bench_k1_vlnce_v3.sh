#!/bin/bash
# K1 VLN-CE v3 benchmark wrapper.
#
# Drives the v3 policy through NaVILA-Bench via the navila_eval_v3.py
# script + VLNEnvWrapperV3 adapter. The wrapper builds the v3 47-dim
# per-step obs + 5-frame term-major history from raw scene state, so the
# bench env can stay on its existing k1_matterport_vision task.
#
# Usage:
#   bench_k1_vlnce_v3.sh smoke [ckpt_path] [episode_idx] [gait_phase_init]
#   bench_k1_vlnce_v3.sh full  [ckpt_path] [start_idx] [end_idx]
#
# ckpt_path default: model_16000.pt (peak reward 111.57).

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

MODE="${1:-smoke}"
DEFAULT_CKPT="$HOME/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/model_16000.pt"
CKPT="${2:-$DEFAULT_CKPT}"
ARG3="${3:-}"
ARG4="${4:-}"

NAVILA_BENCH="$HOME/Projects/k1_research/NaVILA-Bench"
OUT_TAG="v3_model_16000"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi
echo "[bench] checkpoint: $CKPT"

# Start VLM bridge if not running
VLM_LOG=/tmp/vlm_server_vlnce_v3.log
if ! pgrep -f vlm_server_bridge.py > /dev/null; then
    echo "[bench] launching VLM bridge..."
    setsid bash -c "
        cd $NAVILA_BENCH
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate navila
        python scripts/vlm_server_bridge.py \
            --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \
            --port 54321 > $VLM_LOG 2>&1
    " < /dev/null > /dev/null 2>&1 &
    disown
    echo "[bench] waiting up to 600s for VLM bridge..."
    for i in $(seq 1 60); do
        if grep -q "listening on" "$VLM_LOG" 2>/dev/null; then
            echo "[bench] VLM bridge ready."
            break
        fi
        sleep 10
    done
    if ! grep -q "listening on" "$VLM_LOG" 2>/dev/null; then
        echo "[bench] WARNING: VLM bridge didn't announce 'listening on' within 10 min."
    fi
else
    echo "[bench] VLM bridge already running (pid $(pgrep -f vlm_server_bridge.py | head -1))"
fi

cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes

if [[ "$MODE" == "smoke" ]]; then
    EPISODE_IDX="${ARG3:-0}"
    GAIT_INIT="${ARG4:-0.0}"
    SMOKE_LOG=/tmp/k1_v3_smoke.log
    echo "[bench] smoke ep=$EPISODE_IDX gait_phase_init=$GAIT_INIT -> $SMOKE_LOG"
    timeout 300 python scripts/navila_eval_v3.py \
        --task=k1_matterport_vision --num_envs=1 \
        --checkpoint="$CKPT" \
        --episode_idx=$EPISODE_IDX \
        --gait_phase_init=$GAIT_INIT \
        --out_tag=$OUT_TAG \
        --headless --enable_cameras \
        2>&1 | tee "$SMOKE_LOG" | tail -30
    EXIT=${PIPESTATUS[0]}
    echo "[bench] smoke exit=$EXIT"
    MEAS="$NAVILA_BENCH/eval_results/k1_matterport_vision_loco_${OUT_TAG}/measurements/${EPISODE_IDX}.json"
    if [[ -f "$MEAS" ]]; then
        echo "[bench] measurement:"; cat "$MEAS"
    fi
    exit $EXIT
fi

# Full benchmark
START_IDX="${ARG3:-0}"
END_IDX="${ARG4:-1077}"
GAIT_INIT="${5:-0.0}"
BENCH_LOG=/tmp/k1_v3_bench.log

"$TMUX_BIN" kill-session -t bench_v3 2>/dev/null || true

# Use a simple driver: loop episodes, call navila_eval_v3.py one-at-a-time.
# (run_benchmark.py is for the v2 path and assumes navila_eval.py.)
DRIVER=/tmp/k1_v3_bench_driver.sh
cat > "$DRIVER" <<EOF
#!/bin/bash
set -e
export OMNI_KIT_ACCEPT_EULA=yes
cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
for ((i=$START_IDX; i<$END_IDX; i++)); do
    echo "=== episode \$i ==="
    timeout 300 python scripts/navila_eval_v3.py \\
        --task=k1_matterport_vision --num_envs=1 \\
        --checkpoint="$CKPT" \\
        --episode_idx=\$i \\
        --gait_phase_init=$GAIT_INIT \\
        --out_tag=$OUT_TAG \\
        --headless --enable_cameras || echo "[ep \$i timeout/error]"
done
EOF
chmod +x "$DRIVER"

"$TMUX_BIN" new-session -d -s bench_v3 "bash $DRIVER 2>&1 | tee $BENCH_LOG"
echo "[bench] full benchmark started in tmux 'bench_v3' on episodes [$START_IDX, $END_IDX)."
echo "  log:    $BENCH_LOG"
echo "  attach: LD_LIBRARY_PATH=\$HOME/miniconda3/lib \$HOME/miniconda3/bin/tmux attach-session -t bench_v3"
echo "  kill:   LD_LIBRARY_PATH=\$HOME/miniconda3/lib \$HOME/miniconda3/bin/tmux kill-session -t bench_v3"
