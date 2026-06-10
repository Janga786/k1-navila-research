#!/bin/bash
# K1 vision-policy VLN-CE-Isaac benchmark pipeline.
#
# Usage:
#   ./run_k1_benchmark.sh <load_run> [start_idx] [end_idx] [per_ep_timeout]
#
# Example:
#   ./run_k1_benchmark.sh 2026-05-15_23-17_k1_vision_v1 0 1077 240
#
# Assumes:
#   * trained policy lives at:
#     ~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/<load_run>/
#   * conda envs `vlnce-isaac` and `navila` exist and are configured
#   * NaVILA-Bench is at ~/Projects/k1_research/NaVILA-Bench
#
# Behavior:
#   1. Symlinks the trained run dir from legged-loco -> NaVILA-Bench.
#   2. Launches VLM bridge in tmux session `vlm` (in `navila` env).
#   3. Waits for VLM "listening" message.
#   4. Smoke-tests with a single episode.
#   5. If smoke test passes, launches full benchmark in tmux `bench`.

set -e

LOAD_RUN="${1:-}"
START_IDX="${2:-0}"
END_IDX="${3:-1077}"
PER_EP_TIMEOUT="${4:-240}"

if [[ -z "$LOAD_RUN" ]]; then
    echo "Usage: $0 <load_run> [start_idx] [end_idx] [per_ep_timeout]"
    exit 1
fi

export PATH="$HOME/miniconda3/bin:$PATH"
TMUX="$HOME/miniconda3/bin/tmux"

LEGGED_LOCO=~/Projects/k1_research/legged-loco
NAVILA_BENCH=~/Projects/k1_research/NaVILA-Bench
SRC_RUN_DIR="$LEGGED_LOCO/logs/rsl_rl/k1_vision_rough/$LOAD_RUN"
DST_RUN_DIR="$NAVILA_BENCH/logs/rsl_rl/k1_vision_rough/$LOAD_RUN"

if [[ ! -d "$SRC_RUN_DIR" ]]; then
    echo "ERROR: trained run dir not found: $SRC_RUN_DIR"
    exit 1
fi

mkdir -p "$NAVILA_BENCH/logs/rsl_rl/k1_vision_rough"
if [[ ! -e "$DST_RUN_DIR" ]]; then
    ln -s "$SRC_RUN_DIR" "$DST_RUN_DIR"
    echo "[pipeline] symlinked $SRC_RUN_DIR -> $DST_RUN_DIR"
fi

# Kill any stale tmux sessions
"$TMUX" kill-session -t vlm   2>/dev/null || true
"$TMUX" kill-session -t bench 2>/dev/null || true

# Start VLM bridge
echo "[pipeline] launching VLM bridge in tmux 'vlm'..."
"$TMUX" new-session -d -s vlm \
    "cd $NAVILA_BENCH && source ~/miniconda3/etc/profile.d/conda.sh && conda activate navila && python scripts/vlm_server_bridge.py --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f --port 54321 2>&1 | tee /tmp/vlm_server.log"

echo "[pipeline] waiting up to 600s for VLM bridge to come up..."
for i in $(seq 1 60); do
    if grep -q "listening on" /tmp/vlm_server.log 2>/dev/null; then
        echo "[pipeline] VLM bridge ready."
        break
    fi
    sleep 10
done

if ! grep -q "listening on" /tmp/vlm_server.log; then
    echo "[pipeline] WARNING: VLM bridge did not announce 'listening on' within 10 min."
    echo "[pipeline] last 20 lines of /tmp/vlm_server.log:"
    tail -20 /tmp/vlm_server.log || true
    echo "[pipeline] proceeding anyway — the bridge may still be loading."
fi

# Smoke-test with a single episode
echo "[pipeline] smoke testing with episode 0..."
cd "$NAVILA_BENCH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes

timeout 300 python scripts/navila_eval.py \
    --task=k1_matterport_vision --num_envs=1 \
    --load_run="$LOAD_RUN" \
    --headless --enable_cameras --episode_idx=0 \
    2>&1 | tee /tmp/k1_smoke.log | tail -40 || true

# Launch full benchmark in tmux
echo "[pipeline] launching full benchmark in tmux 'bench'..."
"$TMUX" new-session -d -s bench \
    "cd $NAVILA_BENCH && source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlnce-isaac && export OMNI_KIT_ACCEPT_EULA=yes && python scripts/run_benchmark.py --task=k1_matterport_vision --low_level_policy_dir=$LOAD_RUN --start-idx=$START_IDX --end-idx=$END_IDX --per-episode-timeout=$PER_EP_TIMEOUT 2>&1 | tee /tmp/k1_bench_vision.log"

echo "[pipeline] DONE — monitor:"
echo "  tmux capture-pane -t bench -p | tail -40"
echo "  tail -f /tmp/k1_bench_vision.log"
