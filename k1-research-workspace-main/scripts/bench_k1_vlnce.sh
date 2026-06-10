#!/bin/bash
# Run the VLN-CE-Isaac benchmark against the booster_train K1 vision policy.
#
# The new policy is trained in isaacLab_311 (rsl_rl 3.1.2). It saves
# .pt files under booster_train/logs/rsl_rl/k1_velocity_vlnce/<timestamp>_<tag>/.
# This script symlinks that run directory into NaVILA-Bench under a synthetic
# "k1_vision_rough" load_run name so eval_results land in a distinct dir,
# launches the VLM bridge, smoke-tests one episode, then optionally fires the
# full benchmark in tmux.
#
# Usage:
#   bench_k1_vlnce.sh smoke   <run_subdir> [checkpoint_iter]
#   bench_k1_vlnce.sh full    <run_subdir> [start_idx] [end_idx]
#
# run_subdir is e.g. "2026-05-18_10-07-57_k1_vlnce_v1" (the booster_train log dir).
# checkpoint_iter defaults to "latest" (highest iter present).
#
# Smoke test: 1 episode (idx 0), 240s timeout, in foreground.
# Full run:   tmux session 'bench_vlnce' on episodes [start, end), 240s/ep.

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

MODE="${1:-smoke}"
RUN_SUBDIR="${2:-}"
ARG3="${3:-}"
ARG4="${4:-}"

if [[ -z "$RUN_SUBDIR" ]]; then
    echo "Usage:"
    echo "  $0 smoke <run_subdir> [checkpoint_iter]"
    echo "  $0 full  <run_subdir> [start_idx] [end_idx]"
    echo
    echo "Available booster_train runs:"
    ls ~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/ 2>/dev/null
    exit 1
fi

BOOSTER_RUN_DIR="$HOME/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/$RUN_SUBDIR"
NAVILA_BENCH="$HOME/Projects/k1_research/NaVILA-Bench"
LOAD_RUN_NAME="booster_vlnce_${RUN_SUBDIR}"     # appears in eval_results dir
LINK_DST="$NAVILA_BENCH/logs/rsl_rl/k1_vision_rough/$LOAD_RUN_NAME"

if [[ ! -d "$BOOSTER_RUN_DIR" ]]; then
    echo "ERROR: booster_train run dir not found: $BOOSTER_RUN_DIR"
    exit 1
fi

# Build a wrapper directory the benchmark can load.
#
# We can't just symlink the booster_train run dir, because navila_eval.py
# calls update_class_from_dict(agent_cfg, log_agent_cfg_dict) on
# params/agent.yaml, and booster_train (rsl_rl 3.1.2) writes 3.1.2-specific
# fields that rsl_rl 2.0.2's RslRlOnPolicyRunnerCfg rejects:
#   obs_groups, noise_std_type, state_dependent_std,
#   actor_obs_normalization, critic_obs_normalization,
#   normalize_advantage_per_mini_batch, rnd_cfg, symmetry_cfg, clip_actions,
#   class_name (top-level)
#
# So: real directory + symlinks to model_*.pt + sanitized agent.yaml.
mkdir -p "$LINK_DST/params"
# (Re)create per-checkpoint symlinks — picks up new model_*.pt files saved
# during training.
for f in "$BOOSTER_RUN_DIR"/model_*.pt; do
    [[ -f "$f" ]] || continue
    ln -sf "$f" "$LINK_DST/$(basename "$f")"
done
# Write a sanitized agent.yaml (matches legged-loco v2's schema that the
# benchmark accepts).
cat > "$LINK_DST/params/agent.yaml" <<EOF
seed: 42
device: cuda:0
num_steps_per_env: 24
max_iterations: 30000
empirical_normalization: false
policy:
  class_name: ActorCritic
  init_noise_std: 1.0
  actor_hidden_dims:
  - 512
  - 256
  - 128
  critic_hidden_dims:
  - 512
  - 256
  - 128
  activation: elu
  history_length: 0
algorithm:
  class_name: PPO
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  entropy_coef: 0.01
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 0.001
  schedule: adaptive
  gamma: 0.99
  lam: 0.95
  desired_kl: 0.01
  max_grad_norm: 1.0
save_interval: 2000
experiment_name: k1_vision_rough
run_name: $RUN_SUBDIR
logger: tensorboard
resume: false
load_run: $LOAD_RUN_NAME
load_checkpoint: model_.*.pt
EOF
echo "[bench] wrapper dir ready: $LINK_DST"

# Pick the checkpoint
if [[ "$MODE" == "smoke" ]]; then
    CKPT_ITER="${ARG3:-latest}"
    if [[ "$CKPT_ITER" == "latest" ]]; then
        CKPT_FILE=$(ls -1 "$BOOSTER_RUN_DIR"/model_*.pt 2>/dev/null \
            | grep -oP 'model_\d+\.pt' \
            | sort -t_ -k2 -n \
            | tail -1)
    else
        CKPT_FILE="model_${CKPT_ITER}.pt"
    fi
    if [[ -z "$CKPT_FILE" || ! -f "$BOOSTER_RUN_DIR/$CKPT_FILE" ]]; then
        echo "ERROR: no checkpoint matching iter=$CKPT_ITER in $BOOSTER_RUN_DIR"
        exit 1
    fi
    echo "[bench] smoke testing with $CKPT_FILE"
fi

# Start VLM bridge if not already up
VLM_LOG=/tmp/vlm_server_vlnce.log
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
    SMOKE_LOG=/tmp/k1_vlnce_smoke.log
    echo "[bench] smoke episode 0 -> $SMOKE_LOG"
    timeout 300 python scripts/navila_eval.py \
        --task=k1_matterport_vision --num_envs=1 \
        --load_run="$LOAD_RUN_NAME" \
        --headless --enable_cameras --episode_idx=0 \
        2>&1 | tee "$SMOKE_LOG" | tail -30
    EXIT=${PIPESTATUS[0]}
    echo
    echo "[bench] smoke exit=$EXIT"
    if [[ -f "$NAVILA_BENCH/eval_results/k1_matterport_vision_loco_${LOAD_RUN_NAME}/measurements/0.json" ]]; then
        echo "[bench] measurement saved:"
        cat "$NAVILA_BENCH/eval_results/k1_matterport_vision_loco_${LOAD_RUN_NAME}/measurements/0.json"
    fi
    exit $EXIT
fi

# Full benchmark
START_IDX="${ARG3:-0}"
END_IDX="${ARG4:-1077}"
BENCH_LOG=/tmp/k1_bench_vlnce.log

"$TMUX_BIN" kill-session -t bench_vlnce 2>/dev/null || true

"$TMUX_BIN" new-session -d -s bench_vlnce \
    "cd $NAVILA_BENCH && \
     source ~/miniconda3/etc/profile.d/conda.sh && \
     conda activate vlnce-isaac && \
     export OMNI_KIT_ACCEPT_EULA=yes && \
     python scripts/run_benchmark.py \
         --task=k1_matterport_vision \
         --low_level_policy_dir=$LOAD_RUN_NAME \
         --start-idx=$START_IDX --end-idx=$END_IDX \
         --per-episode-timeout=240 \
         2>&1 | tee $BENCH_LOG"

echo "[bench] full benchmark started in tmux 'bench_vlnce' on episodes [$START_IDX, $END_IDX)."
echo "  log:    $BENCH_LOG"
echo "  attach: $TMUX_BIN attach-session -t bench_vlnce"
echo "  kill:   $TMUX_BIN kill-session -t bench_vlnce"
