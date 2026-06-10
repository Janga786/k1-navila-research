#!/bin/bash
# Launch K1 VLN-CE-Isaac compatible training in tmux.
#
# Trains `Booster-K1-Velocity-Vision-VLNCE-v0` (208-dim obs w/ height_scan,
# action_scale=0.5, bad_orientation@1.3 rad, rough terrain curriculum, all
# the booster_train stable PPO settings).
#
# Log:        /tmp/k1_booster_train.log
# Logs dir:   ~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/<timestamp>_<tag>/
# Tmux session: k1_vlnce_train
#
# After launch, the watcher script (watch_k1_vlnce_train.sh) checks reward
# every 10 min and kills training if it stalls for 15k iters or starts dropping.

set -e

TAG="${1:-k1_vlnce_v1}"
NUM_ENVS="${2:-4096}"
MAX_ITERS="${3:-30000}"

export PATH="$HOME/miniconda3/bin:$PATH"
# tmux from miniconda needs its bundled libtinfo (6.5); system has 6.4.
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

BOOSTER_DIR=~/Projects/k1_research/booster/booster_train
LOG=/tmp/k1_booster_train.log

# Kill any stale session
"$TMUX_BIN" kill-session -t k1_vlnce_train 2>/dev/null || true

echo "[$(date +%H:%M:%S)] launching training in tmux 'k1_vlnce_train'..."
echo "  tag=$TAG  num_envs=$NUM_ENVS  max_iters=$MAX_ITERS"
echo "  log=$LOG"

"$TMUX_BIN" new-session -d -s k1_vlnce_train \
  "cd $BOOSTER_DIR && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate isaacLab_311 && \
   python scripts/rsl_rl/train.py \
     --task Booster-K1-Velocity-Vision-VLNCE-v0 \
     --headless \
     --num_envs $NUM_ENVS \
     --max_iterations $MAX_ITERS \
     --logger tensorboard \
     --run_name $TAG \
     2>&1 | tee $LOG"

echo "[$(date +%H:%M:%S)] tmux session 'k1_vlnce_train' started."
echo
echo "Monitor with:"
echo "  tmux capture-pane -t k1_vlnce_train -p | tail -40"
echo "  tail -F $LOG"
echo
echo "To kill: tmux kill-session -t k1_vlnce_train"
