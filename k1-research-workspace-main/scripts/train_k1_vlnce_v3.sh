#!/bin/bash
# Launch K1 VLN-CE v3 training in a tmux session.
#
# v3 starts from the WORKING velocity_env_cfg.py (the one that produced
# the reward-103 walking policy). It adds height_scan as a separate vision
# obs group (160 dim) without touching the actor/critic obs that the
# working policy depends on. See vlnce_v3_env_cfg.py for the rationale.
#
# Usage: train_k1_vlnce_v3.sh [run_name] [num_envs] [max_iters]
#   defaults: run_name=k1_vlnce_v3  num_envs=4096  max_iters=5000

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
# tmux needs the conda libtinfo at runtime
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

BOOSTER_DIR=~/Projects/k1_research/booster/booster_train
LOG=/tmp/k1_booster_train_v3.log
SESSION=k1_vlnce_v3_train
RUN_NAME="${1:-k1_vlnce_v3}"
NUM_ENVS="${2:-4096}"
MAX_ITERS="${3:-5000}"

# Kill any stale session
"$TMUX_BIN" kill-session -t "$SESSION" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] launching K1 VLN-CE v3 training"
echo "  task     = Booster-K1-Velocity-Vision-VLNCE-v3"
echo "  run_name = $RUN_NAME"
echo "  num_envs = $NUM_ENVS"
echo "  max_iter = $MAX_ITERS"
echo "  log      = $LOG"
echo "  tmux     = $SESSION"

"$TMUX_BIN" new-session -d -s "$SESSION" \
  "cd $BOOSTER_DIR && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate isaacLab_311 && \
   python scripts/rsl_rl/train.py \
     --task Booster-K1-Velocity-Vision-VLNCE-v3 \
     --headless \
     --num_envs $NUM_ENVS \
     --max_iterations $MAX_ITERS \
     --logger tensorboard \
     --run_name $RUN_NAME \
     2>&1 | tee $LOG"

echo
echo "Monitor:"
echo "  LD_LIBRARY_PATH=\$HOME/miniconda3/lib \$HOME/miniconda3/bin/tmux capture-pane -t $SESSION -p | tail -40"
echo "  tail -F $LOG"
echo
echo "Stop:"
echo "  LD_LIBRARY_PATH=\$HOME/miniconda3/lib \$HOME/miniconda3/bin/tmux kill-session -t $SESSION"
