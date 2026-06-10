#!/bin/bash
# Resume K1 VLN-CE v3 training from the 5000-cap run that completed at
# 2026-05-19 ~10:00 (model_4999.pt, reward ~100, ep_len ~1451).
#
# rsl_rl interprets --max_iterations as "additional iterations after the
# loaded checkpoint", so 25000 here lands us at iter 30000 total.
#
# Behavior:
#   --resume + --load_run pins to the source v3 run dir (regex anchored
#   so we don't accidentally pick this resume's own fresh model_0.pt).
#   A new timestamped run_dir is created automatically by train.py.

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
# tmux needs the conda libtinfo at runtime
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

BOOSTER_DIR=~/Projects/k1_research/booster/booster_train
LOG=/tmp/k1_booster_train_v3_resume.log
SESSION=k1_vlnce_v3_resume
RUN_REGEX='.*_k1_vlnce_v3$'        # matches "2026-05-19_09-46-39_k1_vlnce_v3"
RESUME_TAG="${1:-k1_vlnce_v3_resumed}"
NUM_ENVS="${2:-4096}"
MAX_ITERS="${3:-25000}"

"$TMUX_BIN" kill-session -t "$SESSION" 2>/dev/null || true

echo "[$(date +%H:%M:%S)] resuming K1 VLN-CE v3 training in tmux '$SESSION'"
echo "  load_run regex = $RUN_REGEX"
echo "  num_envs       = $NUM_ENVS"
echo "  +max_iter      = $MAX_ITERS  (5000 loaded + $MAX_ITERS = $((5000 + MAX_ITERS)) total)"
echo "  log            = $LOG"

"$TMUX_BIN" new-session -d -s "$SESSION" \
  "cd $BOOSTER_DIR && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate isaacLab_311 && \
   python scripts/rsl_rl/train.py \
     --task Booster-K1-Velocity-Vision-VLNCE-v3 \
     --headless \
     --num_envs $NUM_ENVS \
     --resume \
     --load_run '$RUN_REGEX' \
     --max_iterations $MAX_ITERS \
     --logger tensorboard \
     --run_name $RESUME_TAG \
     2>&1 | tee $LOG"

echo
echo "Monitor:"
echo "  tail -F $LOG"
echo "  LD_LIBRARY_PATH=\$HOME/miniconda3/lib \$HOME/miniconda3/bin/tmux capture-pane -t $SESSION -p | tail -40"
