#!/bin/bash
# Resume K1 VLN-CE training from the v2 checkpoint that was killed by the
# integration smoke test before training had a chance to crash on its own.
#
# Source run dir: 2026-05-18_10-42-27_k1_vlnce_v2  (last save: model_2000.pt
# at reward ~56.2; user-reported iter 1000 was at reward ~47.72)
#
# Behavior:
#   - --resume + --load_run pin selection to the v2 directory (default
#     get_checkpoint_path() picks alphabetically-latest, which would land on
#     v3's stub model_0.pt — so we pin explicitly).
#   - --max_iterations is left to the config default (30000); rsl_rl learn()
#     interprets that as "run 30k MORE iters starting from the loaded iter",
#     so the run will end around iter 32000. Config is otherwise unchanged.
#   - A new timestamped log_dir is created automatically by train.py.
#
# Log:           /tmp/k1_booster_train.log
# Tmux session:  k1_vlnce_train

set -e

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

BOOSTER_DIR=~/Projects/k1_research/booster/booster_train
LOG=/tmp/k1_booster_train.log
RUN_REGEX='.*k1_vlnce_v2$'
RESUME_TAG="${1:-k1_vlnce_v2_resumed}"
NUM_ENVS="${2:-4096}"

# Kill any stale session
"$TMUX_BIN" kill-session -t k1_vlnce_train 2>/dev/null || true

echo "[$(date +%H:%M:%S)] resuming K1 VLN-CE training in tmux 'k1_vlnce_train'..."
echo "  resume_from regex=$RUN_REGEX  num_envs=$NUM_ENVS  run_name=$RESUME_TAG"
echo "  log=$LOG"

"$TMUX_BIN" new-session -d -s k1_vlnce_train \
  "cd $BOOSTER_DIR && \
   source ~/miniconda3/etc/profile.d/conda.sh && \
   conda activate isaacLab_311 && \
   python scripts/rsl_rl/train.py \
     --task Booster-K1-Velocity-Vision-VLNCE-v0 \
     --headless \
     --num_envs $NUM_ENVS \
     --resume \
     --load_run '$RUN_REGEX' \
     --logger tensorboard \
     --run_name $RESUME_TAG \
     2>&1 | tee $LOG"

echo "[$(date +%H:%M:%S)] tmux session 'k1_vlnce_train' started."
echo
echo "Monitor with:"
echo "  tmux capture-pane -t k1_vlnce_train -p | tail -40"
echo "  tail -F $LOG"
