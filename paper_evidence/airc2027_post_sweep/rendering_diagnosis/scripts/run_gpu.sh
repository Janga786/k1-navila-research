#!/bin/bash
# Run one simulator script the way the sweep runner runs the evaluator (conda env vlnce-isaac, cwd
# NaVILA-Bench, --headless --enable_cameras), and log its GPU wall time to ../logs/gpu_time.tsv.
# Usage: run_gpu.sh <label> <script.py> [args...]
set -u
LABEL="$1"; shift; SCRIPT="$1"; shift
R=$HOME/Projects/k1_research/airc2027_renders
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd $HOME/Projects/k1_research/NaVILA-Bench
if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -q .; then
  echo "[run_gpu] another compute process is on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; exit 9
fi
t0=$(date +%s)
timeout -k 15 ${GPU_TIMEOUT:-1500} python "$R/scripts/$SCRIPT" --task=k1_matterport_vision --num_envs=1 \
   --checkpoint=$HOME/Projects/k1_research/checkpoints/model_14498.pt --gait_phase_init=0.0 \
   --headless --enable_cameras "$@" > "$R/logs/$LABEL.log" 2>&1
rc=$?
t1=$(date +%s)
printf "%s\t%s\t%s\t%s\t%d\t%s\n" "$LABEL" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" "$rc" "$SCRIPT $*" >> "$R/logs/gpu_time.tsv"
echo "[run_gpu] $LABEL rc=$rc wall=$((t1-t0))s log=$R/logs/$LABEL.log"
exit $rc
