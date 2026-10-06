#!/bin/bash
# Run one walk_check.py process the way the sweep runner runs the evaluator (conda env vlnce-isaac, cwd NaVILA-Bench,
# --num_envs=1 --checkpoint model_14498.pt --gait_phase_init=0.0 --headless --enable_cameras) and log its GPU wall time
# to ../logs/gpu_time.tsv. Refuses to start unless this check holds GPU_LOCK and the GPU has no compute process.
# Usage: run_gpu.sh <label> --task=<task> --mode ... [walk_check.py args...]
set -u
LABEL="$1"; shift
W=$HOME/Projects/k1_research/airc2027_walking
LOCK=$HOME/Projects/k1_research/GPU_LOCK
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
cd $HOME/Projects/k1_research/NaVILA-Bench
if [ "$(cat "$LOCK" 2>/dev/null)" != "walking_check" ]; then
  echo "[run_gpu] GPU_LOCK is not held by walking_check: '$(cat "$LOCK" 2>/dev/null)'"; exit 8
fi
if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | grep -q .; then
  echo "[run_gpu] another compute process is on the GPU:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; exit 9
fi
CMD=(python "$W/scripts/walk_check.py" --num_envs=1 --checkpoint=$HOME/Projects/k1_research/checkpoints/model_14498.pt
     --gait_phase_init=0.0 --headless --enable_cameras "$@")
t0=$(date +%s)
timeout -k 15 ${GPU_TIMEOUT:-600} "${CMD[@]}" > "$W/logs/$LABEL.log" 2>&1
rc=$?
t1=$(date +%s)
printf "%s\t%s\t%s\t%s\t%d\t%s\n" "$LABEL" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" "$rc" \
  "(cwd NaVILA-Bench, env vlnce-isaac) ${CMD[*]}" >> "$W/logs/gpu_time.tsv"
echo "[run_gpu] $LABEL rc=$rc wall=$((t1-t0))s log=$W/logs/$LABEL.log"
exit $rc
