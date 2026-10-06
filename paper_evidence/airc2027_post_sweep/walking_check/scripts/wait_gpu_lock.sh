#!/bin/bash
# Wait for the GPU: every 10 minutes check that ~/Projects/k1_research/GPU_LOCK does not exist and nvidia-smi shows no
# compute process; when both hold, create GPU_LOCK containing "walking_check" atomically (only if still absent) and exit 0.
# Every check is logged to ../logs/gpu_lock_wait.log.
set -u
W=$HOME/Projects/k1_research/airc2027_walking
LOCK=$HOME/Projects/k1_research/GPU_LOCK
LOG=$W/logs/gpu_lock_wait.log
while true; do
  now=$(date -Is)
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
  if [ -e "$LOCK" ]; then
    echo -e "$now\twait\tGPU_LOCK exists ($(cat "$LOCK" 2>/dev/null))\tcompute: ${apps:-none}" >> "$LOG"
  elif [ -n "$apps" ]; then
    echo -e "$now\twait\tno GPU_LOCK\tcompute: $apps" >> "$LOG"
  else
    if ( set -o noclobber; echo "walking_check" > "$LOCK" ) 2>/dev/null; then
      echo -e "$now\tacquired\tGPU_LOCK created (walking_check)\tcompute: none" >> "$LOG"
      echo "[wait_gpu_lock] acquired at $now"
      exit 0
    fi
    echo -e "$now\twait\tlost the race: GPU_LOCK appeared ($(cat "$LOCK" 2>/dev/null))" >> "$LOG"
  fi
  sleep 600
done
