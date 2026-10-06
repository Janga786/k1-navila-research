#!/bin/bash
# FAKE nvidia-smi: compute apps = running fake servers on $FAKE_GPU_PORT (+ PIDs in $FAKE_GPU_PIDS_FILE)
if [[ "$*" == *query-compute-apps* ]]; then
  pgrep -f "^python [^ ]*fake_vlm_server\.py --port ${FAKE_GPU_PORT:-0}( |$)"
  cat "${FAKE_GPU_PIDS_FILE:-/dev/null}" 2>/dev/null; exit 0
fi
if [[ "$*" == *query-gpu* ]]; then echo "fake, 0 MiB, 24576 MiB, 0 %"; exit 0; fi
echo "FAKE nvidia-smi"
