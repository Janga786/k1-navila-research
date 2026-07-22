#!/usr/bin/env bash
# run_ab100.sh — crop-vs-stretch A/B, 100 episodes per arm (episodes 0-99), sequential
# on the lab RTX 3090 (renderer validity: NEVER run this on a Blackwell GPU).
# Both arms: model_14498 (matches the June 10-episode A/B), int8 bridge, EP_TIMEOUT=900,
# crash-resumable (the harness skips already-scored episodes — just rerun this script).
# Launch detached:  setsid bash run_ab100.sh > /tmp/ab100_chain.log 2>&1 &
set -u
cd "$HOME/Projects/k1_research"
export VLM_BRIDGE_EXTRA="--load_8bit"
export EP_TIMEOUT=900

echo "[ab100] $(date) ARM 1/2: stretch, episodes 0-99"
TRANSFORM=stretch bash scripts/run_powered_benchmark.sh stretch100 "" 0 100
echo "[ab100] $(date) ARM 2/2: crop, episodes 0-99"
TRANSFORM=crop bash scripts/run_powered_benchmark.sh crop100 "" 0 100
echo "[ab100] $(date) BOTH ARMS COMPLETE"
