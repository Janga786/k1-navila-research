#!/bin/bash
# Step 1 (pre-registered in PLAN.md): start the model server (session 1, kept for Steps 2-4 cond 1-2),
# run the logging copy once in normal mode on episode_idx 92 (label `validate`), run the checks.
# Stops at the first failure. Progress: logs/batch.log.
set -u
S=$(cd "$(dirname "$0")" && pwd)
R=${FR_ROOT:-$HOME/Projects/k1_research/airc2027_replay}
PYC=${FR_CPU_PY:-$HOME/miniconda3/envs/vlnce-isaac/bin/python}   # CPU scripts: same PNG encoder as the evaluator
SWEEP_REC=${FR_SWEEP_REC:-$HOME/Projects/k1_research/NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_stretchA_300/measurements/137.json}
VAL_IDX=${FR_VAL_IDX:-92}
LOG=$R/logs/batch.log
export PYTHONDONTWRITEBYTECODE=1
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
stop(){ log "STOPPED: $*"; log "BATCH_VALIDATE_FAILED"; exit 1; }
mkdir -p "$R/logs" "$R/runs"

log "=== STEP 1: validation ==="
SPID=$(bash "$S/server.sh" start session1) || stop "model server session1 did not start (see logs/server_session1.log)"
log "model server session1 up: pid $SPID"
export SERVER_PID=$SPID
touch "$R/logs/validate.marker"; sleep 1
log "RUN validate (episode_idx $VAL_IDX)"
bash "$S/run_sim.sh" validate "$VAL_IDX" "$R/runs/validate"; rc=$?
[ $rc -eq 0 ] || stop "validate run rc=$rc (logs/validate.log)"
"$PYC" "$S/validate_run.py" "$R/runs/validate" --marker "$R/logs/validate.marker" \
       --sweep-record "$SWEEP_REC" --replay-root "$R" --out "$R/logs/validate_report.json" > "$R/logs/validate_checks.txt" 2>&1; rc=$?
cat "$R/logs/validate_checks.txt" | cut -c1-300 | tee -a "$LOG"
[ $rc -eq 0 ] || stop "validation checks failed (logs/validate_report.json)"
log "STEP 1 PASSED"
log "BATCH_VALIDATE_DONE"
