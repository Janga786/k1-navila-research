#!/bin/bash
# One simulator run of the logging copy, launched the way run_powered_benchmark.sh launches the
# evaluator (PATH + conda env vlnce-isaac, cwd NaVILA-Bench, OMNI_KIT_ACCEPT_EULA=yes, the sweep's
# flags unchanged), except: the script path (the copy), --out_tag, --log_dir (+ any replay flags
# given as extra args), PYTHONDONTWRITEBYTECODE=1 (no .pyc writes into the sweep tree), and a
# 3600-s safety timeout instead of the runner's 900-s kill.
# Preflight (exit 90-95 without running): GPU_LOCK is ours; the only GPU compute process is the
# model server ($SERVER_PID, which must be alive); >= 15 GB free; the log dir is new.
# Every run is logged (label, start, end, wall s, rc, command) to logs/gpu_time.tsv.
#
# Usage: run_sim.sh <label> <episode_idx> <log_dir> [extra evaluator args...]
# Test overrides (unset in real runs): FR_ROOT (logs), FR_LOCK, NVSMI, FR_PYTHONPATH, FR_TIMEOUT, FR_EXTRA_ARGS
set -u
LABEL=$1; IDX=$2; LOGDIR=$3; shift 3
S=$(cd "$(dirname "$0")" && pwd)
R=${FR_ROOT:-$HOME/Projects/k1_research/airc2027_replay}
NB=$HOME/Projects/k1_research/NaVILA-Bench
LOCK=${FR_LOCK:-$HOME/Projects/k1_research/GPU_LOCK}
NVSMI=${NVSMI:-nvidia-smi}
mkdir -p "$R/logs"

[ "$(cat "$LOCK" 2>/dev/null)" = "frame_replay" ] || { echo "[run_sim] GPU_LOCK missing or not ours"; exit 90; }
[ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null || { echo "[run_sim] model server (SERVER_PID=${SERVER_PID:-unset}) is not running"; exit 95; }
apps=$($NVSMI --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
for p in $apps; do
  [ "$p" = "$SERVER_PID" ] || { echo "[run_sim] unexpected GPU compute process $p"; $NVSMI; exit 91; }
done
free_gb=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 15 ] || { echo "[run_sim] low disk: ${free_gb}G free"; exit 92; }
[ -e "$LOGDIR" ] && { echo "[run_sim] log dir exists: $LOGDIR"; exit 93; }
mkdir -p "$(dirname "$LOGDIR")"
echo "[run_sim] $LABEL preflight ok: compute apps=[${apps//$'\n'/,}] server=$SERVER_PID free=${free_gb}G" >> "$R/logs/preflight.log"
$NVSMI --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$R/logs/preflight.log" 2>&1

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
export PYTHONDONTWRITEBYTECODE=1
[ -n "${FR_PYTHONPATH:-}" ] && export PYTHONPATH="$FR_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
cd "$NB"
CMD=(python "$S/navila_eval_v3_LOG.py" --task=k1_matterport_vision --num_envs=1
     --checkpoint="$HOME/Projects/k1_research/checkpoints/model_14498.pt"
     --episode_idx="$IDX" --gait_phase_init=0.0 --out_tag="fr_$LABEL"
     --closed_loop --max_episode_s 120 --vlm_transform stretch
     --headless --enable_cameras --log_dir "$LOGDIR" "$@" ${FR_EXTRA_ARGS:-})   # FR_EXTRA_ARGS: offline tests only
t0=$(date +%s)
timeout -k 30 "${FR_TIMEOUT:-3600}" "${CMD[@]}" > "$R/logs/$LABEL.log" 2>&1
rc=$?
t1=$(date +%s)
printf "%s\t%s\t%s\t%s\t%d\t%s\n" "$LABEL" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" "$rc" \
       "cd $NB && OMNI_KIT_ACCEPT_EULA=yes PYTHONDONTWRITEBYTECODE=1 timeout -k 30 ${FR_TIMEOUT:-3600} ${CMD[*]}" >> "$R/logs/gpu_time.tsv"
echo "[run_sim] $LABEL rc=$rc wall=$((t1-t0))s log=$R/logs/$LABEL.log"
exit $rc
