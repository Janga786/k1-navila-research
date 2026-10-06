#!/bin/bash
# Model server, started as run_powered_benchmark.sh starts it with VLM_BRIDGE_EXTRA=--load_8bit
# (arms_runner.sh): conda env navila, cwd NaVILA-Bench,
#   python scripts/vlm_server_bridge.py --model_path .../navila-llama3-8b-8f --port 54321 --load_8bit
# Differences, none functional: `exec` (the setsid'd shell becomes the server, so its PID is known),
# PYTHONDONTWRITEBYTECODE=1, and the log goes to airc2027_replay/logs/server_<label>.log.
#   server.sh start <label>   -> prints the server PID once it logs "listening on"
#   server.sh stop  <label>   -> SIGTERM (SIGKILL after 60 s); logs the session to logs/gpu_time.tsv
# Test overrides: FR_ROOT, FR_LOCK, NVSMI, FR_PORT, FR_SERVER_ENV, FR_SERVER_CMD
set -u
ACTION=$1; LABEL=$2
R=${FR_ROOT:-$HOME/Projects/k1_research/airc2027_replay}
NB=$HOME/Projects/k1_research/NaVILA-Bench
LOCK=${FR_LOCK:-$HOME/Projects/k1_research/GPU_LOCK}
NVSMI=${NVSMI:-nvidia-smi}
PORT=${FR_PORT:-54321}
ENVNAME=${FR_SERVER_ENV:-navila}
SCMD=${FR_SERVER_CMD:-python scripts/vlm_server_bridge.py --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f --port 54321 --load_8bit}
STATE=$R/logs/server_$LABEL.state
LOG=$R/logs/server_$LABEL.log
mkdir -p "$R/logs"

case "$ACTION" in
start)
  [ "$(cat "$LOCK" 2>/dev/null)" = "frame_replay" ] || { echo "GPU_LOCK missing or not ours" >&2; exit 90; }
  apps=$($NVSMI --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  [ -z "$apps" ] || { echo "GPU busy: compute processes [$apps]" >&2; $NVSMI >&2; exit 91; }
  ss -ltn | grep -q ":$PORT " && { echo "port $PORT already in use" >&2; exit 94; }
  [ -e "$STATE" ] && { echo "state file exists: $STATE" >&2; exit 93; }
  t0=$(date +%s)
  setsid bash -c "
      cd $NB; source ~/miniconda3/etc/profile.d/conda.sh; conda activate $ENVNAME
      export PYTHONDONTWRITEBYTECODE=1
      exec $SCMD > $LOG 2>&1
  " < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 180); do grep -q "listening on" "$LOG" 2>/dev/null && break; sleep 5; done
  grep -q "listening on" "$LOG" 2>/dev/null || { echo "server not ready after 15 min" >&2; tail -30 "$LOG" >&2; exit 2; }
  SPID=$(pgrep -f "^python [^ ]*vlm_server_bridge\.py --model_path .* --port $PORT( |$)|^python [^ ]*fake_vlm_server\.py --port $PORT( |$)" | head -1)
  [ -n "$SPID" ] || { echo "server PID not found" >&2; exit 2; }
  echo "$SPID $t0 $SCMD" > "$STATE"
  echo "$SPID"
  ;;
stop)
  read -r SPID t0 rest < "$STATE"
  sig=none
  if kill -0 "$SPID" 2>/dev/null; then
    kill "$SPID"; sig=TERM
    for i in $(seq 1 60); do kill -0 "$SPID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$SPID" 2>/dev/null; then kill -9 "$SPID"; sig=KILL; sleep 2; fi
  fi
  t1=$(date +%s)
  rc=$([ "$sig" = TERM ] && echo 143 || { [ "$sig" = KILL ] && echo 137 || echo -1; })
  printf "%s\t%s\t%s\t%s\t%d\t%s\n" "server_$LABEL" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" "$rc" \
         "cd $NB && conda activate $ENVNAME && $rest (stopped by SIG$sig; rc = 128+signal)" >> "$R/logs/gpu_time.tsv"
  mv "$STATE" "$STATE.done"
  echo "server $SPID stopped (SIG$sig) after $((t1-t0)) s"
  ;;
*) echo "usage: server.sh start|stop <label>" >&2; exit 64;;
esac
