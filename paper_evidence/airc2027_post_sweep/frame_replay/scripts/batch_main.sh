#!/bin/bash
# Steps 2-4 (pre-registered in PLAN.md), with the model server session 1 started by batch_validate.sh.
#  Step 2: per episode (table order) runs A and B in fresh processes; if no reply differs, C, then D.
#  Step 3: run R (--replay_frames_from run A) for each episode with a diverging pair.
#  Step 4: rebuild both q* requests; cond 1 = evaluator copy --replay_queries (simulator loaded);
#          cond 2 = client, server alone; restart server (session 2); cond 3 = client, no simulator.
# Stops cleanly at the first failure; never kills other processes; leaves session 2 running (it is
# stopped after the upload, Step 6). Budget guard: GPU clock = session 1 start; a unit starts only if
# elapsed + estimate <= 8 h 45 min (C/D of an episode already started: <= 9 h).
set -u
S=$(cd "$(dirname "$0")" && pwd)
R=${FR_ROOT:-$HOME/Projects/k1_research/airc2027_replay}
PYC=${FR_CPU_PY:-$HOME/miniconda3/envs/vlnce-isaac/bin/python}
NVSMI=${NVSMI:-nvidia-smi}
LOG=$R/logs/batch.log
EPS=(${FR_EPISODES:-3 8 16 20 50 99 92 212})
declare -A MAXLEN=([3]=1493 [8]=1573 [16]=4068 [20]=6001 [50]=6001 [99]=2772 [92]=926 [212]=1049)
SOFT=${FR_SOFT_S:-31500}; HARD=${FR_HARD_S:-32400}
N_SENDS=${FR_N_SENDS:-20}
CLIENT_PORT=${FR_PORT:-54321}
export PYTHONDONTWRITEBYTECODE=1
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
stop(){ log "STOPPED: $*"; log "BATCH_MAIN_FAILED"; exit 1; }
read -r SPID T0 _ < "$R/logs/server_session1.state" || stop "no logs/server_session1.state"
export SERVER_PID=$SPID
elapsed(){ echo $(( $(date +%s) - T0 )); }
fits(){ [ $(( $(elapsed) + $1 )) -le "$2" ]; }
est_run(){ echo $(( 90 + ($1 * 14) / 100 )); }
mkdir -p "$R/replay"
BUDGET_HIT=0
budget_skip(){ BUDGET_HIT=1; log "BUDGET: elapsed $(elapsed) s; skipping $1 and every later GPU unit"; }

sim(){  # sim <label> <episode_idx> <dir> [extra...]   (normal or replay run; must produce a record)
  local label=$1 idx=$2 dir=$3; shift 3
  kill -0 "$SERVER_PID" 2>/dev/null || stop "model server $SERVER_PID is not running"
  log "RUN $label (episode_idx $idx) elapsed=$(elapsed)s"
  bash "$S/run_sim.sh" "$label" "$idx" "$dir" "$@"; local rc=$?
  [ $rc -eq 0 ] || stop "$label rc=$rc (logs/$label.log)"
  [ -e "$dir/error.txt" ] && stop "$label wrote error.txt"
  local info
  info=$("$PYC" -c "
import json, glob
f = glob.glob('$dir/measurements/*.json')
m = json.load(open(f[0])) if len(f) == 1 else None
print('NONE' if m is None else f\"{m['ended_at_step']} {m['term_reason']} {m['success']}\")") || stop "$label: cannot read record"
  [ "$info" = "NONE" ] && stop "$label: no record JSON"
  case "$info" in *vlm_error*|*wall_timeout*) stop "$label ended with: $info";; esac
  log "  $label done: ended_at_step/term_reason/success = $info; queries=$(wc -l < "$dir/queries.jsonl")"
}

only_server_on_gpu(){
  local apps p; apps=$($NVSMI --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  for p in $apps; do [ "$p" = "$SERVER_PID" ] || stop "unexpected GPU compute process $p before $1"; done
}

client(){  # client <cond> <idx>
  local cond=$1 idx=$2 t0 t1 rc
  kill -0 "$SERVER_PID" 2>/dev/null || stop "model server not running before $cond ep$idx"
  only_server_on_gpu "$cond ep$idx"
  t0=$(date +%s)
  "$PYC" "$S/replay_client.py" "$R/replay/ep$idx/requests.json" "$N_SENDS" "$R/replay/ep$idx/$cond/replies.jsonl" "$cond" \
        --port "$CLIENT_PORT" > "$R/logs/${cond}_ep$idx.log" 2>&1; rc=$?
  t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%d\t%s\n" "${cond}_ep$idx" "$(date -d @$t0 -Is)" "$(date -d @$t1 -Is)" "$((t1-t0))" "$rc" \
         "replay_client.py replay/ep$idx/requests.json $N_SENDS (server pid $SERVER_PID)" >> "$R/logs/client_sessions.tsv"
  [ $rc -eq 0 ] || stop "$cond ep$idx client rc=$rc"
  "$PYC" "$S/compare_runs.py" h2 "$R/replay/ep$idx/requests.json" "$R/replay/ep$idx/$cond/replies.jsonl" \
        --n "$N_SENDS" --out "$R/replay/ep$idx/$cond/h2.json" > /dev/null || stop "h2 check failed for $cond ep$idx"
  log "  $cond ep$idx: $("$PYC" -c "import json;d=json.load(open('$R/replay/ep$idx/$cond/h2.json'));print(d['H2'], [(r['label'], r['own_reply_count'], r['sends']) for r in d['requests']])")"
}

# ------------------------------- Step 2 -------------------------------------
log "=== STEP 2: runs A and B (extra runs C, D if no reply differs) ==="
: > "$R/replay/pairs.tsv"
for idx in "${EPS[@]}"; do
  fits $(( 2 * $(est_run ${MAXLEN[$idx]}) )) "$SOFT" || { budget_skip "Step 2 episode $idx"; break; }
  sim "ep${idx}_A" "$idx" "$R/runs/ep${idx}_A"
  sim "ep${idx}_B" "$idx" "$R/runs/ep${idx}_B"
  pair=none
  for X in B C D; do
    if [ "$X" != B ]; then
      fits "$(est_run ${MAXLEN[$idx]})" "$HARD" || { budget_skip "run ep${idx}_$X"; break 2; }
      sim "ep${idx}_$X" "$idx" "$R/runs/ep${idx}_$X"
    fi
    "$PYC" "$S/compare_runs.py" pair "$R/runs/ep${idx}_A" "$R/runs/ep${idx}_$X" --images \
          --out "$R/replay/compare_ep${idx}_A_$X.json" > "$R/logs/compare_ep${idx}_A_$X.txt" 2>&1 \
          || stop "compare_runs failed for ep$idx A vs $X"
    q=$("$PYC" -c "import json;d=json.load(open('$R/replay/compare_ep${idx}_A_$X.json'));q=d['qstar'];print('q*=%s step=%s/%s H1=%s' % ((q['i'], q['step_A'], q['step_X'], d['H1']['verdict']) if q else ('none','-','-','-')) + ' first_state_diff=' + str(d['first_state_diff'] and d['first_state_diff']['label']))")
    log "  ep$idx A vs $X: $q"
    if grep -q "DIVERGED=1" "$R/logs/compare_ep${idx}_A_$X.txt"; then pair=$X; break; fi
  done
  printf "%s\t%s\n" "$idx" "$pair" >> "$R/replay/pairs.tsv"
  log "  ep$idx: pair = A vs $pair"
done
[ $BUDGET_HIT -eq 1 ] && { log "BATCH_MAIN_BUDGET_STOP"; exit 0; }

mapfile -t PAIRS < <(awk -F'\t' '$2 != "none" {print $1 "\t" $2}' "$R/replay/pairs.tsv")
log "diverging pairs: ${#PAIRS[@]} of ${#EPS[@]}"

# ------------------------------- Step 3 -------------------------------------
log "=== STEP 3: run R (model history = run A's saved frames) ==="
for line in "${PAIRS[@]}"; do
  idx=${line%%$'\t'*}
  la=$("$PYC" -c "import json,glob;print(json.load(open(glob.glob('$R/runs/ep${idx}_A/measurements/*.json')[0]))['ended_at_step'])")
  fits "$(est_run "$la")" "$SOFT" || { budget_skip "Step 3 episode $idx"; break; }
  sim "ep${idx}_R" "$idx" "$R/runs/ep${idx}_R" --replay_frames_from "$R/runs/ep${idx}_A"
  "$PYC" "$S/compare_runs.py" h3 "$R/runs/ep${idx}_A" "$R/runs/ep${idx}_R" --out "$R/replay/h3_ep$idx.json" > /dev/null \
        || stop "h3 check failed for ep$idx"
  log "  ep$idx: $("$PYC" -c "import json;d=json.load(open('$R/replay/h3_ep$idx.json'));print('H3', d['H3'], 'mismatch', d['replay_mismatch'] and (d['replay_mismatch']['what'], d['replay_mismatch']['where']))")"
done
[ $BUDGET_HIT -eq 1 ] && { log "BATCH_MAIN_BUDGET_STOP"; exit 0; }

# ------------------------------- Step 4 -------------------------------------
log "=== STEP 4: model replays at q* ==="
for line in "${PAIRS[@]}"; do
  idx=${line%%$'\t'*}; X=${line##*$'\t'}
  "$PYC" "$S/make_qstar_requests.py" "$R/runs/ep${idx}_A" "$R/runs/ep${idx}_$X" "$R/replay/ep$idx" \
        > "$R/logs/qstar_requests_ep$idx.txt" 2>&1 || stop "rebuilt q* request does not match its logged hash (ep$idx)"
  log "  ep$idx q* requests rebuilt and verified: $(tr '\n' ' ' < "$R/logs/qstar_requests_ep$idx.txt")"
done
log "--- condition 1: simulator loaded (evaluator copy --replay_queries) ---"
for line in "${PAIRS[@]}"; do
  idx=${line%%$'\t'*}
  fits 400 "$SOFT" || { budget_skip "condition 1 episode $idx"; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || stop "model server not running before cond1 ep$idx"
  log "RUN ep${idx}_cond1 elapsed=$(elapsed)s"
  bash "$S/run_sim.sh" "ep${idx}_cond1" "$idx" "$R/replay/ep$idx/cond1" \
       --replay_queries "$R/replay/ep$idx/requests.json" --replay_n "$N_SENDS"; rc=$?
  [ $rc -eq 0 ] || stop "ep${idx}_cond1 rc=$rc"
  [ -e "$R/replay/ep$idx/cond1/error.txt" ] && stop "ep${idx}_cond1 wrote error.txt"
  "$PYC" "$S/compare_runs.py" h2 "$R/replay/ep$idx/requests.json" "$R/replay/ep$idx/cond1/replay_queries.jsonl" \
        --n "$N_SENDS" --out "$R/replay/ep$idx/cond1/h2.json" > /dev/null || stop "h2 check failed for cond1 ep$idx"
  log "  cond1 ep$idx: $("$PYC" -c "import json;d=json.load(open('$R/replay/ep$idx/cond1/h2.json'));print(d['H2'], [(r['label'], r['own_reply_count'], r['sends']) for r in d['requests']])")"
done
[ $BUDGET_HIT -eq 1 ] && { log "BATCH_MAIN_BUDGET_STOP"; exit 0; }
log "--- condition 2: server alone ---"
fits 600 "$SOFT" || { budget_skip "condition 2"; log "BATCH_MAIN_BUDGET_STOP"; exit 0; }
for line in "${PAIRS[@]}"; do client cond2 "${line%%$'\t'*}"; done
log "--- condition 3: restarted server, no simulator ---"
fits 1200 "$SOFT" || { budget_skip "condition 3"; log "BATCH_MAIN_BUDGET_STOP"; exit 0; }
bash "$S/server.sh" stop session1 | tee -a "$LOG"
SPID=$(bash "$S/server.sh" start session2) || stop "model server session2 did not start"
export SERVER_PID=$SPID
log "model server session2 up: pid $SPID"
for line in "${PAIRS[@]}"; do client cond3 "${line%%$'\t'*}"; done
log "STEPS 2-4 COMPLETE (server session2 left running until after the upload)"
log "BATCH_MAIN_DONE"
