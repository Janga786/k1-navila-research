#!/usr/bin/env bash
# Sweep runner (driver 580.173.02): stretchA -> stretchB -> crop -> pad -> 6-height sweep.
# Single-pass @900 with guaranteed per-episode records (no retry). Resumable (skips completed
# arms and, within an arm, completed episodes). After each arm: aggregate, copy JSONs into the
# tracked receipts/ dir, git push. Runs in tmux; survives disconnection.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
KR=~/Projects/k1_research
NB=$KR/NaVILA-Bench
RUNNER=$KR/k1-research-workspace-main/scripts/run_powered_benchmark.sh
AGG=$KR/k1-research-workspace-main/aggregate_k1_vision_results.py
CKPT=$KR/checkpoints/model_14498.pt
STATUS=$KR/SWEEP_STATUS.md
SM=$KR/receipts/sweep_measurements
export VLM_BRIDGE_EXTRA=--load_8bit
export EP_TIMEOUT=900
mkdir -p "$SM"

# name transform END extra(cam_z)   [ground height = 0.53 + cam_z]
ARMS=(
"stretchA stretch 300 "
"stretchB stretch 300 "
"crop crop 300 "
"pad pad 300 "
"h060 stretch 200 --cam_z=0.07"
"h078 stretch 200 --cam_z=0.25"
"h095 stretch 200 --cam_z=0.42"
"h110 stretch 200 --cam_z=0.57"
"h130 stretch 200 --cam_z=0.77"
"h150 stretch 200 --cam_z=0.97"
)

[ -f "$STATUS" ] || echo "# Sweep status — driver 580.173.02 — started $(date)" > "$STATUS"
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$STATUS"; }
push(){ cd "$KR"; git add SWEEP_STATUS.md receipts/sweep_measurements 2>/dev/null; \
        git commit -q -m "$1" 2>/dev/null && git push -q origin main 2>/dev/null && say "  pushed: $1"; }

say "SWEEP RUN (re)started"
for entry in "${ARMS[@]}"; do
  read -r name transform end extra <<< "$entry"
  tag="${name}_${end}"
  MEAS="$NB/eval_results/k1_matterport_vision_loco_${tag}/measurements"
  have=$(ls "$MEAS"/*.json 2>/dev/null | wc -l)
  if [ "$have" -ge "$end" ]; then say "ARM SKIP (complete): $name ($have/$end)"; continue; fi
  say "ARM START: $name  transform=$transform  eps 0-$end  extra='${extra:-none}'  (have $have)"
  TRANSFORM=$transform bash "$RUNNER" "$tag" "$CKPT" 0 "$end" "$extra"
  have=$(ls "$MEAS"/*.json 2>/dev/null | wc -l)
  say "ARM DONE: $name — $have/$end scored"
  python3 "$AGG" --measurements-dir "$MEAS" --total-episodes "$end" 2>/dev/null \
    | grep -E 'n_present|success_over_total|oracle_success_over_total' | sed 's/^/    /' | tee -a "$STATUS"
  # term_reason breakdown (records + wall_timeout incidence)
  python3 - "$MEAS" >> "$STATUS" 2>/dev/null <<'PY'
import json,glob,sys,collections
c=collections.Counter()
for f in glob.glob(sys.argv[1]+"/*.json"):
    try: c[json.load(open(f)).get("term_reason","?")]+=1
    except: pass
print("    term_reason:", dict(c))
PY
  mkdir -p "$SM/$tag"; cp "$MEAS"/*.json "$SM/$tag/" 2>/dev/null
  push "Sweep arm complete: $name ($have/$end)"
done
say "ALL ARMS COMPLETE"
push "Sweep COMPLETE — all arms"
