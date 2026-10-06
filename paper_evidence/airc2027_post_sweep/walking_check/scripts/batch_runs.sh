#!/bin/bash
# Step 1 validation, then (only if it passes) the Step 2 trials, one fresh process each, via run_gpu.sh.
# Usage: batch_runs.sh validate | trials
# GPU budget: no run starts if the logged GPU wall time plus a conservative estimate for that run would exceed 5400 s.
set -u
W=$HOME/Projects/k1_research/airc2027_walking
S=$W/scripts
BUDGET=5400
used() { awk -F'\t' 'NR>1 && $1 !~ /^#/ {s+=$4} END {print s+0}' "$W/logs/gpu_time.tsv" 2>/dev/null || echo 0; }
[ -f "$W/logs/gpu_time.tsv" ] || printf "label\tstart\tend\tseconds\trc\tcommand\n" > "$W/logs/gpu_time.tsv"
run() {  # run <label> <estimate_s> args...
  local label=$1 est=$2; shift 2
  local u; u=$(used)
  if [ $((u + est)) -gt $BUDGET ]; then echo "[batch] BUDGET: used ${u}s + est ${est}s > ${BUDGET}s; not starting $label"; exit 7; fi
  bash "$S/run_gpu.sh" "$label" "$@"
}
case "${1:-}" in
  validate)
    run validate_E_matterport_idx0_cz025 120 --task=k1_matterport_vision --mode validate --robot E \
        --episode_idx=0 --cam_z=0.25 --out "$W/logs/validation/E_matterport_idx0_cz025_init50"
    exit $?
    ;;
  trials)
    # robot T stand repeat 1 first (first T run: surfaces configuration errors), then the rest
    order=("T stand 1")
    for rep in 1 2; do
      for trial in stand forward turn_left turn_right sequence; do
        for robot in E T; do
          [ "$robot $trial $rep" = "T stand 1" ] && continue
          order+=("$robot $trial $rep")
        done
      done
    done
    for item in "${order[@]}"; do
      set -- $item; robot=$1; trial=$2; rep=$3
      est=180; [ "$trial" = sequence ] && est=300
      label="${robot}_${trial}_r${rep}"
      [ -f "$W/logs/states/$label.npz" ] && { echo "[batch] $label exists, skipping"; continue; }
      run "$label" $est --task=k1_walkcheck_plane_$robot --mode trial --robot $robot --trial $trial --rep $rep \
          --episode_idx=0 --cam_z=0.25 --out "$W/logs/states/$label"
      rc=$?
      if [ $rc -ne 0 ]; then echo "[batch] $label failed rc=$rc; stopping"; exit $rc; fi
    done
    echo "[batch] trials done; GPU used $(used)s"
    ;;
  *) echo "usage: $0 validate|trials"; exit 2;;
esac
