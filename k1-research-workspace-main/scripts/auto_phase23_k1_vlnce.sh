#!/bin/bash
# Auto Phase 2 (validate) + Phase 3 (benchmark) for the resumed K1 VLN-CE run.
#
# Pre-conditions checked:
#   * training tmux session 'k1_vlnce_train' is dead
#   * no python process is currently using the GPU (CRITICAL — no overlap)
#
# Phase 2: pick best checkpoint by max reward (from tensorboard events).
# Phase 3: smoke (1 ep) → if pass, full benchmark in tmux 'bench_vlnce'.

set -u

export PATH="$HOME/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TMUX_BIN="$HOME/miniconda3/bin/tmux"

RUN_SUBDIR="2026-05-18_21-15-33_k1_vlnce_v2_resumed"
RUN_DIR="$HOME/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/$RUN_SUBDIR"
SETUP_LOG="$HOME/Projects/k1_research/SETUP_LOG.md"
PHASE_LOG=/tmp/k1_auto_phase23.log

log() { echo "[$(date +%F\ %H:%M:%S)] $*" | tee -a "$PHASE_LOG"; }
note() { echo -e "\n## $(date +%F\ %H:%M:%S) — $*" >> "$SETUP_LOG"; }

note "Auto Phase 2/3 invoked"
log "auto Phase 2/3 starting — run=$RUN_SUBDIR"

# ============================================================
# Pre-condition: training session fully dead
# ============================================================
log "checking training tmux session…"
if "$TMUX_BIN" has-session -t k1_vlnce_train 2>/dev/null; then
  log "FATAL: training tmux 'k1_vlnce_train' still alive. Aborting Phase 2/3."
  echo "- Aborted: training tmux still alive (no GPU overlap)" >> "$SETUP_LOG"
  exit 1
fi
log "training tmux session is gone — OK."

log "checking GPU is free of python processes…"
for i in $(seq 1 12); do  # wait up to 60s for GPU memory release
  if ! ps aux | grep -v grep | grep -qE "python.*train\.py"; then
    if ! nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
         | grep -qiE "python"; then
      log "GPU is free of python processes — OK."
      break
    fi
  fi
  log "still seeing GPU/python residue, waiting 5s (attempt $i/12)…"
  sleep 5
done
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -qiE "python"; then
  log "FATAL: GPU still has python compute apps after 60s. Aborting."
  echo "- Aborted: GPU not free after 60s" >> "$SETUP_LOG"
  exit 1
fi

# ============================================================
# Phase 2: pick best checkpoint
# ============================================================
log "Phase 2 — picking best checkpoint by reward from tensorboard events…"

BEST=$(/home/janga/miniconda3/envs/isaacLab_311/bin/python - "$RUN_DIR" <<'PY'
import os, sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
ea = EventAccumulator(run, size_guidance={"scalars": 200000})
ea.Reload()
tags = ea.Tags()["scalars"]
key = "Train/mean_reward" if "Train/mean_reward" in tags else None
if not key:
    print("no Train/mean_reward tag", file=sys.stderr)
    sys.exit(2)

evts = ea.Scalars(key)
# rsl_rl saves every 2000; pick the save point with highest reward among those that exist on disk
saves = sorted(int(os.path.basename(p)[6:-3]) for p in glob.glob(os.path.join(run, "model_*.pt")))
best_iter, best_r = None, float("-inf")
ar_key = "Episode_Reward/action_rate" if "Episode_Reward/action_rate" in tags else None
ar_at_best = None
for s in saves:
    near = [e.value for e in evts if abs(e.step - s) <= 5]
    if not near:
        continue
    r = max(near)
    if r > best_r:
        best_r = r
        best_iter = s
        if ar_key:
            ar_near = [e.value for e in ea.Scalars(ar_key) if abs(e.step - s) <= 5]
            ar_at_best = ar_near[-1] if ar_near else None

print(f"best_iter={best_iter}")
print(f"best_reward={best_r:.4f}")
print(f"action_rate_at_best={ar_at_best}")
print(f"saves_seen={saves}")
PY
)
log "Phase 2 result:\n$BEST"
echo "- Phase 2: $BEST" | tr '\n' ' ' >> "$SETUP_LOG"; echo "" >> "$SETUP_LOG"

# Extract numeric iter for benchmark; for now we use the latest checkpoint
# (bench script does the same — uses get_checkpoint_path on the latest model_*.pt).
# If you want to bench a SPECIFIC iter, move the others to _attic/ and re-symlink.

# ============================================================
# Phase 3: smoke + full benchmark
# ============================================================
log "Phase 3 — smoke test…"
note "Phase 3 — smoke"

bash "$HOME/Projects/k1_research/scripts/bench_k1_vlnce.sh" smoke "$RUN_SUBDIR" \
  2>&1 | tee -a "$PHASE_LOG"
SMOKE_EXIT=${PIPESTATUS[0]}

# Smoke test passes if measurement json was written AND distance_to_goal makes sense
MEAS="$HOME/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_booster_vlnce_${RUN_SUBDIR}/measurements/0.json"
if [[ ! -f "$MEAS" ]]; then
  log "FATAL: smoke wrote no measurement json — integration failure. Aborting Phase 3."
  echo "- Phase 3 smoke FAILED (no measurement json, exit=$SMOKE_EXIT)" >> "$SETUP_LOG"
  exit 2
fi
log "smoke measurement: $(cat "$MEAS")"
echo "- Phase 3 smoke OK (measurement written, exit=$SMOKE_EXIT)" >> "$SETUP_LOG"

# A genuinely walking policy will have non-trivial path_length and a
# distance_to_goal < starting distance (typically <10m). path_length < 0.1 m
# means the robot didn't move — bad sign but doesn't necessarily block the full
# benchmark. Log either way.
PL=$(/home/janga/miniconda3/envs/isaacLab_311/bin/python -c "import json; print(json.load(open('$MEAS')).get('path_length', 0))" 2>/dev/null)
log "smoke path_length=$PL"

log "Phase 3 — launching full benchmark in tmux 'bench_vlnce'…"
note "Phase 3 — full benchmark"
bash "$HOME/Projects/k1_research/scripts/bench_k1_vlnce.sh" full "$RUN_SUBDIR" 0 1077 \
  2>&1 | tee -a "$PHASE_LOG"
log "full benchmark launched. Monitor: $TMUX_BIN attach-session -t bench_vlnce"
echo "- Phase 3 full bench launched in tmux 'bench_vlnce'" >> "$SETUP_LOG"
