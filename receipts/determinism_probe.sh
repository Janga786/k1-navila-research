#!/usr/bin/env bash
# determinism_probe.sh — run 2 episodes twice under IDENTICAL production settings,
# diff the per-episode JSONs byte-for-byte. No --seed wired, no variation introduced.
# Mirrors run_powered_benchmark.sh's exact per-episode invocation.
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
NB="$HOME/Projects/k1_research/NaVILA-Bench"; cd "$NB"
CKPT="$HOME/Projects/k1_research/checkpoints/model_14498.pt"
EPS="3 8"
runone(){ local tag=$1 i=$2
  timeout -k 30 900 python scripts/navila_eval_v3.py --task=k1_matterport_vision --num_envs=1 \
    --checkpoint="$CKPT" --episode_idx=$i --gait_phase_init=0.0 --out_tag=$tag \
    --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras \
    >/tmp/probe_${tag}_ep${i}.log 2>&1
}
for i in $EPS; do runone probe_a $i; done
for i in $EPS; do runone probe_b $i; done
echo "=== DETERMINISM DIFF (probe_a vs probe_b) ==="
A="$NB/eval_results/k1_matterport_vision_loco_probe_a/measurements"
B="$NB/eval_results/k1_matterport_vision_loco_probe_b/measurements"
for f in "$A"/*.json; do
  n=$(basename "$f")
  if diff -q "$f" "$B/$n" >/dev/null 2>&1; then echo "$n: BIT-IDENTICAL"
  else echo "$n: DIFFERS ->"; diff "$f" "$B/$n"; fi
done
echo "=== DONE ==="