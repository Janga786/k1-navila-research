#!/usr/bin/env bash
# Extended determinism probe: 6 deliberately-chosen episodes x2, identical production
# settings, --diag for ended_at_step (step counts) + term_reason. No --seed, no variation.
#   3,8  = short (VLM ~16 ticks in pilot)
#   16,50 = longest completed (36.3 / 35.1 m paths = step-cap timeouts)
#   20,99 = final distance 3.13 / 3.23 m (straddle the 3.0 m success radius)
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
NB="$HOME/Projects/k1_research/NaVILA-Bench"; cd "$NB"
CKPT="$HOME/Projects/k1_research/checkpoints/model_14498.pt"
EPS="3 8 16 50 20 99"
runone(){ local tag=$1 i=$2
  timeout -k 30 1800 python scripts/navila_eval_v3.py --task=k1_matterport_vision --num_envs=1 \
    --checkpoint="$CKPT" --episode_idx=$i --gait_phase_init=0.0 --out_tag=$tag --diag \
    --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras \
    >/tmp/probe_ext_${tag}_ep${i}.log 2>&1
}
for i in $EPS; do runone exta $i; done
for i in $EPS; do runone extb $i; done
echo "=== EXTENDED DETERMINISM DIFF (exta vs extb) ==="
A="$NB/eval_results/k1_matterport_vision_loco_exta/measurements"
B="$NB/eval_results/k1_matterport_vision_loco_extb/measurements"
for f in "$A"/*.json; do
  n=$(basename "$f"); steps=$(python3 -c "import json;print(json.load(open('$f')).get('ended_at_step','?'))")
  if diff -q "$f" "$B/$n" >/dev/null 2>&1; then echo "$n (ended_at_step=$steps): BIT-IDENTICAL"
  else echo "$n (ended_at_step=$steps): DIFFERS ->"; diff "$f" "$B/$n"; fi
done
echo "=== EXT DONE ==="