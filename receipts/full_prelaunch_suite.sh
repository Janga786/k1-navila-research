#!/usr/bin/env bash
# Full pre-launch suite on driver 580.173.02: acceptance tests + isolation (5a/5b) +
# driver smoke (re-run the 6 determinism episodes x2, compare to the 580.159 baseline).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
R=~/Projects/k1_research/receipts
NB=~/Projects/k1_research/NaVILA-Bench
REPORT=/tmp/ALL_TESTS_REPORT.txt
: > "$REPORT"
say(){ echo "$@" | tee -a "$REPORT"; }

say "=== waiting for VLM bridge ==="
for i in $(seq 1 60); do grep -q 'listening on' /tmp/vlm_bridge.log 2>/dev/null && break; sleep 5; done
say "bridge listening: $(grep -c 'listening on' /tmp/vlm_bridge.log 2>/dev/null)"

say ""; say "########## 1+2. ACCEPTANCE + ISOLATION (5a render, 5b VLM) ##########"
bash "$R/acceptance_and_rootcause.sh" >> "$REPORT" 2>&1

say ""; say "########## 3. DRIVER SMOKE on 580.173.02 (6 eps x2) ##########"
conda activate vlnce-isaac; export OMNI_KIT_ACCEPT_EULA=yes; cd "$NB"
CKPT=~/Projects/k1_research/checkpoints/model_14498.pt
EPS="3 8 16 50 20 99"
runone(){ timeout -k 30 1800 python scripts/navila_eval_v3.py --task=k1_matterport_vision --num_envs=1 \
  --checkpoint="$CKPT" --episode_idx=$2 --gait_phase_init=0.0 --out_tag=$1 --diag \
  --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras \
  > /tmp/smoke173_${1}_ep$2.log 2>&1; }
for i in $EPS; do runone smoke173a $i; done
for i in $EPS; do runone smoke173b $i; done

python3 - >> "$REPORT" <<'PY'
import json, glob, os
NB="/home/boosterk1/Projects/k1_research/NaVILA-Bench"
def load(tag):
    d={}
    for f in glob.glob(f"{NB}/eval_results/k1_matterport_vision_loco_{tag}/measurements/*.json"):
        d[os.path.basename(f)]=json.load(open(f))
    return d
na,nb,oa,ob=load("smoke173a"),load("smoke173b"),load("exta"),load("extb")
print(f"new-driver files: a={len(na)} b={len(nb)}   old-driver(580.159): a={len(oa)} b={len(ob)}")
print("\n-- NEW-DRIVER run-to-run (580.173) = second nondeterminism sample --")
flip=0
for k in sorted(na):
    if k in nb:
        sa,sb=na[k].get('success'),nb[k].get('success'); flip+= (sa!=sb)
        print(f"  {k}: success {sa}->{sb} {'[FLIP]' if sa!=sb else ''}  "
              f"path {na[k].get('path_length',0):.1f}/{nb[k].get('path_length',0):.1f}  "
              f"steps {na[k].get('ended_at_step','?')}/{nb[k].get('ended_at_step','?')}")
print(f"  NEW-driver success flips: {flip}/{len(na)}   (old-driver was 2/6)")
print("\n-- CROSS-DRIVER neighborhood + STRUCTURAL-BREAK flags (smoke173a vs exta) --")
for k in sorted(na):
    if k in oa:
        n,o=na[k],oa[k]; flags=[]
        if n.get('ended_at_step',9999)<20: flags.append("INSTANT-END")
        if n.get('path_length',-1)<0: flags.append("NO-PARTIALS/sentinel")
        if o.get('path_length',0)>0 and n.get('path_length',0)>0:
            rr=n['path_length']/o['path_length']
            if rr>10 or rr<0.1: flags.append(f"PATH-{rr:.1f}x")
        print(f"  {k}: new path {n.get('path_length',0):.1f} vs old {o.get('path_length',0):.1f} | "
              f"new dtg {n.get('distance_to_goal',0):.2f} vs old {o.get('distance_to_goal',0):.2f} | "
              f"new succ {n.get('success')} old {o.get('success')} | "
              f"{'*** '+','.join(flags) if flags else 'in-neighborhood'}")
PY
say ""; say "########## SUITE DONE ##########"
