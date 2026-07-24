#!/usr/bin/env bash
# Acceptance tests for the applied fixes + the two root-cause isolation tests (points 5a/5b).
set -u
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
NB="$HOME/Projects/k1_research/NaVILA-Bench"; cd "$NB"
CKPT="$HOME/Projects/k1_research/checkpoints/model_14498.pt"
R="$HOME/Projects/k1_research/receipts"
COMMON="--task=k1_matterport_vision --num_envs=1 --checkpoint=$CKPT --gait_phase_init=0.0 \
  --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras"

echo "########## TEST 1: wall_timeout acceptance (long ep 16 under a 90s wall-clock) ##########"
timeout -k 30 90 python scripts/navila_eval_v3.py $COMMON --episode_idx=16 --out_tag=acc_wt \
  > /tmp/acc_wt.log 2>&1
WTJSON=$(ls eval_results/k1_matterport_vision_loco_acc_wt/measurements/*.json 2>/dev/null | head -1)
echo "-- wall_timeout marker in log:"; grep -c WALL_TIMEOUT /tmp/acc_wt.log
echo "-- JSON produced: ${WTJSON:-NONE}"; [ -n "$WTJSON" ] && cat "$WTJSON"

echo; echo "########## TEST 2: cam_z isolation (native 0.25 vs virtual 0.97) ##########"
python scripts/navila_eval_v3.py $COMMON --episode_idx=3 --out_tag=acc_camA --cam_z 0.25 --diag \
  > /tmp/acc_camA.log 2>&1
python scripts/navila_eval_v3.py $COMMON --episode_idx=3 --out_tag=acc_camB --cam_z 0.97 \
  > /tmp/acc_camB.log 2>&1
echo "-- cam-override A (native):"; grep 'cam-override' /tmp/acc_camA.log
echo "-- cam-override B (1.5m virtual):"; grep 'cam-override' /tmp/acc_camB.log
echo "-- Trunk-z distribution over one walking episode (from diag trajectory):"
python3 - <<'PY'
import numpy as np, glob
t=sorted(glob.glob("/home/boosterk1/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_acc_camA/diag/ep*/trajectory.npy"))
if t:
    a=np.load(t[0])
    z=a[:,2] if a.ndim==2 and a.shape[1]>=3 else None
    if z is not None:
        print(f"   Trunk-z: mean={z.mean():.3f} min={z.min():.3f} max={z.max():.3f} std={z.std():.3f} (m); nominal 0.53")
        print(f"   -> camera abs height (native, +0.25): {0.25+z.mean():.2f}m mean, range [{0.25+z.min():.2f},{0.25+z.max():.2f}]")
else: print("   (no trajectory.npy found)")
PY

echo; echo "########## TEST 5a: render determinism (same pose twice, byte-diff) ##########"
python scripts/l0_camera_dump.py --headless --enable_cameras --episode_idx 3 \
  --cam_z 0.25 --tag detA --out_dir /tmp/rendtest > /tmp/rend_A.log 2>&1
python scripts/l0_camera_dump.py --headless --enable_cameras --episode_idx 3 \
  --cam_z 0.25 --tag detB --out_dir /tmp/rendtest > /tmp/rend_B.log 2>&1
A=/tmp/rendtest/k1_cam_detA_raw512.png; B=/tmp/rendtest/k1_cam_detB_raw512.png
if [ -f "$A" ] && [ -f "$B" ]; then
  if cmp -s "$A" "$B"; then echo "5a RESULT: frames BYTE-IDENTICAL -> rendering is deterministic"
  else
    echo "5a RESULT: frames DIFFER -> sim-side (render/physics) nondeterminism"
    python3 -c "import numpy as np;from PIL import Image;a=np.asarray(Image.open('$A')).astype(int);b=np.asarray(Image.open('$B')).astype(int);d=np.abs(a-b);print(f'   pixel |diff|: max={d.max()} mean={d.mean():.4f} nonzero={100*(d>0).mean():.2f}% of channels')"
  fi
else echo "5a: camera dump did not produce PNGs (see /tmp/rend_A.log)"; fi

echo; echo "########## TEST 5b: VLM determinism (same 8 frames twice, diff tokens) ##########"
python3 - <<'PY'
import socket, json, base64, glob, io
from PIL import Image
frames=sorted(glob.glob("/home/boosterk1/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit/diag/ep7/frames/tick005/frame_*.jpg"))[:8]
if len(frames)<8:
    print("5b: could not find 8 saved frames; skipping"); raise SystemExit
enc=[]
for f in frames:
    buf=io.BytesIO(); Image.open(f).convert("RGB").resize((384,384)).save(buf,format="PNG")
    enc.append(base64.b64encode(buf.getvalue()).decode())
req=json.dumps({"images":enc,"query":"Walk to the goal and stop."}).encode()
def ask():
    s=socket.socket(); s.connect(("localhost",54321)); s.settimeout(120)
    s.sendall(len(req).to_bytes(8,"big")); s.sendall(req)
    n=b""
    while len(n)<8: n+=s.recv(8-len(n))
    ln=int.from_bytes(n,"big"); d=b""
    while len(d)<ln: d+=s.recv(ln-len(d))
    s.close(); return d.decode()
a=ask(); b=ask()
print("   VLM out #1:", a[:120])
print("   VLM out #2:", b[:120])
print("5b RESULT:", "IDENTICAL -> VLM forward is deterministic" if a==b else "DIFFER -> VLM (CUDA non-associativity) is nondeterministic")
PY
echo "########## TESTS DONE ##########"
