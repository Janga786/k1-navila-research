#!/bin/bash
# Offline (CPU-only, no Isaac, no GPU, no real model) tests of navila_eval_v3_LOG.py and the analysis
# scripts, using the fake omni.* modules in fake_site/ and fake_vlm_server.py. Writes only under $1.
#   run_offline_tests.sh <scratch_dir>
set -u
OUT=$1; S=$(cd "$(dirname "$0")/.." && pwd); M=$S/mock
rm -rf "$OUT"; mkdir -p "$OUT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate vlnce-isaac
export PYTHONDONTWRITEBYTECODE=1
pass=0; fail=0
ok(){ if [ "$1" = "0" ]; then echo "PASS  $2"; pass=$((pass+1)); else echo "FAIL  $2"; fail=$((fail+1)); fi; }
python "$M/fake_vlm_server.py" --port 54396 > "$OUT/srv_det.log" 2>&1 &  SD=$!
python "$M/fake_vlm_server.py" --port 54395 --nondet 0.5 > "$OUT/srv_nondet.log" 2>&1 &  SN=$!
sleep 2
ev(){ (cd ~/Projects/k1_research/NaVILA-Bench && PYTHONPATH=$M/fake_site python "$S/navila_eval_v3_LOG.py" \
  --task=k1_matterport_vision --num_envs=1 --checkpoint=$HOME/Projects/k1_research/checkpoints/model_14498.pt \
  --episode_idx=92 --gait_phase_init=0.0 --out_tag=t --closed_loop --max_episode_s 120 --vlm_transform stretch \
  --headless --enable_cameras "$@" > /dev/null 2>&1); }
cd "$S"
# fixed render seeds 11/12: a pair known to diverge (q* = 6), so the request tests always have a q*
FAKE_RENDER_SEED=11 ev --vlm_port 54396 --log_dir "$OUT/A"; ok $? "normal run A completes (rc 0)"
FAKE_RENDER_SEED=12 ev --vlm_port 54396 --log_dir "$OUT/B"; ok $? "normal run B completes (rc 0)"
touch -d '10 minutes ago' "$OUT/marker"
python validate_run.py "$OUT/A" --marker "$OUT/marker" --replay-root "$OUT" \
  --sweep-record ~/Projects/k1_research/NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_stretchA_300/measurements/137.json \
  --out "$OUT/val.json" > /dev/null
python -c "import json,sys;d=json.load(open('$OUT/val.json'));sys.exit(0 if all(d[k]['ok'] for k in ('2_png_roundtrip','3_request_rebuild','4_schema','consistency')) else 1)"
ok $? "validation checks 2-4 + consistency pass on run A (PNG round-trip, request rebuild, schema)"
python compare_runs.py pair "$OUT/A" "$OUT/B" --out "$OUT/ab.json" > /dev/null
python -c "import json,sys;d=json.load(open('$OUT/ab.json'));print('      A vs B:', d['qstar'] and (d['qstar']['i'], d['qstar']['step_A']), d['H1'] and d['H1']['verdict']);sys.exit(0 if d['H1'] and d['H1']['verdict']=='held' else 1)"
ok $? "A vs B (different render noise, identical physics): q* found, H1 verdict held"
ev --vlm_port 54396 --log_dir "$OUT/R" --replay_frames_from "$OUT/A"; ok $? "replay run R completes"
python compare_runs.py h3 "$OUT/A" "$OUT/R" --out "$OUT/h3.json" > /dev/null
python -c "import json,sys;sys.exit(0 if json.load(open('$OUT/h3.json'))['H3']=='held' else 1)"; ok $? "R vs A: H3 held with deterministic fake physics and server"
FAKE_PHYSICS_JITTER_AT=350 ev --vlm_port 54396 --log_dir "$OUT/Rj" --replay_frames_from "$OUT/A"
python -c "import json,sys;m=json.load(open('$OUT/Rj/replay_mismatch.json'));sys.exit(0 if (m['what'],m['where']['label'])==('state','99') else 1)"
ok $? "1e-6 m physics jitter after env step 350 -> replay stops with a state mismatch at main step 99"
ev --vlm_port 54395 --log_dir "$OUT/Rn" --replay_frames_from "$OUT/A"
python -c "import json,sys;m=json.load(open('$OUT/Rn/replay_mismatch.json'));sys.exit(0 if m['what']=='reply' else 1)"
ok $? "nondeterministic server -> replay stops with a reply mismatch"
FAKE_RENDER_SEED=1 ev --vlm_port 54396 --log_dir "$OUT/S1"; FAKE_RENDER_SEED=1 ev --vlm_port 54396 --log_dir "$OUT/S2"
python compare_runs.py pair "$OUT/S1" "$OUT/S2" | grep -q "DIVERGED=0"; ok $? "identical renders -> no divergence, identical records"
FAKE_RENDER_SEED=1 ev --vlm_port 54395 --log_dir "$OUT/S3"
python compare_runs.py pair "$OUT/S1" "$OUT/S3" --out "$OUT/s13.json" > /dev/null
python -c "import json,sys;sys.exit(0 if json.load(open('$OUT/s13.json'))['H1']['verdict']=='failed_requests_identical' else 1)"
ok $? "identical requests + nondeterministic server -> H1 verdict failed_requests_identical"
found=1; for s in 3 5; do FAKE_RENDER_SEED=$s FAKE_PHYSICS_JITTER_AT=300 ev --vlm_port 54396 --log_dir "$OUT/J$s"
  python compare_runs.py pair "$OUT/S1" "$OUT/J$s" --out "$OUT/j$s.json" > /dev/null
  python -c "import json,sys;d=json.load(open('$OUT/j$s.json'));sys.exit(0 if d['H1'] and d['H1']['verdict']=='failed_state_before_qstar' else 1)" && found=0; done
ok $found "physics jitter before q* -> H1 verdict failed_state_before_qstar"
python make_qstar_requests.py "$OUT/A" "$OUT/B" "$OUT/rq" > /dev/null; ok $? "q* requests rebuilt from saved frames match the logged hashes"
ev --vlm_port 54396 --log_dir "$OUT/c1" --replay_queries "$OUT/rq/requests.json" --replay_n 4
python compare_runs.py h2 "$OUT/rq/requests.json" "$OUT/c1/replay_queries.jsonl" --n 4 | grep -q '"H2": "held"'; ok $? "condition 1 (--replay_queries in the copy): each request gets its own reply 4/4"
python replay_client.py "$OUT/rq/requests.json" 4 "$OUT/c2/replies.jsonl" cond2 --port 54396 > /dev/null
python compare_runs.py h2 "$OUT/rq/requests.json" "$OUT/c2/replies.jsonl" --n 4 | grep -q '"H2": "held"'; ok $? "condition 2 client: each request gets its own reply 4/4"
python replay_client.py "$OUT/rq/requests.json" 4 "$OUT/c2n/replies.jsonl" cond2n --port 54395 > /dev/null
python compare_runs.py h2 "$OUT/rq/requests.json" "$OUT/c2n/replies.jsonl" --n 4 | grep -q '"H2": "failed"'; ok $? "nondeterministic server -> H2 failed (distribution reported)"
python - <<'PY'
import ast, sys
def f(p):
    t = ast.parse(open(p).read()); return ast.dump(next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "_vlm_send_bytes"))
sys.exit(0 if f("navila_eval_v3_LOG.py") == f("replay_client.py") else 1)
PY
ok $? "replay_client._vlm_send_bytes is AST-identical to the copy's (the evaluator's socket code)"
(cd ~/Projects/k1_research/NaVILA-Bench && PYTHONPATH=$M/fake_site timeout -k 30 4 python "$S/navila_eval_v3_LOG.py" --task=k1_matterport_vision --num_envs=1 --checkpoint=$HOME/Projects/k1_research/checkpoints/model_14498.pt --episode_idx=92 --gait_phase_init=0.0 --out_tag=t --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras --vlm_port 54396 --log_dir "$OUT/T" > /dev/null 2>&1); rc=$?
python -c "import json,sys,numpy as np;m=json.load(open('$OUT/T/measurements/137.json'));np.load('$OUT/T/states_at_queries.npz');sys.exit(0 if $rc==124 and m['term_reason']=='wall_timeout' else 1)"
ok $? "SIGTERM: record (wall_timeout) and states_at_queries.npz written under --log_dir; rc 124"
# Isaac Sim 4.1 mixes Pillow inside the Kit process (PIL.Image 12.3.0 + Kit's prebundled 10.2.0 plugins; this broke
# attempt 1, logs/validate_attempt1.log). FAKE_KIT_PIL_MIX=1 makes the fake AppLauncher reproduce it.
FAKE_KIT_PIL_MIX=1 ev --vlm_port 54396 --log_dir "$OUT/KA"; ok $? "simulated Kit Pillow mix: normal run completes (in-process PNG decode via cv2)"
python -c "import json,sys;fr=[json.loads(l) for l in open('$OUT/KA/frames.jsonl')];sys.exit(0 if sum(1 for f in fr if f.get('png_reload_verified'))==10 else 1)"
ok $? "simulated Kit Pillow mix: the 10 in-process PNG reload checks are true"
python validate_run.py "$OUT/KA" --marker "$OUT/marker" --replay-root "$OUT" \
  --sweep-record ~/Projects/k1_research/NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_stretchA_300/measurements/137.json \
  --out "$OUT/valK.json" > /dev/null
python -c "import json,sys;d=json.load(open('$OUT/valK.json'));sys.exit(0 if all(d[k]['ok'] for k in ('2_png_roundtrip','3_request_rebuild','4_schema','consistency')) else 1)"
ok $? "simulated Kit Pillow mix: validation checks 2-4 + consistency pass (requests encoded under the mix rebuild exactly)"
FAKE_KIT_PIL_MIX=1 ev --vlm_port 54396 --log_dir "$OUT/KR" --replay_frames_from "$OUT/KA"; ok $? "simulated Kit Pillow mix: replay run R completes"
python compare_runs.py h3 "$OUT/KA" "$OUT/KR" --out "$OUT/h3K.json" > /dev/null
python -c "import json,sys;d=json.load(open('$OUT/h3K.json'));sys.exit(0 if d['H3']=='held' and d['R_history_frames_equal_A_frames'] else 1)"
ok $? "simulated Kit Pillow mix: R vs A H3 held, R's history frames = A's frames"
(cd ~/Projects/k1_research/NaVILA-Bench && PYTHONPATH=$M/fake_site python "$S/navila_eval_v3_LOG.py" --checkpoint=x --log_dir ~/Projects/k1_research/NaVILA-Bench-main/eval_results/x > /dev/null 2>&1); [ $? -ne 0 ] && [ ! -e ~/Projects/k1_research/NaVILA-Bench-main/eval_results/x ]; ok $? "refuses a --log_dir under eval_results/ (and creates nothing there)"
(cd ~/Projects/k1_research/NaVILA-Bench && PYTHONPATH=$M/fake_site python "$S/navila_eval_v3_LOG.py" --checkpoint=x --log_dir "$OUT/A" > /dev/null 2>&1); [ $? -ne 0 ]; ok $? "refuses to overwrite an existing run directory"
kill $SD $SN 2>/dev/null
echo "OFFLINE TESTS: $pass passed, $fail failed"
