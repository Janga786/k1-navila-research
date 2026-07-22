# EVIDENCE.md — claim-by-claim verification against primary sources
Compiled 2026-07-21 on the lab machine. Verdicts: ✅ VERIFIED · ⚠️ VERIFIED-WITH-NUANCE
(number right, framing needs care) · 🔴 CONFLICT (slide number disagrees with primary
source) · ⛔ NOT ON DISK (cannot be verified from this machine — fix or remove).
Generated tables/scripts live in `receipts/`. Paths are relative to
`~/Projects/k1_research/` unless absolute.

---
## SECTION 1 — slide-number verification

### 1. Reward table for the model_14498 run — 🔴 CONFLICT on the count, ✅ on the values
- **Primary source:** the run's own dumped config:
  `~/robots/k1/workspace/booster_train/logs/rsl_rl/k1_navila/2026-06-10_14-30-20/params/env.yaml`
- **Verified: 19 terms** for model_14498. **"20 rewards" is TRUE only for the final
  model_14200 recipe** (19 + `feet_distance_band` −0.4; primary:
  `.../2026-07-19_10-50-11/params/env.yaml`, verified 20 terms). Pick one model per slide.
- **feet_swing: BOTH numbers are real but they are different terms** —
  `feet_swing` (contact schedule) = **+3.0**; `feet_swing_height` (swing clearance /
  circumduction fix) = **−20.0**. A slide saying "feet_swing −20" conflates them.
- **track-linear-velocity = 1.5** (per axis, x and y separately; yaw also 1.5).
  1.0 is Booster Gym T1's value — used only in the rejected T1-recipe experiments.
- **termination = −200.0** ✅.
- Full table + regen one-liner: `receipts/reward_tables.md`.

### 2. PD gains — ✅ (one damping nuance)
- Fixed benchmark config: `NaVILA-Bench-main/.../config/k1/k1_matterport_vision_cfg.py:129-130`
  (stiffness **100.0** / damping **2.0**, hips+knees) and `:142-143` (**50.0 / 1.0**, ankles).
- Broken prior (preserved backup `k1_matterport_vision_cfg.py.orig_gains:128-129,141-142`):
  stiffness **350.0** / damping **7.5** and **250.0 / 5.0**. ⚠️ "350/250" is correct for the
  *stiffnesses*; the damping was 7.5/5 (not 250's pair) — phrase as "stiffness 350/250".
- Deploy config identical: `booster_deploy/tasks/k1_navila/k1_navila.py:179-192`
  (legs/knees 100, ankles 50; damping 2/1). Training identical:
  `booster_train/.../config/k1/navila_env_cfg.py:77` (`("legs",100,2),("feet",50,1)`).

### 3. Observation contract — ✅
- **235 = 47 × 5**: term-major stack of 47-dim frames; layout documented and enforced in
  `booster_deploy/tasks/k1_navila/k1_navila.py:12-21,53-54` and
  `booster_train/.../config/k1/navila_env_cfg.py:5-8,134-138`.
- **Max disagreement ≤1.4e-6**: fresh run 2026-07-19 →
  `booster_train/round5_lateral/logs/validate_contract.log`: obs 1.431e-06, act 1.431e-06,
  `CONTRACT PASS`. Regenerate (~30 s, CPU):
  `cd ~/robots/k1/workspace/booster_deploy && ~/envs/navilaenv/bin/python tasks/k1_navila/validate_contract.py`
- **Three stacks** (state precisely — it is a chain, not one test): (a) benchmark wrapper
  `wrappers_v3.py` validated slice-by-slice at runtime vs sim ground truth
  (`scripts/l2_obs_frame_check.py`, audit gate L1.1/L2c green); (b) the MuJoCo gate
  ObsBuilder validated against the benchmark stack 2026-06-10 (docstring
  `k1_navila.py:8-10`); (c) deploy policy validated byte-level vs that gate reference
  (`validate_contract.py`, the 1.4e-6).

### 4. Camera — ✅ numbers; 🔴 "fixing commit" does not exist
- Broken (backup `k1_matterport_vision_cfg.py.orig_camfix:387-395`): **512×512,
  horizontal_aperture 54.0, z = 0.85 m** — note this is literally H1's camera block
  left unadapted (see below), a good slide line.
- Fixed (live `k1_matterport_vision_cfg.py:391-402`): **1280×720, aperture 47.71
  (= 24.0·1280/fx 643.9 → HFOV ≈ 90°), pos (0.10, 0, 0.25) on Trunk, level** — with the
  0.78 m eye-height derivation in the in-file comment (`:391-393`; 0.53 m standing + 0.25).
- 🔴 **No fixing commit exists**: the fix predates the repo (initial commit `acd6b5c`
  2026-06-10 already contains it). Cite instead: the `.orig_camfix` backup +
  `NaVILA-Bench-main/PATCH_k1_accuracy_benchmark.patch` + audit gate L0.1
  (`~/Desktop/NAVILA_AUDIT_2026-06-10.md`).
- **H1 camera height ≈ 1.5 m is DERIVED, not a literal line**: pelvis-mounted at
  +0.5 m (`.../config/h1/h1_matterport_base_cfg.py:301-305` — pos (0.1, 0, 0.5),
  512×512, aperture 54.0) + H1 pelvis init height 1.05 m
  (`~/IsaacLab/source/isaaclab_assets/.../unitree.py:200`) → **≈1.55 m**. Say "~1.5 m
  (pelvis + 0.5 m)".

### 5. Early-termination bug (0.8 rad → ~61% of episodes) — ⚠️
- Live evidence: `config/k1/k1_matterport_base_cfg.py:261` —
  `{"limit_angle": 1.3},  # was 0.8 (fired mid-stride, killed ~61% of episodes)` and
  `k1_matterport_vision_cfg.py:305` (1.3). Audit bug rank 13 restates it.
- ⚠️ The **raw analysis log behind "61%" is not on disk** — the number survives as a
  config comment + audit text. Present as "~61% (recorded during the June fix)" or
  re-measure before printing a precise figure.

### 6. Parser bug ({15,30,45}° buckets; 90°→15°) — ✅ (fix has no dedicated commit)
- Upstream buggy source, verbatim from the pristine archive:
  `unzip -p ~/Downloads/NaVILA-Bench-main.zip '*/utils/eval_utils.py'` — the
  `get_vel_command` bucket chain matches only "45"/"30"/"15" and falls through to
  `time_to_go=0.5 s`; with wz = π/6 (30°/s), **0.5 s ⇒ 15°** for ANY other angle
  (60/75/90/120/180). The bare `"move"` substring hijack is also visible upstream.
- Fixed version: live `.../utils/eval_utils.py:58-84` (regex angle extraction, comment
  "audit rank 3"; `[PARSE_FAIL]` hold). Diff to regenerate:
  `diff <(unzip -p ~/Downloads/NaVILA-Bench-main.zip '*/utils/eval_utils.py') NaVILA-Bench-main/isaaclab_exts/.../utils/eval_utils.py`
- Like claim 4: the fix predates the repo — no dedicated commit; the audit
  (`NAVILA_AUDIT_2026-06-10.md`, bug 3) is the documented diagnosis.

### 7. v1 baseline SR 0.76% — ⛔ RAW DATA NOT ON DISK
- The number appears in `FINAL_RESULTS_full1077.md:20` ("K1 — prior stack (reference)
  v3 model_16000 … SR 0.76") and `BENCHMARK_32GB_CHECKLIST.md` (with OS ~10%, full 1077).
- **No eval_results directory for the old stack exists on this machine**
  (`NaVILA-Bench-main/eval_results/` holds only 14498-era runs). The raw run predates the
  current tree / lived on the desktop. On a slide, cite it as "recorded prior-stack
  baseline (project docs)" — or soften to "<1%". Do not present as regenerable.

### 8. Final run SR/OS/NE/SPL — ✅ REGENERATED EXACTLY
- `python3 receipts/recompute_final_metrics.py` over the 1,077 per-episode JSONs
  (`NaVILA-Bench/eval_results/k1_matterport_vision_loco_full_14498/measurements/`) →
  **SR 18.29% (197/1077), OS 30.27% (326/1077), NE 7.59 m, SPL 10.93%, 1077/1077 present**.
  Output archived: `receipts/final_metrics_output.txt`.

### 9. Velocity tracking 0.007 m/s — 🔴 CONFLICT with fresh regeneration
- Recorded value: 0.007 m/s (`~/Desktop/PROJECT_DOSSIER_AND_PAPER_ANALYSIS.md:103`,
  `checkpoints/README.md`), measured June via `eval_navila_tracking.py` (32 envs).
- **Fresh regeneration today** (`receipts/tracking_14498_regen.log`, command:
  `~/run_isaac.sh scripts/rsl_rl/eval_navila_tracking.py --headless --num_envs 32 --checkpoint <14498>`):
  **ss_vx error = 0.016 m/s** (0.484 @ 0.5 cmd), onset 0.40 s (matches recorded exactly),
  1.5 s-dist 0.626 (matches recorded exactly). The 0.007-vs-0.016 gap is run-to-run eval
  variance. **Slide-safe phrasing: "steady-state velocity error < 0.02 m/s (measured
  0.007–0.016), ~4× better than the NaVILA paper's 0.066 Go2 reference."**
- ✅ **0.066 VERIFIED against arXiv 2412.04453 HTML (2026-07-22)**: Table V "Linear Vel.
  Error" = 0.066, Section III-B. Also verified: every H1/Go2 baseline in our docs matches
  the paper's Table IV exactly (Go2 blind 6.03/49.0/36.2/33.3, Go2 vision 5.49/58.7/50.2/45.5,
  H1 blind 7.67/33.3/24.4/21.0, H1 vision 5.86/54.6/45.3/40.3).

---
## SECTION 2 — paper evidence chain

### 10. Failure decomposition & stop-conversion — ✅ REGENERATED EXACTLY
- Same script/output as claim 8: **never-reached 751 (69.7%) / reached-no-stop 129
  (12.0%) / success 197 (18.3%)**; stop-given-reached **197/326 = 60.4%**.
- H1-blind 73.3% is a **derivation from paper Table values** (SR 24.4 / OS 33.3 →
  0.244·1077=263, 0.333·1077=359, 263/359=73.3%) — the script prints it with the
  derivation; label it "derived" on the slide.

### 11. SPL-among-successes — ✅ (derived, printed by the same script)
- K1: 10.93/18.29 = **0.597**; H1-blind: 21.0/24.4 = **0.861**. Both are ratios of
  aggregate metrics (standard decomposition); label as derived.

### 12. Crop-vs-stretch A/B — ✅ with one wording fix
- Primary per-episode data (both arms, episodes 0–9):
  `eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit{,_crop}/diag/slice_summary.csv`
  (copies in the 2026-07-13 archive). Verified: **stretch 3/10 success (eps 5,7,8),
  OS 4/10; crop 0/10, OS 3/10 (eps 6,7,8), each with min-distance-to-goal = 0.371 m**.
- ⚠️ Wording: the crop arm **did emit stops — but only after overshooting** (final
  distances 3.82/3.66/6.66 m on eps 6/7/8). Say "failed to stop at the goal /
  stopped only after walking past," not "never stopped."
- The two runs differ **only** in `--vlm_transform` (`stretch` vs `crop`), same
  checkpoint/int8/raw/seed episodes — commands recorded in
  `RESULTS_2026-06-11_smoke.md` and `k1-research-workspace-main/scripts/bench_diag.sh`.
- Videos: `..._crop/videos/output_{6,7,8}.mp4` (arrival-blind), stretch successes
  `output_{5,7,8}.mp4`.
- **100-episode/arm rerun**: `TRANSFORM=crop EP_TIMEOUT=900 bash scripts/run_powered_benchmark.sh crop100 checkpoints/model_14498.pt` (and stretch arm likewise), episodes 0–99.
  ⚠️ Cost estimate depends on which recorded rate you trust: the full-run average was
  ~3 min/ep (54 GPU-h / 1077) but the smoke doc recorded 9–12 min/ep → **≈5–20 GPU-h
  per arm, 10–40 total**. Budget the upper bound.

### 13. Renderer study (5090 vs 3090) — ⛔ RAW 5090 ARTIFACTS NOT ON THIS MACHINE
- On-disk primary: `BENCHMARK_32GB_CHECKLIST.md` ("DECISION 2026-06-11": desktop 5090
  SR 0/10, OS 4/10, fp16 vs lab 3090 SR 3/10, clean render; frame residual **15.6 vs
  0.4–1.3**; deterministic 3× walk-out on ep 7) and `RESULTS_2026-06-11_smoke.md`
  (per-episode desktop-vs-3090 table).
- The 5090 run directories/frames live on the 32 GB desktop, not here. **Do not claim
  same-commit reproducibility from this machine** — cite the two documents, or pull the
  desktop dirs before the deck ships.

### 14. int8 arbiter — ✅ numbers, ⚠️ raw outputs partial
- Primary: `REPLAY_ARBITER_2026-06-11.md` — **424 paired decisions; fp16-vs-int8
  action-level agreement 394/424 = 92.9% (drift 7.1%); int8-replay vs int8-original
  361/424 = 85.1% ⇒ JPEG-re-encode sensitivity 14.9%; 100% agreement on all success
  episodes** (per-episode table in the doc).
- The replayed **inputs** are on disk (`.../smoke.../diag/ep0..ep9/` frame payloads) and
  the tool is `k1-research-workspace-main/scripts/replay_vlm_arbiter.py`; the arbiter's
  raw **answer files were not retained**. Regenerating needs the VLM bridge up
  (`run`+`compare` modes, ~1 GPU-h).

---
## SECTION 3 — "not a download" receipts

### 15. Git ledger — ✅ generated, with two loud flags
`receipts/git_ledger.sh` → `receipts/git_ledger_output.txt` (regen: `bash receipts/git_ledger.sh`):
| Repo | User commits | Dates | LOC added | Notes |
|---|---|---|---|---|
| k1_research (`Janga786/k1-navila-research`) | **18/18 Janga786** | 06-10→07-09 | 29,152 | ⚠️ includes the vendored NaVILA-Bench tree in the initial commit — **K1-authored new files inside the benchmark ≈ 3,071 LOC** (see claim 16), plus modified upstream files & workspace scripts |
| booster_train | 🔴 **0 user commits** (6 = Booster upstream) | — | — | **≈6,321 LOC of authored K1/NaVILA work sits UNCOMMITTED in the working tree** (task cfgs, mdp, gate scripts, pipelines, sweep — 20 untracked paths). Commit before the deck ships. |
| booster_deploy | 1 (7 upstream) | user commit 07-02 | — | +14 uncommitted paths (k1_navila r5/r7 models, deploy.sh, arm_scan validators) |
| k1-vlm-navigation | **10/10 user** (as Janga786 + Bliss Janga) | 05-08→07-09 | 7,883 | fully user-authored repo |
- 🔴 Flag for any "LOC authored" slide: separate **authored** (≈3.1k benchmark + ≈6.3k
  booster_train + 7.9k vlm-nav + workspace scripts) from **vendored** (NaVILA-Bench
  upstream, IsaacLab clone, Booster repos).

### 16. K1 integration surface — ✅ (`receipts/` + this list)
New files added to the vendored NaVILA-Bench (absent from the upstream zip):
`config/k1/{__init__,k1_matterport_base_cfg,k1_matterport_vision_cfg}.py` (robot, camera,
gains, terminations) · `utils/wrappers_v3.py` (235-dim obs bridge + joint permutation +
stall guards) · `scripts/navila_eval_v3.py` (episode loop/executor) ·
`scripts/vlm_server_bridge.py` (int8 VLM server) · `scripts/{nav_diag,l0_camera_dump,l2_obs_frame_check,l2_stall_gate_check,l4_control_episode}.py` (gate harnesses).
Modified upstream: `utils/eval_utils.py` (parser fixes), `utils/measures.py`-adjacent
aggregation via `aggregate_k1_vision_results.py` (NE definition fix). Plus patch records:
`PATCH_k1_accuracy_benchmark.patch`, `PATCH_model8700_wrappers_v3.patch`. Training-side and
deploy-side trees: see claim 15 flags.

### 17. Training-run ledger — 🔴 "~9 runs" UNDERCOUNTS
`receipts/run_ledger.md`: **22 substantive runs** (31 run dirs on disk), ≈**270 GPU-hours**,
every run dated with config/iters/outcome/reason. Winning path: 11999 → 14498 → 14200-r7.
Slide options: "22 runs / ~270 GPU-h" or "3 runs on the winning path, 19 documented
dead-ends & diagnostics."

### 18. Media index — ✅ mostly, two ⛔
`receipts/media_index.md`: verified paths for the 8700-circumduction, 14498, 14200-winner,
trot, reward-farming, near-hit, and crop-arrival-blind clips. **⛔ "Round-4 splay" video
does not exist** (round-4 was never rendered and its checkpoints were deleted — use the
r6 FAILED_WIDE clip). **⛔ "Real-robot volleyball clip" not found anywhere on this
machine** — and the project record states the K1 has never walked under this stack;
source it explicitly or cut it.

---
## Summary of items needing slide edits before shipping
1. "20 rewards" → 19 (14498) or 20 (14200) — pick the model. (Claim 1)
2. feet_swing −20 → that's `feet_swing_height`; `feet_swing` is +3.0. (Claim 1)
3. "350/250" → say *stiffness* 350/250 (damping was 7.5/5). (Claim 2)
4. Camera "fixing commit" → cite `.orig_camfix` + patch file; no commit exists. (Claim 4)
5. H1 1.5 m → "~1.5 m (pelvis + 0.5 m, derived)". (Claim 4)
6. 61% → "~61% (recorded)"; raw log not on disk. (Claim 5)
7. 0.76% → recorded baseline; raw run dir not on this machine. (Claim 7)
8. 0.007 m/s → "<0.02 m/s (0.007–0.016 across runs)"; verify 0.066 vs the paper. (Claim 9)
9. Crop arm "never stopped" → "stopped only after overshooting (3.7–6.7 m)". (Claim 12)
10. Renderer raw data is on the desktop machine — cite docs or fetch dirs. (Claim 13)
11. "~9 training runs" → 22 runs / ~270 GPU-h. (Claim 17)
12. Round-4-splay & volleyball clips don't exist here — substitute or cut. (Claim 18)
13. Commit the ≈6.3k LOC of uncommitted booster_train work before showing git receipts. (Claim 15)
