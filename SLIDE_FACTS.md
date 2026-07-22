# SLIDE_FACTS — verified numbers only (2026-07-22)
Every number below is verified in EVIDENCE.md (claim # in brackets) with primary source
and regeneration command. Phrasings marked ⚠ are the evidence-safe wordings — use them.

## Headline benchmark (VLN-CE-Isaac, R2R val-unseen, n = 1,077, 0 missing) [8]
- **SR 18.29%** (197/1077) · **OS 30.27%** (326/1077) · **NE 7.59 m** · **SPL 10.93%**
- Baseline: **SR 0.76%** (prior stack, v3 model_16000) ⚠ "recorded baseline" — raw run
  dir not on this machine [7]. Improvement: **~24×**.
- Failure decomposition: **751 never-reached (69.7%) / 129 reached-no-stop (12.0%) /
  197 success (18.3%)**; stop-given-reached **60.4%** (H1-blind ⚠ *derived* 73.3%) [10]
- SPL-among-successes: K1 **0.597** vs H1-blind **0.861** ⚠ derived ratios [11]

## Locomotion policy (model_14498 run) [1,2,3,9]
- **19 reward terms** (⚠ the final deployed model_14200 recipe = 20, adding the
  penalty-form set-distance band; pick ONE model per slide)
- `feet_swing` (contact schedule) **+3.0** · `feet_swing_height` (clearance) **−20**
  (⚠ two different terms) · velocity tracking **1.5** per axis · termination **−200**
- PD gains **100/2 hips-knees, 50/1 ankles** (prior broken config: ⚠ *stiffness*
  350/250, damping 7.5/5)
- Obs contract **235 = 47 × 5** (term-major), verified across training/benchmark/deploy;
  byte-level agreement **≤1.5e-6** (measured 1.431e-06, fresh 2026-07-19 log)
- Velocity tracking: ⚠ **steady-state error < 0.02 m/s** (measured 0.007–0.016 across
  eval runs) vs the NaVILA paper's 0.066 Go2 reference (⚠ external citation — verify
  against arXiv 2412.04453 before shipping)

## Training effort [17]
- **22 substantive runs, ≈270 GPU-hours** (31 run dirs on disk); **3 runs on the winning
  path** (11999 → 14498 → 14200) — the rest are documented dead ends & diagnostics

## Controlled studies [12,13]
- Crop-vs-stretch A/B (10 episodes, only `--vlm_transform` differs): **stretch 3/10
  success vs crop 0/10**; crop reached **0.371 m** minimum approach on eps 6/7/8 and
  ⚠ **stopped only after overshooting (final 3.7–6.7 m)** — not "never stopped"
  (100-episode/arm rerun IN PROGRESS since 07-22, see RUN_STATUS.md)
- Renderer study: same code/episodes, **SR 0/10 (RTX 5090, broken render) vs 3/10 =
  30% (RTX 3090)**; frame residual 15.6 vs 0.4–1.3 ⚠ recorded in project docs; raw 5090
  artifacts live on the desktop machine [13]
- int8 arbiter: **424 replayed decisions, 7.1% action drift < 14.9% JPEG-re-encode
  sensitivity, 100% agreement on success episodes** [14]

## Camera fix (the embodiment slide) [4]
- Broken: **512×512, 54°, mounted 0.85 m** above trunk — ⚠ literally H1's camera block
  left unadapted (identical lines in the H1 config)
- Fixed: **1280×720, HFOV ≈ 90° (aperture 47.71 from real fx 643.9), 0.25 m above trunk
  → 0.78 m eye height**; H1 comparison: ⚠ "~1.5 m (pelvis + 0.5 m, derived: 1.05 + 0.5)"
- Early-termination bug: 0.8 rad orientation limit ⚠ "~61% of episodes killed (recorded)"
  → fixed to 1.3 rad [5]
- Parser bug: upstream buckets {15,30,45}° only; **any other angle executed as 15°**
  (90° ⇒ 75° undershoot) — upstream source verbatim in the pristine zip [6]
