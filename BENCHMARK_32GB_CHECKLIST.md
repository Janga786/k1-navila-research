# 32 GB Machine — Benchmark Checklist (2026-06-10)

Everything below is the ONLY remaining benchmark work. All code fixes are in this
repo (15 audit bugs + stall-abort v2); L0–L4 are validated on the lab 3090 —
see memory/session notes. fp16 NaVILA fits 32 GB: **no quantization needed.**

## 1. Get the code + heavy pieces

```
git clone https://github.com/Janga786/k1-navila-research.git
```

Gitignored pieces (the desktop should already have them from the original setup —
point/symlink rather than re-download):
- NaVILA weights → `booster/NaVILA/checkpoints/navila-llama3-8b-8f` (HF: a8cheng/navila-llama3-8b-8f)
- Matterport USD + episodes → `NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/assets/`
  (HF dataset: Zhaojing/VLN-CE-Isaac; needs `matterport_usd/` + `vln_ce_isaac_v1.json.gz`)
- Forks with their own git: `IsaacLab/` (yang-zj1026 fork, Isaac 4.1 / Lab 1.1) and
  `experiments/navila` (AnjieCheng/NaVILA llava)
- Conda envs: `vlnce-isaac` (isaacsim 4.1.0.0 + Lab 1.1 + rsl_rl 2.0.2 + torch 2.2.2) and
  `navila`→navila-vila (torch 2.3 + VILA llava). Known dep pins are in the
  navila-bench-integration memory if rebuild is needed.
- Symlinks: `NaVILA-Bench -> NaVILA-Bench-main`; `booster/booster_assets/robots/K1 ->`
  the booster_assets K1 dir (K1_locomotion.urdf).

⚠️ Do NOT re-run any `PATCH_*.patch` — all patches are pre-applied in the repo files;
re-applying will conflict.

## 2. Copy the policy (gitignored, from the lab box)

```
scp boosterk1@<lab-box>:~/robots/k1/workspace/booster_train/logs/rsl_rl/k1_navila/2026-06-09_13-52-48/model_11999.pt .
# exported TorchScript/ONNX live next to it in exported/ (only model_11999.pt is
# needed for the benchmark: navila_eval_v3 build_v3_actor loads the raw checkpoint)
```

model_11999 = the new production policy (circumduction fixed, full tracking gate PASS,
MuJoCo rated-torque gate PASS). It replaces model_8700 — same 235/12 contract.

## 3. Smoke (10 episodes) — audit steps 17/18 in full VLM form

```
bash scripts/bench_diag.sh 0 10 smoke_11999 /abs/path/model_11999.pt \
  "--closed_loop --clean_render --bright --max_episode_s 120 --vlm_transform stretch"
```

Watch for:
- `[closed-loop] STALL:` lines — the new 3 s stall→re-query firing on jams (expected
  occasionally; episodes should CONTINUE after)
- `[v3-wrapper] hopeless-stall:` — should be RARE (only truly wedged episodes)
- `[PARSE_FAIL]` — should be rare; every line is a VLM output worth eyeballing
- 90° turn commands executing the FULL angle (diag JSON: target_heading vs achieved)
- no socket hangs on VLM hiccups (120 s timeout + clean episode failure instead)
- forward-75 landing ≈ 0.70–0.75 m

Keep `--stop_assist` OFF (prior A/F sweep measured it as a regression).

## 4. Optional stall A/B (10–20 eps)

The executor stall is always-on. For a controlled A/B, temporarily set
`stall_window=0` in ClosedLoopController's ctor defaults (navila_eval_v3.py) for the
B arm. Compare SR/OS + mean episode wall-time. (Skippable: mechanism already
validated; the A/B only quantifies the SR gain.)

## 5. Full run (1,077 episodes, resumable)

```
bash scripts/run_powered_benchmark.sh full_11999 /abs/path/model_11999.pt
```
Crashed episodes are retried on re-invoke; missing episodes score as failures over
the full 1,077 denominator (paper-honest).

## 6. Aggregate + corrected historical NE (audit step 19)

```
python aggregate_k1_vision_results.py eval_results/k1_matterport_vision_loco_full_11999
```
NE column is now FINAL distance_to_goal (paper-comparable vs H1 5.86 / Go2 5.49).
Also re-run the aggregator on the OLD runs' measurement dirs to get corrected
historical NE — no re-simulation needed.

## Reference points
- Old stack (v3 model_16000 era): SR 0.76% / OS ~10%; OS ceiling on old policy ~25%.
- Paper blind-H1 (same VLM): SR ~24% / OS ~33% — the realistic ceiling-zone for a
  blind-policy K1.
- What changed since: model_11999 (clean gait + full tracking gate), 90°-turn parser
  fix, 3 s stall→re-query, VLM socket robustness, BoosterMipi camera geometry,
  proven PD gains/torque limits, NE metric fix.
