# Overnight setup log (2026-05-18 night → morning)

Append-only record of what the autonomous overnight watcher did and why.
The user is asleep; this is the audit trail to wake up to.

---

## 2026-05-18 21:15  — Training resumed
- Loaded `model_2000.pt` from
  `~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/2026-05-18_10-42-27_k1_vlnce_v2/`
  (this is the run the smoke-test killed yesterday morning at iter 2015,
  reward 56.2).
- New run dir created automatically by train.py:
  `2026-05-18_21-15-33_k1_vlnce_v2_resumed/`.
- rsl_rl's `learn(num_learning_iterations=30000)` adds 30k to the loaded
  iter, so the run goes iter 2000 → 32000. Config left untouched per
  user instruction.
- Tmux session: `k1_vlnce_train`. Conda env: `isaacLab_311`.
- Launcher: `~/Projects/k1_research/scripts/resume_k1_vlnce.sh`.

## 2026-05-18 21:15  — 5-min reward monitor armed
- Monitor: `~/Projects/k1_research/scripts/monitor_k1_vlnce_resume.sh`,
  running in the background under nohup.
- Alerts (written to `/tmp/k1_vlnce_monitor.log`, also tail-able live):
  - `action_rate < -1e6` (collapse, mimicking the legged-loco v3 failure mode)
  - `Train/mean_reward < 0.5 * peak` (regression)
  - tmux session gone (crash or natural completion)
- Auto-stop on 15k-iter stall (kills the tmux session, same threshold as the
  original `watch_k1_vlnce_train.sh`).
- Initial monitor regex for action_rate was wrong (`Mean episode action_rate:`)
  → fixed to `Episode_Reward/action_rate:` because the booster env's
  reward-manager keys contain `/`, which makes rsl_rl skip the
  "Mean episode ..." prefix. Monitor restarted with the corrected pattern.
- Dry-run confirmed: at iter 2064 the script extracted
  `iter=2064 reward=53.57 action_rate=-0.3026`.

## 2026-05-18 22:14  — Status at start of autonomous overnight watch
- Iter: 4500 / 32000  (well past the 2015 the v2 run reached pre-kill)
- Reward: 67.9  (peak was 56.2; resumed run is already better)
- action_rate: -0.29  (healthy; floor for collapse alert is -1e6)
- ETA to iter 32000: ~10h 41m  → user will likely return _before_ training
  finishes; Phase 2/3 most likely do not run tonight.
- Checkpoints already saved in the resumed run dir:
  `model_2000.pt`, `model_4000.pt`.

## 2026-05-18 22:20  — Claude event-monitors armed
Two persistent Monitor tools fire stdout lines into Claude as notifications.
They will re-invoke me overnight when something happens.

1. **Alarm monitor** (`tail -F` of both logs + grep):
   - Patterns: `Traceback (most recent call last)`, `RuntimeError`,
     `CUDA out of memory`, `CUDA error`, `Segmentation fault`, `\bKilled\b`,
     `^\[Error\]`, `^Error:`, `ALERT:`, `MemoryError`, `out of memory`,
     `nan loss`, `Loss is nan`.
   - Fires on training crashes AND on any line emitted by
     `monitor_k1_vlnce_resume.sh` that starts with "ALERT:" (collapse,
     regression, completion, plateau-stop).
2. **Checkpoint heartbeat monitor** (polls run dir every 60s):
   - Emits `CHECKPOINT model_NNNN.pt iter=X reward=Y action_rate=Z eta=W`
     each time a new `model_*.pt` file appears.
   - With save_interval=2000 and ~1.4s/iter, expect ~47 min between events.
   - On each event I append a row to OVERNIGHT_RESULTS.md status timeline.

## 2026-05-18 22:20  — Phase 2/3 auto-runner staged
`~/Projects/k1_research/scripts/auto_phase23_k1_vlnce.sh` is the script the
alarm-completion event will invoke. It:

1. Verifies tmux `k1_vlnce_train` is gone.
2. Verifies no python processes hold GPU memory (waits up to 60s).
3. Picks best checkpoint by max reward across save points (tensorboard).
4. Runs `bench_k1_vlnce.sh smoke` (1 ep, 240s timeout).
5. If a measurement JSON lands, runs `bench_k1_vlnce.sh full 0 1077` in tmux
   `bench_vlnce`.
6. Aggregator (`aggregate_k1_vision_results.py`) is invoked from the morning
   report routine, not auto.

## Decisions deferred to morning (if training is still running)
- Whether to let it finish to iter 32000 or interrupt early.
- Whether `--max_iterations 28000` would have been preferable (would have
  capped at iter 30000 total). Currently the run will end at iter 32000 since
  the config was preserved.
- Best-checkpoint selection: latest may or may not be best; the auto-runner
  reads tensorboard to find argmax(reward) across save points.

## 2026-05-19 07:48:33 — Auto Phase 2/3 invoked
- Phase 2: best_iter=28000 best_reward=73.0234 action_rate_at_best=-0.28123870491981506 saves_seen=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000] 

## 2026-05-19 07:48:41 — Phase 3 — smoke

## 2026-05-19 07:48–07:50 — Phase 3 smoke ran and finished
- VLM bridge launched (navila-llama3-8b-8f via vlm_server_bridge.py),
  came up in <10 s.
- `navila_eval.py` ran episode 0 with `model_28000.pt` under
  `k1_matterport_vision`, exited cleanly after ~98 s. No Python error.
- Measurement: `eval_results/k1_matterport_vision_loco_booster_vlnce_2026-05-18_21-15-33_k1_vlnce_v2_resumed/measurements/0.json`
  →  path_length=0.018 m,  success=0,  distance_to_goal=8.24 m,  oracle_success=0.
- Conclusion: state_dict load OK, but **robot did not walk in eval**.
  (Compare to legged-loco v2 = path_length 1.95 m which was already 0% SR,
  so this is "less broken, still broken".)

## 2026-05-19 07:50 — Full benchmark NOT launched (judgment call)
- The auto-runner does not gate on smoke path_length, so it would have
  proceeded to fire 1077 episodes in tmux `bench_vlnce`. I sent SIGTERM
  to the auto-runner (PID 376050) and the smoke wrapper (PID 376202)
  before the full launch crossed the boundary.
- Rationale: 1077 episodes × ~60 s/ep ≈ 18 h of GPU for 0% SR is not
  worth it given the strong smoke signal that integration is broken.
  User explicit instruction: "If benchmark integration is complex and
  you're running low on time, just get training done and write a
  detailed plan — I'll do the benchmark tomorrow."
- Also killed the VLM bridge (PID 376247, 17.5 GB freed). Confirmed GPU
  back to 273 MiB (X11+firefox only), 0 python compute apps.

## 2026-05-19 07:55 — Morning report written
- OVERNIGHT_RESULTS.md top section now contains the smoke failure
  analysis + a priority-ordered investigation plan.
- Most likely root cause documented: training uses
  `BOOSTER_K1_LOCOMOTION_CFG` with tight effort_limit_sim (30–60 Nm) +
  `DelayedImplicitActuatorCfg` (0–4 step delay) + per-joint armature.
  Eval uses `K1_ARTICULATION_CFG` with loose effort_limit=200 Nm +
  `ImplicitActuatorCfg` (no delay) + no armature. Same regex-based
  action ordering, same obs layout, same action_scale, same default
  joint positions — but the actuator physics gap is wide enough to
  silence the policy.

## 2026-05-19 08:50 — Morning session: fixing actuator mismatch
- User awake; task is to fix the training/eval actuator physics gap and
  re-run smoke (then full bench if smoke walks).
- Read OVERNIGHT_RESULTS.md "Working hypothesis: training-vs-eval
  articulation mismatch" — confirms the priority-1 next step.
- Confirmed source of truth for training actuators:
  `booster/booster_train/source/booster_train/booster_train/assets/robots/booster.py:189`
  → `BOOSTER_K1_LOCOMOTION_CFG`.
  The VLNCE training env_cfg uses `BOOSTER_K1_LOCOMOTION_CFG.replace(prim_path=...)`
  with **no** further actuator overrides (`vlnce_env_cfg.py:211`).
- Read `booster/booster_train/source/booster_train/booster_train/assets/robots/actuator.py`:
  `DelayedImplicitActuatorCfg(ImplicitActuatorCfg)` adds only `min_delay` /
  `max_delay` fields. The runtime class subclasses `ImplicitActuator` and
  applies a `DelayBuffer` to joint setpoints (pos/vel/effort) — pure
  domain randomization. No other physics change.
- Confirmed Isaac Lab 1.1 (NaVILA-Bench) has only `ImplicitActuatorCfg` /
  `DelayedPDActuatorCfg` — no `DelayedImplicitActuatorCfg`. Decision:
  use `ImplicitActuatorCfg` (matches training's underlying physics) and
  document that the 0-20 ms random latency is dropped at eval.

## 2026-05-19 09:00 — Eval config patched
File:
  `NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/k1/k1_matterport_vision_cfg.py`

K1_ARTICULATION_CFG.actuators now matches BOOSTER_K1_LOCOMOTION_CFG exactly:

| Group       | Field            | Old eval value | New eval value (= training) |
|-------------|------------------|----------------|------------------------------|
| hips_knees  | effort_limit     | 200.0 scalar   | {Hip_Pitch:30, Hip_Roll:35, Hip_Yaw:20, Knee_Pitch:60} |
| hips_knees  | velocity_limit   | 20.0 scalar    | {Hip_Pitch:8, Hip_Roll:12.9, Hip_Yaw:18, Knee_Pitch:12.5} |
| hips_knees  | stiffness        | 350.0          | 350.0 (unchanged) |
| hips_knees  | damping          | 10.0           | 7.5 |
| hips_knees  | armature         | absent (None)  | {Hip_Pitch:0.0478125, Hip_Roll:0.0339552, Hip_Yaw:0.0282528, Knee_Pitch:0.095625} |
| ankles      | effort_limit     | 100.0          | 20.0 |
| ankles      | velocity_limit   | 20.0           | 18.0 |
| ankles      | stiffness        | 250.0          | 250.0 (unchanged) |
| ankles      | damping          | 5.0            | 5.0 (unchanged) |
| ankles      | armature         | absent (None)  | 0.0565056 (2 × ARMATURE_4310) |

Actuator class change documented in inline comment: training uses
`DelayedImplicitActuatorCfg` (Isaac Lab 2.3) which is `ImplicitActuator` +
0-4 step (0-20 ms) random delay buffer on setpoints. Isaac Lab 1.1 has no
equivalent; eval uses plain `ImplicitActuatorCfg`. The dropped delay is
a domain-randomization feature, not a requirement — the policy was
trained to be robust to latency, not to require it.

## 2026-05-19 09:06 — Pre-fix smoke artifacts preserved
- Renamed:
  `eval_results/.../measurements/0.json` → `0.json.prefix_actuator_v1`
  `/tmp/k1_vlnce_smoke.log` → `/tmp/k1_vlnce_smoke.log.prefix_actuator_v1`
- GPU clean: 273 MiB used, 31829 MiB free.


## 2026-05-19 09:10 — Smoke result with actuator fix
- /tmp/k1_vlnce_smoke.log + measurements/0.json show:
  - path_length = 0.057 m (was 0.018 m pre-fix; 3x better but still ~0)
  - distance_to_goal = 8.29 m, success = 0, oracle_success = 0
  - bad_orientation = False, time_out = True
- Diagnosis: K1 is stable (doesn't fall over) but doesn't walk toward goal.
  Actuator physics gap closed; remaining gap is elsewhere.
- Did NOT launch full benchmark (smoke gate path_length > 1m unmet).

## 2026-05-19 09:30 — MuJoCo sim-to-sim policy diagnostic
- Wrote `scripts/mujoco_test_vlnce_policy.py`: loads model_28000.pt's
  ActorCritic state_dict into a fresh nn.Sequential (matching the rsl_rl
  3.x ActorCritic actor: [Linear(235→512), ELU, Linear(512→256), ELU,
  Linear(256→128), ELU, Linear(128→12)]). Sends commanded velocity
  schedule (vx=0.5 5s → vyaw=0.524 3s → vx=0.5 5s → 0 2s = 15s) and
  records 50fps mp4 via mujoco.Renderer (EGL) → ffmpeg.
- MJCF: `booster_assets/robots/K1/K1_22dof.xml` (22-DoF; the locomotion
  URDF would have arms+head merged into Trunk but the MJCF doesn't
  exist with that merge). Arms/head held rigidly with kp=500/kd=15 PD.
- Critical joint-order discovery: action_cfg has `preserve_order=False`,
  so Isaac Lab sorts the regex-matched joints by **articulation index**
  (URDF declaration order), NOT by the regex iteration order. For
  K1_locomotion the order is therefore left chain first then right chain:
  [LHP, LHR, LHY, LK, LAP, LAR, RHP, RHR, RHY, RK, RAP, RAR].
  Earlier booster_deploy `K1_VELOCITY_POLICY_JOINTS` uses the interleaved
  L/R order — likely valid for a different task/Isaac Lab version, NOT
  for this VLN-CE checkpoint.
- Result: K1 fell at t=0.82s with rigid arms, t=0.88s with flexible arms.
  Forward displacement during first 5s = 0.27-0.34 m before collapse;
  mean vx_body ≈ 0.06 m/s (vs commanded 0.50). Yaw spin ~-140° during
  the brief upright phase. Tilt > 1.3 rad for 711/751 steps.
- Action probe at the standing-pose obs (with vx command varied):
  - L_HipP raw=+1.83, L_HipR=+0.86, L_HipY=+1.02, L_Knee=+1.78
  - R_HipP=-3.15, R_HipR=-3.41, R_HipY=-3.70, R_Knee=-1.93
  Massively asymmetric and outside joint ranges (R_Knee target -0.66
  is below joint limit 0; L_Ankle_Roll target +1.71 is above joint
  limit 0.345). Action varies barely with command magnitude (~0.05
  raw difference between vx=0 and vx=1).
- Conclusion: the trained policy itself appears degenerate; the
  "reward 73.72 peak" in training did NOT correspond to a stable
  walking gait. It corresponds to a brief forward lunge before the
  bad_orientation termination kicks in (which would still accumulate
  ~70 cumulative episode reward at 0.07/step over ~1000 steps if the
  env auto-resets repeatedly while the policy briefly tracks command
  forward velocity each fresh episode).
- Video saved: `/home/janga/Desktop/k1_vlnce_policy_test.mp4` (15s,
  640x480, 50fps, 1.2 MB).
- This invalidates the "fix the eval-time actuator mismatch and you're
  done" hypothesis. The actuator fix helped (path_length 0.018→0.057)
  but the policy itself can't walk. Need to retrain (likely with the
  termination + reward issues addressed) or try a smaller-iter
  checkpoint (e.g., model_4000.pt before the policy collapsed into the
  bad lunge-and-fall behavior).


## 2026-05-19 09:46-14:23 — K1 VLN-CE v3 (start-from-what-works) training
**Verdict: SUCCESS.** Peak reward 111.57, final 105 over 30000 iters.

### Config
- New task `Booster-K1-Velocity-Vision-VLNCE-v3` registered in
  `booster_train/.../velocity/robots/k1/__init__.py`.
- `vlnce_v3_env_cfg.py`: inherits `LocomotionVelocityFlatEnvCfg` (the
  WORKING reward-103 base) and adds only:
  * Separate `vision` obs group with 160-dim height_scan (16x10 grid)
  * `height_scanner` RayCaster on Trunk, casting against /World/ground
  * `bad_orientation` relaxed 1.0 → 1.3 rad
  * K1 robot (BOOSTER_K1_LOCOMOTION_CFG)
- `vlnce_v3_ppo_cfg.py`: inherits `VelocityPPORunnerCfg` verbatim, only
  changes experiment_name + max_iterations + save_interval. Keeps
  `empirical_normalization=True` (v2's biggest bug fix).

### Two-phase training
1. **09:46-10:36 (50 min)** `train_k1_vlnce_v3.sh` → 5000 iter cap.
   Iter 1400 gate passed at reward 92, ep_len 1425 (gate was reward 50,
   ep_len 500). Run dir `2026-05-19_09-46-39_k1_vlnce_v3/`. Final iter
   4999 @ reward ~100, ep_len ~1451.
2. **10:38-14:23 (3h45m)** `resume_k1_vlnce_v3.sh` → +25000 iters
   (total 30000). Run dir `2026-05-19_10-38-52_k1_vlnce_v3_resumed/`.
   Reward trajectory: 103 (iter 5227) → 108 (iter 11998) → 111.57 peak
   (iter 16348) → 102-105 stable through iter 29998.

### Final state
- Total iters: 30000
- Peak reward: 111.57 @ iter 16348 (best save-point ckpt: `model_16000.pt`)
- Final reward: ~105 @ iter 29998 (final ckpt: `model_29998.pt`)
- Final ep_len: ~1498 / 1500 max (99% survival rate)
- Final terminations: bad_orientation 0.29%, base_below 1.27%,
  time_out ~99% (essentially "walked the full episode")
- 51 checkpoints saved in `..._k1_vlnce_v3_resumed/`

### Key delta from v2 (which collapsed at reward 73)
The v2 vlnce_env_cfg.py redesigned the policy obs to match the
NaVILA-Bench wrapper's 235-dim single-step layout, dropping gait_phase,
5-frame history, action_scale 0.25, empirical_normalization, and flat
terrain. v3 keeps all of those (they're what makes the policy walk) and
adds height_scan as a SEPARATE obs group that the actor/critic don't see.

### What's NOT done
The v3 policy can't yet plug into the NaVILA benchmark — the bench
wrapper expects v2's single-step obs layout. Adapter work needed:
either extend `VLNEnvWrapper` to build the per-step + history obs the
v3 actor needs, or write a parallel `k1_matterport_vision_v3` task in
NaVILA-Bench. Documented in the memory.


## 2026-05-19 16:00-17:00 — v3 MuJoCo + bench wrapper

### Task 1: K1_locomotion.xml MJCF with welded arms
- New file: `booster/booster_assets/robots/K1/K1_locomotion.xml`
- Strategy: removed the 10 arm+head `<joint>` elements from K1_22dof.xml,
  keeping the body hierarchy. MuJoCo treats jointless child bodies as
  rigid extensions of their parent → mass/inertia is absorbed into the
  parent's effective articulation inertia automatically — exactly what
  `merge_fixed_joints=True` does in Isaac Lab URDF loading. Mass conserved
  (19.666 kg, same as K1_22dof.xml). nq=19, nv=18, nu=12 (only legs).
- Also bumped ground friction 0.4 → 1.0 in this MJCF to match training.

### Task 1b: v3 in welded MuJoCo
- Updated `scripts/mujoco_test_v3_policy.py`:
  * default MJCF path → K1_locomotion.xml
  * joint-lookup is now tolerant of missing arm joints (the arm/head
    PD inner loop becomes a no-op when joints don't exist)
  * `--gait_phase_init` flag added (training had random per-env offsets;
    MuJoCo physics is sensitive to phase init)
- Swept gait_phase_init 0.0 → 0.95 in 0.05 increments. Result:

  | phase | fall time | distance |
  |-------|-----------|----------|
  | 0.00  | 1.06 s    | 0.69 m   |
  | 0.50  | 1.54 s    | 0.62 m   |
  | 0.65  | 6.40 s    | 0.99 m   |
  | 0.68  | 7.12 s    | 1.54 m   |
  | **0.70**  | **8.32 s**    | **1.47 m**   |
  | 0.72  | 6.24 s    | 1.27 m   |
  | 0.73  | 8.16 s    | 0.93 m   |
  | 0.75  | 3.74 s    | 0.63 m   |
  | 0.80  | 0.72 s    | 1.13 m   |

- **Best: phase=0.70 → 8.32 s walking, 1.61 m traveled in 20 s sim.**
  Saved canonical video to `~/Desktop/k1_v3_locomotion_mujoco.mp4`.
  12× longer than the previous v3 run with articulated arms (0.70 s).
- Remaining gap to "10+ s stable": PhysX (Isaac Sim) vs Newton solver
  (MuJoCo) contact-model differences. Not closeable without retraining
  with MuJoCo-side domain randomization.

### Task 2: Benchmark wrapper adapter for v3
- New file `NaVILA-Bench/.../utils/wrappers_v3.py` with `VLNEnvWrapperV3`:
  * Constructs the v3 47-dim per-step obs from raw scene state
    (`scene["robot"].data.joint_pos/joint_vel/root_quat_w/root_ang_vel_b`).
  * Maintains an internal (N, 5, 47) history buffer.
  * Term-major flattens to 235 dim before feeding the policy.
  * Drives a gait_phase clock at 2 Hz, advancing by `policy_dt` each step.
  * Pre-divides actions by 2 so the bench env's existing `action_scale=0.5`
    yields effective 0.25 (matches v3 training).
  * Ignores the env's "policy" obs group — env is used only for physics,
    Matterport scene, cameras, and measurements.
- New file `NaVILA-Bench/scripts/navila_eval_v3.py`:
  * Loads the v3 actor state-dict directly into a fresh nn.Sequential
    (235→512→256→128→12, ELU) — bypasses rsl_rl OnPolicyRunner so we
    don't need to push the v3 agent.yaml into NaVILA-Bench.
  * Uses VLNEnvWrapperV3.
  * Same VLM image loop, R2R episode handling, measurements, video
    output as `navila_eval.py`.
  * Required args: `--checkpoint=<absolute path>`, `--episode_idx=N`,
    `--gait_phase_init=X`.
- New file `scripts/bench_k1_vlnce_v3.sh`:
  * `smoke [ckpt] [episode] [phase]` → 1 episode (300 s timeout)
  * `full  [ckpt] [start] [end] [phase]` → loops episodes in tmux
  * Auto-launches VLM bridge on port 54321 if not running
  * Default checkpoint: `model_16000.pt` (peak reward 111.57)


### Task 3: smoke + full bench launched
- 17:00 — first smoke attempt failed: argparse `--checkpoint` conflict
  between `parser.add_argument` and `cli_args.add_rsl_rl_args`. Fixed by
  removing my own --checkpoint declaration and using the rsl_rl one;
  added a manual `if not args_cli.checkpoint` guard.
- 17:05 — second smoke (episode 0, model_16000.pt, gait_phase_init=0.0):
  **path_length = 7.81 m**, distance_to_goal = 8.13 m, success = 0,
  oracle_navigation_error = 7.62 m. **First time path_length crossed
  1 m in this benchmark** — the wrapper + policy integration works.
  Measurement: `NaVILA-Bench/eval_results/k1_matterport_vision_loco_v3_model_16000/measurements/0.json`
  Video: same dir, `videos/output_0.mp4` (50 MB).
- 17:08 — full benchmark launched in tmux `bench_v3` on episodes [0, 1077).
  ETA ~28 h. Per-episode timeout 300 s. Driver `/tmp/k1_v3_bench_driver.sh`
  loops episodes calling navila_eval_v3.py one at a time. Measurements
  accumulate in `eval_results/k1_matterport_vision_loco_v3_model_16000/measurements/`.
  Log `/tmp/k1_v3_bench.log`.

### Stack summary (what's running right now)
- VLM bridge: pid 699746, port 54321, ~17 GB GPU
- Full bench: tmux `bench_v3`, Isaac Sim per-episode, ~6 GB additional GPU
- Total GPU: ~23 GB / 32 GB → headroom for the rest of the session

