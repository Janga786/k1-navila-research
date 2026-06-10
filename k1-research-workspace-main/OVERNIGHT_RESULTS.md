# Overnight K1 Vision Benchmark — Morning Summary

**Run started:** 2026-05-15 23:17  
**Hardware:** RTX 5090 (sm_120), Linux 6.17

> Read me first. Detailed timeline + commands below.

---

## Tonight's session (2026-05-18 21:15 → 2026-05-19 ~07:50) — MORNING SUMMARY

**Bottom line:**
1. **Training succeeded** — 27,092 new iters (2000 → 29,092), peak reward
   **73.72 @ iter 14056**, plateau-stop fired at 07:47 (15k iters with no
   improvement > 0.5 since the peak). Best save-point checkpoint:
   **`model_28000.pt`** (reward 73.02, action_rate -0.281 at save).
2. **Phase 2** auto-runner picked `model_28000.pt` and **Phase 3 smoke** ran
   episode 0 successfully — state_dict loads cleanly, measurement JSON
   landed, no Python errors.
3. **But the smoke result is bad:** episode 0 path_length = **0.018 m**,
   success = 0, distance_to_goal = 8.24 m. The robot **did not walk** in
   eval despite training rewards north of 70 (and the v2 iter-2000 policy
   was even worse at path_length 0.003 m, so this is "less broken, still
   broken").
4. **Therefore I did NOT launch the full 1077-episode benchmark** —
   running 18 h of GPU time for an integration-broken policy is wasteful.
   The auto-runner was killed at the "Phase 3 launch full" boundary.
5. **GPU is clean.** All processes (train.py, monitor, VLM bridge,
   navila_eval) are dead. Ready for debugging in the morning.

### Resume run final state
- Run dir: `~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/2026-05-18_21-15-33_k1_vlnce_v2_resumed/`
- Last iter: 29092 (auto-stop) — `model_28000.pt` is the highest saved.
- Total training time: ~10h 32m (21:15 → 07:47).
- All 14 checkpoints from `model_2000.pt` through `model_28000.pt` present.

### What the smoke test produced
- `eval_results/k1_matterport_vision_loco_booster_vlnce_2026-05-18_21-15-33_k1_vlnce_v2_resumed/measurements/0.json`:
  ```
  path_length: 0.018,  distance_to_goal: 8.24,  success: 0
  oracle_navigation_error: 8.24,  oracle_success: 0
  ```
- `/tmp/k1_vlnce_smoke.log` — full Isaac Sim log of the smoke run.
- `/tmp/k1_auto_phase23.log` — Phase 2/3 driver log.

### Working hypothesis: training-vs-eval articulation mismatch
The training and eval envs **disagree on actuator physics** even though
they share obs/action layouts. Most likely root cause of the no-walk:

| Field            | Training (BOOSTER_K1_LOCOMOTION_CFG) | Eval (K1_ARTICULATION_CFG) |
|------------------|--------------------------------------|----------------------------|
| effort_limit     | 30 / 35 / 20 / 60 Nm per joint type  | 200 Nm (hips/knees), 100 (ankles) |
| damping          | 7.5 (hips/knees), 5 (ankles)         | 10 (hips/knees), 5 (ankles) |
| velocity_limit   | 8 / 12.9 / 18 / 12.5 per joint type  | 20 for all |
| actuator class   | `DelayedImplicitActuatorCfg` (0–4 step delay) | `ImplicitActuatorCfg` (no delay) |
| armature         | Per-joint specific (4310, 4315, 6408, 6416) | None |

The training policy learned to walk under low effort limits, with action
latency and per-joint armature. In the eval env, those torques are
4–10× higher and there's no latency — the same actions probably saturate
or oscillate. Combined with `use_default_offset=True`, a policy whose
output is dominated by small deltas around defaults can end up commanding
"basically stand" → path_length ~0.

Other things I ruled OUT:
- **Obs dim**: trained input layer is 235 (verified by inspecting
  `model_28000.pt`'s `actor.0.weight`), eval emits 235-dim policy obs.
  Match.
- **Obs term ordering**: same 8 terms in same order on both sides.
- **Action dim**: 12 on both sides.
- **action_scale**: 0.5 on both sides.
- **use_default_offset**: True on both sides.
- **Default joint positions**: same dict (`.*_Hip_Pitch: -0.15`,
  `.*_Hip_Roll: 0.0`, etc.) on both sides.
- **Stiffness**: 350 (hips/knees), 250 (ankles) on both sides.
- **State_dict load**: `ActorCritic.__init__` only emits a warning about
  ignoring `history_length`, no errors. Policy is functionally loaded.

### Things to try next (priority order)
1. **Align actuator physics in eval to training**: in
   `NaVILA-Bench/isaaclab_exts/.../k1_matterport_vision_cfg.py`, change
   the actuators to match `BOOSTER_K1_LOCOMOTION_CFG` (effort_limit,
   velocity_limit, damping, armature). Re-run smoke. If path_length
   jumps to >1 m, we found it.
2. **Or align training to eval** by training a new policy with the eval's
   actuator physics. Slower but bigger win for fairness vs Go2/H1
   baselines (which were trained in their respective envs).
3. **Disable actuator delay for an eval-time experiment**: temporarily set
   `DelayedImplicitActuatorCfg(min_delay=0, max_delay=0)` in training and
   re-train a short policy to test if delay is the dominant factor.
4. **Check `velocity_commands` source at eval time**: the obs term reads
   `mdp.generated_commands(command_name="base_velocity")`. In training
   this comes from `UniformVelocityCommand`; in eval, NaVILA's
   "move forward 75 cm" needs to be translated to (vx, vy, wz). Verify
   it actually is being injected and not left at zero. Look at
   `VLNEnvWrapper` in `NaVILA-Bench/scripts/`.
5. **Quick sanity test**: in `navila_eval.py`, hard-code
   `velocity_commands = (1.0, 0.0, 0.0)` (1 m/s forward) for one episode.
   If robot now walks straight, the issue is the NaVILA → velocity
   translation, not the policy.

### Reproducible commands
```bash
# Re-run smoke (same args as the auto-runner):
bash ~/Projects/k1_research/scripts/bench_k1_vlnce.sh smoke \
     2026-05-18_21-15-33_k1_vlnce_v2_resumed

# Aggregate any measurements that did land:
python ~/Projects/k1_research/aggregate_k1_vision_results.py \
       --measurements-dir ~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_booster_vlnce_2026-05-18_21-15-33_k1_vlnce_v2_resumed/measurements

# If you want to launch the full bench anyway (gives 1077 episodes of
# 0% SR if the integration is still broken, but documents it cleanly):
bash ~/Projects/k1_research/scripts/bench_k1_vlnce.sh full \
     2026-05-18_21-15-33_k1_vlnce_v2_resumed 0 1077
```

### Benchmark table (placeholder until integration is fixed)
| Robot | Policy | NE↓ | OS↑ | SR↑ | SPL↑ |
|-------|--------|-----|-----|-----|------|
| Go2 (paper) | Blind | 6.03 | 49.0 | 36.2 | 33.3 |
| Go2 (paper) | Vision | 5.49 | 58.7 | 50.2 | 45.5 |
| H1 (paper) | Blind | 7.67 | 33.3 | 24.4 | 21.0 |
| H1 (paper) | Vision | 5.86 | 54.6 | 45.3 | 40.3 |
| **K1 (ours)** | **booster_train v2 resumed, model_28000.pt, smoke ep 0** | 8.24 | 0 | 0 | 0 |

(Single-episode smoke only — not a full benchmark. The 0/0/0 row is
diagnostic, not a defensible reported number.)

### Files written tonight
- `~/Projects/k1_research/scripts/resume_k1_vlnce.sh` — resume launcher.
- `~/Projects/k1_research/scripts/monitor_k1_vlnce_resume.sh` — 5-min
  reward/action_rate/plateau monitor that auto-stopped the training.
- `~/Projects/k1_research/scripts/auto_phase23_k1_vlnce.sh` — driver that
  picked best ckpt → ran smoke → would have run full bench (I killed it).
- `~/Projects/k1_research/SETUP_LOG.md` — append-only audit trail.

### Live state during the night
- Training log: `tail -F /tmp/k1_booster_train.log`
- Monitor log: `tail -F /tmp/k1_vlnce_monitor.log` (5-min cadence)
- Setup log:   `~/Projects/k1_research/SETUP_LOG.md` (what Claude did + when)
- Run dir:     `~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce/2026-05-18_21-15-33_k1_vlnce_v2_resumed/`
- Tmux session: `k1_vlnce_train` (training), monitor PID written in SETUP_LOG.md

### Status timeline (updated as the night progresses)
| Time | Iter | Reward | Action_rate | Notes |
|------|------|--------|-------------|-------|
| 21:15 | 2000 | (loaded) | — | resumed from v2 model_2000.pt |
| 21:17 | 2067 | 54.0 | -0.30 | rewbuffer refilled, near v2 peak (56.2) |
| 22:13 | 4500 | 67.86 | -0.29 | already past v2 peak |
| 22:49 | 5999 | 66.22 | -0.28 | model_6000.pt saved, ETA 10:07 |
| 23:36 | 8014 | 68.17 | -0.30 | model_8000.pt saved, back near peak (68.40), ETA 09:19 |
| 00:23 | 10033 | 69.26 | -0.29 | model_10000.pt saved, NEW PEAK 69.26, ETA 08:31 |
| 01:09 | 12005 | 70.63 | -0.30 | model_12000.pt saved, NEW PEAK 70.63, ETA 07:45 |
| 01:56 | 14024 | 71.50 | -0.29 | model_14000.pt saved, NEW PEAK 71.50, ETA 06:58 |
| 02:43 | 16039 | 68.94 | -0.29 | model_16000.pt saved, slight pullback (-2.56 from peak), ETA 06:11 |
| 03:29 | 18015 | 67.90 | -0.29 | model_18000.pt saved, flat-ish (-3.6 from peak), stall=3991 iters, ETA 05:25 |
| 04:16 | 20034 | 71.03 | -0.30 | model_20000.pt saved, recovered (≈peak 71.50), 2/3 through, ETA 04:38 |
| 05:02 | 22010 | 69.62 | -0.29 | model_22000.pt saved, oscillating in 67-71 range, ETA 03:52 |
| 05:49 | 24030 | 68.98 | -0.29 | model_24000.pt saved, ~3/4 through, ETA 03:05 |
| 06:35 | 26003 | 72.34 | -0.30 | model_26000.pt saved, NEW PEAK 72.34, policy still improving late, ETA 02:19 |
| 07:22 | 28026 | 72.50 | -0.27 | model_28000.pt saved, NEW PEAK 72.50, ETA 01:32 (~90 min to finish) |
| 07:47 | 29092 | 67.37 | -0.27 | **15k-iter plateau auto-stop fired**; true peak was 73.72 @ iter 14056 |
| 07:48 | — | — | — | Phase 2 picked `model_28000.pt` (reward 73.02, action_rate -0.281 at save) |
| 07:48 | — | — | — | Phase 3 smoke launched (VLM bridge up, episode 0 running with model_28000.pt) |

### Pre-flight for Phase 2/3 (done now, in case I lose context overnight)
- Benchmark wrapper: `~/Projects/k1_research/scripts/bench_k1_vlnce.sh`
  - `bench_k1_vlnce.sh smoke <run_subdir>` — 1 episode, validates state_dict load
  - `bench_k1_vlnce.sh full  <run_subdir>` — 1077 episodes in tmux `bench_vlnce`
  - Auto-symlinks `<booster_run_dir>/model_*.pt` and writes a sanitized
    `params/agent.yaml` (strips rsl_rl 3.1.2 fields that 2.0.2 rejects).
  - Launches VLM bridge (`scripts/vlm_server_bridge.py` in `navila` env) if not
    already running.
- Aggregator: `python ~/Projects/k1_research/aggregate_k1_vision_results.py`
- For our run, `<run_subdir>` is `2026-05-18_21-15-33_k1_vlnce_v2_resumed`.
- Auto-runner stage 0: `~/Projects/k1_research/scripts/auto_phase23_k1_vlnce.sh`
  fires when the alarm monitor sees the "training tmux gone" ALERT;
  it verifies GPU is free, picks the best ckpt, runs smoke, then full bench.

### Earlier-run smoke result (for context)
The previous-day v2 run's iter-2000 checkpoint was already smoke-tested
(7 episode files at `eval_results/k1_matterport_vision_loco_booster_vlnce_2026-05-18_10-42-27_k1_vlnce_v2/measurements/`)
and the robot did NOT walk — path_length ≈ 0.003 m on each episode. That's
why we are resuming and training much longer: iter 2000 is too early for the
policy to navigate. The resumed run is already at iter ~4700 with reward 68
(vs 56.2 pre-stop), so by iter ~32k we expect a noticeably better policy.

### Decision tree the watcher is following
1. **Training healthy** → do nothing, log heartbeat on each checkpoint save.
2. **Training crashes** (Traceback / CUDA / Killed / OOM / Segfault)
   → diagnose from log tail, if OOM restart from latest ckpt with
   `num_envs=2048`, log everything.
3. **action_rate < -1e6** → stop training, use best checkpoint, jump to Phase 2.
4. **Reward drops >50% from peak** → stop, use best checkpoint, jump to Phase 2.
5. **15k-iter stall** → already armed in monitor, auto-stops, jump to Phase 2.
6. **Training reaches iter 32000** → process dies cleanly, jump to Phase 2.

Phase 2/3 auto-execute only after the training process is fully dead AND
`nvidia-smi` shows no python processes. The monitor verifies this before
touching the GPU.

---

## TL;DR (previous session, updated 2026-05-18 10:14)

1. **legged-loco v3 (50 k iter) collapsed.** Ran 2026-05-16 → 2026-05-18, iter
   49,999 reached, but `action_rate_l2` diverged from -0.04 → **-3.085e+15**
   starting at iter ~44,910; mean reward fell to -12…-21. Not benchmarked.
   Root cause: `action_rate_l2.weight = -0.005` in legged-loco's
   `k1_low_vision_cfg.py:619` vs `-1.0` in booster_train — 200× weaker penalty
   on unclipped post-tanh actions → late-stage oscillation, then divergence.
2. **legged-loco v2 (9 k iter, Fix-v1 eval) is still the best legged-loco
   checkpoint** but tops out at SR=0%, path_length 1.95 m.
3. **Pivot:** training a fresh policy in **booster_train** (Isaac Lab 2.3 /
   Isaac Sim 5.0, env `isaacLab_311`) with benchmark-matched obs (235 dim:
   proprio + 187-ray height_scan), action_scale 0.5, bad_orientation @ 1.3 rad,
   rough-terrain curriculum, and the booster_train stable PPO settings
   (`action_rate_l2 = -1.0` — the diff that prevents v3-style collapse).
4. **State_dict cross-load verified.** rsl_rl 3.1.2 (train) and 2.0.2 (eval)
   produce identical ActorCritic state_dict keys; drop-in load works, no ONNX,
   no `navila_eval.py` edits.
5. **Training v1 launched 10:07 but dead-on-arrival** — `Mean episode length`
   stuck at **1.00** for 1000 iters, `bad_orientation` termination 80%.
   `ROUGH_TERRAINS_CFG` (Isaac Lab preset) includes 0.23 m stairs and 0.20 m
   boxes, too steep for K1's 0.55-m frame; random spawn landed K1 with one
   foot on a step → instant tip past 1.3 rad → termination on the first
   physics step, before the policy ever got to act.
6. **v2 launched 10:42:27** with a K1-tailored gentle terrain (`K1_VLNCE_TERRAINS_CFG`:
   random_rough 1–6 cm + slopes ±0–0.15 rad, no stairs, no boxes). At iter 500
   reward broke out to **+29.52**, ep_len **978**. By iter 2000 reward 56.34
   and ep_len 1407 — policy walking reliably.
7. **v2 killed by own smoke watcher at iter 2012** (11:29:31). The smoke watcher
   correctly detected the rsl_rl 3.1.2 → 2.0.2 agent.yaml schema mismatch
   (sanitized wrapper not yet in place) and killed training to "save 13 h".
   Integration fix was applied seconds later; smoke then ran cleanly. Net effect:
   ~50 min of v2 compute lost.
8. **v3 launched 11:40:01** — same gentle terrain, paranoid watchers:
   - fire_smoke_when_ready disabled (integration already validated manually)
   - auto_bench requires latest ckpt iter ≥ 15 k before firing the full benchmark
   - no pre-bench smoke (avoids the race that masked the v2 issue)

### Earlier history (legged-loco v1 / v2)

- **`v1` (2000 iters, ~1.7 h, blind-style termination)** — placeholder eval baseline.
- **`v2` (resumed from v1 → +7000 iters, ~6 h, 6h cap)** — reward ~17, best
  legged-loco checkpoint.
- **Fix-v1 eval applied** to v2 (`bad_orientation` 0.8 → 1.3 rad + 8-frame VLM
  warmup) — partial run 140 eps, **SR=0% / path 1.95 m**, K1 walks reliably
  but doesn't navigate Matterport. The fix-v1 *code* (k1_matterport_vision_cfg.py
  `limit_angle=1.3`, `navila_eval.py:325-345`) is now permanent in the eval env
  and applies to all future benchmarks automatically.

---

## Comparison table (live aggregate)

| Robot | Policy | NE↓ | OS↑ | SR↑ | SPL↑ | Path Len |
|-------|--------|-----|-----|-----|------|----------|
| Go2 (paper, Table IV) | Blind  | 6.03 | 49.0 | 36.2 | 33.3 | — |
| Go2 (paper, Table IV) | Vision | 5.49 | 58.7 | 50.2 | 45.5 | — |
| H1  (paper, Table IV) | Blind  | 7.67 | 33.3 | 24.4 | 21.0 | — |
| H1  (paper, Table IV) | Vision | 5.86 | 54.6 | 45.3 | 40.3 | — |
| K1 stub (zero-action, 2026-05-15)              | 10.66 | 0.0 | 0.0 | 0.0 | 0.28 m |
| K1 v2 iter 9000 (broken eval — 0.8 rad termination) | 9.93 | 0.0 | 0.0 | 0.0 | 0.81 m |
| **K1 v2 iter 9000 (Fix-v1 eval — 1.3 rad + VLM warmup, 140 eps)** | **10.15** | **0.0** | **0.0** | **0.0** | **1.95 m** |
| ~~K1 v3 50 k (legged-loco)~~ — *collapsed mid-training, not benchmarked* | — | — | — | — | — |
| ~~K1 booster_train vlnce_v1~~ — *episode length stuck at 1 (ROUGH_TERRAINS_CFG too steep)* | — | — | — | — | — |
| ~~K1 booster_train vlnce_v2~~ — *iter 2012 reward 56, killed by own smoke watcher on agent.yaml schema bug* | — | — | — | — | — |
| **K1 booster_train vlnce_v3 (training, gentle terrain + safer watchers, ETA ~13.6 h)** | TBD | TBD | TBD | TBD | TBD |

> Re-aggregate any time:
> `python ~/Projects/k1_research/aggregate_k1_vision_results.py --paper-table`

---

## Where everything is

### Training runs (legged-loco)
```bash
ls -la ~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/
# 2026-05-15_23-17-29_k1_vision_v1/        model_1999.pt   (2 k iters, 1h54)
# 2026-05-16_01-16-18_k1_vision_v2_10k/    model_9000.pt   (9 k iters, 6h cap)
# 2026-05-16_12-47-..._k1_vision_v3_50k/   in progress     (50 k iters target)
```

### Benchmark results (NaVILA-Bench)
```bash
ls ~/Projects/k1_research/NaVILA-Bench/eval_results/
# k1_matterport_base_loco_stub_zero_2026-05-15            # stub baseline (~19 eps)
# k1_matterport_vision_loco_2026-05-15_23-17-29_k1_vision_v1   # earlier failed v2 attempt (empty)
# k1_matterport_vision_loco_2026-05-16_01-16-18_k1_vision_v2_10k         # broken run (235 eps, SR=0%)
# k1_matterport_vision_loco_2026-05-16_01-16-18_k1_vision_v2_10k_fix_v1  # Fix-v1 run (~140 eps, SR=0%)
```

### Tmux sessions active
```bash
~/miniconda3/bin/tmux ls
# k1train3 — v3 training, ETA ~25-30h
```

### Reports
- `~/Projects/k1_research/OVERNIGHT_RESULTS.md` (this file)
- `~/Projects/k1_research/experiments/navila/SETUP_LOG.md` (per-command log)
- `~/Projects/k1_research/experiments/navila/BENCHMARK_REPORT.md` (broader report)

---

## Status timeline

- **23:17** training launched (tmux `k1train`, conda `vlnce-isaac`)
- **23:20** env builds, 12 reward terms active, contact sensors working
- **23:23** PPO running, iter time 6.4 s, track_lin_vel_xy_exp 0.011
- **23:26** base_contact dropped 117 → 13 per env (8 min in)
- **23:36** iter 200 checkpoint (1.56 base_contact)
- **23:55** user extended overnight; planning two-checkpoint chain
- **01:11** v1 done — iter 1999 (rsl_rl is 0-indexed), 1h54 total, base_contact 0.84
- **01:12** Watcher attempted to chain v2 in tmux but the conda `tmux`
  binary symlink got replaced by a Unix-domain-socket file
  (cause unclear; possibly a tmux-in-tmux race). v2 training killed.
- **01:16** switched to nohup+setsid `mega_pipeline.sh` — no tmux.
  v2 relaunched with `--resume True --max_iterations=8000` (= 10 k total).
  `timeout(1)` 6h cap so benchmark gets ~2h before user wakes.
- **03:46** v2 iter 5000. Mean reward 13.69 (vs v1 end ~8.4 → +63%).
- **05:26** v2 iter 7000. Mean reward 16.76. base_contact 0.5.
- **06:11** v2 iter 8000. Mean reward 16.05.
- **07:01** v2 iter 9000. Mean reward 16.69, base_contact 0.625.
- **07:16:13** v2 hit 6 h wall-clock cap (timeout exit 124). Final
  checkpoint: model_9000.pt. v2 chosen for benchmark.
- **07:16:43** broken benchmark launched. VLM bridge up, episode 0 starts.
- **09:11** broken benchmark @ 199 eps: SR=0%, path 0.81m, NE 9.93m
- **09:46** **DIAGNOSIS:** training had `base_contact` only, eval had
  `bad_orientation` 0.8 rad only. 61% of episodes ended in <2 s. NaVILA
  also emitted "turn left 45 degree" 5-9× in a row at episode start
  because image_observations was padded with black frames (only 1 real
  frame at first VLM call). Both issues fixed.
- **09:50** Fix-v1 benchmark launched (Fix A relaxes bad_orientation,
  adds base_contact + contact_forces; Fix B warms up VLM image obs).
- **09:55** Fix-v1 crashed with "Inplace update to inference tensor"
  — `torch.inference_mode()` block in my Fix B + main-loop
  `update_command` in-place write. Removed inference_mode wrapper.
- **10:06** Fix-v1 relaunched. First 3 eps had path_len 0.054m
  — K1 stood still while NaVILA spammed "move forward 75 cm" because
  the merged-arm Trunk geometry was triggering base_contact on a wall
  near spawn. Removed base_contact entirely (kept just relaxed
  bad_orientation 1.3 rad).
- **10:32** Fix-v1 final relaunch. K1 walking properly now.
- **12:42** Fix-v1 @ 140 eps: SR=0%, path 1.95 m, NE 10.15 m. K1 walks
  reliably but never reaches goals (7-13 m typical).
- **12:47** **v3 50k training launched** (tmux `k1train3`). Added
  `bad_orientation` (1.3 rad) to training terminations so the policy
  trains under the same envelope it will be evaluated in.

---

## v3 plan and what to monitor

### Config delta vs v2
- `k1_low_vision_cfg.py:TerminationsCfg` now has `bad_orientation`
  (limit_angle=1.3 rad) in addition to `base_contact`.
- All other terrain/reward/obs config identical to v2.
- Training starts from scratch (no `--resume`), matching the paper's
  H1 50k schedule.

### Command
```bash
tmux new-session -d -s k1train3 'cd ~/Projects/k1_research/legged-loco \
  && source ~/miniconda3/etc/profile.d/conda.sh && conda activate vlnce-isaac \
  && export OMNI_KIT_ACCEPT_EULA=yes \
  && python scripts/train.py --task=k1_vision --run_name=k1_vision_v3_50k \
       --max_iterations=50000 --save_interval=2000 --headless \
       2>&1 | tee /tmp/k1_train_v3.log'
```

### Monitoring (during the weekend)
```bash
# tmux session
~/miniconda3/bin/tmux capture-pane -t k1train3 -p | tail -30

# checkpoints
ls -la ~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/*v3_50k/model_*.pt

# tensorboard (if you want graphs)
tensorboard --logdir ~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/

# benchmark intermediate checkpoint (test mid-training without killing v3 —
# only safe if no other Isaac Sim is running)
ln -sfn 2026-05-16_12-47-..._k1_vision_v3_50k \
    ~/Projects/k1_research/NaVILA-Bench/logs/rsl_rl/k1_vision_rough/k1_v3_iter_NNNN
bash ~/Projects/k1_research/fix_v1_pipeline.sh    # edit LOAD_RUN inside
```

### Expected outcomes
At 50 k iters with the matched termination, the K1 should:
- Walk faster than v2 (the 1.3-rad envelope is genuinely tighter than the
  base_contact-only envelope, so the policy learns a less wobbly gait)
- Track yaw commands more accurately over the 10 m walks NaVILA asks for
- Plausibly achieve SR 10-30 % (H1 paper got SR 45 % at 50 k, but K1 is
  shorter / less stable so we expect lower)

### Risk if v3 underperforms
- Reward might converge before 50 k iters (humanoid PPO often plateaus
  by iter 20-30 k). If `Mean reward` doesn't grow after iter 30 k,
  consider stopping early and benchmarking that checkpoint.
- Training might be unstable with the new `bad_orientation` termination
  added (more env resets, less effective experience). If episode reset
  rate stays high (`Episode Termination/bad_orientation > 50/env`),
  consider relaxing further to 1.5 rad and resuming.

---

## How to inspect

### Re-aggregate Fix-v1 benchmark
```bash
python ~/Projects/k1_research/aggregate_k1_vision_results.py --paper-table
# (default points to the latest k1_matterport_vision_* dir;
#  use --measurements-dir to pick a specific run)
```

### Watch a Fix-v1 video
```bash
ls -t ~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_2026-05-16_01-16-18_k1_vision_v2_10k_fix_v1/videos/ | head
mpv ~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_2026-05-16_01-16-18_k1_vision_v2_10k_fix_v1/videos/output_69.mp4
```

### Resume v3 if it crashes
```bash
cd ~/Projects/k1_research/legged-loco && conda activate vlnce-isaac
export OMNI_KIT_ACCEPT_EULA=yes
python scripts/train.py --task=k1_vision \
    --resume True --load_run=<v3-timestamped-dir> --max_iterations=50000 --headless
```

---

## Files touched this overnight

### Code (committed)
- `legged-loco` branch `k1-vision`:
  - `isaaclab_exts/.../config/k1/k1_low_vision_cfg.py`
    (NEW; +bad_orientation(1.3) added 12:46 for v3)
  - `isaaclab_exts/.../config/k1/k1_low_base_cfg.py` (NEW, blind fallback)
  - `isaaclab_exts/.../config/k1/__init__.py` (NEW)
  - `isaaclab_exts/.../config/__init__.py` (+1 line)
- `NaVILA-Bench` branch `vlnce-isaac-benchmark`:
  - `isaaclab_exts/.../config/k1/k1_matterport_vision_cfg.py` (NEW;
    relaxed bad_orientation, no base_contact in eval)
  - `isaaclab_exts/.../config/k1/__init__.py` (+1 task registration)
- `NaVILA-Bench` working tree (NOT yet committed — Fix B):
  - `scripts/navila_eval.py` (Fix B: 8-frame VLM image warmup)

### Scripts (top-level)
- `run_k1_benchmark.sh` (pipeline driver)
- `mega_pipeline.sh` (used overnight chained train→bench)
- `fix_v1_pipeline.sh` (post-fix benchmark driver)
- `aggregate_k1_vision_results.py` (run after benchmarks)
- `watch_and_benchmark.sh` (kept for reference; superseded by mega_pipeline)

### Logs
- `/tmp/k1_train.log` (v1)
- `/tmp/k1_train_10k.log` (v2)
- `/tmp/k1_train_v3.log` (v3, live)
- `/tmp/k1_bench_vision.log` (broken eval)
- `/tmp/k1_bench_fix_v1.log` (Fix-v1 eval)
- `/tmp/vlm_server.log` (broken eval VLM bridge)
- `/tmp/vlm_server_fix_v1.log` (Fix-v1 VLM bridge)

---

## What to do when you wake up

1. **Check v3 training progress:**
   ```bash
   ~/miniconda3/bin/tmux capture-pane -t k1train3 -p | tail -30
   ```
   If iter < 5 k after 4 h, something's off. Otherwise let it run.

2. **(Optional) Test an intermediate v3 checkpoint:**
   When `model_20000.pt` exists in the v3 dir, you can run the Fix-v1
   pipeline against it without killing training — but training uses
   ~6 GB GPU, and the benchmark uses ~6 GB + 17 GB VLM = 29 GB, which
   barely fits on 32 GB 5090. Safer to wait until v3 finishes.

3. **When v3 finishes (~Sunday afternoon):**
   - Symlink: `ln -sfn <v3-dir> ~/Projects/k1_research/NaVILA-Bench/logs/rsl_rl/k1_vision_rough/k1_v3_50k`
   - Edit `fix_v1_pipeline.sh`'s `LOAD_RUN` to `k1_v3_50k`
   - Run: `bash ~/Projects/k1_research/fix_v1_pipeline.sh`

---

## v3 training complete (2026-05-18 06:38)

Total wall time: **41h 51m** (= start 2026-05-16 12:47).

### Final checkpoint inventory (v3, model_*.pt)
```
2026-05-16  14:37  model_2000.pt
2026-05-16  16:17  model_4000.pt
2026-05-16  17:57  model_6000.pt
2026-05-16  19:37  model_8000.pt
2026-05-16  21:17  model_10000.pt
2026-05-16  22:57  model_12000.pt
2026-05-17  00:47  model_14000.pt
2026-05-17  02:27  model_16000.pt
2026-05-17  04:07  model_18000.pt
2026-05-17  05:57  model_20000.pt
2026-05-17  07:37  model_22000.pt
2026-05-17  09:17  model_24000.pt
2026-05-17  10:47  model_26000.pt
2026-05-17  12:27  model_28000.pt
2026-05-17  14:07  model_30000.pt
2026-05-17  15:47  model_32000.pt
2026-05-17  17:27  model_34000.pt
2026-05-17  19:07  model_36000.pt
2026-05-17  20:37  model_38000.pt
2026-05-17  22:17  model_40000.pt
2026-05-17  23:57  model_42000.pt
2026-05-18  01:37  model_44000.pt
2026-05-18  03:17  model_46000.pt
2026-05-18  04:57  model_48000.pt
2026-05-18  06:38  model_49999.pt   ← FINAL
```
Location: `~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/2026-05-16_12-47-23_k1_vision_v3_50k/`

### Late-training warning
`Episode Termination/bad_orientation` rate evolved as:
- iter 0-20k: 0.0 (curriculum too easy for K1 to tilt much)
- iter 30k: 0.03 sporadic
- iter 38k: 0.03 sporadic
- iter 44k: 0.06
- **iter 46-50k: 0.5-0.94** (terrain curriculum hit max difficulty; policy struggled)

This means `model_42000.pt` or `model_44000.pt` may benchmark *better* than
the final `model_49999.pt`. Worth benchmarking both.

### To benchmark the final v3
```bash
# Symlink the v3 dir into NaVILA-Bench
ln -sfn /home/janga/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/2026-05-16_12-47-23_k1_vision_v3_50k \
   ~/Projects/k1_research/NaVILA-Bench/logs/rsl_rl/k1_vision_rough/k1_v3_50k

# Edit fix_v1_pipeline.sh's LOAD_RUN to "k1_v3_50k", then:
bash ~/Projects/k1_research/fix_v1_pipeline.sh
```

### To benchmark a specific intermediate checkpoint
get_checkpoint_path picks the latest model file by alphabetical sort.
Easiest way: move all *later* checkpoints out temporarily, e.g.:
```bash
RUN=~/Projects/k1_research/legged-loco/logs/rsl_rl/k1_vision_rough/2026-05-16_12-47-23_k1_vision_v3_50k
mkdir -p $RUN/_attic && mv $RUN/model_4{4,6,8}*.pt $RUN/model_49999.pt $RUN/_attic/
# now the latest checkpoint visible is model_42000.pt — bench that
```
