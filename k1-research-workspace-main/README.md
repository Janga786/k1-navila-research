# K1 Humanoid Research Workspace

Vision-language navigation (VLN-CE) on the Booster K1 humanoid. Trains a
PPO low-level walking policy in Isaac Lab, drives it with NaVILA (LLaMA3-8B
+ SigLIP) as the high-level planner, and evaluates against the NaVILA-Bench
1077-episode test set.

## Where to look first

| File | Purpose |
|------|---------|
| `OVERNIGHT_RESULTS.md` | **Latest** results, the running benchmark status, milestone table, what just changed. **Read this first.** |
| `SETUP_LOG.md` | Append-only audit log of decisions and commands during long-running training/bench runs. |
| `HOW_IT_ALL_WORKS.md` | System tour: real-robot pipeline, VLM internals, velocity mapping, benchmark pipeline, locomotion policy, constants cheat sheet, debugging guide. The canonical "how does this all fit together" doc. |
| `HANDOFF.md` | Earlier handoff snapshot. Older than the docs above, kept for context. |

## Current state (2026-05-20)

- Active policy: **v3** — `Booster-K1-Velocity-Vision-VLNCE-v3`,
  `booster/booster_train/logs/rsl_rl/k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/`.
  30,000 iters. Peak reward 111.57 @ iter 16348. Best save-point: `model_16000.pt`.
- v3 architecture: subclasses `LocomotionVelocityFlatEnvCfg` (the reward-103
  working base) and adds a separate `vision` obs group with a 160-dim
  height_scan. Actor input is 235-dim (term-major flatten of a 5-step
  history of 47-dim per-step proprio + cmd + gait phase). `action_scale=0.25`.
- v3 in MuJoCo (sim-to-sim): `K1_locomotion.xml` (arm joints removed →
  rigid extensions of Trunk; mass conserved at 19.666 kg) → **8.32 s of
  walking, 1.61 m traveled** at `--gait_phase_init=0.70`.
- v3 in NaVILA-Bench: live full run on episodes [42, 1077) in tmux `bench_v3`
  (resumed 2026-05-20 00:55 after a paused earlier run). At n≈300 the K1
  averages NE 7.58 m, SR 0.7%, SPL 0.5%, OS 8.0%, mean path_length 8.12 m
  — robot walks reliably; nav accuracy is the open problem.
- Old v2 (`k1_velocity_vlnce/2026-05-18_21-15-33_k1_vlnce_v2_resumed/`) is
  **kept for reference only.** Its peak-reward 73 was a lunge-and-fall
  episode, not a stable gait. Don't bench, don't reuse — use v3
  `model_16000.pt`.

## Layout

```
k1_research/
├── OVERNIGHT_RESULTS.md   # latest results + running bench status
├── SETUP_LOG.md           # append-only decisions/commands log
├── HOW_IT_ALL_WORKS.md    # canonical system tour
├── HANDOFF.md             # older handoff snapshot
├── README.md              # this file
│
├── booster/
│   ├── booster_train/     # Isaac Lab 2.3 / rsl_rl 3.1.2 training repo
│   │                      #   tasks/manager_based/velocity/robots/k1/
│   │                      #     vlnce_v3_env_cfg.py   ← USE FOR NEW WORK
│   │                      #     vlnce_env_cfg.py      ← v2, do not use
│   │                      #   logs/rsl_rl/k1_velocity_vlnce_v3/...
│   ├── booster_assets/    # K1 URDFs / MJCFs / meshes / motions
│   │     robots/K1/{K1_22dof,K1_locomotion}.{urdf,xml}
│   ├── NaVILA/            # NaVILA training code + checkpoints
│   │     checkpoints/navila-llama3-8b-8f/   ← VLM weights
│   └── booster_robotics_sdk, booster_deploy
│
├── NaVILA-Bench/          # Isaac Lab 1.1 / Isaac Sim 4.1 bench env
│   ├── isaaclab_exts/.../config/k1/k1_matterport_vision_cfg.py
│   │                      #   (K1 articulation, camera, observations)
│   ├── scripts/
│   │     navila_eval.py        ← v2 driver
│   │     navila_eval_v3.py     ← v3 driver (with VLNEnvWrapperV3)
│   │     vlm_server_bridge.py  ← runs NaVILA in 'navila' env, port 54321
│   │     run_benchmark.py      ← v2 multi-ep wrapper
│   └── eval_results/<task>_loco_<tag>/measurements/*.json
│
├── experiments/navila/    # real-robot bridge, demos, mocap relay
├── legged-loco/           # earlier training repo (deprecated for vlnce)
├── K1_Robot_Project/      # robot hardware notes
├── Documentation/         # snapshots
├── papers/                # PDFs
├── checkpoints/           # symlinks to best policies
├── notes/                 # research notes
├── results/               # figures, plots, videos
└── scripts/               # workspace-level launchers + monitors
```

## Key scripts (in `scripts/`)

| Script | Purpose |
|--------|---------|
| `train_k1_vlnce_v3.sh` / `resume_k1_vlnce_v3.sh` | Launch / resume v3 training in tmux `k1_vlnce_train` (env `isaacLab_311`). Logs to `/tmp/k1_booster_train.log`. |
| `monitor_k1_vlnce_resume.sh` | 5-min watcher: alerts on action_rate < -1e6, reward < 50% of peak, 15k-iter plateau (auto-stops). |
| `bench_k1_vlnce_v3.sh smoke <ckpt> [ep] [phase]` | 1-episode integration test with the v3 wrapper. |
| `bench_k1_vlnce_v3.sh full <ckpt> [start] [end]` | Full 1077-ep bench in tmux `bench_v3`. |
| `mujoco_test_v3_policy.py` | Sim-to-sim test of a v3 checkpoint against a MJCF. |
| `auto_phase23_k1_vlnce.sh` | Picks best ckpt from tensorboard, runs smoke, then full bench. Triggered when training plateaus. |
| `fire_cam050_experiment.sh` | Camera-height ablation (K1 rgb_camera z=0.85 → 0.50, 50 eps). |

## Common operations

```bash
# Resume v3 training (if you stopped it)
~/Projects/k1_research/scripts/resume_k1_vlnce_v3.sh

# Watch training live
tail -F /tmp/k1_booster_train.log
LD_LIBRARY_PATH=$HOME/miniconda3/lib $HOME/miniconda3/bin/tmux attach -t k1_vlnce_train

# Test a checkpoint with one bench episode (smoke)
~/Projects/k1_research/scripts/bench_k1_vlnce_v3.sh smoke \
    ~/Projects/k1_research/booster/booster_train/logs/rsl_rl/k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/model_16000.pt

# Run the full 1077-episode bench in tmux 'bench_v3'
~/Projects/k1_research/scripts/bench_k1_vlnce_v3.sh full <ckpt> 0 1077

# Aggregate measurements into a paper-table-formatted row
python ~/Projects/k1_research/aggregate_k1_vision_results.py \
    --measurements-dir ~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_v3_model_16000/measurements \
    --paper-table

# MuJoCo sim-to-sim sanity check on a checkpoint
python ~/Projects/k1_research/scripts/mujoco_test_v3_policy.py \
    --ckpt <model_16000.pt> --mjcf K1_locomotion.xml --gait_phase_init 0.70
```

## Tooling

| Component | Where / how |
|-----------|-------------|
| **Training** | `isaacLab_311` conda env, Isaac Lab 2.3, rsl_rl 3.1.2 |
| **Benchmark** | `vlnce-isaac` conda env, Isaac Lab 1.1, rsl_rl 2.0.2 |
| **VLM server** | `navila` conda env, port 54321, `~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f` (SigLIP 384×384 + LLaMA3-8B + S2 scaling 336/672/1008) |
| **Real-robot bridge** | `~/Projects/k1_research/experiments/navila/` (`navila_k1_bridge.py`, `navila_server.py`) |
| **GPU** | RTX 5090 (sm_120, 32 GB) |
| **tmux** | `$HOME/miniconda3/bin/tmux`. Needs `LD_LIBRARY_PATH=$HOME/miniconda3/lib` to find a new-enough libtinfo. |

## K1 articulation models

| File | Purpose |
|------|---------|
| `K1_22dof.urdf` | Full 22-DOF rig (legs + arms + head) — real robot, BeyondMimic, demos |
| `K1_22dof.xml` | Same in MJCF — sim-to-sim baseline. **Fails** (0.70 s walking). |
| `K1_locomotion.urdf` | 12-DOF legs only — training URDF in `booster_train` |
| `K1_locomotion.xml` | MJCF with arm/head joints removed (welded to Trunk). Sim-to-sim **winner**: 8.32 s walking, 1.61 m. |
| `K1_22dof-ZED.urdf` | Real-robot variant with ZED camera mount |

## NaVILA-Bench paper baselines

| Policy | NE↓ | OS↑ | SR↑ | SPL↑ |
|--------|-----|-----|-----|------|
| Go2 (paper, blind) | 6.03 | 49.0 | 36.2 | 33.3 |
| Go2 (paper, vision) | 5.49 | 58.7 | 50.2 | 45.5 |
| H1 (paper, blind) | 7.67 | 33.3 | 24.4 | 21.0 |
| H1 (paper, vision) | 5.86 | 54.6 | 45.3 | 40.3 |
| **K1 (ours, v3 model_16000)** | _running_ | _running_ | _running_ | _running_ |

Live K1 row is updated in `OVERNIGHT_RESULTS.md` as the bench progresses.

## When you sit down at this machine

1. `tail -20 ~/Projects/k1_research/OVERNIGHT_RESULTS.md` — what just happened.
2. `tail -10 /tmp/k1_v3_bench.log` — if a bench is running, what it's on.
3. `ls -la NaVILA-Bench/eval_results/k1_matterport_vision_loco_*/measurements/ | wc -l` — episode count.
4. `LD_LIBRARY_PATH=$HOME/miniconda3/lib $HOME/miniconda3/bin/tmux ls` — what's alive.
5. `nvidia-smi` — what's on the GPU. If anything is, do **not** start a second GPU process until the first finishes.
