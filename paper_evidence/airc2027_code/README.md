# airc2027_code — the code that produced the sweep

This directory holds the code that ran the NaVILA–K1 transform and camera-height sweep
(2026-07-24 to 2026-08-17) and the June full run (`full_14498`). It contains only files that this project
wrote or modified. Unmodified upstream files are not copied; where to get them is listed below.

The runner at the top of this repository, `arms_runner.sh`, drove the whole sweep. It was committed at
launch (2026-07-24) and has not changed since. The code here was installed at these paths on the sweep
workstation:

| Here | Path on the sweep workstation | Sweep variable |
|---|---|---|
| `navila_bench/` | `~/Projects/k1_research/NaVILA-Bench-main/` (symlink `NaVILA-Bench`) | `$NB` |
| `workspace/scripts/run_powered_benchmark.sh` | `~/Projects/k1_research/k1-research-workspace-main/scripts/run_powered_benchmark.sh` | `$RUNNER` |
| `workspace/aggregate_k1_vision_results.py` | `~/Projects/k1_research/k1-research-workspace-main/aggregate_k1_vision_results.py` | `$AGG` |
| `isaaclab_fork/` | `~/Projects/k1_research/IsaacLab/` (only the one modified file) | — |
| `training_config_model_14498/` | `~/robots/k1/workspace/booster_train/logs/rsl_rl/k1_navila/<run>/` | `$CKPT` provenance |

`checkpoints/model_14498.pt` is not committed (weights are gitignored). Its SHA-256 is in
`receipts/provenance/checkpoint_sha256.txt`.

## How `arms_runner.sh` uses each file

```
arms_runner.sh                      env: VLM_BRIDGE_EXTRA=--load_8bit, EP_TIMEOUT=900
├─ per arm: TRANSFORM=<stretch|crop|pad> $RUNNER <tag> $CKPT 0 <300|200> "[--cam_z=…]"
│  └─ workspace/scripts/run_powered_benchmark.sh
│     ├─ if no bridge is listening: conda env `navila` →
│     │    navila_bench/scripts/vlm_server_bridge.py --model_path …/navila-llama3-8b-8f --port 54321 --load_8bit
│     │    (imports `llava` from AnjieCheng/NaVILA @ 76b98f2, unmodified)
│     ├─ inline Python: episode index → measurement-file index (from vln_ce_isaac_v1.json.gz)
│     ├─ per episode, conda env `vlnce-isaac`:
│     │    timeout -k 30 $EP_TIMEOUT python navila_bench/scripts/navila_eval_v3.py
│     │      --task=k1_matterport_vision --num_envs=1 --checkpoint=$CKPT --episode_idx=<i>
│     │      --gait_phase_init=0.0 --out_tag=<tag> --closed_loop --max_episode_s 120
│     │      --vlm_transform $TRANSFORM [--cam_z=…] --headless --enable_cameras
│     │    ├─ scripts/nav_diag.py            (imported; diagnostics only with --diag, which the sweep did not pass)
│     │    ├─ omni.isaac.vlnce: config/k1/*  (K1 task: robot, camera, terminations)
│     │    │                    utils/wrappers_v3.py (235-dim policy observation, action scatter, stall backstop)
│     │    │                    utils/eval_utils.py  (VLM text → velocity command parser)
│     │    │                    vlnce/mdp/observations.py (camera observation term)
│     │    └─ Isaac Lab fork (+ isaaclab_fork/…/simulation_cfg.py: reduced PhysX GPU buffers), Isaac Sim 4.1.0
│     ├─ if an episode left no JSON: inline Python writes a stub record
│     │    (rc 124/137 → term_reason "wall_timeout", else "eval_crash"; distances = -1.0 sentinel)
│     └─ python ~/Projects/k1_research/aggregate_k1_vision_results.py --paper-table
├─ python3 $AGG --measurements-dir … --total-episodes <end>   → lines appended to SWEEP_STATUS.md
├─ inline Python: term_reason counts                          → SWEEP_STATUS.md
└─ copies the per-episode JSONs to receipts/sweep_measurements/<tag>/, then git commit + push
```

Arms, in order: `stretchA_300`, `stretchB_300`, `crop_300`, `pad_300` (episodes 0–299), then
`h060_200` … `h150_200` (episodes 0–199, stretch, `--cam_z` = 0.07, 0.25, 0.42, 0.57, 0.77, 0.97).
`full_14498` (June, 1,077 episodes) used earlier versions of three of these files: the evaluator from
2026-06-11 (no camera overrides, no SIGTERM handler, no `term_reason` fields), the runner from
2026-06-11 (no stub record; a timed-out episode left no JSON and was re-run on the next invocation, the
last pass with EP_TIMEOUT=1800), and the aggregator from 2026-06-10. Each change from June to the sweep is
purely additive. The changes are in `git_state/navila_eval_v3_june_to_sweep.diff`,
`git_state/run_powered_benchmark_june_to_sweep.diff` and `git_state/aggregate_june_to_sweep.diff`;
reverse-apply them to recover the June versions. Per git history, every other file here was last changed
before the June run started (2026-06-11 ~15:30). For the imported modules this is also confirmed by
their bytecode caches.

## File roles

### `navila_bench/` — the benchmark (upstream: yang-zj1026/NaVILA-Bench, MIT, `navila_bench/LICENSE`)
Base: upstream commit `e9d2db12ce5788c0f987d734c0094100b6bc0d3a`. 95 of the 121 benchmark files tracked on
the workstation are byte-identical to it. These are not copied here (for example `scripts/cli_args.py`,
`utils/measures.py`, `utils/wrappers.py`, the rest of `vlnce/mdp/`, all of `omni.isaac.matterport`, and the
vendored `scripts/rsl_rl`).

Written by this project:

| File | Role |
|---|---|
| `scripts/navila_eval_v3.py` | Episode loop. Builds the K1 actor (235→512→256→128→12), loads the episode, applies `--cam_*` overrides, runs the pre-VLM warm-up, queries the VLM (8 frames), executes commands (closed-loop executor `ClosedLoopController`), applies the 500-step stall guard, step cap and wall-clock SIGTERM handler, writes `measurements/<file>.json` and `videos/output_<file>.mp4`. |
| `scripts/vlm_server_bridge.py` | TCP model server (port 54321) for `navila-llama3-8b-8f`. Implements `--load_8bit` (bitsandbytes int8 via llava `load_pretrained_model`, fp16 for the rest, SDPA attention), the NaVILA prompt, and greedy decoding. |
| `scripts/nav_diag.py` | Per-episode diagnostics (`--diag`). Imported by the evaluator, inactive in the sweep. |
| `scripts/l0_camera_dump.py` | Not on the sweep path. It is used by `receipts/acceptance_and_rootcause.sh`, the two-process render comparison ("5a"). |
| `…/config/k1/k1_matterport_vision_cfg.py` | The task `k1_matterport_vision`: K1 articulation and actuators, camera (offset, 1280×720, aperture), terminations, sim dt/decimation, episode length. |
| `…/config/k1/k1_matterport_base_cfg.py`, `…/config/k1/__init__.py` | Base K1 task and gym registration of both K1 tasks. |
| `…/utils/wrappers_v3.py` | Policy-observation contract (term-major 5-frame history), joint-order permutation, action scaling, gait clock, 50-step reset warm-up, hopeless-stall backstop. |

Upstream files modified by this project (what changed):

| File | Change |
|---|---|
| `…/config/__init__.py` | adds `from .k1 import *` (registers the K1 tasks) |
| `…/utils/__init__.py` | exports `VLNEnvWrapperV3` |
| `…/utils/eval_utils.py` | `get_vel_command` rewritten: extracts the turn angle instead of bucketing to {15, 30, 45}; numberless turn → 45°; unrecognised text → hold for 0.5 s and re-query instead of walking forward |
| `…/vlnce/mdp/observations.py` | `isaac_camera_data` / `process_depth_image` return a zero placeholder while the camera sensor is uninitialised, during shape discovery at env construction |
| `…/vlnce/mdp/actions/navigation_actions.py`, `vlm_navigation_actions.py`, `vlm_navigation_actions_gpt.py` | remove an unused `import torchvision` (it pulled an incompatible Pillow); imported but unused by the K1 task |

The full per-file diffs are in `git_state/navila_bench_diff_vs_upstream_e9d2db1.diff`.

### `workspace/` — runner and aggregator (written by this project)
`run_powered_benchmark.sh` is the resumable per-episode runner described above. `aggregate_k1_vision_results.py`
computes per-arm means and rates over the full intended set. Missing episodes count as failures, and the
-1.0 distance sentinel is excluded from the NE/ONE means. The repository-root `aggregate_k1_vision_results.py`
is the same file.

### `isaaclab_fork/` — one modified upstream file (Isaac Lab, BSD-3-Clause, `isaaclab_fork/LICENSE`)
Base: yang-zj1026/IsaacLab @ `4d558ec83878c4892a46591c85ba91ac9d3c1834`. In `simulation_cfg.py` (`PhysxCfg`), seven
GPU buffer capacities are reduced so that Isaac Sim plus the int8 VLM fit on a 24 GB RTX 3090. The diff is in
`git_state/isaaclab_diff_vs_HEAD_4d558ec.diff`, identical to `receipts/provenance/isaaclab_simulation_cfg.patch`.
Nothing else in the fork is modified.

### `training_config_model_14498/` — how the walking policy was trained
Isaac Lab's resolved `params/env.yaml` and `params/agent.yaml` plus the git-status dump (`git/booster_train.diff`)
for both training runs behind `model_14498.pt`:
- `run_2026-06-09_13-52-48`: from scratch, 12,000 iterations, seed 42 → `model_11999`
- `run_2026-06-10_14-30-20`: resumes `model_11999.pt`, 2,500 iterations → `model_14498` (SHA-256 `8e7559b5…`, equal to the swept checkpoint)

The locomotion task source was untracked in `booster_train` when these runs were trained; it was first committed
there on 2026-07-21. The resolved `env.yaml` is therefore the authoritative record of the training configuration.

### `git_state/` — status and diffs of every checkout involved
| File | Content |
|---|---|
| `k1-navila-research_checkout.txt` | commit/remote/status of `~/Projects/k1_research`; explains the 2026 history rewrite |
| `commit_hash_map_pre_post_rewrite.tsv` | old (pre-rewrite) → current GitHub commit hashes. Older receipts quote old hashes, e.g. the `-e git+…@07a5263…` lines in `receipts/provenance/pip_freeze_vlnce-isaac.txt` |
| `navila_bench_checkout.txt` | provenance, per-file history, status and file classification of `$NB` |
| `navila_bench_diff_vs_last_commit.diff` | empty: the working tree equals its last commit |
| `navila_bench_diff_vs_upstream_e9d2db1.diff` | everything this project changed or added relative to upstream NaVILA-Bench |
| `navila_bench_diff_vs_zip_source_c60d042.diff` | changes made on the sweep workstation after the code arrived (2026-06-05) |
| `workspace_checkout.txt`, `workspace_diff_vs_last_commit.diff`, `workspace_diff_vs_zip_source_1031552.diff` | the same for `k1-research-workspace-main/` |
| `isaaclab_checkout.txt`, `isaaclab_diff_vs_HEAD_4d558ec.diff` | the Isaac Lab fork |
| `navila_vlm_repo_checkout.txt` | the NaVILA/VILA code the bridge imports (clean) |
| `sweep_time_verification.txt` | evidence that these files are what ran during the sweep, and what cannot be proven |

## Rebuilding the runtime layout
1. Clone yang-zj1026/NaVILA-Bench at `e9d2db1` as `~/Projects/k1_research/NaVILA-Bench-main`, symlink
   `NaVILA-Bench` to it, and copy `navila_bench/` over it.
2. Put `workspace/` at `~/Projects/k1_research/k1-research-workspace-main/`.
3. Clone yang-zj1026/IsaacLab at `4d558ec`, copy `isaaclab_fork/source/…/simulation_cfg.py` over it, and
   install as described in the NaVILA-Bench README (Isaac Sim 4.1.0, conda env `vlnce-isaac`). The package lists
   are in `receipts/provenance/`.
4. Clone AnjieCheng/NaVILA at `76b98f2` as `~/Projects/k1_research/experiments/navila` (conda env `navila`),
   and download `a8cheng/navila-llama3-8b-8f` and the VLN-CE-Isaac scenes and `vln_ce_isaac_v1.json.gz`
   (SHA-256 `ceb2a2a9…0eec`) from Hugging Face.
5. Run `bash arms_runner.sh`.
