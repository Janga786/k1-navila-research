# 16 — AIRC 2027 walking check: PLAN

Written 2026-10-01, before any GPU run, on the workstation that ran the NaVILA–K1 sweep (`~/Projects/k1_research`,
RTX 3090, driver 580.173.02, Isaac Sim 4.1 + Isaac Lab fork `4d558ec`, conda env `vlnce-isaac`). Work directory:
`~/Projects/k1_research/airc2027_walking/`. Descriptive context for the paper: nothing is tuned and no reported
result changes.

## Question

How well does the walking policy `model_14498` follow the commands that the evaluator's closed-loop executor sends
(forward 0.5 m/s, turns ±π/6 rad/s, zero command between commands), in two setups, both on Isaac Sim 4.1:

- **E** — the robot exactly as evaluated;
- **T** — the training robot model of `model_14498`.

## Files used (read-only; SHA-256 checked today, identical to the Oct 1 provenance record)

| File | SHA-256 |
|---|---|
| `NaVILA-Bench-main/scripts/navila_eval_v3.py` (evaluator) | `9b010715…` |
| `…/omni.isaac.vlnce/config/k1/k1_matterport_vision_cfg.py` (evaluated config) | `e294b61e…` |
| `…/omni.isaac.vlnce/utils/wrappers_v3.py` (`VLNEnvWrapperV3`) | `be607d92…` |
| `…/omni.isaac.vlnce/utils/eval_utils.py`, `measures.py` | `2add9a77…`, `20ec8def…` |
| `IsaacLab/…/sim/simulation_cfg.py` (fork's PhysX buffer settings) | `8d13b5bf…` |
| `checkpoints/model_14498.pt` | `8e7559b5…` |
| `K1_locomotion.urdf` / `K1_22dof.urdf` | `5cf5951d…` / `9312ecc9…` |
| `airc2027_renders/scripts/sim_setup.py` (Oct 1: the evaluator's own lines) | `8a205c61…` |
| `airc2027_renders/configs/k1_corrected_robot_cfg.py` (Oct 1) | `380741f5…` |
| training config `booster_train/logs/rsl_rl/k1_navila/2026-06-10_14-30-20/params/env.yaml` | (read only for values) |

## Robot models

| | **E** (evaluated) | **T** (training robot model) |
|---|---|---|
| Config | evaluated `K1_ARTICULATION_CFG`, unmodified | copy of `k1_corrected_robot_cfg.py` with `fix_base=False` (`scripts/configs/k1_walkcheck_T_robot_cfg.py`) |
| URDF | `K1_locomotion.urdf`, `merge_fixed_joints=True`: head and arms merged into the Trunk at zero joint angles (arms straight out sideways) | `K1_22dof.urdf`: all 22 joints revolute, nothing merged |
| Joints / bodies | 12 / 13 | 22 / 23 |
| Legs | implicit PD: hips+knees stiffness 100, damping 2, effort 68/76/38.3/112 (hip pitch/roll/yaw, knee), velocity 14.66/12.57/17.59/12.57, armature 0.0478/0.0340/0.0283/0.0956; ankles stiffness 50, damping 1, effort 38.3, velocity 17.59, armature 0.0565 | identical to E |
| Arms (Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw ×2) | none (fixed, merged) | implicit PD, stiffness 20, damping 2, armature 0.001, effort 14, velocity 33.51 |
| Head (AAHead_yaw, Head_pitch) | none (fixed, merged) | implicit PD, stiffness 4, damping 1, armature 0.001, effort 6, velocity 7.85 |
| Head/arm pose | zero angles (T-pose), rigid | initial state and position targets at the training pose: Shoulder_Pitch 0.2, Shoulder_Roll −1.25 (L) / +1.25 (R), Elbow_Pitch 0, Elbow_Yaw −0.5 (L) / +0.5 (R), head 0 |
| Total mass | 19.666 kg | 19.666 kg |

- **Delay:** training used `BoosterDelayedPDActuator` with a random 0–3-step delay (`min_delay 0`, `max_delay 3` in
  `env.yaml`, all four actuator groups). Isaac Lab 1.1 has no equivalent, so the evaluation used plain implicit PD; both
  E and T here use plain implicit PD without delay.
- Known property of the fork, to be confirmed by reading the values back from PhysX: for implicit actuators it writes
  stiffness, damping, effort limit, armature and friction into the simulation but not the velocity limit (the PhysX joint
  velocity limits then come from the URDF import). This applies equally to E and T and to the evaluation.
- Every run reads back and records the per-joint PhysX stiffness, damping, armature, friction, effort and velocity
  limits, the body masses and the root-body centre of mass.

## Ground and environment

- `scripts/configs/k1_walkcheck_plane_cfg.py`: a copy of the evaluated config with only these changes: the Matterport
  importer replaced by a flat plane (`TerrainImporterCfg(terrain_type="plane")` at `/World/ground`, plane at z = 0) with
  the Matterport importer's physics material (static and dynamic friction 1.0, friction and restitution combine
  "multiply"); the height scanner ray-casts that plane; the spawn below. Kept unchanged: dt 0.005 s, decimation 4,
  render interval 4, `sim.physics_material`, the fork's PhysX settings, `disable_contact_processing`, robot E,
  actuators, sensors, cameras, observations, actions, the 1.3-rad `bad_orientation` termination.
- `scripts/configs/k1_walkcheck_plane_T_cfg.py`: the same plane config with robot T.
- A diff of each copied config (and of the copied wrapper) against the evaluated file goes to `scripts/diffs/`.
- **Cameras are kept.** `VLNEnvWrapperV3.reset()` and `.step()` return `infos["observations"]["camera_obs"]` (the
  evaluator's `high_level_obs_key`), which needs the robot camera, so the wrapper does not work without the cameras.
- **Spawn:** Trunk at (0, 0, 0.729) m, quaternion (1, 0, 0, 0) (heading +x): dataset index 0's start point lies 0.178 m
  above the Matterport floor, plus the configured 0.55 m. The evaluator's own `reset_start_pos_rot` still runs (it sets
  the episode's Matterport start pose); a configuration hook applied after the evaluator's edits (the same mechanism
  as the Oct 1 `env_cfg_hook`) then sets the plane spawn. The env origin is (0, 0, 0). The dataset index 0 episode is
  passed to the wrapper as in the evaluator; it only feeds the wrapper's distance measures.
- `--cam_z 0.25` (equal to the config default) in all runs, as in the validation reference.

## Policy pipeline

`scripts/walk_check.py` imports the evaluator's code and does not reimplement it: `sim_setup.build_env` (the
evaluator's `main()` lines up to `env.reset()`), `build_v3_actor` with `checkpoint model_14498.pt`, `VLNEnvWrapperV3`
with `--gait_phase_init=0.0` (5-frame observation history, 1.5-Hz gait clock with the stand gate, action scale 0.25,
legs in the training order). Flags as the sweep runner: `--num_envs=1 --headless --enable_cameras`, cwd `NaVILA-Bench`.

- **E:** the unmodified `VLNEnvWrapperV3`.
- **T:** `scripts/wrappers_v3_T.py`, a copy that selects the 12 leg joints by name with Isaac Lab's resolver in the
  policy's training order and asserts that order (`Left_Hip_Pitch, Right_Hip_Pitch, Left_Hip_Roll, Right_Hip_Roll,
  Left_Hip_Yaw, Right_Hip_Yaw, Left_Knee_Pitch, Right_Knee_Pitch, Left_Ankle_Pitch, Right_Ankle_Pitch, Left_Ankle_Roll,
  Right_Ankle_Roll`), observes only those, and writes the 12 leg actions into the env's 22-joint action vector; the 10
  head/arm entries stay 0, so their targets stay at the training pose. It is substituted for `VLNEnvWrapperV3` in
  `omni.isaac.vlnce.utils` before `build_env` imports it (T runs only).

## Commands and trials (plane, robots E and T)

Commands are `(vx m/s, vy m/s, yaw rate rad/s)`, sent as `env.step(torch.tensor(cmd, float32))` like the evaluator's
main loop; π/6 is the evaluator's `np.pi / 6.0` (float32 0.5235988). Control step 0.02 s.

| Trial | Command | Duration |
|---|---|---|
| `stand` | (0, 0, 0) | 12 s = 600 steps |
| `forward` | (0.5, 0, 0) | 12 s = 600 steps |
| `turn_left` | (0, 0, +π/6) | 12 s = 600 steps |
| `turn_right` | (0, 0, −π/6) | 12 s = 600 steps |
| `sequence` | repeat forward 75 steps → zero 8 → turn left 75 → zero 8 → forward 75 → zero 8 → turn right 75 → zero 8 (1.5 s / 0.16 s, the executor's 8 settle steps) | 30 s = 1500 steps: 4 full cycles + forward 75, zero 8, turn left 75, zero 8 + a final partial forward of 6 steps |

## Protocol per trial (one fresh process per trial and repeat)

1. Build the environment with the evaluator's lines; this includes `VLNEnvWrapperV3`'s 1-s reset warm-up (50 steps,
   zero command).
2. The evaluator's 4-s zero-command warm-up: `int(8 × 0.5 / (dt × decimation))` = 200 steps of `env.step(zeros(3))`.
3. The test command, from the next step on.

- **Stop early on a fall:** the env's `bad_orientation` termination (tilt = arccos(−g_z) in the base frame > 1.3 rad,
  the benchmark's fall criterion), checked after every control step in all three phases. The env resets itself in the
  step where the termination fires, so the state recorded in that step is a reset state; it is flagged and excluded,
  and the time of that step is the fall time.
- The wrapper's own `done` from its hopeless-stall check (not a fall; no reset) is recorded but does not end a trial.
- Each trial is run twice (repeats 1 and 2) in fresh processes.

## Recorded data (every control step of all three phases; float32 as produced)

Phase, command, base pose (Trunk position, quaternion w x y z), base velocity in the base frame (Isaac Lab
`root_lin_vel_b`, `root_ang_vel_b`), world-frame root velocities, projected gravity, joint positions and velocities
(all joints, articulation order, with names), the action sent to the env, env termination flags (`bad_orientation`,
terminated, time-out), the wrapper's `done`. Plus a JSON per run: arguments, file hashes, joint and body names, PhysX
read-back, env origin, default root state, the command schedule, wall times. Files: `logs/states/<robot>_<trial>_r<k>.npz`
and `.json`.

## Validation (must pass before Step 2)

Robot E on the unmodified Matterport scene of dataset index 0 (record 0, zsNo4HB9uLZ), with the flags of the reference
run (`--task=k1_matterport_vision --num_envs=1 --checkpoint=…/model_14498.pt --gait_phase_init=0.0 --headless
--enable_cameras --episode_idx=0 --cam_z=0.25`), the same harness and recording as the trials, and protocol `init50` of
`airc2027_renders/scripts/sim_capture.py`: states after the 1-s reset warm-up (`init`); then the command (0.5, 0, 0)
from the first control step after the reset warm-up (no 4-s warm-up in `init50`) for 50 control steps; states after the
50th (`after50`). States are `root_state_w` (13 float32) and `[body_pos_w | body_quat_w]` (13 × 7 float32), as
`sim_capture.py` saves them.

**Pass** iff all four arrays (`init_root_state`, `init_body_states`, `after50_root_state`, `after50_body_states`) are
byte-identical (dtype, shape and bytes) to those in
`airc2027_renders/diagnosis/repeatability/step4_1_idx0_cz025_proc1/`, and the root states' float32 hex strings equal
those in its `capture.json`. If any value differs: stop, report, no Step 2.

## Repeatability

For each robot and trial, every recorded array of repeat 1 is compared bitwise with repeat 2. Report identical, or the
first differing step and the maximum absolute difference per quantity. Metrics come from repeat 1; if the repeats
differ, the metrics of both are reported.

## Metrics (CPU, from the recorded states)

Time 0 = start of the test command; the state after command step j (j = 0, 1, …) is at t = (j + 1) × 0.02 s; the state
after the last warm-up step is t = 0.

- **Base velocity** = velocity of the Trunk frame origin, expressed in the base (Trunk) frame:
  `v = root_lin_vel_b − root_ang_vel_b × r_com`, with `r_com` the root body's centre of mass in the Trunk frame (read
  back from PhysX). Reason: Isaac Lab's `root_lin_vel_b` is the velocity of the root body's centre of mass, and E's root
  body includes the merged head and arms (centre of mass about 0.12 m above the Trunk origin, vs about 0.07 m for T's
  Trunk alone), while the Trunk origin is the same point in both models. The centre-of-mass value is reported as well.
  Yaw rate = `root_ang_vel_b` z. Check: the mean of the finite-difference velocity of the Trunk position over the steady
  state agrees with the mean of `v`.
- **Steady state** = the last 8 s of a 12-s command: 4 s < t ≤ 12 s (400 samples).
- **All 12-s trials:** steady-state mean forward speed, lateral speed and yaw rate; each mean's ratio to the commanded
  value (not defined where the command is 0); RMS error √mean((x − command)²) over the steady state.
- **All trials (incl. the sequence):** trunk height above the plane (z of the Trunk origin; plane at z = 0), mean and
  minimum over the command phase; maximum |roll| and |pitch| (Z-Y-X Euler angles of the Trunk) and maximum tilt over
  the command phase; fell or not, and when (phase and time).
- **Stand:** horizontal drift ‖p_xy(12 s) − p_xy(0)‖ and yaw drift ψ(12 s) − ψ(0) (wrapped).
- **Sequence:** every complete segment, from the state before its first step to the state after its last step:
  - forward: displacement along the heading at the segment start (the evaluator's `ClosedLoopController`
    definition, `dx cos ψ0 + dy sin ψ0`) against 0.75 m; lateral displacement and yaw change; also the same
    displacement at the end of the following zero segment;
  - turns: yaw change (wrapped) against +45° / −45°; planar displacement; also the yaw change at the end of the
    following zero segment;
  - zero segments: residual planar displacement, yaw change, mean forward speed and mean |yaw rate| over the 0.16 s;
  - summaries per segment type (n, mean, SD, min, max, ratio of the mean to the implied value); the final 6-step partial
    forward segment is excluded;
  - the RMS error of forward speed, lateral speed and yaw rate against the step-wise command over the 30 s.
- One table of all metrics (`tables/`) and one figure of forward-speed and yaw-rate traces, E against T (`figures/`).

## GPU use

- All CPU work first. Before any GPU use, wait until `~/Projects/k1_research/GPU_LOCK` does not exist and `nvidia-smi`
  shows no compute process, checking every 10 minutes; then create `GPU_LOCK` containing `walking_check` (atomically,
  only if absent). Delete it at the end, also when stopping early.
- Every GPU process is run by `scripts/run_gpu.sh`, which refuses to start unless this check holds the lock and the GPU
  has no compute process, and logs label, start, end, wall seconds, return code and command to `logs/gpu_time.tsv`.
- Budget 1.5 h of GPU wall time in total (expected ≈ 40 min: 1 validation run + 20 trial runs). No run starts if it
  could take the total over 1.5 h.
- Order: validation → (only if it passes) robot T `stand` repeat 1 (first T run, to surface configuration errors) →
  the remaining 19 trial runs.

## Rules kept

Sweep files are read-only; all work in `airc2027_walking/`; nothing written to `eval_results/`; no model server.
Results go to the private repo `Janga786/NaVILA-Complete-Archive` under `16_airc2027_walking_check/` (this file,
unchanged, then `RESULTS.md`, `scripts/` with the copied configs and diffs, `logs/` incl. the per-step states,
`tables/`, `figures/`); files staged by explicit path, scanned for secrets, each under 50 MB; no force-push; nothing
from the never-upload list (scenes, dataset, weights, meshes, environments, transcripts). Any departure from this plan
goes in a "Deviations" section of `RESULTS.md`; "not determined" where a fact was not established; a failed check stops
the work and is reported.
