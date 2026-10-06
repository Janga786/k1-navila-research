# 16 — AIRC 2027 walking check: RESULTS

Run 2026-10-01 23:40 → 2026-10-02 00:06 (MDT) on the workstation that ran the NaVILA–K1 sweep (`~/Projects/k1_research`,
RTX 3090, driver 580.173.02, Isaac Sim 4.1, Isaac Lab fork `4d558ec`, conda env `vlnce-isaac`). Protocol: `PLAN.md`
(pushed before any GPU run in commit `52ec62c28dc39f56c7b3d937f32a4f9f656dcf86`, unchanged here). Descriptive context
for the paper: nothing was tuned and no reported result changes. All numbers below come from files in this folder;
the tables are copied verbatim from `tables/` (built by `scripts/metrics.py` and `scripts/repeatability.py`).

## Answer

**On a flat plane in Isaac Sim 4.1, the evaluated robot (E) follows the executor's commands about as well as the
training robot model (T).** Steady state (last 8 s of a 12-s command): forward 0.453 vs 0.451 m/s for 0.5 m/s
commanded (0.91 vs 0.90 of the command); turn left +0.529 vs +0.533 rad/s (1.01 vs 1.02 × π/6); turn right −0.507 vs
−0.495 rad/s (0.97 vs 0.94); standing drift 8 vs 10 mm over 12 s. Neither robot fell in any trial (largest tilt 11.1°;
the fall criterion is 74.5°), and the trunk height is the same (0.51 m mean while walking). In the executor-like
sequence both cover 0.61 m of each 0.75-m forward segment (0.81; 0.65 m, 0.87, by the end of the following 0.16-s zero
segment) and turn 43.0° (E) / 44.0° (T) per +45° segment and −43.0° / −43.0° per −45° segment. The largest E−T
difference in any steady-state mean is 0.019 m/s in forward speed (turn right), 0.007 m/s in lateral speed and
0.012 rad/s in yaw rate. The per-step oscillation differs more than the means: E's forward speed oscillates more
(SD 0.072 vs 0.057 m/s while walking forward), T's yaw rate more (SD 0.12 vs 0.09 rad/s).

## Validation (Step 1): PASS — bit for bit

Robot E on the unmodified Matterport scene of dataset index 0 (record 0, zsNo4HB9uLZ), with the reference run's flags
(`--task=k1_matterport_vision --num_envs=1 --checkpoint=…/model_14498.pt --gait_phase_init=0.0 --headless
--enable_cameras --episode_idx=0 --cam_z=0.25`), the trial harness with its per-step recording, and protocol `init50`
of `sim_capture.py` (states after the 1-s reset warm-up; command (0.5, 0, 0) from the first control step after it;
states after 50 control steps):

| Array | vs `airc2027_renders/diagnosis/repeatability/step4_1_idx0_cz025_proc1/` | SHA-256 (both) |
|---|---|---|
| `init_root_state.npy` (13 float32) | byte-identical | `0116d340…` |
| `init_body_states.npy` (13 × 7 float32) | byte-identical | `342258ae…` |
| `after50_root_state.npy` | byte-identical | `6a3566ea…` |
| `after50_body_states.npy` | byte-identical | `0a9891df…` |
| float32 hex of both root states vs `capture.json` | equal | |

Files: `logs/validation/E_matterport_idx0_cz025_init50/` (`compare.json`, the four arrays, per-step `steps.npz/.json`);
log `logs/validate_E_matterport_idx0_cz025.log`. The robot configuration the env was built with differs from the
evaluated scene's robot configuration only in the spawn pose (`steps.json` → `robot_cfg_vs_evaluated_scene_robot`).

## Repeatability (Step 2): every trial bit-identical across fresh processes

| Robot | Trial | Rows (r1 / r2) | All recorded arrays bit-identical | Differences |
|---|---|---|---|---|
| E | stand | 850 / 850 | yes | — |
| E | forward | 850 / 850 | yes | — |
| E | turn_left | 850 / 850 | yes | — |
| E | turn_right | 850 / 850 | yes | — |
| E | sequence | 1750 / 1750 | yes | — |
| T | stand | 850 / 850 | yes | — |
| T | forward | 850 / 850 | yes | — |
| T | turn_left | 850 / 850 | yes | — |
| T | turn_right | 850 / 850 | yes | — |
| T | sequence | 1750 / 1750 | yes | — |

All 18 recorded arrays per run (base pose, base-frame and world-frame velocities, projected gravity, joint positions and
velocities, actions, commands, termination flags) are bit-identical between repeats 1 and 2 (`logs/repeatability.json`).
Across trials, the first 250 recorded steps (both warm-ups) of all five trials are bit-identical for each robot, and
every trial first differs from `stand` at the first command step; the first forward segment of `sequence` is
bit-identical to the first 75 command steps of `forward`. The metrics below are from repeat 1.

## Results by robot and trial (repeat 1)

Forward speed and yaw rate: steady-state means (last 8 s) for the 12-s trials, with the ratio to the command; for the
sequence, the mean over the complete forward segments and over the complete left / right turn segments (each segment
includes its start-up from the preceding zero command). Trunk height: Trunk origin above the plane over the command
phase. Max tilt: arccos(−g_z) over the command phase.

| Robot | Trial | Forward speed (m/s) | Yaw rate (rad/s) | Fell | Trunk height mean / min (m) | Max tilt (°) |
|---|---|---|---|---|---|---|
| E | stand | -0.000 | -0.003 | no | 0.488 / 0.487 | 1.4 |
| E | forward | 0.453 (0.91 × cmd) | +0.017 | no | 0.512 / 0.481 | 9.1 |
| E | turn_left | -0.042 | +0.529 (1.01 × cmd) | no | 0.511 / 0.488 | 8.7 |
| E | turn_right | -0.000 | -0.507 (0.97 × cmd) | no | 0.508 / 0.487 | 10.7 |
| E | sequence | 0.405 (mean over forward segments) | +0.504 / -0.502 (mean over left / right segments) | no | 0.510 / 0.476 | 10.7 |
| T | stand | 0.000 | -0.010 | no | 0.486 / 0.485 | 1.7 |
| T | forward | 0.451 (0.90 × cmd) | +0.025 | no | 0.511 / 0.480 | 8.5 |
| T | turn_left | -0.043 | +0.533 (1.02 × cmd) | no | 0.511 / 0.487 | 8.7 |
| T | turn_right | -0.019 | -0.495 (0.94 × cmd) | no | 0.508 / 0.487 | 8.4 |
| T | sequence | 0.406 (mean over forward segments) | +0.524 / -0.498 (mean over left / right segments) | no | 0.509 / 0.476 | 11.1 |

## All metrics (one table)

Definitions as in `PLAN.md`: base velocity = Trunk-origin velocity in the base frame; steady state = 4 s < t ≤ 12 s;
RMS error = √mean((x − command)²) over the steady state (sequence: over the 30 s against the step-wise command).
Rows that are not in the PLAN's list (Deviations, item 1) are descriptive extras.

### All metrics (repeat 1)

Command: stand (0, 0, 0); forward (0.5, 0, 0); turn_left (0, 0, +π/6); turn_right (0, 0, −π/6) for 12 s; steady state = last 8 s. Sequence: 30 s executor-like sequence (RMS errors against the step-wise command over 30 s; segment results below).

| Metric | Unit | stand E | stand T | forward E | forward T | turn_left E | turn_left T | turn_right E | turn_right T | sequence E | sequence T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forward speed, steady-state mean | m/s | -0.000 | 0.000 | 0.453 | 0.451 | -0.042 | -0.043 | -0.000 | -0.019 |  |  |
| forward speed / command |  |  |  | 0.906 | 0.901 |  |  |  |  |  |  |
| forward speed RMS error | m/s | 0.000 | 0.001 | 0.086 | 0.076 | 0.071 | 0.064 | 0.072 | 0.052 | 0.140 | 0.134 |
| forward speed SD (stepping oscillation), steady state | m/s | 0.000 | 0.001 | 0.072 | 0.057 | 0.057 | 0.048 | 0.072 | 0.049 |  |  |
| lateral speed, steady-state mean | m/s | 0.001 | 0.001 | -0.003 | -0.002 | -0.005 | -0.011 | -0.029 | -0.030 |  |  |
| lateral speed RMS error | m/s | 0.001 | 0.001 | 0.063 | 0.058 | 0.080 | 0.079 | 0.093 | 0.090 | 0.085 | 0.078 |
| yaw rate, steady-state mean | rad/s | -0.003 | -0.010 | 0.017 | 0.025 | 0.529 | 0.533 | -0.507 | -0.495 |  |  |
| yaw rate / command |  |  |  |  |  | 1.011 | 1.018 | 0.968 | 0.944 |  |  |
| yaw rate RMS error | rad/s | 0.003 | 0.013 | 0.095 | 0.125 | 0.086 | 0.110 | 0.111 | 0.138 | 0.142 | 0.178 |
| yaw rate SD, steady state | rad/s | 0.001 | 0.008 | 0.093 | 0.122 | 0.086 | 0.110 | 0.110 | 0.135 |  |  |
| forward speed of the root centre of mass (Isaac Lab root_lin_vel_b), steady-state mean | m/s | 0.001 | 0.001 | 0.453 | 0.451 | -0.040 | -0.041 | -0.001 | -0.020 |  |  |
| trunk height above the plane, mean | m | 0.488 | 0.486 | 0.512 | 0.511 | 0.511 | 0.511 | 0.508 | 0.508 | 0.510 | 0.509 |
| trunk height above the plane, min | m | 0.487 | 0.485 | 0.481 | 0.480 | 0.488 | 0.487 | 0.487 | 0.487 | 0.476 | 0.476 |
| max absolute roll | deg | 0.493 | 0.792 | 8.535 | 8.127 | 8.719 | 8.731 | 5.536 | 5.721 | 8.535 | 8.127 |
| max absolute pitch | deg | 1.303 | 1.541 | 7.353 | 7.852 | 5.836 | 6.092 | 10.249 | 7.715 | 8.621 | 9.906 |
| max tilt (fall criterion: 74.5°) | deg | 1.387 | 1.687 | 9.072 | 8.504 | 8.736 | 8.733 | 10.719 | 8.380 | 10.697 | 11.074 |
| time of max tilt (from command start) | s | 11.940 | 10.040 | 0.400 | 0.380 | 0.340 | 0.340 | 10.540 | 7.200 | 5.240 | 5.240 |
| yaw change over the command phase (unwrapped) | deg | 0.928 | 1.139 | 8.267 | 5.871 | 363.321 | 363.462 | -348.886 | -348.460 | 61.983 | 62.039 |
| planar displacement over the command phase | m | 0.008 | 0.010 | 5.474 | 5.429 | 0.098 | 0.081 | 0.150 | 0.104 | 5.134 | 5.205 |
| robot T: max absolute deviation of a head/arm joint from the training pose | rad | n/a | 0.022 | n/a | 0.069 | n/a | 0.046 | n/a | 0.052 | n/a | 0.069 |
| horizontal drift over 12 s | m | 0.008 | 0.010 |  |  |  |  |  |  |  |  |
| yaw drift over 12 s | deg | 0.928 | 1.139 |  |  |  |  |  |  |  |  |
| check: absolute difference, finite-difference vs v, steady-state mean (world xy) | m/s | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 |  |  |
| fell | | no | no | no | no | no | no | no | no | no | no |

### Executor-like sequence: achieved vs implied per complete segment

| Segment type | Quantity | Implied | E: n, mean ± SD [min, max] | E ratio | T: n, mean ± SD [min, max] | T ratio |
|---|---|---|---|---|---|---|
| forward | displacement along start heading (m) | 0.75 | 9, 0.610 ± 0.013 [0.590, 0.625] | 0.813 | 9, 0.610 ± 0.011 [0.599, 0.629] | 0.814 |
| forward | same, at end of following zero segment (m) | 0.75 | 9, 0.651 ± 0.013 [0.630, 0.667] | 0.868 | 9, 0.651 ± 0.010 [0.641, 0.669] | 0.869 |
| forward | lateral displacement (m) | 0 | 9, 0.001 ± 0.036 [-0.043, 0.080] | n/a | 9, -0.010 ± 0.036 [-0.048, 0.074] | n/a |
| forward | yaw change (deg) | 0 | 9, 1.555 ± 1.103 [0.631, 4.307] | n/a | 9, 0.542 ± 1.976 [-3.408, 3.976] | n/a |
| turn_left | yaw change (deg) | 45 | 5, 42.987 ± 0.854 [41.952, 44.036] | 0.955 | 5, 43.952 ± 0.958 [42.867, 44.906] | 0.977 |
| turn_left | same, at end of following zero segment (deg) | 45 | 5, 44.851 ± 0.890 [43.542, 45.737] | 0.997 | 5, 46.087 ± 1.234 [44.599, 47.561] | 1.024 |
| turn_left | planar displacement (m) | 0 | 5, 0.039 ± 0.035 [0.007, 0.091] | n/a | 5, 0.037 ± 0.035 [0.013, 0.095] | n/a |
| turn_right | yaw change (deg) | -45 | 4, -42.966 ± 0.531 [-43.467, -42.343] | 0.955 | 4, -42.960 ± 1.627 [-44.523, -41.165] | 0.955 |
| turn_right | same, at end of following zero segment (deg) | -45 | 4, -43.318 ± 0.623 [-44.060, -42.562] | 0.963 | 4, -42.402 ± 1.427 [-43.704, -41.059] | 0.942 |
| turn_right | planar displacement (m) | 0 | 4, 0.028 ± 0.017 [0.007, 0.046] | n/a | 4, 0.039 ± 0.008 [0.032, 0.049] | n/a |
| zero after forward (0.16 s) | residual planar displacement (m) | 0 | 9, 0.041 ± 0.001 [0.039, 0.043] |  | 9, 0.041 ± 0.002 [0.038, 0.044] |  |
| zero after forward (0.16 s) | residual absolute yaw change (deg) | 0 | 9, 0.361 ± 0.239 [0.088, 0.863] |  | 9, 0.315 ± 0.306 [0.003, 0.970] |  |
| zero after forward (0.16 s) | mean forward speed (m/s) | 0 | 9, 0.240 ± 0.012 [0.213, 0.251] |  | 9, 0.240 ± 0.009 [0.228, 0.256] |  |
| zero after forward (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 9, 0.135 ± 0.034 [0.096, 0.205] |  | 9, 0.178 ± 0.037 [0.109, 0.230] |  |
| zero after turn left (0.16 s) | residual planar displacement (m) | 0 | 5, 0.014 ± 0.001 [0.013, 0.016] |  | 5, 0.014 ± 0.002 [0.012, 0.016] |  |
| zero after turn left (0.16 s) | residual absolute yaw change (deg) | 0 | 5, 2.062 ± 1.047 [0.495, 3.297] |  | 5, 2.135 ± 1.045 [0.894, 3.637] |  |
| zero after turn left (0.16 s) | mean forward speed (m/s) | 0 | 5, -0.088 ± 0.025 [-0.115, -0.049] |  | 5, -0.094 ± 0.025 [-0.126, -0.063] |  |
| zero after turn left (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 5, 0.233 ± 0.058 [0.183, 0.317] |  | 5, 0.256 ± 0.085 [0.172, 0.352] |  |
| zero after turn right (0.16 s) | residual planar displacement (m) | 0 | 4, 0.015 ± 0.003 [0.012, 0.019] |  | 4, 0.014 ± 0.005 [0.007, 0.018] |  |
| zero after turn right (0.16 s) | residual absolute yaw change (deg) | 0 | 4, 0.508 ± 0.809 [0.003, 1.717] |  | 4, 0.615 ± 0.370 [0.114, 0.955] |  |
| zero after turn right (0.16 s) | mean forward speed (m/s) | 0 | 4, -0.118 ± 0.016 [-0.133, -0.099] |  | 4, -0.106 ± 0.029 [-0.126, -0.063] |  |
| zero after turn right (0.16 s) | mean absolute yaw rate (rad/s) | 0 | 4, 0.177 ± 0.042 [0.135, 0.226] |  | 4, 0.224 ± 0.018 [0.212, 0.250] |  |

RMS error against the step-wise command over the 30 s — E: vx 0.140 m/s, vy 0.085 m/s, wz 0.142 rad/s; T: vx 0.134 m/s, vy 0.078 m/s, wz 0.178 rad/s.

Per-segment values: `tables/sequence_segments.csv`; all numbers incl. per-segment rows: `logs/metrics.json`.

## Figure

`figures/speed_yawrate_E_vs_T.png` (and `.pdf`): forward speed in the forward trial, yaw rate in the two turn trials
(top row: every 0.02-s step thin, 0.66-s moving average bold, steady-state window shaded), and forward speed and yaw
rate through the 30-s executor-like sequence (every step, unsmoothed); E blue, T orange, command gray; repeat 1.

## Facts established along the way

1. **Velocity limits in the evaluated simulator are the URDF's, not the config's.** In this Isaac Lab fork implicit
   actuators write stiffness, damping, effort limit, armature and friction into PhysX, but not the velocity limit
   (`assets/articulation/articulation.py`, implicit branch of `_process_actuators_cfg`). Read back from PhysX for both
   robots: hip pitch 7.1, hip roll 12.9, hip yaw 18.1, knee 12.5, ankles 18.1 rad/s; T's arms and head 18.0 rad/s. The
   configs say 14.66 / 12.57 / 17.59 / 12.57 / 17.59 (legs), 33.51 (arms), 7.85 (head), as in training, where Isaac
   Lab 2.x writes them. This applies to the evaluation as well. In these trials the left hip pitch reaches the 7.1-rad/s
   limit on 17 of 600 steps of `forward` (E and T alike) and 11 (E) / 10 (T) of 1500 steps of `sequence`, never in
   `stand` or the turns; no other leg joint reaches 99 % of its limit (`tables/velocity_limits.md`).
2. Gains, effort limits and armatures read back from PhysX equal the configured values for both robots; total mass
   19.666 kg for both; E's Trunk (merged with head and arms) 10.66 kg with its centre of mass at
   (−0.0010, −0.0005, 0.1196) m in the Trunk frame, T's Trunk 6.5 kg at (−0.0043, −0.0007, 0.0657) m.
3. The two base-velocity definitions agree in the means: Trunk origin vs Isaac Lab's root centre of mass differ by at most
   0.0034 m/s in any steady-state mean of forward or lateral speed; the finite-difference check of the Trunk position
   agrees with the Trunk-origin velocity to ≤ 0.0015 m/s.
4. Robot T's head and arms stayed within 0.069 rad of the training pose in every trial (largest: head pitch while
   walking forward; mean deviation 0.008 rad).
5. In `forward` the largest tilt is at the first steps from standstill (t = 0.40 s E, 0.38 s T). The left ankle pitch's
   peak velocity in every run (14.0 rad/s E, 12.6 rad/s T) is the landing during the reset warm-up (spawn 0.729 m);
   while walking, the right ankle pitch reaches 11.4–13.0 rad/s (limit 18.1).
6. Walking forward, both robots yaw slowly to the left: +8.3° (E) and +5.9° (T) over 12 s (steady-state yaw rate +0.017
   and +0.025 rad/s). Turning in place, both move slightly backward in left turns (−0.04 m/s) and sideways in right turns
   (−0.03 m/s), ending 0.08–0.15 m from the start after about one full turn.
7. No env termination (fall) and no wrapper hopeless-stall flag in any run.

## Deviations from PLAN.md

The protocol, robot models, commands, validation criterion and metric definitions were followed as written. Departures
and additions:

1. **Additional metrics** beyond the PLAN's list (nothing in the PLAN was removed or redefined): standard deviations of
   forward speed and yaw rate in steady state; yaw change and planar displacement over the command phase; time of the
   maximum tilt; robot T's head/arm deviation from the training pose; leg-joint velocities against the PhysX velocity
   limits (`tables/velocity_limits.md`).
2. **Figure:** the top row adds a 0.66-s moving average (about one gait cycle) on top of the unsmoothed per-step traces,
   because the 3-Hz stepping oscillation hides the comparison otherwise. The dataviz palette validator could not be run
   (no Node.js on the workstation); the figure uses slots 1–2 of the skill's reference palette, which the skill documents
   as validated.
3. **Implementation details that do not change the protocol:** the PhysX and ground-USD read-backs are taken after the
   last recorded step of each run (they are read-only; this keeps the validation path identical to the reference run up
   to the last snapshot); the validation run passes a record-only configuration hook (stores the robot configuration
   for the comparison above and changes nothing), because the scene rewrites the robot's `prim_path` in place.
4. A bug found while analysing (a duplicate key dropped `ratio_of_mean` for turn segments from `logs/metrics.json`) was
   fixed before the final analysis; the tables were byte-identical before and after.

## Not determined

1. How the two robots walk on the Matterport floors themselves (mesh collider, thresholds, slopes, clutter): only the flat
   plane was tested, by design.
2. The effect of the training-time actuator delay (random 0–3 steps): modelled neither here nor in the evaluation.
3. How `model_14498` tracks these commands in its training simulator (Isaac Sim 5.0 / Isaac Lab 2.x with the
   `booster_train` actuators and the configured velocity limits written to PhysX): not run, so how much Isaac Sim 4.1
   itself changes tracking relative to training is not determined; this check compares two robot models within Isaac
   Sim 4.1 only.
4. Whether the evaluator's closed-loop executor, which holds a command until the displacement or heading is reached and
   then settles for 8 steps, gets the same per-command result: the sequence uses the nominal open-loop durations.
5. The cause of the left/right asymmetries seen in both robots (only the left hip pitch reaches its velocity limit; left
   turns run at 1.01–1.02 × command, right turns at 0.94–0.97 ×; leftward yaw while walking forward).
6. Behaviour with a gait-phase initialisation other than 0.0.

## GPU use

`logs/gpu_time.tsv`: 21 GPU processes (1 validation, 20 trials), all return code 0, **1508 s (25.1 min) in total** of
the 1.5-h budget. Before GPU use, `GPU_LOCK` was held by the frame-replay experiment (`frame_replay`) with no compute
process running; checks every 10 minutes (`logs/gpu_lock_wait.log`: 23:10, 23:20, 23:30) until the lock was gone; the
lock was created atomically with `walking_check` at 23:40:02 and is deleted after this push. `run_gpu.sh` verified the
lock and an empty GPU before every process.

## Files

| Path | Content |
|---|---|
| `PLAN.md` | the plan, unchanged (commit `52ec62c`) |
| `scripts/walk_check.py` | harness (validation and trials); imports the Oct 1 `sim_setup.py` = the evaluator's own lines |
| `scripts/configs/k1_walkcheck_plane_cfg.py`, `k1_walkcheck_plane_T_cfg.py`, `k1_walkcheck_T_robot_cfg.py` | copied configs (plane E, plane T, robot T) |
| `scripts/wrappers_v3_T.py` | copy of `VLNEnvWrapperV3` for the 22-joint model |
| `scripts/diffs/` | `diff -u` of each copy against the file it was copied from (evaluated config, evaluated wrapper, Oct 1 corrected robot config) |
| `scripts/run_gpu.sh`, `batch_runs.sh`, `wait_gpu_lock.sh` | GPU runner (lock check, logging), batch order, lock wait |
| `scripts/repeatability.py`, `metrics.py`, `make_figure.py` | Step 2 comparison, Step 3 metrics and figure (CPU) |
| `scripts/secret_scan.sh` | secret scan run on every staged file |
| `logs/` | `gpu_time.tsv`, `gpu_lock_wait.log`, `batch_trials.out`, one stdout log per GPU process, `validation/`, `states/` (per-step states and metadata of all 20 runs, `<robot>_<trial>_r<repeat>.npz/.json`), `repeatability.json`, `metrics.json`, `plan_commit.txt` |
| `tables/` | `summary_by_robot_trial.md`, `metrics_all.md/.csv`, `sequence_summary.md`, `sequence_segments.csv`, `velocity_limits.md`, `repeatability.md` |
| `figures/` | `speed_yawrate_E_vs_T.png/.pdf` |

Nothing from the sweep was edited, nothing was written to `eval_results/`, no model server was started. No Matterport
scene, dataset file, model weight or checkpoint, robot mesh, conda environment or session transcript is included.
