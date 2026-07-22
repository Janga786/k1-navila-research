# Reward tables — extracted from the runs' own dumped params (primary source)

## model_14498 run (`booster_train/logs/rsl_rl/k1_navila/2026-06-10_14-30-20/params/env.yaml`) — 19 terms
| Term | Weight | | Term | Weight |
|---|---|---|---|---|
| alive | 0.025 | | feet_yaw_diff | −1.0 |
| track_lin_vel_x | **1.5** | | feet_yaw_mean | −1.0 |
| track_lin_vel_y | **1.5** | | feet_swing_height | **−20.0** |
| track_ang_vel_z | **1.5** | | hip_deviation | −0.1 |
| base_height | −20.0 | | feet_distance | −1.0 |
| orientation | −5.0 | | dof_torques | −1.0e−05 |
| feet_swing | **+3.0** | | dof_acc | −2.5e−07 |
| feet_slide | −0.1 | | dof_pos_limits | −1.0 |
| action_rate | −0.002 | | undesired_contacts | −1.0 |
| termination_penalty | **−200.0** | | | |

## model_14200 run (`.../2026-07-19_10-50-11/params/env.yaml`) — 20 terms
Same 19 as above **+ `feet_distance_band` = −0.4** (penalty-form set-distance band,
[0.14, 0.32] m).

Regenerate either table:
`awk '/^rewards:/{r=1;next} r&&/^[a-z]/{r=0} r&&/^  [a-z_]+:$/{n=$1} r&&/weight:/{print n,$2}' <run>/params/env.yaml`

## Slide-claim resolutions
- **"20 rewards"** → TRUE only for the final model_14200 recipe. model_14498 = **19**.
- **feet_swing 3.0 or −20?** → BOTH exist, different terms: `feet_swing` (contact
  schedule) = **+3.0**; `feet_swing_height` (swing clearance, the circumduction fix)
  = **−20.0**. Do not conflate on a slide.
- **track-linear-velocity 1.5 or 1.0?** → **1.5** (per-axis x and y). 1.0 is Booster
  Gym T1's value, which was tried in the (rejected) T1-recipe line only.
- **fall/termination −200** → confirmed, `termination_penalty: -200.0`.
