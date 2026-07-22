# Media index (verified paths, 2026-07-21)

| Asset | Path | What it shows |
|---|---|---|
| model_8700 circumduction baseline | `~/Desktop/Official_Locomotion_Policy.webm` (also `~/Videos/Screencasts/`) | The original accepted policy's lateral swing-arc (Jun 5) |
| Round-2 winner era | `~/Desktop/Production_Locomotion_Policy.webm` | model_11999-era production gait (Jun 10) |
| model_14498 sim2sim | `~/Desktop/k1_gait_14498_mujoco.mp4` | Production policy in MuJoCo under rated torque clamps (Jun 29) |
| model_14498 reference render | `~/Desktop/gait_14498_reference.mp4` | Side-by-side reference arm (Jul 16) |
| **model_14200 (installed winner)** | `~/Desktop/gait_r5_14200.mp4` (copy: `booster_train/round5_lateral/`) | Final all-gates-pass gait (Jul 19) |
| T1-recipe trot (user-rejected) | `~/Desktop/gait_t1v3_11999.mp4` | The split-stance trot despite best lateral metrics |
| Reward-farming wide gait | `~/Desktop/gait_r6_11999_FAILED_WIDE.mp4` | Positive-band farming via circumduction (Jul 18) |
| Rails near-hit gait ("so close") | `~/Desktop/gait_r5a_20497_FAILED_PROBE.mp4` | Zero-waiver tracking, feet converge at touchdown |
| Crop arrival-blind episodes | `NaVILA-Bench/eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit_crop/videos/output_6.mp4`, `output_7.mp4`, `output_8.mp4` | Crop arm reaches 0.371 m, stops only after overshooting (final 3.7–6.7 m) |
| Stretch success episodes | `.../smoke14498_lab3090_raw8bit/videos/output_5.mp4`, `output_7.mp4`, `output_8.mp4` | The 3/10 stretch conversions |

## ⚠️ NOT ON DISK
- **"Round-4 splay" video: DOES NOT EXIST.** Round-4 (07-13 fine-tune) was killed at the
  eval stage; no render was ever made. Nearest existing "splayed/wide gait" footage is
  `gait_r6_11999_FAILED_WIDE.mp4`. Fix the slide caption or render round-4's checkpoint
  (`k1_navila/2026-07-13_14-08-34` was DELETED after the negative result — cannot render;
  use r6's video instead).
- **"Real-robot volleyball clip": NOT FOUND anywhere on this machine** (searched Desktop,
  Videos, Screencasts, Downloads, Documents, robocup_demo). Also note the project record:
  the K1 has never walked under this stack; if such a clip exists it is on a phone/other
  machine and is not part of this project's evidence chain. Remove from deck or source it
  explicitly.
