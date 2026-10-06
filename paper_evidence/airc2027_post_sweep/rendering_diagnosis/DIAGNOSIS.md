# DIAGNOSIS — gray robot-camera frames and the floating K1 head and arms

Done 2026-10-01 on the workstation that ran the NaVILA–K1 sweep (`~/Projects/k1_research`), driver 580.173.02,
Isaac Sim 4.1 / Isaac Lab fork 4d558ec (unmodified for everything "as evaluated"). Every script is in `scripts/`,
every number below comes from a file in `diagnosis/` (paths are relative to this folder). Nothing in the sweep tree
was edited; no navigation episode or sweep was run; nothing was written to `eval_results/`.

## Answers in one paragraph

**The gray is the robot's own geometry, drawn in the wrong place.** In the evaluated K1 model the head, the arms and
the chest logo are separate visual meshes inside the Trunk's instanceable `visuals` prototype (URDF fixed joints merged
by the converter). In this Isaac Sim 4.1 + Fabric render path those meshes are drawn at the Trunk's **spawn pose**
(the USD pose: episode `start_position` + 0.55 m, which physics never writes back under Fabric) instead of at the
Trunk's current pose. The torso shell (mesh_0, the first mesh of the Trunk prototype) and the leg links (one mesh per
prototype) follow their bodies; the other eleven Trunk meshes (logo, head, arms) stay behind. Because
the evaluator spawns the Trunk 0.55 m above `start_position`, and `start_position` itself lies above the floor by an
episode-dependent amount (0.178 m in episode 0), the settled Trunk ends up 0.09–0.24 m below the spawn height and the
head is left floating that far above the torso, in the T-pose of the merged zero joint angles. A camera mounted 0.42 or
0.57 m above the Trunk then sits inside, or 1–2 cm from, the leftover head: the uniform RGB ≈ 53 is the unlit default
material of that head seen from inside or from below. It clears as soon as the robot walks away from its start pose. The
collision shapes and masses are correct and move with the Trunk; only rendering is affected. Rendering repeats across
fresh processes to ±1 level for >99.9 % of pixels and physics is bit-identical, so neither explains the three-episode
lead; two of its three cases are near-threshold measurements on real scenery.

---

## Step 1 — the existing videos (CPU)

Frames were decoded with ffmpeg (`scripts/step1_extract_frames.py`); crops are PNG in `diagnosis/videos/`.

### 1.1 Does the floating head-and-arms group stay with the robot? — **No. It stays where the robot started.**
Videos h110_200 `output_0`, h095_200 `output_158`, h078_200 `output_0`, frames 0, 1, 5, 6, 10, 20, 50, 100; both halves
in `diagnosis/videos/1_1_group_motion/`, sheets `1_1_*_sheet.png`, gray fractions in `1_1_group_motion.csv`.
- The chase camera is rigidly mounted on the Trunk, and the white torso and legs stay at the same pixel position in every
  frame (e.g. h110 rec 0 frames 0 and 100). The gray head and T-posed arms do not: they change size and position
  relative to the torso from frame to frame (frames 10 and 20 show them nearer and larger as the robot walks toward
  them), and in h110 rec 0 frame 100 the robot's own camera sees a K1 head face-on and an arm a short distance in front
  of it (`1_1_group_motion/h110_200_rec0_f0100_robotcam.png`). Step 3 confirms quantitatively that the group is drawn at
  the spawn pose.
- The same group is visible in h078 (0.25 m), where the robot camera is never gray at the start: the geometry is
  displaced at every height; only the camera height decides whether the camera is inside it.

### 1.2 The lead — **two of three cases are near-threshold measurements of real scenery; one is a whole-patch shading difference on the robot's leftover geometry. Chase-camera halves are identical between runs.**
Frames in `diagnosis/videos/1_2_lead/`, gray fractions in `1_2_lead.csv` / `1_2_lead_context.csv`, side-by-side sheets
`1_2_rec217_f41_compare.png`, `1_2_rec{39,40,41}_f11_compare.png`, gray-mask overlays `1_2_gray_masks_with_vs_without.png`.

| Record (dataset idx) | Video frame (history step) | stretchA | stretchB | h078 | June full_14498 |
|---|---|---|---|---|---|
| 217 (148) | 40 (175) | 0.4657 | 0.4658 | 0.4658 | — |
| 217 (148) | **41–45 (200)** | **0.1440** | **0.1441** | **0.0616** | — |
| 217 (148) | 46 (225) | 0.0003 | 0.0003 | 0.0003 | — |
| 39 (21) | 10 (25) | 0.0023 | 0.0022 | 0.0023 | 0.0022 |
| 39 (21) | **11–15 (50)** | **0.1277** | **0.1279** | **0.1270** | **0.0662** |
| 41 (23) | **11–15 (50)** | **0.1278** | **0.1281** | **0.1283** | **0.0651** |
| 40 (22), control | 11–15 (50) | 0.1318 | 0.1316 | 0.1316 | 0.1321 |
| 39, 40, 41 | 16 (75) | ≈0.010 | ≈0.010 | ≈0.010 | ≈0.010 |

- **Records 39 and 41: near the 10 % threshold, not a patch present/absent.** The "gray" here is the dark underside of a
  kitchen counter (Matterport geometry) seen while the robot pitches forward under it; the 53 ± 6 mask is scattered over
  that textured surface. In June the same surface rendered darker (mean 43 vs 49–50.5 inside the sweep's mask), which
  pushes half of it out of the ±6 band (6.5 % vs 12.8 %). The geometry is identical: edge maps agree as well as two sweep
  runs agree with each other (0.950 vs 0.952), and the June−sweep difference is a smooth shading offset concentrated on the
  counter (`1_2_rec39_f11_June_minus_stretchA_x4.png`). Within June, record 40 (same view) rendered like the sweep.
- **Record 217: a whole patch with the same outline in all three runs, shaded differently.** At step 200 the robot is
  turning in place next to its start pose and its camera passes through the leftover K1 head/arm geometry (step 150:
  98.6 % gray in all three runs). In stretchA/B the region is uniform RGB ≈ 55 (8.4 % of the image); in h078 the same
  region renders RGB ≈ 126 with darker rectangles (the lit default-material surface with its lettering), so only 6.2 % is
  counted. Outline identical; shading differs.
- **Chase-camera halves at those frames are identical between runs** up to H.264 noise: pairwise mean |diff| 1.15–1.86,
  99th percentile 9–13 (`1_2_pairwise_halves_mae_p99.json`; the same pairs at unaffected frames give the same numbers).

### 1.3 Default height (0.25 m): what is the mid-episode gray? — **one is real scenery, one is the robot's own leftover geometry.**
From `occlusion_scan_full_length/stretchA_300.csv`: record 13 (frames 116–200) and record 198 (frames 86–110);
`diagnosis/videos/1_3_*`.
- **Record 13**: the robot walks under a table; the "gray" (11.7–14.8 %) is the shadowed, textured underside of the table
  top (Matterport geometry, scattered mask, `1_3_gray_masks.png` left).
- **Record 198**: the robot camera looks at the K1 head (eye socket visible) and a K1 forearm/hand, centimetres in front
  of the lens next to a tiled wall (`1_3_rec198_sheet.png`); gray 10.9–15.0 %. The robot is near its start pose, i.e.
  near where the head and arms were left.

## Step 2 — the robot models (CPU)

### 2.1 The two URDFs (`diagnosis/models/urdf_compare.md`, `urdf_merged_mass_properties.json`)
`diff K1_22dof.urdf K1_locomotion.urdf`: the only differences are the 10 head/arm joint `type` attributes
(revolute → fixed) and two XML comment delimiters. Origins, axes, limits, masses, inertias, visuals and collisions are
byte-identical.

| Joint | Parent → child | K1_locomotion (evaluated) | K1_22dof (training) | origin xyz (m) | rpy | training pose (rad) |
|---|---|---|---|---|---|---|
| AAHead_yaw | Trunk → Head_1 | fixed | revolute | 0.0056 0 0.2149 | 0 0 0 | 0 |
| Head_pitch | Head_1 → Head_2 | fixed | revolute | 0 0 0.033 | 0 0 0 | 0 |
| ALeft_Shoulder_Pitch | Trunk → Left_Arm_1 | fixed | revolute | 0 0.077 0.1845 | 0 0 0 | 0.2 |
| Left_Shoulder_Roll | Left_Arm_1 → Left_Arm_2 | fixed | revolute | 0.0025 0.068 −0.0135 | 0 0 0 | −1.25 |
| Left_Elbow_Pitch | Left_Arm_2 → Left_Arm_3 | fixed | revolute | 0 0.044428 0 | 0 0 0 | 0 |
| Left_Elbow_Yaw | Left_Arm_3 → left_hand_link | fixed | revolute | 0 0.1215 0 | 0 0 0 | −0.5 |
| ARight_Shoulder_Pitch | Trunk → Right_Arm_1 | fixed | revolute | 0 −0.077 0.1845 | 0 0 0 | 0.2 |
| Right_Shoulder_Roll | Right_Arm_1 → Right_Arm_2 | fixed | revolute | 0.0025 −0.068 −0.0135 | 0 0 0 | +1.25 |
| Right_Elbow_Pitch | Right_Arm_2 → Right_Arm_3 | fixed | revolute | 0 −0.044428 0 | 0 0 0 | 0 |
| Right_Elbow_Yaw | Right_Arm_3 → right_hand_link | fixed | revolute | 0 −0.1215 0 | 0 0 0 | +0.5 |

Arm pose produced by the fixed joints: every angle 0, i.e. **arms straight out sideways (T-pose)**: left hand frame at
(0.0025, 0.3109, 0.171) m in the Trunk frame (right mirrored), head frame (Head_2) at (0.0056, 0, 0.2479) m.
Masses (both files): Trunk 6.5, Head_1 0.3, Head_2 0.7, each arm chain 0.5 + 0.09 + 0.8 + 0.19 = 1.58, legs 2 × 4.503;
**total 19.666 kg**. A rigid merge of Trunk + head + arms at the zero pose: 10.660 kg, COM (−0.00098, −0.00048, 0.11961),
inertia about the COM diag (0.37997, 0.16092, 0.23678) kg·m² (Trunk link alone: diag 0.0962, 0.0895, 0.0202).

### 2.2 The converted USD
- **The sweep-time conversion no longer exists.** Isaac Lab 1.1 (`asset_converter_base.py:64-67`) writes every
  conversion to a new `/tmp/IsaacLab/usd_<date>_<time>_<random>` folder per process (`usd_dir=None`), so nothing is
  cached across processes, and `/tmp` was wiped by the 2026-09-20 reboot. A search of the root filesystem found no
  `K1_locomotion*.usd`. The hierarchy below is from conversions made today by the unmodified converter with the
  unmodified config (`diagnosis/sim/explore_cz057/`; the converter's `config.yaml` is generated from the unmodified K1 spawn config:
  `make_instanceable: true`, `merge_fixed_joints: true`, `import_inertia_tensor: true`, …).
- `K1_locomotion.usd` is a **flattened** stage ("Generated from Composed Stage"): every link's `visuals` and `collisions`
  are instanceable prims that reference in-file prototypes `/Flattened_Prototype_<n>`; the side file
  `Props/instanceable_meshes.usd` is written but not referenced (the converter also logs "Failed to open layer
  …/Props/instanceable_meshes.usd").
- Full hierarchy with local and USD-world transforms: `diagnosis/sim/explore_cz057/usd_hierarchy_after_reset.txt`;
  Fabric attributes: `fabric_attrs_after_reset.json`.

| Visual mesh | URDF source | Parent prim | Local transform (translate; orient = identity) | Instanceable | Material binding |
|---|---|---|---|---|---|
| mesh_0 | Trunk.STL | `Robot/Trunk/visuals` (instanceable, → /Flattened_Prototype_1) | (0, 0, 0) | instance proxy | material_C0C0C0 |
| mesh_1 | K1logo.STL (Trunk) | same | (0, 0, 0) | instance proxy | none |
| mesh_2 | Head_1.STL | same | (0.0056, 0, 0.2149) | instance proxy | none |
| mesh_3 | Head_2.STL | same | (0.0056, 0, 0.2479) | instance proxy | none |
| mesh_4 / mesh_8 | Left/Right_Arm_1 | same | (0, ±0.077, 0.1845) | instance proxy | none |
| mesh_5 / mesh_9 | Left/Right_Arm_2 | same | (0.0025, ±0.145, 0.171) | instance proxy | none |
| mesh_6 / mesh_10 | Left/Right_Arm_3 | same | (0.0025, ±0.189428, 0.171) | instance proxy | none |
| mesh_7 / mesh_11 | Left/Right_Arm_4 (hand) | same | (0.0025, ±0.310928, 0.171) | instance proxy | none |

The local offsets are the correctly composed fixed-joint origins. Empty Xforms `Trunk/Head_1`, `Trunk/Left_Arm_1`,
`Trunk/Right_Arm_1` are also created (no geometry). **Collision shapes of the Trunk** (`Robot/Trunk/collisions`,
instanceable, all `purpose=guide`): mesh_0 box 0.12×0.18×0.2 at (0,0,0.1); mesh_1 cylinder r 0.05 at (0,0,−0.06);
mesh_2 Head_1 mesh at (0.0056,0,0.2149); mesh_3 Head_2 mesh at (0.0056,0,0.2479); mesh_4/mesh_7 Arm_2 cylinders at
(−0.0045,±0.145,0.171); mesh_5/mesh_8 Arm_3 meshes at (0.0025,±0.189428,0.171); mesh_6/mesh_9 hand meshes at
(0.0025,±0.310928,0.171). Left/Right_Arm_1 have no collision in the URDF.

## Step 3 — reproduce and explain (GPU)

`scripts/sim_setup.py` executes the evaluator's own lines (verbatim copy `scripts/navila_eval_v3_COPY.py`, SHA-256
9b0107…, identical to the sweep file) from `main()` to `env.reset()`: task `k1_matterport_vision`,
`reset_start_pos_rot`, the `--cam_z` override, `VLNEnvWrapperV3` with its 1-s reset warm-up, `model_14498.pt`, no model
server. Dataset index 0 (record 0, zsNo4HB9uLZ). Probe: `scripts/sim_step3.py`; outputs `diagnosis/sim/step3_cz{057,042,025}/`.

### 3.1 Reproduce (cam_z 0.57) — **reproduced exactly.**
The reset robot-camera frame (`step3_cz057/A_reset_robotcam.png`, lossless) is 100.00 % gray; with the evaluator's
text overlay drawn on it, it matches frame 0 of h110 `output_0.mp4` at the H.264 floor (gray 0.9911 vs 0.9910, mean |diff|
1.03, p99 2; `diagnosis/step5a_raw/step5a_vs_video_frame0.json`). Its depth is a constant **0.0151 m** over the whole image.
World poses after the reset warm-up (all heights give the same Trunk pose; full tables in `diagnosis/tables.md`):

| Item | World position (m) |
|---|---|
| Trunk (Isaac Lab = Fabric `_worldPosition`) | (15.1070, 4.4618, 0.4899), yaw 150.21° |
| Trunk USD pose = spawn pose (`start_position` + 0.55; never updated under Fabric) | (15.0686, 4.4848, 0.7216), yaw 150.00° |
| Robot camera (Isaac Lab `pos_w` = Trunk × offset (0.10, 0, 0.57)) | (15.0210, 4.5077, 1.0603) |
| Head_2 (mesh_3): USD world / Trunk(now) × local | (15.0637, 4.4876, 0.9695) / (15.1025, 4.4629, 0.7378) |
| Head_1 (mesh_2) | (15.0637, 4.4876, 0.9365) / (15.1025, 4.4631, 0.7048) |
| Left hand (mesh_7) | (14.9110, 4.2168, 0.8926) / (14.9506, 4.1920, 0.6593) |
| Right hand (mesh_11) | (15.2219, 4.7553, 0.8926) / (15.2596, 4.7317, 0.6625) |

Fabric holds `_worldPosition/_worldOrientation` only for the 13 rigid bodies; the visual meshes carry only their local
xform ops. **Which of the two the renderer uses was measured, not assumed**: the pixels that change when only the ten
head/arm meshes are made invisible (removal run, physics bit-identical) were compared with the head/arm STL meshes
rasterised under both hypotheses (`scripts/step3_verify.py`, overlays `diagnosis/sim/verify_cz057/*_overlay.png`:
red = spawn pose, cyan = attached to the current Trunk):

| cam_z 0.57 | IoU attached-to-Trunk | IoU spawn pose |
|---|---|---|
| robot camera, reset (footprint 916 097 px) | 0.000 | **0.994** |
| chase camera, reset | 0.000 | **0.712** |
| chase camera, steps 25 / 50 / 75 / 100 of the walk | 0.04 / 0.17 / 0.06 / 0.07 | **0.71 / 0.51 / 0.55 / 0.60** |

Predicted depth of the inside of the spawn-pose Head_2 front plate from the robot camera: 0.0163 m (median over a 20-px
grid; rendered 0.0151 m, median error 1.1 mm); the attached hypothesis predicts no hit at all. The robot camera itself
renders from the current Trunk pose. The camera sits 1.4 cm behind the front face of the left-behind Head_2.

### 3.2 Movement — **the Trunk, legs and cameras move; the head and arm visuals do not.**
Constant command (0.5, 0, 0) through the walking policy for 2 s (100 steps) right after reset: the Trunk moved 0.26 m
(policy start-up; yaw 150.2° → 144.0°), the head/arm visuals stayed at the spawn pose in every frame (table above;
`step3_cz057/D_walk_step*_{robotcam,chasecam}.png`). The robot camera cleared by step 25 (gray 0.0004).

### 3.3 Removal test — **the gray disappears when (only) the head meshes are hidden.**
The visibility of chosen visual meshes was authored in a copy of the converted USD before load (`--hide_meshes`;
runtime USD edits were tried first and do not reach the renderer, see `diagnosis/sim/_invalid_removal_props_layer/` and
`step3_v1_cz057/`). Everything else identical; Trunk poses bit-identical to the as-evaluated run.

| Reset frame, dataset index 0 | cam_z | robot-cam gray |
|---|---|---|
| as evaluated | 0.57 | 1.0000 (depth 0.0151 m) |
| head + arm meshes hidden | 0.57 | **0.0009** |
| arm meshes hidden, head kept | 0.57 | 1.0000 |
| head meshes hidden, arms kept | 0.57 | **0.0010** |
| as evaluated / head+arms hidden | 0.42 | 0.0007 / 0.0006 |
| as evaluated / logo (mesh_1) hidden | 0.42 | lettering present / **lettering gone** (`diagnosis/sim/logo_removal_cz042_compare.png`) |
| as evaluated / head+arms hidden | 0.25 | 0.0016 / 0.0015 |

### 3.4 Other heights
- **0.42**: the reset frame is not gray (0.0007; the sweep's frame 0 at h095 is never gray). The lighter **glyph-like
  shapes are the "BOOSTER" lettering of the K1 logo mesh** (mesh_1, also left at the spawn pose) 1.6–2.1 cm in front of
  and just below the lens; hiding mesh_1 removes them. At the end of the 4-s warm-up the lettering crosses the image
  mirrored (`images/as_evaluated/robot_camera/idx0_rec0_camz0.42_warmup_last.png`). Lit default material renders at
  RGB ≈ 100–130, outside the 53 ± 6 band, so the measure does not count it.
- **0.25**: no gray at reset or while walking in episode 0; the chase camera shows the same floating group.
- **Why only some h095/h110 episodes**: the gray needs the camera to be inside, or within a few cm of, the stale head,
  but beyond the camera's 1-cm near clipping plane (`clipping_range=(0.01, 1e6)`). That depends on how far the Trunk
  settles below its spawn height (0.09–0.24 m in the episodes checked, set by `start_position` z above the floor) and on
  the small drift during the warm-ups. A ray cast of the stale meshes with the 1-cm near plane reproduces the rendered
  gray (`diagnosis/stale_geometry_check_all.json`, table in `diagnosis/tables.md`):

| Episode | cam_z | frame | rendered gray (sim) | predicted coverage | sweep video |
|---|---|---|---|---|---|
| idx 0 (rec 0) | 0.57 | reset / end of warm-up | 1.000 / 1.000 | 1.000 / 1.000 (Head_2, back faces) | frame 0: 0.991 |
| idx 75 (rec 111) | 0.42 | reset / end of warm-up | 0.002 / 0.683 | 0.000 / 0.700 (Head_1) | frame 1: 0.683 |
| idx 81 (rec 120) | 0.42 | reset / end of warm-up | 0.000 / 0.879 | 0.000 / 0.880 (Head_1) | frame 1: 0.873 |
| idx 72 (rec 102) | 0.42 | reset / end of warm-up | 0.005 / 0.006 | 0.000 / 0.000 (front plate 8 mm away, inside the near plane) | frame 1: 0.005 |
| idx 0 (rec 0) | 0.07, 0.25, 0.77, 0.97 | all | ≤ 0.002 | 0.000 | ≤ 0.0015 |

The first main-loop history frame (`main000`, fixed forward command) also matches the sweep's video frame 1 for
episode 0 at 0.57 (gray 1.000 vs 0.991, MAE 1.01) and 0.42 (0.0014 vs 0.0014). Episodes 75/81 were not compared
frame-by-frame with their videos because the sweep's first command for them is not known.

### 3.5 Real heights (dataset index 0)
Floor under the Trunk (ray cast onto the Matterport mesh): z = −0.0069; `start_position` z = 0.1716, i.e. **0.178 m above
the floor**. The Trunk is spawned 0.55 m above `start_position` (0.73 m above the floor), drops during the 1-s reset
warm-up and settles **0.497 m above the floor (0.318 m above `start_position`)**; during the 2-s walk it rides
0.490–0.553 m (mean 0.524) above the floor.

| cam_z | nominal (0.53 + cam_z) | camera above floor after reset | at end of 4-s warm-up | walking, mean (range) | camera above `start_position` z after reset |
|---|---|---|---|---|---|
| 0.07 | 0.60 | 0.567 | 0.562 | 0.594 (0.560–0.623) | 0.389 |
| 0.25 | 0.78 | 0.747 | 0.742 | 0.774 (0.740–0.803) | 0.569 |
| 0.42 | 0.95 | 0.917 | 0.912 | 0.944 (0.910–0.973) | 0.739 |
| 0.57 | 1.10 | 1.067 | 1.062 | 1.094 (1.060–1.123) | 0.889 |
| 0.77 | 1.30 | 1.267 | 1.262 | 1.294 (1.260–1.323) | 1.089 |
| 0.97 | 1.50 | 1.467 | 1.462 | 1.494 (1.460–1.523) | 1.289 |

Other episodes settle differently relative to `start_position` (e.g. Trunk − spawn = −0.086 m in idx 72, −0.175 in idx 75,
−0.190 in idx 7, −0.232 in idx 0), so heights quoted relative to `start_position` vary by episode; heights above the floor
were measured for episode 0 only (**not determined** for the others).

### 3.6 Physics — **the mass properties match the URDF exactly, and the collision shapes move with the Trunk.**
PhysX tensor view (`step3_cz057/step3_results.json` → `E_physics`): total 19.666 kg (URDF 19.666); Trunk 10.660 kg (=
rigid merge, Step 2.1); Trunk COM (−0.00098, −0.00048, 0.11961) and inertia [[0.37997, −0.00005, −0.00578], [−0.00005,
0.16092, −0.00068], [−0.00578, −0.00068, 0.23678]] — identical to five decimals with the composite computed from the URDF
at the zero (T-) pose. Leg-link masses match the URDF. (The training model has the same total mass but the head and arms
are separate actuated bodies with the arms down, so the evaluated Trunk's inertia differs from training: diag 0.380 vs
the Trunk's own 0.096 about x.)
Collision check (`diagnosis/sim/physics_cpu_diag_cz025/`; diagnostic run on the CPU PhysX pipeline with scene queries
enabled, because Isaac Lab disables scene queries by default and the GPU pipeline does not expose the articulation to
them): vertical rays through the head and both hands, after reset and after the 2-s walk, hit
`Robot/Trunk/collisions/mesh_3` (Head_2), `…/mesh_6` and `…/mesh_9` (hands), rigid body `Robot/Trunk`, at the CURRENT
Trunk pose (head top predicted 0.209 m below the ray origin, hit at 0.211 m), and nothing at the spawn pose.

### Mechanism (causal tests, not as evaluated)
- **Material binding is not the cause.** Giving mesh_1…mesh_11 exactly mesh_0's material binding (before load) turns the
  floating parts white but leaves them at the spawn pose; the camera at 0.57 is still inside the head, which now renders
  uniform light gray, so the 53 ± 6 measure reads 0.0000 although the view is still fully blocked
  (`diagnosis/sim/causal_material_binding_compare.png`).
- **Instancing is.** Setting `instanceable = False` on `Trunk/visuals` in a copy of the converted USD before load puts the
  head and arms on the torso and makes them move with it (chase-camera IoU attached / spawn: 0.79 / 0.00 at reset,
  0.65–0.74 / 0.05–0.20 while walking; `diagnosis/sim/verify_deinstanced_cz057/`,
  `diagnosis/sim/causal_deinstanced_compare.png`); the 0.57 reset frame is clear (gray 0.0009). The arms are still in the
  T-pose, which comes from the merged URDF, not from rendering. Implication for a fix: spawn the K1 with
  `make_instanceable=False` (or with `K1_22dof.urdf` and actuated arms, as in training). Why the Fabric/RTX path in Isaac
  Sim 4.1 leaves all but the first mesh of an instanced prototype at the authored pose is **not determined**.
- Earlier runtime attempts (de-instancing after load in `step3_v1_cz057`: robot camera clear but group still at the spawn
  pose in the chase camera) only changed how the mesh was drawn after repopulation; they are kept for the record.

## Step 4 — does rendering repeat? (GPU)

`scripts/sim_capture.py`, fresh processes, as evaluated; `diagnosis/repeatability/`.

### 4.1 Pixel diff across processes (dataset index 0) — **physics bit-identical; rendering repeats to ±1 except for a few pixels.**

| cam_z | frame | max abs diff | mean abs diff | share of channels differing | Trunk/root and all body states bit-identical |
|---|---|---|---|---|---|
| 0.25 | initial | 10 | 0.159 | 15.7 % | yes |
| 0.25 | after 50 steps of (0.5, 0, 0) | 59 | 0.125 | 12.4 % | yes |
| 0.57 | initial (all gray) | 1 | 0.193 | 19.3 % | yes |
| 0.57 | after 50 steps | 9 | 0.147 | 14.5 % | yes |

Example (0.25, after 50 steps): 31 % of pixels differ in at least one channel, almost all by 1 level (99th percentile of
the differing pixels: 1, 99.9th: 2); 0.010 % of all pixels differ by more than 2 levels, 0.001 % by more than 10, and 5
pixels by more than 20. The gray fraction is identical in both processes in every case.

### 4.2 Lead replay (dataset index 21, record 39, cam_z 0.25, four fresh processes) — **the gray patch appears in all four.**
Evaluator sequence (reset, 4-s warm-up, main loop) with a fixed forward command instead of the model:

| Main-loop step | gray fraction, processes 1–4 | max / mean abs diff vs process 1 | bit-identical physics |
|---|---|---|---|
| 0 | 0.0020 ×4 | 7–19 / 0.098 | yes |
| 25 | 0.0021 ×4 | 7–13 / 0.108 | yes |
| 50 | 0.1534, 0.1533, 0.1533, 0.1530 | 9 / 0.157 | yes |
| 75 | 0.0101 ×4 | 7–8 / 0.117 | yes |

The fixed forward command reproduces the sweep's trajectory for this start pose: my step-50 frame with the instruction
overlay matches video frame 11 of stretchA and h078 record 39, and the sweep's records 40 and 41 and June record 40, at the
codec floor (MAE 1.34–1.35, p99 8), but differs from June records 39 and 41 (MAE 3.9 / 4.2, p99 26 / 28)
(`diagnosis/repeatability/step4_2_vs_sweep_videos*.json`).

## What explains the three-episode lead
- Not physics (bit-identical across processes) and not ordinary render noise (≤ ±1–2 for 99.9 % of pixels, identical gray
  fractions across processes).
- Records 39 and 41: the measure sits on a real dark surface whose mean shade (≈ 50) is 3 levels from the band edge; in the
  June run (driver 580.159.03, before the 2026-07-24 driver update) that surface rendered ≈ 7 levels darker in two of the
  three episodes, halving the counted area. The same view rendered like the sweep in June's record 40 and in all four
  fresh processes today. What made June render those two episodes darker is **not determined** (candidates not tested:
  the older driver, renderer state at that point of the June run, GPU memory pressure from the co-resident model server).
- Record 217: identical trajectories and identical outlines, but the leftover head geometry close to the lens was shaded
  dark (≈ 55) in stretchA/B and lit (≈ 126) in h078, all with the same driver and config. **Not determined.** Today's runs
  could not replay this trajectory (commands were not logged; they are visible only as overlay text in the chase half of
  the videos, which could be used to reconstruct them).

## Step 6 — does the model repeat?

**Yes — every reply was identical.** The model server was started exactly as `run_powered_benchmark.sh` starts it
(env `navila`, `scripts/vlm_server_bridge.py --model_path …/navila-llama3-8b-8f --port 54321 --load_8bit`, as set by
`VLM_BRIDGE_EXTRA=--load_8bit` in `arms_runner.sh`), in three sessions (started, used, killed; restarted twice).
Payload: the eight frames of `…/smoke14498_lab3090_raw8bit/diag/ep7/frames/tick005/` (the July two-query check's
payload), encoded as the evaluator sends them (lossless PNG of each stored frame, base64) with that episode's
instruction "Walk across the floor and wait the archway. " (dataset index 7). Also, twice per session, the July check's
exact request (384×384 PNG, query "Walk to the goal and stop.").

| Request | Replies | Distinct replies |
|---|---|---|
| evaluator encoding + ep7 instruction | 20 per session × 3 sessions = 60 | 1: "The next action is move forward 75 cm." |
| July 5b exact request | 2 per session × 3 sessions = 6 | 1: "The next action is move forward 75 cm." |

The reply equals the one the June int8 run logged for this tick. Byte-identical request hashes in every send
(`diagnosis/model_repeat/replies.jsonl`, `summary.json`; server logs `logs/step6_server_session{1,2,3}.log`). With
Step 4 (rendering repeats to ±1–2 levels, physics bit-identical), this payload gives no sign of inference
nondeterminism; a single payload whose answer is not close to a decision boundary cannot rule out flips on
knife-edge payloads (in the June replay arbiter, `REPLAY_ARBITER_2026-06-11.md`, replaying the saved JPEG frames through
int8 reproduced the original reply on 361/424 ticks, 85.1 %; that comparison mixes the JPEG-vs-PNG input change with any
nondeterminism, so it does not measure nondeterminism on its own).

## Not determined
1. Why Isaac Sim 4.1's Fabric/RTX path draws every mesh except the first of an instanced visuals prototype at the authored
   (spawn) pose. The cause is pinned to the instanced merged prototype (de-instancing fixes it; material binding does not),
   but not to a specific code path.
2. The renderer-side cause of the June shading difference (records 39/41) and of the record-217 shading difference.
3. The exact USD files converted during the sweep (deleted with `/tmp`); today's conversions use the same code and config.
4. Camera heights above the floor for episodes other than 0; the small robot-camera footprints at 0.42/0.25 that change
   when the head/arms are hidden but match neither projection (68–4 041 px; probably shadows/indirect light from the stale
   meshes).
5. Whether any sweep episode's VLM decisions were changed by the gray frames (not part of this task).

## GPU use
`logs/gpu_time.tsv` lists every GPU process (wall time, return code, command): 49 processes, 37.5 min of wall time in total (including one 337-s path-tracing run that hung after an exception and was killed, and the three Step 6 server sessions, 324 s). The GPU had no other compute process before any run (checked with `nvidia-smi` by `run_gpu.sh` and the Step 6 script).
