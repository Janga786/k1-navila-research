# 14_airc2027_renders — gray-frame diagnosis and Fig. 1 images (AIRC 2027)

Made 2026-10-01 on the sweep workstation (`~/Projects/k1_research`, RTX 3090, driver 580.173.02, Isaac Sim 4.1,
Isaac Lab fork 4d558ec, conda env `vlnce-isaac`). The answers and evidence are in **`DIAGNOSIS.md`**.

| Path | Content |
|---|---|
| `DIAGNOSIS.md` | answers to Steps 1–4 and 6 with the evidence |
| `images/as_evaluated/robot_camera/` | Fig. 1(c): robot-camera frames exactly as the evaluator captured them |
| `images/as_evaluated/third_person/` | Fig. 1(b): path-traced views of the K1 as evaluated (head and arms as they render) |
| `images/corrected/` | the same views with the training robot model (`corrected_*`), plus `studio/` |
| `diagnosis/` | every intermediate result (video crops, simulator frames, poses, overlays, comparisons, tables) |
| `scripts/` | every script used (see the list below); `configs/` the copied corrected robot config |
| `logs/` | `gpu_time.tsv` (every GPU process, wall time, return code, command), run logs |

Not uploaded (size): 253 raw `.npy` arrays over 100 KB (785 MB) — per-frame robot-camera RGB arrays, which the lossless
PNGs duplicate exactly, and per-frame robot-camera depth maps, whose statistics are in the JSON files. They remain on the
workstation in `~/Projects/k1_research/airc2027_renders/`. Small `.npy` files (Trunk/body states used for the bit-identity
checks) are included.

Nothing from the sweep was edited. "As evaluated" images come from the unmodified evaluator code
(`scripts/sim_setup.py` executes the evaluator's own lines; `scripts/navila_eval_v3_COPY.py` is a byte-identical copy,
SHA-256 `9b010715…`), the unmodified K1 config and robot model (`K1_locomotion.urdf`, `merge_fixed_joints=True`) and the
unmodified render settings. Provenance hashes: `diagnosis/provenance_sweep_files.txt`. No Matterport scene, model weight or
robot mesh is included; images are renders.

## How each image was made

### `images/as_evaluated/robot_camera/` (Fig. 1(c))
- Script: `scripts/sim_capture.py --protocol mainloop --main_steps 1` via `scripts/run_gpu.sh` (fresh process per image
  pair; task `k1_matterport_vision`, `--num_envs=1 --gait_phase_init=0.0 --headless --enable_cameras`, checkpoint
  `model_14498.pt`, `--episode_idx` and `--cam_z` as in the file name). No model server.
- `*_initial.png`: `infos["observations"]["camera_obs"]` returned by `env.reset()` after `VLNEnvWrapperV3`'s 1-s reset
  warm-up — the frame the evaluator stores as history frame 0 and writes as video frame 0 (here without the text overlay).
- `*_warmup_last.png`: the 8th history frame of the evaluator's 4-s zero-command warm-up (after warm-up step 200), the
  last frame before the first model query. Not in the videos.
- Renderer: Isaac Lab `Camera` annotator `rgb`, RTX real-time (the evaluator's `isaaclab.python.headless.rendering.kit`),
  1280×720, lossless PNG. Not path traced.
- Each PNG has a JSON sidecar (script, code commit, config overrides, dataset index, record, `cam_z`, render mode,
  samples per pixel, resolution, `as_evaluated`, gray fraction, Trunk pose, match to the sweep video).
- Match to the sweep's video frame 0 (sim frame with the evaluator's text overlay drawn on it vs the H.264 video;
  `diagnosis/step5a_raw/step5a_vs_video_frame0.json`, side-by-sides `*_sim_vs_video.png`):

| Image | Sweep video | gray: image / video | mean abs diff (codec floor ≈ 1–3) |
|---|---|---|---|
| idx0_rec0_camz0.07_initial | h060_200 output_0 | 0.0015 / 0.0014 | 1.75 |
| idx0_rec0_camz0.25_initial | h078_200 output_0 (and stretchA_300) | 0.0015 / 0.0013 | 1.76 |
| idx0_rec0_camz0.42_initial | h095_200 output_0 | 0.0007 / 0.0009 | 1.77 |
| idx0_rec0_camz0.57_initial | h110_200 output_0 | 1.0000 (0.9911 with text) / 0.9910 | 1.03 |
| idx0_rec0_camz0.77_initial | h130_200 output_0 | 0.0007 / 0.0009 | 1.78 |
| idx0_rec0_camz0.97_initial | h150_200 output_0 | 0.0006 / 0.0007 | 1.77 |
| idx7_rec10_camz0.25_initial | h078_200 output_10 | 0.0004 / 0.0004 | 2.63 |

The 0.57 initial frame is the uniform gray of the sweep; the 0.42 frames show the lighter "BOOSTER" lettering (the K1 logo
left at the spawn pose; mirrored across the middle of the warm-up-last frame).

### `images/as_evaluated/third_person/` (Fig. 1(b))
- Script: `scripts/sim_render_pt.py --model as_evaluated --episode_idx=0` (default `cam_z`). The environment is built with
  the evaluator's lines (unmodified robot model, scene and lights) and left in the state after the 1-s reset warm-up of
  dataset index 0 (record 0, zsNo4HB9uLZ). The head and arms are shown exactly as they render: left at the spawn pose,
  floating above the torso, arms in the T-pose of the merged URDF.
- Renderer: a Replicator render product on a new USD camera prim (Isaac Lab's `Camera` has no path-tracing switch);
  carb settings `/rtx/rendermode=PathTracing`, `/rtx/pathtracing/spp=32` per frame, `/rtx/pathtracing/totalSpp=256`,
  `/rtx/pathtracing/optixDenoiser/enabled=true` (read back after setting; max bounces 4). 12 frames rendered without
  stepping physics; 3 more frames changed nothing (mean |diff| 0.000), so the image is converged. 1920×1080, PNG.
- Nothing was added to the scene except the (invisible) camera prims. Lights are the evaluator's (distant light 1000,
  disk lights over the start and the goal).
- Views (`views.json`; positions chosen with a warp ray cast against the Matterport mesh so that no camera sits inside a
  wall): `behind_left` — eye 1.4 m above the floor behind the robot, looking at the body (the search over directions
  135–180° from the heading, back-left to straight back, found the most free space, 1.5 m, straight behind, so the view
  is from behind rather than behind-left; the eye ended 1.58 m from the body target; focal length auto-fitted 22 mm);
  `side` — 70° to the right of the heading, 1.5 m, focal 20.8 mm; `upper_body_close` — front-left, 0.86 m from a point 0.92 m above the floor (between the torso top and the floating head), focal 24 mm.

### `images/corrected/` (`corrected_*.png`)
- Script: `scripts/sim_render_pt.py --model corrected --views_json images/as_evaluated/third_person/views.json`
  (identical cameras, scene, lights and render settings).
- Only the robot model changed, in a copied config (`configs/k1_corrected_robot_cfg.py`): `K1_22dof.urdf` (the training
  model of `model_14498`), `fix_base=True`, Trunk held at the as-evaluated Trunk pose after the reset warm-up
  (`diagnosis/trunk_pose_idx0_after_reset.json`), head and arms actuated with the training gains and held at the training
  pose (Shoulder_Pitch 0.2, Shoulder_Roll −1.25 / +1.25, Elbow_Pitch 0, Elbow_Yaw −0.5 / +0.5, head 0; rendered joint
  angles in each JSON, within 0.02 rad). The walking-policy wrapper is not used (it is built for the 12-joint model);
  the PD holds the pose for 1 s before rendering.
- Caveat: with a fixed base the Trunk never leaves its spawn pose, so these images would look right even if the same
  stale-transform defect affected this model. They show the training model's geometry and pose, not a fix to the renderer.
- `studio/corrected_studio_three_quarter.png`: `--model studio`; the same corrected K1 alone on an 8 × 8 m neutral-gray
  floor (PreviewSurface albedo 0.62, roughness 0.85), dome light 900 + distant key light 1600 (8° angle, soft shadows),
  three-quarter view, focal 50 mm, 2048×2048, path traced 256 spp, OptiX denoiser.

## Scripts
| Script | Purpose |
|---|---|
| `run_gpu.sh` | runs one simulator script like the sweep runner (env `vlnce-isaac`, cwd `NaVILA-Bench`, headless + cameras), refuses if another compute process is on the GPU, logs wall time |
| `sim_setup.py` | argument parser and environment construction copied verbatim from the evaluator |
| `navila_eval_v3_COPY.py` | byte-identical copy of the sweep's evaluator (helpers are loaded from it) |
| `step1_extract_frames.py`, `step1_gray_masks.py`, `montage.py` | Step 1 video crops, gray masks, contact sheets |
| `step2_urdf_compare.py` | Step 2.1 tables and the URDF rigid-merge mass properties |
| `sim_explore.py` | Step 2.2 converted-USD hierarchy and Fabric attributes |
| `sim_step3.py` (v2), `sim_step3_v1.py` | Step 3 probe; removal (`--hide_meshes`) and causal tests (`--bind_material_meshes`, `--deinstance_trunk_visuals`) on a copy of the converted USD; `--physics_cpu` collision check |
| `step3_verify.py` | rendered footprint of the head/arm meshes vs the two pose hypotheses (IoU, depth) |
| `stale_geometry_check.py` | camera vs the spawn-pose meshes, predicted coverage with the 1-cm near plane |
| `sim_capture.py` | fresh-process captures for Step 4 and Step 5A |
| `step4_compare.py`, `compare_to_video.py` | pixel / pose comparisons across processes and against the sweep videos |
| `sim_render_pt.py` | Step 5B/5C path-traced renders |
| `step6_model_repeat.sh`, `step6_client.py` | Step 6 model-server repeatability |
| `build_tables.py` | tables in `diagnosis/tables.md` |
| `batch_*.sh` | the batches that were run |
