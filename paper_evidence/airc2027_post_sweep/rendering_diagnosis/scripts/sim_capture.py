"""Fresh-process captures, as evaluated (GPU, short). Used by Step 4 (repeatability) and Step 5A (paper robot-camera
images). The environment is built exactly as the evaluator does (sim_setup.build_env: verbatim evaluator lines,
unmodified config, robot model and render settings). No model server; never writes eval_results/.

--protocol init50
    Step 4.1: save the initial frame (after the 1-s reset warm-up), then drive the walking policy with a fixed
    command (default [0.5, 0, 0]) for 50 control steps and save that frame. Trunk / body states saved bit-exact.
--protocol mainloop
    Step 4.2 and Step 5A: replicate the evaluator's own sequence: initial frame (= video frame 0, the first history
    frame), the 4-s pre-VLM warm-up with zero command (200 steps; a history frame every 25 steps, the 8th = the last
    warm-up frame), then main-loop steps with a fixed command, saving the history frames the evaluator would add at
    main-loop steps 0, 25, 50, 75 (num_steps % 25 == 0 after env.step). The fixed command replaces the VLM.

Frames are saved lossless (PNG) without any text overlay; bit-exact copies of the robot-camera arrays (.npy) and
Trunk states (.npy, float32 as produced) are saved for diffing.
"""
import functools
import json
import os
import subprocess
import sys

print = functools.partial(print, flush=True)  # noqa: A001
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_setup  # noqa: E402


def extra(p):
    p.add_argument("--out_dir", required=True)
    p.add_argument("--protocol", choices=["init50", "mainloop"], required=True)
    p.add_argument("--cmd", type=str, default="0.5,0,0")
    p.add_argument("--main_steps", type=int, default=76)


args, simulation_app = sim_setup.launch("capture", extra)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

GRAY = lambda a: float((np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6).mean())  # noqa: E731


def main():
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    env, episode, env_cfg, obs, infos = sim_setup.build_env(args)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    ti = robot.body_names.index("Trunk")
    commit = subprocess.run(["git", "-C", os.path.expanduser("~/Projects/k1_research"), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = {"script": "airc2027_renders/scripts/sim_capture.py", "protocol": args.protocol,
            "k1_research_commit": commit, "evaluator_sha256": "9b010715ced0a46efb17b86e918653e933bdc26b5fd765f3d8a0912d7b1281fc",
            "task": args.task, "episode_idx": args.episode_idx, "record": int(episode["episode_id"]) - 1,
            "scene": os.path.splitext(os.path.basename(episode["scene_id"]))[0], "cam_z": args.cam_z,
            "config_overrides": {"cam_z": args.cam_z} if args.cam_z is not None else {},
            "robot_model": "K1_locomotion.urdf (merge_fixed_joints=True), unmodified", "as_evaluated": True,
            "render_mode": "RTX real-time (Isaac Lab 1.1 headless.rendering kit, unmodified), Isaac Lab Camera annotator 'rgb'",
            "samples_per_pixel": "n/a (real-time ray-traced lighting, not path tracing)", "resolution": [1280, 720],
            "text_overlay": False, "frames": {}}

    def snap(tag, inf):
        rgb = inf["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
        Image.fromarray(rgb).save(f"{out}/{tag}_robotcam.png")
        np.save(f"{out}/{tag}_robotcam.npy", rgb)
        rs = robot.data.root_state_w[0].cpu().numpy()
        bs = torch.cat([robot.data.body_pos_w[0], robot.data.body_quat_w[0]], -1).cpu().numpy()
        np.save(f"{out}/{tag}_root_state.npy", rs)
        np.save(f"{out}/{tag}_body_states.npy", bs)
        meta["frames"][tag] = {"robotcam_gray_frac": GRAY(rgb), "trunk_pos": robot.data.body_pos_w[0, ti].cpu().numpy().tolist(),
                               "trunk_quat_wxyz": robot.data.body_quat_w[0, ti].cpu().numpy().tolist(),
                               "root_state_float32_hex": rs.astype(np.float32).tobytes().hex()}
        print(f"[snap] {tag} gray={GRAY(rgb):.4f} trunk={np.round(meta['frames'][tag]['trunk_pos'], 5)}")

    snap("init", infos)
    cmd = torch.tensor([float(v) for v in args.cmd.split(",")], device=obs.device)
    if args.protocol == "init50":
        for _ in range(50):
            obs, _, done, infos = env.step(cmd)
        snap("after50", infos)
    else:
        # evaluator: steps_per_image = 25; warmup_total_steps = 8 * 25 = 200, zero command
        zero_cmd = torch.zeros(3, device=obs.device)
        for warm_i in range(200):
            obs, _, done, infos = env.step(zero_cmd)
            if (warm_i + 1) % 25 == 0:
                snap(f"warmup{(warm_i + 1) // 25}", infos)  # warmup8 = last frame of the 4-s warm-up
        for num_steps in range(args.main_steps):
            obs, _, done, infos = env.step(cmd)
            if bool(done):
                meta["done_at_main_step"] = num_steps
                print("[main] done at", num_steps)
                break
            if num_steps % 25 == 0:
                snap(f"main{num_steps:03d}", infos)
    meta["command"] = [float(v) for v in args.cmd.split(",")]
    json.dump(meta, open(f"{out}/capture.json", "w"), indent=1)
    print("[done]", out)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001  (Kit keeps the process alive after an uncaught exception)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    simulation_app.close()
