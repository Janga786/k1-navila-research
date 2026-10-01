# SPDX-License-Identifier: BSD-3-Clause
"""L0 camera gate: dump ONE rgb_camera frame from the real K1 Matterport-vision
benchmark env (no NaVILA loaded) and report the camera's actual world pose.

Reuses the benchmark's own build path (parse_env_cfg + gym.make), episode start
pose (mirrors reset_start_pos_rot in navila_eval_v3.py), and the rgb_camera
sensor. Saves the raw 512 frame, the 384 the VLM actually consumes, and the
--bright variant the eval sends. Lets you override the camera height with --cam_z
to compare the current 0.85 m mount vs the ground-truth ZED height (~0.25 m).

Run (benchmark env):
    conda run -n vlnce-isaac python scripts/l0_camera_dump.py \
        --headless --enable_cameras --num_envs 1 --cam_z 0.85 --tag z085
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="L0 K1 camera frame dump.")
parser.add_argument("--task", type=str, default="k1_matterport_vision")
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--cam_z", type=float, default=0.85,
                    help="camera z offset above Trunk (0.85=current cfg, 0.25=real ZED).")
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--tag", type=str, default="z085")
parser.add_argument("--out_dir", type=str, default="/tmp/l0_cam")
parser.add_argument("--cam_width", type=int, default=None, help="override camera width (e.g. 1280 for BoosterMipi). (--width is taken by AppLauncher)")
parser.add_argument("--cam_height", type=int, default=None, help="override camera height (e.g. 720).")
parser.add_argument("--aperture", type=float, default=None,
                    help="override horizontal_aperture (47.7=BoosterMipi ~90deg, 54.0=current, 60.3=ZED).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import os
import math

import numpy as np
import torch
from PIL import Image

import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

from omni.isaac.vlnce.config import *  # noqa: F401,F403  (registers gym tasks)
from omni.isaac.vlnce.utils import ASSETS_DIR
from omni.isaac.vlnce.utils.eval_utils import read_episodes


def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    """(H,W,3|4) tensor -> (H,W,3) uint8 numpy."""
    a = t.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0 + 1e-3:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)

    episodes = read_episodes(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz"))
    ep = episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)

    # --- mirror reset_start_pos_rot (navila_eval_v3.py) so the robot stands in a real room ---
    scene_id = os.path.splitext(os.path.basename(ep["scene_id"]))[0]
    env_cfg.scene.terrain.obj_filepath = os.path.join(
        ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd")
    sp, sr = ep["start_position"], ep["start_rotation"]
    gp = ep["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = sr
    env_cfg.scene.robot.init_state.pos = (sp[0], sp[1], sp[2] + 0.55)
    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos
    if hasattr(env_cfg.scene, "disk_1"):
        env_cfg.scene.disk_1.init_state.pos = [sp[0], sp[1], sp[2] + 2.5]
    if hasattr(env_cfg.scene, "disk_2"):
        env_cfg.scene.disk_2.init_state.pos = [gp[0], gp[1], gp[2] + 2.5]

    # --- camera height override (0.85 current vs 0.25 ground-truth ZED) ---
    old = tuple(env_cfg.scene.rgb_camera.offset.pos)
    env_cfg.scene.rgb_camera.offset.pos = (old[0], old[1], args_cli.cam_z)
    if args_cli.cam_width:
        env_cfg.scene.rgb_camera.width = args_cli.cam_width
    if args_cli.cam_height:
        env_cfg.scene.rgb_camera.height = args_cli.cam_height
    if args_cli.aperture:
        env_cfg.scene.rgb_camera.spawn.horizontal_aperture = args_cli.aperture
    cam = env_cfg.scene.rgb_camera
    print(f"[L0cam] scene={scene_id} ep={args_cli.episode_idx}  "
          f"cam offset pos {old} -> {cam.offset.pos}  "
          f"{cam.width}x{cam.height} haperture={cam.spawn.horizontal_aperture}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    obs, _ = env.reset()

    act_dim = uenv.action_manager.total_action_dim
    zero = torch.zeros((1, act_dim), device=uenv.device)
    for _ in range(args_cli.settle_steps):
        env.step(zero)

    cam = uenv.scene.sensors["rgb_camera"]
    rgb = cam.data.output["rgb"][0]
    img = to_uint8_rgb(rgb)

    # camera + robot world pose
    cam_pos = cam.data.pos_w[0].detach().cpu().numpy()
    robot = uenv.scene["robot"]
    trunk_z = robot.data.root_pos_w[0, 2].item()
    pg_z = robot.data.projected_gravity_b[0, 2].item()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -pg_z))))

    base = os.path.join(args_cli.out_dir, f"k1_cam_{args_cli.tag}")
    Image.fromarray(img).save(base + "_raw512.png")
    Image.fromarray(img).resize((384, 384)).save(base + "_vlm384.png")
    try:
        import cv2
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        bright = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        bright = (((bright / 255.0) ** 0.7) * 255).astype(np.uint8)
        Image.fromarray(bright).save(base + "_bright.png")
    except Exception as e:
        print(f"[L0cam] bright skipped: {e!r}", flush=True)

    # frame sanity
    mean_lum = float(img.mean())
    frac_dark = float((img.max(axis=-1) < 10).mean())
    print(f"[L0cam] frame shape={img.shape} mean_lum={mean_lum:.1f} "
          f"frac_near_black={frac_dark:.3f}", flush=True)
    print(f"[L0cam] camera world pos = ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, "
          f"{cam_pos[2]:.3f})  -> eye height {cam_pos[2]:.3f} m", flush=True)
    print(f"[L0cam] trunk world z = {trunk_z:.3f} m   robot tilt = {tilt:.1f} deg", flush=True)
    print(f"[L0cam] cam-above-trunk = {cam_pos[2]-trunk_z:.3f} m", flush=True)
    print(f"[L0cam] SAVED {base}_raw512.png / _vlm384.png", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
