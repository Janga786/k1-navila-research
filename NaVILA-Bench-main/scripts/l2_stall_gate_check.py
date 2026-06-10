# SPDX-License-Identifier: BSD-3-Clause
"""Stall-abort validation (audit rank 5 / plan steps 17-18) — runs the REAL
VLNEnvWrapperV3.check_same_pos (cmd-gated + ang-gated, 150 steps) with no VLM.

--mode free  (use an OPEN-room episode, e.g. --episode_idx 12):
    false-positive suite — 4 s stand, 2 s walk, 90-deg left turn, 90-deg right
    turn, 1 s walk. The stuck counter must never fire; turns must land ~90 deg
    (this also closes audit step 17: post-parser-fix 90-deg commands execute
    fully at the policy level).
--mode jam   (use the wall-facing episode, --episode_idx 0):
    true-positive — walk straight into the obstacle; the wrapper must end the
    episode ~3 s (150 steps) after progress stops, with the robot UPRIGHT.

Run:
  conda run -n vlnce-isaac python scripts/l2_stall_gate_check.py \
      --headless --enable_cameras --mode free --episode_idx 12 --checkpoint <pt>
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stall-abort gate validation.")
parser.add_argument("--task", type=str, default="k1_matterport_vision")
parser.add_argument("--episode_idx", type=int, default=12)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--mode", choices=["free", "jam"], default="free")
parser.add_argument("--jam_max_steps", type=int, default=900)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from omni.isaac.vlnce.config import *  # noqa: F401,F403
from omni.isaac.vlnce.utils import ASSETS_DIR, VLNEnvWrapperV3
from omni.isaac.vlnce.utils.eval_utils import read_episodes

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"[stallgate] {'PASS' if ok else 'FAIL'}  {name}"
          f"{('  -- ' + detail) if detail else ''}", flush=True)


def build_v3_actor(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(235, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12),
    )
    own = actor.state_dict()
    actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items()
                           if k.startswith("actor.") and k[len("actor."):] in own},
                          strict=True)
    actor.to(device).eval()
    return actor


def yaw_of(env):
    q = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_phase(env, cmd_vec, steps, label):
    """Step with a fixed command; return (done_fired, done_step, max_count, dyaw, disp, ang_hist)."""
    device = env.unwrapped.device
    cmd = torch.tensor(cmd_vec, device=device, dtype=torch.float32)
    p0 = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy().copy()
    y0 = yaw_of(env)
    yaw_acc, prev_y = 0.0, y0
    max_count, ang_hist = 0, []
    for k in range(1, steps + 1):
        obs, _, done, info = env.step(cmd)
        cy = yaw_of(env)
        yaw_acc += wrap_pi(cy - prev_y)
        prev_y = cy
        max_count = max(max_count, int(env.same_pos_count))
        ang_hist.append(float(np.linalg.norm(
            env.unwrapped.scene["robot"].data.root_vel_w[0, 3:].detach().cpu().numpy())))
        if bool(done):
            p1 = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
            return True, k, max_count, yaw_acc, float(np.linalg.norm((p1 - p0)[:2])), ang_hist
    p1 = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    print(f"[stallgate] phase {label}: steps={steps} disp={np.linalg.norm((p1 - p0)[:2]):.3f} m "
          f"dyaw={math.degrees(yaw_acc):+.1f} deg max_count={max_count}", flush=True)
    return False, steps, max_count, yaw_acc, float(np.linalg.norm((p1 - p0)[:2])), ang_hist


def main():
    episodes = read_episodes(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz"))
    ep = episodes[args_cli.episode_idx]
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    device = env.unwrapped.device
    actor = build_v3_actor(args_cli.checkpoint, str(device))

    def policy_fn(o):
        with torch.no_grad():
            return actor(o)

    env = VLNEnvWrapperV3(env, policy_fn, args_cli.task, ep,
                          high_level_obs_key="camera_obs", gait_phase_init=0.0)
    env.reset()
    print(f"[stallgate] mode={args_cli.mode} ep={args_cli.episode_idx} scene={scene_id} "
          f"ckpt={os.path.basename(args_cli.checkpoint)}", flush=True)

    if args_cli.mode == "free":
        # same_pos_count now == windowed-buffer length (fires only if 500 reached AND
        # net progress tiny). The suite keeps commands active for 600 straight steps,
        # so the window FILLS during legit motion — proving net-progress spares it.
        d, _, mc, _, _, _ = run_phase(env, [0.0, 0.0, 0.0], 200, "STAND-4s")
        check("stand 4 s: window stays empty (cmd gate)", not d and mc == 0, f"buf={mc}")
        d, _, mc, _, disp, _ = run_phase(env, [0.4, 0.0, 0.0], 250, "WALK-5s")
        check("walk 5 s: no fire, robot moved", not d and disp > 1.0, f"disp={disp:.2f} buf={mc}")
        d, _, mc, dy, _, ah = run_phase(env, [0.0, 0.0, math.pi / 6], 150, "TURN-L90")
        check("turn-in-place L90: no fire", not d, f"buf={mc} min_ang={min(ah):.2f}")
        check("L90 lands 70-105 deg [audit step 17; incl walk->turn transition]",
              70 <= math.degrees(dy) <= 105, f"dyaw={math.degrees(dy):+.1f}")
        d, _, mc, dy, _, _ = run_phase(env, [0.0, 0.0, -math.pi / 6], 150, "TURN-R90")
        # window is FULL (>=500 commanded steps) somewhere in here — must still not fire
        check("turn-in-place R90: no fire with FULL window", not d, f"buf={mc}")
        check("R90 lands -80..-100 deg (settled turn)", -100 <= math.degrees(dy) <= -80,
              f"dyaw={math.degrees(dy):+.1f}")
        d, _, mc, _, disp, _ = run_phase(env, [0.4, 0.0, 0.0], 50, "WALK-1s")
        check("walk 1 s after turns: no fire (full window)", not d, f"disp={disp:.2f} buf={mc}")
    else:  # jam
        d, k, mc, dy, disp, ah = run_phase(env, [0.4, 0.0, 0.0], args_cli.jam_max_steps,
                                           "JAM-WALK")
        zraw = float(env.unwrapped.scene["robot"].data.root_pos_w[0, 2].detach())
        check("hopeless-stall FIRED before step cap", d,
              f"done_step={k} disp={disp:.2f} m dyaw={math.degrees(dy):+.1f}")
        check("fired with FULL 500-step window (10 s no-progress)", mc >= 500, f"buf={mc}")
        check("fired in plausible band [500, cap]", 500 <= k <= args_cli.jam_max_steps,
              f"k={k}")
        check("robot UPRIGHT at abort (z > 0.40 above start floor)",
              zraw - (env_cfg.scene.robot.init_state.pos[2] - 0.55) > 0.40,
              f"z_raw={zraw:.3f}")
        if not d:
            tail = ah[-100:]
            print(f"[stallgate] NOT-FIRED diagnostics: base ang-speed last-100 "
                  f"mean={np.mean(tail):.3f} max={np.max(tail):.3f}; total disp={disp:.2f}; "
                  f"check net-progress thresholds (0.25 m / 30 deg per 10 s)", flush=True)

    print(f"\n[stallgate] TOTAL: {PASS} pass / {FAIL} fail", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
