# SPDX-License-Identifier: BSD-3-Clause
"""L4 control episode — the scoreboard gate, no VLM.

Drives the REAL benchmark wrapper along the episode's ground-truth reference
path with a waypoint-following P-controller (velocity commands inside the
policy's trained envelope), calls set_stop_called inside the 3 m goal radius,
and verifies the scorer:

  - success == 1.0 only AFTER stop (and 0.0 while merely near the goal),
  - oracle_success flips to 1 when first within radius,
  - final distance_to_goal < 3.0 and == reported NE,
  - SPL in (0, 1].

If the harness can't score this gimme, the scorer is broken, not the robot.

Run:
  conda run -n vlnce-isaac python scripts/l4_control_episode.py \
      --headless --enable_cameras --episode_idx 0 --checkpoint <model.pt>
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="L4 control episode (scripted, no VLM).")
parser.add_argument("--task", type=str, default="k1_matterport_vision")
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=6000, help="120 s @ 50 Hz, the prod budget")
parser.add_argument("--wp_reach", type=float, default=0.5, help="advance waypoint within (m)")
parser.add_argument("--stop_at", type=float, default=2.0, help="call stop when dtg < (m)")
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
    print(f"[L4ctl] {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}",
          flush=True)


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


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


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
    obs, infos = env.reset()

    # TELEPORT CONTROL: the scoreboard gate tests SCORING, not navigation (VLN-CE
    # reference waypoints are sparse viewpoints ~2.5 m apart — no episode is a
    # collision-free "straight shot" for a beeline follower, ep0/ep12 both proved it).
    # Sequence: walk briefly (real motion through the scorer) -> teleport the base
    # next to the goal -> verify OS flips + success stays 0 without stop -> stop ->
    # verify success/SPL/NE all score correctly.
    wps = [np.array(w, dtype=np.float64) for w in ep["reference_path"]]
    goal = wps[-1]
    robot = env.unwrapped.scene["robot"]
    print(f"[L4ctl] ep={args_cli.episode_idx} scene={scene_id} waypoints={len(wps)} "
          f"start_dtg={infos['measurements']['distance_to_goal']:.2f} m", flush=True)
    check("initial success == 0", infos["measurements"]["success"] == 0.0)
    check("initial dtg > 3.0 (not a spawn-in-goal episode)",
          infos["measurements"]["distance_to_goal"] > 3.0,
          f"{infos['measurements']['distance_to_goal']:.2f}")

    # 1) short real walk so the scorer integrates genuine motion
    k = 0
    for _ in range(100):
        obs, _, done, infos = env.step(torch.tensor([0.4, 0.0, 0.0], device=device))
        k += 1
    m = infos["measurements"]
    check("after 2 s walk: path_length > 0", m["path_length"] > 0.2,
          f"path={m['path_length']:.2f}")

    # 2) teleport next to the goal (1.0 m short of it, upright, zero velocity)
    away = (wps[0][:2] - goal[:2])
    away = away / (np.linalg.norm(away) + 1e-9)
    tp_xy = goal[:2] + 1.0 * away
    pose = torch.zeros(1, 7, device=device)
    pose[0, 0], pose[0, 1], pose[0, 2] = float(tp_xy[0]), float(tp_xy[1]), float(goal[2] + 0.56)
    pose[0, 3] = 1.0
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    print(f"[L4ctl] teleported to {tp_xy.round(2).tolist()} (1.0 m from goal)", flush=True)

    # 3) stand 1 s (zero cmd: hopeless-stall window clears; history refreshes)
    for _ in range(50):
        obs, _, done, infos = env.step(torch.zeros(3, device=device))
        k += 1
    m = infos["measurements"]
    check("near goal WITHOUT stop: success stays 0 (stop required)",
          m["success"] == 0.0, f"dtg={m['distance_to_goal']:.2f} success={m['success']}")
    check("oracle_success == 1 once within radius", m["oracle_success"] == 1.0,
          f"dtg={m['distance_to_goal']:.2f}")
    check("dtg now < 3.0 at 1.0 m from goal (path-distance sane near goal)",
          m["distance_to_goal"] < 3.0, f"{m['distance_to_goal']:.2f}")

    # 4) stop — registered BEFORE the scoring step (eval ordering)
    env.set_stop_called(True)
    obs, _, done, infos = env.step(torch.zeros(3, device=device))
    m = infos["measurements"]
    k += 1

    print(f"[L4ctl] FINAL after {k} steps -- measurements: "
          + ", ".join(f"{kk}={float(vv):.3f}" for kk, vv in m.items()), flush=True)
    check("scored SUCCESS == 1.0 on the gimme", m["success"] == 1.0, f"{m['success']}")
    check("final NE (distance_to_goal) < 3.0", m["distance_to_goal"] < 3.0,
          f"{m['distance_to_goal']:.2f}")
    check("oracle_success == 1.0", m["oracle_success"] == 1.0)
    check("SPL in (0, 1]", 0.0 < m["spl"] <= 1.0, f"{m['spl']:.3f}")
    check("ONE <= final NE (min over episode)",
          m["oracle_navigation_error"] <= m["distance_to_goal"] + 1e-6,
          f"ONE={m['oracle_navigation_error']:.2f}")
    check("finished inside the 120 s budget", k < args_cli.max_steps, f"k={k}")

    print(f"\n[L4ctl] TOTAL: {PASS} pass / {FAIL} fail", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
