# SPDX-License-Identifier: BSD-3-Clause
"""L2c/L2d/L2e runtime gate: drive the REAL VLNEnvWrapperV3 in the benchmark env
(no VLM) and verify, component by component, the 235-dim policy input it builds;
dump the 8 VLM frames the sampler would pick; assert the 50 Hz step.

  L2c: slice-by-slice check of the term-major 235 vector against live robot state
       (offsets cmd[0:15] gait[15:25] grav[25:40] angvel[40:55] jpos[55:115]
        jvel[115:175] act[175:235]; newest frame = last 1/5 of each block),
       gait stand-gate, history ordering via gait-phase progression.
  L2d: sampler indices ([0,2,4,6,8,10,12,15] @ n=16, first/last kept, 8 unique)
       + saves the 8 sampled frames to /tmp/l2_frames for eyeballing.
  L2e: step_dt == 0.02 (50 Hz).

Run:
  conda run -n vlnce-isaac python scripts/l2_obs_frame_check.py \
      --headless --enable_cameras --checkpoint /abs/model_11999.pt
"""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="L2c/d/e bridge runtime gate.")
parser.add_argument("--task", type=str, default="k1_matterport_vision")
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="/tmp/l2_frames")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ----
import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import gymnasium as gym

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

from omni.isaac.vlnce.config import *  # noqa: F401,F403
from omni.isaac.vlnce.utils import ASSETS_DIR, VLNEnvWrapperV3
from omni.isaac.vlnce.utils.eval_utils import read_episodes

PASS = 0
FAIL = 0


def check(tag, name, cond, detail=""):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"[{tag}] {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}",
          flush=True)
    return ok


def build_v3_actor(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(235, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    own = actor.state_dict()
    mapped = {k[len("actor."):]: v for k, v in sd.items()
              if k.startswith("actor.") and k[len("actor."):] in own}
    actor.load_state_dict(mapped, strict=True)
    actor.to(device).eval()
    print(f"[L2] actor loaded from {os.path.basename(ckpt_path)} iter={ck.get('iter', '?')}",
          flush=True)
    return actor


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    episodes = read_episodes(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz"))
    ep = episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    # mirror reset_start_pos_rot (same as l0_camera_dump.py)
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
    obs, infos = env.reset()  # includes the 50-step zero-cmd warmup

    # ---------------- L2e: 50 Hz two-clock fast loop ----------------
    u = env.unwrapped
    step_dt = u.cfg.sim.dt * u.cfg.decimation
    check("L2e", "policy step_dt == 0.02 s (50 Hz)", abs(step_dt - 0.02) < 1e-9,
          f"sim.dt={u.cfg.sim.dt} x decim={u.cfg.decimation} = {step_dt}")

    # ---------------- drive forward, collecting camera frames ----------------
    cmd = torch.tensor([0.4, 0.0, 0.0], device=device)
    frames = []
    for k in range(80):
        obs, _, done, infos = env.step(cmd)
        if k % 5 == 0 and len(frames) < 16:
            f = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            frames.append(Image.fromarray(f.astype(np.uint8)))

    # ---------------- L2c: the 235 vector, slice by slice ----------------
    robot = env.unwrapped.scene["robot"]
    jperm = env._jperm
    # live state S (captured BEFORE the manual build, which appends a frame from S)
    S_cmd = env._command[0].detach().clone()
    S_phase = float(env._gait_phase[0].item())
    S_grav = robot.data.projected_gravity_b[0].detach().clone()
    S_angv = robot.data.root_ang_vel_b[0].detach().clone()
    S_q = (robot.data.joint_pos[0, jperm] - robot.data.default_joint_pos[0, jperm]).detach().clone()
    S_qd = robot.data.joint_vel[0, jperm].detach().clone()
    S_act = env._last_action[0].detach().clone()

    pi = env._build_policy_input()
    check("L2c", "policy input shape (1, 235)", tuple(pi.shape) == (1, 235), f"{tuple(pi.shape)}")
    check("L2c", "all finite", bool(torch.isfinite(pi).all()))
    v = pi[0]

    OFF = {"cmd": (0, 3), "gait": (15, 2), "grav": (25, 3), "angvel": (40, 3),
           "jpos": (55, 12), "jvel": (115, 12), "act": (175, 12)}
    newest = {k: v[o + 4 * d: o + 5 * d] for k, (o, d) in OFF.items()}

    t = 1e-4
    check("L2c", "cmd newest == live command (0.4,0,0)",
          torch.allclose(newest["cmd"], S_cmd, atol=t), f"{newest['cmd'].tolist()}")
    g_exp = torch.tensor([math.cos(2 * math.pi * S_phase), math.sin(2 * math.pi * S_phase)],
                         device=device)
    cos_first = torch.allclose(newest["gait"], g_exp, atol=1e-3)
    sin_first = torch.allclose(newest["gait"], g_exp.flip(0), atol=1e-3)
    check("L2c", "gait newest == [cos,sin](2*pi*phase) (cos-first)", cos_first,
          f"{newest['gait'].tolist()} vs cos-first {g_exp.tolist()} (sin-first match: {sin_first})")
    check("L2c", "grav newest == projected_gravity_b (raw)",
          torch.allclose(newest["grav"], S_grav, atol=t),
          f"{[round(x, 3) for x in newest['grav'].tolist()]} norm={newest['grav'].norm():.4f}")
    check("L2c", "angvel newest == root_ang_vel_b x 0.25",
          torch.allclose(newest["angvel"], S_angv * 0.25, atol=t))
    check("L2c", "jpos newest == (q - q0)[LEG order]",
          torch.allclose(newest["jpos"], S_q, atol=t),
          f"max|jpos|={newest['jpos'].abs().max():.3f}")
    check("L2c", "jvel newest == qd[LEG order] x 0.1",
          torch.allclose(newest["jvel"], S_qd * 0.1, atol=t))
    check("L2c", "prev-action slot == wrapper._last_action (raw)",
          torch.allclose(newest["act"], S_act, atol=t))

    # history ordering: gait phase must INCREASE left->right within the block (oldest first)
    gait_block = v[15:25].view(5, 2)
    ph = torch.atan2(gait_block[:, 1], gait_block[:, 0])
    diffs = torch.remainder(ph[1:] - ph[:-1], 2 * math.pi)
    exp_d = 2 * math.pi * 1.5 * step_dt
    check("L2c", "history order oldest->newest (gait phase advances L->R)",
          bool(((diffs - exp_d).abs() < 1e-2).all()),
          f"frame-to-frame dphase={[round(x, 4) for x in diffs.tolist()]} expected {exp_d:.4f}")
    a_out = policy_fn(pi)
    check("L2c", "actor consumes vector: out (1,12) finite",
          tuple(a_out.shape) == (1, 12) and bool(torch.isfinite(a_out).all()))

    # stand-gate: zero command -> gait slots must zero
    zero = torch.tensor([0.0, 0.0, 0.0], device=device)
    for _ in range(6):
        obs, _, _, _ = env.step(zero)
    v0 = env._build_policy_input()[0]
    check("L2c", "gait stand-gate: ||cmd||<0.1 -> newest gait == 0",
          bool((v0[23:25].abs() < 1e-6).all()), f"{v0[23:25].tolist()}")

    # ---------------- L2d: sampler + frame dump ----------------
    def sample_indices(n):
        return [int(i * (n - 1) / 7) for i in range(7)] + [n - 1]

    idx16 = sample_indices(16)
    check("L2d", "indices @ n=16 == [0,2,4,6,8,10,12,15]",
          idx16 == [0, 2, 4, 6, 8, 10, 12, 15], f"{idx16}")
    ok = True
    for n in range(8, 41):
        ix = sample_indices(n)
        ok &= (len(ix) == 8 and ix[0] == 0 and ix[-1] == n - 1
               and ix == sorted(ix) and len(set(ix)) == 8)
    check("L2d", "n=8..40: always 8 unique sorted, first+last kept", ok)
    picked = [frames[i] for i in sample_indices(len(frames))]
    for j, im in enumerate(picked):
        im.save(os.path.join(args_cli.out_dir, f"vlm_frame_{j}.jpg"))
    check("L2d", f"dumped {len(picked)} sampled frames", len(picked) == 8,
          f"-> {args_cli.out_dir}/vlm_frame_*.jpg (eyeball: coherent forward walk)")

    print(f"\n[L2] TOTAL: {PASS} pass / {FAIL} fail", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
