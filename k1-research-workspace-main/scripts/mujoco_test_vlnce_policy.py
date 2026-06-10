"""
Run the VLN-CE-trained K1 policy (model_28000.pt) in MuJoCo on flat ground,
send a deterministic velocity-command sequence, record an mp4, and print
quantitative stats.

Purpose: visually verify whether the trained policy itself walks correctly.
If it does, the eval-time bug must be in the NaVILA-Bench integration
(observation injection / actuator wiring / scene geometry). If it doesn't,
the policy itself is the problem.

Obs layout the policy expects (235 dims, no history):
  base_lin_vel(3) base_ang_vel(3) projected_gravity(3) velocity_commands(3)
  joint_pos_rel(12) joint_vel_rel(12) last_action(12) height_scan(187)

Joint order (leg-only, used for joint_pos/joint_vel/last_action and action):
  Left_Hip_Pitch, Right_Hip_Pitch,
  Left_Hip_Roll,  Right_Hip_Roll,
  Left_Hip_Yaw,   Right_Hip_Yaw,
  Left_Knee_Pitch, Right_Knee_Pitch,
  Left_Ankle_Pitch, Right_Ankle_Pitch,
  Left_Ankle_Roll,  Right_Ankle_Roll

PD gains and effort limits are taken verbatim from BOOSTER_K1_LOCOMOTION_CFG
in booster_train.assets.robots.booster.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

# Force MuJoCo to use EGL for offscreen GPU rendering.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco


# --------------------------------------------------------------------------- #
# Constants — every value below comes from booster_train's training config.
# --------------------------------------------------------------------------- #
MJCF_PATH = (
    "/home/janga/Projects/k1_research/booster/booster_assets/robots/K1/K1_22dof.xml"
)
CKPT_PATH = (
    "/home/janga/Projects/k1_research/booster/booster_train/logs/rsl_rl/"
    "k1_velocity_vlnce/2026-05-18_21-15-33_k1_vlnce_v2_resumed/model_28000.pt"
)

PHYSICS_DT = 0.005     # sim.dt in training
DECIMATION = 4         # control runs every 4 sim steps -> 50 Hz policy
ACTION_SCALE = 0.5     # JointPositionActionCfg.scale in vlnce_env_cfg.py

# 12 leg joints in URDF/articulation order. Isaac Lab's regex expansion in
# JointPositionActionCfg uses preserve_order=False (the default), which sorts
# the matching joints by their articulation index — i.e., URDF declaration
# order. For K1_locomotion.urdf that means left chain first, then right chain.
# Both obs (joint_pos_rel/joint_vel_rel) and action use this same ordering.
POLICY_LEG_JOINTS: list[str] = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
LEG_DEFAULT_POS = np.array([
    -0.15, 0.00, 0.00, 0.30, -0.15, 0.00,    # left chain
    -0.15, 0.00, 0.00, 0.30, -0.15, 0.00,    # right chain
], dtype=np.float32)
# kp/kd per leg joint (matches BOOSTER_K1_LOCOMOTION_CFG):
#   hips/knees: stiffness=350, damping=7.5    ankles: stiffness=250, damping=5.0
LEG_KP = np.array([350, 350, 350, 350, 250, 250,
                   350, 350, 350, 350, 250, 250], dtype=np.float32)
LEG_KD = np.array([7.5, 7.5, 7.5, 7.5, 5.0, 5.0,
                   7.5, 7.5, 7.5, 7.5, 5.0, 5.0], dtype=np.float32)
LEG_EFFORT_LIMIT = np.array(
    [30, 35, 20, 60, 20, 20,
     30, 35, 20, 60, 20, 20], dtype=np.float32
)

# All 22 joints in MJCF order, with their hold-pose defaults (matching k1_velocity.py).
# Arms hold a relaxed pose to match the booster_deploy task setup.
ALL_JOINT_NAMES_DEFAULTS: list[tuple[str, float]] = [
    ("AAHead_yaw", 0.0), ("Head_pitch", 0.0),
    ("ALeft_Shoulder_Pitch", 0.2), ("Left_Shoulder_Roll", -1.25),
    ("Left_Elbow_Pitch", 0.0),     ("Left_Elbow_Yaw", -0.5),
    ("ARight_Shoulder_Pitch", 0.2), ("Right_Shoulder_Roll", 1.25),
    ("Right_Elbow_Pitch", 0.0),    ("Right_Elbow_Yaw", 0.5),
    ("Left_Hip_Pitch", -0.15), ("Left_Hip_Roll", 0.0), ("Left_Hip_Yaw", 0.0),
    ("Left_Knee_Pitch", 0.30), ("Left_Ankle_Pitch", -0.15), ("Left_Ankle_Roll", 0.0),
    ("Right_Hip_Pitch", -0.15), ("Right_Hip_Roll", 0.0), ("Right_Hip_Yaw", 0.0),
    ("Right_Knee_Pitch", 0.30), ("Right_Ankle_Pitch", -0.15), ("Right_Ankle_Roll", 0.0),
]


# --------------------------------------------------------------------------- #
# Tiny MLP actor matching the rsl_rl 3.x ActorCritic actor head.
# Architecture: [512, 256, 128], activation=ELU, in=235, out=12.
# We only need the actor for inference; build it directly from state_dict
# to avoid rsl_rl version drift between save (3.1.2) and load (whatever env).
# --------------------------------------------------------------------------- #
def build_actor_from_state_dict(state_dict: dict, in_dim: int, hidden: list[int],
                                out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ELU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    actor = nn.Sequential(*layers)

    own = actor.state_dict()
    mapped = {}
    for k, v in state_dict.items():
        if not k.startswith("actor."):
            continue
        local_k = k[len("actor."):]
        if local_k in own:
            mapped[local_k] = v
    missing = [k for k in own if k not in mapped]
    if missing:
        raise RuntimeError(f"Missing actor keys when mapping state_dict: {missing}")
    actor.load_state_dict(mapped, strict=True)
    actor.eval()
    return actor


# --------------------------------------------------------------------------- #
# Quaternion utilities (Isaac Lab's quat_apply_inverse: passive rotation by q^-1).
# Isaac Lab quaternion order: (w, x, y, z).
# --------------------------------------------------------------------------- #
def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return (v - w * t + np.cross(qv, t)).astype(np.float32)


def quat_to_yaw(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


# --------------------------------------------------------------------------- #
# Velocity command schedule.
# --------------------------------------------------------------------------- #
def vel_command_at(t: float) -> np.ndarray:
    """(vx, vy, wz) in robot frame, as commanded to the policy."""
    if t < 5.0:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
    if t < 8.0:
        return np.array([0.0, 0.0, 0.524], dtype=np.float32)  # ~30 deg/s
    if t < 13.0:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


SEGMENTS = [
    ("vx=0.5 forward",   0.0,  5.0),
    ("vyaw=0.524 turn",  5.0,  8.0),
    ("vx=0.5 forward",   8.0, 13.0),
    ("zero (stop)",     13.0, 15.0),
]


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/home/janga/Desktop/k1_vlnce_policy_test.mp4")
    p.add_argument("--fps", type=int, default=50)         # 50 Hz policy -> 50 fps
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="zero-command policy steps before commanded sequence "
                        "(not recorded). default 0 so video starts at the "
                        "clean standing pose")
    p.add_argument("--initial_z", type=float, default=0.55,
                   help="Trunk drop height; matches training init_state.pos[2]")
    args = p.parse_args()

    # ----- load policy state dict -----
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    actor = build_actor_from_state_dict(
        sd, in_dim=235, hidden=[512, 256, 128], out_dim=12
    )
    print(f"[policy] loaded {CKPT_PATH}", file=sys.stderr)
    print(f"[policy] iter={ck.get('iter')} actor params={sum(p.numel() for p in actor.parameters())}",
          file=sys.stderr)

    # ----- load MuJoCo model -----
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_model.opt.timestep = PHYSICS_DT
    mj_data = mujoco.MjData(mj_model)

    # joint name -> qpos/qvel addresses
    def jaddr(name: str) -> tuple[int, int]:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint '{name}' not in MJCF")
        return int(mj_model.jnt_qposadr[jid]), int(mj_model.jnt_dofadr[jid])

    qadr_all = {n: jaddr(n) for (n, _) in ALL_JOINT_NAMES_DEFAULTS}
    leg_qpos_idx = np.array([qadr_all[n][0] for n in POLICY_LEG_JOINTS], dtype=np.int64)
    leg_qvel_idx = np.array([qadr_all[n][1] for n in POLICY_LEG_JOINTS], dtype=np.int64)

    # actuator name -> ctrl index
    def aid(name: str) -> int:
        a = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a < 0:
            raise RuntimeError(f"actuator '{name}' not in MJCF")
        return int(a)
    leg_ctrl_idx = np.array([aid(n) for n in POLICY_LEG_JOINTS], dtype=np.int64)
    all_ctrl_idx = {n: aid(n) for (n, _) in ALL_JOINT_NAMES_DEFAULTS}

    # ----- initial state: place K1 at standing pose, arms held -----
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = np.array([0.0, 0.0, args.initial_z])
    mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    for n, def_pos in ALL_JOINT_NAMES_DEFAULTS:
        qadr, _ = qadr_all[n]
        mj_data.qpos[qadr] = def_pos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # ----- renderer (offscreen, EGL) -----
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 110.0
    cam.elevation = -15.0
    cam.distance = 3.2
    cam.lookat[:] = mj_data.qpos[0:3]

    # ----- ffmpeg writer over stdin -----
    if os.path.exists(args.output):
        os.remove(args.output)
    ff_cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{args.width}x{args.height}",
        "-pix_fmt", "rgb24",
        "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20",
        args.output,
    ]
    print(f"[ffmpeg] writing -> {args.output}", file=sys.stderr)
    ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    # ----- run loop -----
    last_action = np.zeros(12, dtype=np.float32)
    height_scan = np.ones(187, dtype=np.float32)   # saturated at +1 in training (offset of 20m raycaster + clip)

    # Stats accumulators
    trajectory_pos = []        # (t, x, y, z)
    trajectory_yaw = []        # (t, yaw_world)
    trajectory_cmd_lin = []    # (t, cmd_vx, actual_vx_world, actual_vx_robot)
    trajectory_cmd_yaw = []    # (t, cmd_wz, actual_wz)
    upright_violations = 0
    upright_threshold = 1.3    # rad - matches training bad_orientation
    fallen = False
    fall_time = None

    # Warmup with zero command so the standing pose settles.
    sim_time = 0.0
    for _ in range(args.warmup_steps):
        cmd = np.zeros(3, dtype=np.float32)
        last_action = step_policy(actor, mj_model, mj_data, cmd, last_action,
                                  leg_qpos_idx, leg_qvel_idx, leg_ctrl_idx,
                                  all_ctrl_idx, height_scan)
        # do not record the warmup frames; just advance state

    # Re-zero trajectory start
    start_xyz = mj_data.qpos[0:3].copy()
    start_yaw = quat_to_yaw(mj_data.qpos[3:7])

    sim_time = 0.0
    total_time = 15.0
    step_count = 0
    record_every = 1   # record every policy step (50 fps if policy_dt=20ms)

    while sim_time < total_time:
        cmd = vel_command_at(sim_time)
        last_action = step_policy(actor, mj_model, mj_data, cmd, last_action,
                                  leg_qpos_idx, leg_qvel_idx, leg_ctrl_idx,
                                  all_ctrl_idx, height_scan)
        sim_time += PHYSICS_DT * DECIMATION
        step_count += 1

        # log stats
        xyz = mj_data.qpos[0:3].copy()
        quat = mj_data.qpos[3:7].copy()
        yaw = quat_to_yaw(quat)
        lin_vel_w = mj_data.qvel[0:3].copy()
        ang_vel_w = mj_data.qvel[3:6].copy()
        proj_grav = quat_apply_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))
        upright_angle = math.acos(max(-1.0, min(1.0, -proj_grav[2])))   # 0 = perfectly upright

        # forward speed in robot's body frame
        # rotate world lin_vel by inverse yaw rotation:
        actual_vx_b =  math.cos(-yaw) * lin_vel_w[0] - math.sin(-yaw) * lin_vel_w[1]
        actual_vy_b =  math.sin(-yaw) * lin_vel_w[0] + math.cos(-yaw) * lin_vel_w[1]

        trajectory_pos.append((sim_time, *xyz))
        trajectory_yaw.append((sim_time, yaw))
        trajectory_cmd_lin.append((sim_time, cmd[0], lin_vel_w[0], actual_vx_b))
        trajectory_cmd_yaw.append((sim_time, cmd[2], ang_vel_w[2]))
        if upright_angle > upright_threshold and not fallen:
            fallen = True
            fall_time = sim_time
        if upright_angle > upright_threshold:
            upright_violations += 1

        # render every step at 50 fps
        if step_count % record_every == 0:
            # follow the robot with a soft tracking camera
            cam.lookat[:] = xyz
            renderer.update_scene(mj_data, camera=cam)
            pixels = renderer.render()   # (h, w, 3) uint8
            pixels = pixels.copy()
            # Overlay text using OpenCV (BGR space; flip rgb to bgr, write, flip back).
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"t={sim_time:5.2f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bgr, f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(bgr, f"actual vx_body={actual_vx_b:+.2f}  wz_world={ang_vel_w[2]:+.2f}",
                        (10, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)
            cv2.putText(bgr, f"trunk z={xyz[2]:+.2f}m  tilt={math.degrees(upright_angle):4.1f}deg",
                        (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 255) if fallen else (200, 200, 255), 2)
            if fallen:
                cv2.putText(bgr, "FALLEN", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            pixels = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                ff.stdin.write(pixels.tobytes())
            except BrokenPipeError:
                print("[ffmpeg] pipe broke", file=sys.stderr); break

    ff.stdin.close()
    ff.wait()
    renderer.close()

    # ----- compute stats per segment -----
    pos = np.array(trajectory_pos)         # (N, 4): t, x, y, z
    yaws = np.array(trajectory_yaw)        # (N, 2): t, yaw
    cmd_lin = np.array(trajectory_cmd_lin) # (N, 4): t, cmd_vx, vx_world, vx_body
    cmd_yaw = np.array(trajectory_cmd_yaw) # (N, 3): t, cmd_wz, wz_world

    def segment_mask(t0: float, t1: float) -> np.ndarray:
        return (pos[:, 0] >= t0) & (pos[:, 0] < t1)

    def yaw_change(t0: float, t1: float) -> float:
        m = segment_mask(t0, t1)
        if not m.any():
            return 0.0
        seg_yaw = yaws[m, 1]
        # unwrap
        seg_yaw = np.unwrap(seg_yaw)
        return float(seg_yaw[-1] - seg_yaw[0])

    print("\n=========  K1 VLN-CE policy MuJoCo test  =========")
    print(f"checkpoint : {CKPT_PATH}")
    print(f"iter       : {ck.get('iter')}")
    print(f"output mp4 : {args.output}")
    print(f"total sim  : 15.0 s, dt={PHYSICS_DT}, decimation={DECIMATION},"
          f" policy steps={len(pos)}")
    print(f"warmup     : {args.warmup_steps} zero-command policy steps before recording")
    print()
    print(f"trunk pos start  = ({start_xyz[0]:+.3f}, {start_xyz[1]:+.3f}, {start_xyz[2]:+.3f})")
    print(f"trunk pos end    = ({pos[-1, 1]:+.3f}, {pos[-1, 2]:+.3f}, {pos[-1, 3]:+.3f})")
    end_dist = float(np.linalg.norm(pos[-1, 1:4] - start_xyz))
    print(f"total displacement (Euclidean, x/y only) = "
          f"{math.hypot(pos[-1, 1]-start_xyz[0], pos[-1, 2]-start_xyz[1]):.3f} m")
    print()
    print("Per-segment summary:")
    for (name, t0, t1) in SEGMENTS:
        m = segment_mask(t0, t1)
        if not m.any():
            continue
        seg = pos[m]
        seg_cmd_lin = cmd_lin[m]
        seg_cmd_yaw = cmd_yaw[m]
        # displacement in x,y over segment
        dx = seg[-1, 1] - seg[0, 1]
        dy = seg[-1, 2] - seg[0, 2]
        d  = math.hypot(dx, dy)
        # mean actual_vx_body
        mean_vx_b = float(seg_cmd_lin[:, 3].mean())
        cmd_vx = float(seg_cmd_lin[:, 1].mean())
        cmd_wz = float(seg_cmd_yaw[:, 1].mean())
        mean_wz_w = float(seg_cmd_yaw[:, 2].mean())
        dyaw = yaw_change(t0, t1) * 180.0 / math.pi
        print(f"  [{t0:4.1f}-{t1:4.1f}s] {name:20s}  "
              f"dx,dy=({dx:+.2f},{dy:+.2f})m  dist={d:.2f}m  "
              f"vx_cmd={cmd_vx:+.2f}  mean_vx_body={mean_vx_b:+.2f}  "
              f"wz_cmd={cmd_wz:+.2f}  mean_wz={mean_wz_w:+.2f}  "
              f"yaw_change={dyaw:+.1f}deg")
    print()
    print(f"upright (|tilt| < {upright_threshold} rad) : "
          f"{'FELL at t={:.2f}s'.format(fall_time) if fallen else 'YES (stayed upright the whole run)'}")
    print(f"upright violations (steps over threshold) : {upright_violations}/{len(pos)}")
    print("====================================================\n")


def step_policy(actor: nn.Sequential, mj_model, mj_data, cmd: np.ndarray,
                last_action: np.ndarray, leg_qpos_idx: np.ndarray,
                leg_qvel_idx: np.ndarray, leg_ctrl_idx: np.ndarray,
                all_ctrl_idx: dict, height_scan: np.ndarray) -> np.ndarray:
    """Build obs, run actor, apply PD torque for DECIMATION sim steps."""
    # Build obs (235 dims). World/body frame conventions match Isaac Lab.
    quat = mj_data.qpos[3:7].astype(np.float32)        # (w,x,y,z)
    # MuJoCo stores qvel as world-frame linear vel + body-frame angular vel for free joint
    # Actually for free joint: qvel[0:3] = root_lin_vel_world, qvel[3:6] = root_ang_vel_local
    # Isaac Lab obs uses base_lin_vel (body frame) and base_ang_vel (body frame)
    lin_vel_w = mj_data.qvel[0:3].astype(np.float32)
    ang_vel_b = mj_data.qvel[3:6].astype(np.float32)   # already body frame in MuJoCo free joint
    lin_vel_b = quat_apply_inverse(quat, lin_vel_w)
    proj_grav = quat_apply_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))

    leg_pos = mj_data.qpos[leg_qpos_idx].astype(np.float32)
    leg_vel = mj_data.qvel[leg_qvel_idx].astype(np.float32)
    joint_pos_rel = leg_pos - LEG_DEFAULT_POS
    joint_vel_rel = leg_vel  # default joint vel is zero in training

    obs = np.concatenate([
        lin_vel_b,       # 0:3
        ang_vel_b,       # 3:6
        proj_grav,       # 6:9
        cmd.astype(np.float32),  # 9:12
        joint_pos_rel,   # 12:24
        joint_vel_rel,   # 24:36
        last_action,     # 36:48
        height_scan,     # 48:235
    ]).astype(np.float32)
    assert obs.shape == (235,), obs.shape

    with torch.no_grad():
        action = actor(torch.from_numpy(obs)).numpy().astype(np.float32)
    action = np.clip(action, -100.0, 100.0)

    # leg targets = default + action * action_scale
    leg_targets = LEG_DEFAULT_POS + action * ACTION_SCALE

    # PD torque computation for decimation steps; arms held by feedback to fixed defaults.
    for _ in range(DECIMATION):
        # Legs: PD to leg_targets
        cur_pos = mj_data.qpos[leg_qpos_idx].astype(np.float32)
        cur_vel = mj_data.qvel[leg_qvel_idx].astype(np.float32)
        tau_legs = LEG_KP * (leg_targets - cur_pos) - LEG_KD * cur_vel
        tau_legs = np.clip(tau_legs, -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
        mj_data.ctrl[leg_ctrl_idx] = tau_legs

        # Arms + head: hard PD to hold them rigidly at default. Training used
        # K1_locomotion.urdf with merge_fixed_joints=True, fusing the arms+head
        # into the Trunk rigid body. MuJoCo loads them as separate articulated
        # bodies, so flexible arms here add dynamics the policy never saw.
        # Use very stiff PD (kp=500, kd=15) clipped to MJCF actuator forcerange.
        for (name, def_pos) in ALL_JOINT_NAMES_DEFAULTS:
            if name in POLICY_LEG_JOINTS:
                continue
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qadr = int(mj_model.jnt_qposadr[jid]); vadr = int(mj_model.jnt_dofadr[jid])
            lim = 6.0 if "Head" in name else 14.0
            tau = 500.0 * (def_pos - float(mj_data.qpos[qadr])) - 15.0 * float(mj_data.qvel[vadr])
            mj_data.ctrl[all_ctrl_idx[name]] = float(np.clip(tau, -lim, lim))

        mujoco.mj_step(mj_model, mj_data)

    return action


if __name__ == "__main__":
    main()
