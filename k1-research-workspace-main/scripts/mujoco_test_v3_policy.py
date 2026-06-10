"""
Run the K1 VLN-CE-v3 policy (model_16000.pt) in MuJoCo on flat ground,
send a deterministic velocity-command sequence, record an mp4, and print
quantitative stats.

v3 obs layout (235 dim flattened, term-major from a 5-frame history of
47-dim per-step observations):

  per-step: cmd(3) | gait_phase(2) | gravity(3) | ang_vel(3)
            | jp_rel(12) | jv(12) | last_action(12)   = 47
  history : 5 frames, term-major flatten
            → [cmd_t0..t4 | gait_t0..t4 | grav_t0..t4 | angv_t0..t4
               | jp_t0..t4 | jv_t0..t4 | la_t0..t4]
            → 47 * 5 = 235

Joint order (URDF order, preserve_order=False):
  [LHP, LHR, LHY, LK, LAP, LAR, RHP, RHR, RHY, RK, RAP, RAR]

Action scale = 0.25 (vs v2's 0.5). Gait clock at 2 Hz.
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

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco


MJCF_PATH = (
    "/home/janga/Projects/k1_research/booster/booster_assets/robots/K1/K1_locomotion.xml"
)
CKPT_PATH = (
    "/home/janga/Projects/k1_research/booster/booster_train/logs/rsl_rl/"
    "k1_velocity_vlnce_v3/2026-05-19_10-38-52_k1_vlnce_v3_resumed/model_16000.pt"
)

PHYSICS_DT = 0.005
DECIMATION = 4
ACTION_SCALE = 0.25       # v3 action_scale (v2 used 0.5)
GAIT_FREQ_HZ = 2.0
HISTORY_LENGTH = 5

# per-step term dims, in PolicyCfg declaration order
_TERM_DIMS = (3, 2, 3, 3, 12, 12, 12)  # cmd, gait, grav, ang_vel, jp, jv, last_a
_PER_STEP_DIM = sum(_TERM_DIMS)        # 47
assert _PER_STEP_DIM * HISTORY_LENGTH == 235

# 12 leg joints in URDF/articulation order.
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
LEG_KP = np.array([350, 350, 350, 350, 250, 250,
                   350, 350, 350, 350, 250, 250], dtype=np.float32)
LEG_KD = np.array([7.5, 7.5, 7.5, 7.5, 5.0, 5.0,
                   7.5, 7.5, 7.5, 7.5, 5.0, 5.0], dtype=np.float32)
LEG_EFFORT_LIMIT = np.array(
    [30, 35, 20, 60, 20, 20,
     30, 35, 20, 60, 20, 20], dtype=np.float32
)

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


def build_actor_from_state_dict(state_dict, in_dim, hidden, out_dim) -> nn.Sequential:
    layers = []
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
    actor.load_state_dict(mapped, strict=True)
    actor.eval()
    return actor


def quat_apply_inverse(q_wxyz, v):
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return (v - w * t + np.cross(qv, t)).astype(np.float32)


def quat_to_yaw(q_wxyz):
    w, x, y, z = q_wxyz
    return float(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def vel_command_at(t):
    if t < 5.0:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
    if t < 8.0:
        return np.array([0.0, 0.0, 0.524], dtype=np.float32)
    if t < 13.0:
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


SEGMENTS = [
    ("vx=0.5 forward",   0.0,  5.0),
    ("vyaw=0.524 turn",  5.0,  8.0),
    ("vx=0.5 forward",   8.0, 13.0),
    ("zero (stop)",     13.0, 15.0),
]


def build_per_step_obs(cmd, gait_t, mj_data, leg_qpos_idx, leg_qvel_idx, last_action):
    quat = mj_data.qpos[3:7].astype(np.float32)
    ang_vel_b = mj_data.qvel[3:6].astype(np.float32)   # body frame (free joint)
    proj_grav = quat_apply_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))
    leg_pos = mj_data.qpos[leg_qpos_idx].astype(np.float32)
    leg_vel = mj_data.qvel[leg_qvel_idx].astype(np.float32)
    jp_rel = leg_pos - LEG_DEFAULT_POS

    angle = 2.0 * math.pi * gait_t
    gait = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    return np.concatenate([
        cmd.astype(np.float32),  # 3
        gait,                    # 2
        proj_grav,               # 3
        ang_vel_b,               # 3
        jp_rel,                  # 12
        leg_vel,                 # 12
        last_action,             # 12
    ])


def flatten_history_term_major(buf):
    """buf: (HISTORY_LENGTH, _PER_STEP_DIM). Output: (235,) term-major."""
    parts = []
    s = 0
    for d in _TERM_DIMS:
        # buf[:, s:s+d] is (H, d); flatten gives [term_t0..t(H-1)] = d*H values
        parts.append(buf[:, s:s+d].reshape(-1))
        s += d
    return np.concatenate(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="/home/janga/Desktop/k1_v3_locomotion_mujoco.mp4")
    p.add_argument("--mjcf", default=MJCF_PATH)
    p.add_argument("--ckpt", default=CKPT_PATH)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--initial_z", type=float, default=0.55)
    p.add_argument("--total_time", type=float, default=15.0)
    p.add_argument("--gait_phase_init", type=float, default=0.0,
                   help="Initial gait phase offset (matches training's per-env "
                        "random offset). Try 0.0, 0.25, 0.5, 0.75.")
    p.add_argument("--seed", type=int, default=0,
                   help="If nonzero, set numpy seed and use a random gait phase.")
    args = p.parse_args()
    if args.seed != 0:
        np.random.seed(args.seed)
        args.gait_phase_init = float(np.random.rand())

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    actor = build_actor_from_state_dict(sd, in_dim=235,
                                         hidden=[512, 256, 128], out_dim=12)
    print(f"[policy] loaded {args.ckpt}", file=sys.stderr)
    print(f"[policy] iter={ck.get('iter')}  params={sum(p.numel() for p in actor.parameters())}",
          file=sys.stderr)

    mj_model = mujoco.MjModel.from_xml_path(args.mjcf)
    print(f"[mjcf] {args.mjcf}", file=sys.stderr)
    print(f"[mjcf] njnt={mj_model.njnt} nu={mj_model.nu} nbody={mj_model.nbody} "
          f"mass={sum(mj_model.body_mass):.3f}kg", file=sys.stderr)
    mj_model.opt.timestep = PHYSICS_DT
    # Bump ground friction to match training (training used 1.0 static/dynamic;
    # MJCF default ground is 0.4 — that slipperier ground spins the K1 out).
    gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if gid >= 0:
        mj_model.geom_friction[gid] = np.array([1.0, 0.005, 0.0001], dtype=np.float64)
    # Note: training URDF used merge_fixed_joints=True which fused arms+head
    # into the Trunk rigid body. MuJoCo's K1_22dof.xml leaves them articulated.
    # We hold them with stiff PD (kp=500, kd=15) in the policy loop below;
    # there's still residual arm-swing inertia the policy never saw at train
    # time. This is the dominant sim-to-sim gap.
    mj_data = mujoco.MjData(mj_model)

    def jaddr(name):
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint '{name}' not found")
        return int(mj_model.jnt_qposadr[jid]), int(mj_model.jnt_dofadr[jid])

    # Joint lookup. If a joint isn't in the MJCF (e.g., K1_locomotion.xml has
    # arms/head welded to Trunk and no joint elements), we just skip it.
    qadr_all = {}
    for (n, _) in ALL_JOINT_NAMES_DEFAULTS:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            qadr_all[n] = (int(mj_model.jnt_qposadr[jid]),
                           int(mj_model.jnt_dofadr[jid]))
    missing_leg = [n for n in POLICY_LEG_JOINTS if n not in qadr_all]
    if missing_leg:
        raise RuntimeError(f"leg joints missing from MJCF: {missing_leg}")
    leg_qpos_idx = np.array([qadr_all[n][0] for n in POLICY_LEG_JOINTS], dtype=np.int64)
    leg_qvel_idx = np.array([qadr_all[n][1] for n in POLICY_LEG_JOINTS], dtype=np.int64)

    def aid(name):
        a = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        return int(a)  # may be -1 if absent
    leg_ctrl_idx = np.array([aid(n) for n in POLICY_LEG_JOINTS], dtype=np.int64)
    if (leg_ctrl_idx < 0).any():
        raise RuntimeError(f"leg actuators missing from MJCF "
                           f"(indices: {leg_ctrl_idx.tolist()})")
    all_ctrl_idx = {n: aid(n) for (n, _) in ALL_JOINT_NAMES_DEFAULTS}

    # Stiff PD arrays for arms/head (held at training defaults). With the
    # K1_locomotion MJCF the arm/head joints don't exist, so this list is
    # empty and the arm-PD inner loop becomes a no-op.
    arm_head_info = []
    for (name, def_pos) in ALL_JOINT_NAMES_DEFAULTS:
        if name in POLICY_LEG_JOINTS:
            continue
        if name not in qadr_all or all_ctrl_idx[name] < 0:
            continue
        qadr, vadr = qadr_all[name]
        lim = 6.0 if "Head" in name else 14.0
        arm_head_info.append((qadr, vadr, all_ctrl_idx[name], def_pos, lim))
    if not arm_head_info:
        print("[info] no arm/head joints in MJCF — arms are welded to Trunk",
              file=sys.stderr)

    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = np.array([0.0, 0.0, args.initial_z])
    mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    for n, def_pos in ALL_JOINT_NAMES_DEFAULTS:
        if n not in qadr_all:
            continue  # joint welded away in K1_locomotion.xml
        qadr, _ = qadr_all[n]
        mj_data.qpos[qadr] = def_pos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Renderer
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 110.0
    cam.elevation = -15.0
    cam.distance = 3.2
    cam.lookat[:] = mj_data.qpos[0:3]

    # ffmpeg writer
    if os.path.exists(args.output):
        os.remove(args.output)
    ff_cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{args.width}x{args.height}",
        "-pix_fmt", "rgb24", "-r", str(args.fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "20",
        args.output,
    ]
    print(f"[ffmpeg] writing -> {args.output}", file=sys.stderr)
    ff = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    # State
    last_action = np.zeros(12, dtype=np.float32)
    history_buf: np.ndarray | None = None  # (H, 47)
    gait_phase = float(args.gait_phase_init)
    print(f"[gait] initial phase = {gait_phase:.3f}", file=sys.stderr)

    trajectory_pos = []
    trajectory_yaw = []
    trajectory_cmd_lin = []
    trajectory_cmd_yaw = []
    upright_violations = 0
    upright_threshold = 1.3
    fallen = False
    fall_time = None

    POLICY_DT = PHYSICS_DT * DECIMATION   # 0.02 s

    def policy_step(cmd):
        nonlocal last_action, history_buf, gait_phase
        obs = build_per_step_obs(cmd, gait_phase, mj_data,
                                  leg_qpos_idx, leg_qvel_idx, last_action)
        if history_buf is None:
            history_buf = np.tile(obs[None, :], (HISTORY_LENGTH, 1))
        else:
            history_buf = np.roll(history_buf, shift=-1, axis=0)
            history_buf[-1] = obs
        flat = flatten_history_term_major(history_buf)
        assert flat.shape == (235,), flat.shape

        with torch.no_grad():
            action = actor(torch.from_numpy(flat)).numpy().astype(np.float32)
        action = np.clip(action, -100.0, 100.0)

        leg_targets = LEG_DEFAULT_POS + action * ACTION_SCALE

        for _ in range(DECIMATION):
            cur_pos = mj_data.qpos[leg_qpos_idx].astype(np.float32)
            cur_vel = mj_data.qvel[leg_qvel_idx].astype(np.float32)
            tau_legs = LEG_KP * (leg_targets - cur_pos) - LEG_KD * cur_vel
            tau_legs = np.clip(tau_legs, -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
            mj_data.ctrl[leg_ctrl_idx] = tau_legs

            # Arms/head stiff PD to defaults
            for qadr, vadr, cidx, def_pos, lim in arm_head_info:
                tau = 500.0 * (def_pos - float(mj_data.qpos[qadr])) \
                      - 15.0 * float(mj_data.qvel[vadr])
                mj_data.ctrl[cidx] = float(np.clip(tau, -lim, lim))

            mujoco.mj_step(mj_model, mj_data)

        last_action = action
        gait_phase = (gait_phase + GAIT_FREQ_HZ * POLICY_DT) % 1.0
        return action

    # Warmup with zero command if requested (not recorded)
    for _ in range(args.warmup_steps):
        policy_step(np.zeros(3, dtype=np.float32))

    start_xyz = mj_data.qpos[0:3].copy()
    start_yaw = quat_to_yaw(mj_data.qpos[3:7])

    sim_time = 0.0
    step_count = 0

    while sim_time < args.total_time:
        cmd = vel_command_at(sim_time)
        policy_step(cmd)
        sim_time += POLICY_DT
        step_count += 1

        xyz = mj_data.qpos[0:3].copy()
        quat = mj_data.qpos[3:7].copy()
        yaw = quat_to_yaw(quat)
        lin_vel_w = mj_data.qvel[0:3].copy()
        ang_vel_w = mj_data.qvel[3:6].copy()
        proj_grav = quat_apply_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))
        upright_angle = math.acos(max(-1.0, min(1.0, -proj_grav[2])))

        actual_vx_b = math.cos(-yaw) * lin_vel_w[0] - math.sin(-yaw) * lin_vel_w[1]

        trajectory_pos.append((sim_time, *xyz))
        trajectory_yaw.append((sim_time, yaw))
        trajectory_cmd_lin.append((sim_time, cmd[0], lin_vel_w[0], actual_vx_b))
        trajectory_cmd_yaw.append((sim_time, cmd[2], ang_vel_w[2]))
        if upright_angle > upright_threshold and not fallen:
            fallen = True
            fall_time = sim_time
        if upright_angle > upright_threshold:
            upright_violations += 1

        cam.lookat[:] = xyz
        renderer.update_scene(mj_data, camera=cam)
        pixels = renderer.render().copy()
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"t={sim_time:5.2f}s  v3 model_16000.pt", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bgr, f"cmd  vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(bgr, f"actual vx_body={actual_vx_b:+.2f}  wz={ang_vel_w[2]:+.2f}",
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
            break

    ff.stdin.close(); ff.wait(); renderer.close()

    pos = np.array(trajectory_pos)
    yaws = np.array(trajectory_yaw)
    cmd_lin = np.array(trajectory_cmd_lin)
    cmd_yaw = np.array(trajectory_cmd_yaw)

    def segment_mask(t0, t1):
        return (pos[:, 0] >= t0) & (pos[:, 0] < t1)

    def yaw_change(t0, t1):
        m = segment_mask(t0, t1)
        if not m.any(): return 0.0
        return float(np.unwrap(yaws[m, 1])[-1] - np.unwrap(yaws[m, 1])[0])

    print("\n=========  K1 VLN-CE v3 policy MuJoCo test  =========")
    print(f"checkpoint : {args.ckpt}")
    print(f"iter       : {ck.get('iter')}")
    print(f"output mp4 : {args.output}")
    print(f"total sim  : {args.total_time}s  dt={PHYSICS_DT}  dec={DECIMATION}  "
          f"policy steps={len(pos)}")
    print(f"warmup     : {args.warmup_steps}")
    print()
    print(f"trunk pos start = ({start_xyz[0]:+.3f}, {start_xyz[1]:+.3f}, {start_xyz[2]:+.3f})")
    print(f"trunk pos end   = ({pos[-1, 1]:+.3f}, {pos[-1, 2]:+.3f}, {pos[-1, 3]:+.3f})")
    print(f"total displacement (x/y) = "
          f"{math.hypot(pos[-1, 1]-start_xyz[0], pos[-1, 2]-start_xyz[1]):.3f} m")
    print()
    print("Per-segment summary:")
    for (name, t0, t1) in SEGMENTS:
        m = segment_mask(t0, t1)
        if not m.any(): continue
        seg = pos[m]; sc = cmd_lin[m]; sy = cmd_yaw[m]
        dx = seg[-1, 1] - seg[0, 1]; dy = seg[-1, 2] - seg[0, 2]
        d = math.hypot(dx, dy)
        mean_vx_b = float(sc[:, 3].mean())
        cmd_vx = float(sc[:, 1].mean())
        cmd_wz = float(sy[:, 1].mean())
        mean_wz = float(sy[:, 2].mean())
        dyaw = yaw_change(t0, t1) * 180.0 / math.pi
        print(f"  [{t0:4.1f}-{t1:4.1f}s] {name:20s}  "
              f"dx,dy=({dx:+.2f},{dy:+.2f})m  dist={d:.2f}m  "
              f"vx_cmd={cmd_vx:+.2f}  mean_vx_body={mean_vx_b:+.2f}  "
              f"wz_cmd={cmd_wz:+.2f}  mean_wz={mean_wz:+.2f}  "
              f"yaw={dyaw:+.1f}deg")
    print()
    fall_str = 'FELL at t={:.2f}s'.format(fall_time) if fallen else 'YES (stayed upright the whole run)'
    print(f"upright (|tilt|<{upright_threshold}rad): {fall_str}")
    print(f"upright violations: {upright_violations}/{len(pos)}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
