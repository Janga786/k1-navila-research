"""FAKE env + VLNEnvWrapperV3 for the offline test harness of navila_eval_v3_LOG.py. NOT Isaac.

Deterministic 'physics' (lagged unicycle driven by the velocity command, joints lagging the real
policy's targets) and fake renders: a smooth pattern of the pose whose brightness grows with the
distance walked, plus per-process noise of +-1 on ~15 % of channels (like the real renderer).
Knobs (environment variables):
  FAKE_CAM_HW="h,w"          camera size (default 90,160; the real camera is 720,1280)
  FAKE_RENDER_SEED=<int>     render-noise seed (default: the PID, i.e. differs per process)
  FAKE_PHYSICS_JITTER_AT=<t> add 1e-6 m to x after env step t (counted over all env steps)
"""
import math
import os
from types import SimpleNamespace as NS

import gymnasium as gym
import numpy as np
import torch

ASSETS_DIR = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/assets")
JOINTS = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
          "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"]
LEG_JOINTS = ["Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Hip_Roll", "Right_Hip_Roll", "Left_Hip_Yaw", "Right_Hip_Yaw",
              "Left_Knee_Pitch", "Right_Knee_Pitch", "Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll",
              "Right_Ankle_Roll"]


class _Data:
    def __init__(self, env):
        self._env = env
        self.joint_names = list(JOINTS)
        self.default_joint_pos = torch.zeros(1, 12)
        self.joint_pos_target = torch.zeros(1, 12)
        self.joint_pos = torch.zeros(1, 12)
        self.joint_vel = torch.zeros(1, 12)

    @property
    def root_state_w(self):
        e = self._env
        return torch.tensor([[e.x, e.y, e.z, math.cos(e.yaw / 2), 0.0, 0.0, math.sin(e.yaw / 2),
                              e.vx * math.cos(e.yaw), e.vx * math.sin(e.yaw), 0.0, 0.0, 0.0, e.wz]],
                            dtype=torch.float32)

    @property
    def root_pos_w(self):
        return self.root_state_w[:, 0:3]

    @property
    def root_quat_w(self):
        return self.root_state_w[:, 3:7]

    @property
    def root_vel_w(self):
        return self.root_state_w[:, 7:13]

    @property
    def root_ang_vel_b(self):
        return self.root_state_w[:, 10:13]

    @property
    def body_state_w(self):
        rs = self.root_state_w
        return rs.unsqueeze(1).repeat(1, 13, 1) + torch.arange(13, dtype=torch.float32).reshape(1, 13, 1) * 0.01


class _Robot:
    def __init__(self, env):
        self.data = _Data(env)


class FakeEnv(gym.Env):
    def __init__(self, cfg=None, render_mode=None):
        self.cfg = NS(sim=NS(dt=0.005), decimation=4, is_finite_horizon=False)
        self.device = "cpu"
        self.num_envs = 1
        self.sim = NS(set_camera_view=lambda eye, target: None)
        pos = cfg.scene.robot.init_state.pos
        rot = cfg.scene.robot.init_state.rot
        self.x0, self.y0 = float(pos[0]), float(pos[1])
        self.x, self.y, self.z = float(pos[0]), float(pos[1]), float(pos[2]) - 0.05
        self.yaw = math.atan2(2 * (rot[0] * rot[3] + rot[1] * rot[2]), 1 - 2 * (rot[2] ** 2 + rot[3] ** 2))
        self.vx = self.wz = 0.0
        self.t = 0
        self._vel_cmd = [0.0, 0.0, 0.0]
        self.is_stop_called = False
        self.scene = {"robot": _Robot(self)}
        h, w = os.environ.get("FAKE_CAM_HW", "90,160").split(",")
        self.h, self.w = int(h), int(w)
        self.rng = np.random.default_rng(int(os.environ.get("FAKE_RENDER_SEED", os.getpid())))
        self.jitter_at = int(os.environ.get("FAKE_PHYSICS_JITTER_AT", "-1"))
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        self._yy, self._xx = yy / self.h, xx / self.w

    @property
    def unwrapped(self):
        return self

    def _render(self):
        d = math.hypot(self.x - self.x0, self.y - self.y0)
        base = (100 + 15 * d + 8 * math.sin(2 * self.yaw) + 40 * np.sin(6.283 * 2 * self._xx + 3 * self.yaw) * np.cos(6.283 * self._yy + self.x)
                + 20 * np.sin(self.y + 9 * self._xx))
        img = np.stack([base, base * 0.9 + 10, base * 0.8 + 20], axis=-1)
        img = np.round(img)
        noise = (self.rng.random(img.shape) < 0.15) * self.rng.choice([-1.0, 1.0], size=img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        # 3 channels so the evaluator's [..., :3] slice stays contiguous on the CPU (on the real GPU
        # path .cpu() makes the copy; cv2.putText needs a contiguous array)
        return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)

    def _obs(self):
        return {"policy": torch.zeros(1, 235), "camera_obs": self._render(),
                "viz_camera_obs": torch.full((1, 64, 64, 3), 200, dtype=torch.uint8)}

    def reset(self, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        vx, _, wz = self._vel_cmd
        dt = 0.02
        self.vx += 0.2 * (vx - self.vx)
        self.wz += 0.2 * (wz - self.wz)
        self.yaw += self.wz * dt
        self.x += self.vx * math.cos(self.yaw) * dt
        self.y += self.vx * math.sin(self.yaw) * dt
        self.t += 1
        if self.t == self.jitter_at:
            self.x += 1e-6
        d = self.scene["robot"].data
        d.joint_pos_target = (action * 0.5).float()
        new = d.joint_pos + 0.5 * (d.joint_pos_target - d.joint_pos)
        d.joint_vel = (new - d.joint_pos) / dt
        d.joint_pos = new
        return self._obs(), torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}


class VLNEnvWrapperV3:
    """Mirrors the real wrapper's interface (reset with a 50-step internal warm-up, step(cmd),
    _history_buf / _last_action / _gait_phase / _command, measurements, set_stop_called)."""

    def __init__(self, env, low_level_policy, task_name, episode, max_length=10000,
                 high_level_obs_key="camera_obs", gait_phase_init=0.0, measure_names=None):
        self.env = env
        self.low_level_policy = low_level_policy
        self.episode = episode
        self.high_level_obs_key = high_level_obs_key
        self.is_stop_called = False
        self._history_buf = None
        self._last_action = torch.zeros(1, 12)
        self._command = torch.zeros(1, 3)
        self._gait_phase_init = float(gait_phase_init)
        self._gait_phase = torch.full((1,), self._gait_phase_init)
        self._jperm = torch.tensor([JOINTS.index(n) for n in LEG_JOINTS])
        self.goal = episode["reference_path"][-1]

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _per_step_obs(self):
        d = self.unwrapped.scene["robot"].data
        angle = 2.0 * math.pi * self._gait_phase
        gait = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        if float(self._command.norm()) < 0.1:
            gait = torch.zeros_like(gait)
        return torch.cat([self._command, gait, torch.tensor([[0.0, 0.0, -1.0]]), d.root_ang_vel_b * 0.25,
                          d.joint_pos[:, self._jperm], d.joint_vel[:, self._jperm] * 0.1, self._last_action], dim=-1)

    def _build_policy_input(self):
        obs = self._per_step_obs()
        if self._history_buf is None:
            self._history_buf = obs.unsqueeze(1).expand(-1, 5, -1).clone()
        else:
            self._history_buf = torch.roll(self._history_buf, shifts=-1, dims=1)
            self._history_buf[:, -1, :] = obs
        offs = [0, 3, 5, 8, 11, 23, 35, 47]
        return torch.cat([self._history_buf[:, :, offs[i]:offs[i + 1]].reshape(1, -1) for i in range(7)], dim=-1)

    def _advance(self):
        action = self.low_level_policy(self._build_policy_input())
        self._gait_phase = (self._gait_phase + 1.5 * 0.02) % 1.0
        self._last_action = action
        full = torch.zeros_like(action)
        full[:, self._jperm] = action
        self.unwrapped._vel_cmd = [float(c) for c in self._command[0]]
        return self.env.step(full * 0.5)

    def _measure(self):
        e = self.unwrapped
        d = math.hypot(e.x - self.goal[0], e.y - self.goal[1])
        if self._prev is not None:
            self._pl += math.hypot(e.x - self._prev[0], e.y - self._prev[1])
        self._prev = (e.x, e.y)
        self._one = min(self._one, d)
        succ = float(self.is_stop_called and d < 3.0)
        return {"path_length": self._pl, "distance_to_goal": d, "success": succ, "spl": succ,
                "oracle_navigation_error": self._one, "oracle_success": float(self._one < 3.0)}

    def reset(self):
        _, infos = self.env.reset()
        self._history_buf = None
        self._last_action = torch.zeros(1, 12)
        self._command = torch.zeros(1, 3)
        self._gait_phase = torch.full((1,), self._gait_phase_init)
        for _ in range(50):
            _, _, _, infos = self._advance()
        self._pl, self._prev, self._one = 0.0, None, float("inf")
        infos["measurements"] = self._measure()
        return infos["observations"][self.high_level_obs_key], infos

    def step(self, action):
        self._command = (action if torch.is_tensor(action) else torch.tensor(action)).float().reshape(1, 3).clone()
        _, reward, done, info = self._advance()
        info["measurements"] = self._measure()
        return info["observations"][self.high_level_obs_key], reward, bool(done[0]) or self.is_stop_called, info

    def set_stop_called(self, v):
        self.unwrapped.is_stop_called = v
        self.is_stop_called = v

    def close(self):
        pass
