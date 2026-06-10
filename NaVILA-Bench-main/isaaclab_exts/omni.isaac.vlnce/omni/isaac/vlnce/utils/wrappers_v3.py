# SPDX-License-Identifier: BSD-3-Clause
"""
VLNEnvWrapperV3 — drives the K1 VLN-CE v3 policy from the NaVILA-Bench
benchmark loop.

The v3 policy was trained with a 47-dim per-step observation +
5-frame history, term-major flattened to 235 dims:

  per-step (in order):
    velocity_commands(3)  gait_phase(2)  projected_gravity(3)
    base_ang_vel(3)  joint_pos_rel(12)  joint_vel_rel(12)  last_action(12)
  history = 5 frames
  flatten = [cmd_t0..t4 | gait_t0..t4 | grav_t0..t4 | angv_t0..t4
             | jp_t0..t4 | jv_t0..t4 | la_t0..t4]   (235 dim)

action_scale was 0.25 in training. The bench's existing
k1_matterport_vision env uses scale=0.5, so we pre-divide the policy
action by 2 before calling env.step(), giving an effective scale of
0.25. (action × 0.5 by env × 0.5 by us = action × 0.25.)

This wrapper IGNORES the env's "policy" observation group (the 235-dim
single-step layout from k1_matterport_vision_cfg.py); it constructs the
v3 obs directly from scene state. The env is used only as a physics
engine + Matterport scene + camera renderer + measurement manager.
"""

from __future__ import annotations

import math
from collections import deque

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from .measures import add_measurement


def _yaw_from_quat_w(q: torch.Tensor) -> float:
    """world yaw from a (w,x,y,z) quaternion tensor."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


GAIT_FREQ_HZ = 1.5   # PATCHED for model_8700 (was 2.0): contract gait clock is 1.5 Hz
HISTORY_LENGTH = 5
PER_STEP_DIM = 47   # 3+2+3+3+12+12+12
ACTION_SCALE_RATIO = 0.5   # v3 wants 0.25; env uses 0.5 → pre-divide by 2

# ---- PATCHED for model_8700 (the lab-box velocity policy) ----
# Trained with the 12 leg joints in this exact interleaved L/R order (preserve_order=True
# in navila_env_cfg.py). The benchmark articulation may enumerate joints in a different
# internal order, so we permute obs reads into this order and scatter the action back —
# correct regardless of the articulation's native ordering.
LEG_JOINTS = [
    "Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Hip_Roll", "Right_Hip_Roll",
    "Left_Hip_Yaw", "Right_Hip_Yaw", "Left_Knee_Pitch", "Right_Knee_Pitch",
    "Left_Ankle_Pitch", "Right_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Roll",
]
ANGVEL_SCALE = 0.25   # PATCHED: contract base_ang_vel obs scale (benchmark fed it raw)
JVEL_SCALE = 0.1      # PATCHED: contract joint_vel obs scale (benchmark fed it raw)


def quat_apply_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Passive rotation by q^-1 applied to vector v. q_wxyz: (..., 4),
    v: (..., 3). Returns (..., 3)."""
    w = q_wxyz[..., 0:1]
    qv = q_wxyz[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v - w * t + torch.cross(qv, t, dim=-1)


class VLNEnvWrapperV3:
    """Drives the K1 v3 policy through the bench's Matterport env.

    Use with a single env (num_envs=1). Maintains an internal 47-dim ×
    5-frame history buffer and feeds the policy the term-major flattened
    235-dim observation each step.
    """

    def __init__(
        self,
        env,                         # already wrapped with RslRlVecEnvWrapper
        low_level_policy,
        task_name: str,
        episode,
        max_length: int = 10000,
        high_level_obs_key: str = "camera_obs",
        gait_phase_init: float = 0.0,
        measure_names=("PathLength", "DistanceToGoal", "Success",
                       "SPL", "OracleNavigationError", "OracleSuccess"),
    ):
        self.env = env
        self.task_name = task_name
        self.episode = episode
        self.high_level_obs_key = high_level_obs_key
        self.max_length = max_length
        self.measure_names = list(measure_names)

        self.low_level_policy = low_level_policy
        self.low_level_action = None

        self.env_step = 0
        self.same_pos_count = 0
        self.curr_pos = None
        self.prev_pos = None
        self.is_stop_called = False
        # windowed hopeless-stall buffer: (x, y, yaw) per commanded step, 500 = 10 s
        self._stall_buf: deque = deque(maxlen=500)

        # v3 obs / gait state, allocated lazily on first reset() since
        # the env hasn't computed obs yet at construct time.
        self.device = self.env.unwrapped.device
        self.num_envs = self.env.unwrapped.num_envs
        self._history_buf = None     # (num_envs, HISTORY_LENGTH, PER_STEP_DIM)
        self._jperm = None           # PATCHED: articulation-order -> LEG_JOINTS-order index map
        self._last_action = torch.zeros(
            self.num_envs, 12, device=self.device, dtype=torch.float32
        )
        self._gait_phase_init = float(gait_phase_init)
        self._gait_phase = torch.full(
            (self.num_envs,), self._gait_phase_init,
            device=self.device, dtype=torch.float32,
        )
        # cmd is whatever the VLM most-recently said.
        self._command = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self._policy_dt = (
            self.env.unwrapped.cfg.sim.dt * self.env.unwrapped.cfg.decimation
        )

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.env.unwrapped

    def set_measures(self):
        self.measure_manager = add_measurement(
            self.env, self.episode, self.measure_names
        )

    # ---- v3 obs construction ----
    def _per_step_obs(self) -> torch.Tensor:
        """Build the 47-dim per-step v3 obs from raw scene state.
        Returns (num_envs, 47)."""
        robot = self.unwrapped.scene["robot"].data
        # PATCHED: build articulation-order -> LEG_JOINTS-order permutation once.
        if self._jperm is None:
            names = list(robot.joint_names)
            self._jperm = torch.tensor(
                [names.index(n) for n in LEG_JOINTS],
                device=self.device, dtype=torch.long,
            )
        # body-frame angular velocity (PATCHED: contract obs scale 0.25)
        ang_vel_b = robot.root_ang_vel_b * ANGVEL_SCALE      # (num_envs, 3)
        # projected_gravity: gravity in body frame
        # Isaac Lab convention: gravity_world = (0, 0, -1), projected = q^-1 * g
        gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, 3)
        proj_grav = quat_apply_inverse(robot.root_quat_w, gravity_w)
        # joint_pos_rel = current - default; joint_vel_rel = current vel
        # PATCHED: permute into LEG_JOINTS order; joint_vel scaled by 0.1 (contract)
        jp_rel = (robot.joint_pos - robot.default_joint_pos)[:, self._jperm]
        jv = robot.joint_vel[:, self._jperm] * JVEL_SCALE

        # gait_phase obs: cos/sin of 2π × phase
        angle = 2.0 * math.pi * self._gait_phase
        gait = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        # PATCHED: contract stand-gate — gait = [0,0] when ||cmd|| < 0.1 (stand vs walk)
        stand = self._command.norm(dim=-1, keepdim=True) < 0.1
        gait = torch.where(stand, torch.zeros_like(gait), gait)

        return torch.cat([
            self._command,    # 3
            gait,             # 2
            proj_grav,        # 3
            ang_vel_b,        # 3
            jp_rel,           # 12
            jv,               # 12
            self._last_action,  # 12
        ], dim=-1)

    def _flatten_history_term_major(self) -> torch.Tensor:
        """history_buf: (N, H, 47). Return (N, 235) term-major flatten."""
        # term boundaries: (3, 2, 3, 3, 12, 12, 12)
        offsets = [0, 3, 5, 8, 11, 23, 35, 47]
        parts = []
        for i in range(7):
            s, e = offsets[i], offsets[i + 1]
            # buf[:, :, s:e] is (N, H, d); reshape to (N, H*d) row-major
            # → time outer, joint inner.
            parts.append(self._history_buf[:, :, s:e].reshape(self.num_envs, -1))
        return torch.cat(parts, dim=-1)

    def _build_policy_input(self) -> torch.Tensor:
        """Pull one per-step obs into the history buffer and return the
        flattened 235-dim policy input."""
        obs = self._per_step_obs()
        if self._history_buf is None:
            # Initialize buffer by replicating current obs in all H slots —
            # matches what booster_deploy's K1VelocityPolicy.inference does
            # on first step (and is reasonable for a clean reset).
            self._history_buf = obs.unsqueeze(1).expand(
                -1, HISTORY_LENGTH, -1
            ).clone()
        else:
            self._history_buf = torch.roll(
                self._history_buf, shifts=-1, dims=1
            )
            self._history_buf[:, -1, :] = obs
        return self._flatten_history_term_major()

    # ---- gym-ish API ----
    def reset(self):
        low_level_obs, infos = self.env.reset()
        # First-reset bookkeeping: state buffers + gait clock + history.
        self._history_buf = None
        self._last_action.zero_()
        self._command.zero_()
        self._gait_phase.fill_(self._gait_phase_init)  # t=0 gait clock each episode (deploy contract, rank 10)
        self._stall_buf.clear()

        # Warmup: same as the original VLNEnvWrapper. Step with zero cmd
        # for the task-specific number of frames so the VLM image buffer
        # has real (not black) frames before NaVILA's first call.
        if "go2" in self.task_name:
            warmup_steps = 100
        elif "h1" in self.task_name or "g1" in self.task_name:
            warmup_steps = 200
        else:
            warmup_steps = 50
        for i in range(warmup_steps):
            if i % 100 == 0 or i == warmup_steps - 1:
                print(f"[v3-wrapper] warmup step {i}/{warmup_steps}")
            policy_input = self._build_policy_input()
            action = self.low_level_policy(policy_input)
            # advance gait clock
            self._gait_phase = (
                self._gait_phase + GAIT_FREQ_HZ * self._policy_dt
            ) % 1.0
            self._last_action = action
            # PATCHED: scatter LEG_JOINTS-order action into articulation order, then pre-divide
            full = torch.zeros_like(action)
            full[:, self._jperm] = action
            scaled = full * ACTION_SCALE_RATIO
            low_level_obs, _, _, infos = self.env.step(scaled)

        self.env_step = 0
        self.same_pos_count = 0
        self.set_measures()
        self.measure_manager.reset_measures()
        infos["measurements"] = self.measure_manager.get_measurements()
        self.prev_pos = (
            self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        )
        obs = infos["observations"][self.high_level_obs_key]
        return obs, infos

    def update_command(self, command) -> None:
        """High-level planner pushes a (vx, vy, wz) command. We just store
        it; it shows up in the next per-step obs at term[0]."""
        if not torch.is_tensor(command):
            command = torch.tensor(
                command, device=self.device, dtype=torch.float32
            )
        if command.dim() == 1:
            command = command.unsqueeze(0).expand(self.num_envs, -1)
        self._command = command.clone()

    def step(self, action):
        """High-level action = velocity command. The low-level policy is
        called internally and produces the joint action."""
        self.update_command(action)
        policy_input = self._build_policy_input()
        low_level_action = self.low_level_policy(policy_input)
        self._gait_phase = (
            self._gait_phase + GAIT_FREQ_HZ * self._policy_dt
        ) % 1.0
        self._last_action = low_level_action
        # PATCHED: scatter LEG_JOINTS-order action into articulation order, then pre-divide
        full = torch.zeros_like(low_level_action)
        full[:, self._jperm] = low_level_action
        scaled = full * ACTION_SCALE_RATIO
        low_level_obs, reward, done, info = self.env.step(scaled)
        obs = info["observations"][self.high_level_obs_key]
        self.env_step += 1
        self.measure_manager.update_measures()
        info["measurements"] = self.measure_manager.get_measurements()
        same_pos = self.check_same_pos()
        done = done[0] or same_pos or self.env_step >= self.max_length \
            or self.is_stop_called
        return obs, reward, done, info

    def check_same_pos(self) -> bool:
        # Episode-level HOPELESS-STALL backstop (audit rank 5, validated 2026-06-10).
        # WINDOWED net progress over the last 500 commanded steps (10 s): a wall-jammed
        # K1 wobbles its base at ~1 rad/s and creeps ~0.05 m/s sideways (measured), so
        # instantaneous stillness gates NEVER fire on real jams — only integrated
        # progress separates them. Gates:
        #   - only while motion is COMMANDED (||cmd|| >= 0.1); zero-cmd (warmup, settle,
        #     stand) CLEARS the window so intentional stillness can never fire it.
        #   - hopeless = net displacement < 0.25 m AND net |yaw| < 30 deg over 10 s of
        #     continuous commands (legit walking covers ~4-5 m; a legit 90-deg turn is
        #     3x the yaw bar). The fast per-command stall-requery (3 s) lives in
        #     ClosedLoopController — THIS guard only ends truly-wedged episodes.
        curr_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].detach()
        curr_yaw = _yaw_from_quat_w(
            self.env.unwrapped.scene["robot"].data.root_quat_w[0].detach()
        )
        if torch.norm(self._command[0]) < 0.1:
            self._stall_buf.clear()
        else:
            self._stall_buf.append(
                (float(curr_pos[0]), float(curr_pos[1]), float(curr_yaw))
            )
        self.same_pos_count = len(self._stall_buf)  # kept for external introspection
        self.prev_pos = curr_pos
        if len(self._stall_buf) >= self._stall_buf.maxlen:
            x0, y0, w0 = self._stall_buf[0]
            x1, y1, w1 = self._stall_buf[-1]
            net_disp = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            net_yaw = abs((w1 - w0 + math.pi) % (2 * math.pi) - math.pi)
            if net_disp < 0.25 and net_yaw < math.radians(30.0):
                print(f"[v3-wrapper] hopeless-stall: {net_disp:.2f} m / "
                      f"{math.degrees(net_yaw):.0f} deg net over 500 commanded steps "
                      f"(10 s) — ending episode")
                return True
        return False

    def set_stop_called(self, is_stop_called: bool) -> None:
        self.env.is_stop_called = is_stop_called
        self.is_stop_called = is_stop_called

    def close(self) -> None:
        self.env.close()
