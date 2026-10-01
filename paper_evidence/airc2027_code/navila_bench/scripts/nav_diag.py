"""
Per-episode navigation diagnostics for the K1 VLN-CE-Isaac eval (Step 1).

Captures, per episode, exactly what we were flying blind on:
  (a) top-down trajectory: robot path vs R2R ground-truth reference path
  (b) the exact 8 RGB frames sent to the VLM at each decision step
  (c) the VLM raw text, the parsed action, and the commanded velocity per step
  (d) achieved displacement / heading change per command vs what was requested

Purely additive: enabled via --diag, writes under <result_dir>/diag/ep<idx>/.
Nothing here changes scored behavior, so paper parity is preserved.
"""

import os
import csv
import json
import math
import numpy as np


def yaw_from_quat(q):
    """ZYX yaw from (w, x, y, z), matching navila_eval_v3.quat2eulers."""
    q0, q1, q2, q3 = q
    return math.atan2(2 * (q1 * q2 + q0 * q3),
                      q0**2 + q1**2 - q2**2 - q3**2)


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def classify_vlm_text(text):
    """Mirror get_vel_command's branch order to label what the env actually did.

    Returns one of: turn_left, turn_right, forward, stop, fallback_forward.
    'fallback_forward' = the else-branch (no recognized keyword) which silently
    walks forward — this is the rate we most want to measure (RC-2).
    """
    t = (text or "").lower()
    if "turn left" in t:
        return "turn_left"
    if "turn right" in t:
        return "turn_right"
    if "move forward" in t or "move" in t:
        return "forward"
    if "stop" in t:
        return "stop"
    return "fallback_forward"


class EpisodeDiagnostics:
    def __init__(self, result_dir, episode, episode_idx, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
        self.idx = episode_idx
        self.ep_dir = os.path.join(result_dir, "diag", f"ep{episode_idx}")
        self.frames_dir = os.path.join(self.ep_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.result_dir = result_dir

        ref = np.array(episode["reference_path"], dtype=float)  # (N,3)
        self.ref_xy = ref[:, :2]
        self.goal_xy = ref[-1, :2]
        self.start_xy = ref[0, :2]
        self.instruction = episode.get("instruction", {}).get("instruction_text", "")

        self.traj = []        # (num_steps, x, y, yaw, dist_to_goal)
        self.ticks = []       # per-VLM-decision records
        self.segments = []    # per-command achieved-vs-requested
        self._pending = None  # the command currently executing
        self._tick_k = 0

    # ---- per-sim-step ----
    def log_step(self, num_steps, robot_pos, robot_yaw, dist_to_goal):
        if not self.enabled:
            return
        self.traj.append((int(num_steps), float(robot_pos[0]), float(robot_pos[1]),
                          float(robot_yaw), float(dist_to_goal)))

    # ---- per-VLM-decision ----
    def on_vlm_tick(self, num_steps, frames_sent, raw_output, parsed_cmd,
                    time_to_go, robot_pos, robot_yaw, dist_to_goal):
        if not self.enabled:
            return
        # close out the command that was executing up to now
        self._close_segment(robot_pos, robot_yaw)

        # save the exact frames the VLM saw
        tick_frame_dir = os.path.join(self.frames_dir, f"tick{self._tick_k:03d}")
        os.makedirs(tick_frame_dir, exist_ok=True)
        try:
            for i, im in enumerate(frames_sent or []):
                im.save(os.path.join(tick_frame_dir, f"frame_{i}.jpg"), format="JPEG")
        except Exception as e:  # never let diagnostics break the run
            print(f"[diag] frame save failed: {e}")

        action_type = classify_vlm_text(raw_output)
        rec = {
            "tick": self._tick_k,
            "num_steps": int(num_steps),
            "vlm_raw": raw_output,
            "parsed_cmd": [float(x) for x in parsed_cmd],
            "time_to_go": float(time_to_go),
            "action_type": action_type,
            "robot_xy": [float(robot_pos[0]), float(robot_pos[1])],
            "robot_yaw": float(robot_yaw),
            "dist_to_goal": float(dist_to_goal),
            "frames_dir": os.path.relpath(tick_frame_dir, self.ep_dir),
        }
        self.ticks.append(rec)

        # open the new segment (target derived from the fixed-speed scheme)
        cmd_type = ("forward" if parsed_cmd[0] > 0 else
                    "turn" if abs(parsed_cmd[2]) > 0 else "stop")
        self._pending = {
            "tick": self._tick_k,
            "type": cmd_type,
            "start_xy": [float(robot_pos[0]), float(robot_pos[1])],
            "start_yaw": float(robot_yaw),
            "target_disp_m": 0.5 * float(time_to_go) if cmd_type == "forward" else 0.0,
            "target_heading_rad": float(parsed_cmd[2]) * float(time_to_go) if cmd_type == "turn" else 0.0,
        }
        self._tick_k += 1

    def _close_segment(self, robot_pos, robot_yaw):
        p = self._pending
        if p is None:
            return
        dx = float(robot_pos[0]) - p["start_xy"][0]
        dy = float(robot_pos[1]) - p["start_xy"][1]
        h = p["start_yaw"]
        achieved_forward = dx * math.cos(h) + dy * math.sin(h)  # along start heading
        achieved_total = math.hypot(dx, dy)
        achieved_heading = wrap_pi(float(robot_yaw) - p["start_yaw"])
        self.segments.append({
            **p,
            "achieved_forward_m": achieved_forward,
            "achieved_total_disp_m": achieved_total,
            "achieved_heading_rad": achieved_heading,
            "lateral_drift_m": math.sqrt(max(achieved_total**2 - achieved_forward**2, 0.0)),
        })
        self._pending = None

    # ---- end of episode ----
    def finalize(self, measurements, robot_pos=None, robot_yaw=None, extra=None):
        if not self.enabled:
            return
        if robot_pos is not None and robot_yaw is not None:
            self._close_segment(robot_pos, robot_yaw)

        n_stop = sum(1 for t in self.ticks if t["action_type"] == "stop")
        n_fallback = sum(1 for t in self.ticks if t["action_type"] == "fallback_forward")
        n_forward = sum(1 for t in self.ticks if t["action_type"] == "forward")
        n_turn = sum(1 for t in self.ticks if t["action_type"].startswith("turn"))
        dists = [t["dist_to_goal"] for t in self.ticks] or [float("nan")]
        traj_dists = [r[4] for r in self.traj] or [float("nan")]

        summary = {
            "episode_idx": self.idx,
            "instruction": self.instruction,
            "n_vlm_ticks": len(self.ticks),
            "n_stop": n_stop,
            "n_fallback_forward": n_fallback,
            "n_forward": n_forward,
            "n_turn": n_turn,
            "emitted_stop": n_stop > 0,
            "min_dist_to_goal": float(np.nanmin(traj_dists)),
            "final_dist_to_goal": float(traj_dists[-1]),
            "reached_within_3m": bool(np.nanmin(traj_dists) < 3.0),
            "measurements": measurements,
        }
        if extra:
            summary.update(extra)
        with open(os.path.join(self.ep_dir, "ticks.jsonl"), "w") as f:
            for t in self.ticks:
                f.write(json.dumps(t) + "\n")
        with open(os.path.join(self.ep_dir, "segments.json"), "w") as f:
            json.dump(self.segments, f, indent=2)
        with open(os.path.join(self.ep_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        np.save(os.path.join(self.ep_dir, "trajectory.npy"), np.array(self.traj))

        self._append_slice_summary(summary, measurements)
        self._plot(summary)
        self._print(summary)
        return summary

    def _append_slice_summary(self, summary, measurements):
        path = os.path.join(self.result_dir, "diag", "slice_summary.csv")
        header = ["episode_idx", "n_vlm_ticks", "n_stop", "n_fallback_forward",
                  "emitted_stop", "reached_within_3m", "min_dist_to_goal",
                  "final_dist_to_goal", "success", "oracle_success", "spl",
                  "distance_to_goal", "path_length",
                  "term_reason", "hit_step_cap", "ended_at_step",
                  "closed_loop", "clean_render"]
        row = {
            "episode_idx": summary["episode_idx"],
            "n_vlm_ticks": summary["n_vlm_ticks"],
            "n_stop": summary["n_stop"],
            "n_fallback_forward": summary["n_fallback_forward"],
            "emitted_stop": summary["emitted_stop"],
            "reached_within_3m": summary["reached_within_3m"],
            "min_dist_to_goal": round(summary["min_dist_to_goal"], 3),
            "final_dist_to_goal": round(summary["final_dist_to_goal"], 3),
            "success": (measurements or {}).get("success"),
            "oracle_success": (measurements or {}).get("oracle_success"),
            "spl": (measurements or {}).get("spl"),
            "distance_to_goal": (measurements or {}).get("distance_to_goal"),
            "path_length": (measurements or {}).get("path_length"),
            "term_reason": summary.get("term_reason"),
            "hit_step_cap": summary.get("hit_step_cap"),
            "ended_at_step": summary.get("ended_at_step"),
            "closed_loop": summary.get("closed_loop"),
            "clean_render": summary.get("clean_render"),
        }
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new:
                w.writeheader()
            w.writerow(row)

    def _plot(self, summary):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[diag] matplotlib unavailable ({e}); trajectory.npy saved for offline plotting")
            return
        traj = np.array(self.traj)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # top-down
        ax1.plot(self.ref_xy[:, 0], self.ref_xy[:, 1], "g--o", ms=3, lw=1.5, label="GT reference path")
        if len(traj):
            ax1.plot(traj[:, 1], traj[:, 2], "b-", lw=1.5, label="robot trajectory")
        ax1.plot(*self.start_xy, "gs", ms=12, label="start")
        ax1.plot(*self.goal_xy, "r*", ms=18, label="goal")
        ax1.add_patch(plt.Circle(self.goal_xy, 3.0, color="r", fill=False, ls=":", label="3 m success"))
        for t in self.ticks:
            c = {"stop": "k", "fallback_forward": "m"}.get(t["action_type"], "c")
            ax1.plot(t["robot_xy"][0], t["robot_xy"][1], "o", color=c, ms=4)
        ax1.set_aspect("equal"); ax1.legend(fontsize=8); ax1.set_title(
            f"ep{self.idx}  ticks={summary['n_vlm_ticks']} stop={summary['n_stop']} "
            f"fallback={summary['n_fallback_forward']}\nSR={summary['measurements'].get('success')} "
            f"NE={summary['measurements'].get('distance_to_goal'):.2f} minD={summary['min_dist_to_goal']:.2f}")
        # distance over time
        if len(traj):
            ax2.plot(traj[:, 0], traj[:, 4], "b-")
        ax2.axhline(3.0, color="r", ls=":", label="3 m")
        for t in self.ticks:
            c = {"stop": "k", "fallback_forward": "m"}.get(t["action_type"], "c")
            ax2.axvline(t["num_steps"], color=c, alpha=0.3)
        ax2.set_xlabel("sim step"); ax2.set_ylabel("distance_to_goal (m)")
        ax2.legend(fontsize=8); ax2.set_title("distance to goal over time")
        fig.tight_layout()
        fig.savefig(os.path.join(self.ep_dir, "trajectory.png"), dpi=110)
        plt.close(fig)

    def _print(self, summary):
        print(f"[diag] ep{self.idx}: ticks={summary['n_vlm_ticks']} "
              f"stop={summary['n_stop']} fallback={summary['n_fallback_forward']} "
              f"reached_3m={summary['reached_within_3m']} "
              f"minD={summary['min_dist_to_goal']:.2f} finalD={summary['final_dist_to_goal']:.2f} "
              f"SR={summary['measurements'].get('success')}")
