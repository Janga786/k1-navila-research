"""
NaVILA-Bench evaluation for the K1 VLN-CE v3 policy.

This is a sibling of `navila_eval.py` that:
  * loads the v3 ActorCritic state-dict directly into a fresh
    nn.Sequential (skips rsl_rl OnPolicyRunner so we don't need the
    v3 agent.yaml to live inside NaVILA-Bench)
  * uses `VLNEnvWrapperV3` which builds the v3 47-dim per-step obs +
    5-frame term-major history flatten from raw scene state
  * everything else (VLM image loop, R2R episode handling, measurements,
    video writer) mirrors navila_eval.py.

Usage:
  python scripts/navila_eval_v3.py
      --task=k1_matterport_vision
      --num_envs=1
      --headless --enable_cameras
      --checkpoint=<absolute path to v3 model_*.pt>
      [--episode_idx=N]
      [--gait_phase_init=0.0]
"""
# [LOG] FRAME-REPLAY LOGGING COPY (airc2027_replay, 2026-10-01) of NaVILA-Bench-main/scripts/
# navila_eval_v3.py (SHA-256 9b010715...281fc). Every change is marked "[LOG]" and listed in
# evaluator.diff. Added, behind new flags: --log_dir (mandatory: all outputs go there, never
# eval_results/), per-frame PNG + frames.jsonl, per-query queries.jsonl, per-step state hashes,
# --replay_frames_from (run R: the model's history gets run A's saved frames) and --replay_queries
# (resend saved requests with the simulator loaded). Outside replay mode nothing changes the
# simulation, the random state, the control flow or the bytes sent to the model.

import argparse
import gymnasium as gym
import os
import json
import math
import signal
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import imageio
from PIL import Image
import base64
import io
import socket
import hashlib    # [LOG]
import sys        # [LOG]
import time       # [LOG]
import traceback  # [LOG]

# [LOG] This copy lives outside NaVILA-Bench/scripts. Put that directory first on sys.path so the
# sweep's own cli_args / nav_diag are imported (read-only), as `python scripts/navila_eval_v3.py`
# resolves them (there sys.path[0] is the script's directory).
_NB_SCRIPTS = os.path.realpath(os.path.expanduser("~/Projects/k1_research/NaVILA-Bench/scripts"))
sys.path.insert(0, _NB_SCRIPTS)

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import nav_diag  # isort: skip  (Step-1 per-episode diagnostics)

parser = argparse.ArgumentParser(description="Eval K1 VLN-CE v3 policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="k1_matterport_vision")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episode_idx", type=int, default=0)
parser.add_argument("--gait_phase_init", type=float, default=0.0,
                    help="Initial gait clock offset in [0, 1).")
parser.add_argument("--vlm_host", type=str, default="localhost")
parser.add_argument("--vlm_port", type=int, default=54321)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--out_tag", type=str, default="v3",
                    help="Suffix for eval_results/k1_matterport_vision_loco_<tag>/")
parser.add_argument("--diag", action="store_true", default=False,
                    help="Write per-episode navigation diagnostics (trajectory, "
                         "VLM frames/text, per-command achieved-vs-requested) under "
                         "eval_results/.../diag/. Purely additive; does not change scoring.")
parser.add_argument("--clean_render", action="store_true", default=False,
                    help="Median-5 denoise the VLM-bound frames to remove the salt-and-pepper "
                         "render speckle (RC-7) that is off-distribution vs NaVILA's clean "
                         "Habitat training. Scoped to this process.")
parser.add_argument("--bright", action="store_true", default=False,
                    help="Brighten dim scenes toward NaVILA's well-lit training exposure "
                         "(CLAHE on L + gamma 0.7), applied BEFORE the median denoise. Adaptive, "
                         "so already-bright scenes aren't blown out. Composes with --clean_render.")
parser.add_argument("--vlm_transform", choices=["stretch", "crop", "pad"], default="stretch",
                    help="Squarify the 16:9 BoosterMipi frame for NaVILA before its resize-to-384. "
                         "'stretch'=none, the op NaVILA was trained with; keeps full ~90deg HFOV "
                         "(the path) but distorts geometry [default]. 'crop'=center-crop "
                         "(undistorted, ~58deg, loses periphery). 'pad'=letterbox (undistorted, "
                         "full HFOV, but black bars = likely OOD). A/B these; use the SAME mode on "
                         "the real robot (NAVILA_VLM_TRANSFORM) to keep the benchmark predictive.")
parser.add_argument("--closed_loop", action="store_true", default=False,
                    help="Execute each VLM command in CLOSED LOOP: hold the velocity until "
                         "the robot has actually achieved the requested displacement/heading "
                         "(then briefly settle), instead of an open-loop fixed duration. "
                         "Robust to locomotion under-tracking and the acceleration ramp.")
parser.add_argument("--max_episode_s", type=float, default=50.0,
                    help="Per-episode wall budget in sim-seconds (default 50 = the original "
                         "2500-step cap). Loosen for closed-loop so the settle/full-distance "
                         "overhead isn't penalized as a controller regression.")
parser.add_argument("--max_chunk_m", type=float, default=0.0,
                    help="Closed-loop only: cap each forward command's executed displacement "
                         "to this many meters before re-querying the VLM (0=disabled=full "
                         "distance). Bounds overshoot on short final approaches and gives the "
                         "VLM more frequent chances to emit 'stop'. Legitimate (no ground truth).")
parser.add_argument("--stop_assist", action="store_true", default=False,
                    help="Legitimate near-goal stop: emit 'stop' when the robot has netted "
                         "<0.6 m over the last 8 VLM decisions (rotate-without-progress). Uses "
                         "only the robot's own pose, no goal info.")
parser.add_argument("--proximity_stop", type=float, default=0.0,
                    help="DIAGNOSTIC ONLY (uses ground-truth goal distance, NOT benchmark-fair): "
                         "auto-stop within this radius (m). Use to measure the OS->SR ceiling "
                         "(success if the robot always stopped on arrival). 0=disabled.")
# Camera-offset overrides (2026-07-23) for the height sweep. Offset is in the Trunk frame;
# absolute ground height = Trunk-standing (~0.53 m) + cam_z. None => keep the cfg default
# (0.25 -> ~0.78 m). Aperture left as an explicit override so FOV stays FIXED across a height
# sweep by default. Ported from l0_camera_dump.py:23-101. Scoring/termination/robot cfg untouched.
parser.add_argument("--cam_z", type=float, default=None, help="camera z-OFFSET above Trunk (m)")
parser.add_argument("--cam_width", type=int, default=None, help="camera width override")
parser.add_argument("--cam_height", type=int, default=None, help="camera height override")
parser.add_argument("--cam_aperture", type=float, default=None, help="horizontal_aperture (FOV) override")
# [LOG] frame-replay experiment flags (see the header note and evaluator.diff)
parser.add_argument("--log_dir", type=str, default=None,
                    help="[LOG] MANDATORY. All outputs (record JSON, video, frame/query/state logs) go here.")
parser.add_argument("--replay_frames_from", type=str, default=None,
                    help="[LOG] Run R: store run A's saved frames DIR/frames/f{k:04d}.png in the model's history "
                         "instead of the fresh renders (still rendered and saved as fresh_frames/); compare with A's "
                         "logs and end the episode at the first mismatch.")
parser.add_argument("--replay_queries", type=str, default=None,
                    help="[LOG] JSON file of saved requests: build the env, reset the episode, send each request "
                         "--replay_n times through the evaluator's socket code, log the replies, exit.")
parser.add_argument("--replay_n", type=int, default=20, help="[LOG] sends per request for --replay_queries")
cli_args.add_rsl_rl_args(parser)   # adds --checkpoint among others
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    raise SystemExit("--checkpoint is required (absolute path to v3 model_*.pt)")
# [LOG] The logging copy writes only under --log_dir, never into eval_results/, and never over a run.
if not args_cli.log_dir:
    raise SystemExit("[LOG] --log_dir is required in the logging copy")
args_cli.log_dir = os.path.realpath(args_cli.log_dir)
if "eval_results" in args_cli.log_dir.split(os.sep):
    raise SystemExit(f"[LOG] refusing to write under eval_results/: {args_cli.log_dir}")
if args_cli.replay_frames_from and args_cli.replay_queries:
    raise SystemExit("[LOG] --replay_frames_from and --replay_queries are mutually exclusive")
if args_cli.replay_frames_from:
    args_cli.replay_frames_from = os.path.realpath(args_cli.replay_frames_from)
    for _f in ("frames", "frames.jsonl", "queries.jsonl", "state_hashes.txt"):
        if not os.path.exists(os.path.join(args_cli.replay_frames_from, _f)):
            raise SystemExit(f"[LOG] --replay_frames_from lacks {_f}: {args_cli.replay_frames_from}")
    if args_cli.replay_frames_from == args_cli.log_dir:
        raise SystemExit("[LOG] --log_dir must differ from --replay_frames_from")
for _f in ("frames.jsonl", "queries.jsonl", "replay_queries.jsonl"):
    if os.path.exists(os.path.join(args_cli.log_dir, _f)):
        raise SystemExit(f"[LOG] --log_dir already holds a run ({_f}): {args_cli.log_dir}")
os.makedirs(args_cli.log_dir, exist_ok=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Render fix (RC-7): the egocentric RGB carries heavy salt-and-pepper speckle
# (off-distribution vs NaVILA's clean Habitat training). The Isaac render-side
# carb tweak (disabling sampledLighting) did NOT remove it and darkened the
# scene, so --clean_render instead applies a deterministic median-5 denoise to
# the VLM-bound frames (see sample_images_and_send_to_vlm). ~40x speckle drop,
# structure preserved, ~1 ms/frame, source-independent.
if args_cli.clean_render or args_cli.bright:
    print(f"[render] VLM-frame fix: bright={args_cli.bright} (CLAHE+gamma) "
          f"denoise={args_cli.clean_render} (median-5)")

# imports that need the SimulationApp running
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
import omni.isaac.lab.sim as sim_utils

from omni.isaac.vlnce.config import *
from omni.isaac.vlnce.utils import ASSETS_DIR, VLNEnvWrapperV3
from omni.isaac.vlnce.utils.eval_utils import (
    get_vel_command, read_episodes,
    add_instruction_on_img, InstructionData,
)


# --------------------------------------------------------------------------- #
# v3 actor builder — must match the training arch exactly (235 → 512 → 256
# → 128 → 12, ELU). The checkpoint has only `actor.*` and `critic.*` weights
# (no normalizer), so we just lift the actor.* keys.
# --------------------------------------------------------------------------- #
def build_v3_actor(ckpt_path: str, device: str) -> nn.Sequential:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    actor = nn.Sequential(
        nn.Linear(235, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 12),
    )
    own = actor.state_dict()
    mapped = {}
    for k, v in sd.items():
        if not k.startswith("actor."):
            continue
        local_k = k[len("actor."):]
        if local_k in own:
            mapped[local_k] = v
    missing = [k for k in own if k not in mapped]
    if missing:
        raise RuntimeError(f"missing actor keys: {missing}")
    actor.load_state_dict(mapped, strict=True)
    actor.to(device).eval()
    iter_ = ck.get("iter", "?")
    print(f"[v3] loaded actor from {ckpt_path}  iter={iter_}  "
          f"params={sum(p.numel() for p in actor.parameters())}")
    return actor


def quat2eulers(q0, q1, q2, q3):
    roll = math.atan2(2 * (q2 * q3 + q0 * q1),
                       q0**2 - q1**2 - q2**2 + q3**2)
    pitch = math.asin(2 * (q1 * q3 - q0 * q2))
    yaw = math.atan2(2 * (q1 * q2 + q0 * q3),
                      q0**2 + q1**2 - q2**2 - q3**2)
    return roll, pitch, yaw


def reset_start_pos_rot(env_cfg, args_cli, episode):
    scene_id = os.path.splitext(os.path.basename(episode["scene_id"]))[0]
    env_cfg.scene.terrain.obj_filepath = os.path.join(
        ASSETS_DIR, f"matterport_usd/{scene_id}/{scene_id}.usd"
    )
    start_pos = episode["start_position"]
    start_rot = episode["start_rotation"]
    goal_pos = episode["reference_path"][-1]
    env_cfg.scene.robot.init_state.rot = start_rot
    env_cfg.scene.robot.init_state.pos = (
        start_pos[0], start_pos[1], start_pos[2] + 0.55
    )
    env_cfg.scene.terrain.origins = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.disk_1.init_state.pos = (
        [start_pos[0], start_pos[1], start_pos[2] + 2.5]
    )
    env_cfg.scene.disk_2.init_state.pos = (
        [goal_pos[0], goal_pos[1], goal_pos[2] + 2.5]
    )
    return env_cfg


def brighten_frame(arr_rgb):
    """CLAHE on the L channel + mild gamma — lifts dim Isaac scenes toward
    NaVILA's well-lit training exposure. Adaptive (per-tile), so already-bright
    scenes aren't blown out. Input/output: uint8 RGB."""
    import cv2
    lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return (((rgb / 255.0) ** 0.7) * 255).astype(np.uint8)


def square_frame(pil_img, mode):
    """Make a frame square so NaVILA's process_images resize-to-384 is geometry-
    preserving. The real K1 BoosterMipi feed is 16:9; pick how to squarify it:
      'stretch' -> no-op (the VLM's resize distorts 16:9->1:1, = legacy behavior)
      'crop'    -> center-crop to square (undistorted, narrower ~58deg FOV)
      'pad'     -> letterbox to square (undistorted, full ~90deg HFOV, black bars)
    Apply the SAME mode on the real robot before inference to keep sim predictive."""
    if mode == "stretch":
        return pil_img
    W, H = pil_img.size
    if W == H:
        return pil_img
    if mode == "crop":
        s = min(W, H)
        l, t = (W - s) // 2, (H - s) // 2
        return pil_img.crop((l, t, l + s, t + s))
    # 'pad' (default): letterbox onto a black square
    s = max(W, H)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(pil_img, ((s - W) // 2, (s - H) // 2))
    return canvas


def _vlm_send_bytes(data_bytes, vlm_host, vlm_port, log=None):
    """[LOG] The evaluator's socket exchange, moved verbatim out of sample_images_and_send_to_vlm
    so that --replay_queries sends saved requests through the very same code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))
        s.settimeout(120.0)  # bound every recv: a VLM crash/OOM must not hang the eval forever (rank 4)
        s.sendall(len(data_bytes).to_bytes(8, 'big'))
        s.sendall(data_bytes)
        size_data = b''                       # recv(8) may return <8 bytes on a partial read (rank 12)
        while len(size_data) < 8:
            chunk = s.recv(8 - len(size_data))
            if not chunk:
                raise ConnectionError("VLM closed connection before sending the size header")
            size_data += chunk
        size = int.from_bytes(size_data, 'big')
        response_data = b''
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet
        if log is not None:  # [LOG]
            log["response_sha256"] = _sha256(response_data)
            log["response_len"] = len(response_data)
        return json.loads(response_data.decode())


def sample_images_and_send_to_vlm(image_list, vlm_host, vlm_port, query,
                                  denoise=False, bright=False, transform="stretch",
                                  log=None):  # [LOG] log: per-query record (None = no logging)
    _n_hist = len(image_list)  # [LOG]
    if len(image_list) == 0:
        return None
    if len(image_list) < 8:
        image_list = image_list.copy()
        for _ in range(8 - len(image_list)):
            image_list.insert(0, Image.new('RGB', image_list[-1].size, (0, 0, 0)))
    else:
        image_list = image_list.copy()
    num_images = len(image_list)
    indices = [int(i * (num_images - 1) / 7) for i in range(7)]
    sampled_images = [image_list[i] for i in indices]
    sampled_images.append(image_list[-1])
    if log is not None:  # [LOG] append indices k of the 8 frames sent (-1 = black padding frame)
        _pad = num_images - _n_hist
        log["frames_in_history"] = _n_hist
        log["indices"] = [i - _pad if i >= _pad else -1 for i in indices + [num_images - 1]]

    encoded_images = []
    pil_images = []  # the exact 8 frames sent, for diagnostics
    for image in sampled_images:
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = ((arr * 255.0) if arr.max() <= 1.0 else arr).clip(
                    0, 255
                ).astype(np.uint8)
            pil_image = Image.fromarray(arr)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(np.array(image, dtype=np.uint8))
        if bright or denoise:
            import cv2  # lazy import: avoid cv2-before-omni ordering issues
            arr = np.array(pil_image)
            # ORDER MATTERS (desktop smoke diagnosis 2026-06-11): denoise BEFORE
            # CLAHE — brightening first amplifies render speckle into high-contrast
            # blotches the median can no longer remove.
            if denoise:  # median-5 kills the salt-and-pepper render speckle (RC-7)
                arr = cv2.medianBlur(arr, 5)
            if bright:
                arr = brighten_frame(arr)
            pil_image = Image.fromarray(arr)
        pil_image = square_frame(pil_image, transform)
        pil_images.append(pil_image)
        buf = io.BytesIO()
        # PNG, not JPEG: lossless on the wire (the 6/8 A/B's winning config; JPEG
        # adds compression artifacts on top of render noise). Localhost: size is free.
        pil_image.save(buf, format="PNG")
        encoded_images.append(base64.b64encode(buf.getvalue()).decode())
        if log is not None:  # [LOG] SHA-256 of each PNG byte string sent
            log["png_sha256"].append(_sha256(buf.getvalue()))

    request_data = {'images': encoded_images, 'query': query}
    data_bytes = json.dumps(request_data).encode()  # [LOG] moved out of the socket block; same bytes
    if log is not None:  # [LOG]
        log["request_sha256"] = _sha256(data_bytes)
        log["request_len"] = len(data_bytes)
        _exp = log.get("expect_request_sha256")  # set in replay mode only
        if _exp is not None and _exp != log["request_sha256"]:
            log["request_mismatch"] = True
            raise ReplayMismatch("request")      # replay mode: end before sending a differing request
        log["t_send_unix"] = time.time()
    _resp = _vlm_send_bytes(data_bytes, vlm_host, vlm_port, log=log)  # [LOG] original socket code
    if log is not None:  # [LOG]
        log["request_wall_s"] = round(time.time() - log["t_send_unix"], 4)
    return _resp, pil_images


class ClosedLoopController:
    """Step 2: hold each VLM velocity command until the robot has ACTUALLY
    achieved the requested forward displacement / heading change (measured from
    Isaac ground-truth base pose), then briefly settle at zero velocity, then
    signal a re-query. Replaces the open-loop fixed-duration hold, which trusts
    that vx=0.5 m/s for distance/0.5 s moves `distance` — false for a humanoid
    with an acceleration ramp or steady-state tracking error. A per-command
    deadline (timeout) keeps a stuck robot from hanging the episode.
    """

    def __init__(self, dt_step, fwd_tol=0.05, yaw_tol_deg=3.0,
                 timeout_mult=4.0, min_timeout_steps=25, settle_steps=8,
                 max_chunk_m=0.0, stall_window=150, stall_fwd_m=0.10,
                 stall_turn_deg=15.0):
        self.dt_step = dt_step
        self.fwd_tol = fwd_tol
        self.yaw_tol = math.radians(yaw_tol_deg)
        self.timeout_mult = timeout_mult
        self.min_timeout = min_timeout_steps
        self.settle_steps = settle_steps
        self.max_chunk_m = max_chunk_m  # cap forward exec distance per re-query (0=off)
        # Stall early-requery (audit rank 5, validated 2026-06-10): if windowed progress
        # over the last `stall_window` steps (3 s) is below threshold, end the command and
        # re-query the VLM instead of burning the full 4x deadline (6.2 s fwd75 / 12.2 s
        # turn90). WINDOWED net progress, not instantaneous stillness: a wall-jammed K1's
        # base wobbles at ~1 rad/s (measured), so only integration separates jam (0.14 m,
        # 8 deg per 3 s) from legit motion (1.2 m walk, 90 deg turn per 3 s).
        self.stall_window = stall_window
        self.stall_fwd_m = stall_fwd_m
        self.stall_turn = math.radians(stall_turn_deg)
        self.active = None

    def start(self, cmd, time_to_go, pos, yaw, num_steps):
        typ = "forward" if cmd[0] > 0 else ("turn" if abs(cmd[2]) > 0 else "stop")
        nominal = int(time_to_go / self.dt_step)
        target_disp = 0.5 * time_to_go if typ == "forward" else 0.0
        if typ == "forward" and self.max_chunk_m > 0:
            target_disp = min(target_disp, self.max_chunk_m)  # bound overshoot (fix c)
        self.active = {
            "type": typ,
            "cmd": [float(c) for c in cmd],
            "start_xy": (float(pos[0]), float(pos[1])),
            "start_yaw": float(yaw),
            "target_disp": target_disp,
            "target_heading": float(cmd[2]) * time_to_go if typ == "turn" else 0.0,
            "deadline": num_steps + max(int(nominal * self.timeout_mult), self.min_timeout),
            "phase": "move",
            "settle_end": None,
            "ach_hist": [],     # per-step achieved progress, for the windowed stall check
            "stall_fired": False,
        }

    def velocity(self):
        if self.active is None or self.active["phase"] == "settle":
            return [0.0, 0.0, 0.0]
        return self.active["cmd"]

    def update(self, pos, yaw, num_steps):
        """Advance one step; return True when ready to (re)query the VLM."""
        a = self.active
        if a is None or a["type"] == "stop":
            return True
        if a["phase"] == "move":
            if a["type"] == "forward":
                dx = float(pos[0]) - a["start_xy"][0]
                dy = float(pos[1]) - a["start_xy"][1]
                achieved = dx * math.cos(a["start_yaw"]) + dy * math.sin(a["start_yaw"])
                reached = achieved >= a["target_disp"] - self.fwd_tol
            else:  # turn
                achieved = nav_diag.wrap_pi(float(yaw) - a["start_yaw"])
                reached = abs(achieved) >= abs(a["target_heading"]) - self.yaw_tol
            # windowed stall: no meaningful progress over the last stall_window steps
            # -> give up on this command early and let the VLM re-plan
            a["ach_hist"].append(achieved)
            stalled = False
            if self.stall_window > 0 and len(a["ach_hist"]) > self.stall_window:
                prog = abs(achieved - a["ach_hist"][-self.stall_window - 1])
                thr = self.stall_fwd_m if a["type"] == "forward" else self.stall_turn
                if prog < thr:
                    stalled = True
                    a["stall_fired"] = True
                    print(f"[closed-loop] STALL: {a['type']} progress "
                          f"{prog:.3f} over last {self.stall_window} steps (3 s) "
                          f"— aborting command, re-querying VLM.", flush=True)
            if reached or stalled or num_steps >= a["deadline"]:
                if self.settle_steps > 0:
                    a["phase"] = "settle"
                    a["settle_end"] = num_steps + self.settle_steps
                    return False
                return True
            return False
        return num_steps >= a["settle_end"]  # settle phase


def _match_heights(a, b):
    """Resize b to a's height (keep aspect) before hstack — the nav camera is
    1280x720 (BoosterMipi patch) while viz_rgb_camera is still 512x512."""
    if a.shape[0] == b.shape[0]:
        return b
    import cv2
    h = a.shape[0]
    w = max(1, int(round(b.shape[1] * h / b.shape[0])))
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)


# =================================================================================== #
# [LOG] Frame-replay logging. Observers only: they copy tensors to the host, hash them #
# and write files under --log_dir. They never write to the simulator, never draw      #
# random numbers and never change the bytes sent to the model. Replay mode            #
# (--replay_frames_from) is the one documented exception: it changes the source of    #
# the frames stored in the model's history and ends the episode at the first mismatch.#
# =================================================================================== #
STATE_PARTS = ("root_state_w", "joint_pos", "joint_vel", "body_state_w",
               "joint_pos_target", "policy_obs_history")


class ReplayMismatch(Exception):
    """[LOG] Replay mode only: run R differs from run A."""


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def _state_arrays(env):
    """[LOG] Host copies of the state hashed after every env.step, in hash order. The policy
    observation is the wrapper's 5 x 47 history buffer (the 235-dim input is its flatten)."""
    d = env.unwrapped.scene["robot"].data
    return [
        ("root_state_w", d.root_state_w.detach().cpu().numpy()),
        ("joint_pos", d.joint_pos.detach().cpu().numpy()),
        ("joint_vel", d.joint_vel.detach().cpu().numpy()),
        ("body_state_w", d.body_state_w.detach().cpu().numpy()),
        ("joint_pos_target", d.joint_pos_target.detach().cpu().numpy()),
        ("policy_obs_history", env._history_buf.detach().cpu().numpy()),
    ]


def _context_arrays(env):
    """[LOG] Extra wrapper state saved at queries (not part of the state hash)."""
    return [
        ("last_action", env._last_action.detach().cpu().numpy()),
        ("gait_phase", env._gait_phase.detach().cpu().numpy()),
        ("command", env._command.detach().cpu().numpy()),
    ]


class ReplayLog:
    """[LOG] frames/ + frames.jsonl, queries.jsonl, state_hashes.txt (+ per-part hashes in
    state_components.tsv) and states_at_queries.npz for one run. With replay_from (run R):
    serves run A's frames to the model's history and compares with A's logs."""

    VERIFY_FIRST = 10  # frames whose PNG is reloaded and compared with the array in-process

    def __init__(self, log_dir, replay_from=None, meta=None):
        self.dir = log_dir
        self.replay_from = replay_from
        self.k = 0               # append index into image_observations
        self.qi = 0              # query index
        self.mismatch = None     # replay mode: the first difference from run A
        self.last_state = None   # (label, hash) of the latest logged state
        self.layout = None
        self.q_arrays = {}
        self._closed = False
        self.frames_sub = "frames" if replay_from is None else "fresh_frames"
        os.makedirs(os.path.join(log_dir, self.frames_sub), exist_ok=True)
        self.f_frames = open(os.path.join(log_dir, "frames.jsonl"), "w", buffering=1)
        self.f_queries = open(os.path.join(log_dir, "queries.jsonl"), "w", buffering=1)
        self.f_states = open(os.path.join(log_dir, "state_hashes.txt"), "w", buffering=1)
        self.f_comp = open(os.path.join(log_dir, "state_components.tsv"), "w", buffering=1)
        self.f_comp.write("label\t" + "\t".join(STATE_PARTS) + "\n")
        self.ref_frames, self.ref_queries, self.ref_states = {}, {}, {}
        if replay_from is not None:
            with open(os.path.join(replay_from, "frames.jsonl")) as fh:
                for line in fh:
                    r = json.loads(line)
                    self.ref_frames[r["k"]] = r
            with open(os.path.join(replay_from, "queries.jsonl")) as fh:
                for line in fh:
                    r = json.loads(line)
                    self.ref_queries[r["query_index"]] = r
            with open(os.path.join(replay_from, "state_hashes.txt")) as fh:
                for line in fh:
                    lab, hx = line.split()
                    self.ref_states[lab] = hx
        if meta is not None:
            with open(os.path.join(log_dir, "run_meta.json"), "w") as fh:
                json.dump(meta, fh, indent=2, default=str)

    def _set_mismatch(self, what, where, run_a, run_r):
        if self.mismatch is not None:
            return
        self.mismatch = {"what": what, "where": where, "run_A": run_a, "run_R": run_r,
                         "replay_frames_from": self.replay_from, "frames_appended": self.k,
                         "queries_started": self.qi,
                         "last_state_label": self.last_state[0] if self.last_state else None}
        with open(os.path.join(self.dir, "replay_mismatch.json"), "w") as fh:
            json.dump(self.mismatch, fh, indent=2, default=str)
        print(f"[LOG] REPLAY MISMATCH ({what}) at {where}: A={run_a!r} R={run_r!r}", flush=True)

    def hist_frame(self, arr, label):
        """Called where the evaluator appends a frame to image_observations; returns the image
        to append. Normal mode: Image.fromarray(arr), as before, and the frame is saved as a
        lossless PNG. Replay mode: run A's frame k wrapped the same way (Image.fromarray of the
        uint8 array); the fresh render is saved under fresh_frames/."""
        k = self.k
        self.k += 1
        fresh = np.ascontiguousarray(arr)
        rec = {"k": k, "label": label, "shape": list(fresh.shape), "dtype": str(fresh.dtype),
               "fresh_sha256": _sha256(fresh.tobytes())}
        path = os.path.join(self.dir, self.frames_sub, f"f{k:04d}.png")
        Image.fromarray(fresh).save(path, format="PNG")
        if self.replay_from is None:
            img = Image.fromarray(arr)
            rec["sha256"] = rec["fresh_sha256"]
            if k < self.VERIFY_FIRST:
                back = _read_png_rgb(path)
                ok = (back.dtype == fresh.dtype and back.shape == fresh.shape
                      and np.array_equal(back, fresh) and np.array_equal(np.asarray(img), fresh))
                rec["png_reload_verified"] = bool(ok)
                if not ok:
                    self.f_frames.write(json.dumps(rec) + "\n")
                    raise RuntimeError(f"[LOG] PNG reload check failed for frame {k}")
        else:
            src = os.path.join(self.replay_from, "frames", f"f{k:04d}.png")
            ref = self.ref_frames.get(k)
            if ref is None or not os.path.exists(src):
                self._set_mismatch("frame_missing_in_A", {"k": k, "label": label}, None, rec["fresh_sha256"])
                img = Image.fromarray(arr)   # the episode ends at the next hook
                rec["sha256"] = rec["fresh_sha256"]
            else:
                rarr = _read_png_rgb(src)
                img = Image.fromarray(rarr)
                rec["sha256"] = _sha256(np.ascontiguousarray(rarr).tobytes())
                rec["A_sha256"] = ref["sha256"]
                if (rec["sha256"] != ref["sha256"] or rarr.shape != fresh.shape
                        or rarr.dtype != fresh.dtype or ref.get("label") != label):
                    self._set_mismatch("replay_frame", {"k": k, "label": label, "A_label": ref.get("label")},
                                       ref["sha256"], rec["sha256"])
        self.f_frames.write(json.dumps(rec) + "\n")
        return img

    def log_state(self, env, label):
        """After env.reset and after every env.step: `label hash` in state_hashes.txt (SHA-256
        over the bytes of STATE_PARTS, in order). Returns False in replay mode once run R
        differs from run A (the caller ends the episode)."""
        parts = _state_arrays(env)
        if self.layout is None:
            self.layout = [(n, list(a.shape), str(a.dtype)) for n, a in parts]
            with open(os.path.join(self.dir, "state_layout.json"), "w") as fh:
                json.dump(self.layout, fh, indent=2)
        h = hashlib.sha256()
        comps = []
        for _, a in parts:
            b = np.ascontiguousarray(a).tobytes()
            h.update(b)
            comps.append(_sha256(b))
        hx = h.hexdigest()
        self.f_states.write(f"{label} {hx}\n")
        self.f_comp.write(f"{label}\t" + "\t".join(comps) + "\n")
        self.last_state = (str(label), hx)
        if self.replay_from is not None and self.mismatch is None:
            ref = self.ref_states.get(str(label))
            if ref != hx:
                self._set_mismatch("state", {"label": str(label)}, ref, hx)
        return self.mismatch is None

    def query_begin(self, env, num_steps, infos):
        """At each model query, before the request is built: state hash at the query, raw state
        arrays (states_at_queries.npz), distance_to_goal. Returns the per-query record that
        sample_images_and_send_to_vlm fills in (indices, PNG hashes, request hash, wall time)."""
        qi = self.qi
        self.qi += 1
        h = hashlib.sha256()
        for name, a in _state_arrays(env):
            h.update(np.ascontiguousarray(a).tobytes())
            self.q_arrays[f"q{qi:04d}_{name}"] = a
        for name, a in _context_arrays(env):
            self.q_arrays[f"q{qi:04d}_{name}"] = a
        q = {"query_index": qi, "num_steps": int(num_steps),
             "state_hash": h.hexdigest(),
             "state_label_before": self.last_state[0] if self.last_state else None,
             "distance_to_goal": float(infos.get("measurements", {}).get("distance_to_goal", float("nan"))),
             "frames_in_history": None, "indices": None, "png_sha256": [],
             "request_sha256": None, "request_len": None}
        if self.replay_from is not None:
            ref = self.ref_queries.get(qi)
            if self.mismatch is not None:
                q["abort"] = "earlier_mismatch"
            elif ref is None:
                self._set_mismatch("query_missing_in_A", {"query_index": qi, "num_steps": int(num_steps)},
                                   None, int(num_steps))
                q["abort"] = "query_missing_in_A"
            elif int(ref["num_steps"]) != int(num_steps):
                self._set_mismatch("query_step", {"query_index": qi}, int(ref["num_steps"]), int(num_steps))
                q["abort"] = "query_step"
            else:
                q["expect_request_sha256"] = ref["request_sha256"]
        return q

    def query_end(self, q, instruction, reply, cmd, time_to_go, error=None):
        """After the reply (or a failed/aborted query): one line in queries.jsonl."""
        q = dict(q)
        q["instruction"] = instruction
        q["reply"] = reply
        q["parsed_cmd"] = None if cmd is None else [float(c) for c in cmd]
        q["time_to_go"] = None if time_to_go is None else float(time_to_go)
        if error is not None:
            q["error"] = error
        if self.replay_from is not None and self.mismatch is None:
            ref = self.ref_queries.get(q["query_index"])
            where = {"query_index": q["query_index"], "num_steps": q["num_steps"]}
            if q.get("request_mismatch"):
                self._set_mismatch("request", dict(where, A_png_sha256=ref.get("png_sha256"),
                                                   R_png_sha256=q.get("png_sha256")),
                                   ref["request_sha256"], q["request_sha256"])
            elif reply is not None and reply != ref["reply"]:
                self._set_mismatch("reply", where, ref["reply"], reply)
        self.f_queries.write(json.dumps(q) + "\n")

    def close(self):
        """Write states_at_queries.npz and close the line logs (normal end, SIGTERM, errors)."""
        if self._closed:
            return
        self._closed = True
        np.savez(os.path.join(self.dir, "states_at_queries.npz"), **self.q_arrays)
        for fh in (self.f_frames, self.f_queries, self.f_states, self.f_comp):
            fh.close()


def _read_png_rgb(path):
    """[LOG] Decode a lossless PNG to its uint8 RGB array without PIL's plugin path: inside Isaac Sim 4.1
    the PIL package mixes Pillow 12.3.0 (PIL.Image, ~/.local) with Kit's prebundled 10.2.0 plugins
    (PIL.ImageFile / PIL.PngImagePlugin) and Image.open(...).load() raises there. cv2 is already imported
    by the evaluator; PNG decoding is exact in any conforming decoder."""
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise RuntimeError(f"[LOG] cannot decode {path} as an 8-bit RGB PNG")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _file_sha256(path):
    with open(path, "rb") as fh:
        return _sha256(fh.read())


def _run_meta(episode):
    """[LOG] Provenance written once per run (run_meta.json)."""
    import platform
    import zlib
    import PIL
    return {
        "argv": sys.argv, "args": vars(args_cli),
        "episode_idx": args_cli.episode_idx, "episode_id": episode["episode_id"],
        "record": int(episode["episode_id"]) - 1, "scene_id": episode["scene_id"],
        "instruction": episode["instruction"]["instruction_text"],
        "pid": os.getpid(), "host": platform.node(), "cwd": os.getcwd(),
        "start_unix": time.time(), "start_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evaluator_copy_sha256": _file_sha256(os.path.abspath(__file__)),
        "sweep_evaluator_sha256": _file_sha256(os.path.join(_NB_SCRIPTS, "navila_eval_v3.py")),
        "python": sys.version, "torch": torch.__version__,
        "numpy": [np.__version__, np.__file__], "PIL": [PIL.__version__, PIL.__file__],
        "zlib": [zlib.ZLIB_VERSION, zlib.ZLIB_RUNTIME_VERSION],
        "env_PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
    }


def _replay_queries(args_cli):
    """[LOG] --replay_queries (Step 4, condition 1): with the simulator built and the episode
    reset, send each saved request --replay_n times through _vlm_send_bytes (the evaluator's
    socket code) and log every reply to <log_dir>/replay_queries.jsonl."""
    with open(args_cli.replay_queries) as fh:
        spec = json.load(fh)
    base = os.path.dirname(os.path.abspath(args_cli.replay_queries))
    with open(os.path.join(args_cli.log_dir, "replay_queries.jsonl"), "w", buffering=1) as out:
        for req in spec["requests"]:
            path = req["path"] if os.path.isabs(req["path"]) else os.path.join(base, req["path"])
            with open(path, "rb") as fh:
                data_bytes = fh.read()
            sha = _sha256(data_bytes)
            if sha != req["sha256"]:
                raise RuntimeError(f"[LOG] request {req['label']}: sha256 {sha} != {req['sha256']}")
            for i in range(args_cli.replay_n):
                log = {}
                t0 = time.time()
                reply = _vlm_send_bytes(data_bytes, args_cli.vlm_host, args_cli.vlm_port, log=log)
                out.write(json.dumps({"label": req["label"], "i": i, "request_sha256": sha,
                                      "reply": reply, "response_sha256": log.get("response_sha256"),
                                      "wall_s": round(time.time() - t0, 4), "t_unix": t0}) + "\n")
                print(f"[LOG] replay {req['label']} {i}: {reply!r}", flush=True)


def main():
    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    episode = all_episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg = reset_start_pos_rot(env_cfg, args_cli, episode)

    # Camera-offset overrides (height sweep). Camera-only; scoring/termination untouched.
    if any(v is not None for v in (args_cli.cam_z, args_cli.cam_width,
                                   args_cli.cam_height, args_cli.cam_aperture)):
        _cam = env_cfg.scene.rgb_camera
        _old = tuple(_cam.offset.pos)
        if args_cli.cam_z is not None:       _cam.offset.pos = (_old[0], _old[1], args_cli.cam_z)
        if args_cli.cam_width is not None:   _cam.width = args_cli.cam_width
        if args_cli.cam_height is not None:  _cam.height = args_cli.cam_height
        if args_cli.cam_aperture is not None: _cam.spawn.horizontal_aperture = args_cli.cam_aperture
        print(f"[cam-override] offset {_old}->{_cam.offset.pos}  {_cam.width}x{_cam.height}  "
              f"aperture={_cam.spawn.horizontal_aperture}  "
              f"abs_ground_height~={0.53 + _cam.offset.pos[2]:.2f}m (virtual if >~0.9m)", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # Build the v3 actor directly (skip rsl_rl runner setup).
    device = env.unwrapped.device
    actor = build_v3_actor(args_cli.checkpoint, device=str(device))

    # Build a callable that turns the 235-dim policy input into 12-dim
    # actions deterministically (mean of the policy, no exploration).
    def policy_fn(obs_235: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return actor(obs_235)

    # Drop the v3-specific wrapper on top.
    env = VLNEnvWrapperV3(
        env, policy_fn, args_cli.task, episode,
        high_level_obs_key="camera_obs",
        gait_phase_init=args_cli.gait_phase_init,
    )

    # Camera setup
    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    _, _, yaw = quat2eulers(*robot_quat_w)
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw),
               robot_pos_w[1] - 0.8 * math.cos(-yaw),
               robot_pos_w[2] + 0.8)
    cam_target = tuple(robot_pos_w)
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    obs, infos = env.reset()

    if args_cli.replay_queries:  # [LOG] Step 4, condition 1: simulator built, episode reset; no episode
        _replay_queries(args_cli)
        env.close()
        return
    rlog = ReplayLog(args_cli.log_dir, args_cli.replay_frames_from, meta=_run_meta(episode))  # [LOG]
    _reset_ok = rlog.log_state(env, "reset")  # [LOG] False only in replay mode, if R's reset state differs

    result_dir = args_cli.log_dir  # [LOG] was f"eval_results/{args_cli.task}_loco_{args_cli.out_tag}"
    diag = nav_diag.EpisodeDiagnostics(
        result_dir, episode, args_cli.episode_idx, enabled=args_cli.diag
    ) if args_cli.diag else None

    steps_per_image = 0.5 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    steps_per_viz_image = 0.1 / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)

    rgb_obs = infos["observations"]["camera_obs"]
    init_frame = rgb_obs[0, :, :, :3].cpu().numpy()
    instruction = InstructionData(**episode["instruction"])
    image_observations = [rlog.hist_frame(init_frame, "init")]  # [LOG] was [Image.fromarray(init_frame)]

    add_instruction_on_img(init_frame, instruction.instruction_text)
    vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    add_instruction_on_img(vis_frame, "")
    rgb_obses = [np.concatenate([init_frame, _match_heights(init_frame, vis_frame)], axis=1)]

    num_steps = 0
    target_steps = 0
    same_pos_count = 0
    stall_buf = deque(maxlen=500)  # (x, y, yaw) per commanded step — hopeless-stall window
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    max_episode_steps = args_cli.max_episode_s / (env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation)
    term_reason = "running"
    if not _reset_ok:  # [LOG] replay mode only: R's reset state differs from run A's
        term_reason = "replay_mismatch"

    # --- Wall-clock-timeout capture (2026-07-23) -----------------------------------------
    # The runner kills us with SIGTERM at --EP_TIMEOUT wall-seconds (timeout -k 30). Without
    # this, that kill lands mid-loop BEFORE the measurement write -> NO JSON (a pilot lost
    # 29/100). Catch SIGTERM and write a recorded FAILURE row so every pinned episode index
    # produces a file. Schema guaranteed: all six production fields defaulted (distances use
    # a -1.0 sentinel; 0.0 would read as 'at the goal'); real mid-episode partials overlay.
    # Only success & spl are FORCED to 0 (both require a stop); oracle_success / ONE /
    # path_length keep their real partial values (a wall-timeout can legitimately have OS=1).
    _WALL_STUB = {"success": 0.0, "spl": 0.0, "oracle_success": 0.0,
                  "distance_to_goal": -1.0, "oracle_navigation_error": -1.0, "path_length": -1.0}
    _latest = {"infos": None, "num_steps": 0}   # updated each loop iteration
    def _on_sigterm(signum, frame):
        m = dict(_WALL_STUB)
        m.update((_latest["infos"] or {}).get("measurements", {}))   # real partials win
        m["success"] = 0.0; m["spl"] = 0.0                            # only these require a stop
        m["term_reason"] = "wall_timeout"
        m["ended_at_step"] = int(_latest["num_steps"])
        m["max_episode_steps"] = int(max_episode_steps)
        m["hit_step_cap"] = False
        _rd = f"{args_cli.log_dir}/measurements"  # [LOG] was f"eval_results/{args_cli.task}_loco_{args_cli.out_tag}/measurements"
        os.makedirs(_rd, exist_ok=True)
        with open(f"{_rd}/{int(episode['episode_id'])-1}.json", "w") as _f:
            json.dump(m, _f, indent=4)
        rlog.close()  # [LOG] flush the logs
        print("[WALL_TIMEOUT] SIGTERM -> wrote recorded-failure row, exiting", flush=True)
        os._exit(0)
    signal.signal(signal.SIGTERM, _on_sigterm)   # set late (post-Isaac-init) so it sticks

    # Pre-VLM warmup: fill the 8-frame VLM image buffer with REAL frames.
    zero_cmd = torch.zeros(3, device=obs.device)
    warmup_target_frames = 8
    warmup_total_steps = int(warmup_target_frames * steps_per_image)
    for warm_i in range(warmup_total_steps):
        if term_reason == "replay_mismatch":  # [LOG] replay mode only
            break
        obs, _, done, infos = env.step(zero_cmd)
        if not rlog.log_state(env, f"w{warm_i}"):  # [LOG] replay mode only: state differs from run A
            term_reason = "replay_mismatch"
            break
        if (warm_i + 1) % int(steps_per_image) == 0:
            curr_frame = (
                infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            )
            image_observations.append(rlog.hist_frame(curr_frame, f"w{warm_i}"))  # [LOG] was Image.fromarray(curr_frame)
    prev_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()

    vlm_vel_commands = [0.0, 0.0, 0.0]
    env_steps_to_go = 0
    stream_output = ""
    sent_frames = None
    dt_step = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    controller = (ClosedLoopController(dt_step, max_chunk_m=args_cli.max_chunk_m)
                  if args_cli.closed_loop else None)
    recent_tick_xy = []  # for --stop_assist circling detection

    while term_reason != "replay_mismatch" and simulation_app.is_running():  # [LOG] 1st clause: replay mode only
        cur_p = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        cur_y = nav_diag.yaw_from_quat(
            env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy())

        # Diagnostic proximity stop (uses ground-truth goal distance — NOT a fair
        # benchmark number; only to measure the OS->SR ceiling). Set before env.step
        # so update_measures() scores Success at this pose.
        if args_cli.proximity_stop > 0 and not env.is_stop_called:
            _pd = float(infos.get("measurements", {}).get("distance_to_goal", 1e9))
            if _pd < args_cli.proximity_stop:
                env.set_stop_called(True)
                term_reason = "proximity_stop"

        # --- decide whether to (re)query the VLM ---
        if controller is not None:
            requery = controller.update(cur_p, cur_y, num_steps)
        else:
            requery = (num_steps == target_steps)
        if env.is_stop_called:  # proximity-stop already fired -> no point querying
            requery = False

        # [LOG] state at the query + per-query record; read here, outside inference_mode, so a
        # lazy Isaac Lab buffer can never be refilled with an inference tensor
        _q = rlog.query_begin(env, num_steps, infos) if requery else None
        with torch.inference_mode():
            if requery:
                try:
                    if _q.get("abort"):  # [LOG] replay mode only
                        raise ReplayMismatch(_q["abort"])
                    stream_output, sent_frames = sample_images_and_send_to_vlm(
                        image_observations, args_cli.vlm_host, args_cli.vlm_port,
                        instruction.instruction_text,
                        denoise=args_cli.clean_render, bright=args_cli.bright,
                        transform=args_cli.vlm_transform,
                        log=_q,  # [LOG]
                    )
                except ReplayMismatch:  # [LOG] replay mode only: end the episode with a partial record
                    rlog.query_end(_q, instruction.instruction_text, None, None, None)
                    term_reason = "replay_mismatch"
                    break
                except (socket.timeout, OSError, ValueError) as e:
                    # VLM hung/crashed/OOM or sent malformed JSON: end the episode as a
                    # FAILURE (no set_stop_called -> not scored as success) and write the
                    # partial measurement JSON, instead of blocking the whole run (rank 4/12).
                    print(f"[VLM_ERROR] {type(e).__name__}: {e} -- ending episode", flush=True)
                    rlog.query_end(_q, instruction.instruction_text, None, None, None,
                                   error=f"{type(e).__name__}: {e}")  # [LOG]
                    term_reason = "vlm_error"
                    break
                vlm_vel_commands, time_to_go = get_vel_command(stream_output)
                env_steps_to_go = int(time_to_go / dt_step)
                target_steps = num_steps + env_steps_to_go
                print(f"VLM: {stream_output}\nCmd: {vlm_vel_commands}  "
                      f"steps_to_go: {env_steps_to_go}")
                rlog.query_end(_q, instruction.instruction_text, stream_output,
                               vlm_vel_commands, time_to_go)  # [LOG]
                if rlog.mismatch is not None:  # [LOG] replay mode only: reply differs from run A
                    term_reason = "replay_mismatch"
                    break

                # Stop-ordering fix: register 'stop' BEFORE the next env.step so
                # update_measures() scores Success at the stop pose (matching the
                # upstream demo_planner.py), not one drifted step later.
                if time_to_go == 0.0:
                    env.set_stop_called(True)

                # Legitimate near-goal stop: no net progress over last 8 decisions.
                if args_cli.stop_assist and not env.is_stop_called:
                    recent_tick_xy.append((float(cur_p[0]), float(cur_p[1])))
                    recent_tick_xy = recent_tick_xy[-8:]
                    if len(recent_tick_xy) >= 8:
                        last = recent_tick_xy[-8:]
                        span = max(math.hypot(a[0] - b[0], a[1] - b[1])
                                   for a in last for b in last)
                        if span < 0.6:
                            env.set_stop_called(True)
                            term_reason = "stop_assist"

                if controller is not None:
                    controller.start(vlm_vel_commands, time_to_go, cur_p, cur_y, num_steps)

                if diag is not None:
                    _d = float(infos.get("measurements", {}).get(
                        "distance_to_goal", float("nan")))
                    diag.on_vlm_tick(num_steps, sent_frames, stream_output,
                                     vlm_vel_commands, time_to_go, cur_p, cur_y, _d)

        applied_cmd = controller.velocity() if controller is not None else vlm_vel_commands
        obs, _, done, infos = env.step(
            torch.tensor(applied_cmd, device=obs.device, dtype=torch.float32)
        )
        _latest["infos"] = infos; _latest["num_steps"] = num_steps   # for the SIGTERM handler
        if not rlog.log_state(env, num_steps):  # [LOG] replay mode only: state differs from run A
            term_reason = "replay_mismatch"
            break

        if done or env.is_stop_called or num_steps > max_episode_steps:
            if term_reason == "running":  # preserve proximity_stop / stop_assist
                term_reason = ("stop" if env.is_stop_called
                               else "sim_done" if done else "step_cap")
            break

        cur_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        root_vel_w = env.unwrapped.scene["robot"].data.root_vel_w[0].detach().cpu().numpy()
        if diag is not None:
            _cq = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
            _cd = float(infos.get("measurements", {}).get("distance_to_goal", float("nan")))
            diag.log_step(num_steps, cur_pos, nav_diag.yaw_from_quat(_cq), _cd)
        # Hopeless-stall backstop, mirrors wrappers_v3.check_same_pos (audit rank 5,
        # validated 2026-06-10): WINDOWED net progress over 500 commanded steps (10 s).
        # Instantaneous gates cannot work — a wall-jammed K1 wobbles its base ~1 rad/s
        # and creeps ~0.05 m/s, so only integrated displacement/yaw separates a jam
        # (~0.15 m, ~8 deg / 10 s) from legit motion (~4 m walk or 90+ deg turn).
        # Zero command (warmup/settle/stand) clears the window. The FAST per-command
        # stall lever (3 s -> re-query) lives in ClosedLoopController.update().
        _cq_all = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        if np.linalg.norm(applied_cmd) < 0.1:
            stall_buf.clear()
        else:
            stall_buf.append((float(cur_pos[0]), float(cur_pos[1]),
                              float(nav_diag.yaw_from_quat(_cq_all))))
        same_pos_count = len(stall_buf)
        prev_pos = cur_pos
        if len(stall_buf) >= 500:
            _x0, _y0, _w0 = stall_buf[0]
            _x1, _y1, _w1 = stall_buf[-1]
            _nd = math.hypot(_x1 - _x0, _y1 - _y0)
            _ny = abs(nav_diag.wrap_pi(_w1 - _w0))
            if _nd < 0.25 and _ny < math.radians(30.0):
                print(f"Hopeless-stall: {_nd:.2f} m / {math.degrees(_ny):.0f} deg net "
                      f"over 500 commanded steps (10 s) — breaking.")
                term_reason = "stuck"
                break

        if num_steps % steps_per_image == 0:
            curr_frame = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            image_observations.append(rlog.hist_frame(curr_frame, num_steps))  # [LOG] was Image.fromarray(curr_frame)
            curr_frame_copy = curr_frame.copy()
            add_instruction_on_img(curr_frame_copy, instruction.instruction_text)
        if num_steps % steps_per_viz_image == 0:
            curr_vis_frame = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            add_instruction_on_img(curr_vis_frame, stream_output)
            rgb_obses.append(np.concatenate(
                [curr_frame_copy, _match_heights(curr_frame_copy, curr_vis_frame)], axis=1))

        num_steps += 1
        # (stop is now registered before env.step at the VLM tick above)

    measurements = infos["measurements"]
    if diag is not None:
        _fp = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
        _fq = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
        diag.finalize(measurements, _fp, nav_diag.yaw_from_quat(_fq), extra={
            "ended_at_step": int(num_steps),
            "max_episode_steps": int(max_episode_steps),
            "hit_step_cap": bool(num_steps > max_episode_steps),
            "term_reason": term_reason,
            "closed_loop": bool(args_cli.closed_loop),
            "clean_render": bool(args_cli.clean_render),
            "bright": bool(args_cli.bright),
            "max_episode_s": float(args_cli.max_episode_s),
            "max_chunk_m": float(args_cli.max_chunk_m),
            "stop_assist": bool(args_cli.stop_assist),
            "proximity_stop": float(args_cli.proximity_stop),
        })
    # ADDITIVE (2026-07-23): always record termination diagnostics in the production JSON,
    # not only under --diag. Authoritative (set AFTER the diag block). hit_step_cap derives
    # from the terminator's own verdict (term_reason), not a recomputed >/>= boundary. No
    # scored metric is touched.
    measurements["term_reason"]       = term_reason
    measurements["ended_at_step"]     = int(num_steps)
    measurements["max_episode_steps"] = int(max_episode_steps)
    measurements["hit_step_cap"]      = (term_reason == "step_cap")
    result_dir = args_cli.log_dir  # [LOG] was f"eval_results/{args_cli.task}_loco_{args_cli.out_tag}"
    os.makedirs(result_dir, exist_ok=True)
    measurement_dir = os.path.join(result_dir, "measurements")
    os.makedirs(measurement_dir, exist_ok=True)
    with open(f"{measurement_dir}/{int(episode['episode_id'])-1}.json", "w") as f:
        json.dump(measurements, f, indent=4)
    rlog.close()  # [LOG] states_at_queries.npz; flush the line logs

    video_dir = os.path.join(result_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    writer = imageio.get_writer(
        f"{video_dir}/output_{int(episode['episode_id'])-1}.mp4", fps=10
    )
    for frame in rgb_obses:
        writer.append_data(frame.astype(np.uint8))
    writer.close()

    env.close()


if __name__ == "__main__":
    try:  # [LOG] Kit stays alive after an uncaught exception: print it, save it, hard-exit with 3
        main()
    except Exception:
        traceback.print_exc()
        try:
            with open(os.path.join(args_cli.log_dir, "error.txt"), "w") as _fh:
                _fh.write(traceback.format_exc())
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
    simulation_app.close()
