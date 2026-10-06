"""Shared simulator setup for the AIRC 2027 render/diagnosis scripts.

`launch(extra_args_fn)` parses the evaluator's own command-line arguments (copied verbatim from
NaVILA-Bench-main/scripts/navila_eval_v3.py, sweep-time version, SHA-256 9b0107...) plus script-specific
extras, then starts Isaac Sim exactly as the evaluator does.

`build_env(args)` executes the evaluator's `main()` lines up to and including `obs, infos = env.reset()`
verbatim (same task, same `reset_start_pos_rot`, same `--cam_z` override, same `VLNEnvWrapperV3` with its
1-s reset warm-up, same checkpoint loader, same viewport camera call). Nothing in the sweep tree is edited;
the evaluator's helper modules are imported read-only from NaVILA-Bench-main/scripts.

Must be run from ~/Projects/k1_research/NaVILA-Bench (as the runner does) in conda env vlnce-isaac with
OMNI_KIT_ACCEPT_EULA=yes. Never writes into eval_results/.
"""
import argparse
import os
import sys

KR = os.path.expanduser("~/Projects/k1_research")
NB_SCRIPTS = os.path.join(KR, "NaVILA-Bench-main", "scripts")
if NB_SCRIPTS not in sys.path:
    sys.path.insert(0, NB_SCRIPTS)  # read-only import of cli_args / nav_diag, as `python scripts/...` does

from omni.isaac.lab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402  isort: skip


def make_parser(description):
    # ---- verbatim copy of navila_eval_v3.py argument definitions ----
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--task", type=str, default="k1_matterport_vision")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--gait_phase_init", type=float, default=0.0)
    parser.add_argument("--vlm_host", type=str, default="localhost")
    parser.add_argument("--vlm_port", type=int, default=54321)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_tag", type=str, default="v3")
    parser.add_argument("--diag", action="store_true", default=False)
    parser.add_argument("--clean_render", action="store_true", default=False)
    parser.add_argument("--bright", action="store_true", default=False)
    parser.add_argument("--vlm_transform", choices=["stretch", "crop", "pad"], default="stretch")
    parser.add_argument("--closed_loop", action="store_true", default=False)
    parser.add_argument("--max_episode_s", type=float, default=50.0)
    parser.add_argument("--max_chunk_m", type=float, default=0.0)
    parser.add_argument("--stop_assist", action="store_true", default=False)
    parser.add_argument("--proximity_stop", type=float, default=0.0)
    parser.add_argument("--cam_z", type=float, default=None)
    parser.add_argument("--cam_width", type=int, default=None)
    parser.add_argument("--cam_height", type=int, default=None)
    parser.add_argument("--cam_aperture", type=float, default=None)
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def launch(description, extra_args_fn=None, argv=None):
    parser = make_parser(description)
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args(argv)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def build_env(args_cli, env_cfg_hook=None):
    """Verbatim copy of navila_eval_v3.main() from its first line to `obs, infos = env.reset()`.
    `env_cfg_hook(env_cfg)` (default None = as evaluated) may modify the cfg AFTER the evaluator's own
    edits; it is used only for the 'corrected' robot-model images."""
    import math

    import gymnasium as gym
    import omni.isaac.lab_tasks  # noqa: F401
    import torch  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    import omni.isaac.vlnce.config  # noqa: F401  (evaluator: `from ... import *`; registers the gym tasks)
    from omni.isaac.vlnce.utils import ASSETS_DIR, VLNEnvWrapperV3
    from omni.isaac.vlnce.utils.eval_utils import read_episodes

    ev = load_eval_helpers()

    r2r_data_path = os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz")
    all_episodes = read_episodes(r2r_data_path)
    episode = all_episodes[args_cli.episode_idx]

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg = ev.reset_start_pos_rot(env_cfg, args_cli, episode)

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

    if env_cfg_hook is not None:
        env_cfg_hook(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    device = env.unwrapped.device
    actor = ev.build_v3_actor(args_cli.checkpoint, device=str(device))

    def policy_fn(obs_235):
        with torch.no_grad():
            return actor(obs_235)

    env = VLNEnvWrapperV3(
        env, policy_fn, args_cli.task, episode,
        high_level_obs_key="camera_obs",
        gait_phase_init=args_cli.gait_phase_init,
    )

    robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w[0].detach().cpu().numpy()
    _, _, yaw = ev.quat2eulers(*robot_quat_w)
    cam_eye = (robot_pos_w[0] - 0.8 * math.sin(-yaw),
               robot_pos_w[1] - 0.8 * math.cos(-yaw),
               robot_pos_w[2] + 0.8)
    cam_target = tuple(robot_pos_w)
    env.unwrapped.sim.set_camera_view(eye=cam_eye, target=cam_target)

    obs, infos = env.reset()
    return env, episode, env_cfg, obs, infos


_EV = None


def load_eval_helpers():
    """Pull the pure helper functions (reset_start_pos_rot, build_v3_actor, quat2eulers, square_frame,
    _match_heights) out of the verbatim evaluator copy without executing its module-level app launch."""
    global _EV
    if _EV is not None:
        return _EV
    import ast
    import types
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navila_eval_v3_COPY.py")
    src = open(src_path).read()
    tree = ast.parse(src)
    keep = {"build_v3_actor", "quat2eulers", "reset_start_pos_rot", "square_frame", "_match_heights",
            "brighten_frame"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    mod = types.ModuleType("navila_eval_v3_helpers")
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from omni.isaac.vlnce.utils import ASSETS_DIR
    mod.__dict__.update(dict(os=os, math=math, np=np, torch=torch, nn=nn, Image=Image, ASSETS_DIR=ASSETS_DIR))
    exec(compile(ast.Module(body=body, type_ignores=[]), src_path, "exec"), mod.__dict__)
    _EV = mod
    return mod
