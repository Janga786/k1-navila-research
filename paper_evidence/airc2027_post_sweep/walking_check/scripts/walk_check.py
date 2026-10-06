"""AIRC 2027 walking check (GPU): how well does model_14498 follow the closed-loop executor's commands?

Robot E = the robot exactly as evaluated (unmodified k1_matterport_vision config, K1_locomotion.urdf);
robot T = the training robot model (K1_22dof.urdf, copied config configs/k1_walkcheck_T_robot_cfg.py).
Both on Isaac Sim 4.1 / Isaac Lab fork 4d558ec (conda env vlnce-isaac). See ../PLAN.md.

The environment is built with the evaluator's own lines: airc2027_renders/scripts/sim_setup.build_env (Oct 1) executes
navila_eval_v3.main() verbatim up to `obs, infos = env.reset()`, incl. build_v3_actor(model_14498.pt) and
VLNEnvWrapperV3 with its 1-s reset warm-up. Nothing here reimplements that code:
  * plane trials: `--task` names a copied config registered below at runtime (configs/k1_walkcheck_plane*_cfg.py);
    after the evaluator's reset_start_pos_rot, an env_cfg hook (sim_setup's own mechanism) sets the plane spawn;
  * robot T: VLNEnvWrapperV3 in omni.isaac.vlnce.utils is replaced by the copy wrappers_v3_T.VLNEnvWrapperV3T before
    build_env imports it (robot E uses the unmodified class).
Every control step of every phase is recorded by wrapping RslRlVecEnvWrapper.step (reads state after the original step
returns; changes nothing). Nothing in the sweep tree is edited; nothing is written to eval_results/; no model server.

Modes
  --mode validate  robot E, unmodified Matterport scene (task k1_matterport_vision, --episode_idx 0, --cam_z 0.25),
                   protocol init50 of airc2027_renders/scripts/sim_capture.py; the states are compared byte-for-byte
                   with the Oct 1 reference (exit 0 = identical, 3 = differs).
  --mode trial     plane: reset warm-up (1 s) -> the evaluator's 4-s zero-command warm-up -> the test command
                   (--trial stand | forward | turn_left | turn_right | sequence). Stops early on the env's
                   bad_orientation termination (tilt > 1.3 rad).
Run through run_gpu.sh (cwd NaVILA-Bench, --num_envs=1 --checkpoint=.../model_14498.pt --gait_phase_init=0.0
--headless --enable_cameras).
"""
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import time

print = functools.partial(print, flush=True)  # noqa: A001

HERE = os.path.dirname(os.path.abspath(__file__))
KR = os.path.expanduser("~/Projects/k1_research")
sys.path.insert(0, os.path.join(KR, "airc2027_renders", "scripts"))  # Oct 1 sim_setup = the evaluator's own lines
sys.path.insert(0, os.path.join(HERE, "configs"))  # copied configs
sys.path.insert(0, HERE)  # wrappers_v3_T
import sim_setup  # noqa: E402

PI6 = math.pi / 6.0  # evaluator get_vel_command: sign * np.pi / 6.0
FWD, ZERO, LEFT, RIGHT = (0.5, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, PI6), (0.0, 0.0, -PI6)
TRIAL_CMDS = {"stand": ZERO, "forward": FWD, "turn_left": LEFT, "turn_right": RIGHT}
SEQ_CYCLE = [("forward", 75, FWD), ("zero", 8, ZERO), ("turn_left", 75, LEFT), ("zero", 8, ZERO),
             ("forward", 75, FWD), ("zero", 8, ZERO), ("turn_right", 75, RIGHT), ("zero", 8, ZERO)]
TRIAL_S = {"stand": 12.0, "forward": 12.0, "turn_left": 12.0, "turn_right": 12.0, "sequence": 30.0}
TASK_PLANE = {"E": "k1_walkcheck_plane_E", "T": "k1_walkcheck_plane_T"}
REF_DIR = os.path.join(KR, "airc2027_renders", "diagnosis", "repeatability", "step4_1_idx0_cz025_proc1")
PHASE_ID = {"reset_warmup": 0, "zero_warmup": 1, "command": 2}


def command_schedule(trial, dt_step):
    """Per-step test commands and the segment table [type, start, end (exclusive), complete, command]."""
    n = int(round(TRIAL_S[trial] / dt_step))
    if trial in TRIAL_CMDS:
        return [TRIAL_CMDS[trial]] * n, [[trial, 0, n, True, list(TRIAL_CMDS[trial])]]
    cmds, segs, k, i = [], [], 0, 0
    while k < n:
        typ, length, c = SEQ_CYCLE[i % len(SEQ_CYCLE)]
        end = min(k + length, n)
        cmds += [c] * (end - k)
        segs.append([typ, k, end, end - k == length, list(c)])
        k, i = end, i + 1
    return cmds, segs


def extra(p):
    p.add_argument("--mode", choices=["validate", "trial"], required=True)
    p.add_argument("--robot", choices=["E", "T"], required=True)
    p.add_argument("--trial", choices=list(TRIAL_S), default=None)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--out", required=True, help="validate: output directory; trial: output stem (<stem>.npz/.json)")


args, simulation_app = sim_setup.launch("AIRC 2027 walking check", extra)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.isaac.lab_tasks  # noqa: E402,F401
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.isaac.vlnce.config  # noqa: E402,F401  (registers the evaluator's tasks)
import omni.isaac.vlnce.utils as vlnce_utils  # noqa: E402
from omni.isaac.vlnce.config.k1 import k1_matterport_vision_cfg as EVAL_CFG  # noqa: E402
import k1_walkcheck_plane_cfg as PLANE_E  # noqa: E402
import k1_walkcheck_plane_T_cfg as PLANE_T  # noqa: E402
from wrappers_v3_T import VLNEnvWrapperV3T  # noqa: E402

ORIG_WRAPPER = vlnce_utils.VLNEnvWrapperV3
PLANE_MODS = {"E": PLANE_E, "T": PLANE_T}
for _r, _mod in PLANE_MODS.items():  # runtime registration of the copied configs (same entry point as the evaluator's)
    gym.register(id=TASK_PLANE[_r], entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": _mod.K1MatterportVisionCfg,
                         "rsl_rl_cfg_entry_point": _mod.K1VisionRoughPPORunnerCfg})


def _np(t):
    return t.detach().to("cpu").numpy().copy()


class Recorder:
    """State after every control step (RslRlVecEnvWrapper.step), float32 as produced."""

    def __init__(self):
        self.phase = "reset_warmup"
        self.vln = None  # set after build_env; during the reset warm-up the wrapper's command is zero (reset() zeroes it)
        self.rows = []
        self.fall = None

    def __call__(self, vecenv, actions, dones):
        u = vecenv.unwrapped
        d = u.scene["robot"].data
        cmd = self.vln._command[0] if self.vln is not None else torch.zeros(3, device=u.device)
        row = {
            "phase": np.int8(PHASE_ID[self.phase]),
            "cmd": _np(cmd).astype(np.float32),
            "root_pos_w": _np(d.root_pos_w[0]), "root_quat_w": _np(d.root_quat_w[0]),
            "root_lin_vel_w": _np(d.root_lin_vel_w[0]), "root_ang_vel_w": _np(d.root_ang_vel_w[0]),
            "root_lin_vel_b": _np(d.root_lin_vel_b[0]), "root_ang_vel_b": _np(d.root_ang_vel_b[0]),
            "projected_gravity_b": _np(d.projected_gravity_b[0]),
            "joint_pos": _np(d.joint_pos[0]), "joint_vel": _np(d.joint_vel[0]),
            "action": _np(actions[0]),
            "env_done": np.bool_(bool(dones[0])),
            "terminated": np.bool_(bool(u.reset_terminated[0])),
            "time_out": np.bool_(bool(u.reset_time_outs[0])),
            "bad_orientation": np.bool_(bool(u.termination_manager.get_term("bad_orientation")[0])),
            "episode_length": np.int32(int(u.episode_length_buf[0])),
            "wrapper_done": np.bool_(False),
        }
        self.rows.append(row)
        if self.fall is None and (row["env_done"] or row["terminated"] or row["bad_orientation"]):
            self.fall = {"row": len(self.rows) - 1, "phase": self.phase,
                         "bad_orientation": bool(row["bad_orientation"]), "terminated": bool(row["terminated"]),
                         "time_out": bool(row["time_out"]),
                         "note": "env reset itself in this step; the state recorded in this row is the reset state"}
            print(f"[walk_check] ENV TERMINATION at row {len(self.rows) - 1} phase {self.phase}: {self.fall}")


REC = Recorder()
_ORIG_STEP = RslRlVecEnvWrapper.step


def _recording_step(self, actions):
    out = _ORIG_STEP(self, actions)
    REC(self, actions, out[2])
    return out


RslRlVecEnvWrapper.step = _recording_step


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def cfg_diff(a, b, path=""):
    """Paths where two to_dict() trees differ (tuples and lists compared alike)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append([p, jsonable(a.get(k, "<absent>")), jsonable(b.get(k, "<absent>"))])
            else:
                out += cfg_diff(a[k], b[k], p)
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [[path, jsonable(a), jsonable(b)]]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += cfg_diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [[path, jsonable(a), jsonable(b)]]


HOOK_LOG = {}
ROBOT_CFG_AT_GYM_MAKE = {}  # robot cfg as passed to gym.make (the scene later formats prim_path in place)


def capture_hook(env_cfg):
    """Validation: changes nothing, only records the robot cfg the env is built with."""
    ROBOT_CFG_AT_GYM_MAKE["cfg"] = env_cfg.scene.robot.to_dict()


def make_plane_hook(cfgmod):
    def hook(env_cfg):
        HOOK_LOG["set_by_evaluator_reset_start_pos_rot"] = {
            "pos": jsonable(env_cfg.scene.robot.init_state.pos), "rot": jsonable(env_cfg.scene.robot.init_state.rot)}
        env_cfg.scene.robot.init_state.pos = tuple(cfgmod.PLANE_SPAWN_POS)
        env_cfg.scene.robot.init_state.rot = tuple(cfgmod.PLANE_SPAWN_ROT)
        HOOK_LOG["plane_spawn"] = {"pos": list(cfgmod.PLANE_SPAWN_POS), "rot": list(cfgmod.PLANE_SPAWN_ROT)}
        print(f"[walk_check] plane spawn hook: {HOOK_LOG}")
        capture_hook(env_cfg)
    return hook


def physx_readback(robot):
    v = robot.root_physx_view

    def c(t):
        return t.detach().to("cpu")
    coms = c(v.get_coms()[0])
    return {
        "units": "as returned by the PhysX tensor API (Isaac Sim 4.1)",
        "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
        "dof_stiffness": c(v.get_dof_stiffnesses()[0]).tolist(), "dof_damping": c(v.get_dof_dampings()[0]).tolist(),
        "dof_armature": c(v.get_dof_armatures()[0]).tolist(),
        "dof_friction": c(v.get_dof_friction_coefficients()[0]).tolist(),
        "dof_max_force": c(v.get_dof_max_forces()[0]).tolist(),
        "dof_max_velocity": c(v.get_dof_max_velocities()[0]).tolist(),
        "body_masses": c(v.get_masses()[0]).tolist(), "total_mass": float(c(v.get_masses()[0]).sum()),
        "body_coms_in_link_frame_pos_quat": coms.tolist(), "body_inertias": c(v.get_inertias()[0]).tolist(),
        "root_body": robot.body_names[0], "root_com_b": coms[0, :3].tolist(),
    }


def ground_usd_info(prim_path):
    """Collision/material attributes authored under the ground prim (plane runs)."""
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    info = []
    if not root.IsValid():
        return {"error": f"no prim at {prim_path}"}
    for prim in Usd.PrimRange(root):
        attrs = {a.GetName(): jsonable(a.Get()) for a in prim.GetAttributes()
                 if a.GetName().startswith(("physics", "physx", "physxCollision", "physxMaterial"))}
        rels = {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()
                if "material:binding" in r.GetName()}
        if attrs or rels or prim.GetTypeName() in ("Plane", "Mesh"):
            info.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                         "apis": [str(s) for s in prim.GetAppliedSchemas()], "attrs": attrs, "rels": rels})
    return info


def snap_like_sim_capture(robot, out_dir, tag):
    """Exactly the arrays sim_capture.py saves (root_state_w; [body_pos_w | body_quat_w])."""
    rs = robot.data.root_state_w[0].cpu().numpy()
    bs = torch.cat([robot.data.body_pos_w[0], robot.data.body_quat_w[0]], -1).cpu().numpy()
    np.save(f"{out_dir}/{tag}_root_state.npy", rs)
    np.save(f"{out_dir}/{tag}_body_states.npy", bs)
    return rs, bs


def compare_with_reference(out_dir):
    cap = json.load(open(os.path.join(REF_DIR, "capture.json")))
    res = {"reference_dir": REF_DIR, "items": {}, "hex": {}}
    ok = True
    for tag in ("init", "after50"):
        for kind in ("root_state", "body_states"):
            name = f"{tag}_{kind}.npy"
            a = np.load(os.path.join(out_dir, name))
            b = np.load(os.path.join(REF_DIR, name))
            same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            item = {"dtype": [str(a.dtype), str(b.dtype)], "shape": [list(a.shape), list(b.shape)],
                    "bytes_identical": bool(same),
                    "file_sha256": [sha256(os.path.join(out_dir, name)), sha256(os.path.join(REF_DIR, name))]}
            if a.shape == b.shape:
                diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                item["max_abs_diff"] = float(diff.max())
                item["n_values_differing"] = int((a.view(np.uint32) != b.view(np.uint32)).sum()) \
                    if a.dtype == b.dtype == np.float32 else None
            res["items"][name] = item
            ok &= same
        rs = np.load(os.path.join(out_dir, f"{tag}_root_state.npy"))
        hx = rs.astype(np.float32).tobytes().hex()
        ref_hx = cap["frames"][tag]["root_state_float32_hex"]
        res["hex"][tag] = {"harness": hx, "reference": ref_hx, "equal": hx == ref_hx}
        ok &= hx == ref_hx
    res["pass"] = bool(ok)
    return res


def main():
    t0 = time.time()
    meta = {"script": "airc2027_walking/scripts/walk_check.py", "argv": sys.argv, "mode": args.mode,
            "robot": args.robot, "trial": args.trial, "rep": args.rep, "task": args.task,
            "episode_idx": args.episode_idx, "cam_z": args.cam_z, "gait_phase_init": args.gait_phase_init,
            "checkpoint": args.checkpoint, "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.mode == "validate":
        assert args.robot == "E" and args.task == "k1_matterport_vision" and args.episode_idx == 0, \
            "validation = robot E on the unmodified task, dataset index 0"
        hook = capture_hook  # records only; the config stays as the evaluator built it
        os.makedirs(args.out, exist_ok=True)
    else:
        assert args.trial is not None and args.task == TASK_PLANE[args.robot], \
            f"trial mode needs --trial and --task={TASK_PLANE[args.robot]}"
        hook = make_plane_hook(PLANE_MODS[args.robot])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.robot == "T":
        vlnce_utils.VLNEnvWrapperV3 = VLNEnvWrapperV3T  # build_env does `from omni.isaac.vlnce.utils import VLNEnvWrapperV3`

    env, episode, env_cfg, obs, infos = sim_setup.build_env(args, env_cfg_hook=hook)  # incl. the 1-s reset warm-up
    t_built = time.time()
    REC.vln = env
    u = env.unwrapped
    robot = u.scene["robot"]
    expected = VLNEnvWrapperV3T if args.robot == "T" else ORIG_WRAPPER
    assert type(env) is expected, f"wrapper {type(env)} != {expected}"
    names = list(robot.joint_names)
    if args.robot == "E":
        assert len(names) == 12 and robot.num_bodies == 13, (names, robot.body_names)
    else:
        assert len(names) == 22 and robot.num_bodies == 23, (names, robot.body_names)
    leg_joints = list(vlnce_utils.wrappers_v3.LEG_JOINTS)  # the policy's training order
    _, leg_names = robot.find_joints(leg_joints, preserve_order=True)
    assert list(leg_names) == leg_joints, leg_names
    used_order = [names[i] for i in env._jperm.tolist()]  # order the wrapper actually used (obs and actions)
    assert used_order == leg_joints, used_order

    # robot config vs the evaluated scene's robot config (E: only the spawn pose may differ)
    eval_robot = EVAL_CFG.K1MatterportVisionCfg().scene.robot
    rdiff = cfg_diff(eval_robot.to_dict(), ROBOT_CFG_AT_GYM_MAKE["cfg"])
    if args.robot == "E":
        assert all(p[0].startswith(("init_state.pos", "init_state.rot")) for p in rdiff), rdiff
    meta.update({
        "wrapper_class": f"{type(env).__module__}.{type(env).__name__}",
        "policy_leg_order_used": used_order,
        "robot_cfg_vs_evaluated_scene_robot": rdiff,
        "robot_urdf": env_cfg.scene.robot.spawn.asset_path, "fix_base": env_cfg.scene.robot.spawn.fix_base,
        "merge_fixed_joints": env_cfg.scene.robot.spawn.merge_fixed_joints,
        "action_dim": int(u.action_manager.total_action_dim),
        "action_term_joint_names": list(u.action_manager.get_term("joint_pos")._joint_names),
        "default_root_state": _np(robot.data.default_root_state[0]).tolist(),
        "default_joint_pos": _np(robot.data.default_joint_pos[0]).tolist(),
        "env_origin": _np(u.scene.env_origins[0]).tolist(),
        "terrain": {"type": u.cfg.scene.terrain.terrain_type, "prim_path": u.cfg.scene.terrain.prim_path,
                    "physics_material": jsonable(u.cfg.scene.terrain.physics_material.to_dict())},
        "sim": {"dt": u.cfg.sim.dt, "decimation": u.cfg.decimation, "render_interval": u.cfg.sim.render_interval,
                "disable_contact_processing": u.cfg.sim.disable_contact_processing,
                "physics_material": jsonable(u.cfg.sim.physics_material.to_dict()),
                "physx": jsonable(u.cfg.sim.physx.to_dict()), "device": u.cfg.sim.device,
                "use_fabric": u.cfg.sim.use_fabric},
        "terminations": {k: jsonable(getattr(u.cfg.terminations, k).params)
                         for k in ("bad_orientation",) if hasattr(u.cfg.terminations, k)},
        "plane_spawn_hook": HOOK_LOG,
        "episode": {"episode_id": episode["episode_id"], "scene_id": episode["scene_id"]},
        "rows_reset_warmup": len(REC.rows),
    })
    if args.mode == "trial":
        assert u.cfg.scene.terrain.terrain_type == "plane"
        assert np.allclose(meta["env_origin"], [0.0, 0.0, 0.0])
        assert np.allclose(meta["default_root_state"][:7], list(PLANE_MODS[args.robot].PLANE_SPAWN_POS)
                           + list(PLANE_MODS[args.robot].PLANE_SPAWN_ROT), atol=1e-7)
    print(f"[walk_check] built: robot {args.robot} wrapper {meta['wrapper_class']} joints {len(names)} "
          f"action_dim {meta['action_dim']} leg order {used_order}")

    rc = 0
    t_cmd0 = t_cmd1 = time.time()
    if args.mode == "validate":
        snap_like_sim_capture(robot, args.out, "init")
        REC.phase = "command"
        cmd = torch.tensor([0.5, 0.0, 0.0], device=obs.device)  # sim_capture: torch.tensor([0.5, 0, 0], device=obs.device)
        for _ in range(50):
            obs, _, done, infos = env.step(cmd)
        snap_like_sim_capture(robot, args.out, "after50")
        t_cmd1 = time.time()
        res = compare_with_reference(args.out)
        json.dump(res, open(os.path.join(args.out, "compare.json"), "w"), indent=1)
        print(f"[walk_check] VALIDATION {'PASS' if res['pass'] else 'FAIL'}: "
              f"{ {k: v['bytes_identical'] for k, v in res['items'].items()} } hex {[v['equal'] for v in res['hex'].values()]}")
        rc = 0 if res["pass"] else 3
        stem = os.path.join(args.out, "steps")
    else:
        dt_step = u.cfg.sim.dt * u.cfg.decimation
        assert abs(dt_step - 0.02) < 1e-12, dt_step
        steps_per_image = 0.5 / (u.cfg.sim.dt * u.cfg.decimation)  # evaluator
        warmup_total_steps = int(8 * steps_per_image)  # evaluator: warmup_target_frames = 8 -> 200
        cmds, segs = command_schedule(args.trial, dt_step)
        meta["schedule"] = {"dt_step": dt_step, "zero_warmup_steps": warmup_total_steps, "command_steps": len(cmds),
                            "segments": segs}
        if REC.fall is None:
            REC.phase = "zero_warmup"
            zero_cmd = torch.zeros(3, device=obs.device)  # evaluator: zero_cmd = torch.zeros(3, device=obs.device)
            for _ in range(warmup_total_steps):
                obs, _, done, infos = env.step(zero_cmd)
                REC.rows[-1]["wrapper_done"] = np.bool_(bool(done))
                if REC.fall is not None:
                    break
        t_cmd0 = time.time()
        if REC.fall is None:
            REC.phase = "command"
            for c in cmds:
                obs, _, done, infos = env.step(torch.tensor(c, device=obs.device, dtype=torch.float32))
                REC.rows[-1]["wrapper_done"] = np.bool_(bool(done))
                if REC.fall is not None:
                    break
        t_cmd1 = time.time()
        stem = args.out
    # read-backs AFTER the last recorded step (read-only; masses, COMs and joint parameters do not change in a run)
    meta["physx"] = physx_readback(robot)
    if args.mode == "trial":
        meta["ground_usd"] = ground_usd_info(u.cfg.scene.terrain.prim_path)
    print(f"[walk_check] PhysX read-back: mass {meta['physx']['total_mass']:.4f} root_com_b {meta['physx']['root_com_b']}")
    rows = REC.rows
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(stem + ".npz", **arrays)
    phases = arrays["phase"]
    meta.update({
        "rows": {"total": len(rows), "reset_warmup": int((phases == 0).sum()), "zero_warmup": int((phases == 1).sum()),
                 "command": int((phases == 2).sum())},
        "fall": REC.fall,
        "wrapper_done_rows": np.flatnonzero(arrays["wrapper_done"]).tolist(),
        "wall_s": {"build_incl_reset_warmup": t_built - t0, "zero_warmup": t_cmd0 - t_built,
                   "command": t_cmd1 - t_cmd0, "total_before_close": time.time() - t0},
        "file_sha256": {p: sha256(os.path.join(KR, p)) for p in (
            "NaVILA-Bench-main/scripts/navila_eval_v3.py",
            "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/config/k1/k1_matterport_vision_cfg.py",
            "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/wrappers_v3.py",
            "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py",
            "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/measures.py",
            "IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/sim/simulation_cfg.py",
            "checkpoints/model_14498.pt",
            "airc2027_renders/scripts/sim_setup.py", "airc2027_renders/scripts/navila_eval_v3_COPY.py",
            "airc2027_renders/configs/k1_corrected_robot_cfg.py",
            "airc2027_walking/scripts/walk_check.py", "airc2027_walking/scripts/wrappers_v3_T.py",
            "airc2027_walking/scripts/configs/k1_walkcheck_plane_cfg.py",
            "airc2027_walking/scripts/configs/k1_walkcheck_plane_T_cfg.py",
            "airc2027_walking/scripts/configs/k1_walkcheck_T_robot_cfg.py",
            "booster/booster_assets/robots/K1/K1_locomotion.urdf", "booster/booster_assets/robots/K1/K1_22dof.urdf")},
        "k1_research_commit": git_head(KR), "isaaclab_commit": git_head(os.path.join(KR, "IsaacLab")),
        "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
        "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True).stdout.strip(),
        "array_keys": {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()},
    })
    json.dump(jsonable(meta), open(stem + ".json", "w"), indent=1)
    print(f"[walk_check] saved {stem}.npz/.json rows={meta['rows']} fall={REC.fall} "
          f"wall={meta['wall_s']['total_before_close']:.1f}s")
    env.close()
    return rc


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001  (Kit keeps the process alive after an uncaught exception)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    if rc != 0:
        sys.stdout.flush()
        os._exit(rc)
    simulation_app.close()
