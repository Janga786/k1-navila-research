"""Step 5B/5C: path-traced third-person images of the K1 (GPU).

--model as_evaluated : environment built exactly as the evaluator does (sim_setup.build_env, unmodified code, robot
                       model and scene), state right after the 1-s reset warm-up of dataset index --episode_idx.
--model corrected    : same evaluator lines and scene, but the robot model replaced (copied config
                       configs/k1_corrected_robot_cfg.py: K1_22dof.urdf, fix_base=True, training upper-body pose) and
                       its Trunk held at the as-evaluated Trunk pose after the reset warm-up (read from --trunk_pose_json).
                       The walking-policy wrapper is not used (it is built for the 12-joint model); the PD holds the
                       training pose for 50 control steps (1 s) before rendering.
--model studio       : the corrected K1 alone: neutral floor, dome light + soft distant light, three-quarter view.

Renderer: a Replicator render product on a new USD camera (the Isaac Lab Camera class has no path-tracing switch),
carb settings /rtx/rendermode=PathTracing, accumulation to --total_spp samples per pixel, OptiX denoiser on.
Physics is not stepped while accumulating (sim.render() only). Nothing is added to the Matterport scene except the
render camera prim itself (cameras are not rendered).
"""
import functools
import json
import math
import os
import subprocess
import sys

print = functools.partial(print, flush=True)  # noqa: A001
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"))
import sim_setup  # noqa: E402


def extra(p):
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", choices=["as_evaluated", "corrected", "studio"], required=True)
    p.add_argument("--views_json", type=str, default="", help="reuse camera poses from a previous run (same views)")
    p.add_argument("--trunk_pose_json", type=str, default="")
    p.add_argument("--total_spp", type=int, default=256)
    p.add_argument("--spp_per_frame", type=int, default=32)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)


args, simulation_app = sim_setup.launch("path-traced renders", extra)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def git_commit():
    return subprocess.run(["git", "-C", os.path.expanduser("~/Projects/k1_research"), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def build_corrected(trunk_pose):
    """Evaluator lines up to RslRlVecEnvWrapper, with the robot model replaced by the copied corrected config."""
    import gymnasium as gym
    import omni.isaac.lab_tasks  # noqa: F401
    import omni.isaac.vlnce.config  # noqa: F401
    from omni.isaac.lab_tasks.utils import parse_env_cfg
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
    from omni.isaac.vlnce.utils import ASSETS_DIR
    from omni.isaac.vlnce.utils.eval_utils import read_episodes
    from k1_corrected_robot_cfg import make_corrected_robot_cfg
    ev = sim_setup.load_eval_helpers()
    episode = read_episodes(os.path.join(ASSETS_DIR, "vln_ce_isaac_v1.json.gz"))[args.episode_idx]
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg = ev.reset_start_pos_rot(env_cfg, args, episode)
    rc = make_corrected_robot_cfg(trunk_pose["pos"], trunk_pose["quat_wxyz"])
    env_cfg.scene.robot = rc.replace(prim_path=env_cfg.scene.robot.prim_path)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg, render_mode=None))
    env.reset()
    act = torch.zeros(1, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for _ in range(50):  # PD settles at the default (= training) joint pose; base is fixed
        env.step(act)
    return env, episode, env_cfg


def build_studio(trunk_pose):
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import Articulation
    from k1_corrected_robot_cfg import make_corrected_robot_cfg
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    floor = sim_utils.CuboidCfg(size=(8.0, 8.0, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.62, 0.62, 0.60), roughness=0.85))
    floor.func("/World/floor", floor, translation=(0.0, 0.0, -0.01))
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0))
    dome.func("/World/dome", dome)
    key = sim_utils.DistantLightCfg(intensity=1600.0, color=(1.0, 0.98, 0.95), angle=8.0)
    key.func("/World/key", key, orientation=(0.88, 0.2, -0.35, 0.25))
    z = trunk_pose["height_above_floor"]
    rc = make_corrected_robot_cfg((0.0, 0.0, z), (1.0, 0.0, 0.0, 0.0)).replace(prim_path="/World/K1")
    robot = Articulation(rc)
    sim.reset()
    for _ in range(200):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(0.005)
    return sim, robot


def main():
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    tp = json.load(open(args.trunk_pose_json)) if args.trunk_pose_json else None
    meta_common = {"script": "airc2027_renders/scripts/sim_render_pt.py", "k1_research_commit": git_commit(),
                   "evaluator_sha256": "9b010715ced0a46efb17b86e918653e933bdc26b5fd765f3d8a0912d7b1281fc"}
    if args.model == "as_evaluated":
        env, episode, env_cfg, obs, infos = sim_setup.build_env(args)
        sim = env.unwrapped.sim
        robot = env.unwrapped.scene["robot"]
        ti = robot.body_names.index("Trunk")
        tt = robot.data.body_pos_w[0, ti].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, ti].cpu().numpy().astype(float)
        meta_common.update({"robot_model": "K1_locomotion.urdf (merge_fixed_joints=True), unmodified", "variant": "as_evaluated",
                            "state": "after the evaluator's 1-s reset warm-up (VLNEnvWrapperV3.reset)"})
    elif args.model == "corrected":
        env, episode, env_cfg = build_corrected(tp)
        sim = env.unwrapped.sim
        robot = env.unwrapped.scene["robot"]
        ti = robot.body_names.index("Trunk")
        tt = robot.data.body_pos_w[0, ti].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, ti].cpu().numpy().astype(float)
        meta_common.update({"robot_model": "K1_22dof.urdf, fix_base=True, training upper-body pose (configs/k1_corrected_robot_cfg.py)",
                            "variant": "corrected", "state": "fixed base at the as-evaluated Trunk pose after the reset warm-up; PD settled 1 s",
                            "joint_pos_rendered": dict(zip(robot.joint_names, robot.data.joint_pos[0].cpu().numpy().round(4).tolist()))})
    else:
        sim, robot = build_studio(tp)
        episode, env_cfg = None, None
        tt, tq = np.array([0.0, 0.0, tp["height_above_floor"]]), np.array([1.0, 0, 0, 0])
        meta_common.update({"robot_model": "K1_22dof.urdf, fix_base=True, training upper-body pose", "variant": "corrected",
                            "scene": "studio: 8x8 m neutral-gray floor (PreviewSurface 0.62 albedo, roughness 0.85), dome light 900, distant light 1600 (angle 8 deg)",
                            "joint_pos_rendered": dict(zip(robot.joint_names, robot.data.joint_pos[0].cpu().numpy().round(4).tolist()))})
    print("[trunk]", tt, tq)
    stage = omni.usd.get_context().get_stage()
    fwd = qrot(tq)[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0], 0.0])

    # ---- camera poses ----
    if args.views_json:
        views = json.load(open(args.views_json))
    elif args.model == "studio":
        W, H = 2048, 2048
        tgt = tt + np.array([0, 0, -0.08])
        eye = tgt + 1.85 * (0.80 * fwd + 0.60 * left) + np.array([0, 0, 0.42])
        views = {"studio_three_quarter": {"eye": eye.tolist(), "target": tgt.tolist(), "focal_length": 50.0, "horizontal_aperture": 36.0,
                                          "width": W, "height": H}}
    else:
        from omni.isaac.lab.sensors import RayCaster
        from omni.isaac.lab.utils.warp import raycast_mesh
        mesh = RayCaster.meshes["/World/matterport"]

        def cast(o, d, maxd):
            s = torch.tensor([o], dtype=torch.float32, device=str(mesh.device))
            dd = torch.tensor([d / np.linalg.norm(d)], dtype=torch.float32, device=str(mesh.device))
            hit = raycast_mesh(s, dd, mesh, max_dist=maxd)[0][0].cpu().numpy()
            return float(np.linalg.norm(hit - np.array(o))) if np.all(np.isfinite(hit)) else math.inf

        floor = tt[2] - cast(tt, np.array([0, 0, -1.0]), 3.0)

        def clear(target, eye, margin=0.12):
            d = np.array(eye) - target
            L = np.linalg.norm(d)
            hit = cast(target, d, L + 0.01)
            if hit < L:
                eye = target + d / L * max(0.35, hit - margin)
            return np.array(eye), hit

        tgt_body = np.array([tt[0], tt[1], floor + 0.60])

        def best_dir(angles_deg, origin, maxd=4.0):
            best = None
            for ang in angles_deg:
                a = math.radians(ang)
                dirv = math.cos(a) * fwd + math.sin(a) * left
                d = cast(origin, dirv, maxd)
                if best is None or d > best[0]:
                    best = (d, dirv, ang)
            return best

        def auto_focal(dist, half_extent=0.62, ha=36.0, w=args.width, h=args.height):
            va = ha * h / w  # focal so that +-half_extent around the target fills ~85 % of the image height
            return float(np.clip(va * dist / (2 * half_extent / 0.85), 12.0, 40.0))

        views = {}
        # behind and to the left: search 135..180 deg (180 = straight behind), eye 1.4 m above the floor, <= 1.8 m back
        free, dirv, ang = best_dir(range(135, 181, 5), np.array([tt[0], tt[1], floor + 1.0]))
        hd = min(1.8, free - 0.15)
        e, h = clear(tgt_body, tgt_body + dirv * hd + np.array([0, 0, floor + 1.4 - tgt_body[2]]))
        dist = float(np.linalg.norm(e - tgt_body))
        views["behind_left"] = {"eye": e.tolist(), "target": tgt_body.tolist(), "focal_length": auto_focal(dist),
                                "requested": "about 1.8 m back, to the left, 1.4 m above the floor",
                                "direction_deg_from_heading": ang, "horizontal_free_distance_m": free, "first_hit_from_target_m": h}
        # side: the side (left or right, 70..110 deg) with the most free space, <= 1.8 m
        bl = best_dir(range(70, 111, 5), tgt_body)
        br = best_dir(range(-110, -69, 5), tgt_body)
        free, dirv, ang = max(bl, br, key=lambda b: b[0])
        dist = min(1.8, free - 0.15)
        e = tgt_body + dirv * dist + np.array([0, 0, 0.25])
        views["side"] = {"eye": e.tolist(), "target": tgt_body.tolist(), "focal_length": auto_focal(dist),
                         "requested": "from the side", "direction_deg_from_heading": ang, "distance_m": dist, "free_distance_m": free}
        tgt_up = np.array([tt[0], tt[1], floor + 0.92])
        e, h = clear(tgt_up, tgt_up + 0.85 * (0.75 * fwd + 0.66 * left) + np.array([0, 0, 0.12]))
        views["upper_body_close"] = {"eye": e.tolist(), "target": tgt_up.tolist(), "focal_length": 24.0,
                                     "requested": "closer view of the upper body, front-left", "first_hit_from_target_m": h}
        for v in views.values():
            v.update({"horizontal_aperture": 36.0, "width": args.width, "height": args.height})
        meta_common["floor_z"] = float(floor)

    # ---- path tracing ----
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/spp", args.spp_per_frame)
    s.set("/rtx/pathtracing/totalSpp", args.total_spp)
    s.set("/rtx/pathtracing/optixDenoiser/enabled", True)
    n_frames = int(math.ceil(args.total_spp / args.spp_per_frame)) + 4
    settings_read = {k: s.get(k) for k in ("/rtx/rendermode", "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp",
                                           "/rtx/pathtracing/optixDenoiser/enabled", "/rtx/pathtracing/maxBounces",
                                           "/rtx/pathtracing/clampSpp", "/rtx/pathtracing/adaptiveSampling/enabled")}
    print("[pt settings]", settings_read)

    def render_once():
        if hasattr(sim, "render"):
            sim.render()
        else:
            simulation_app.update()

    for name, v in views.items():
        cam = UsdGeom.Camera.Define(stage, f"/World/PTCam_{name}")
        cam.GetFocalLengthAttr().Set(v["focal_length"])
        cam.GetHorizontalApertureAttr().Set(v["horizontal_aperture"])
        cam.GetVerticalApertureAttr().Set(v["horizontal_aperture"] * v["height"] / v["width"])
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        m = Gf.Matrix4d()
        m.SetLookAt(Gf.Vec3d(*v["eye"]), Gf.Vec3d(*v["target"]), Gf.Vec3d(0, 0, 1))
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(m.GetInverse())
        rp = rep.create.render_product(f"/World/PTCam_{name}", (v["width"], v["height"]))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        for _ in range(n_frames):
            render_once()
        img = np.array(ann.get_data())[:, :, :3]
        for _ in range(3):
            render_once()
        img2 = np.array(ann.get_data())[:, :, :3]
        conv = float(np.abs(img.astype(np.int16) - img2.astype(np.int16)).mean())
        try:  # stop rendering this product for the next views
            ann.detach([rp.path])
            rp.hydra_texture.set_updates_enabled(False)
        except Exception as e:  # noqa: BLE001
            print("[pt] render product not detached/disabled:", repr(e))
        prefix = "corrected_" if meta_common["variant"] == "corrected" else ""
        fn = f"{prefix}{name}"
        Image.fromarray(img).save(f"{out}/{fn}.png")
        meta = dict(meta_common)
        meta.update({"image": f"{fn}.png", "as_evaluated_or_corrected": meta_common["variant"],
                     "dataset_index": args.episode_idx if episode is not None else None,
                     "record": int(episode["episode_id"]) - 1 if episode is not None else None,
                     "scene": os.path.splitext(os.path.basename(episode["scene_id"]))[0] if episode is not None else "studio",
                     "config_overrides": ({"cam_z": args.cam_z} if args.cam_z is not None else {}) if args.model == "as_evaluated"
                     else {"robot": "configs/k1_corrected_robot_cfg.py"},
                     "cam_z": args.cam_z, "render_mode": "RTX PathTracing via Replicator render product + rgb annotator",
                     "samples_per_pixel": args.total_spp, "spp_per_frame": args.spp_per_frame, "frames_rendered": n_frames,
                     "denoiser": "OptiX (/rtx/pathtracing/optixDenoiser/enabled=True)", "rtx_settings_read_back": settings_read,
                     "resolution": [v["width"], v["height"]], "camera": v,
                     "mean_abs_change_after_3_more_frames": conv,
                     "trunk_pose": {"pos": tt.tolist(), "quat_wxyz": tq.tolist()}})
        json.dump(meta, open(f"{out}/{fn}.json", "w"), indent=1)
        print(f"[pt] {fn}: mean {img.mean():.1f}  conv-delta {conv:.3f}")
    json.dump(views, open(f"{out}/views.json", "w"), indent=1)
    if args.model != "studio":
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001  (Kit keeps the process alive after an uncaught exception)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    simulation_app.close()
