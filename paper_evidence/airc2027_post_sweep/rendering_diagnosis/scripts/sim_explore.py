"""Exploratory probe (GPU, short): build the environment exactly as the evaluator does, reset, then dump
the robot prim hierarchy (USD, incl. instance proxies), the URDF-converter output folder, and the Fabric
attributes of the robot prims. Saves the reset robot-camera / chase-camera frames. No stepping.

Run (from ~/Projects/k1_research/NaVILA-Bench):
  python ../airc2027_renders/scripts/sim_explore.py --task=k1_matterport_vision --num_envs=1 \
     --checkpoint=$KR/checkpoints/model_14498.pt --episode_idx=0 --gait_phase_init=0.0 --cam_z=0.57 \
     --headless --enable_cameras --out_dir <dir>
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_setup  # noqa: E402


def extra(p):
    p.add_argument("--out_dir", required=True)


args, simulation_app = sim_setup.launch("explore", extra)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    env, episode, env_cfg, obs, infos = sim_setup.build_env(args)
    from pxr import Usd, UsdGeom, UsdPhysics

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = "/World/envs/env_0/Robot"
    lines = []
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        p = prim.GetPath().pathString
        apis = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): apis.append("RigidBody")
        if prim.HasAPI(UsdPhysics.CollisionAPI): apis.append("Collision")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI): apis.append("ArtRoot")
        if prim.HasAPI(UsdPhysics.MassAPI): apis.append("Mass")
        xf = UsdGeom.Xformable(prim)
        ops = []
        if xf:
            for op in xf.GetOrderedXformOps():
                v = op.Get()
                ops.append(f"{op.GetOpName()}={v}")
        wt = ""
        if xf:
            m = xc.GetLocalToWorldTransform(prim)
            t = m.ExtractTranslation()
            wt = f"usd_world_t=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})"
        vis = ""
        im = UsdGeom.Imageable(prim)
        if im and im.GetVisibilityAttr().HasAuthoredValue():
            vis = f"vis={im.GetVisibilityAttr().Get()}"
        purpose = ""
        if im and im.GetPurposeAttr().HasAuthoredValue():
            purpose = f"purpose={im.GetPurposeAttr().Get()}"
        lines.append(f"{p} | {prim.GetTypeName()} | inst={prim.IsInstanceable()} proxy={prim.IsInstanceProxy()} | "
                     f"{','.join(apis)} | {' '.join(ops)} | {wt} {vis} {purpose}")
    open(os.path.join(args.out_dir, "usd_hierarchy_after_reset.txt"), "w").write("\n".join(lines) + "\n")
    print(f"[explore] {len(lines)} prims under {root}")

    # Converter output dir (Isaac Lab 1.1: /tmp/IsaacLab/usd_<time>_<rand>)
    dirs = sorted(glob.glob("/tmp/IsaacLab/usd_*"), key=os.path.getmtime)
    info = {"converter_dirs": dirs}
    for d in dirs:
        info[d] = sorted(os.path.relpath(f, d) for f in glob.glob(d + "/**", recursive=True))
    json.dump(info, open(os.path.join(args.out_dir, "converter_dirs.json"), "w"), indent=1)
    print("[explore] converter dirs:", dirs)

    # Fabric attributes
    try:
        import usdrt
        rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        fab = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            p = prim.GetPath().pathString
            rp = rt.GetPrimAtPath(p)
            if not rp or not rp.IsValid():
                fab[p] = "NOT IN FABRIC"
                continue
            d = {}
            for a in rp.GetAttributes():
                n = a.GetName()
                if any(k in n for k in ("world", "World", "xformOp", "local", "Local", "visib")):
                    try:
                        d[n] = str(a.Get())
                    except Exception as e:  # noqa: BLE001
                        d[n] = f"<{e}>"
            fab[p] = d
        json.dump(fab, open(os.path.join(args.out_dir, "fabric_attrs_after_reset.json"), "w"), indent=1)
        print("[explore] fabric prims:", len(fab))
    except Exception as e:  # noqa: BLE001
        print("[explore] usdrt failed:", repr(e))

    rgb = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
    viz = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    Image.fromarray(rgb).save(os.path.join(args.out_dir, "reset_robotcam.png"))
    Image.fromarray(viz).save(os.path.join(args.out_dir, "reset_chasecam.png"))
    g = (np.abs(rgb.astype(np.int16) - 53).max(axis=2) <= 6).mean()
    print(f"[explore] reset robot-cam gray fraction {g:.4f}")
    robot = env.unwrapped.scene["robot"]
    print("[explore] body_names:", robot.body_names)
    print("[explore] joint_names:", robot.joint_names)
    print("[explore] root_pos_w:", robot.data.root_pos_w[0].tolist(), "start:", episode["start_position"])
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
