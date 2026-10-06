"""Step 3 probe v2 (GPU, short): reproduce and explain the gray robot-camera frames.

Builds the environment exactly as the evaluator does (sim_setup.build_env = verbatim evaluator lines:
task k1_matterport_vision, reset_start_pos_rot, --cam_z override, VLNEnvWrapperV3 with its 1-s reset
warm-up, model_14498.pt; no model server), then:

  A. reset frames (robot camera + chase camera, lossless, + robot-camera depth) as the evaluator captures them
  B. poses: Trunk (Isaac Lab, Fabric, USD), robot camera, every Trunk visual mesh (parent, local transform,
     USD world, Trunk(now) x local); floor height under the Trunk by a warp ray cast onto the Matterport mesh
  D. walk: constant forward command (default 0.5 m/s) through the walking policy; frames + full Trunk pose every
     25 steps; heights every step (2 s = 100 steps is the specified test; --walk_steps may extend it)
  E. articulation masses / inertias / COMs (PhysX tensor view)
  F. back to the start pose (env.reset(), same 1-s warm-up), re-render without stepping physics; optional runtime
     deletion of mesh specs in the flattened prototype (skipped with --skip_removal; the removal test used for the
     report is --hide_meshes, which authors visibility in a copy of the converted USD BEFORE load)
  --physics_cpu (diagnostic only, NOT as evaluated): CPU PhysX pipeline so that PhysX scene queries see the
     articulation; vertical ray casts through the head/hand collision shapes at the current Trunk pose and at the
     spawn pose, after reset and after the walk.

Writes only under --out_dir. Never writes eval_results/.
"""
import functools
import glob
import json
import math
import os
import sys

print = functools.partial(print, flush=True)  # noqa: A001  (Kit fast-shutdown drops unflushed stdout)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_setup  # noqa: E402


def extra(p):
    p.add_argument("--out_dir", required=True)
    p.add_argument("--walk_steps", type=int, default=100)
    p.add_argument("--walk_vx", type=float, default=0.5)
    p.add_argument("--physics_cpu", action="store_true", default=False)
    p.add_argument("--skip_removal", action="store_true", default=False)
    p.add_argument("--hide_meshes", type=str, default="",
                   help="REMOVAL TEST (not as evaluated): comma list of Trunk visual meshes (e.g. mesh_2,...,mesh_11) made "
                        "invisible in a copy of the converted USD before load; collisions, masses and everything else unchanged")
    p.add_argument("--work_dir", type=str, default="/tmp/claude-1000/airc2027_work")
    p.add_argument("--deinstance_trunk_visuals", action="store_true", default=False,
                   help="CAUSAL TEST (not as evaluated): set instanceable=False on Trunk/visuals in a copy of the converted USD before load")
    p.add_argument("--bind_material_meshes", type=str, default="",
                   help="CAUSAL TEST (not as evaluated): comma list of Trunk visual meshes that get mesh_0's exact material "
                        "binding (relationship + MaterialBindingAPI) in a copy of the converted USD before load")


args, simulation_app = sim_setup.launch("step3 probe v2", extra)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

GRAY = lambda a: float((np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6).mean())  # noqa: E731
MESH_LINK = {"mesh_0": "Trunk (Trunk.STL)", "mesh_1": "Trunk (K1logo.STL)", "mesh_2": "Head_1", "mesh_3": "Head_2",
             "mesh_4": "Left_Arm_1", "mesh_5": "Left_Arm_2", "mesh_6": "Left_Arm_3", "mesh_7": "left_hand_link",
             "mesh_8": "Right_Arm_1", "mesh_9": "Right_Arm_2", "mesh_10": "Right_Arm_3", "mesh_11": "right_hand_link"}
ARMS = [f"mesh_{i}" for i in range(4, 12)]
HEAD = ["mesh_2", "mesh_3"]
ROOT = "/World/envs/env_0/Robot"


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def save_png(arr, path):
    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def cpu_hook(cfg):
    cfg.sim.device = "cpu"
    cfg.sim.use_gpu_pipeline = False
    cfg.sim.physx.use_gpu = False
    cfg.sim.enable_scene_query_support = True  # Isaac Lab default False -> PhysX builds no scene-query structures


HIDE_INFO = {}


def hide_hook(cfg):
    """Convert the URDF exactly as the spawner would (same UrdfFileCfg), copy the output, author
    visibility=invisible on the chosen meshes in the copy's instanceable-mesh layer, and spawn that copy with the
    identical file-spawner settings (UsdFileCfg). Only visibility of the chosen visual meshes differs."""
    import shutil

    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.sim.converters import UrdfConverter
    from pxr import Sdf
    old = cfg.scene.robot.spawn
    conv = UrdfConverter(old)
    dst = os.path.join(args.work_dir, f"k1_hidden_{os.getpid()}")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(conv.usd_dir, dst)
    # The converter output K1_locomotion.usd is a FLATTENED stage: the instanceable 'visuals' / 'collisions' prims
    # point at in-file prototypes /Flattened_Prototype_<n> (Props/instanceable_meshes.usd is not referenced).
    # The Trunk-visuals prototype is the one holding mesh_0..mesh_11 with a Looks scope.
    lay = Sdf.Layer.FindOrOpen(os.path.join(dst, conv.usd_file_name))
    names = [n for n in args.hide_meshes.split(",") if n]
    proto = [r for r in lay.rootPrims if r.name.startswith("Flattened_Prototype")
             and "mesh_11" in r.nameChildren and "Looks" in r.nameChildren
             and r.nameChildren["mesh_11"].typeName == "Mesh"]
    assert len(proto) == 1, [r.name for r in proto]
    proto = proto[0]
    for n in names:
        ps = proto.nameChildren[n]
        a = Sdf.AttributeSpec(ps, "visibility", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
        a.default = "invisible"
    bind = [n for n in args.bind_material_meshes.split(",") if n]
    if bind:
        src = proto.nameChildren["mesh_0"]
        srel = src.relationships["material:binding"]
        targets = list(srel.targetPathList.GetAddedOrExplicitItems())
        schemas = src.GetInfo("apiSchemas")
        for n in bind:
            ps = proto.nameChildren[n]
            r = Sdf.RelationshipSpec(ps, "material:binding", custom=False)
            for t in targets:
                r.targetPathList.Prepend(t)
            ps.SetInfo("apiSchemas", schemas)
        HIDE_INFO["material_binding_copied_from_mesh_0"] = {"targets": [str(t) for t in targets],
                                                            "apiSchemas": str(schemas), "meshes": bind}
    if args.deinstance_trunk_visuals:
        vs = lay.GetPrimAtPath("/K1/Trunk/visuals")
        HIDE_INFO["trunk_visuals_spec_before"] = {"instanceable": vs.instanceable if vs.HasInfo("instanceable") else None,
                                                  "references": [str(r) for r in vs.referenceList.GetAddedOrExplicitItems()],
                                                  "inherits": [str(r) for r in vs.inheritPathList.GetAddedOrExplicitItems()]}
        vs.instanceable = False
        HIDE_INFO["deinstanced"] = "/K1/Trunk/visuals"
    lay.Save()
    HIDE_INFO["edited_prototype"] = str(proto.path)
    kw = {}
    for f in sim_utils.UsdFileCfg.__dataclass_fields__:
        if f not in ("usd_path", "func") and hasattr(old, f):
            kw[f] = getattr(old, f)
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=os.path.join(dst, conv.usd_file_name), **kw)
    HIDE_INFO.update({"hidden_meshes": names, "source_converter_dir": conv.usd_dir, "copy": dst,
                      "spawn_kwargs_copied": sorted(kw)})
    print("[hide] spawning", cfg.scene.robot.spawn.usd_path, "with hidden", names)


def main():
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    hooks = [h for h, on in ((cpu_hook, args.physics_cpu),
                              (hide_hook, bool(args.hide_meshes) or bool(args.bind_material_meshes) or args.deinstance_trunk_visuals)) if on]
    env, episode, env_cfg, obs, infos = sim_setup.build_env(
        args, env_cfg_hook=(lambda c: [h(c) for h in hooks]) if hooks else None)
    import omni.usd
    import usdrt
    from omni.isaac.lab.sensors import RayCaster
    from omni.isaac.lab.utils.warp import raycast_mesh
    from pxr import Gf, Sdf, Usd, UsdGeom

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    rcam = uenv.scene["rgb_camera"]
    vcam = uenv.scene["viz_rgb_camera"]
    stage = omni.usd.get_context().get_stage()
    rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    trunk_idx = robot.body_names.index("Trunk")
    start = np.array(episode["start_position"], dtype=float)
    cam_off = np.array(env_cfg.scene.rgb_camera.offset.pos, dtype=float)
    R = {"script": "sim_step3.py v2", "episode_idx": args.episode_idx, "record": int(episode["episode_id"]) - 1,
         "scene": os.path.basename(episode["scene_id"]), "cam_z": args.cam_z, "physics_cpu": args.physics_cpu,
         "start_position": start.tolist(), "start_rotation": episode["start_rotation"],
         "spawn_pos_cfg": list(env_cfg.scene.robot.init_state.pos), "robot_camera_offset_trunk_frame": cam_off.tolist(),
         "robot_camera_rot_cfg(ros)": list(env_cfg.scene.rgb_camera.offset.rot),
         "chase_camera_offset_trunk_frame": list(env_cfg.scene.viz_rgb_camera.offset.pos),
         "converter_dir": sorted(glob.glob("/tmp/IsaacLab/usd_*"), key=os.path.getmtime)[-1],
         "removal_test_hidden_meshes": HIDE_INFO or None}

    def trunk():
        return (robot.data.body_pos_w[0, trunk_idx].cpu().numpy().astype(float),
                robot.data.body_quat_w[0, trunk_idx].cpu().numpy().astype(float))

    def depth_stats(dep, rgb):
        g = (np.abs(rgb.astype(np.int16) - 53).max(axis=2) <= 6)
        if not g.any():
            return {"gray_px": 0}
        v = dep[g]
        v = v[np.isfinite(v)]
        return {"gray_px": int(g.sum()), "depth_median": float(np.median(v)), "depth_p5": float(np.percentile(v, 5)),
                "depth_p95": float(np.percentile(v, 95))}

    def grab(tag, from_infos=None):
        if from_infos is not None:
            rgb = from_infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            viz = from_infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
        else:
            rgb = rcam.data.output["rgb"][0, :, :, :3].cpu().numpy()
            viz = vcam.data.output["rgb"][0, :, :, :3].cpu().numpy()
        dep = rcam.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy()
        save_png(rgb, f"{out}/{tag}_robotcam.png")
        save_png(viz, f"{out}/{tag}_chasecam.png")
        np.save(f"{out}/{tag}_robotcam_depth.npy", dep.astype(np.float32))
        tt, tq = trunk()
        rec = {"robotcam_gray": GRAY(rgb), "chasecam_gray": GRAY(viz), "robotcam_depth_in_gray": depth_stats(dep, rgb),
               "trunk_pos": tt.tolist(), "trunk_quat_wxyz": tq.tolist()}
        R.setdefault("frames", {})[tag] = rec
        print(f"[frame {tag}] robot gray {rec['robotcam_gray']:.4f} chase gray {rec['chasecam_gray']:.4f} depth {rec['robotcam_depth_in_gray']}")
        return rgb

    # ---------- A ----------
    rgb0 = grab("A_reset", infos)

    # ---------- B ----------
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def usd_world(path):
        m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        return np.array([t[0], t[1], t[2]]), np.array([q.GetReal(), *q.GetImaginary()])

    meshes = {}
    for i in range(12):
        n = f"mesh_{i}"
        p = stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals/{n}")
        ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(p).GetOrderedXformOps()}
        t = ops.get("xformOp:translate", Gf.Vec3d(0, 0, 0))
        q = ops.get("xformOp:orient", Gf.Quatd(1, 0, 0, 0))
        pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
        meshes[n] = {"local_t": [t[0], t[1], t[2]], "local_q": [q.GetReal(), *q.GetImaginary()],
                     "bbox_local_min": pts.min(0).tolist(), "bbox_local_max": pts.max(0).tolist(),
                     "is_instance_proxy": p.IsInstanceProxy(), "parent": str(p.GetParent().GetPath()),
                     "parent_instanceable": p.GetParent().IsInstanceable()}
    spawn_t, spawn_q = usd_world(f"{ROOT}/Trunk")
    R["spawn_pose_usd"] = {"pos": spawn_t.tolist(), "quat_wxyz": spawn_q.tolist()}
    R["meshes"] = meshes

    def fabric_world(path):
        p = rt.GetPrimAtPath(path)
        d = {}
        for n in ("_worldPosition", "_worldOrientation"):
            a = p.GetAttribute(n) if p and p.IsValid() else None
            d[n] = str(a.Get()) if a and a.IsValid() and a.HasValue() else None
        return d

    rc_mesh = RayCaster.meshes.get("/World/matterport")

    def floor_z(xy, z_from):
        if rc_mesh is None:
            return None
        dev = rc_mesh.device
        s = torch.tensor([[xy[0], xy[1], z_from]], dtype=torch.float32, device=str(dev))
        d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=str(dev))
        hit = raycast_mesh(s, d, rc_mesh, max_dist=5.0)[0][0].cpu().numpy()
        return float(hit[2]) if np.isfinite(hit[2]) else None

    def poses(tag):
        tt, tq = trunk()
        Rt = qrot(tq)
        fz = floor_z(tt[:2], tt[2])
        camw = tt + Rt @ cam_off
        d = {"trunk_isaaclab": {"pos": tt.tolist(), "quat_wxyz": tq.tolist(), "yaw_deg": yaw_of(tq)},
             "trunk_fabric": fabric_world(f"{ROOT}/Trunk"),
             "trunk_usd": {"pos": spawn_t.tolist(), "quat_wxyz": spawn_q.tolist()},
             "floor_z_below_trunk": fz,
             "trunk_height_above_episode_start_z": float(tt[2] - start[2]),
             "trunk_height_above_floor": None if fz is None else float(tt[2] - fz),
             "robot_camera": {"isaaclab_data_pos_w": rcam.data.pos_w[0].cpu().numpy().astype(float).tolist(),
                              "trunk_x_offset_pos_w": camw.tolist(),
                              "usd_world_pos": usd_world(f"{ROOT}/Trunk/rgb_camera")[0].tolist(),
                              "height_above_episode_start_z": float(camw[2] - start[2]),
                              "height_above_floor": None if fz is None else float(camw[2] - fz)},
             "meshes": {}}
        for n, m in meshes.items():
            lt = np.array(m["local_t"])
            d["meshes"][n] = {"link": MESH_LINK[n], "usd_world_pos": usd_world(f"{ROOT}/Trunk/visuals/{n}")[0].tolist(),
                              "trunk_now_x_local_pos": (tt + Rt @ lt).tolist(),
                              "fabric_world": fabric_world(f"{ROOT}/Trunk/visuals/{n}")}
        R.setdefault("poses", {})[tag] = d
        print(f"[poses {tag}] trunk {np.round(tt, 4)} yaw {yaw_of(tq):.2f}  floor {fz}  trunk-floor "
              f"{d['trunk_height_above_floor']}  cam-floor {d['robot_camera']['height_above_floor']}  "
              f"cam-start {d['robot_camera']['height_above_episode_start_z']:.4f}")

    poses("B_after_reset")

    # ---------- E ----------
    pv = robot.root_physx_view
    masses = pv.get_masses()[0].cpu().numpy().astype(float)
    inert = pv.get_inertias()[0].cpu().numpy().astype(float).reshape(-1, 3, 3)
    coms = pv.get_coms()[0].cpu().numpy().astype(float)
    R["E_physics"] = {"body_names": list(robot.body_names), "masses": masses.tolist(), "total_mass": float(masses.sum()),
                      "trunk_mass": float(masses[trunk_idx]), "trunk_inertia_3x3": inert[trunk_idx].tolist(),
                      "trunk_com_pose_pos_quat": coms[trunk_idx].tolist(), "all_inertias": inert.tolist(),
                      "all_coms": coms.tolist(), "joint_names": list(robot.joint_names)}
    print("[E] total mass", masses.sum(), "trunk", masses[trunk_idx])

    sq = None
    if args.physics_cpu:
        from omni.physx import get_physx_scene_query_interface
        sq = get_physx_scene_query_interface()

    def ray_probe(tag):
        if sq is None:
            return
        tt, tq = trunk()
        res = {}
        hits_all = []

        def cb(h):
            hits_all.append((float(getattr(h, 'distance', -1)), str(getattr(h, 'collision', '')), str(getattr(h, 'rigid_body', getattr(h, 'rigidBody', '')))))
            return True
        for n in ("mesh_3", "mesh_7", "mesh_11"):
            m = meshes[n]
            lo, hi = np.array(m["bbox_local_min"]), np.array(m["bbox_local_max"])
            centre_local = np.array(m["local_t"]) + qrot(m["local_q"]) @ ((lo + hi) / 2)
            for hyp, (ht, hq) in {"current_trunk": (tt, tq), "spawn_pose": (spawn_t, spawn_q)}.items():
                c = ht + qrot(hq) @ centre_local
                o = c + np.array([0, 0, 0.30])
                hits_all.clear()
                sq.raycast_all(tuple(o.tolist()), (0.0, 0.0, -1.0), 0.45, cb)
                rb = sorted(hits_all)
                res[f"{n}({MESH_LINK[n]})@{hyp}"] = {"ray_origin": o.tolist(), "ray_length": 0.45,
                                                    "hits(distance,collision,rigid_body)": rb,
                                                    "expected_top_z": float(c[2] + (hi[2] - lo[2]) / 2)}
                print(f"[ray {tag}] {n}@{hyp}: {[(round(a, 3), b.replace(ROOT, 'R'), c2.replace(ROOT, 'R')) for a, b, c2 in rb]}")
        # sanity: does the scene query see the floor below the trunk?
        hits_all.clear()
        sq.raycast_all(tuple((tt + np.array([0.3, 0.3, 0])).tolist()), (0.0, 0.0, -1.0), 3.0, cb)
        res["sanity_floor_ray_0.3m_beside_trunk"] = sorted(hits_all)
        R.setdefault("E_raycasts_cpu_physics", {})[tag] = res

    ray_probe("after_reset")

    # ---------- D ----------
    cmd = torch.tensor([args.walk_vx, 0.0, 0.0], device=obs.device)
    traj = []
    for i in range(args.walk_steps):
        obs, _, done, infos = env.step(cmd)
        tt, tq = trunk()
        camw = tt + qrot(tq) @ cam_off
        traj.append([i + 1, *tt.tolist(), *tq.tolist(), yaw_of(tq), camw[2], float(tt[2] - start[2]), float(camw[2] - start[2])])
        if (i + 1) % 25 == 0:
            grab(f"D_walk_step{i + 1:03d}", infos)
        if (i + 1) == 100:
            poses("D_after_walk_2s")
            ray_probe("after_walk_2s")
        if bool(done):
            R["D_walk_done_at"] = i + 1
            print("[D] env reported done at", i + 1)
            break
    tr = np.array(traj)
    if len(tr) == 0:
        tr = np.zeros((0, 12))
    np.savetxt(f"{out}/D_walk_trajectory.csv", tr, delimiter=",",
               header="step,trunk_x,trunk_y,trunk_z,qw,qx,qy,qz,yaw_deg,camera_z,trunk_z_minus_start_z,camera_z_minus_start_z", comments="")
    t0 = np.array(R["poses"]["B_after_reset"]["trunk_isaaclab"]["pos"])
    fz0 = R["poses"]["B_after_reset"]["floor_z_below_trunk"]
    w2 = tr[:100]
    R["D_walk_heights_2s"] = None if len(w2) == 0 else {"trunk_z_minus_start_z": {"mean": float(w2[:, 10].mean()), "min": float(w2[:, 10].min()), "max": float(w2[:, 10].max())},
                              "camera_z_minus_start_z": {"mean": float(w2[:, 11].mean()), "min": float(w2[:, 11].min()), "max": float(w2[:, 11].max())},
                              "trunk_z_minus_floor_at_start": None if fz0 is None else {"mean": float((w2[:, 3] - fz0).mean()), "min": float((w2[:, 3] - fz0).min()), "max": float((w2[:, 3] - fz0).max())},
                              "camera_z_minus_floor_at_start": None if fz0 is None else {"mean": float((w2[:, 9] - fz0).mean()), "min": float((w2[:, 9] - fz0).min()), "max": float((w2[:, 9] - fz0).max())},
                              "net_displacement_xy_m": float(np.linalg.norm(w2[-1, 1:3] - t0[:2]))}
    if len(tr) > 100:
        R["D_walk_extra"] = {"steps": len(tr), "net_displacement_xy_m": float(np.linalg.norm(tr[-1, 1:3] - t0[:2]))}
        poses(f"D_after_walk_{len(tr)}steps")
        ray_probe(f"after_walk_{len(tr)}steps")
    print("[D] 2-s walk:", R["D_walk_heights_2s"])

    # ---------- F ----------
    obs, infos = env.reset()
    rgbF = grab("F0_rereset", infos)
    d = np.abs(rgbF.astype(np.int16) - rgb0.astype(np.int16))
    R["F0_rereset_vs_first_reset"] = {"max_abs_diff": int(d.max()), "mean_abs_diff": float(d.mean()),
                                     "share_channels_differ": float((d > 0).mean())}
    poses("F0_after_rereset")
    print("[F0]", R["F0_rereset_vs_first_reset"])

    def rerender(tag):
        for _ in range(3):
            uenv.sim.render()
        rcam.update(0.0, force_recompute=True)
        vcam.update(0.0, force_recompute=True)
        return grab(tag)

    if not args.skip_removal:
        rerender("F1_rerender_no_change")
        layer = Sdf.Layer.FindOrOpen(os.path.join(R["converter_dir"], "K1_locomotion.usd"))  # flattened file
        R["F_removal_layer"] = layer.identifier
        proto_spec = [r for r in layer.rootPrims if r.name.startswith("Flattened_Prototype")
                      and "mesh_11" in r.nameChildren and "Looks" in r.nameChildren][0]

        def delete_specs(names):
            parent = proto_spec
            done_ = []
            for n in names:
                if n in parent.nameChildren:
                    del parent.nameChildren[n]
                    done_.append(n)
            return done_
        R["F2_deleted"] = delete_specs(ARMS)
        rerender("F2_arm_meshes_deleted")
        R["F3_deleted"] = delete_specs(HEAD)
        rerender("F3_head_and_arm_meshes_deleted")
        R["F_stage_prims_after_delete"] = [str(p.GetPath()).replace(ROOT, "R") for p in
                                           Usd.PrimRange(stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals"), Usd.TraverseInstanceProxies())
                                           if p.GetTypeName() == "Mesh"]

    json.dump(R, open(f"{out}/step3_results.json", "w"), indent=1, default=str)
    print("[done] wrote", f"{out}/step3_results.json")
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
