"""[v1 — superseded by sim_step3.py v2; kept because its outputs (diagnosis/sim/step3_v1_cz057) hold the de-instancing/culling result] Step 3 probe (GPU, short): reproduce and explain the gray robot-camera frames.

Builds the environment exactly as the evaluator does (sim_setup.build_env = verbatim evaluator lines:
task k1_matterport_vision, reset_start_pos_rot, --cam_z override, VLNEnvWrapperV3 with its 1-s reset
warm-up, model_14498.pt; no model server), then:

  A. reset frames (robot camera + chase camera, lossless) exactly as the evaluator captures them
  B. poses: Trunk (Isaac Lab + Fabric), robot camera, and every Trunk visual mesh (local offset, USD
     world, and Trunk(now) x local)
  C. a fixed diagnostic camera (Replicator render product: rgb + distance_to_image_plane +
     instance_id_segmentation_fast), rendered with the Matterport mesh hidden, to locate in 3-D where the
     renderer actually draws every robot mesh; each rendered point is tested against each mesh's local
     bounding box under two hypotheses: attached to the CURRENT Trunk pose, or at the SPAWN (USD) pose
  D. walk: constant forward command 0.5 m/s for 2 s (100 control steps) through the walking policy;
     frames + poses every 25 steps; trunk/camera heights every step; C repeated at the end
  E. physics: articulation masses / inertias / COMs vs URDF; PhysX ray casts through the head and hands
     at the current pose and at the spawn pose (do the merged collision shapes move with the Trunk?)
  F. back to the start pose (env.reset(), the same 1-s warm-up), then the removal test: hide the head and
     arm meshes and re-render without stepping physics (method A: edit visibility in the converter's
     instanceable-mesh layer; method B: de-instance Trunk/visuals and hide the mesh prims)

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


args, simulation_app = sim_setup.launch("step3 probe", extra)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

GRAY = lambda a: float((np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6).mean())  # noqa: E731

# URDF link each merged Trunk visual came from (matched by the composed fixed-joint offsets; see Step 2)
MESH_LINK = {"mesh_0": "Trunk (Trunk.STL)", "mesh_1": "Trunk (K1logo.STL)", "mesh_2": "Head_1", "mesh_3": "Head_2",
             "mesh_4": "Left_Arm_1", "mesh_5": "Left_Arm_2", "mesh_6": "Left_Arm_3", "mesh_7": "left_hand_link",
             "mesh_8": "Right_Arm_1", "mesh_9": "Right_Arm_2", "mesh_10": "Right_Arm_3", "mesh_11": "right_hand_link"}
HEAD_ARM = [f"mesh_{i}" for i in range(2, 12)]
ROOT = "/World/envs/env_0/Robot"


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


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


def main():
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    env, episode, env_cfg, obs, infos = sim_setup.build_env(args)
    import omni.usd
    import usdrt
    from pxr import Gf, Sdf, Usd, UsdGeom

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    rcam = uenv.scene["rgb_camera"]
    vcam = uenv.scene["viz_rgb_camera"]
    stage = omni.usd.get_context().get_stage()
    rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    trunk_idx = robot.body_names.index("Trunk")
    start = np.array(episode["start_position"], dtype=float)
    R = {"episode_idx": args.episode_idx, "record": int(episode["episode_id"]) - 1,
         "scene": os.path.basename(episode["scene_id"]), "cam_z": args.cam_z, "start_position": start.tolist(),
         "start_rotation": episode["start_rotation"], "spawn_pos_cfg": list(env_cfg.scene.robot.init_state.pos)}

    # ---------- A. reset frames, exactly as the evaluator captures them ----------
    rgb0 = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
    viz0 = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    save_png(rgb0, f"{out}/A_reset_robotcam.png")
    save_png(viz0, f"{out}/A_reset_chasecam.png")
    dep0 = rcam.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy()
    g0 = (np.abs(rgb0.astype(np.int16) - 53).max(axis=2) <= 6)
    R["A_reset"] = {"robotcam_gray_frac": GRAY(rgb0), "chasecam_gray_frac": GRAY(viz0),
                    "robotcam_depth_m_in_gray_region_median": float(np.median(dep0[g0])) if g0.any() else None,
                    "robotcam_depth_m_in_gray_region_p5_p95": [float(np.percentile(dep0[g0], 5)), float(np.percentile(dep0[g0], 95))] if g0.any() else None,
                    "robotcam_depth_m_overall_median": float(np.nanmedian(dep0[np.isfinite(dep0)]))}
    np.save(f"{out}/A_reset_robotcam_depth.npy", dep0.astype(np.float32))
    print("[A] reset gray", R["A_reset"])

    # ---------- B. poses ----------
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    def usd_world(path):
        m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()
        return np.array([t[0], t[1], t[2]]), np.array([q.GetReal(), *q.GetImaginary()])

    def mesh_local(name):
        p = stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals/{name}")
        ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(p).GetOrderedXformOps()}
        t = ops.get("xformOp:translate", Gf.Vec3d(0, 0, 0))
        q = ops.get("xformOp:orient", Gf.Quatd(1, 0, 0, 0))
        pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
        return np.array([t[0], t[1], t[2]]), np.array([q.GetReal(), *q.GetImaginary()]), pts.min(0), pts.max(0)

    meshes = {f"mesh_{i}": mesh_local(f"mesh_{i}") for i in range(12)}
    spawn_t, spawn_q = usd_world(f"{ROOT}/Trunk")  # USD (not updated by physics under Fabric) = spawn pose

    def fabric_world(path):
        p = rt.GetPrimAtPath(path)
        d = {}
        for n in ("_worldPosition", "_worldOrientation"):
            a = p.GetAttribute(n) if p and p.IsValid() else None
            d[n] = str(a.Get()) if a and a.IsValid() and a.HasValue() else None
        return d

    def poses(tag):
        tt = robot.data.body_pos_w[0, trunk_idx].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, trunk_idx].cpu().numpy().astype(float)
        Rt = qrot(tq)
        cam_off = np.array(env_cfg.scene.rgb_camera.offset.pos)
        d = {"trunk_isaaclab": {"pos": tt.tolist(), "quat_wxyz": tq.tolist(), "yaw_deg": yaw_of(tq)},
             "trunk_root_state": robot.data.root_state_w[0, :7].cpu().numpy().astype(float).tolist(),
             "trunk_fabric": fabric_world(f"{ROOT}/Trunk"),
             "trunk_usd(spawn)": {"pos": spawn_t.tolist(), "quat_wxyz": spawn_q.tolist()},
             "trunk_height_above_episode_start_z": float(tt[2] - start[2]),
             "robot_camera": {"isaaclab_data_pos_w": rcam.data.pos_w[0].cpu().numpy().astype(float).tolist(),
                              "trunk_x_offset_pos_w": (tt + Rt @ cam_off).tolist(),
                              "usd_world": usd_world(f"{ROOT}/Trunk/rgb_camera")[0].tolist(),
                              "height_above_episode_start_z": float((tt + Rt @ cam_off)[2] - start[2])},
             "meshes": {}}
        hs = uenv.scene["height_scanner"].data.ray_hits_w[0].cpu().numpy()
        k = np.argmin(np.linalg.norm(hs[:, :2] - tt[:2], axis=1))
        d["floor_z_below_trunk_raycast"] = float(hs[k, 2])
        d["trunk_height_above_floor"] = float(tt[2] - hs[k, 2])
        d["robot_camera"]["height_above_floor"] = float((tt + Rt @ cam_off)[2] - hs[k, 2])
        for n, (lt, lq, _, _) in meshes.items():
            uw = usd_world(f"{ROOT}/Trunk/visuals/{n}")[0]
            d["meshes"][n] = {"link": MESH_LINK[n], "parent_prim": f"{ROOT}/Trunk/visuals (instanceable)",
                              "local_translate": lt.tolist(), "local_orient_wxyz": lq.tolist(),
                              "usd_world_pos": uw.tolist(), "trunk_now_x_local_pos": (tt + Rt @ lt).tolist(),
                              "fabric": fabric_world(f"{ROOT}/Trunk/visuals/{n}")}
        R.setdefault("poses", {})[tag] = d
        return tt, tq

    tt0, tq0 = poses("B_after_reset")
    print("[B] trunk", R["poses"]["B_after_reset"]["trunk_isaaclab"], "spawn", spawn_t,
          "cam", R["poses"]["B_after_reset"]["robot_camera"])

    # ---------- C. diagnostic camera ----------
    import omni.replicator.core as rep

    terrain = stage.GetPrimAtPath("/World/matterport")
    fwd = qrot(tq0)[:, 0]
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0], 0.0])
    W, H, FOC, HA = 1600, 900, 18.0, 36.0
    diag = {}

    def make_cam(name, eye, target):
        cam = UsdGeom.Camera.Define(stage, f"/World/Diag_{name}")
        cam.GetFocalLengthAttr().Set(FOC)
        cam.GetHorizontalApertureAttr().Set(HA)
        cam.GetVerticalApertureAttr().Set(HA * H / W)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        m = Gf.Matrix4d()
        m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        c2w = m.GetInverse()
        xf = UsdGeom.Xformable(cam)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(c2w)
        rp = rep.create.render_product(f"/World/Diag_{name}", (W, H))
        anns = {}
        for an, ip in (("rgb", None), ("distance_to_image_plane", None),
                       ("instance_id_segmentation_fast", {"colorize": False})):
            a = rep.AnnotatorRegistry.get_annotator(an, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(an)
            a.attach([rp])
            anns[an] = a
        diag[name] = {"c2w": np.array(c2w), "anns": anns, "eye": list(eye), "target": list(target)}

    mid = tt0 + fwd * 0.5 + np.array([0, 0, 0.15])
    make_cam("side", tt0 + fwd * 0.5 + left * 2.6 + np.array([0, 0, 0.35]), mid)
    make_cam("rear", tt0 - fwd * 2.2 + left * 0.6 + np.array([0, 0, 0.9]), mid)

    def diag_capture(tag):
        UsdGeom.Imageable(terrain).MakeInvisible()
        for _ in range(4):
            uenv.sim.render()
        tt = robot.data.body_pos_w[0, trunk_idx].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, trunk_idx].cpu().numpy().astype(float)
        res = {}
        for name, dc in diag.items():
            rgb = np.array(dc["anns"]["rgb"].get_data())[:, :, :3]
            dep = np.array(dc["anns"]["distance_to_image_plane"].get_data()).squeeze()
            seg = dc["anns"]["instance_id_segmentation_fast"].get_data()
            ids, id2lab = np.array(seg["data"]).squeeze(), seg["info"]["idToLabels"]
            save_png(rgb, f"{out}/C_{tag}_diag_{name}.png")
            fx = W * FOC / HA
            vv, uu = np.mgrid[0:H, 0:W]
            ok = np.isfinite(dep) & (dep > 0) & (dep < 50)
            x = (uu + 0.5 - W / 2) * dep / fx
            y = -(vv + 0.5 - H / 2) * dep / fx
            P = np.stack([x, y, -dep, np.ones_like(dep)], -1)[ok] @ dc["c2w"]
            labs = ids[ok]
            per = {}
            for idv, lab in id2lab.items():
                lab = str(lab)
                if "Robot" not in lab:
                    continue
                pts = P[labs == int(idv)][:, :3]
                if len(pts) == 0:
                    continue
                # which mesh bbox explains each point, under the CURRENT trunk pose vs the SPAWN pose
                fits = {}
                for hyp, (ht, hq) in {"current": (tt, tq), "spawn": (spawn_t, spawn_q)}.items():
                    Rh = qrot(hq)
                    loc = (pts - ht) @ Rh  # world -> trunk frame
                    inside_any = np.zeros(len(pts), bool)
                    by = {}
                    for n, (lt, lq, lo, hi) in meshes.items():
                        q = (loc - lt) @ qrot(lq)
                        ins = np.all((q >= lo - 0.005) & (q <= hi + 0.005), axis=1)
                        by[n] = int(ins.sum())
                        inside_any |= ins
                    fits[hyp] = {"n_points_inside_some_trunk_mesh_bbox": int(inside_any.sum()), "by_mesh": by}
                per[lab] = {"n_pixels": int(len(pts)), "centroid_world": pts.mean(0).tolist(),
                            "z_min_max": [float(pts[:, 2].min()), float(pts[:, 2].max())], "fits": fits}
            res[name] = {"eye": dc["eye"], "target": dc["target"], "robot_prims": per}
        UsdGeom.Imageable(terrain).MakeVisible()
        uenv.sim.render()
        R.setdefault("C_rendered_geometry", {})[tag] = {"trunk_pos": tt.tolist(), "trunk_quat": tq.tolist(), "views": res}
        for name in res:
            for lab, v in res[name]["robot_prims"].items():
                print(f"[C:{tag}:{name}] {lab.replace(ROOT, 'R')} px={v['n_pixels']} cur={v['fits']['current']['n_points_inside_some_trunk_mesh_bbox']} spawn={v['fits']['spawn']['n_points_inside_some_trunk_mesh_bbox']}")

    diag_capture("reset")

    # ---------- E1. physics: masses / inertias / COMs ----------
    pv = robot.root_physx_view
    masses = pv.get_masses()[0].cpu().numpy().astype(float)
    inert = pv.get_inertias()[0].cpu().numpy().astype(float).reshape(-1, 3, 3)
    coms = pv.get_coms()[0].cpu().numpy().astype(float)
    R["E_physics"] = {"body_names": list(robot.body_names), "masses": masses.tolist(), "total_mass": float(masses.sum()),
                      "trunk_mass": float(masses[trunk_idx]), "trunk_inertia_3x3": inert[trunk_idx].tolist(),
                      "trunk_com_pose": coms[trunk_idx].tolist(), "all_inertias": inert.tolist(), "all_coms": coms.tolist()}
    print("[E] total mass", masses.sum(), "trunk", masses[trunk_idx])

    from omni.physx import get_physx_scene_query_interface
    sq = get_physx_scene_query_interface()

    def ray_probe(tag):
        tt = robot.data.body_pos_w[0, trunk_idx].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, trunk_idx].cpu().numpy().astype(float)
        out_r = {}
        for n in ("mesh_3", "mesh_7", "mesh_11"):
            lt, lq, lo, hi = meshes[n]
            centre_local = lt + qrot(lq) @ ((lo + hi) / 2)
            for hyp, (ht, hq) in {"current": (tt, tq), "spawn": (spawn_t, spawn_q)}.items():
                c = ht + qrot(hq) @ centre_local
                o = c + np.array([0, 0, 0.35])
                hit = sq.raycast_closest(tuple(o.tolist()), (0.0, 0.0, -1.0), 0.6)
                out_r[f"{n}@{hyp}"] = {"ray_origin": o.tolist(), "hit": bool(hit["hit"]),
                                       "collision": str(hit.get("collision", "")), "rigidBody": str(hit.get("rigidBody", "")),
                                       "hit_z": float(hit["position"][2]) if hit["hit"] else None,
                                       "expected_top_z": float(c[2] + (hi[2] - lo[2]) / 2)}
        R.setdefault("E_raycasts", {})[tag] = out_r
        for k, v in out_r.items():
            print(f"[E-ray:{tag}] {k} hit={v['hit']} {v['collision'].replace(ROOT, 'R')} z={v['hit_z']} top~{v['expected_top_z']:.3f}")

    ray_probe("reset")

    # ---------- D. walk 2 s at 0.5 m/s through the walking policy ----------
    cmd = torch.tensor([args.walk_vx, 0.0, 0.0], device=obs.device)
    traj = []
    for i in range(args.walk_steps):
        obs, _, done, infos = env.step(cmd)
        tt = robot.data.body_pos_w[0, trunk_idx].cpu().numpy().astype(float)
        tq = robot.data.body_quat_w[0, trunk_idx].cpu().numpy().astype(float)
        camw = tt + qrot(tq) @ np.array(env_cfg.scene.rgb_camera.offset.pos)
        traj.append([i + 1, *tt.tolist(), yaw_of(tq), camw[2], float(tt[2] - start[2]), float(camw[2] - start[2])])
        if (i + 1) % 25 == 0:
            rgb = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
            viz = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
            save_png(rgb, f"{out}/D_walk_step{i + 1:03d}_robotcam.png")
            save_png(viz, f"{out}/D_walk_step{i + 1:03d}_chasecam.png")
            R.setdefault("D_walk_gray", {})[i + 1] = {"robotcam": GRAY(rgb), "chasecam": GRAY(viz)}
        if bool(done):
            R["D_walk_done_at"] = i + 1
            print("[D] env reported done at", i + 1)
            break
    tr = np.array(traj)
    np.savetxt(f"{out}/D_walk_trajectory.csv", tr, delimiter=",",
               header="step,trunk_x,trunk_y,trunk_z,yaw_deg,camera_z,trunk_z_minus_start_z,camera_z_minus_start_z", comments="")
    R["D_walk_heights"] = {"trunk_z_minus_start_z": {"mean": float(tr[:, 6].mean()), "min": float(tr[:, 6].min()), "max": float(tr[:, 6].max())},
                           "camera_z_minus_start_z": {"mean": float(tr[:, 7].mean()), "min": float(tr[:, 7].min()), "max": float(tr[:, 7].max())},
                           "displacement_m": float(np.linalg.norm(tr[-1, 1:3] - tt0[:2]))}
    poses("D_after_walk")
    print("[D] walked", R["D_walk_heights"])
    diag_capture("after_walk")
    ray_probe("after_walk")

    # ---------- F. back to the start pose, then the removal test ----------
    obs, infos = env.reset()
    rgbF = infos["observations"]["camera_obs"][0, :, :, :3].cpu().numpy()
    vizF = infos["observations"]["viz_camera_obs"][0, :, :, :3].cpu().numpy()
    save_png(rgbF, f"{out}/F0_rereset_robotcam.png")
    save_png(vizF, f"{out}/F0_rereset_chasecam.png")
    d = np.abs(rgbF.astype(np.int16) - rgb0.astype(np.int16))
    R["F0_rereset_vs_first_reset"] = {"gray": GRAY(rgbF), "max_abs_diff": int(d.max()), "mean_abs_diff": float(d.mean()),
                                     "share_channels_differ": float((d > 0).mean())}
    poses("F0_after_rereset")
    print("[F0] re-reset", R["F0_rereset_vs_first_reset"])

    def render_robotcam(tag):
        for _ in range(3):
            uenv.sim.render()
        rcam.update(0.0, force_recompute=True)
        vcam.update(0.0, force_recompute=True)
        a = rcam.data.output["rgb"][0, :, :, :3].cpu().numpy()
        v = vcam.data.output["rgb"][0, :, :, :3].cpu().numpy()
        save_png(a, f"{out}/F_{tag}_robotcam.png")
        save_png(v, f"{out}/F_{tag}_chasecam.png")
        R.setdefault("F_removal", {})[tag] = {"robotcam_gray": GRAY(a), "chasecam_gray": GRAY(v)}
        print(f"[F] {tag}: robotcam gray {GRAY(a):.4f}  chasecam gray {GRAY(v):.4f}")
        return a

    render_robotcam("1_rerender_no_change")
    # method A: visibility edit inside the converter's instanceable-mesh layer (the prototype source)
    lay_paths = sorted(glob.glob("/tmp/IsaacLab/usd_*/Props/instanceable_meshes.usd"), key=os.path.getmtime)
    layer = Sdf.Layer.FindOrOpen(lay_paths[-1])
    R["F_removal_layer"] = lay_paths[-1]
    for n in HEAD_ARM:
        ps = layer.GetPrimAtPath(f"/K1/Trunk/visuals/{n}")
        a = ps.attributes.get("visibility") if ps else None
        if a is None and ps is not None:
            a = Sdf.AttributeSpec(ps, "visibility", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
        if a is not None:
            a.default = "invisible"
    render_robotcam("2_methodA_head_arms_hidden")
    for n in HEAD_ARM:
        ps = layer.GetPrimAtPath(f"/K1/Trunk/visuals/{n}")
        if ps is not None and "visibility" in ps.attributes:
            ps.attributes["visibility"].default = "inherited"
    render_robotcam("3_methodA_restored")
    # method B: de-instance Trunk/visuals, then hide the (now ordinary) mesh prims
    vis_prim = stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals")
    vis_prim.SetInstanceable(False)
    render_robotcam("4_methodB_deinstanced_only")
    for n in HEAD_ARM:
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals/{n}")).MakeInvisible()
    render_robotcam("5_methodB_head_arms_hidden")
    for n in HEAD_ARM:
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{ROOT}/Trunk/visuals/{n}")).MakeVisible()
    render_robotcam("6_methodB_restored")

    json.dump(R, open(f"{out}/step3_results.json", "w"), indent=1, default=str)
    print("[done] wrote", f"{out}/step3_results.json")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
