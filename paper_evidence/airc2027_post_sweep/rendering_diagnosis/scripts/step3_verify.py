"""Step 3 verification (CPU): where does the renderer draw the merged head and arm meshes?

Ground truth: the pixels where an as-evaluated frame differs from the frame of the same step in the removal run
(head and arm visual meshes invisible, everything else identical; physics checked bit-identical).
Prediction: the head/arm STL meshes (K1 meshes, local link frames) placed with the merged local offsets from the
converted USD, under two hypotheses:
   current : attached to the Trunk pose of that frame  (what a rigid merge should give)
   spawn   : at the Trunk's spawn pose (the USD pose, which Fabric-mode physics never updates)
rasterised into the robot camera (1280x720, fx = 1280*24/47.71) and the chase camera (512x512, fx = 512*24/100),
both mounted on the CURRENT Trunk pose with the configured offsets and ROS-convention rotation.
Reports IoU / precision / recall of each hypothesis against the ground truth, and for the robot camera the depth
the hypotheses predict (Moller-Trumbore ray cast on a pixel grid) against the rendered depth.

Usage: python step3_verify.py <as_evaluated_dir> <removal_dir> <out_dir>
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image

MESHDIR = os.path.expanduser("~/robots/k1/workspace/booster_assets/robots/K1/meshes")
STL = {"mesh_2": "Head_1", "mesh_3": "Head_2", "mesh_4": "Left_Arm_1", "mesh_5": "Left_Arm_2", "mesh_6": "Left_Arm_3",
       "mesh_7": "Left_Arm_4", "mesh_8": "Right_Arm_1", "mesh_9": "Right_Arm_2", "mesh_10": "Right_Arm_3",
       "mesh_11": "Right_Arm_4"}
MOUNT = (-0.5, 0.5, -0.5, 0.5)  # CameraCfg.OffsetCfg rot, convention "ros" (z fwd, x right, y down)
NEAR = 0.005


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_meshes(R):
    out = {}
    for k, name in STL.items():
        m = trimesh.load(f"{MESHDIR}/{name}.STL")
        lo, hi = np.array(R["meshes"][k]["bbox_local_min"]), np.array(R["meshes"][k]["bbox_local_max"])
        assert np.allclose(m.bounds[0], lo, atol=2e-3) and np.allclose(m.bounds[1], hi, atol=2e-3), (k, m.bounds, lo, hi)
        out[k] = (np.asarray(m.vertices) + np.array(R["meshes"][k]["local_t"]), np.asarray(m.faces))
    return out


def world_tris(meshes, t, q):
    Rm = qrot(q)
    return [((v @ Rm.T) + t)[f] for v, f in meshes.values()]  # list of (F,3,3)


def cam_pose(trunk_t, trunk_q, offset):
    Rt = qrot(trunk_q)
    return trunk_t + Rt @ np.array(offset), Rt @ qrot(MOUNT)


def raster(tris_w, cam_t, R_wc, W, H, fx):
    mask = np.zeros((H, W), np.uint8)
    for T in tris_w:
        Pc = (T - cam_t) @ R_wc  # (F,3,3) camera coords
        z = Pc[..., 2]
        front = (z > NEAR).all(1)
        polys = []
        P = Pc[front]
        uv = np.stack([fx * P[..., 0] / P[..., 2] + W / 2, fx * P[..., 1] / P[..., 2] + H / 2], -1)
        polys.extend(list(uv))
        for tri in Pc[~front & (z > NEAR).any(1)]:  # clip triangles crossing the near plane
            pts = []
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                if a[2] > NEAR:
                    pts.append(a)
                if (a[2] > NEAR) != (b[2] > NEAR):
                    s = (NEAR - a[2]) / (b[2] - a[2])
                    pts.append(a + s * (b - a))
            pts = np.array(pts)
            polys.append(np.stack([fx * pts[:, 0] / pts[:, 2] + W / 2, fx * pts[:, 1] / pts[:, 2] + H / 2], -1))
        for p in polys:
            if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6:
                continue
            cv2.fillConvexPoly(mask, np.round(p * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return mask.astype(bool)


def raycast_depth(tris_w, cam_t, R_wc, W, H, fx, step=20):
    """z-depth (distance to image plane) of the nearest hit, either face, on a pixel grid."""
    vs, us = np.mgrid[step // 2:H:step, step // 2:W:step]
    d_cam = np.stack([(us + 0.5 - W / 2) / fx, (vs + 0.5 - H / 2) / fx, np.ones_like(us, float)], -1).reshape(-1, 3)
    D = d_cam @ R_wc.T  # world directions (unnormalised: z_cam component = 1 => t is z-depth)
    best = np.full(len(D), np.inf)
    T = np.concatenate(tris_w, 0)
    for c in range(0, len(T), 800):
        v0, v1, v2 = T[c:c + 800, 0], T[c:c + 800, 1], T[c:c + 800, 2]
        e1, e2 = v1 - v0, v2 - v0
        p = np.cross(D[:, None, :], e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        s = cam_t - v0
        u = (s[None] * p).sum(-1) * inv
        qv = np.cross(s[None], e1[None])
        v = (D[:, None, :] * qv).sum(-1) * inv
        t = (e2[None] * qv).sum(-1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        t = np.where(hit, t, np.inf)
        best = np.minimum(best, t.min(1))
    return vs, us, best.reshape(vs.shape)


def score(gt, pred):
    i, u = (gt & pred).sum(), (gt | pred).sum()
    return {"iou": float(i / u) if u else None, "precision": float(i / pred.sum()) if pred.sum() else None,
            "recall": float(i / gt.sum()) if gt.sum() else None, "gt_px": int(gt.sum()), "pred_px": int(pred.sum())}


def main():
    ev_dir, rm_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    Re = json.load(open(f"{ev_dir}/step3_results.json"))
    Rr = json.load(open(f"{rm_dir}/step3_results.json"))
    meshes = load_meshes(Re)
    spawn_t, spawn_q = np.array(Re["spawn_pose_usd"]["pos"]), np.array(Re["spawn_pose_usd"]["quat_wxyz"])
    res = {"as_evaluated": ev_dir, "removal": rm_dir, "cam_z": Re["cam_z"], "frames": {}}
    rows = []
    for tag, fe in Re["frames"].items():
        if tag not in Rr["frames"] or not os.path.exists(f"{rm_dir}/{tag}_robotcam.png"):
            continue
        fr = Rr["frames"][tag]
        same_phys = fe["trunk_pos"] == fr["trunk_pos"] and fe["trunk_quat_wxyz"] == fr["trunk_quat_wxyz"]
        tt, tq = np.array(fe["trunk_pos"]), np.array(fe["trunk_quat_wxyz"])
        rec = {"trunk_pose_bit_identical_to_removal_run": same_phys}
        for cam, (W, H, fx, off) in {"robotcam": (1280, 720, 1280 * 24 / 47.71, Re["robot_camera_offset_trunk_frame"]),
                                     "chasecam": (512, 512, 512 * 24 / 100.0, Re["chase_camera_offset_trunk_frame"])}.items():
            A = np.array(Image.open(f"{ev_dir}/{tag}_{cam}.png")).astype(np.int16)
            B = np.array(Image.open(f"{rm_dir}/{tag}_{cam}.png")).astype(np.int16)
            gt = np.abs(A - B).max(2) > 10
            gt = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            ct, Rwc = cam_pose(tt, tq, off)
            pc = raster(world_tris(meshes, tt, tq), ct, Rwc, W, H, fx)
            ps = raster(world_tris(meshes, spawn_t, spawn_q), ct, Rwc, W, H, fx)
            rec[cam] = {"current_trunk": score(gt, pc), "spawn_pose": score(gt, ps),
                        "max_abs_pixel_diff_eval_vs_removal": int(np.abs(A - B).max())}
            # overlay: ground truth (white tint), spawn hypothesis outline (red), current hypothesis outline (cyan)
            vis = A.astype(np.uint8).copy()
            vis[gt] = (0.5 * vis[gt] + 0.5 * 255).astype(np.uint8)
            for m, col in ((pc, (0, 255, 255)), (ps, (255, 0, 0))):
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cnt, -1, col, 2 if cam == "robotcam" else 1)
            Image.fromarray(vis).save(f"{out}/{tag}_{cam}_overlay.png")
            if cam == "robotcam" and gt.sum() >= 1000:  # depth check only where head/arms cover the robot camera
                for hyp, (ht, hq) in {"current_trunk": (tt, tq), "spawn_pose": (spawn_t, spawn_q)}.items():
                    vs, us, zd = raycast_depth(world_tris(meshes, ht, hq), ct, Rwc, W, H, fx)
                    dep = np.load(f"{ev_dir}/{tag}_robotcam_depth.npy")[vs, us]
                    g = gt[vs, us]
                    hit = np.isfinite(zd)
                    both = g & hit
                    rec[cam][hyp]["depth_check_grid20px"] = {
                        "grid_pts_in_gt": int(g.sum()), "grid_pts_predicted_hit": int(hit.sum()),
                        "gt_pts_with_predicted_hit": int(both.sum()),
                        "median_abs_depth_err_m": float(np.median(np.abs(zd[both] - dep[both]))) if both.any() else None,
                        "median_rendered_depth_m": float(np.median(dep[g])) if g.any() else None,
                        "median_predicted_depth_m": float(np.median(zd[both])) if both.any() else None}
        res["frames"][tag] = rec
        rows.append(f"{tag:18s} phys_identical={same_phys}  " + "  ".join(
            f"{c}: IoU cur={rec[c]['current_trunk']['iou'] if rec[c]['current_trunk']['iou'] is None else round(rec[c]['current_trunk']['iou'], 3)} "
            f"spawn={rec[c]['spawn_pose']['iou'] if rec[c]['spawn_pose']['iou'] is None else round(rec[c]['spawn_pose']['iou'], 3)} (gt {rec[c]['current_trunk']['gt_px']}px)"
            for c in ("robotcam", "chasecam")))
        print(rows[-1], flush=True)
    json.dump(res, open(f"{out}/verify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
