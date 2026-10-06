"""Per-frame check (CPU): where is the robot camera relative to the visual meshes left at the spawn pose?

For captures from sim_capture.py (trunk pose per history frame), the robot camera position is expressed in the SPAWN
frame of the Trunk (episode start_position + (0, 0, 0.55), start_rotation; this is the USD pose, which is where the
renderer draws the merged head/arm meshes and the logo, see Step 3), and tested against each mesh's bounding box
(local offsets and bounds read from the converted USD, diagnosis/sim/step3_cz057/step3_results.json).
Prints: camera height above/below spawn, whether it is inside each stale mesh's bounding box, distance to the Head_2
box, and the rendered gray fraction of that frame.
Usage: python stale_geometry_check.py <out_json> <capture_dir> [...]"""
import gzip
import json
import os
import sys

import numpy as np
import trimesh

KR = os.path.expanduser("~/Projects/k1_research")
EPS = json.load(gzip.open(f"{KR}/NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz", "rt"))["episodes"]
S3 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "diagnosis/sim/step3_cz057/step3_results.json")))
NAMES = {"mesh_1": "K1logo", "mesh_2": "Head_1", "mesh_3": "Head_2", "mesh_4": "L_Arm_1", "mesh_5": "L_Arm_2", "mesh_6": "L_Arm_3",
         "mesh_7": "L_hand", "mesh_8": "R_Arm_1", "mesh_9": "R_Arm_2", "mesh_10": "R_Arm_3", "mesh_11": "R_hand"}


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


STL = {"mesh_1": "K1logo", "mesh_2": "Head_1", "mesh_3": "Head_2", "mesh_4": "Left_Arm_1", "mesh_5": "Left_Arm_2",
       "mesh_6": "Left_Arm_3", "mesh_7": "Left_Arm_4", "mesh_8": "Right_Arm_1", "mesh_9": "Right_Arm_2", "mesh_10": "Right_Arm_3",
       "mesh_11": "Right_Arm_4"}
TRIS = None


def stale_tris():
    """All stale meshes in the spawn Trunk frame: (T,3,3) triangles, (T,3) outward normals, (T,) mesh index."""
    global TRIS
    if TRIS is None:
        T, N, I = [], [], []
        for i, (k, n) in enumerate(STL.items()):
            m = trimesh.load(os.path.expanduser(f"~/robots/k1/workspace/booster_assets/robots/K1/meshes/{n}.STL"))
            v = np.asarray(m.vertices) + np.array(S3["meshes"][k]["local_t"])
            T.append(v[m.faces]); N.append(np.asarray(m.face_normals)); I.append(np.full(len(m.faces), i))
        TRIS = (np.concatenate(T), np.concatenate(N), np.concatenate(I))
    return TRIS


def first_hits(o, D, tmin=1e-5):
    """Moller-Trumbore, nearest hit per ray beyond tmin (D has unit z_cam, so t is the z-depth): t, triangle (-1 = none)."""
    T, _, _ = stale_tris()
    best = np.full(len(D), np.inf)
    arg = np.full(len(D), -1)
    for c in range(0, len(T), 2000):
        v0, v1, v2 = T[c:c + 2000, 0], T[c:c + 2000, 1], T[c:c + 2000, 2]
        e1, e2 = v1 - v0, v2 - v0
        p = np.cross(D[:, None, :], e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        s = o - v0
        u = (s[None] * p).sum(-1) * inv
        q = np.cross(s[None], e1[None])
        v = (D[:, None, :] * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        t = np.where(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin), t, np.inf)
        j = t.argmin(1)
        tm = t[np.arange(len(D)), j]
        better = tm < best
        best[better], arg[better] = tm[better], (c + j)[better]
    return best, arg


def predicted_gray(cam_spawn, R_cam_spawn, W=1280, H=720, fx=1280 * 24 / 47.71, nu=32, nv=18):
    """On a 32x18 ray grid: fraction of rays whose first stale-mesh hit beyond the camera's 1-cm near clipping plane
    (Isaac Lab PinholeCameraCfg clipping_range=(0.01, 1e6)) lies within 0.5 m (= stale geometry fills that pixel),
    and how many of those hits are back faces."""
    _, N, I = stale_tris()
    us = (np.arange(nu) + 0.5) * W / nu
    vs = (np.arange(nv) + 0.5) * H / nv
    uu, vv = np.meshgrid(us, vs)
    d_cam = np.stack([(uu - W / 2) / fx, (vv - H / 2) / fx, np.ones_like(uu)], -1).reshape(-1, 3)  # ROS camera
    D = d_cam @ R_cam_spawn.T
    t, a = first_hits(cam_spawn, D, tmin=0.01)
    hit = (a >= 0) & (t < 0.5)
    back = np.zeros(len(D), bool)
    back[hit] = (N[a[hit]] * D[hit]).sum(-1) > 0
    names = list(STL.values())
    return float(back.mean()), float(hit.mean()), sorted({names[I[j]] for j in a[hit]})


MOUNT = (-0.5, 0.5, -0.5, 0.5)


def box_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    res = {}
    for d in dirs:
        m = json.load(open(f"{d}/capture.json"))
        ep = EPS[m["episode_idx"]]
        st = np.array(ep["start_position"], float)
        spawn_t, spawn_R = st + np.array([0, 0, 0.55]), qrot(np.array(ep["start_rotation"], float))
        off = np.array([0.10, 0.0, m["cam_z"]])
        rows = {}
        for tag, f in m["frames"].items():
            tt, tq = np.array(f["trunk_pos"]), np.array(f["trunk_quat_wxyz"])
            cam = tt + qrot(tq) @ off
            c = spawn_R.T @ (cam - spawn_t)  # camera in the spawn Trunk frame
            inside = []
            for k, n in NAMES.items():
                mm = S3["meshes"][k]
                lo = np.array(mm["bbox_local_min"]) + np.array(mm["local_t"])
                hi = np.array(mm["bbox_local_max"]) + np.array(mm["local_t"])
                if np.all(c >= lo) and np.all(c <= hi):
                    inside.append(n)
            h2 = S3["meshes"]["mesh_3"]
            R_cs = spawn_R.T @ qrot(tq) @ qrot(MOUNT)  # ROS camera axes in the spawn Trunk frame
            pg, ph, pn = predicted_gray(c, R_cs)
            rows[tag] = {"gray": round(f["robotcam_gray_frac"], 4), "predicted_backface_fraction": round(pg, 4),
                         "predicted_any_stale_hit_fraction": round(ph, 4), "meshes_covering": pn, "trunk_z_minus_spawn_z": round(float(tt[2] - spawn_t[2]), 4),
                         "camera_in_spawn_frame": np.round(c, 4).tolist(), "inside_stale_bbox_of": inside,
                         "dist_to_stale_Head_2_bbox_m": round(box_dist(c, np.array(h2["bbox_local_min"]) + h2["local_t"],
                                                                       np.array(h2["bbox_local_max"]) + h2["local_t"]), 4)}
        res[d] = {"episode_idx": m["episode_idx"], "record": m["record"], "cam_z": m["cam_z"], "frames": rows}
        print(f"== idx {m['episode_idx']} rec {m['record']} cam_z {m['cam_z']}")
        for tag, r in rows.items():
            print(f"  {tag:9s} gray {r['gray']:.4f} predicted coverage {r['predicted_any_stale_hit_fraction']:.3f} (back {r['predicted_backface_fraction']:.3f}) {r['meshes_covering']}  trunk-spawn dz {r['trunk_z_minus_spawn_z']:+.3f}  cam(spawn frame) {r['camera_in_spawn_frame']}  "
                  f"inside {r['inside_stale_bbox_of']}  d(Head_2) {r['dist_to_stale_Head_2_bbox_m']}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
