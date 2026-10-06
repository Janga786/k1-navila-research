"""Step 2.1 (CPU): compare K1_locomotion.urdf (evaluated) and K1_22dof.urdf (training) and compute the mass
properties a rigid merge of Trunk + head + arms (fixed joints at their zero pose) should have.
Writes diagnosis/models/urdf_compare.md and urdf_merged_mass_properties.json."""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

K1 = os.path.expanduser("~/robots/k1/workspace/booster_assets/robots/K1")
OUT = os.path.expanduser("~/Projects/k1_research/airc2027_renders/diagnosis/models")
HEAD_ARM_JOINTS = ["AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
                   "Left_Elbow_Yaw", "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
TRAIN_POSE = {"ALeft_Shoulder_Pitch": 0.2, "ARight_Shoulder_Pitch": 0.2, "Left_Shoulder_Roll": -1.25,
              "Right_Shoulder_Roll": 1.25, "Left_Elbow_Pitch": 0.0, "Right_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": -0.5,
              "Right_Elbow_Yaw": 0.5, "AAHead_yaw": 0.0, "Head_pitch": 0.0}


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_mat(ax, q):
    ax = np.array(ax) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


def parse(fn):
    r = ET.parse(fn).getroot()
    links, joints = {}, {}
    for l in r.findall("link"):
        i = l.find("inertial")
        o = i.find("origin")
        I = i.find("inertia").attrib
        links[l.get("name")] = {"mass": float(i.find("mass").get("value")),
                                "com": [float(v) for v in o.get("xyz").split()],
                                "I": np.array([[float(I["ixx"]), float(I["ixy"]), float(I["ixz"])],
                                               [float(I["ixy"]), float(I["iyy"]), float(I["iyz"])],
                                               [float(I["ixz"]), float(I["iyz"]), float(I["izz"])]])}
    for j in r.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        joints[j.get("name")] = {"type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                                 "xyz": [float(v) for v in o.get("xyz").split()], "rpy": [float(v) for v in o.get("rpy").split()],
                                 "axis": [float(v) for v in a.get("xyz").split()] if a is not None else None}
    return links, joints


def link_frames(joints, q):
    """World(=Trunk) frames of every link for joint values q (default 0)."""
    F = {"Trunk": (np.zeros(3), np.eye(3))}
    changed = True
    while changed:
        changed = False
        for n, j in joints.items():
            if j["parent"] in F and j["child"] not in F:
                pt, pR = F[j["parent"]]
                Rj = rpy_mat(*j["rpy"])
                if j["axis"] is not None:
                    Rj = Rj @ axis_mat(j["axis"], q.get(n, 0.0))
                F[j["child"]] = (pt + pR @ np.array(j["xyz"]), pR @ Rj)
                changed = True
    return F


def composite(links, F, names):
    m = sum(links[n]["mass"] for n in names)
    c = sum(links[n]["mass"] * (F[n][0] + F[n][1] @ np.array(links[n]["com"])) for n in names) / m
    I = np.zeros((3, 3))
    for n in names:
        R = F[n][1]
        r = F[n][0] + R @ np.array(links[n]["com"]) - c
        I += R @ links[n]["I"] @ R.T + links[n]["mass"] * (r @ r * np.eye(3) - np.outer(r, r))
    return m, c, I


def main():
    os.makedirs(OUT, exist_ok=True)
    lo_links, lo_j = parse(f"{K1}/K1_locomotion.urdf")
    tr_links, tr_j = parse(f"{K1}/K1_22dof.urdf")
    merged = ["Trunk", "Head_1", "Head_2", "Left_Arm_1", "Left_Arm_2", "Left_Arm_3", "left_hand_link",
              "Right_Arm_1", "Right_Arm_2", "Right_Arm_3", "right_hand_link"]
    F0 = link_frames(lo_j, {})
    m, c, I = composite(lo_links, F0, merged)
    Ft = link_frames(tr_j, TRAIN_POSE)
    mt, ct, It = composite(tr_links, Ft, merged)
    res = {"total_mass_K1_locomotion": sum(l["mass"] for l in lo_links.values()),
           "total_mass_K1_22dof": sum(l["mass"] for l in tr_links.values()),
           "trunk_link_alone": {"mass": lo_links["Trunk"]["mass"], "com": lo_links["Trunk"]["com"], "inertia": lo_links["Trunk"]["I"].tolist()},
           "merged_trunk_zero_pose(=evaluated geometry)": {"mass": m, "com": c.tolist(), "inertia_about_com_trunk_axes": I.tolist(),
                                                          "principal_moments": sorted(np.linalg.eigvalsh(I).tolist())},
           "same_bodies_in_training_pose(for reference; in training they are separate actuated bodies)": {
               "mass": mt, "com": ct.tolist(), "inertia_about_com_trunk_axes": It.tolist()},
           "link_frames_zero_pose": {n: {"pos": F0[n][0].round(5).tolist()} for n in merged},
           "link_frames_training_pose": {n: {"pos": Ft[n][0].round(5).tolist(), "R": Ft[n][1].round(4).tolist()} for n in merged}}
    json.dump(res, open(f"{OUT}/urdf_merged_mass_properties.json", "w"), indent=1)

    L = ["# K1_locomotion.urdf (evaluated) vs K1_22dof.urdf (training)", "",
         f"SHA-256: K1_locomotion.urdf `5cf5951d…`, K1_22dof.urdf `9312ecc9…` (full hashes in `../provenance_sweep_files.txt`).",
         "`diff` of the two files: the only differences are the 10 head/arm joint `type` attributes (revolute → fixed) and two",
         "XML comment delimiters. Origins, axes, limits, masses, inertias, visuals and collisions are byte-identical.", "",
         "## Head and arm joints", "",
         "| Joint | Parent → child | K1_locomotion | K1_22dof | origin xyz | origin rpy | axis | training pose (rad) |",
         "|---|---|---|---|---|---|---|---|"]
    for n in HEAD_ARM_JOINTS:
        a, b = lo_j[n], tr_j[n]
        L.append(f"| {n} | {a['parent']} → {a['child']} | {a['type']} | {b['type']} | {a['xyz']} | {a['rpy']} | {a['axis']} | {TRAIN_POSE[n]} |")
    L += ["", "With every joint fixed at its origin (angle 0), the arm pose is the zero pose: shoulder roll 0 puts the",
          "arms straight out sideways (T-pose). Link frame positions in the Trunk frame:", "",
          "| Link | zero pose (evaluated) xyz | training pose xyz |", "|---|---|---|"]
    for n in merged[1:]:
        L.append(f"| {n} | {F0[n][0].round(4).tolist()} | {Ft[n][0].round(4).tolist()} |")
    L += ["", "## Masses", "", "| Link | mass (kg), both files |", "|---|---|"]
    for n, l in lo_links.items():
        L.append(f"| {n} | {l['mass']} |")
    L += [f"| **total** | **{res['total_mass_K1_locomotion']:.3f}** (K1_22dof: {res['total_mass_K1_22dof']:.3f}) |", "",
          "## Rigid merge (what the evaluated Trunk body should be)", "",
          f"Trunk + Head_1 + Head_2 + both arm chains at the zero pose: mass **{m:.3f} kg**, COM {np.round(c, 4).tolist()} (Trunk frame),",
          f"inertia about the COM (Trunk axes) diag ≈ {np.round(np.diag(I), 4).tolist()}, principal moments {np.round(sorted(np.linalg.eigvalsh(I)), 4).tolist()}.",
          f"Trunk link alone: 6.5 kg, diag {np.round(np.diag(lo_links['Trunk']['I']), 4).tolist()}."]
    open(f"{OUT}/urdf_compare.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
