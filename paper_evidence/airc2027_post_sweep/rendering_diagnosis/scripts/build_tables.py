"""Build the Markdown tables used in DIAGNOSIS.md from the result JSON files (CPU). Writes diagnosis/tables.md."""
import json
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
J = lambda p: json.load(open(os.path.join(R, p)))  # noqa: E731
f3 = lambda v: "(" + ", ".join(f"{x:.4f}" for x in v) + ")"  # noqa: E731
out = []

# ---- Step 3.1 / 3.2 poses ----
for cz in ("057", "042", "025"):
    S = J(f"diagnosis/sim/step3_cz{cz}/step3_results.json")
    out.append(f"### Poses, cam_z {S['cam_z']} (dataset index 0, record 0, zsNo4HB9uLZ)\n")
    out.append("| Item | after the reset warm-up | after the 2-s walk (100 steps, 0.5 m/s) |\n|---|---|---|")
    a, b = S["poses"]["B_after_reset"], S["poses"]["D_after_walk_2s"]
    out.append(f"| Trunk position (Isaac Lab = Fabric `_worldPosition`) | {f3(a['trunk_isaaclab']['pos'])} | {f3(b['trunk_isaaclab']['pos'])} |")
    out.append(f"| Trunk yaw (deg) | {a['trunk_isaaclab']['yaw_deg']:.2f} | {b['trunk_isaaclab']['yaw_deg']:.2f} |")
    out.append(f"| Trunk USD pose (never updated under Fabric = spawn pose) | {f3(a['trunk_usd']['pos'])} | {f3(b['trunk_usd']['pos'])} |")
    out.append(f"| Robot camera (Isaac Lab `data.pos_w`) | {f3(a['robot_camera']['isaaclab_data_pos_w'])} | {f3(b['robot_camera']['isaaclab_data_pos_w'])} |")
    out.append(f"| Robot camera = Trunk(now) x offset | {f3(a['robot_camera']['trunk_x_offset_pos_w'])} | {f3(b['robot_camera']['trunk_x_offset_pos_w'])} |")
    for k in ("mesh_2", "mesh_3", "mesh_4", "mesh_7", "mesh_8", "mesh_11", "mesh_1", "mesh_0"):
        ma, mb = a["meshes"][k], b["meshes"][k]
        out.append(f"| {k} ({ma['link']}): USD world / Trunk(now) x local | {f3(ma['usd_world_pos'])} / {f3(ma['trunk_now_x_local_pos'])} | "
                   f"{f3(mb['usd_world_pos'])} / {f3(mb['trunk_now_x_local_pos'])} |")
    out.append("")

# ---- Step 3 verification (rendered footprint vs hypotheses) ----
out.append("### Where the head/arm meshes are rendered (IoU of the rendered footprint with each hypothesis)\n")
out.append("Footprint = pixels that change when only the 10 head/arm meshes are made invisible (identical physics, checked bit-exact).\n")
out.append("| Run | Frame | robot cam: IoU current / spawn (footprint px) | chase cam: IoU current / spawn (footprint px) |\n|---|---|---|---|")
for lab, d in (("as evaluated, cam_z 0.57", "verify_cz057"), ("as evaluated, cam_z 0.42", "verify_cz042"), ("as evaluated, cam_z 0.25", "verify_cz025"),
               ("de-instanced Trunk/visuals, cam_z 0.57", "verify_deinstanced_cz057")):
    V = J(f"diagnosis/sim/{d}/verify.json")
    for t, r in V["frames"].items():
        g = lambda c: (f"{r[c]['current_trunk']['iou'] if r[c]['current_trunk']['iou'] is None else round(r[c]['current_trunk']['iou'], 3)} / "  # noqa: E731
                       f"{r[c]['spawn_pose']['iou'] if r[c]['spawn_pose']['iou'] is None else round(r[c]['spawn_pose']['iou'], 3)} ({r[c]['current_trunk']['gt_px']})")
        out.append(f"| {lab} | {t} | {g('robotcam')} | {g('chasecam')} |")
out.append("")

# ---- removal / causal tests (reset frame, cam_z) ----
out.append("### Removal and causal tests (reset frame, dataset index 0)\n")
out.append("| Run | cam_z | robot-cam gray | robot-cam median depth in gray (m) | Trunk pose identical to as-evaluated |\n|---|---|---|---|---|")
base = {cz: J(f"diagnosis/sim/step3_cz{cz}/step3_results.json")["frames"]["A_reset"] for cz in ("057", "042", "025")}
for lab, d, cz in (("as evaluated", "step3_cz057", "057"), ("head+arm meshes hidden", "removal_headarms_cz057", "057"),
                   ("arm meshes hidden (head kept)", "removal_armsonly_cz057", "057"), ("head meshes hidden (arms kept)", "removal_headonly_cz057", "057"),
                   ("mesh_1..11 given mesh_0's material binding", "causal_material_binding_cz057", "057"),
                   ("Trunk/visuals de-instanced before load", "causal_deinstanced_cz057", "057"),
                   ("as evaluated", "step3_cz042", "042"), ("head+arm meshes hidden", "removal_headarms_cz042", "042"),
                   ("logo mesh (mesh_1) hidden", "removal_logoonly_cz042", "042"),
                   ("as evaluated", "step3_cz025", "025"), ("head+arm meshes hidden", "removal_headarms_cz025", "025")):
    f = J(f"diagnosis/sim/{d}/step3_results.json")["frames"]["A_reset"]
    same = f["trunk_pos"] == base[cz]["trunk_pos"] and f["trunk_quat_wxyz"] == base[cz]["trunk_quat_wxyz"]
    dm = f["robotcam_depth_in_gray"].get("depth_median")
    out.append(f"| {lab} | 0.{cz[1:]} | {f['robotcam_gray']:.4f} | {'' if dm is None else f'{dm:.4f}'} | {same} |")
out.append("")

# ---- camera heights ----
H = J("diagnosis/camera_heights.json")
out.append("### Actual camera heights (dataset index 0; floor under the Trunk by ray cast onto the Matterport mesh)\n")
out.append(f"Floor z = {H['floor_z_below_trunk']:.4f}; episode start_position z = {H['episode_start_z']:.4f} "
           f"({H['episode_start_z'] - H['floor_z_below_trunk']:.3f} m above the floor).\n")
out.append("| cam_z | nominal (0.53 + cam_z) | Trunk above floor after reset | camera above floor after reset | camera above start z | camera above floor at end of 4-s warm-up | camera above floor while walking (mean, min-max) |\n|---|---|---|---|---|---|---|")
for r in H["rows"]:
    w = r["walking_2s_0.5mps_camera_above_floor_approx"]
    out.append(f"| {r['cam_z']:.2f} | {r['nominal_label_0.53+cam_z']:.2f} | {r['init']['trunk_above_floor']:.3f} | {r['init']['camera_above_floor']:.3f} | "
               f"{r['init']['camera_above_start_z']:.3f} | {r['warmup8']['camera_above_floor']:.3f} | {w['mean']:.3f} ({w['min']:.3f}-{w['max']:.3f}) |")
out.append("")

# ---- stale geometry predictions ----
G = J("diagnosis/stale_geometry_check_all.json")
out.append("### Prediction from the stale geometry (spawn pose, 1-cm near plane) vs rendered gray\n")
out.append("| Episode | cam_z | frame | rendered gray | predicted coverage by stale meshes (32x18 rays) | of which back faces | meshes | Trunk z - spawn z |\n|---|---|---|---|---|---|---|---|")
for d, e in G.items():
    for t in ("init", "warmup8", "main000"):
        r = e["frames"][t]
        out.append(f"| idx {e['episode_idx']} (rec {e['record']}) | {e['cam_z']} | {t} | {r['gray']:.4f} | {r['predicted_any_stale_hit_fraction']:.3f} | "
                   f"{r['predicted_backface_fraction']:.3f} | {', '.join(r['meshes_covering'])} | {r['trunk_z_minus_spawn_z']:+.3f} |")
out.append("")

# ---- step 4 ----
out.append("### Step 4 repeatability (fresh processes)\n")
out.append("| Comparison | frame | max abs diff | mean abs diff | share of channels differing | root state bit-identical | all body states bit-identical | gray per process |\n|---|---|---|---|---|---|---|---|")
for lab, p in (("idx 0, cam_z 0.25, proc 2 vs 1", "diagnosis/repeatability/step4_1_cz025_compare.json"),
               ("idx 0, cam_z 0.57, proc 2 vs 1", "diagnosis/repeatability/step4_1_cz057_compare.json"),
               ("idx 21, cam_z 0.25, procs 2-4 vs 1", "diagnosis/repeatability/step4_2_idx21_compare.json")):
    C = J(p)
    for t, r in C["frames"].items():
        if p.endswith("idx21_compare.json") and not (t in ("init", "warmup8") or t.startswith("main")):
            continue
        v = r["vs_process_1"]
        mx = ", ".join(str(x["max_abs_diff"]) for x in v)
        mn = ", ".join(f"{x['mean_abs_diff']:.3f}" for x in v)
        sh = ", ".join(f"{x['share_channels_differ']:.3f}" for x in v)
        gr = ", ".join(f"{g:.4f}" for g in r["gray_frac_per_process"])
        rb = all(x["root_state_bit_identical"] for x in v)
        bb = all(x["all_body_states_bit_identical"] for x in v)
        out.append(f"| {lab} | {t} | {mx} | {mn} | {sh} | {rb} | {bb} | {gr} |")
out.append("")
open(os.path.join(R, "diagnosis", "tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
