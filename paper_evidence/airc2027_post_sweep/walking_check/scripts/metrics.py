"""Step 3 metrics (CPU) from the recorded per-step states, as defined in ../PLAN.md.

Time 0 = start of the test command; the state after command step j is at t = (j + 1) * dt; the state after the last
warm-up step is t = 0. Base velocity = velocity of the Trunk frame origin in the base frame,
v = root_lin_vel_b - root_ang_vel_b x r_com (r_com = root body's centre of mass in the Trunk frame, PhysX read-back);
Isaac Lab's root_lin_vel_b (root centre of mass) is reported alongside. Yaw rate = root_ang_vel_b z.

Usage: python metrics.py [--states_dir DIR] [--rep 1]
Writes ../logs/metrics.json, ../tables/metrics_all.{csv,md}, ../tables/sequence_segments.csv,
../tables/sequence_summary.md, ../tables/summary_by_robot_trial.md.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS = ["stand", "forward", "turn_left", "turn_right", "sequence"]
PI6 = math.pi / 6.0
CMD = {"stand": (0.0, 0.0, 0.0), "forward": (0.5, 0.0, 0.0), "turn_left": (0.0, 0.0, PI6),
       "turn_right": (0.0, 0.0, -PI6)}
IMPLIED = {"forward": 0.75, "turn_left": 45.0, "turn_right": -45.0, "zero": 0.0}  # m, deg, deg
SS_FROM_S = 4.0  # steady state = last 8 s of a 12-s command: 4 s < t <= 12 s


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def euler_zyx(q):
    """(roll, pitch, yaw) from quaternions (w, x, y, z), shape (N, 4)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_rotate(q, v):
    """Rotate vectors v (N, 3) by quaternions q (N, 4) (w, x, y, z): body -> world."""
    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def load(states_dir, robot, trial, rep):
    stem = os.path.join(states_dir, f"{robot}_{trial}_r{rep}")
    d = np.load(stem + ".npz")
    meta = json.load(open(stem + ".json"))
    return {k: d[k] for k in d.files}, meta


def trial_series(d, meta):
    """Command-phase series (float64) incl. the t = 0 start state; excludes a reset row after a termination."""
    phase = d["phase"]
    n_rows = len(phase)
    fall = meta.get("fall")
    valid = np.ones(n_rows, bool)
    if fall is not None:
        valid[fall["row"]:] = False  # the env reset itself in the termination step
    cmd_rows = np.flatnonzero((phase == 2) & valid)
    pre_rows = np.flatnonzero((phase < 2) & valid)
    if len(pre_rows) == 0:
        return None
    rows = np.concatenate([[pre_rows[-1]], cmd_rows])  # t = 0 state, then command steps
    dt = meta["schedule"]["dt_step"]
    t = np.arange(len(rows)) * dt
    r_com = np.asarray(meta["physx"]["root_com_b"], dtype=np.float64)
    f = {k: d[k][rows].astype(np.float64) for k in ("root_pos_w", "root_quat_w", "root_lin_vel_b", "root_ang_vel_b",
                                                    "root_lin_vel_w", "projected_gravity_b", "cmd")}
    v_origin_b = f["root_lin_vel_b"] - np.cross(f["root_ang_vel_b"], r_com)
    roll, pitch, yaw = euler_zyx(f["root_quat_w"])
    tilt = np.arccos(np.clip(-f["projected_gravity_b"][:, 2], -1.0, 1.0))
    return {"rows": rows, "t": t, "p": f["root_pos_w"], "q": f["root_quat_w"], "v": v_origin_b,
            "v_com": f["root_lin_vel_b"], "w": f["root_ang_vel_b"], "roll": roll, "pitch": pitch, "yaw": yaw,
            "tilt": tilt, "cmd": f["cmd"], "r_com": r_com, "dt": dt, "n_cmd": len(cmd_rows)}


def fall_info(d, meta):
    fall = meta.get("fall")
    if fall is None:
        return {"fell": False}
    phase = d["phase"]
    row = fall["row"]
    dt = meta["schedule"]["dt_step"] if "schedule" in meta else 0.02
    # time of the termination step from the start of the phase it happened in (= from the command start if
    # phase "command"); the termination step itself counts (state after step k is at (k + 1) dt)
    t = int(((phase == phase[row]) & (np.arange(len(phase)) <= row)).sum()) * dt
    return {"fell": True, "phase": fall["phase"], "t_s": t, "t_note": f"seconds from the start of phase {fall['phase']}",
            "row": row, "bad_orientation": fall.get("bad_orientation"), "terminated": fall.get("terminated")}


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
            "min": float(x.min()), "max": float(x.max())}


def steady_metrics(s, cmd):
    m = {}
    sel = s["t"] > SS_FROM_S + 1e-9
    sel[0] = False
    n_ss = int(sel.sum())
    m["steady_state_samples"] = n_ss
    comp = {"vx": (s["v"][:, 0], cmd[0]), "vy": (s["v"][:, 1], cmd[1]), "wz": (s["w"][:, 2], cmd[2])}
    for name, (x, c) in comp.items():
        xs = x[sel]
        mean = float(xs.mean()) if n_ss else float("nan")
        m[f"{name}_mean"] = mean
        m[f"{name}_ratio"] = (mean / c) if c != 0 else None
        m[f"{name}_rms_err"] = float(np.sqrt(np.mean((xs - c) ** 2))) if n_ss else float("nan")
        m[f"{name}_sd"] = float(xs.std(ddof=1)) if n_ss > 1 else float("nan")
    m["vx_com_mean"] = float(s["v_com"][sel, 0].mean()) if n_ss else float("nan")
    m["vy_com_mean"] = float(s["v_com"][sel, 1].mean()) if n_ss else float("nan")
    # check: finite-difference mean velocity of the Trunk position vs the mean of v (both world frame)
    if n_ss > 1:
        idx = np.flatnonzero(sel)
        i0, i1 = idx[0] - 1, idx[-1]
        v_fd_w = (s["p"][i1, :2] - s["p"][i0, :2]) / (s["t"][i1] - s["t"][i0])
        v_w = quat_rotate(s["q"][sel], s["v"][sel])[:, :2].mean(axis=0)
        m["check_fd_mean_vel_w"] = v_fd_w.tolist()
        m["check_v_mean_w"] = v_w.tolist()
        m["check_abs_diff_max"] = float(np.abs(v_fd_w - v_w).max())
    return m


def common_metrics(s):
    c = slice(1, None)  # command-phase states (t > 0)
    return {"height_mean": float(s["p"][c, 2].mean()), "height_min": float(s["p"][c, 2].min()),
            "max_abs_roll_deg": float(np.degrees(np.abs(s["roll"][c]).max())),
            "max_abs_pitch_deg": float(np.degrees(np.abs(s["pitch"][c]).max())),
            "max_tilt_deg": float(np.degrees(s["tilt"][c].max())),
            "max_tilt_t_s": float(s["t"][1:][np.argmax(s["tilt"][1:])]),
            "yaw_change_deg": float(np.degrees(np.unwrap(s["yaw"])[-1] - s["yaw"][0])),
            "planar_displacement_m": float(np.hypot(*(s["p"][-1, :2] - s["p"][0, :2]))),
            "command_steps_completed": int(s["n_cmd"])}


UPPER_BODY = ("Shoulder", "Elbow", "Head")


def upper_body_deviation(d, meta, s):
    """Robot T: max and mean |joint position - default (= training pose)| of the 10 head/arm joints, command phase."""
    names = meta["physx"]["joint_names"]
    idx = [i for i, n in enumerate(names) if any(u in n for u in UPPER_BODY)]
    if not idx:
        return {}
    q0 = np.asarray(meta["default_joint_pos"])[idx]
    dev = np.abs(d["joint_pos"][s["rows"][1:]][:, idx].astype(np.float64) - q0)
    return {"upper_body_max_dev_rad": float(dev.max()), "upper_body_mean_dev_rad": float(dev.mean())}


LEG_KEYS = ("Hip", "Knee", "Ankle")


def velocity_limit_use(d, meta):
    """Per leg joint: max |joint velocity| over all phases (and the phase), the PhysX velocity limit read back, and the
    number of command-phase steps at >= 99 % of that limit."""
    names = meta["physx"]["joint_names"]
    lim = np.asarray(meta["physx"]["dof_max_velocity"], dtype=np.float64)
    jv = np.abs(d["joint_vel"].astype(np.float64))
    cmd = d["phase"] == 2
    out = {}
    for i, n in enumerate(names):
        if not any(k in n for k in LEG_KEYS):
            continue
        r = int(np.argmax(jv[:, i]))
        out[n] = {"max_abs_vel": float(jv[r, i]), "at_phase": ["reset_warmup", "zero_warmup", "command"][int(d["phase"][r])],
                  "physx_limit": float(lim[i]), "cmd_steps_at_99pct_limit": int((jv[cmd, i] >= 0.99 * lim[i]).sum()),
                  "cmd_steps": int(cmd.sum())}
    return out


def sequence_metrics(s, segs):
    """Per complete segment, from the state before its first step to the state after its last step."""
    out = []
    n = s["n_cmd"]
    for i, (typ, a, b, complete, c) in enumerate(segs):
        if not complete or b > n:
            out.append({"i": i, "type": typ, "start": a, "end": b, "complete": False})
            continue
        p0, p1 = s["p"][a], s["p"][b]  # s index k = state after command step k-1; index a = before step a
        y0, y1 = s["yaw"][a], s["yaw"][b]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        row = {"i": i, "type": typ, "start": a, "end": b, "complete": True, "t0_s": a * s["dt"],
               "forward_m": dx * math.cos(y0) + dy * math.sin(y0),
               "lateral_m": -dx * math.sin(y0) + dy * math.cos(y0),
               "planar_m": math.hypot(dx, dy), "dyaw_deg": math.degrees(wrap(y1 - y0)),
               "mean_vx": float(s["v"][a + 1:b + 1, 0].mean()), "mean_wz": float(s["w"][a + 1:b + 1, 2].mean()),
               "mean_abs_wz": float(np.abs(s["w"][a + 1:b + 1, 2]).mean())}
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if nxt is not None and nxt[0] == "zero" and nxt[3] and nxt[2] <= n:
            e2 = nxt[2]
            dx2, dy2 = s["p"][e2][0] - p0[0], s["p"][e2][1] - p0[1]
            row["forward_m_incl_next_zero"] = dx2 * math.cos(y0) + dy2 * math.sin(y0)
            row["dyaw_deg_incl_next_zero"] = math.degrees(wrap(s["yaw"][e2] - y0))
        if typ == "zero":
            row["after"] = segs[i - 1][0] if i > 0 else None
        out.append(row)
    return out


def sequence_summary(rows):
    summ = {}
    comp = [r for r in rows if r["complete"]]
    for typ, key, implied in (("forward", "forward_m", 0.75), ("turn_left", "dyaw_deg", 45.0),
                              ("turn_right", "dyaw_deg", -45.0)):
        rs = [r for r in comp if r["type"] == typ]
        if not rs:
            continue
        st = stats([r[key] for r in rs])
        st["implied"] = implied
        st["ratio_of_mean"] = st["mean"] / implied
        summ[typ] = {"lateral_m": stats([r["lateral_m"] for r in rs]), "planar_m": stats([r["planar_m"] for r in rs]),
                     "dyaw_deg": stats([r["dyaw_deg"] for r in rs]), "mean_vx": stats([r["mean_vx"] for r in rs]),
                     "mean_wz": stats([r["mean_wz"] for r in rs]),
                     "incl_next_zero": stats([r[key + "_incl_next_zero"] for r in rs if key + "_incl_next_zero" in r])
                     if any(key + "_incl_next_zero" in r for r in rs) else None}
        summ[typ][key] = st  # the achieved-vs-implied entry (with implied and ratio_of_mean) wins over the generic one
        if summ[typ]["incl_next_zero"] is not None:
            summ[typ]["incl_next_zero"]["implied"] = implied
            summ[typ]["incl_next_zero"]["ratio_of_mean"] = summ[typ]["incl_next_zero"]["mean"] / implied
    for after in ("forward", "turn_left", "turn_right", None):
        rs = [r for r in comp if r["type"] == "zero" and (after is None or r["after"] == after)]
        if not rs:
            continue
        summ[f"zero_after_{after or 'any'}"] = {
            "planar_m": stats([r["planar_m"] for r in rs]), "dyaw_deg": stats([r["dyaw_deg"] for r in rs]),
            "abs_dyaw_deg": stats([abs(r["dyaw_deg"]) for r in rs]), "mean_vx": stats([r["mean_vx"] for r in rs]),
            "mean_abs_wz": stats([r["mean_abs_wz"] for r in rs])}
    return summ


def sequence_rms(s):
    c = slice(1, None)
    e = np.stack([s["v"][c, 0] - s["cmd"][c, 0], s["v"][c, 1] - s["cmd"][c, 1], s["w"][c, 2] - s["cmd"][c, 2]], 1)
    r = np.sqrt((e ** 2).mean(axis=0))
    return {"vx_rms_err": float(r[0]), "vy_rms_err": float(r[1]), "wz_rms_err": float(r[2])}


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", default=os.path.join(W, "logs", "states"))
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--out_dir", default=W)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    res = {"rep": a.rep, "definitions": __doc__, "results": {}}
    long_rows, seg_rows = [], []
    for robot in ("E", "T"):
        for trial in TRIALS:
            try:
                d, meta = load(a.states_dir, robot, trial, a.rep)
            except FileNotFoundError:
                res["results"][f"{robot}_{trial}"] = {"not_determined": "no recorded run"}
                continue
            s = trial_series(d, meta)
            r = {"fall": fall_info(d, meta), "wrapper_done_rows": meta.get("wrapper_done_rows", []),
                 "r_com_b": meta["physx"]["root_com_b"], "total_mass": meta["physx"]["total_mass"],
                 "velocity_limit_use": velocity_limit_use(d, meta)}
            if s is None or s["n_cmd"] == 0:
                r["not_determined"] = "no command-phase states (fell before the command)"
                res["results"][f"{robot}_{trial}"] = r
                continue
            r.update(common_metrics(s))
            r.update(upper_body_deviation(d, meta, s))
            expected = meta["schedule"]["command_steps"]
            r["command_completed"] = bool(s["n_cmd"] == expected)
            if not r["command_completed"]:
                r["note"] = (f"trial ended after {s['n_cmd']} of {expected} command steps (t = {s['t'][-1]:.2f} s); "
                             "height/tilt are up to that point; steady-state values and drifts not determined")
            if trial in CMD:
                r["command"] = list(CMD[trial])
                if r["command_completed"]:
                    r.update(steady_metrics(s, CMD[trial]))
                    if trial == "stand":
                        r["stand_drift_m"] = float(np.hypot(*(s["p"][-1, :2] - s["p"][0, :2])))
                        r["stand_yaw_drift_deg"] = float(np.degrees(wrap(s["yaw"][-1] - s["yaw"][0])))
                        r["stand_duration_s"] = float(s["t"][-1])
            else:
                segs = meta["schedule"]["segments"]
                rows = sequence_metrics(s, segs)
                r["segments"] = rows
                r["segment_summary"] = sequence_summary(rows)
                r.update(sequence_rms(s))
                for row in rows:
                    seg_rows.append({"robot": robot, **row})
            res["results"][f"{robot}_{trial}"] = r
            for k, v in r.items():
                if isinstance(v, (int, float)) or v is None:
                    long_rows.append({"robot": robot, "trial": trial, "metric": k, "value": v})
    json.dump(res, open(os.path.join(a.out_dir, "logs", "metrics.json"), "w"), indent=1, default=float)
    write_tables(res, long_rows, seg_rows, a.out_dir)


METRIC_ROWS = [  # (key, label, unit, trials)
    ("vx_mean", "forward speed, steady-state mean", "m/s", ("stand", "forward", "turn_left", "turn_right")),
    ("vx_ratio", "forward speed / command", "", ("forward",)),
    ("vx_rms_err", "forward speed RMS error", "m/s", TRIALS),
    ("vx_sd", "forward speed SD (stepping oscillation), steady state", "m/s",
     ("stand", "forward", "turn_left", "turn_right")),
    ("vy_mean", "lateral speed, steady-state mean", "m/s", ("stand", "forward", "turn_left", "turn_right")),
    ("vy_rms_err", "lateral speed RMS error", "m/s", TRIALS),
    ("wz_mean", "yaw rate, steady-state mean", "rad/s", ("stand", "forward", "turn_left", "turn_right")),
    ("wz_ratio", "yaw rate / command", "", ("turn_left", "turn_right")),
    ("wz_rms_err", "yaw rate RMS error", "rad/s", TRIALS),
    ("wz_sd", "yaw rate SD, steady state", "rad/s", ("stand", "forward", "turn_left", "turn_right")),
    ("vx_com_mean", "forward speed of the root centre of mass (Isaac Lab root_lin_vel_b), steady-state mean", "m/s",
     ("stand", "forward", "turn_left", "turn_right")),
    ("height_mean", "trunk height above the plane, mean", "m", TRIALS),
    ("height_min", "trunk height above the plane, min", "m", TRIALS),
    ("max_abs_roll_deg", "max absolute roll", "deg", TRIALS),
    ("max_abs_pitch_deg", "max absolute pitch", "deg", TRIALS),
    ("max_tilt_deg", "max tilt (fall criterion: 74.5°)", "deg", TRIALS),
    ("max_tilt_t_s", "time of max tilt (from command start)", "s", TRIALS),
    ("yaw_change_deg", "yaw change over the command phase (unwrapped)", "deg", TRIALS),
    ("planar_displacement_m", "planar displacement over the command phase", "m", TRIALS),
    ("upper_body_max_dev_rad", "robot T: max absolute deviation of a head/arm joint from the training pose", "rad", TRIALS),
    ("stand_drift_m", "horizontal drift over 12 s", "m", ("stand",)),
    ("stand_yaw_drift_deg", "yaw drift over 12 s", "deg", ("stand",)),
    ("check_abs_diff_max", "check: absolute difference, finite-difference vs v, steady-state mean (world xy)", "m/s",
     ("stand", "forward", "turn_left", "turn_right")),
]


def write_tables(res, long_rows, seg_rows, out_dir):
    R = res["results"]
    with open(os.path.join(out_dir, "tables", "metrics_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["robot", "trial", "metric", "value"])
        w.writeheader()
        w.writerows(long_rows)
    with open(os.path.join(out_dir, "tables", "sequence_segments.csv"), "w", newline="") as f:
        keys = ["robot", "i", "type", "after", "start", "end", "complete", "t0_s", "forward_m", "lateral_m",
                "planar_m", "dyaw_deg", "forward_m_incl_next_zero", "dyaw_deg_incl_next_zero", "mean_vx", "mean_wz",
                "mean_abs_wz"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(seg_rows)
    # one table of all metrics: rows = metric, columns = trial x robot
    L = ["| Metric | Unit | " + " | ".join(f"{t} E | {t} T" for t in TRIALS) + " |",
         "|---|---|" + "---|" * (2 * len(TRIALS))]
    for key, label, unit, trials in METRIC_ROWS:
        cells = []
        for t in TRIALS:
            for robot in ("E", "T"):
                v = R.get(f"{robot}_{t}", {}).get(key) if t in trials else None
                cells.append(fmt(v) if t in trials else "")
        L.append(f"| {label} | {unit} | " + " | ".join(cells) + " |")
    cells = []
    for t in TRIALS:
        for robot in ("E", "T"):
            fi = R.get(f"{robot}_{t}", {}).get("fall")
            cells.append("not determined" if fi is None else
                         ("no" if not fi["fell"] else f"yes, {fi['phase']} t={fi['t_s']:.2f} s"))
    L.append("| fell | | " + " | ".join(cells) + " |")
    seq_lines = sequence_tables(R)
    open(os.path.join(out_dir, "tables", "metrics_all.md"), "w").write(
        "## All metrics (repeat %d)\n\nCommand: stand (0, 0, 0); forward (0.5, 0, 0); turn_left (0, 0, +π/6); "
        "turn_right (0, 0, −π/6) for 12 s; steady state = last 8 s. Sequence: 30 s executor-like sequence (RMS errors "
        "against the step-wise command over 30 s; segment results below).\n\n" % res["rep"]
        + "\n".join(L) + "\n\n" + seq_lines)
    open(os.path.join(out_dir, "tables", "sequence_summary.md"), "w").write(seq_lines)
    open(os.path.join(out_dir, "tables", "summary_by_robot_trial.md"), "w").write(summary_table(R))
    open(os.path.join(out_dir, "tables", "velocity_limits.md"), "w").write(velocity_table(R))
    print("\n".join(L))
    print(seq_lines)
    print(summary_table(R))


def sequence_tables(R):
    L = ["## Executor-like sequence: achieved vs implied per complete segment\n",
         "| Segment type | Quantity | Implied | E: n, mean ± SD [min, max] | E ratio | T: n, mean ± SD [min, max] | T ratio |",
         "|---|---|---|---|---|---|---|"]

    def cell(st):
        return f"{st['n']}, {st['mean']:.3f} ± {st['sd']:.3f} [{st['min']:.3f}, {st['max']:.3f}]" if st else "n/a"
    specs = [("forward", "forward_m", "displacement along start heading (m)", 0.75),
             ("forward", "incl_next_zero", "same, at end of following zero segment (m)", 0.75),
             ("forward", "lateral_m", "lateral displacement (m)", 0.0),
             ("forward", "dyaw_deg", "yaw change (deg)", 0.0),
             ("turn_left", "dyaw_deg", "yaw change (deg)", 45.0),
             ("turn_left", "incl_next_zero", "same, at end of following zero segment (deg)", 45.0),
             ("turn_left", "planar_m", "planar displacement (m)", 0.0),
             ("turn_right", "dyaw_deg", "yaw change (deg)", -45.0),
             ("turn_right", "incl_next_zero", "same, at end of following zero segment (deg)", -45.0),
             ("turn_right", "planar_m", "planar displacement (m)", 0.0)]
    for typ, key, label, implied in specs:
        cs = []
        for robot in ("E", "T"):
            su = R.get(f"{robot}_sequence", {}).get("segment_summary", {}).get(typ)
            st = su.get(key) if su else None
            ratio = (st["mean"] / implied) if (st and implied) else None
            cs += [cell(st), fmt(ratio)]
        L.append(f"| {typ} | {label} | {implied:g} | " + " | ".join(cs) + " |")
    for z in ("zero_after_forward", "zero_after_turn_left", "zero_after_turn_right"):
        for key, label in (("planar_m", "residual planar displacement (m)"), ("abs_dyaw_deg", "residual absolute yaw change (deg)"),
                           ("mean_vx", "mean forward speed (m/s)"), ("mean_abs_wz", "mean absolute yaw rate (rad/s)")):
            cs = []
            for robot in ("E", "T"):
                su = R.get(f"{robot}_sequence", {}).get("segment_summary", {}).get(z)
                cs += [cell(su.get(key) if su else None), ""]
            L.append(f"| {z.replace('_', ' ')} (0.16 s) | {label} | 0 | " + " | ".join(cs) + " |")
    rms = []
    for robot in ("E", "T"):
        r = R.get(f"{robot}_sequence", {})
        rms.append(f"{robot}: vx {fmt(r.get('vx_rms_err'))} m/s, vy {fmt(r.get('vy_rms_err'))} m/s, "
                   f"wz {fmt(r.get('wz_rms_err'))} rad/s")
    L.append("\nRMS error against the step-wise command over the 30 s — " + "; ".join(rms) + ".\n")
    return "\n".join(L) + "\n"


def velocity_table(R):
    L = ["## Leg joint velocities vs the PhysX velocity limits read back (repeat %s)\n" % "1",
         "Max |joint velocity| over all phases of each trial, and the number of command-phase steps at >= 99 % of the "
         "PhysX limit (the limits are the URDF values; the configs' velocity_limit is not written by the fork).\n",
         "| Robot | Joint | PhysX limit (rad/s) | " + " | ".join(f"{t}: max / steps at limit" for t in TRIALS) + " |",
         "|---|---|---|" + "---|" * len(TRIALS)]
    for robot in ("E", "T"):
        joints = None
        for t in TRIALS:
            v = R.get(f"{robot}_{t}", {}).get("velocity_limit_use")
            if v:
                joints = list(v)
                break
        for n in joints or []:
            cells, lim = [], None
            for t in TRIALS:
                v = R.get(f"{robot}_{t}", {}).get("velocity_limit_use", {}).get(n)
                if v is None:
                    cells.append("n/a")
                    continue
                lim = v["physx_limit"]
                cells.append(f"{v['max_abs_vel']:.2f} ({v['at_phase'].replace('_', ' ')}) / "
                             f"{v['cmd_steps_at_99pct_limit']} of {v['cmd_steps']}")
            L.append(f"| {robot} | {n} | {lim:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def summary_table(R):
    L = ["| Robot | Trial | Forward speed (m/s) | Yaw rate (rad/s) | Fell | Trunk height mean / min (m) | Max tilt (°) |",
         "|---|---|---|---|---|---|---|"]
    for robot in ("E", "T"):
        for t in TRIALS:
            r = R.get(f"{robot}_{t}", {})
            if "height_mean" not in r:
                L.append(f"| {robot} | {t} | not determined | | | | |")
                continue
            fi = r["fall"]
            fell = "no" if not fi["fell"] else f"yes ({fi['phase']}, t = {fi['t_s']:.2f} s)"
            if t != "sequence" and not r.get("command_completed", False):
                L.append(f"| {robot} | {t} | not determined (ended early) | not determined | {fell} | "
                         f"{r['height_mean']:.3f} / {r['height_min']:.3f} (until end) | {r['max_tilt_deg']:.1f} |")
                continue
            if t == "sequence":
                su = r["segment_summary"]
                fs = f"{su['forward']['mean_vx']['mean']:.3f} (mean over forward segments)" if "forward" in su else "n/a"
                ys = (f"{su['turn_left']['mean_wz']['mean']:+.3f} / {su['turn_right']['mean_wz']['mean']:+.3f} "
                      f"(mean over left / right segments)") if "turn_left" in su and "turn_right" in su else "n/a"
            else:
                fs = f"{r['vx_mean']:.3f}" + (f" ({r['vx_ratio']:.2f} × cmd)" if r.get("vx_ratio") is not None else "")
                ys = f"{r['wz_mean']:+.3f}" + (f" ({r['wz_ratio']:.2f} × cmd)" if r.get("wz_ratio") is not None else "")
            L.append(f"| {robot} | {t} | {fs} | {ys} | {fell} | {r['height_mean']:.3f} / {r['height_min']:.3f} | "
                     f"{r['max_tilt_deg']:.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
