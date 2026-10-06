"""Step 5 analysis (CPU only). Reads airc2027_replay outputs; writes tables/*.csv, figures/*.png,
qstar_frames/ep<idx>/ (the 8 frames each run sent at q* + difference images) and
tables/results_tables.md (markdown tables for RESULTS.md). Verdicts come from compare_runs.py
(pinned in PLAN.md); this script only collects and formats them.
"""
import csv
import json
import os
import shutil

import numpy as np
from PIL import Image

import fr_common as C
from compare_runs import check_h2, check_h3, compare_pair

R = C.R
RUNS = REP = TAB = FIG = QF = None


def set_root(root):
    """Experiment root (default airc2027_replay; offline tests use a scratch root)."""
    global R, RUNS, REP, TAB, FIG, QF
    R = root
    RUNS, REP, TAB, FIG, QF = (os.path.join(R, d) for d in ("runs", "replay", "tables", "figures", "qstar_frames"))


set_root(C.R)
N_SENDS = 20   # sends per request in Step 4 (PLAN §8); --n-sends exists for offline tests only
CONDS = ("cond1", "cond2", "cond3")
COND_NAME = {"cond1": "simulator loaded", "cond2": "server alone", "cond3": "restarted server"}
# chart ink + the reference sequential blue ramp and one categorical accent (dataviz palette.md)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE = {200: "#9ec5f4", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 700: "#0d366b"}
ACCENT = "#eb6834"   # categorical slot 2 (orange): "differs"
SAME = "#d9d8d2"     # neutral: "identical"


def rd(idx, tag):
    return os.path.join(RUNS, f"ep{idx}_{tag}")


def write_csv(name, rows, fields=None):
    os.makedirs(TAB, exist_ok=True)
    path = os.path.join(TAB, name)
    if not rows:
        with open(path, "w") as fh:
            fh.write("(no rows)\n")
        return path
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    return path


def load_pairs():
    out = {}
    p = os.path.join(REP, "pairs.tsv")
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                idx, x = line.split()
                out[int(idx)] = x
    return out


def run_summary(idx, tag):
    d = rd(idx, tag)
    if not os.path.isdir(d):
        return None
    rec = C.load_record(d) or {}
    return {"ended_at_step": rec.get("ended_at_step"), "term_reason": rec.get("term_reason"),
            "success": rec.get("success"), "n_queries": len(C.load_jsonl(os.path.join(d, "queries.jsonl")))}


def diff_vis(d):
    """Binned single-hue rendering of a max-over-channels |difference| map (0 = surface)."""
    bins = [(0, 0, SURF), (1, 1, BLUE[200]), (2, 2, BLUE[350]), (3, 9, BLUE[450]), (10, 49, BLUE[550]),
            (50, 255, BLUE[700])]
    out = np.zeros(d.shape + (3,), np.uint8)
    for lo, hi, hexc in bins:
        rgb = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)]
        out[(d >= lo) & (d <= hi)] = rgb
    return out, bins


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    for d in (TAB, FIG, QF):
        os.makedirs(d, exist_ok=True)
    pairs = load_pairs()
    ep_rows, hyp_rows, h2_rows, rvr_rows, tl_rows, qf_rows, rr_rows, rr_sum = [], [], [], [], [], [], [], []
    first_div = None
    timelines = []
    for idx, rec, d1, d2 in C.EPISODES:
        X = pairs.get(idx)
        runs = {t: run_summary(idx, t) for t in "ABCDR"}
        row = {"episode_idx": idx, "record": rec, "sweep_D1_D2": f"{d1}/{d2}"}
        for t in "ABCDR":
            s = runs[t]
            row[f"run_{t}"] = "" if s is None else f"{s['ended_at_step']} steps, {s['term_reason']}, success={s['success']}"
        hyp = {"episode_idx": idx, "H1": "not tested", "H1_detail": "", "H3": "not tested", "H3_first_mismatch": ""}
        for c in CONDS:
            hyp[f"H2_{c}"] = "not tested"
        if X is None:
            row["pair"] = "not run"
        elif X == "none":
            n = sum(1 for t in "BCD" if runs[t] is not None)
            row["pair"] = f"no divergence in {n + 1} runs"
            # report any state difference without a reply difference
            A = C.load_run(rd(idx, "A"))
            for t in "BCD":
                if runs[t] is None:
                    continue
                res = compare_pair(A, C.load_run(rd(idx, t)))
                row[f"first_state_diff_A_vs_{t}"] = res["first_state_diff"] and res["first_state_diff"]["label"]
                row[f"records_equal_A_vs_{t}"] = res["records_equal_six_fields"]
        else:
            A, Xr = C.load_run(rd(idx, "A")), C.load_run(rd(idx, X))
            res = compare_pair(A, Xr, images=True)
            q = res["qstar"]
            row.update({"pair": f"A vs {X}", "qstar_index": q["i"], "qstar_step_A": q["step_A"],
                        "qstar_step_X": q["step_X"],
                        "first_state_diff": ("none before q*" if res["H1"]["a_states_before_step_identical"]
                                             and res["H1"]["b_state_at_qstar_identical"] else
                                             (res["first_state_diff"] or {}).get("label")),
                        "first_state_diff_label_any": (res["first_state_diff"] or {}).get("label"),
                        "first_request_diff_index": (res["first_request_diff"] or {}).get("i"),
                        "first_request_diff_step": (res["first_request_diff"] or {}).get("step_A"),
                        "frames_differ_at_qstar": f"{q['n_frames_differ']} of 8",
                        "frames_differ_slots": [f["slot"] + 1 for f in q["frames"] if not f["frame_sha_same"]],
                        "qstar_max_abs_diff": max(f.get("max_abs_diff", 0) for f in q["frames"]),
                        "qstar_mean_abs_diff_mean": float(np.mean([f.get("mean_abs_diff", 0) for f in q["frames"]])),
                        "reply_A": q["reply_A"], "reply_X": q["reply_X"],
                        "records_equal_six_fields": res["records_equal_six_fields"]})
            hyp["H1"] = res["H1"]["verdict"]
            hyp["H1_detail"] = (f"a={res['H1']['a_states_before_step_identical']} b={res['H1']['b_state_at_qstar_identical']} "
                                f"c={res['H1']['c_requests_at_qstar_differ']}; first state diff "
                                f"{(res['first_state_diff'] or {}).get('label')}")
            rvr_rows.append({"episode_idx": idx, "pair": f"A vs {X}", "queries_before_qstar": res["n_queries_before_qstar"],
                             "requests_differed_replies_matched_before_qstar": res["n_requests_differ_replies_match_before_qstar"],
                             "first_request_diff_index": (res["first_request_diff"] or {}).get("i")})
            for t in res["timeline"]:
                tl_rows.append(dict(episode_idx=idx, pair=f"A vs {X}", **t))
            timelines.append((idx, X, res["timeline"], q["i"], len(A["queries"]), len(Xr["queries"])))
            for f in q["frames"]:
                qf_rows.append(dict(episode_idx=idx, **{k: (v + 1 if k == "slot" else v) for k, v in f.items()}))
            # q* frames + difference images
            od = os.path.join(QF, f"ep{idx}")
            os.makedirs(od, exist_ok=True)
            a_q, x_q = A["queries"][q["i"]], Xr["queries"][q["i"]]
            for j, (ka, kx) in enumerate(zip(a_q["indices"], x_q["indices"])):
                shutil.copy2(os.path.join(rd(idx, "A"), "frames", f"f{ka:04d}.png"),
                             os.path.join(od, f"A_slot{j + 1}_f{ka:04d}.png"))
                shutil.copy2(os.path.join(rd(idx, X), "frames", f"f{kx:04d}.png"),
                             os.path.join(od, f"{X}_slot{j + 1}_f{kx:04d}.png"))
                fa, fx = C.load_frame(rd(idx, "A"), ka), C.load_frame(rd(idx, X), kx)
                dmax = np.abs(fa.astype(np.int16) - fx.astype(np.int16)).max(axis=-1).astype(np.uint8)
                Image.fromarray(dmax).save(os.path.join(od, f"absdiff_slot{j + 1}_raw.png"))
                Image.fromarray(diff_vis(dmax)[0]).save(os.path.join(od, f"absdiff_slot{j + 1}_vis.png"))
            if first_div is None:
                first_div = (idx, X, q, a_q, x_q)
            # H3
            if os.path.isdir(rd(idx, "R")):
                Rr = C.load_run(rd(idx, "R"))
                h3 = check_h3(A, Rr)
                hyp["H3"] = h3["H3"]
                mm = h3["replay_mismatch"]
                hyp["H3_first_mismatch"] = "" if mm is None else f"{mm['what']} at {mm['where']}: A={mm['run_A']} R={mm['run_R']}"
                if h3["H3"] == "failed" and mm is None:
                    hyp["H3_first_mismatch"] = (f"records: {[(k, v['A'], v['R']) for k, v in h3['fields'].items() if not v['equal']]}; "
                                                f"first state diff {h3['first_state_diff']}; first query diff {h3['first_query_diff']}")
                # R fresh renders vs A frames
                stats = []
                for fr in Rr["frames"]:
                    k = fr["k"]
                    pa = os.path.join(rd(idx, "A"), "frames", f"f{k:04d}.png")
                    pr = os.path.join(rd(idx, "R"), "fresh_frames", f"f{k:04d}.png")
                    if not (os.path.exists(pa) and os.path.exists(pr)):
                        continue
                    s = C.img_diff_stats(C.load_frame(rd(idx, "A"), k), C.load_frame(rd(idx, "R"), k, "fresh_frames"))
                    s.update({"episode_idx": idx, "k": k, "label": fr["label"],
                              "fresh_equals_A": fr["fresh_sha256"] == fr.get("A_sha256")})
                    rr_rows.append(s)
                    stats.append(s)
                if stats:
                    rr_sum.append({"episode_idx": idx, "frames_compared": len(stats),
                                   "frames_bit_identical": sum(1 for s in stats if s["max_abs_diff"] == 0),
                                   "max_abs_diff_max": max(s["max_abs_diff"] for s in stats),
                                   "max_abs_diff_median": float(np.median([s["max_abs_diff"] for s in stats])),
                                   "mean_abs_diff_mean": float(np.mean([s["mean_abs_diff"] for s in stats])),
                                   "share_channels_differ_median": float(np.median([s["share_channels_differ"] for s in stats])),
                                   "share_channels_differ_min": min(s["share_channels_differ"] for s in stats),
                                   "share_channels_differ_max": max(s["share_channels_differ"] for s in stats)})
            # H2
            rq = os.path.join(REP, f"ep{idx}", "requests.json")
            for c in CONDS:
                fn = "replay_queries.jsonl" if c == "cond1" else "replies.jsonl"
                path = os.path.join(REP, f"ep{idx}", c, fn)
                if os.path.exists(rq) and os.path.exists(path):
                    h2 = check_h2(rq, path, n_expected=N_SENDS)
                    hyp[f"H2_{c}"] = h2["H2"]
                    for r in h2["requests"]:
                        h2_rows.append({"episode_idx": idx, "condition": c, "condition_name": COND_NAME[c],
                                        "request": r["label"], "logged_reply": r["logged_reply"], "sends": r["sends"],
                                        "own_reply_count": r["own_reply_count"], "distribution": r["distribution"],
                                        "request_sha_ok": r["request_sha_ok"], "requests_identical": h2["requests_identical"]})
        ep_rows.append(row)
        hyp_rows.append(hyp)

    ep_fields = ["episode_idx", "record", "sweep_D1_D2", "run_A", "run_B", "run_C", "run_D", "run_R", "pair",
                 "qstar_index", "qstar_step_A", "qstar_step_X", "first_state_diff", "first_state_diff_label_any",
                 "first_request_diff_index", "first_request_diff_step", "frames_differ_at_qstar", "frames_differ_slots",
                 "qstar_max_abs_diff", "qstar_mean_abs_diff_mean", "reply_A", "reply_X", "records_equal_six_fields",
                 "first_state_diff_A_vs_B", "records_equal_A_vs_B", "first_state_diff_A_vs_C", "records_equal_A_vs_C",
                 "first_state_diff_A_vs_D", "records_equal_A_vs_D"]
    write_csv("episodes.csv", ep_rows, ep_fields)
    write_csv("hypotheses.csv", hyp_rows)
    write_csv("h2_replies.csv", h2_rows)
    write_csv("requests_vs_replies.csv", rvr_rows)
    write_csv("query_timeline.csv", tl_rows)
    write_csv("qstar_frames.csv", qf_rows)
    write_csv("runR_fresh_vs_A.csv", rr_rows)
    write_csv("runR_fresh_vs_A_summary.csv", rr_sum)

    # GPU time
    gpu = []
    gp = os.path.join(R, "logs", "gpu_time.tsv")
    if os.path.exists(gp):
        for line in open(gp):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 5 and f[3].lstrip("-").isdigit():
                gpu.append({"label": f[0], "start": f[1], "end": f[2], "wall_s": int(f[3]), "rc": f[4],
                            "command": f[5] if len(f) > 5 else ""})
    write_csv("gpu_time.csv", gpu)

    # totals
    def tot(key):
        vals = [h[key] for h in hyp_rows if h[key] != "not tested"]
        return f"{sum(1 for v in vals if v == 'held')} of {len(vals)}"
    totals = {"H1": tot("H1"), "H3": tot("H3"), **{f"H2_{c}": tot(f"H2_{c}") for c in CONDS}}
    with open(os.path.join(TAB, "totals.json"), "w") as fh:
        json.dump(totals, fh, indent=2)

    # ---------------- figures ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    if first_div is not None:
        idx, X, q, a_q, x_q = first_div
        ka, kx = a_q["indices"][7], x_q["indices"][7]
        fa, fx = C.load_frame(rd(idx, "A"), ka), C.load_frame(rd(idx, X), kx)
        dmax = np.abs(fa.astype(np.int16) - fx.astype(np.int16)).max(axis=-1)
        vis, bins = diff_vis(dmax.astype(np.uint8))
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1]})
        for ax, img, ttl in ((axs[0], fa, f"run A, frame {ka}"), (axs[1], fx, f"run {X}, frame {kx}"),
                             (axs[2], vis, f"|A − {X}|, max over channels")):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(ttl, color=INK, fontsize=9, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        counts = [int(((dmax >= lo) & (dmax <= hi)).sum()) for lo, hi, _ in bins]
        n = dmax.size
        def blab(lo, hi):
            return "0 (identical)" if hi == 0 else ("1 level" if hi == 1 else (f"{lo} levels" if lo == hi else f"{lo}–{hi} levels"))
        handles = [Patch(facecolor=c, edgecolor=GRID, label=f"{blab(lo, hi)}  ({100 * k / n:.2f} %)")
                   for (lo, hi, c), k in zip(bins, counts)]
        axs[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                      title="pixels by max |difference|", title_fontsize=8, labelcolor=INK2)
        fig.suptitle(f"Episode idx {idx}: the current-observation frame (8th of 8) sent at q* = query {q['i']} "
                     f"(step {q['step_A']}). Replies: A “{q['reply_A']}” / {X} “{q['reply_X']}”",
                     x=0.01, ha="left", color=INK, fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(FIG, f"qstar_diff_heatmap_ep{idx}.png"), dpi=160)
        plt.close(fig)
    if timelines:
        nmax = max(max(t[4], t[5]) for t in timelines)
        fig, ax = plt.subplots(figsize=(max(8, min(18, 0.11 * nmax + 3)), 0.62 * len(timelines) * 2 + 1.2))
        yt, yl = [], []
        for r, (idx, X, tl, qs, na, nx) in enumerate(timelines):
            for lane, key, name in ((0, "request_same", "request"), (1, "reply_same", "reply")):
                y = r * 2.6 + lane
                for t in tl:
                    ax.add_patch(plt.Rectangle((t["i"] - 0.42, y - 0.38), 0.84, 0.76,
                                               color=SAME if t[key] else ACCENT, lw=0))
                yt.append(y); yl.append(f"ep {idx} (A vs {X}) · {name}")
            ax.plot([qs, qs], [r * 2.6 - 0.6, r * 2.6 + 1.6], color=INK, lw=1.2)
            ax.text(qs + 0.6, r * 2.6 + 1.45, f"q* = {qs}", color=INK, fontsize=7.5, va="center")
            common = len(tl)
            if max(na, nx) > common:
                ax.text(common + 0.2, r * 2.6 + 0.5, f"→ A: {na} queries, {X}: {nx}", color=MUTED, fontsize=7.5, va="center")
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8, color=INK2)
        ax.set_xlim(-1, nmax + 8); ax.set_ylim(len(timelines) * 2.6 - 0.9, -0.9)
        ax.set_xlabel("query index (common to both runs)")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(handles=[Patch(color=SAME, label="identical in both runs"), Patch(color=ACCENT, label="differs"),
                           Line2D([0], [0], color=INK, lw=1.2, label="q* (first reply that differs)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12 if len(timelines) > 2 else -0.25), ncol=3,
                  frameon=False, fontsize=8, labelcolor=INK2)
        ax.set_title("Request-hash and reply agreement per query, run A vs the first run that diverged",
                     loc="left", color=INK, fontsize=9.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "timeline_request_reply.png"), dpi=160)
        plt.close(fig)

    # ---------------- markdown tables ----------------
    def md(rows, cols, hdr=None):
        if not rows:
            return "(none)\n"
        hdr = hdr or cols
        out = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(cols) + "\n"
        for r in rows:
            out += "| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |\n"
        return out
    with open(os.path.join(TAB, "results_tables.md"), "w") as fh:
        fh.write("## Episodes\n\n" + md(ep_rows, ["episode_idx", "record", "run_A", "run_B", "run_C", "run_D", "pair",
                                                   "qstar_index", "qstar_step_A", "first_state_diff",
                                                   "first_request_diff_index", "frames_differ_at_qstar",
                                                   "qstar_max_abs_diff", "reply_A", "reply_X"]))
        fh.write("\n## Hypotheses\n\n" + md(hyp_rows, ["episode_idx", "H1", "H1_detail", "H2_cond1", "H2_cond2",
                                                        "H2_cond3", "H3", "H3_first_mismatch"]))
        fh.write("\nTotals: " + json.dumps(totals) + "\n")
        fh.write("\n## H2 reply counts\n\n" + md(h2_rows, ["episode_idx", "condition", "request", "own_reply_count",
                                                           "sends", "distribution"]))
        fh.write("\n## Requests vs replies before q*\n\n" + md(rvr_rows, list(rvr_rows[0].keys()) if rvr_rows else []))
        fh.write("\n## Frames at q*\n\n" + md(qf_rows, ["episode_idx", "slot", "k_A", "label_A", "frame_sha_same",
                                                        "max_abs_diff", "mean_abs_diff", "share_channels_differ"]))
        fh.write("\n## Run R fresh renders vs run A frames\n\n" + md(rr_sum, list(rr_sum[0].keys()) if rr_sum else []))
    print(json.dumps(totals))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=C.R)
    ap.add_argument("--n-sends", type=int, default=20)
    _a = ap.parse_args()
    N_SENDS = _a.n_sends
    set_root(os.path.realpath(_a.root))
    main()
