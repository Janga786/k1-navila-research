"""Numbers for two findings from the retained episode videos (Sections III-D, V-B, V-C).

1. Relabelled wall-clock records. 34 records labelled wall_timeout belong to episodes that had
   finished: the evaluator's SIGTERM handler stayed armed after the result was written and, when
   the 900-s limit arrived while the episode video was being written, overwrote the record's label
   and set success/spl to 0. The list (data/relabelled_wall_timeouts.csv) comes from the
   workstation's file listing: these are the only wall_timeout records whose episode video exists
   (a video is written only after the episode loop ends). common.py excludes them from the
   interruption indicator; this script summarises them.
2. Gray robot-camera frames. data/gray_frames_start.csv holds, for every episode video, the share of
   the robot-camera half (left 1280 px) of video frames 0 and 1 whose pixels are all within +-6 of
   RGB 53 (frame 0: initial frame after the reset warm-up, which enters every model query; frame 1:
   the frame recorded at the first model query). data/gray_frames_full_length/<run>.csv scans every
   frame of every video of all 11 runs. A frame counts as gray when the share is >= 0.10. The
   post-sweep diagnosis (private archive, 14_airc2027_renders/DIAGNOSIS.md) traced the region at
   episode start in the 0.92-m and 1.07-m runs to the robot's own head, drawn at the trunk's spawn pose.
   Both scans were made on the workstation with ffmpeg (scripts in the private archive,
   NaVILA-Complete-Archive/12_airc2027_materials/analysis/scripts/).
Writes tables/corrections_and_gray.json. Deterministic.
"""
import json, os
import numpy as np
import pandas as pd
from common import load_runs, pinned_sets, wide, TABLES, HERE

df = load_runs(); S300, S200 = pinned_sets(df)
out = {}

# ---------------- 1. relabelled records ----------------
rel = df[df.relabelled == 1]
out["relabelled"] = dict(
    total=int(len(rel)),
    per_run={r: int(n) for r, n in rel.groupby("run").size().items()},
    at_cap=int((rel.ended_at_step >= 6001).sum()),
    ended_early=[dict(run=r.run, record_idx=int(r.record_idx), step=int(r.ended_at_step),
                      distance=round(float(r.distance_to_goal), 2)) for r in rel[rel.ended_at_step < 6001].itertuples()],
    within_3m=int((rel.distance_to_goal < 3.0).sum()),
    within_3m_at_cap=int(((rel.distance_to_goal < 3.0) & (rel.ended_at_step >= 6001)).sum()),
    with_reset_signature=int(rel.reset_sig.sum()))
sw = df[df.run != "base"]
out["labelled_wall_timeouts"] = int(sw.timeout_label.sum())
out["true_interruptions"] = dict(total=int(sw.timeout.sum()),
                                 stubs=int(((sw.timeout == 1) & (sw.sentinel == 1)).sum()),
                                 handler_written=int(((sw.timeout == 1) & (sw.sentinel == 0)).sum()))
hw = sw[(sw.timeout == 1) & (sw.sentinel == 0)]
out["true_interruptions"].update(min_step=int(hw.ended_at_step.min()),
                                 after_5500=int((hw.ended_at_step > 5500).sum()),
                                 within_3m=int((hw.distance_to_goal < 3.0).sum()),
                                 within_3m_by_run={r: int(n) for r, n in hw[hw.distance_to_goal < 3.0].groupby("run").size().items()})

# ---------------- 2. gray robot-camera frames ----------------
g = pd.read_csv(os.path.join(HERE, "data", "gray_frames_start.csv"))
TAG2RUN = {"full_14498": "base", "stretchA_300": "A", "stretchB_300": "B", "crop_300": "crop", "pad_300": "pad",
           "h060_200": "h060", "h078_200": "h078", "h095_200": "h095", "h110_200": "h110",
           "h130_200": "h130", "h150_200": "h150"}
g["run"] = g["run"].map(TAG2RUN)
start = {}
for r, x in g.groupby("run"):
    start[r] = dict(videos=int(len(x)), frame0_gray=int((x.frame0_gray_frac >= 0.10).sum()),
                    frame1_gray=int((x.frame1_gray_frac >= 0.10).sum()),
                    frame0_gray_ge50=int((x.frame0_gray_frac >= 0.50).sum()))
out["gray_start"] = start
others = [r for r in start if r not in ("h095", "h110")]
out["gray_start_other_runs"] = dict(videos=int(sum(start[r]["videos"] for r in others)),
                                    frame0_or_1_gray=int(sum(start[r]["frame0_gray"] + start[r]["frame1_gray"] for r in others)))
# Full-length scans of every frame of every video (decoded at reduced size). Note: the
# robot-camera half of each video is refreshed only when a frame is added to the model's
# frame history (every 25 control steps = 0.5 s; navila_eval_v3.py), so gray video frames
# come in blocks of five, each block one history frame; frame 0 is the initial frame.
FULL = {}
for tag, r in TAG2RUN.items():
    f = pd.read_csv(os.path.join(HERE, "data", "gray_frames_full_length", f"{tag}.csv"))
    f["run"] = r
    FULL[r] = f
full = {}
for r, f in FULL.items():
    n_any = int((f.n_robotcam_gray_frames > 0).sum())
    late = int(((f.n_robotcam_gray_frames > 0) & (f.last_gray_frame.fillna(-1) > 50)).sum())
    at_start = int(((f.frame0_gray_frac >= 0.10) | (f.frame1_gray_frac >= 0.10)).sum())
    full[r] = dict(videos=int(len(f)), any_gray=n_any, any_gray_pct=100 * n_any / len(f),
                   gray_frames=int(f.n_robotcam_gray_frames.sum()), all_frames=int(f.n_frames.sum()),
                   gray_frame_pct=100 * float(f.n_robotcam_gray_frames.sum()) / float(f.n_frames.sum()),
                   videos_with_gray_after_5s=late, videos_gray_at_start=at_start,
                   gray_frames_in_first_5s=int(f.n_gray_frames_in_first_5s_of_main_loop.sum()) +
                                           int((f.frame0_gray_frac >= 0.10).sum()))
out["gray_full_length"] = full
oth = {r: v for r, v in full.items() if r not in ("h095", "h110")}
out["gray_full_length_other_runs"] = dict(
    any_gray_pct_range=[min(v["any_gray_pct"] for v in oth.values()), max(v["any_gray_pct"] for v in oth.values())],
    gray_frame_pct_max=max(v["gray_frame_pct"] for v in oth.values()),
    videos_gray_at_start=int(sum(v["videos_gray_at_start"] for v in oth.values())),
    default_config_any_gray_pct={r: full[r]["any_gray_pct"] for r in ("base", "A", "B", "h078")})

# Gray frames and rerun divergence: pairs of default-configuration runs on shared episodes
# with videos in both runs; identical = bit-identical records.
F = pd.concat(FULL.values()).rename(columns={"file_idx": "record_idx"}).set_index(["run", "record_idx"])
sigs = df.set_index(["run", "record_idx"])["sig"]
pairs = {}
for a, b, idx in [("A", "B", S300), ("A", "h078", S200), ("B", "h078", S200),
                  ("base", "A", S300), ("base", "B", S300), ("base", "h078", S200)]:
    rows = []
    for i in idx:
        if (a, i) not in F.index or (b, i) not in F.index:
            continue
        ga, gb = F.loc[(a, i)], F.loc[(b, i)]
        same_gray = (int(ga.n_robotcam_gray_frames) == int(gb.n_robotcam_gray_frames) and
                     str(ga.first_gray_frame) == str(gb.first_gray_frame) and str(ga.last_gray_frame) == str(gb.last_gray_frame))
        rows.append(dict(i=i, ident=sigs.loc[(a, i)] == sigs.loc[(b, i)],
                         gray_a=int(ga.n_robotcam_gray_frames) > 0, gray_b=int(gb.n_robotcam_gray_frames) > 0,
                         same_gray=same_gray))
    P = pd.DataFrame(rows); I = P[P.ident]; D = P[~P.ident]
    Ig = I[I.gray_a | I.gray_b]; Dg = D[D.gray_a | D.gray_b]
    pairs[f"{a}-{b}"] = dict(pairs_with_videos=int(len(P)), identical=int(len(I)), identical_with_gray=int(len(Ig)),
                             identical_with_gray_same_pattern=int(Ig.same_gray.sum()),
                             identical_gray_mismatch_records=[int(x) for x in Ig[~Ig.same_gray].i],
                             diverged=int(len(D)), diverged_with_gray_in_either=int(len(Dg)),
                             diverged_with_gray_in_both=int((D.gray_a & D.gray_b).sum()))
out["gray_and_divergence"] = pairs
# episodes that share a route share the start pose: does the frame-1 gray status agree within routes?
cons = {}
for r in ("h095", "h110"):
    x = g[(g.run == r)].dropna(subset=["episode_idx"]).copy()
    x["route"] = df.set_index(["run", "record_idx"]).loc[[(r, int(i)) for i in x.file_idx], "route"].values
    grp = [v.frame1_gray_frac.ge(0.10).tolist() for _, v in x.groupby("route") if len(v) >= 2]
    cons[r] = dict(routes_with_2plus_videos=len(grp), routes_agreeing=int(sum(len(set(v)) == 1 for v in grp)))
out["gray_route_consistency"] = cons
# descriptive: SR shift against the reruns on episodes with and without a gray frame 1
sub = {}
for r in ("h095", "h110"):
    x = g[(g.run == r)]
    aff = set(x[x.frame1_gray_frac >= 0.10].file_idx.astype(int)); vid = set(x.file_idx.astype(int))
    Y = wide(df, "success", [r, "A", "B", "h078"], S200)
    res = {}
    for name, ids in [("gray_at_first_query", aff), ("no_gray_at_first_query", vid - aff)]:
        idx = [i for i in S200 if i in ids]
        y = Y.loc[idx]
        res[name] = dict(n=len(idx), SR_config=100 * float(y[r].mean()),
                         SR_reruns=100 * float(y[["A", "B", "h078"]].values.mean()),
                         delta=100 * float(y[r].mean() - y[["A", "B", "h078"]].values.mean()))
    sub[r] = res
out["gray_subsets_descriptive"] = sub

with open(os.path.join(TABLES, "corrections_and_gray.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
