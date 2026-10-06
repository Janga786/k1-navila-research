"""Shared data layer for the AIRC 2027 reanalysis.

Reads the committed per-episode receipts (read-only) and the VLN-CE-Isaac episode
file, and returns one tidy table with one row per (run, episode).

Conventions (all verified against the evaluator source in the public archive):
  * success/spl are 0 for wall_timeout records (forced by the runner/handler);
  * sim_done records whose distance_to_goal equals the post-reset start distance
    (0.55 m vertical offset + reference-path length) are orientation-limit
    terminations (the env resets before the wrapper reads its measures);
  * episodes are keyed by record_idx = episode_id - 1 (sparse ids).
"""
import glob, gzip, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Repository with the committed receipts (default: two levels above this folder, i.e. the
# scripts live in <repo>/paper_evidence/<this folder>/). Override with K1_REPO.
REPO = os.environ.get("K1_REPO", os.path.abspath(os.path.join(HERE, "..", "..")))
REC = f"{REPO}/receipts"
# VLN-CE-Isaac episode file released with NaVILA-Bench (SHA-256 ceb2a2a9...). Override with
# VLNCE_DATASET; default: a copy next to these scripts.
DATASET = os.environ.get("VLNCE_DATASET", os.path.join(HERE, "vln_ce_isaac_v1.json.gz"))
TABLES = os.path.join(HERE, "tables")
os.makedirs(TABLES, exist_ok=True)

RUN_DIRS = {
    "base": f"{REC}/baseline_full_14498",
    "A": f"{REC}/sweep_measurements/stretchA_300",
    "B": f"{REC}/sweep_measurements/stretchB_300",
    "crop": f"{REC}/sweep_measurements/crop_300",
    "pad": f"{REC}/sweep_measurements/pad_300",
    "h060": f"{REC}/sweep_measurements/h060_200",
    "h078": f"{REC}/sweep_measurements/h078_200",
    "h095": f"{REC}/sweep_measurements/h095_200",
    "h110": f"{REC}/sweep_measurements/h110_200",
    "h130": f"{REC}/sweep_measurements/h130_200",
    "h150": f"{REC}/sweep_measurements/h150_200",
}
# configuration label for each run
CONFIG = {"base": "stretch", "A": "stretch", "B": "stretch", "h078": "stretch",
          "crop": "crop", "pad": "pad", "h060": "h060", "h095": "h095",
          "h110": "h110", "h130": "h130", "h150": "h150"}
REPLICATES = ["A", "B", "h078"]          # same config, same driver, same protocol
FIELDS = ["path_length", "distance_to_goal", "oracle_navigation_error",
          "success", "spl", "oracle_success"]


def _reset_distance(e):
    """Benchmark distance (measures.DistanceToGoal) evaluated at the reset pose:
    robot root at start_position + 0.55 m in z (init_state), KD-tree nearest
    reference point plus the remaining reference-path length."""
    from scipy.spatial import KDTree
    g = np.array(e["gt_locations"], float)
    p = np.array(e["start_position"], float) + np.array([0.0, 0.0, 0.55])
    dist, k = KDTree(g).query(p)
    return float(dist + np.sum(np.linalg.norm(np.diff(g[k:], axis=0), axis=1)))


def load_dataset():
    d = json.load(gzip.open(DATASET, "rt"))["episodes"]
    rows = []
    for order, e in enumerate(d):
        g = np.array(e["gt_locations"])
        L = float(np.sum(np.linalg.norm(np.diff(g, axis=0), axis=1)))
        rows.append(dict(list_order=order, episode_id=e["episode_id"],
                         record_idx=e["episode_id"] - 1,
                         scene=e["scene_id"].split("/")[1],
                         route=e["trajectory_id"],
                         geodesic=e["info"]["geodesic_distance"],
                         ref_len=L, reset_dist=_reset_distance(e)))
    return pd.DataFrame(rows).set_index("record_idx")


def load_runs():
    ds = load_dataset()
    rows = []
    for run, d in RUN_DIRS.items():
        for f in glob.glob(d + "/*.json"):
            k = int(os.path.basename(f).split(".")[0])
            r = json.load(open(f))
            r = dict(r)
            r["run"] = run
            r["record_idx"] = k
            rows.append(r)
    df = pd.DataFrame(rows)
    df = df.join(ds, on="record_idx")
    df["config"] = df["run"].map(CONFIG)
    # term_reason only exists in sweep records
    df["term_reason"] = df.get("term_reason").fillna("n/a")
    # 34 records labelled wall_timeout belong to episodes that had already finished: the
    # evaluator's SIGTERM handler stayed armed after the result was written and overwrote the
    # label (and success/spl, already 0) while the episode video was being written. Their
    # videos exist (data/relabelled_wall_timeouts.csv, from the workstation's file listing).
    # They are completed failures, not interruptions; all other fields are the final values.
    rel = pd.read_csv(os.path.join(HERE, "data", "relabelled_wall_timeouts.csv"))
    dir2run = {os.path.basename(v): k for k, v in RUN_DIRS.items()}
    rel_keys = set(zip(rel["run_dir"].map(dir2run), rel["record_idx"].astype(int)))
    df["relabelled"] = [int((r, k) in rel_keys) for r, k in zip(df["run"], df["record_idx"])]
    assert df["relabelled"].sum() == len(rel_keys) == 34
    assert (df.loc[df.relabelled == 1, "term_reason"] == "wall_timeout").all()
    df["timeout_label"] = (df["term_reason"] == "wall_timeout").astype(int)
    df["timeout"] = ((df["timeout_label"] == 1) & (df["relabelled"] == 0)).astype(int)
    df["sentinel"] = (df["distance_to_goal"] < 0).astype(int)
    # reset signature: final distance equals the benchmark distance at the reset pose
    df["reset_sig"] = (np.abs(df["distance_to_goal"] - df["reset_dist"]) < 1e-5).astype(int)
    # sweep: every sim_done record carries the signature (checked in paper_numbers.py);
    # J (no term labels): falls identified by the signature alone
    df["fall_reset"] = np.where(df["run"] == "base", df["reset_sig"],
                                ((df["term_reason"] == "sim_done") & (df["reset_sig"] == 1)).astype(int))
    df["sig"] = list(zip(*[df[c] for c in FIELDS]))
    return df


def pinned_sets(df):
    s300 = sorted(df.loc[df.run == "A", "record_idx"])
    s200 = sorted(df.loc[df.run == "h078", "record_idx"])
    return s300, s200


def wide(df, col, runs, idx):
    w = df[df.run.isin(runs)].pivot(index="record_idx", columns="run", values=col)
    return w.loc[idx, runs]
