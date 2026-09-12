#!/usr/bin/env python3
"""
paper_common.py — shared, read-only data access for the AIRC-2027 paper reanalyses.

Every paper number is derived through this module so that episode indexing, scene
mapping and the success/timeout conventions are defined exactly once.

READ-ONLY CONTRACT: this module never writes to receipts/. Analyses write only under
paper_evidence/.

EPISODE INDEXING (the trap, see receipts/analysis/analyze_sweep.py):
  The eval writes each record as `{episode_id - 1}.json`. Benchmark episode_ids are
  SPARSE, so `--episode_idx=0..299` (the first 300 entries of the episode LIST) lands
  on file indices 0..488 with gaps. The file stem therefore equals `record_idx` in
  receipts/analysis/episode_index.csv, NOT the list order. Pairing and scene joins are
  done on that file stem / record_idx key, never on a positional assumption.

SCORING CONVENTIONS (unchanged from the benchmark):
  * success / oracle_success are 0.0/1.0 floats; thresholded at >= 0.5.
  * A wall_timeout episode is recorded with success=0 (intention-to-treat).
  * distance_to_goal == -1.0 is a sentinel (timeout kill before a valid final pose)
    and is EXCLUDED from NE means.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_EVIDENCE = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER_EVIDENCE)
RECEIPTS = os.path.join(ROOT, "receipts")

TABLES = os.path.join(PAPER_EVIDENCE, "tables")
FIGURES = os.path.join(PAPER_EVIDENCE, "figures")
PROVENANCE = os.path.join(PAPER_EVIDENCE, "provenance")

EPISODE_INDEX = os.path.join(RECEIPTS, "analysis", "episode_index.csv")
BASELINE_DIR = os.path.join(RECEIPTS, "baseline_full_14498")

# tag -> (relative receipts dir, pinned n_total, human label)
ARMS = {
    "stretchA": ("sweep_measurements/stretchA_300", 300, "stretch (replicate A)"),
    "stretchB": ("sweep_measurements/stretchB_300", 300, "stretch (replicate B)"),
    "crop":     ("sweep_measurements/crop_300",     300, "crop"),
    "pad":      ("sweep_measurements/pad_300",      300, "pad"),
    "h060":     ("sweep_measurements/h060_200",     200, "camera 0.60 m"),
    "h078":     ("sweep_measurements/h078_200",     200, "camera 0.78 m"),
    "h095":     ("sweep_measurements/h095_200",     200, "camera 0.95 m"),
    "h110":     ("sweep_measurements/h110_200",     200, "camera 1.10 m"),
    "h130":     ("sweep_measurements/h130_200",     200, "camera 1.30 m"),
    "h150":     ("sweep_measurements/h150_200",     200, "camera 1.50 m"),
}

TRANSFORM_ARMS = ["stretchA", "stretchB", "crop", "pad"]
WALL_CLOCK_BUDGET_S = 900.0
NE_SENTINEL = -1.0


def load_records(subdir):
    """{record_idx: record} for one arm directory (path relative to receipts/)."""
    d = subdir if os.path.isabs(subdir) else os.path.join(RECEIPTS, subdir)
    if not os.path.isdir(d):
        raise SystemExit(f"missing receipts directory: {d}")
    recs = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            recs[int(os.path.splitext(fn)[0])] = json.load(f)
    return recs


def load_arm(tag):
    """{record_idx: record} for a named arm; asserts the pinned episode count."""
    subdir, n_total, _ = ARMS[tag]
    recs = load_records(subdir)
    if len(recs) != n_total:
        raise SystemExit(f"arm {tag}: expected {n_total} records, found {len(recs)}")
    return recs


def load_baseline():
    return load_records(BASELINE_DIR)


def is_success(rec):
    return float(rec.get("success", 0.0)) >= 0.5


def is_oracle_success(rec):
    return float(rec.get("oracle_success", 0.0)) >= 0.5


def is_timeout(rec):
    return rec.get("term_reason") == "wall_timeout"


def scene_map():
    """{record_idx: scene} for all 1077 benchmark episodes."""
    out = {}
    with open(EPISODE_INDEX, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["record_idx"])] = row["scene"]
    return out


def arm_scenes(tag):
    """{record_idx: scene} restricted to the episodes an arm actually covers."""
    sm = scene_map()
    recs = load_arm(tag)
    missing = [i for i in recs if i not in sm]
    if missing:
        raise SystemExit(f"arm {tag}: {len(missing)} episodes absent from episode_index.csv")
    return {i: sm[i] for i in recs}


def paired_episodes(tag_a, tag_b):
    """Sorted record_idx list present in BOTH arms; hard-fails on any asymmetry."""
    a, b = load_arm(tag_a), load_arm(tag_b)
    ka, kb = set(a), set(b)
    if ka != kb:
        raise SystemExit(
            f"{tag_a}/{tag_b} episode sets differ: "
            f"{len(ka - kb)} only in {tag_a}, {len(kb - ka)} only in {tag_b}"
        )
    return sorted(ka), a, b


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def fmt(x, nd=4):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


if __name__ == "__main__":
    for tag in TRANSFORM_ARMS:
        r = load_arm(tag)
        n = len(r)
        k = sum(is_success(x) for x in r.values())
        t = sum(is_timeout(x) for x in r.values())
        sc = len(set(arm_scenes(tag).values()))
        print(f"{tag:10s} n={n} succ={k} SR={k/n:.4f} timeouts={t} ({t/n:.4f}) scenes={sc}")
    b = load_baseline()
    n = len(b)
    print(f"baseline   n={n} SR={sum(is_success(x) for x in b.values())/n:.4f} "
          f"OSR={sum(is_oracle_success(x) for x in b.values())/n:.4f}")
