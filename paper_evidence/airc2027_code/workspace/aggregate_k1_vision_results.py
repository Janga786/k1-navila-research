#!/usr/bin/env python3
"""Aggregate K1 vision benchmark results into a Markdown row.

Usage:
    python aggregate_k1_vision_results.py
        [--measurements-dir DIR]
        [--paper-table]   # also print the paper Table IV rows for context

Reads `<measurements-dir>/*.json` (one file per episode) and prints:
* per-metric mean (and count)
* a Markdown table row formatted to match OVERNIGHT_RESULTS.md
"""

import argparse
import glob
import json
import os
import statistics


PAPER_TABLE = [
    ("Go2 (paper, blind)",  6.03, 49.0, 36.2, 33.3),
    ("Go2 (paper, vision)", 5.49, 58.7, 50.2, 45.5),
    ("H1  (paper, blind)",  7.67, 33.3, 24.4, 21.0),
    ("H1  (paper, vision)", 5.86, 54.6, 45.3, 40.3),
]


RATE_KEYS = ("success", "spl", "oracle_success")


def aggregate(mdir, total_episodes=None):
    files = sorted(glob.glob(f"{mdir}/*.json"))
    if not files:
        return None
    keys = ["success", "spl", "distance_to_goal",
            "oracle_success", "oracle_navigation_error", "path_length"]
    acc = {k: [] for k in keys}
    # Distance keys use a -1.0 sentinel on a wall-timeout kill before the first step
    # (no valid final pose). Exclude sentinels from NE/ONE means; they still count as
    # failures for SR/OS/SPL via success/oracle_success=0. Real distances are >= 0.
    DIST_KEYS = {"distance_to_goal", "oracle_navigation_error"}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k in keys:
            if k in d:
                if k in DIST_KEYS and d[k] < 0:
                    continue   # -1.0 sentinel: no valid distance, skip from the mean
                acc[k].append(d[k])
    n_present = len(files)
    out = {
        "n_present": n_present,
        "n_total": total_episodes or n_present,
        "means": {k: (statistics.mean(v) if v else float("nan"))
                  for k, v in acc.items()},
    }
    # Honest paper-comparable rates: episodes that crashed/timed out before
    # writing a JSON are scored as failures (denominator = full R2R subset).
    if total_episodes:
        out["rates_over_total"] = {
            k: (sum(acc[k]) / total_episodes if acc[k] else float("nan"))
            for k in RATE_KEYS
        }
    return out


def fmt(x, ndp=2, percent=False):
    if x != x:  # NaN
        return "n/a"
    if percent:
        return f"{x*100:.1f}"
    return f"{x:.{ndp}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurements-dir", default=None,
        help="Defaults to most recent eval_results/k1_matterport_vision_*/measurements/",
    )
    parser.add_argument("--paper-table", action="store_true")
    parser.add_argument("--total-episodes", type=int, default=1077,
                        help="Full R2R subset size; missing episodes are scored "
                             "as failures for SR/OS/SPL (0 disables).")
    args = parser.parse_args()

    if args.measurements_dir is None:
        nb = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench/eval_results")
        candidates = sorted(glob.glob(f"{nb}/k1_matterport_vision*"))
        if not candidates:
            print(f"No k1_matterport_vision results dirs found under {nb}/")
            return
        args.measurements_dir = os.path.join(candidates[-1], "measurements")

    print(f"# Aggregation of {args.measurements_dir}")
    total = args.total_episodes or None
    r = aggregate(args.measurements_dir, total_episodes=total)
    if r is None:
        print(f"  (no JSONs found)")
        return
    m = r["means"]
    n_present, n_total = r["n_present"], r["n_total"]
    n_missing = max(n_total - n_present, 0)
    print(f"  n_present = {n_present} / n_total = {n_total} "
          f"({n_missing} missing — scored as failures for SR/OS/SPL)")
    for k, v in m.items():
        print(f"  {k}_mean(present) = {fmt(v)}")
    if "rates_over_total" in r:
        print(f"  --- honest rates over n_total={n_total} (paper-comparable) ---")
        for k, v in r["rates_over_total"].items():
            print(f"  {k}_over_total = {fmt(v, percent=True)}%")

    print()
    print("## Markdown row for OVERNIGHT_RESULTS.md:")
    print()
    if args.paper_table:
        print("| Robot | Policy | NE↓ | OS↑ | SR↑ | SPL↑ |")
        print("|-------|--------|-----|-----|-----|------|")
        for name, ne, os_, sr, spl in PAPER_TABLE:
            print(f"| {name} | — | {ne:.2f} | {os_:.1f} | {sr:.1f} | {spl:.1f} |")
    # NE is over present episodes (missing have no final pose); SR/OS/SPL over n_total.
    # NE column = FINAL distance_to_goal (standard VLN-CE NE, paper-comparable vs H1/Go2).
    # NOT oracle_navigation_error (ONE = min-dtg-over-episode), which understated K1 NE
    # because ONE <= final dtg by definition (audit rank 2).
    rates = r.get("rates_over_total", {})
    sr = rates.get("success", m["success"])
    os_ = rates.get("oracle_success", m["oracle_success"])
    spl = rates.get("spl", m["spl"])
    print(
        f"| **K1 (ours, vision)** | k1_vision (n={n_present}/{n_total}) "
        f"| {fmt(m['distance_to_goal'])} "
        f"| {fmt(os_, percent=True)} "
        f"| {fmt(sr, percent=True)} "
        f"| {fmt(spl, percent=True)} |"
    )


if __name__ == "__main__":
    main()
