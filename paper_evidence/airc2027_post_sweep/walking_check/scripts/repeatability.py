"""Step 2 repeatability (CPU): compare every recorded array of repeat 1 with repeat 2, bitwise, per robot and trial.

Writes ../logs/repeatability.json and ../tables/repeatability.md. Exit 0 always (differences are results, reported).
Usage: python repeatability.py [--states_dir DIR]
"""
import argparse
import json
import os

import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIALS = ["stand", "forward", "turn_left", "turn_right", "sequence"]


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    out = {"identical": True, "arrays": {}}
    for k in sorted(set(a.files) | set(b.files)):
        if k not in a.files or k not in b.files:
            out["arrays"][k] = {"identical": False, "note": "missing in one repeat"}
            out["identical"] = False
            continue
        x, y = a[k], b[k]
        same = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        item = {"identical": bool(same), "dtype": str(x.dtype), "shape": [list(x.shape), list(y.shape)]}
        if not same:
            out["identical"] = False
            n = min(len(x), len(y))
            xb, yb = x[:n].reshape(n, -1), y[:n].reshape(n, -1)
            neq = np.any(xb.view(np.uint8).reshape(n, -1) != yb.view(np.uint8).reshape(n, -1), axis=1) \
                if xb.dtype == yb.dtype else np.ones(n, bool)
            item["first_differing_row"] = int(np.argmax(neq)) if neq.any() else None
            item["rows_differing"] = int(neq.sum())
            if x.shape == y.shape and np.issubdtype(x.dtype, np.number):
                item["max_abs_diff"] = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
        out["arrays"][k] = item
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", default=os.path.join(W, "logs", "states"))
    ap.add_argument("--out_json", default=os.path.join(W, "logs", "repeatability.json"))
    ap.add_argument("--out_md", default=os.path.join(W, "tables", "repeatability.md"))
    a = ap.parse_args()
    res, lines = {}, ["| Robot | Trial | Rows (r1 / r2) | All recorded arrays bit-identical | Differences |",
                      "|---|---|---|---|---|"]
    for robot in ("E", "T"):
        for trial in TRIALS:
            p1 = os.path.join(a.states_dir, f"{robot}_{trial}_r1.npz")
            p2 = os.path.join(a.states_dir, f"{robot}_{trial}_r2.npz")
            key = f"{robot}_{trial}"
            if not (os.path.exists(p1) and os.path.exists(p2)):
                res[key] = {"identical": None, "note": "missing repeat"}
                lines.append(f"| {robot} | {trial} | — | not determined (missing repeat) | |")
                continue
            r = compare(p1, p2)
            n1, n2 = len(np.load(p1)["phase"]), len(np.load(p2)["phase"])
            r["rows"] = [n1, n2]
            res[key] = r
            diffs = "; ".join(f"{k}: first row {v.get('first_differing_row')}, max abs diff {v.get('max_abs_diff', 'n/a')}"
                              for k, v in r["arrays"].items() if not v["identical"])
            lines.append(f"| {robot} | {trial} | {n1} / {n2} | {'yes' if r['identical'] else 'NO'} | {diffs or '—'} |")
    json.dump(res, open(a.out_json, "w"), indent=1)
    open(a.out_md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
