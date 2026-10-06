"""Episode-level agreement on diverged episodes, with route-resampled intervals.

For each configuration X with reruns R on the same episodes, "diverged" episodes are those
where no two reruns have bit-identical records. On those episodes:
  kappa_X_div = mean Cohen's kappa between X and each rerun,
  kappa_R_div = mean pairwise Cohen's kappa among the reruns,
and we report kappa_X_div - kappa_R_div with a 95% route-cluster bootstrap interval.
The same is computed for the June run J against D1 and D2.
Writes tables/kappa_div_diff.json. Deterministic (fixed seed).
"""
import itertools, json, os
import numpy as np
from common import load_runs, load_dataset, pinned_sets, wide, TABLES

B = 10000
SEED = 20260930


def kappa(a, b):
    po = np.mean(a == b); p1 = a.mean(); p2 = b.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def ident_any(df, runs, idx):
    S = wide(df, "sig", runs, idx)
    out = np.zeros(len(idx), bool)
    for r1, r2 in itertools.combinations(runs, 2):
        out |= np.array([x == y for x, y in zip(S[r1], S[r2])])
    return out


def diff_stat(Y):
    """Y: n x (k+1) success matrix on diverged episodes, column 0 = X."""
    k = Y.shape[1] - 1
    kx = np.mean([kappa(Y[:, 0], Y[:, j]) for j in range(1, k + 1)])
    kr = np.mean([kappa(Y[:, i], Y[:, j]) for i, j in itertools.combinations(range(1, k + 1), 2)])
    return kx, kr, kx - kr


def main():
    df = load_runs(); ds = load_dataset(); S300, S200 = pinned_sets(df)
    rng = np.random.default_rng(SEED)
    spec = [("crop", ["A", "B"], S300), ("pad", ["A", "B"], S300)] + \
           [(h, ["A", "B", "h078"], S200) for h in ["h060", "h095", "h110", "h130", "h150"]] + \
           [("base", ["A", "B"], S300)]
    out = {}
    for X, reps, idx in spec:
        div = ~ident_any(df, reps, idx)
        idx_div = [i for i, d in zip(idx, div) if d]
        Y = wide(df, "success", [X] + reps, idx_div).values.astype(float)
        kx, kr, d = diff_stat(Y)
        r = ds.loc[idx_div, "route"].values
        groups = [np.where(r == u)[0] for u in np.unique(r)]
        bs = []
        for _ in range(B):
            sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            bs.append(diff_stat(Y[sel])[2])
        name = "J" if X == "base" else X
        out[name] = dict(n=len(idx_div), kappa_X_div=float(kx), kappa_R_div=float(kr), diff=float(d),
                         ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))])
    with open(os.path.join(TABLES, "kappa_div_diff.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(f"{k:5s} n={v['n']:3d} kX={v['kappa_X_div']:.3f} kR={v['kappa_R_div']:.3f} "
              f"diff={v['diff']:+.3f} CI=[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
