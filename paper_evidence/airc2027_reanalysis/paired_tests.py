"""Audit of the configuration test: is within-episode exchangeability tenable?

Reruns of one configuration share bit-identical executions on 12-17% of episodes, which a
configuration run cannot share. Under the null of "no effect on per-episode success
probability", the configuration run is then an independent draw while the reruns are
positively dependent, so runs are not exchangeable within episodes. This script
 (1) compares the exact permutation p-values used so far with a paired test that does not
     need exchangeability: d_i = Y_X,i - mean_j Y_Rj,i has mean zero under the null for every
     episode, episodes are independent, so z = sum(d) / sqrt(sum(d^2)) is asymptotically N(0,1)
     (for one rerun this is McNemar's z);
 (2) estimates the type-I error of both tests by simulation under a null model with rerun
     dependence calibrated to the observed bit-identity by episode length;
 (3) reports the variability a configuration comparison actually has (discordance between
     configuration runs and reruns) versus rerun-vs-rerun variability.
Writes tables/paired_tests.json. Deterministic.
"""
import itertools, json, os
import numpy as np
from scipy import stats
from common import load_runs, load_dataset, pinned_sets, wide, TABLES

df = load_runs(); ds = load_dataset(); S300, S200 = pinned_sets(df)


def M(col, runs, idx):
    return wide(df, col, runs, idx).values.astype(float)


def exact_shift_p(Y):
    n, k1 = Y.shape; t = Y.sum(1); p = t / k1
    pmf = np.zeros(n + 1); pmf[0] = 1.0
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi; pmf[0] *= (1 - pi)
    S_obs = Y[:, 0].sum(); mu = p.sum()
    dev = np.abs(np.arange(n + 1) - mu)
    return float(pmf[dev >= abs(S_obs - mu) - 1e-9].sum())


def sn_test(Y):
    d = Y[:, 0] - Y[:, 1:].mean(1)
    v = float((d ** 2).sum()); s = float(d.sum())
    z = s / np.sqrt(v) if v > 0 else 0.0
    return dict(delta=100 * d.mean(), se=100 * np.sqrt(v) / len(d), z=z, p=float(2 * stats.norm.sf(abs(z))),
                p_t=float(stats.ttest_1samp(d, 0).pvalue), nonzero=int((d != 0).sum()))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj.tolist()


out = {}
spec = [("crop", ["A", "B"], S300), ("pad", ["A", "B"], S300)] + \
       [(h, ["A", "B", "h078"], S200) for h in ["h060", "h095", "h110", "h130", "h150"]]
rows = []
for X, reps, idx in spec:
    Ys = M("success", [X] + reps, idx); Yo = M("oracle_success", [X] + reps, idx)
    r = dict(config=X, perm_p_SR=exact_shift_p(Ys), perm_p_OS=exact_shift_p(Yo))
    r.update({f"SR_{k}": v for k, v in sn_test(Ys).items()})
    r.update({f"OS_{k}": v for k, v in sn_test(Yo).items()})
    rows.append(r)
for key in ["perm_p_SR", "SR_p", "perm_p_OS", "OS_p"]:
    for r, a in zip(rows, holm([r[key] for r in rows])):
        r[key + "_holm"] = a
out["configs"] = rows

# 0.95 m robustness and J
Yh = M("success", ["h095", "A", "B", "h078"], S200)
Th = M("timeout", ["h095", "A", "B", "h078"], S200).sum(1) == 0
out["h095_noTO"] = dict(n=int(Th.sum()), perm_p=exact_shift_p(Yh[Th]), **sn_test(Yh[Th]))
YhJ = M("success", ["h095", "base", "A", "B", "h078"], S200)
out["h095_withJ"] = dict(perm_p=exact_shift_p(YhJ), **sn_test(YhJ))
Yb = M("success", ["base", "A", "B"], S300)
TO = M("timeout", ["A", "B"], S300).sum(1) > 0
out["J"] = dict(perm_p=exact_shift_p(Yb), **sn_test(Yb), noTO=dict(perm_p=exact_shift_p(Yb[~TO]), **sn_test(Yb[~TO])))

# variability: rerun-vs-rerun versus configuration-vs-rerun (fixed-set SD of a difference)
def disc(a, b, idx):
    y = M("success", [a, b], idx); return int((y[:, 0] != y[:, 1]).sum())
var = {}
var["rerun_pairs"] = {f"{a}-{b}": disc(a, b, idx) for a, b, idx in
                      [("A", "B", S300), ("A", "h078", S200), ("B", "h078", S200)]}
var["config_vs_rerun"] = {}
for X, reps, idx in spec:
    var["config_vs_rerun"][X] = {r: disc(X, r, idx) for r in reps}
out["discordance"] = var
# SD (pp) of the SR difference between two single runs implied by discordance D on n episodes:
# Var(SR_1 - SR_2) = D / n^2 under the null; MDE(80%, alpha .05 two-sided) = 2.80 * SD
def sd_mde(D, n):
    sd = 100 * np.sqrt(D) / n; return dict(sd=sd, mde=2.8016 * sd)
out["mde"] = dict(
    reruns_300=sd_mde(33, 300), reruns_200=sd_mde(np.mean(list(var["rerun_pairs"].values())[1:]), 200),
    config_300={X: sd_mde(np.mean(list(v.values())), 300) for X, v in var["config_vs_rerun"].items() if X in ("crop", "pad")},
    config_200={X: sd_mde(np.mean(list(v.values())), 200) for X, v in var["config_vs_rerun"].items() if X.startswith("h")})

# ---------------- simulation of type-I error under a null with rerun dependence ----------------
rng = np.random.default_rng(20260930)
runs_all = [r for r in ["A", "B", "h078", "h060", "h095", "h110", "h130", "h150", "crop", "pad", "base"]]
Yall = M("success", runs_all, S200)
p_hat = (Yall.sum(1) + 0.5) / (Yall.shape[1] + 1.0)        # shrunken per-episode success probability
L = wide(df, "ended_at_step", ["A"], S200)["A"].values
L = np.where(L < 0, 6001, L)
pi_by_bin = np.array([0.722, 0.369, 0.168, 0.097, 0.049, 0.067])   # observed rerun identity by length
bins = np.array([1000, 2000, 3000, 4000, 5000])
pi = pi_by_bin[np.searchsorted(bins, L, side="left")]


def sim(k, n_sims=4000, dependent=True):
    rej_perm = rej_sn = 0
    n = len(p_hat)
    for _ in range(n_sims):
        shared = rng.random(n) < p_hat
        R = np.empty((n, k))
        follow_prob = np.sqrt(pi) if dependent else np.zeros(n)
        for j in range(k):
            follow = rng.random(n) < follow_prob
            own = rng.random(n) < p_hat
            R[:, j] = np.where(follow, shared, own)
        X = (rng.random(n) < p_hat).astype(float)
        Y = np.column_stack([X, R])
        rej_perm += exact_shift_p(Y) < 0.05
        rej_sn += sn_test(Y)["p"] < 0.05
    return dict(perm=rej_perm / n_sims, sn=rej_sn / n_sims)


out["typeI_sim"] = dict(k3_dependent=sim(3), k2_dependent=sim(2), k3_independent=sim(3, dependent=False))

with open(os.path.join(TABLES, "paired_tests.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
