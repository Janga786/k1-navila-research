"""Additions/corrections after independent verification and review (v2).

 * exact permutation p-values (within-episode exchangeability; Poisson-binomial) for
   SR and OS shifts, Holm within each 7-test family;
 * agreement (kappa) stratified by whether the reruns executed identically;
 * reliability by episode segment; falls = sim_done (all carry the reset signature);
 * interruption records within 3 m; 0.95 m robustness; J agreement.
Writes tables/paper_numbers_v2.json. Deterministic.
"""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats
from common import *

OUT = os.path.join(TABLES, "paper_numbers_v2.json")
V = {}
df = load_runs(); S300, S200 = pinned_sets(df); ds = load_dataset()
route = ds["route"]


def sub(r, idx):
    return df[df.run == r].set_index("record_idx").loc[idx]


def M(col, runs, idx):
    return wide(df, col, runs, idx).values.astype(float)


def kappa(a, b):
    po = np.mean(a == b); p1 = a.mean(); p2 = b.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def exact_shift_p(Y):
    """Y: n x (k+1), column 0 = configuration run. Under within-episode exchangeability the
    configuration's outcome in episode i is Bernoulli(t_i/(k+1)); Delta is affine in
    S = sum_i Y_i0, so its two-sided p-value follows from the Poisson-binomial law of S."""
    n, k1 = Y.shape; t = Y.sum(1); p = t / k1
    pmf = np.zeros(n + 1); pmf[0] = 1.0
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi; pmf[0] *= (1 - pi)
    S_obs = Y[:, 0].sum(); mu = p.sum()
    dev = np.abs(np.arange(n + 1) - mu)
    return float(pmf[dev >= abs(S_obs - mu) - 1e-9].sum())


# ---------- falls, step caps ----------
sw = df[df.run != "base"]
V["all_sim_done_have_reset_sig"] = bool((sw[sw.term_reason == "sim_done"].reset_sig == 1).all())
V["non_sim_done_with_sig"] = int(((sw.term_reason != "sim_done") & (sw.reset_sig == 1)).sum())
V["falls"] = {r: int(df[df.run == r].fall_reset.sum()) for r in RUN_DIRS}
V["falls"]["base_300"] = int(sub("base", S300).fall_reset.sum())
V["step_cap_pct"] = {r: 100 * float((df[df.run == r].term_reason == "step_cap").mean()) for r in RUN_DIRS if r != "base"}

# ---------- divergence magnitude (non-identical D1-D2, no interruption, no fall, no sentinel) ----------
a, b = sub("A", S300), sub("B", S300)
ident = np.array([x == y for x, y in zip(a.sig, b.sig)])
ok = (~ident) & (a.timeout.values == 0) & (b.timeout.values == 0) & (a.fall_reset.values == 0) & \
     (b.fall_reset.values == 0) & (a.distance_to_goal.values >= 0) & (b.distance_to_goal.values >= 0)
dd = np.abs(a.distance_to_goal.values - b.distance_to_goal.values)[ok]
V["divergence"] = dict(n=int(ok.sum()), median=float(np.median(dd)), q25=float(np.percentile(dd, 25)),
                       q75=float(np.percentile(dd, 75)), frac_gt1=float(np.mean(dd > 1)))

# ---------- reliability by segment ----------
order = ds.loc[S300, "list_order"].values
Y = M("success", ["A", "B"], S300); Yj = M("success", ["base", "A"], S300)
V["kappa_segments"] = dict(AB_first233=kappa(Y[order < 233, 0], Y[order < 233, 1]),
                           AB_last67=kappa(Y[order >= 233, 0], Y[order >= 233, 1]),
                           JA_first233=kappa(Yj[order < 233, 0], Yj[order < 233, 1]),
                           JA_last67=kappa(Yj[order >= 233, 0], Yj[order >= 233, 1]))


# ---------- identity masks ----------
def ident_any(runs, idx):
    S = wide(df, "sig", runs, idx)
    out = np.zeros(len(idx), bool)
    for r1, r2 in itertools.combinations(runs, 2):
        out |= np.array([x == y for x, y in zip(S[r1], S[r2])])
    return out


# ---------- configuration comparisons ----------
spec = [("crop", ["A", "B"], S300), ("pad", ["A", "B"], S300)] + \
       [(h, ["A", "B", "h078"], S200) for h in ["h060", "h095", "h110", "h130", "h150"]]
rng = np.random.default_rng(20260930)
C = []
for X, reps, idx in spec:
    Ys = M("success", [X] + reps, idx); Yo = M("oracle_success", [X] + reps, idx)
    k = len(reps)
    d = Ys[:, 0] - Ys[:, 1:].mean(1)
    r = route.loc[idx].values; groups = [np.where(r == u)[0] for u in np.unique(r)]
    bs = [d[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(10000)]
    idn = ident_any(reps, idx)          # episodes where some rerun pair executed identically
    kX_all = np.mean([kappa(Ys[:, 0], Ys[:, j]) for j in range(1, k + 1)])
    kR_all = np.mean([kappa(Ys[:, i], Ys[:, j]) for i, j in itertools.combinations(range(1, k + 1), 2)])
    kX_div = np.mean([kappa(Ys[~idn, 0], Ys[~idn, j]) for j in range(1, k + 1)])
    kR_div = np.mean([kappa(Ys[~idn, i], Ys[~idn, j]) for i, j in itertools.combinations(range(1, k + 1), 2)])
    mcn = []
    for j in range(1, k + 1):
        b_ = int(((Ys[:, 0] == 1) & (Ys[:, j] == 0)).sum()); c_ = int(((Ys[:, 0] == 0) & (Ys[:, j] == 1)).sum())
        mcn.append(dict(x_only=b_, r_only=c_, p=float(stats.binomtest(b_, b_ + c_, 0.5).pvalue)))
    C.append(dict(config=X, n=len(idx), reps=reps, succ=int(Ys[:, 0].sum()), succ_reps=Ys[:, 1:].sum(0).astype(int).tolist(),
                  SR=100 * Ys[:, 0].mean(), dSR=100 * d.mean(), p_dSR=exact_shift_p(Ys),
                  ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)],
                  OS=100 * Yo[:, 0].mean(), OS_reps=100 * Yo[:, 1:].mean(), dOS=100 * (Yo[:, 0].mean() - Yo[:, 1:].mean()),
                  p_dOS=exact_shift_p(Yo), kappa_X=kX_all, kappa_R=kR_all,
                  n_diverged=int((~idn).sum()), kappa_X_div=kX_div, kappa_R_div=kR_div, mcnemar=mcn))
for key in ["p_dSR", "p_dOS"]:
    for c, a_ in zip(C, holm([c[key] for c in C])):
        c[key + "_holm"] = float(a_)
V["configs"] = C
# kappa uncertainty (route bootstrap) for reruns and for crop / 0.95 m
def kboot(runs, idx, B=4000):
    Y = M("success", runs, idx); r = route.loc[idx].values; groups = [np.where(r == u)[0] for u in np.unique(r)]
    out = []
    for _ in range(B):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        out.append(kappa(Y[sel, 0], Y[sel, 1]))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]
V["kappa_ci"] = dict(AB=kboot(["A", "B"], S300), cropA=kboot(["crop", "A"], S300), h095_h078=kboot(["h095", "h078"], S200))

# kappa among non-default configuration runs, and with J
runs200 = ["h060", "h095", "h110", "h130", "h150", "crop", "pad"]
K = {}
for r1, r2 in itertools.combinations(runs200, 2):
    y = M("success", [r1, r2], S200); K[f"{r1}-{r2}"] = kappa(y[:, 0], y[:, 1])
V["kappa_between_configs_200"] = dict(min=min(K.values()), max=max(K.values()), pairs=K)
hts = {"h060": 0.60, "h095": 0.95, "h110": 1.10, "h130": 1.30, "h150": 1.50}
kk = [c["kappa_X"] for c in C if c["config"] in hts]; dh = [abs(hts[c["config"]] - 0.78) for c in C if c["config"] in hts]
V["spearman_dh_kappa"] = [float(x) for x in stats.spearmanr(dh, kk)]
# J vs reruns
TO = M("timeout", ["A", "B"], S300).sum(1) > 0
Yb = M("success", ["base", "A", "B"], S300)
V["J"] = dict(kappa_all=float(np.mean([kappa(Yb[:, 0], Yb[:, j]) for j in (1, 2)])),
              kappa_noTO=float(np.mean([kappa(Yb[~TO, 0], Yb[~TO, j]) for j in (1, 2)])),
              p_dSR=exact_shift_p(Yb), p_dSR_noTO=exact_shift_p(Yb[~TO]),
              dSR_noTO=100 * float(Yb[~TO, 0].mean() - Yb[~TO, 1:].mean()),
              succ_TO=Yb[TO].sum(0).astype(int).tolist())
idnJ = ident_any(["A", "B"], S300)
V["J"]["kappa_div"] = float(np.mean([kappa(Yb[~idnJ, 0], Yb[~idnJ, j]) for j in (1, 2)]))

# 0.95 m robustness
Yh = M("success", ["h095", "A", "B", "h078"], S200); Th = M("timeout", ["h095", "A", "B", "h078"], S200).sum(1) == 0
V["h095"] = dict(n_noTO=int(Th.sum()), dSR_noTO=100 * float(Yh[Th, 0].mean() - Yh[Th, 1:].mean()),
                 p_noTO=exact_shift_p(Yh[Th]),
                 with_J=dict(dSR=100 * float(M("success", ["h095", "base", "A", "B", "h078"], S200)[:, 0].mean()
                                             - M("success", ["h095", "base", "A", "B", "h078"], S200)[:, 1:].mean()),
                             p=exact_shift_p(M("success", ["h095", "base", "A", "B", "h078"], S200))),
                 SR_all_default_200={r: 100 * float(sub(r, S200).success.mean()) for r in ["base", "A", "B", "h078"]},
                 neighbors={"h078": 100 * float(sub("h078", S200).success.mean()), "h110": 100 * float(sub("h110", S200).success.mean())},
                 runs_as_units_min_p_with_J=1 / 5, runs_as_units_min_p=1 / 4)

# interruptions: handler records within 3 m, and worst-case variants for crop - pad
to = sw[(sw.timeout == 1) & (sw.ended_at_step >= 0)]
near = to[(to.distance_to_goal >= 0) & (to.distance_to_goal < 3.0)]
V["interrupt_near_goal"] = dict(total=int(len(near)), at_cap=int((near.ended_at_step >= 6000).sum()),
                                by_run=near.groupby("run").size().to_dict())
comp = sw[sw.timeout == 0]
late = comp[comp.ended_at_step > 5500]
V["late_uninterrupted"] = dict(n=int(len(late)), success=int(late.success.sum()),
                               cap_within3=int(((late.term_reason == "step_cap") & (late.distance_to_goal < 3)).sum()))
s_crop, s_pad = 50, 40
V["crop_pad_if_near_goal_succeed"] = 100 * ((s_crop + V["interrupt_near_goal"]["by_run"].get("crop", 0)) -
                                            (s_pad + V["interrupt_near_goal"]["by_run"].get("pad", 0))) / 300
V["interrupt_after_4800"] = int((to.ended_at_step >= 4800).sum())
V["interrupt_min_step"] = float(to.ended_at_step.min())
# paired McNemar for crop vs pad interruptions
w = wide(df, "timeout", ["crop", "pad"], S300)
b_ = int(((w["crop"] == 1) & (w["pad"] == 0)).sum()); c_ = int(((w["crop"] == 0) & (w["pad"] == 1)).sum())
V["crop_pad_interrupt_mcnemar"] = dict(crop_only=b_, pad_only=c_, p=float(stats.binomtest(b_, b_ + c_, 0.5).pvalue))
# odd-one-out (descriptive)
json.dump(V, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:9000])
