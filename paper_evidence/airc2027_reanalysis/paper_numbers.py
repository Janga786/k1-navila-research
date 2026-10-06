"""Single source for every number cited in the revised AIRC 2027 manuscript.

Reads committed per-episode receipts (k1-navila-research @ fa41994) and the
VLN-CE-Isaac episode file (sha256 ceb2a2a9...), recomputes everything, and writes
tables/paper_numbers.json. Deterministic (fixed seeds). CPU only.

Run: python3 paper_numbers.py
"""
import json, itertools, hashlib
import numpy as np, pandas as pd
from scipy import stats
from common import *

OUT = os.path.join(TABLES, "paper_numbers.json")
N = {}
df = load_runs(); S300, S200 = pinned_sets(df)
ds = load_dataset()
N["dataset_sha256"] = hashlib.sha256(open(DATASET, "rb").read()).hexdigest()


def sub(run, idx):
    return df[df.run == run].set_index("record_idx").loc[idx]


def M(col, runs, idx):
    return wide(df, col, runs, idx).values.astype(float)


def kappa(a, b):
    po = np.mean(a == b); p1 = a.mean(); p2 = b.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


# ---------------- design facts ----------------
first300 = ds.sort_values("list_order").iloc[:300]; first200 = ds.sort_values("list_order").iloc[:200]
N["design"] = dict(
    n_all=len(ds), routes_all=int(ds.route.nunique()), scenes_all=int(ds.scene.nunique()),
    routes_300=int(first300.route.nunique()), scenes_300=first300.scene.value_counts().to_dict(),
    routes_200=int(first200.route.nunique()),
    instr_per_route_200=first200.route.value_counts().value_counts().to_dict(),
    set300_is_first300=sorted(first300.index) == S300, set200_is_first200=sorted(first200.index) == S200)

# ---------------- Table I: per-run summary ----------------
rows = {}
for run in RUN_DIRS:
    x = df[df.run == run]
    sets = [("all", x)] if run != "base" else [("all", x), ("300", x[x.record_idx.isin(S300)]), ("200", x[x.record_idx.isin(S200)])]
    for tag, xs in sets:
        tr = xs.term_reason.value_counts().to_dict()
        rows[f"{run}_{tag}"] = dict(n=len(xs), succ=int(xs.success.sum()), SR=100 * xs.success.mean(),
                                    OS=100 * xs.oracle_success.mean(), OSn=int(xs.oracle_success.sum()),
                                    SPL=100 * xs.spl.mean(), NE=float(xs.loc[xs.distance_to_goal >= 0, "distance_to_goal"].mean()),
                                    stop=tr.get("stop", 0), step_cap=tr.get("step_cap", 0), sim_done=tr.get("sim_done", 0),
                                    timeouts=int(xs.timeout.sum()), falls=int(xs.fall_reset.sum()) if run != "base" else
                                    int((np.abs(xs.distance_to_goal - xs.reset_dist) < 1e-5).sum()),
                                    sentinels=int(xs.sentinel.sum()))
for r in ["A", "B"]:
    xs = sub(r, S200); rows[f"{r}_200"] = dict(n=200, succ=int(xs.success.sum()), SR=100 * xs.success.mean(),
                                               timeouts=int(xs.timeout.sum()), falls=int(xs.fall_reset.sum()))
N["runs"] = rows

# ---------------- Repeatability ----------------
def pair(r1, r2, idx):
    a, b = sub(r1, idx), sub(r2, idx)
    s1, s2 = a.success.values, b.success.values
    ident = np.array([x == y for x, y in zip(a.sig, b.sig)])
    to = (a.timeout.values == 1) | (b.timeout.values == 1)
    both_to = (a.timeout.values == 1) & (b.timeout.values == 1)
    n11 = int(((s1 == 1) & (s2 == 1)).sum()); n10 = int(((s1 == 1) & (s2 == 0)).sum())
    n01 = int(((s1 == 0) & (s2 == 1)).sum()); n00 = int(((s1 == 0) & (s2 == 0)).sum())
    return dict(n=len(idx), n11=n11, n10=n10, n01=n01, n00=n00, disc=n10 + n01, kappa=kappa(s1, s2),
                shared_succ_frac=n11 / (n11 + n10 + n01), ident=int(ident.sum()), ident_pct=100 * ident.mean(),
                ident_shared_succ=int((ident & (s1 == 1) & (s2 == 1)).sum()),
                to_any=int(to.sum()), to_both=int(both_to.sum()), n_noTO=int((~to).sum()),
                disc_noTO=int(((s1 != s2) & ~to).sum()), disc_withTO=int(((s1 != s2) & to).sum()),
                mcnemar_p=stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
P = {}
for r1, r2, idx, tag in [("A", "B", S300, "300"), ("A", "B", S200, "200"), ("A", "h078", S200, "200"),
                         ("B", "h078", S200, "200"), ("base", "A", S300, "300"), ("base", "B", S300, "300"),
                         ("base", "h078", S200, "200"), ("crop", "pad", S300, "300")]:
    P[f"{r1}-{r2}_{tag}"] = pair(r1, r2, idx)
# identity between configurations (min/max over different-config pairs)
diff_cfg = []
allruns = [r for r in RUN_DIRS]
for r1, r2 in itertools.combinations(allruns, 2):
    if CONFIG[r1] == CONFIG[r2]:
        continue
    idx = S300 if (r1 in ["base", "A", "B", "crop", "pad"] and r2 in ["base", "A", "B", "crop", "pad"]) else S200
    diff_cfg.append(pair(r1, r2, idx)["ident_pct"])
P["diff_config_ident_pct_range"] = [min(diff_cfg), max(diff_cfg)]
N["pairs"] = P

# three-run view on 200
Y = M("success", ["A", "B", "h078"], S200); T3 = M("timeout", ["A", "B", "h078"], S200); s = Y.sum(1)
N["three_runs_200"] = dict(succ=Y.sum(0).astype(int).tolist(), all3=int((s == 3).sum()), any=int((s >= 1).sum()),
                           varies=int(((s > 0) & (s < 3)).sum()), n_noTO=int((T3.sum(1) == 0).sum()),
                           varies_noTO=int((((s > 0) & (s < 3)) & (T3.sum(1) == 0)).sum()))

# identity by episode length (non-interrupted pairs)
def pair_rows(r1, r2, idx):
    a, b = sub(r1, idx), sub(r2, idx)
    ident = np.array([x == y for x, y in zip(a.sig, b.sig)])
    la = a.ended_at_step.values if r1 != "base" else b.ended_at_step.values
    lb = b.ended_at_step.values
    ta = (a.timeout.values == 1) if r1 != "base" else np.zeros(len(idx), bool)
    keep = ~(ta | (b.timeout.values == 1)) & (la >= 0) & (lb >= 0)
    return pd.DataFrame(dict(ident=ident[keep], L=np.minimum(la, lb)[keep],
                             dA=a.distance_to_goal.values[keep], dB=b.distance_to_goal.values[keep],
                             fA=a.fall_reset.values[keep] if r1 != "base" else np.zeros(keep.sum()),
                             fB=b.fall_reset.values[keep]))
bins = [0, 1000, 2000, 3000, 4000, 5000, 6002]
sw_pairs = pd.concat([pair_rows("A", "B", S300), pair_rows("A", "h078", S200), pair_rows("B", "h078", S200)])
jn_pairs = pd.concat([pair_rows("base", "A", S300), pair_rows("base", "B", S300), pair_rows("base", "h078", S200)])
def by_len(d):
    g = d.assign(b=pd.cut(d.L, bins)).groupby("b", observed=False).ident
    return dict(pct=(100 * g.mean()).round(2).tolist(), n=g.count().tolist(), k=g.sum().astype(int).tolist())
N["ident_by_length"] = dict(bins=bins, sweep=by_len(sw_pairs), june=by_len(jn_pairs),
                            june_ident_long=int(((jn_pairs.ident) & (jn_pairs.L > 2000)).sum()),
                            june_ident_maxL=float(jn_pairs[jn_pairs.ident].L.max()))
ab = pair_rows("A", "B", S300)
nd = ab[(~ab.ident) & (ab.dA >= 0) & (ab.dB >= 0) & (ab.fA == 0) & (ab.fB == 0)]
dd = np.abs(nd.dA - nd.dB)
N["divergence_magnitude_AB"] = dict(n=len(nd), median=float(np.median(dd)), q25=float(np.percentile(dd, 25)),
                                    q75=float(np.percentile(dd, 75)), frac_gt1m=float(np.mean(dd > 1)))

# variance decomposition
RNG = np.random.default_rng(20260930)
route = ds["route"]
def within_share(Y):
    k = Y.shape[1]; m = Y.mean(1); w = (k / (k - 1)) * m * (1 - m); p = Y.mean()
    return w.mean(), p * (1 - p), w.mean() / (p * (1 - p))
VD = {}
for name, runs, idx in [("AB_300", ["A", "B"], S300), ("ABh078_200", ["A", "B", "h078"], S200)]:
    Y = M("success", runs, idx); w, t, f = within_share(Y); n = len(idx)
    r = route.loc[idx].values; groups = [np.where(r == u)[0] for u in np.unique(r)]
    bs = []
    for _ in range(4000):
        sel = np.concatenate([groups[j] for j in RNG.integers(0, len(groups), len(groups))])
        bs.append(within_share(Y[sel])[2])
    VD[name] = dict(within=w, total=t, share=f, share_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    sd_single_pp=100 * np.sqrt(w / n), sd_diff_pp=100 * np.sqrt(2 * w / n),
                    sd_binom_pp=100 * np.sqrt(t / n))
N["variance"] = VD

# June vs sweep reruns (exchangeability)
def perm(Y, nperm=20000, seed=11):
    rng = np.random.default_rng(seed); n, k = Y.shape
    def st(Y):
        a = Y[:, 0].mean() - Y[:, 1:].mean()
        dX = np.mean([np.sum(Y[:, 0] != Y[:, j]) for j in range(1, k)])
        dR = np.mean([np.sum(Y[:, a_] != Y[:, b_]) for a_, b_ in itertools.combinations(range(1, k), 2)])
        return a, dX - dR, dX, dR
    o = st(Y); ca = cT = 0
    for _ in range(nperm):
        Pm = np.argsort(rng.random((n, k)), axis=1); Yp = np.take_along_axis(Y, Pm, axis=1)
        a, t, _, _ = st(Yp)
        ca += abs(a) >= abs(o[0]) - 1e-12; cT += t >= o[1] - 1e-12
    return dict(n=n, dSR_pp=100 * o[0], p_agg=(ca + 1) / (nperm + 1), disc_X=o[2], disc_rep=o[3], T=o[1],
                p_exch=(cT + 1) / (nperm + 1))
TOab = M("timeout", ["A", "B"], S300).sum(1) > 0
Yb = M("success", ["base", "A", "B"], S300)
N["june_vs_reruns"] = dict(all300=perm(Yb), noTO300=perm(Yb[~TOab]),
                           succ_in_TO_eps=dict(zip(["base", "A", "B"], Yb[TOab].sum(0).astype(int).tolist())),
                           n_TO_eps=int(TOab.sum()),
                           stub_eps_where_base_succeeded=int(sum(1 for i, rid in enumerate(S300)
                               if TOab[i] and Yb[i, 0] == 1 and (sub("A", [rid]).ended_at_step.values[0] < 0 or sub("B", [rid]).ended_at_step.values[0] < 0))))

# ---------------- configuration comparisons ----------------
spec = [("crop", ["A", "B"], S300), ("pad", ["A", "B"], S300)] + \
       [(h, ["A", "B", "h078"], S200) for h in ["h060", "h095", "h110", "h130", "h150"]]
C = []
for i, (X, reps, idx) in enumerate(spec):
    Ys = M("success", [X] + reps, idx); Yo = M("oracle_success", [X] + reps, idx)
    ps = perm(Ys, seed=100 + i); po = perm(Yo, seed=200 + i)
    r = route.loc[idx].values; groups = [np.where(r == u)[0] for u in np.unique(r)]
    d = Ys[:, 0] - Ys[:, 1:].mean(1); bs = []
    rng = np.random.default_rng(300 + i)
    for _ in range(10000):
        sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]); bs.append(d[sel].mean())
    C.append(dict(config=X, n=len(idx), succ=int(Ys[:, 0].sum()), succ_reps=Ys[:, 1:].sum(0).astype(int).tolist(),
                  SR=100 * Ys[:, 0].mean(), SR_reps=100 * Ys[:, 1:].mean(), dSR=ps["dSR_pp"], p_dSR=ps["p_agg"],
                  ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)],
                  OS=100 * Yo[:, 0].mean(), OS_reps=100 * Yo[:, 1:].mean(), dOS=po["dSR_pp"], p_dOS=po["p_agg"],
                  kappa_X=float(np.mean([kappa(Ys[:, 0], Ys[:, j]) for j in range(1, len(reps) + 1)])),
                  kappa_reps=float(np.mean([kappa(Ys[:, a], Ys[:, b]) for a, b in itertools.combinations(range(1, len(reps) + 1), 2)])),
                  disc_X=ps["disc_X"], disc_reps=ps["disc_rep"], p_exch=ps["p_exch"]))
for key in ["p_dSR", "p_dOS", "p_exch"]:
    adj = holm([c[key] for c in C])
    for c, a in zip(C, adj):
        c[key + "_holm"] = float(a)
N["configs"] = C
# sensitivity: h095 with June run added to the reference
N["h095_with_june"] = perm(M("success", ["h095", "base", "A", "B", "h078"], S200), seed=999)
# pre-registered unpaired contrasts (Fisher) for reference
def fisher(a, b, n1, n2):
    return stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]
N["prereg_unpaired"] = dict(crop_vs_pad=fisher(50, 40, 300, 300), h095_vs_h078=fisher(37, 25, 200, 200),
                            A_vs_B=fisher(45, 48, 300, 300))
# crop reach/stop descriptive
N["reach_stop"] = {r: dict(OSn=int(sub(r, S300).oracle_success.sum()), succ=int(sub(r, S300).success.sum()))
                   for r in ["A", "B", "crop", "pad"]}
# pilot vs sweep on the first ten dataset entries
first10 = sorted(ds[ds.list_order < 10].index)
N["first10"] = {r: int(sub(r, first10).success.sum()) for r in ["base", "A", "B", "crop", "pad"]}

# ---------------- interruptions ----------------
sw = df[df.run != "base"]
to = sw[sw.timeout == 1]
comp = sw[sw.timeout == 0]
def cont_rate(k):
    alive = comp[comp.ended_at_step > k]
    return float((alive.success == 1).mean()), int(len(alive))
TOinfo = {}
for r, g in to.groupby("run"):
    ks = g.ended_at_step.values
    handler = ks[ks >= 0]
    exp = sum(cont_rate(k)[0] for k in handler if k < 6000)
    TOinfo[r] = dict(total=len(g), stubs=int((ks < 0).sum()), at_cap=int((ks >= 6000).sum()),
                     open=int(((ks >= 0) & (ks < 6000)).sum()), median_step=float(np.median(handler)),
                     exp_extra_succ=exp)
N["timeouts"] = dict(per_run=TOinfo, total=int(len(to)), stubs=int((to.ended_at_step < 0).sum()),
                     at_cap=int((to.ended_at_step >= 6000).sum()),
                     median_kill_step_all=float(np.median(to.ended_at_step[to.ended_at_step >= 0])),
                     cont_rate={k: cont_rate(k) for k in [5000, 5500, 5700, 5900]},
                     succ_end_median=float(sw[sw.success == 1].ended_at_step.median()),
                     succ_after_5500=int((sw[sw.success == 1].ended_at_step > 5500).sum()),
                     succ_total=int((sw.success == 1).sum()),
                     crop_pad_fisher=stats.fisher_exact([[6, 294], [37, 263]])[1])
w = wide(df, "timeout", ["A", "B", "crop", "pad"], S300)
scn = ds.loc[S300, "scene"]
N["timeouts"]["by_scene_EU6"] = {r: int(w[r][scn == "EU6Fwq7SyZv"].sum()) for r in w.columns}
N["timeouts"]["outside_EU6"] = {r: int(w[r][scn != "EU6Fwq7SyZv"].sum()) for r in w.columns}
N["timeouts"]["crop_pad_overlap"] = dict(both=int(((w["crop"] == 1) & (w["pad"] == 1)).sum()),
                                         crop_only=int(((w["crop"] == 1) & (w["pad"] == 0)).sum()),
                                         pad_only=int(((w["crop"] == 0) & (w["pad"] == 1)).sum()),
                                         neither=int(((w["crop"] == 0) & (w["pad"] == 0)).sum()))
# bounds (worst case), classic and refined (handler records at the cap are determined failures)
def bounds(run, idx):
    xs = sub(run, idx); s_ = int(xs.success.sum()); t_ = int(xs.timeout.sum())
    cap = int(((xs.timeout == 1) & (xs.ended_at_step >= 6000)).sum())
    n = len(idx)
    return dict(lo=100 * s_ / n, hi=100 * (s_ + t_) / n, hi_refined=100 * (s_ + t_ - cap) / n)
B_ = {r: bounds(r, S300) for r in ["A", "B", "crop", "pad"]}
N["bounds"] = dict(per_run=B_,
                   crop_minus_pad=[B_["crop"]["lo"] - B_["pad"]["hi"], B_["crop"]["hi"] - B_["pad"]["lo"]],
                   crop_minus_pad_refined=[B_["crop"]["lo"] - B_["pad"]["hi_refined"], B_["crop"]["hi_refined"] - B_["pad"]["lo"]])
# per-scene SR range across the four transform runs
ps_ = []
for r in ["A", "B", "crop", "pad"]:
    g = sub(r, S300).groupby("scene").success.mean() * 100; ps_ += g.tolist()
N["per_scene_SR_range"] = [min(ps_), max(ps_)]
N["Z6MF_SR"] = {r: float(sub(r, S300)[sub(r, S300).scene == "Z6MFQCViBuw"].success.mean() * 100) for r in ["A", "B", "crop", "pad"]}

json.dump(N, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(N, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:20000])
