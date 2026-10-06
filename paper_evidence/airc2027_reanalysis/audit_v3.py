"""Corrected configuration analysis (v3) and audit checks raised in review.

 * Paired configuration test that does not assume within-episode exchangeability:
   d_i = Y_X,i - mean_j Y_Rj,i; Delta = mean(d); route-clustered (CR1) standard error;
   Wald 95% CI and two-sided p; Holm across the seven configurations (SR and OS separately).
 * Variance check: permutation-null variance of Delta versus the empirical paired variance.
 * Planned analyses: McNemar (crop, pad vs D1; heights vs D3; crop vs pad), Fisher, and the
   Cochran-Armitage trend test over camera height.
 * Variability for comparing configurations: discordance between single runs, SD and MDE,
   with route-bootstrap intervals for the rerun SD.
 * Selection check for the diverged-episode kappa comparison.
 * Degenerate episodes shared across configurations.
Writes tables/audit_v3.json. Deterministic.
"""
import itertools, json, os
import numpy as np
from scipy import stats
from common import load_runs, load_dataset, pinned_sets, wide, TABLES

df = load_runs(); ds = load_dataset(); S300, S200 = pinned_sets(df)
route = ds["route"]
rng = np.random.default_rng(20260930)


def M(col, runs, idx):
    return wide(df, col, runs, idx).values.astype(float)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj.tolist()


def paired_cluster(Y, idx):
    """Y n x (k+1), col 0 = X. Route-clustered (CR1) paired comparison."""
    d = Y[:, 0] - Y[:, 1:].mean(1); n = len(d); delta = d.mean()
    r = route.loc[idx].values; G = np.unique(r)
    D = np.array([d[r == g].sum() for g in G]); nr = np.array([(r == g).sum() for g in G])
    u = D - nr * delta
    se = np.sqrt(len(G) / (len(G) - 1) * (u ** 2).sum()) / n
    z = delta / se if se > 0 else 0.0
    se_ep = np.sqrt(((d - delta) ** 2).sum() * n / (n - 1)) / n          # episode-level paired SE
    return dict(delta=100 * delta, se=100 * se, ci=[100 * (delta - 1.96 * se), 100 * (delta + 1.96 * se)],
                p=float(2 * stats.norm.sf(abs(z))), se_episode=100 * se_ep,
                p_episode=float(2 * stats.norm.sf(abs(delta / se_ep))) if se_ep > 0 else 1.0)


def perm_var(Y):
    """variance of Delta under within-episode exchangeability (permutation null), in pp^2"""
    n, k1 = Y.shape; k = k1 - 1; q = Y.sum(1) / k1
    # Delta = S(1+1/k)/n - T/(nk); Var(S) = sum q(1-q)
    return float(((1 + 1 / k) / n) ** 2 * (q * (1 - q)).sum()) * 1e4


spec = [("crop", ["A", "B"], S300), ("pad", ["A", "B"], S300)] + \
       [(h, ["A", "B", "h078"], S200) for h in ["h060", "h095", "h110", "h130", "h150"]]
V = {}
rows = []
for X, reps, idx in spec:
    Ys = M("success", [X] + reps, idx); Yo = M("oracle_success", [X] + reps, idx)
    sr = paired_cluster(Ys, idx); os_ = paired_cluster(Yo, idx)
    d = Ys[:, 0] - Ys[:, 1:].mean(1)
    rows.append(dict(config=X, SR=sr, OS=os_, var_ratio_perm_over_paired=perm_var(Ys) / (100 * d.std(ddof=1) / np.sqrt(len(d))) ** 2))
for key in ["SR", "OS"]:
    for r, a in zip(rows, holm([r[key]["p"] for r in rows])):
        r[key]["p_holm"] = a
    for r, a in zip(rows, holm([r[key]["p_episode"] for r in rows])):
        r[key]["p_episode_holm"] = a
V["configs"] = rows
# placebo: J treated as the configuration
Yb = M("success", ["base", "A", "B"], S300); db = Yb[:, 0] - Yb[:, 1:].mean(1)
V["placebo_J_var_ratio"] = perm_var(Yb) / (100 * db.std(ddof=1) / np.sqrt(len(db))) ** 2
V["J"] = paired_cluster(Yb, S300)
TO = M("timeout", ["A", "B"], S300).sum(1) > 0
idx_noTO = [i for i, t in zip(S300, TO) if not t]
V["J_noTO"] = paired_cluster(M("success", ["base", "A", "B"], idx_noTO), idx_noTO)
# 0.95 m robustness
Th = M("timeout", ["h095", "A", "B", "h078"], S200).sum(1) == 0
idx_h = [i for i, t in zip(S200, Th) if t]
V["h095_noTO"] = dict(n=len(idx_h), **paired_cluster(M("success", ["h095", "A", "B", "h078"], idx_h), idx_h))
V["h095_withJ"] = paired_cluster(M("success", ["h095", "base", "A", "B", "h078"], S200), S200)

# planned analyses
def mcnemar(a, b, idx):
    y = M("success", [a, b], idx); b_ = int(((y[:, 0] == 1) & (y[:, 1] == 0)).sum()); c_ = int(((y[:, 0] == 0) & (y[:, 1] == 1)).sum())
    return dict(a_only=b_, b_only=c_, p=float(stats.binomtest(b_, b_ + c_, 0.5).pvalue))
V["planned_mcnemar"] = dict(crop_D1=mcnemar("crop", "A", S300), pad_D1=mcnemar("pad", "A", S300),
                            crop_pad=mcnemar("crop", "pad", S300),
                            **{f"{h}_D3": mcnemar(h, "h078", S200) for h in ["h060", "h095", "h110", "h130", "h150"]})
hts = [("h060", 0.60), ("h078", 0.78), ("h095", 0.95), ("h110", 1.10), ("h130", 1.30), ("h150", 1.50)]
def cochran_armitage(col):
    s = np.array([M(col, [h], S200).sum() for h, _ in hts]); n = np.full(6, 200.0); x = np.array([v for _, v in hts])
    N = n.sum(); pbar = s.sum() / N
    T = (x * (s - n * pbar)).sum()
    varT = pbar * (1 - pbar) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / N)
    z = T / np.sqrt(varT); return dict(successes=s.astype(int).tolist(), z=float(z), p=float(2 * stats.norm.sf(abs(z))))
V["trend_SR"] = cochran_armitage("success"); V["trend_OS"] = cochran_armitage("oracle_success")

# variability for comparing configurations
def disc(a, b, idx):
    y = M("success", [a, b], idx); return int((y[:, 0] != y[:, 1]).sum())
V["disc"] = dict(reruns={"A-B_300": disc("A", "B", S300), "A-B_200": disc("A", "B", S200),
                         "A-h078": disc("A", "h078", S200), "B-h078": disc("B", "h078", S200)},
                 config_vs_rerun={X: {r: disc(X, r, idx) for r in reps} for X, reps, idx in spec},
                 config_vs_config_200={f"{a}-{b}": disc(a, b, S200) for a, b in itertools.combinations(
                     ["crop", "pad", "h060", "h095", "h110", "h130", "h150"], 2)})
def sd_pp(D, n): return 100 * np.sqrt(D) / n
cv300 = [np.mean(list(V["disc"]["config_vs_rerun"][X].values())) for X in ("crop", "pad")]
cv200 = [np.mean(list(V["disc"]["config_vs_rerun"][X].values())) for X in ("h060", "h095", "h110", "h130", "h150")]
V["sd_diff"] = dict(reruns_300=sd_pp(33, 300), config_300=[sd_pp(c, 300) for c in cv300],
                    reruns_200=[sd_pp(V["disc"]["reruns"][k], 200) for k in ("A-B_200", "A-h078", "B-h078")],
                    config_200=[sd_pp(c, 200) for c in cv200],
                    config_config_200=[sd_pp(c, 200) for c in V["disc"]["config_vs_config_200"].values()])
V["mde80"] = {k: (np.array(v) * 2.8016).tolist() if isinstance(v, list) else v * 2.8016 for k, v in V["sd_diff"].items()}
# route-bootstrap interval for the rerun SD of a difference (300 episodes)
Y = M("success", ["A", "B"], S300); r = route.loc[S300].values; groups = [np.where(r == u)[0] for u in np.unique(r)]
bs = []
for _ in range(10000):
    sel = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
    bs.append(100 * np.sqrt((Y[sel, 0] != Y[sel, 1]).sum()) / len(sel))
V["sd_diff_reruns_300_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# selection check for the diverged-episode kappa comparison (200-episode set)
Sg = wide(df, "sig", ["A", "B", "h078"], S200)
ident = np.zeros(len(S200), bool)
for a, b in itertools.combinations(["A", "B", "h078"], 2):
    ident |= np.array([x == y for x, y in zip(Sg[a], Sg[b])])
div = ~ident
SRdiv = {rr: 100 * float(M("success", [rr], S200)[div, 0].mean()) for rr in
         ["A", "B", "h078", "base", "h060", "h095", "h110", "h130", "h150", "crop", "pad"]}
V["diverged_200"] = dict(n=int(div.sum()), SR=SRdiv)
Sg3 = wide(df, "sig", ["A", "B"], S300); div3 = ~np.array([x == y for x, y in zip(Sg3["A"], Sg3["B"])])
V["diverged_300"] = dict(n=int(div3.sum()), SR={rr: 100 * float(M("success", [rr], S300)[div3, 0].mean()) for rr in ["A", "B", "base", "crop", "pad"]})

# degenerate episodes: records identical across all sweep runs that cover them
sweep = ["A", "B", "crop", "pad", "h060", "h078", "h095", "h110", "h130", "h150"]
S_all = wide(df, "sig", sweep, S200); L_all = wide(df, "ended_at_step", sweep, S200)
deg = []
for i, k in enumerate(S200):
    sigs = list(S_all.loc[k]); same = max(sum(1 for s in sigs if s == t) for t in sigs)
    if same >= 8:
        deg.append(dict(record_idx=int(k), episode_id=int(k) + 1, identical_runs=same, step=float(L_all.loc[k].median()),
                        term=df[(df.run == "A") & (df.record_idx == k)].term_reason.iloc[0],
                        route=int(route.loc[k])))
V["degenerate_200"] = deg
falls_deg = int(df[(df.run != "base") & (df.record_idx.isin([d["record_idx"] for d in deg])) & (df.term_reason == "sim_done")].shape[0])
V["falls_in_degenerate"] = falls_deg
with open(os.path.join(TABLES, "audit_v3.json"), "w") as f:
    json.dump(V, f, indent=1, default=float)
print(json.dumps(V, indent=1, default=float)[:12000])

# ---------------- Table III: interruptions in the 300-episode runs ----------------
T3 = {}
sw = df[df.run != "base"]
cont = sw[sw.timeout == 0]
def cont_rate(s):
    alive = cont[cont.ended_at_step > s]
    return float(alive.success.mean()) if len(alive) else 0.0
for r in ["A", "B", "crop", "pad"]:
    d = sw[(sw.run == r) & (sw.record_idx.isin(S300))]
    # true interruptions only (timeout excludes the 34 relabelled, finished episodes)
    to = d[d.timeout == 1]; stubs = to[to.sentinel == 1]; hw = to[to.sentinel == 0]
    rel = d[d.relabelled == 1]
    at_cap = hw[hw.ended_at_step >= 6001]                          # expected 0 after the correction
    near = hw[(hw.distance_to_goal >= 0) & (hw.distance_to_goal < 3.0)]
    free = len(to)                                                 # every true interruption may take either outcome
    free_label = int(d.timeout_label.sum())                       # treating all labelled records as interruptions
    succ = int(d.success.sum())
    T3[r] = dict(labelled=int(d.timeout_label.sum()), relabelled=int(len(rel)),
                 relabelled_at_cap=int((rel.ended_at_step >= 6001).sum()),
                 interruptions=int(len(to)), stubs=int(len(stubs)), handler_written=int(len(hw)),
                 at_cap=int(len(at_cap)), within3=int(len(near)),
                 extra=float(sum(cont_rate(s) for s in hw.ended_at_step)), SR=100 * succ / 300,
                 bounds=[100 * succ / 300, 100 * (succ + free) / 300],
                 bounds_label=[100 * succ / 300, 100 * (succ + free_label) / 300])
T3["crop_minus_pad"] = [T3["crop"]["bounds"][0] - T3["pad"]["bounds"][1], T3["crop"]["bounds"][1] - T3["pad"]["bounds"][0]]
T3["crop_minus_pad_label"] = [T3["crop"]["bounds_label"][0] - T3["pad"]["bounds_label"][1],
                              T3["crop"]["bounds_label"][1] - T3["pad"]["bounds_label"][0]]
V["table3"] = T3
with open(os.path.join(TABLES, "audit_v3.json"), "w") as f:
    json.dump(V, f, indent=1, default=float)
print(json.dumps(T3, indent=1))
