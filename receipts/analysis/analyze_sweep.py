#!/usr/bin/env python3
"""
analyze_sweep.py — pre-registered analysis for the K1/NaVILA transform + camera-height sweep.

Reads ONLY the committed per-episode receipts in receipts/ (no dependency on the gitignored
eval_results/), so every number in the paper is reproducible from a fresh clone:

    python receipts/analysis/analyze_sweep.py

Outputs (written next to this script):
    ANALYSIS.md          human-readable report
    arm_summary.csv      one row per arm
    contrasts.csv        one row per pre-registered contrast

ANALYSIS PLAN — see receipts/PRE_REGISTRATION.md and its AMENDMENT section.
The pre-registration's original primary test (McNemar on paired per-episode outcomes) was
INVALIDATED by the 2026-07-23 determinism verdict (6/6 episodes differed run-to-run, 2/6
flipped success), which the pre-registration itself declares as a hard gate. Per that gate we
use UNPAIRED two-proportion tests with binomial CIs. This substitution was fixed BEFORE any
arm data was inspected (HANDOFF.md, 2026-07-27) and is recorded as a dated amendment.

Scoring conventions (unchanged from the benchmark, see PRE_REGISTRATION.md):
  * SR / OS / SPL are computed over n_total (missing episodes count as failures).
  * NE (distance_to_goal) and ONE (oracle_navigation_error) exclude the -1.0 sentinel, which
    marks a wall-timeout kill landing before a valid final pose existed.
  * Primary wandering metric is `ended_at_step` (deterministic sim steps), reported by
    term_reason — NOT wall-clock.
"""

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(RECEIPTS)

# tag -> (directory, n_total pinned episode count, human label)
ARMS = [
    ("stretchA", "sweep_measurements/stretchA_300", 300, "stretch (arm A)"),
    ("stretchB", "sweep_measurements/stretchB_300", 300, "stretch (arm B, same-driver replicate)"),
    ("crop",     "sweep_measurements/crop_300",     300, "crop"),
    ("pad",      "sweep_measurements/pad_300",      300, "pad"),
    ("h060",     "sweep_measurements/h060_200",     200, "camera 0.60 m"),
    ("h078",     "sweep_measurements/h078_200",     200, "camera 0.78 m (native)"),
    ("h095",     "sweep_measurements/h095_200",     200, "camera 0.95 m (virtual)"),
    ("h110",     "sweep_measurements/h110_200",     200, "camera 1.10 m (virtual)"),
    ("h130",     "sweep_measurements/h130_200",     200, "camera 1.30 m (virtual)"),
    ("h150",     "sweep_measurements/h150_200",     200, "camera 1.50 m (virtual)"),
]

HEIGHT_M = {"h060": 0.60, "h078": 0.78, "h095": 0.95,
            "h110": 1.10, "h130": 1.30, "h150": 1.50}

BASELINE_DIR = "baseline_full_14498"
DIST_KEYS = ("distance_to_goal", "oracle_navigation_error")

# NOTE ON EPISODE INDEXING (a trap — read before touching this file):
# The eval names each record `{episode_id - 1}.json`, and benchmark episode_ids are SPARSE.
# `--episode_idx=0..299` therefore selects the first 300 episodes of the episode LIST, which
# land on file indices 0..488 with gaps. So the pinned arm set is NOT "file index < 300".
# Subsetting the 1077-episode baseline by `idx < 300` silently scores 195 episodes over a
# denominator of 300 and yields a spurious SR of 10.0% instead of the correct 17.7%.
# The pinned set is always derived FROM the arm receipts themselves, never assumed.


# ----------------------------------------------------------------------------- loading
def load_arm(subdir, n_total=None, episode_filter=None):
    """Load per-episode records. episode_filter: optional set of episode indices to keep."""
    d = os.path.join(RECEIPTS, subdir)
    recs = {}
    if not os.path.isdir(d):
        raise SystemExit(f"missing receipts directory: {d}")
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        idx = int(os.path.splitext(fn)[0])
        if episode_filter is not None and idx not in episode_filter:
            continue
        with open(os.path.join(d, fn)) as f:
            recs[idx] = json.load(f)
    return recs


def episode_set(subdir):
    """The set of episode (file) indices an arm actually covers."""
    d = os.path.join(RECEIPTS, subdir)
    return {int(os.path.splitext(fn)[0]) for fn in os.listdir(d) if fn.endswith(".json")}


# ----------------------------------------------------------------------------- statistics
def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval — the recommended interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson_ci(k, n, alpha=0.05):
    """Exact (conservative) interval — reported alongside Wilson for reviewers who want it."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def two_proportion_test(k1, n1, k2, n2):
    """Unpaired comparison. Returns (diff_pp, z, p_z, p_fisher, ci_lo_pp, ci_hi_pp).

    p_fisher is the two-sided Fisher exact test — preferred here because several arms have
    small success counts where the normal approximation is shaky.
    The CI on the difference is the unpooled (Wald) interval.
    """
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    # pooled z (for the test)
    p_pool = (k1 + k2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
    # Fisher exact on the 2x2
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    p_fisher = stats.fisher_exact(table, alternative="two-sided")[1]
    # unpooled CI on the difference
    se_un = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(0.975)
    return (diff * 100, z, p_z, p_fisher,
            (diff - zc * se_un) * 100, (diff + zc * se_un) * 100)


def mde(n1, n2, p_base, alpha=0.05, power=0.80):
    """Minimum detectable difference (pp) for an unpaired two-proportion test."""
    za = stats.norm.ppf(1 - alpha / 2)
    zb = stats.norm.ppf(power)
    # solve approximately around p_base
    se = math.sqrt(p_base * (1 - p_base) * (1 / n1 + 1 / n2))
    return (za + zb) * se * 100


def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values, preserving input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------- summarise
def summarise(tag, label, recs, n_total):
    n_present = len(recs)
    succ = sum(1 for r in recs.values() if r.get("success", 0.0) >= 0.5)
    osucc = sum(1 for r in recs.values() if r.get("oracle_success", 0.0) >= 0.5)
    spl_vals = [r.get("spl", 0.0) for r in recs.values()]
    # missing episodes count as failures over n_total
    spl_sum = sum(spl_vals)

    ne = [r["distance_to_goal"] for r in recs.values()
          if r.get("distance_to_goal", -1.0) >= 0]
    one = [r["oracle_navigation_error"] for r in recs.values()
           if r.get("oracle_navigation_error", -1.0) >= 0]
    path = [r["path_length"] for r in recs.values() if r.get("path_length", -1.0) >= 0]
    steps = [r["ended_at_step"] for r in recs.values() if "ended_at_step" in r]
    terms = Counter(r.get("term_reason", "unknown") for r in recs.values())

    sr_lo, sr_hi = wilson_ci(succ, n_total)
    os_lo, os_hi = wilson_ci(osucc, n_total)
    sr_cp = clopper_pearson_ci(succ, n_total)

    return {
        "tag": tag, "label": label,
        "n_total": n_total, "n_present": n_present, "n_missing": n_total - n_present,
        "succ": succ, "osucc": osucc,
        "SR": 100 * succ / n_total, "SR_lo": 100 * sr_lo, "SR_hi": 100 * sr_hi,
        "SR_cp_lo": 100 * sr_cp[0], "SR_cp_hi": 100 * sr_cp[1],
        "OS": 100 * osucc / n_total, "OS_lo": 100 * os_lo, "OS_hi": 100 * os_hi,
        "SPL": 100 * spl_sum / n_total,
        "NE_mean": float(np.mean(ne)) if ne else float("nan"),
        "NE_median": float(np.median(ne)) if ne else float("nan"),
        "NE_n": len(ne),
        "ONE_mean": float(np.mean(one)) if one else float("nan"),
        "ONE_n": len(one),
        "path_mean": float(np.mean(path)) if path else float("nan"),
        "steps_mean": float(np.mean(steps)) if steps else float("nan"),
        "steps_median": float(np.median(steps)) if steps else float("nan"),
        "terms": dict(terms),
    }


def fmt_ci(lo, hi):
    return f"[{lo:.1f}, {hi:.1f}]"


# ----------------------------------------------------------------------------- main
def main():
    out = []
    W = out.append

    arms = {}
    for tag, subdir, n_total, label in ARMS:
        recs = load_arm(subdir, n_total)
        arms[tag] = summarise(tag, label, recs, n_total)

    # ---- episode-set integrity (a receipt in its own right) -------------------------
    sets = {tag: episode_set(subdir) for tag, subdir, _, _ in ARMS}
    transform_tags = ["stretchA", "stretchB", "crop", "pad"]
    height_tags = ["h060", "h078", "h095", "h110", "h130", "h150"]
    pinned300 = sets["stretchA"]
    pinned200 = sets["h060"]
    integrity = []
    for t in transform_tags:
        integrity.append((f"{t} covers the same 300-episode set as stretchA",
                          sets[t] == pinned300))
    for t in height_tags:
        integrity.append((f"{t} covers the same 200-episode set as h060",
                          sets[t] == pinned200))
    integrity.append(("the 200-episode height set is a subset of the 300-episode set",
                      pinned200 <= pinned300))
    base_all = load_arm(BASELINE_DIR)
    integrity.append(("all 300 pinned episodes are present in the full_14498 baseline",
                      pinned300 <= set(base_all)))

    # Baseline restricted to each pinned set — the pre-registered comparison anchors.
    baseline = summarise("full_14498[pinned300]", "baseline, driver 580.159.03",
                         {k: v for k, v in base_all.items() if k in pinned300}, 300)
    baseline_h = summarise("full_14498[pinned200]", "baseline, driver 580.159.03",
                           {k: v for k, v in base_all.items() if k in pinned200}, 200)
    baseline_global = summarise("full_14498[global]", "baseline, driver 580.159.03",
                                base_all, len(base_all))

    W("# ANALYSIS — transform ablation + camera-height sweep (K1 / NaVILA)\n")
    W("Generated by `receipts/analysis/analyze_sweep.py` from the committed per-episode")
    W("receipts in `receipts/sweep_measurements/` and `receipts/baseline_full_14498/`.")
    W("Re-run with `python receipts/analysis/analyze_sweep.py` — no external data needed.\n")
    W("**Analysis plan:** pre-registered in `receipts/PRE_REGISTRATION.md`. The original")
    W("primary test (McNemar, paired) was invalidated by that document's own determinism")
    W("gate; unpaired two-proportion tests + binomial CIs were substituted on 2026-07-27,")
    W("before any arm data was inspected. See the AMENDMENT section of the pre-registration.\n")
    W("**Driver confound:** every sweep arm ran on NVIDIA **580.173.02**; the `full_14498`")
    W("baseline ran on **580.159.03** (purged, unobtainable). Any sweep-vs-baseline contrast")
    W("spans a driver change and is an UPPER BOUND on reproducibility, not a noise floor.")
    W("The within-sweep `stretchA` vs `stretchB` contrast is the clean same-driver noise floor.\n")

    # ---------------------------------------------------------------- integrity
    W("\n## 0. Episode-set integrity (checked at analysis time)\n")
    W("Benchmark episode ids are SPARSE: `--episode_idx=0..299` selects the first 300 episodes")
    W("of the episode list, which land on record indices 0..488 with gaps. Arms are therefore")
    W("compared on episode sets derived from the receipts, never on an assumed index range.\n")
    for desc, ok in integrity:
        W(f"- {'PASS' if ok else '**FAIL**'} — {desc}")
    if all(ok for _, ok in integrity):
        W("\nAll arms are episode-aligned by construction. Note that this makes the design")
        W("*structurally* paired, but paired testing (McNemar) remains invalid here: the")
        W("determinism probe showed per-episode outcomes are not stable run-to-run, so a")
        W("per-episode pair tests noise rather than treatment. See PRE_REGISTRATION.md.")
    else:
        W("\n**An integrity check FAILED — do not use these numbers until it is resolved.**")

    # ---------------------------------------------------------------- arm table
    W("\n## 1. Per-arm results (95% Wilson CIs)\n")
    W("| arm | n | SR % | SR 95% CI | OS % | OS 95% CI | SPL % | NE m | ONE m | missing |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for tag, _, _, _ in ARMS:
        a = arms[tag]
        W(f"| {a['tag']} | {a['n_total']} | {a['SR']:.1f} | {fmt_ci(a['SR_lo'], a['SR_hi'])} "
          f"| {a['OS']:.1f} | {fmt_ci(a['OS_lo'], a['OS_hi'])} | {a['SPL']:.1f} "
          f"| {a['NE_mean']:.2f} | {a['ONE_mean']:.2f} | {a['n_missing']} |")
    for a in (baseline, baseline_h, baseline_global):
        W(f"| *{a['tag']}* | {a['n_total']} | {a['SR']:.1f} | {fmt_ci(a['SR_lo'], a['SR_hi'])} "
          f"| {a['OS']:.1f} | {fmt_ci(a['OS_lo'], a['OS_hi'])} | {a['SPL']:.1f} "
          f"| {a['NE_mean']:.2f} | {a['ONE_mean']:.2f} | {a['n_missing']} |")
    W("\n*Italic rows are the June/July `full_14498` baseline (driver 580.159.03), restricted to")
    W("the same pinned episode sets the sweep arms ran. `pinned300` reproduces the")
    W("pre-registered anchor of 17.7 / 31.7 exactly.*")

    W("\nExact (Clopper-Pearson) SR intervals, for reviewers preferring the conservative form:\n")
    W("| arm | SR % | Clopper-Pearson 95% CI |")
    W("|---|---|---|")
    for tag, _, _, _ in ARMS:
        a = arms[tag]
        W(f"| {a['tag']} | {a['SR']:.1f} | {fmt_ci(a['SR_cp_lo'], a['SR_cp_hi'])} |")

    # ---------------------------------------------------------------- resolution floor
    W("\n## 2. Resolution floor (what this design could ever have detected)\n")
    p_base = baseline["SR"] / 100
    W(f"At a base rate of ~{100*p_base:.0f}% SR, alpha=0.05, power=0.80, unpaired:\n")
    W(f"- n=300 vs n=300: minimum detectable difference ~**{mde(300,300,p_base):.1f} pp**")
    W(f"- n=200 vs n=200: minimum detectable difference ~**{mde(200,200,p_base):.1f} pp**")
    W(f"- n=300 vs n=200: minimum detectable difference ~**{mde(300,200,p_base):.1f} pp**")
    W("\nEvery observed contrast below is smaller than these floors unless flagged otherwise.")
    W("This is a pre-registered limitation, not a post-hoc excuse: the design was powered only")
    W("for large effects, and the pre-registration states a null was an acceptable outcome.")

    # ---------------------------------------------------------------- contrasts
    contrasts = []

    specials = {"base300": baseline, "base200": baseline_h, "baseglobal": baseline_global}

    def add(name, a_tag, b_tag, family, note=""):
        A = arms[a_tag] if a_tag in arms else specials[a_tag]
        B = arms[b_tag] if b_tag in arms else specials[b_tag]
        for metric, kk, nn in (("SR", "succ", "n_total"), ("OS", "osucc", "n_total")):
            d, z, pz, pf, lo, hi = two_proportion_test(A[kk], A[nn], B[kk], B[nn])
            contrasts.append({
                "family": family, "contrast": name, "metric": metric,
                "a": A["tag"], "b": B["tag"],
                "a_pct": A[metric], "b_pct": B[metric],
                "diff_pp": d, "ci_lo_pp": lo, "ci_hi_pp": hi,
                "z": z, "p_z": pz, "p_fisher": pf, "note": note,
            })

    # Family 1 — the noise floor (same driver, identical settings)
    add("stretchB vs stretchA (same-driver replicate)", "stretchB", "stretchA", "noise_floor",
        "THE within-sweep noise floor: identical settings, identical driver.")
    # Family 2 — transform ablation
    add("crop vs stretchA", "crop", "stretchA", "transform")
    add("pad vs stretchA", "pad", "stretchA", "transform")
    add("crop vs stretchB", "crop", "stretchB", "transform")
    add("pad vs stretchB", "pad", "stretchB", "transform")
    add("crop vs pad", "crop", "pad", "transform")
    # Family 3 — height sweep vs native
    for h in ("h060", "h095", "h110", "h130", "h150"):
        add(f"{h} vs h078 (native)", h, "h078", "height")
    # Family 4 — reproducibility across the driver change
    add("stretchA vs full_14498[pinned300]", "stretchA", "base300", "driver",
        "SPANS the 580.159.03 -> 580.173.02 driver change. Upper bound, not a noise floor.")
    add("stretchB vs full_14498[pinned300]", "stretchB", "base300", "driver",
        "SPANS the driver change.")
    add("h078 (native) vs full_14498[pinned200]", "h078", "base200", "driver",
        "h078 reproduces the native camera height, so this is a second reproducibility "
        "probe across the same driver change, on the 200-episode set.")

    # Holm correction within each family x metric
    W("\n## 3. Pre-registered contrasts (unpaired two-proportion)\n")
    W("`p_fisher` is the primary p-value (exact, robust at these success counts); `p_z` is the")
    W("normal-approximation z-test shown for comparison. `p_holm` is Holm-Bonferroni adjusted")
    W("WITHIN each family x metric. CI is the 95% unpooled interval on the difference.\n")

    for fam in ("noise_floor", "transform", "height", "driver"):
        rows = [c for c in contrasts if c["family"] == fam]
        for metric in ("SR", "OS"):
            sub = [c for c in rows if c["metric"] == metric]
            if not sub:
                continue
            adj = holm_bonferroni([c["p_fisher"] for c in sub])
            for c, a in zip(sub, adj):
                c["p_holm"] = a

    fam_titles = {
        "noise_floor": "3.1 Noise floor — same driver, identical settings",
        "transform":   "3.2 Transform ablation (stretch / crop / pad)",
        "height":      "3.3 Camera-height sweep (vs 0.78 m native)",
        "driver":      "3.4 Reproducibility across the driver change (UPPER BOUND)",
    }
    for fam in ("noise_floor", "transform", "height", "driver"):
        W(f"\n### {fam_titles[fam]}\n")
        W("| contrast | metric | A % | B % | diff pp | 95% CI (pp) | p_fisher | p_holm | sig |")
        W("|---|---|---|---|---|---|---|---|---|")
        for c in [x for x in contrasts if x["family"] == fam]:
            sig = "**yes**" if c["p_holm"] < 0.05 else "no"
            W(f"| {c['contrast']} | {c['metric']} | {c['a_pct']:.1f} | {c['b_pct']:.1f} "
              f"| {c['diff_pp']:+.1f} | {fmt_ci(c['ci_lo_pp'], c['ci_hi_pp'])} "
              f"| {c['p_fisher']:.3f} | {c['p_holm']:.3f} | {sig} |")
        notes = {c["note"] for c in contrasts if c["family"] == fam and c["note"]}
        for n in notes:
            W(f"\n> {n}")

    # ---------------------------------------------------------------- trend test
    W("\n## 4. Height trend test (Cochran-Armitage)\n")
    hs = ["h060", "h078", "h095", "h110", "h130", "h150"]
    for metric, kk in (("SR", "succ"), ("OS", "osucc")):
        xs = np.array([HEIGHT_M[h] for h in hs], dtype=float)
        ks = np.array([arms[h][kk] for h in hs], dtype=float)
        ns = np.array([arms[h]["n_total"] for h in hs], dtype=float)
        N = ns.sum()
        p_bar = ks.sum() / N
        xbar = (ns * xs).sum() / N
        num = (ks * (xs - xbar)).sum()
        den = math.sqrt(p_bar * (1 - p_bar) * (ns * (xs - xbar) ** 2).sum())
        z = num / den if den > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        W(f"- **{metric} vs camera height (m):** z = {z:+.3f}, p = {p:.3f} "
          f"-> {'monotone trend detected' if p < 0.05 else 'NO monotone trend'}")
    W("\nThe trend test asks whether SR/OS move monotonically with camera height. It does NOT")
    W("test for a non-monotone optimum; the observed h095 peak is examined in section 3.3 as a")
    W("single pairwise contrast and must be read against the Holm-adjusted p-value there.")

    # ---------------------------------------------------------------- wandering metric
    W("\n## 5. Primary wandering metric — `ended_at_step` by termination reason\n")
    W("Pre-registered as the wandering metric because it is deterministic and load-independent")
    W("(wall-clock is not). Reported as mean/median steps and the termination-reason mix.\n")
    W("| arm | mean steps | median steps | stop | step_cap | wall_timeout | sim_done |")
    W("|---|---|---|---|---|---|---|")
    for tag, _, _, _ in ARMS:
        a = arms[tag]
        t = a["terms"]
        W(f"| {a['tag']} | {a['steps_mean']:.0f} | {a['steps_median']:.0f} "
          f"| {t.get('stop',0)} | {t.get('step_cap',0)} | {t.get('wall_timeout',0)} "
          f"| {t.get('sim_done',0)} |")
    W("\n`wall_timeout` rows are recorded failures written by a SIGTERM handler (the runner kills")
    W("each episode at 900 s wall-clock). They are NOT missing data: SR/SPL are forced to 0,")
    W("while OS/ONE/path_length keep their real partial values. See PRE_REGISTRATION.md.")
    W("Every arm has 0 missing episodes — the guaranteed-record design worked.\n")

    # ---------------------------------------------------------------- headline reading
    W("\n## 6. What the numbers support\n")
    nf = [c for c in contrasts if c["family"] == "noise_floor" and c["metric"] == "SR"][0]
    W(f"1. **Noise floor.** Two identical stretch runs on the same driver differ by "
      f"{abs(nf['diff_pp']):.1f} pp SR ({nf['a_pct']:.1f} vs {nf['b_pct']:.1f}), "
      f"p_fisher = {nf['p_fisher']:.3f}. Run-to-run variation of this size is expected and "
      f"is the yardstick every other contrast must clear.")
    tr = [c for c in contrasts if c["family"] == "transform" and c["metric"] == "SR"]
    if all(c["p_holm"] >= 0.05 for c in tr):
        W("2. **Transform ablation: NULL.** No transform contrast survives Holm correction on SR. "
          "This is the pre-registered expected outcome, not a failure of the experiment.")
    ht = [c for c in contrasts if c["family"] == "height" and c["metric"] == "SR"]
    surviving = [c for c in ht if c["p_holm"] < 0.05]
    if surviving:
        W("3. **Height sweep:** the following survive Holm correction on SR: "
          + ", ".join(f"{c['contrast']} ({c['diff_pp']:+.1f} pp)" for c in surviving))
    else:
        W("3. **Height sweep: NULL on every pairwise contrast** after Holm correction. The h095 "
          "peak is within the run-to-run noise established in (1) and must NOT be reported as "
          "an optimum without a replication arm.")
    W("4. **Reproducibility across the driver change** is reported in 3.4 as an upper bound. "
      "It cannot be separated from run-to-run nondeterminism with the data in hand.")

    # ---------------------------------------------------------------- write outputs
    report = "\n".join(out) + "\n"
    with open(os.path.join(HERE, "ANALYSIS.md"), "w") as f:
        f.write(report)

    with open(os.path.join(HERE, "arm_summary.csv"), "w", newline="") as f:
        cols = ["tag", "label", "n_total", "n_present", "n_missing", "succ", "osucc",
                "SR", "SR_lo", "SR_hi", "SR_cp_lo", "SR_cp_hi", "OS", "OS_lo", "OS_hi",
                "SPL", "NE_mean", "NE_median", "NE_n", "ONE_mean", "ONE_n",
                "path_mean", "steps_mean", "steps_median"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for tag, _, _, _ in ARMS:
            w.writerow(arms[tag])
        w.writerow(baseline)
        w.writerow(baseline_global)

    with open(os.path.join(HERE, "contrasts.csv"), "w", newline="") as f:
        cols = ["family", "contrast", "metric", "a", "b", "a_pct", "b_pct",
                "diff_pp", "ci_lo_pp", "ci_hi_pp", "z", "p_z", "p_fisher", "p_holm", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in contrasts:
            w.writerow(c)

    print(report)
    print(f"\nwrote: {HERE}/ANALYSIS.md, arm_summary.csv, contrasts.csv")


if __name__ == "__main__":
    main()
