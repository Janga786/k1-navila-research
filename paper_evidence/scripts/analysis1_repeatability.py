#!/usr/bin/env python3
"""
analysis1_repeatability.py — run-to-run outcome instability, stretchA vs stretchB.

INPUTS (read-only):
  receipts/sweep_measurements/stretchA_300/*.json
  receipts/sweep_measurements/stretchB_300/*.json
  receipts/analysis/episode_index.csv           (via paper_common.arm_scenes)
  All access goes through paper_evidence/scripts/paper_common.py (load_arm,
  paired_episodes, is_success, arm_scenes). No receipt file is written to.

OUTPUTS:
  paper_evidence/tables/repeatability_contingency.csv
  paper_evidence/tables/repeatability_statistics.csv
  paper_evidence/figures/repeatability_paired.pdf
  paper_evidence/figures/repeatability_paired.png

METHOD:
  stretchA and stretchB are the identical 300 episodes run under nominally
  identical configuration (same transform, weights, driver 580.173.02). Every
  episode is paired on record_idx via paper_common.paired_episodes, which
  hard-fails on any key mismatch. From the paired 2x2 outcome table we compute
  raw agreement/discordance, Cohen's kappa with two 95% CIs (asymptotic
  Fleiss/Cohen large-sample SE, and an episode-level nonparametric bootstrap),
  positive/negative percent agreement, an exact (binomial) McNemar test, the
  fail->success / success->fail flip fractions, the union-of-successes overlap
  statistic, and a SCENE-level cluster bootstrap (resampling the 8 covered
  scenes with replacement, carrying all paired episodes in a drawn scene) for
  kappa, discordance, and the SR(B)-SR(A) difference.

SEED: numpy.random.default_rng(20260911), used for both the episode-level and
  the scene-cluster bootstraps (10,000 resamples each).
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import (  # noqa: E402
    FIGURES, TABLES, arm_scenes, is_success, paired_episodes, write_csv,
)

SEED = 20260911
N_BOOT = 10000


def cohens_kappa(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    po = (n11 + n00) / n
    pa_marg = (n11 + n10) / n   # A success marginal
    pb_marg = (n11 + n01) / n   # B success marginal
    pe = pa_marg * pb_marg + (1 - pa_marg) * (1 - pb_marg)
    if pe == 1.0:
        return float("nan")
    kappa = (po - pe) / (1 - pe)
    return kappa


def kappa_asymptotic_se(n11, n10, n01, n00):
    """Fleiss/Cohen large-sample SE for kappa in the 2x2 case (Fleiss, Levin & Paik 2003,
    eq. for the standard error of an estimated kappa; equivalent to the classic
    Cohen (1960) / Fleiss (1969) asymptotic variance formula for two raters, two
    categories)."""
    n = n11 + n10 + n01 + n00
    p11, p10, p01, p00 = n11 / n, n10 / n, n01 / n, n00 / n
    po = p11 + p00
    pa_marg = p11 + p10
    pb_marg = p11 + p01
    pe = pa_marg * pb_marg + (1 - pa_marg) * (1 - pb_marg)
    kappa = (po - pe) / (1 - pe)

    # Row/col marginals for rater A and B (successes)
    pA1, pA0 = pa_marg, 1 - pa_marg
    pB1, pB0 = pb_marg, 1 - pb_marg

    # Closed-form large-sample variance of kappa (Fleiss, Levin & Paik, 2003, sec 18.2.1):
    #   var(kappa) = (1/n) * { [po(1-po)] / (1-pe)^2
    #       + 2*(1-po)*(2*po*pe - A2) / (1-pe)^3
    #       + (1-po)^2 * (B2 - 4*pe^2) / (1-pe)^4 }
    # where A2 = sum over diagonal (agreement) cells of p_ii*(p_i.+p_.i),
    #       B2 = sum over all cells of p_ij*(p_i.+p_.j)^2.
    p = {("A1", "B1"): p11, ("A1", "B0"): p10, ("A0", "B1"): p01, ("A0", "B0"): p00}
    row_marg = {"A1": pA1, "A0": pA0}
    col_marg = {"B1": pB1, "B0": pB0}

    A2 = 0.0
    for cat, pij in ((("A1", "B1"), p11), (("A0", "B0"), p00)):
        i, j = cat
        A2 += pij * (row_marg[i] + col_marg[j])

    B2 = 0.0
    for i in ("A1", "A0"):
        for j in ("B1", "B0"):
            B2 += p[(i, j)] * (row_marg[i] + col_marg[j]) ** 2

    var_kappa = (1.0 / n) * (
        (po * (1 - po)) / (1 - pe) ** 2
        + 2 * (1 - po) * (2 * po * pe - A2) / (1 - pe) ** 3
        + ((1 - po) ** 2) * (B2 - 4 * pe ** 2) / (1 - pe) ** 4
    )
    se = np.sqrt(max(var_kappa, 0.0))
    return kappa, se


def main():
    idx, A, B = paired_episodes("stretchA", "stretchB")
    n = len(idx)
    scenesA = arm_scenes("stretchA")

    a_succ = {k: is_success(A[k]) for k in idx}
    b_succ = {k: is_success(B[k]) for k in idx}

    n11 = sum(1 for k in idx if a_succ[k] and b_succ[k])
    n10 = sum(1 for k in idx if a_succ[k] and not b_succ[k])
    n01 = sum(1 for k in idx if not a_succ[k] and b_succ[k])
    n00 = sum(1 for k in idx if not a_succ[k] and not b_succ[k])
    assert n11 + n10 + n01 + n00 == n

    srA = sum(a_succ.values()) / n
    srB = sum(b_succ.values()) / n

    print("=" * 78)
    print("ANALYSIS 1 — repeatability (stretchA vs stretchB), n =", n)
    print("=" * 78)
    print(f"stretchA: succ={sum(a_succ.values())} SR={srA:.4f}")
    print(f"stretchB: succ={sum(b_succ.values())} SR={srB:.4f}")
    print()
    print("2x2 contingency (rows=A, cols=B):")
    print(f"  A succ & B succ (n11) = {n11}")
    print(f"  A succ & B fail (n10) = {n10}")
    print(f"  A fail & B succ (n01) = {n01}")
    print(f"  A fail & B fail (n00) = {n00}")
    hist = {"n11": 30, "n10": 15, "n01": 18, "n00": 237}
    got = {"n11": n11, "n10": n10, "n01": n01, "n00": n00}
    for key in hist:
        if got[key] != hist[key]:
            print(f"  *** MISMATCH vs historical {key}: got {got[key]}, expected {hist[key]} ***")
    if got == hist:
        print("  MATCHES historical 30/15/18/237.")

    # --- agreement / discordance ---
    disc = n10 + n01
    agree = n11 + n00
    p_agree = agree / n
    p_disc = disc / n

    # --- kappa ---
    kappa, se_kappa = kappa_asymptotic_se(n11, n10, n01, n00)
    z975 = stats.norm.ppf(0.975)
    kappa_ci_analytic = (kappa - z975 * se_kappa, kappa + z975 * se_kappa)

    # --- positive / negative agreement ---
    PA = 2 * n11 / (2 * n11 + n10 + n01)
    NA = 2 * n00 / (2 * n00 + n10 + n01)

    # --- exact McNemar ---
    mcnemar_p = stats.binomtest(n10, disc, 0.5).pvalue

    # --- flips ---
    ever = n11 + n10 + n01  # union of successes
    frac01_of_n = n01 / n
    frac10_of_n = n10 / n
    frac01_of_union = n01 / ever if ever else float("nan")
    frac10_of_union = n10 / ever if ever else float("nan")
    union_both_frac = n11 / ever if ever else float("nan")

    print()
    print(f"Raw agreement = {p_agree:.4f}   discordance = {p_disc:.4f}  ({disc}/{n})")
    print(f"Cohen's kappa = {kappa:.4f}   analytic SE = {se_kappa:.4f}   "
          f"95% CI (asymptotic) = [{kappa_ci_analytic[0]:.4f}, {kappa_ci_analytic[1]:.4f}]")
    print(f"Positive agreement PA = {PA:.4f}   Negative agreement NA = {NA:.4f}")
    print(f"McNemar exact (binomial on discordants, p=0.5): n10={n10} n01={n01} p={mcnemar_p:.4f}")
    print(f"Union of successes = {ever}; both-succeeded fraction of union = {union_both_frac:.4f} "
          f"({n11}/{ever})")
    print(f"fail->success (n01) = {n01} = {frac01_of_n:.4f} of n, {frac01_of_union:.4f} of union")
    print(f"success->fail (n10) = {n10} = {frac10_of_n:.4f} of n, {frac10_of_union:.4f} of union")

    # ------------------------------------------------------------------
    # Episode-level nonparametric bootstrap for kappa CI
    # ------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    a_arr = np.array([a_succ[k] for k in idx], dtype=bool)
    b_arr = np.array([b_succ[k] for k in idx], dtype=bool)

    boot_kappa = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sel = rng.integers(0, n, size=n)
        aa, bb = a_arr[sel], b_arr[sel]
        bn11 = np.sum(aa & bb)
        bn10 = np.sum(aa & ~bb)
        bn01 = np.sum(~aa & bb)
        bn00 = np.sum(~aa & ~bb)
        boot_kappa[i] = cohens_kappa(bn11, bn10, bn01, bn00)

    valid = np.isfinite(boot_kappa)
    kappa_ci_boot = (
        np.percentile(boot_kappa[valid], 2.5),
        np.percentile(boot_kappa[valid], 97.5),
    )
    print(f"Episode-bootstrap ({N_BOOT} resamples, seed {SEED}) kappa 95% CI = "
          f"[{kappa_ci_boot[0]:.4f}, {kappa_ci_boot[1]:.4f}]  "
          f"(n_valid={int(valid.sum())}/{N_BOOT})")

    # ------------------------------------------------------------------
    # Scene-cluster bootstrap: resample 8 scenes w/ replacement, carry all
    # paired episodes belonging to a drawn scene.
    # ------------------------------------------------------------------
    scene_of = {k: scenesA[k] for k in idx}
    scenes = sorted(set(scene_of.values()))
    n_scenes = len(scenes)
    scene_to_idx = {s: [k for k in idx if scene_of[k] == s] for s in scenes}

    rng2 = np.random.default_rng(SEED)
    boot_kappa_sc = np.empty(N_BOOT)
    boot_disc_sc = np.empty(N_BOOT)
    boot_srdiff_sc = np.empty(N_BOOT)
    for i in range(N_BOOT):
        drawn = rng2.choice(scenes, size=n_scenes, replace=True)
        keys = []
        for s in drawn:
            keys.extend(scene_to_idx[s])
        keys = np.array(keys)
        aa = np.array([a_succ[k] for k in keys], dtype=bool)
        bb = np.array([b_succ[k] for k in keys], dtype=bool)
        m = len(keys)
        bn11 = np.sum(aa & bb)
        bn10 = np.sum(aa & ~bb)
        bn01 = np.sum(~aa & bb)
        bn00 = np.sum(~aa & ~bb)
        boot_kappa_sc[i] = cohens_kappa(bn11, bn10, bn01, bn00)
        boot_disc_sc[i] = (bn10 + bn01) / m
        boot_srdiff_sc[i] = (bb.mean() - aa.mean())

    valid_sc = np.isfinite(boot_kappa_sc)
    kappa_ci_scene = (np.percentile(boot_kappa_sc[valid_sc], 2.5),
                       np.percentile(boot_kappa_sc[valid_sc], 97.5))
    disc_ci_scene = (np.percentile(boot_disc_sc, 2.5), np.percentile(boot_disc_sc, 97.5))
    srdiff_ci_scene = (np.percentile(boot_srdiff_sc, 2.5), np.percentile(boot_srdiff_sc, 97.5))
    sr_diff_point = srB - srA

    print()
    print(f"Scene-cluster bootstrap ({N_BOOT} resamples, seed {SEED}, {n_scenes} scenes, "
          f"resampling WHOLE SCENES):")
    print(f"  kappa scene-cluster 95% CI      = [{kappa_ci_scene[0]:.4f}, {kappa_ci_scene[1]:.4f}]"
          f"  (n_valid={int(valid_sc.sum())}/{N_BOOT})")
    print(f"  discordance scene-cluster 95% CI = [{disc_ci_scene[0]:.4f}, {disc_ci_scene[1]:.4f}]")
    print(f"  SR(B)-SR(A) point = {sr_diff_point:+.4f}, scene-cluster 95% CI = "
          f"[{srdiff_ci_scene[0]:+.4f}, {srdiff_ci_scene[1]:+.4f}]")
    print(f"  CAVEAT: only {n_scenes} scenes are covered by these 300 episodes; a cluster")
    print(f"  bootstrap over {n_scenes} clusters has very coarse resolution (as few as "
          f"{n_scenes} distinct resample compositions along some dimensions) and these")
    print("  scene-cluster CIs should be read as indicative of instability, not precise.")

    # ------------------------------------------------------------------
    # WRITE contingency table
    # ------------------------------------------------------------------
    contingency_rows = [
        ("success", "success", n11, n11 / n),
        ("success", "failure", n10, n10 / n),
        ("failure", "success", n01, n01 / n),
        ("failure", "failure", n00, n00 / n),
        ("TOTAL", "TOTAL", n, 1.0),
    ]
    write_csv(
        os.path.join(TABLES, "repeatability_contingency.csv"),
        ["stretchA_outcome", "stretchB_outcome", "n_episodes", "fraction_of_total"],
        [(a, b, c, f"{f:.6f}") for a, b, c, f in contingency_rows],
    )

    # ------------------------------------------------------------------
    # WRITE statistics table
    # ------------------------------------------------------------------
    stat_rows = []

    def row(statistic, value, lo, hi, method, nn, notes):
        stat_rows.append((statistic, value, lo, hi, method, nn, notes))

    row("stretchA_SR", f"{srA:.6f}", "", "", "", n,
        f"successes={sum(a_succ.values())}/{n}")
    row("stretchB_SR", f"{srB:.6f}", "", "", "", n,
        f"successes={sum(b_succ.values())}/{n}")
    row("n11_A_succ_B_succ", n11, "", "", "", n, "contingency cell")
    row("n10_A_succ_B_fail", n10, "", "", "", n, "contingency cell")
    row("n01_A_fail_B_succ", n01, "", "", "", n, "contingency cell")
    row("n00_A_fail_B_fail", n00, "", "", "", n, "contingency cell")
    row("raw_agreement", f"{p_agree:.6f}", "", "", "", n, "(n11+n00)/n")
    row("discordance_rate", f"{p_disc:.6f}", "", "", "", n, "(n10+n01)/n")
    row("cohens_kappa", f"{kappa:.6f}", f"{kappa_ci_analytic[0]:.6f}",
        f"{kappa_ci_analytic[1]:.6f}", "asymptotic_fleiss_cohen_SE", n,
        f"SE={se_kappa:.6f}")
    row("cohens_kappa", f"{kappa:.6f}", f"{kappa_ci_boot[0]:.6f}",
        f"{kappa_ci_boot[1]:.6f}", "episode_bootstrap_percentile",
        n, f"{N_BOOT} resamples, seed {SEED}, n_valid={int(valid.sum())}")
    row("positive_agreement_PA", f"{PA:.6f}", "", "", "", n, "2*n11/(2*n11+n10+n01)")
    row("negative_agreement_NA", f"{NA:.6f}", "", "", "", n, "2*n00/(2*n00+n10+n01)")
    row("mcnemar_exact_p", f"{mcnemar_p:.6f}", "", "", "binomial_two_sided", disc,
        f"n10={n10} n01={n01} vs p=0.5")
    row("n01_fail_to_success", n01, "", "", "", n,
        f"fraction_of_n={frac01_of_n:.6f}; fraction_of_union={frac01_of_union:.6f}")
    row("n10_success_to_fail", n10, "", "", "", n,
        f"fraction_of_n={frac10_of_n:.6f}; fraction_of_union={frac10_of_union:.6f}")
    row("union_of_successes", ever, "", "", "", n,
        f"episodes succeeding in either run out of {n}")
    row("union_both_succeeded_fraction", f"{union_both_frac:.6f}", "", "", "", ever,
        f"{n11}/{ever} succeeded in both runs (historically 30/63=48%)")
    row("scene_count", n_scenes, "", "", "", n,
        "number of distinct scenes covered by the 300 paired episodes; "
        "scene-cluster bootstrap resamples only this many clusters -> coarse resolution")
    row("cohens_kappa_scene_cluster", f"{kappa:.6f}", f"{kappa_ci_scene[0]:.6f}",
        f"{kappa_ci_scene[1]:.6f}", "scene_cluster_bootstrap_percentile", n,
        f"{N_BOOT} resamples, seed {SEED}, {n_scenes} scenes resampled with replacement; "
        f"n_valid={int(valid_sc.sum())}; ONLY {n_scenes} CLUSTERS -> coarse/unstable CI")
    row("discordance_rate_scene_cluster", f"{p_disc:.6f}", f"{disc_ci_scene[0]:.6f}",
        f"{disc_ci_scene[1]:.6f}", "scene_cluster_bootstrap_percentile", n,
        f"{N_BOOT} resamples, seed {SEED}, {n_scenes} scenes; ONLY {n_scenes} CLUSTERS -> coarse/unstable CI")
    row("sr_diff_B_minus_A_scene_cluster", f"{sr_diff_point:.6f}",
        f"{srdiff_ci_scene[0]:.6f}", f"{srdiff_ci_scene[1]:.6f}",
        "scene_cluster_bootstrap_percentile", n,
        f"{N_BOOT} resamples, seed {SEED}, {n_scenes} scenes; ONLY {n_scenes} CLUSTERS -> coarse/unstable CI")

    write_csv(
        os.path.join(TABLES, "repeatability_statistics.csv"),
        ["statistic", "value", "ci95_lo", "ci95_hi", "ci_method", "n", "notes"],
        stat_rows,
    )

    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })

    cats = ["Both success", "A only", "B only", "Both failure"]
    counts = [n11, n10, n01, n00]
    colors = ["#2c7fb8", "#7fcdbb", "#41b6c4", "#bdbdbd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel 1: full-scale horizontal bars, log x-axis so all 4 are visible
    y = np.arange(len(cats))
    ax1.barh(y, counts, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(cats)
    ax1.invert_yaxis()
    ax1.set_xscale("log")
    ax1.set_xlim(1, 400)
    ax1.set_xlabel("episodes (log scale)")
    for yi, c in zip(y, counts):
        ax1.text(c * 1.08, yi, str(c), va="center", fontsize=7)
    ax1.set_title("(a) all four categories", loc="left")

    # Panel 2: zoom on the three small (non-both-failure) categories, linear scale
    small_cats = cats[:3]
    small_counts = counts[:3]
    y2 = np.arange(len(small_cats))
    ax2.bar(y2, small_counts, color=colors[:3])
    ax2.set_xticks(y2)
    ax2.set_xticklabels(small_cats, rotation=0)
    ax2.set_ylabel("episodes")
    ax2.set_ylim(0, max(small_counts) * 1.25)
    for xi, c in zip(y2, small_counts):
        ax2.text(xi, c + max(small_counts) * 0.03, str(c), ha="center", fontsize=7)
    ax2.set_title(f"(b) zoom (both-failure = {n00} off-scale)", loc="left")

    fig.suptitle("")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"repeatability_paired.{ext}"))
    plt.close(fig)

    print()
    print("Wrote:")
    print(" ", os.path.join(TABLES, "repeatability_contingency.csv"))
    print(" ", os.path.join(TABLES, "repeatability_statistics.csv"))
    print(" ", os.path.join(FIGURES, "repeatability_paired.pdf"))
    print(" ", os.path.join(FIGURES, "repeatability_paired.png"))


if __name__ == "__main__":
    main()
