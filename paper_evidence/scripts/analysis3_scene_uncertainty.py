#!/usr/bin/env python3
"""
analysis3_scene_uncertainty.py — how much the 95% CI on SR widens once scene
(cluster) dependence is acknowledged instead of treating every episode as an
independent Bernoulli draw.

INPUTS (read-only, via paper_evidence/scripts/paper_common.py):
  - receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json  (primary arms)
  - receipts/sweep_measurements/{h060,h078,h095,h110,h130,h150}_200/*.json (secondary arms)
  - receipts/analysis/episode_index.csv (record_idx -> scene, via pc.arm_scenes)

METHOD
  A. Naive episode-level uncertainty
     - Wilson score 95% CI, same closed-form as receipts/analysis/analyze_robustness.py's
       wilson(k, n, alpha): z = Phi^-1(1 - alpha/2); center/half-width formula below.
     - Episode-level nonparametric bootstrap: resample n episodes WITH replacement from
       the flat list of per-episode outcomes, 10,000 reps, percentile 95% CI. This still
       treats episodes as exchangeable/independent -- it is a second "naive" method, not
       a fix for clustering.
  B. Scene(cluster)-resampled bootstrap
     - Resample the K observed SCENES with replacement (K = number of unique scenes the
       arm covers, i.e. 8 for every transform arm). Each replicate draws K scenes; every
       episode belonging to a drawn scene is included WHOLESALE (a scene drawn twice
       contributes its episodes twice). The resampled episode COUNT varies replicate to
       replicate. Per-replicate SR = (total successes across included episodes) /
       (total episodes across included episodes) -- i.e. an unweighted pooled rate over
       the resampled episode multiset (matches analyze_robustness.py section D). 10,000
       reps, percentile 95% CI.
     - Code path is isolated in `scene_bootstrap()`, which only ever indexes into a list
       of SCENES, never into a flat list of episodes -- see that function and the
       n_scenes_used print/assert in main() for the enforcement of this requirement.
  C. Clustering diagnostics (indicative only, NOT a substitute for B)
     - Per-scene SR and n for each arm.
     - Binary-outcome ICC via the one-way ANOVA / method-of-moments estimator for
       UNEQUAL cluster sizes (Fleiss & Cuzick 1979 / Donner 1986):
         N = total episodes, k = number of scenes, m_i = size of scene i
         p_i = scene i's success rate, pbar = overall success rate
         SSB = sum_i m_i * (p_i - pbar)^2          (between-scene sum of squares)
         SSW = sum_i m_i * p_i * (1 - p_i)         (within-scene sum of squares, exact
                                                     for a 0/1 outcome: sum_j (y_ij-p_i)^2
                                                     = m_i*p_i*(1-p_i))
         MSB = SSB / (k - 1), MSW = SSW / (N - k)
         m_bar = (N - sum_i m_i^2 / N) / (k - 1)     <- the standard unequal-cluster-size
                                                          correction (this IS the m_bar used
                                                          both inside the ICC formula and in
                                                          the DEFF formula below)
         ICC = (MSB - MSW) / (MSB + (m_bar - 1) * MSW)
     - Design effect DEFF = 1 + (m_bar - 1) * ICC, using the SAME m_bar as above.
     - Effective sample size n_eff = N / DEFF.
     - With only k=8 scenes (transform arms) these estimates carry very wide sampling
       uncertainty and are reported for context ONLY. The scene bootstrap in (B) is the
       headline, directly-observable result. See the printed CAVEAT and the CSV `notes`
       column, repeated here deliberately: DO NOT treat ICC/DEFF/n_eff as precise with
       k=8 clusters.

ASSUMPTIONS
  - "Success" is pc.is_success (success field thresholded at >= 0.5), matching every
    other paper_evidence analysis and analyze_robustness.py.
  - Episodes are grouped into scenes exactly as returned by pc.arm_scenes(tag); this is
    the same scene assignment used in receipts/analysis/ROBUSTNESS.md section D.
  - Primary block: the 4 transform arms (stretchA, stretchB, crop, pad), each n=300,
    8 scenes. Secondary block (marked priority="secondary" throughout): the 6 height
    arms (h060..h150), each n=200 -- reported because it costs nothing extra, not
    because it was requested as a primary deliverable.

SEED: numpy.random.default_rng(20260911), fixed once at the top of main() and reused
      in encounter order (episode-bootstrap then scene-bootstrap, arm by arm, in the
      order PRIORITY + SECONDARY) so the run is fully deterministic end to end.

OUTPUTS:
  - paper_evidence/tables/scene_uncertainty.csv
  - paper_evidence/tables/scene_per_scene_sr.csv
  - paper_evidence/figures/scene_uncertainty.{pdf,png}

Comparison with receipts/analysis/ROBUSTNESS.md section D is printed at the end of
main(); differences are expected (different seed: 20260911 here vs 20260822 there) and
are explained rather than tuned away.
"""

import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

SEED = 20260911
N_BOOT = 10000
PRIORITY = list(pc.TRANSFORM_ARMS)                       # stretchA, stretchB, crop, pad
SECONDARY = ["h060", "h078", "h095", "h110", "h130", "h150"]

# Reference numbers from receipts/analysis/ROBUSTNESS.md section D (seed 20260822),
# for the end-of-run comparison only -- never used in any computation here.
ROBUSTNESS_MD_SCENE_CI = {
    "stretchA": (10.5, 19.8),
    "stretchB": (8.6, 24.2),
    "crop": (11.8, 19.9),
    "pad": (6.0, 22.1),
}


def wilson(k, n, alpha=0.05):
    """Wilson score 95% CI -- identical formula to analyze_robustness.py's wilson()."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def by_scene(tag):
    """{scene: [0/1, ...]} of per-episode success outcomes for one arm."""
    recs = pc.load_arm(tag)
    scenes = pc.arm_scenes(tag)
    out = defaultdict(list)
    for idx, sc in scenes.items():
        out[sc].append(1 if pc.is_success(recs[idx]) else 0)
    return out


def episode_bootstrap(flat_outcomes, rng, n_boot=N_BOOT):
    """Naive nonparametric bootstrap over individual EPISODES (not scenes)."""
    arr = np.asarray(flat_outcomes, dtype=float)
    n = len(arr)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = arr[idx].mean(axis=1)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return lo, hi


def scene_bootstrap(byscene, rng, n_boot=N_BOOT):
    """
    Cluster bootstrap that resamples SCENES, never individual episodes.

    `scenes` is the list of cluster keys; every index drawn by rng.integers() below
    selects a whole scene, and ALL of that scene's episodes are pulled in via
    `sizes[i]`/`succ[i]`. There is no code path here that indexes a flat episode array.
    """
    scenes = list(byscene.keys())
    k = len(scenes)
    sizes = np.array([len(byscene[s]) for s in scenes])
    succ = np.array([sum(byscene[s]) for s in scenes])

    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, size=k)          # <-- indices into the SCENE list
        reps[b] = succ[pick].sum() / sizes[pick].sum()
    lo, hi = np.percentile(reps * 100, [2.5, 97.5])
    return lo, hi, k


def icc_deff(byscene):
    """
    One-way ANOVA / method-of-moments ICC for a binary outcome with unequal cluster
    sizes (Fleiss & Cuzick 1979 / Donner 1986), plus DEFF and n_eff using the same
    m_bar in both places. See module docstring section C for the exact formulas.
    """
    scenes = list(byscene.keys())
    m = np.array([len(byscene[s]) for s in scenes], dtype=float)
    p_i = np.array([np.mean(byscene[s]) for s in scenes])
    N = m.sum()
    k = len(scenes)
    pbar = np.array([x for v in byscene.values() for x in v]).mean()

    ssb = np.sum(m * (p_i - pbar) ** 2)
    ssw = np.sum(m * p_i * (1 - p_i))
    msb = ssb / (k - 1)
    msw = ssw / (N - k) if N > k else float("nan")

    mbar = (N - np.sum(m ** 2) / N) / (k - 1)

    denom = msb + (mbar - 1) * msw
    icc = (msb - msw) / denom if denom != 0 else float("nan")
    icc_trunc = max(icc, 0.0)
    deff = 1 + (mbar - 1) * icc_trunc
    n_eff = N / deff
    return dict(icc=icc, icc_truncated=icc_trunc, mbar=mbar, msb=msb, msw=msw,
                deff=deff, n_eff=n_eff, k=k, N=N)


def main():
    rng = np.random.default_rng(SEED)

    table_rows = []
    per_scene_rows = []
    fig_data = {}  # tag -> dict of stats needed for the figure

    print(f"analysis3_scene_uncertainty.py  (seed={SEED}, n_boot={N_BOOT})\n")
    print(f"{'arm':10s} {'n':>5s} {'succ':>5s} {'SR%':>7s} "
          f"{'Wilson95':>18s} {'epBoot95':>18s} {'sceneBoot95':>18s} "
          f"{'w_naive':>8s} {'w_scene':>8s} {'ratio':>7s}")

    for tag in PRIORITY + SECONDARY:
        priority_tag = "primary" if tag in PRIORITY else "secondary"
        recs = pc.load_arm(tag)
        n = len(recs)
        outcomes = [1 if pc.is_success(r) else 0 for r in recs.values()]
        k_succ = sum(outcomes)
        sr = k_succ / n

        byscene = by_scene(tag)
        n_scenes = len(byscene)

        wlo, whi = wilson(k_succ, n)
        wlo_pct, whi_pct = 100 * wlo, 100 * whi
        w_width = whi_pct - wlo_pct

        eblo, ebhi = episode_bootstrap(outcomes, rng)
        eblo_pct, ebhi_pct = 100 * eblo, 100 * ebhi
        eb_width = ebhi_pct - eblo_pct

        sblo, sbhi, n_scenes_used = scene_bootstrap(byscene, rng)
        assert n_scenes_used == n_scenes, "scene bootstrap must use every observed scene"
        sb_width = sbhi - sblo
        ratio = sb_width / w_width if w_width > 0 else float("nan")

        diag = icc_deff(byscene)

        notes = (
            f"k={diag['k']} scenes, sizes range "
            f"{min(len(v) for v in byscene.values())}-{max(len(v) for v in byscene.values())}; "
            "ICC/DEFF/n_eff are INDICATIVE ONLY -- with k=8 clusters they have very wide "
            "sampling uncertainty; the scene-bootstrap CI (directly observable) is the "
            "headline result, not these derived quantities."
        )
        if diag["icc"] < 0:
            notes += (
                " Raw ICC estimate is negative (method-of-moments artifact: between-scene "
                "mean square fell below within-scene mean square in this finite sample, "
                "which is possible and does not imply negative true dependence); truncated "
                "to 0 for DEFF/n_eff, consistent with ICC>=0 for a random-effects clustering "
                "model."
            )

        table_rows.append([
            tag, pc.ARMS[tag][2], priority_tag, n, k_succ, n_scenes,
            f"{100*sr:.4f}", f"{wlo_pct:.4f}", f"{whi_pct:.4f}", f"{w_width:.4f}",
            f"{eblo_pct:.4f}", f"{ebhi_pct:.4f}", f"{eb_width:.4f}",
            f"{sblo:.4f}", f"{sbhi:.4f}", f"{sb_width:.4f}",
            f"{ratio:.4f}",
            f"{diag['icc']:.4f}", f"{diag['icc_truncated']:.4f}",
            f"{diag['mbar']:.4f}", f"{diag['deff']:.4f}", f"{diag['n_eff']:.2f}",
            notes,
        ])

        for sc, vals in sorted(byscene.items(), key=lambda kv: -len(kv[1])):
            m_i = len(vals)
            s_i = sum(vals)
            per_scene_rows.append([tag, sc, m_i, s_i, f"{100*s_i/m_i:.4f}"])

        if tag in PRIORITY:
            fig_data[tag] = dict(sr=100 * sr, wlo=wlo_pct, whi=whi_pct,
                                  sblo=sblo, sbhi=sbhi)

        print(f"{tag:10s} {n:5d} {k_succ:5d} {100*sr:7.2f} "
              f"[{wlo_pct:6.2f},{whi_pct:6.2f}]  [{eblo_pct:6.2f},{ebhi_pct:6.2f}]  "
              f"[{sblo:6.2f},{sbhi:6.2f}]  {w_width:8.2f} {sb_width:8.2f} {ratio:7.2f}")

    header = [
        "arm", "label", "priority", "n", "successes", "n_scenes", "sr_pct",
        "wilson_lo_pct", "wilson_hi_pct", "wilson_width_pp",
        "episode_boot_lo_pct", "episode_boot_hi_pct", "episode_boot_width_pp",
        "scene_boot_lo_pct", "scene_boot_hi_pct", "scene_boot_width_pp",
        "width_ratio_scene_over_wilson",
        "icc", "icc_truncated", "mbar", "deff", "n_eff", "notes",
    ]
    out_csv = pc.write_csv(os.path.join(pc.TABLES, "scene_uncertainty.csv"), header, table_rows)

    per_scene_header = ["arm", "scene", "n_episodes", "successes", "sr_pct"]
    per_scene_csv = pc.write_csv(os.path.join(pc.TABLES, "scene_per_scene_sr.csv"),
                                  per_scene_header, per_scene_rows)

    print(f"\nwrote {out_csv}")
    print(f"wrote {per_scene_csv}")

    # -------------------------------------------------------------- diagnostics printout
    print("\nCLUSTERING DIAGNOSTICS (indicative only -- see CAVEAT below):")
    for row in table_rows:
        if row[2] != "primary":
            continue
        tag, label = row[0], row[1]
        icc, icc_t, mbar, deff, n_eff = row[17], row[18], row[19], row[20], row[21]
        print(f"  {tag:10s} ICC(raw)={icc:>8s}  ICC(trunc)={icc_t:>7s}  "
              f"mbar={mbar:>7s}  DEFF={deff:>6s}  n_eff={n_eff:>7s}")
    print(
        "\nCAVEAT: with only k=8 scenes for every transform arm (cluster sizes 9-84), the\n"
        "ANOVA/method-of-moments ICC, DEFF and n_eff above have VERY WIDE sampling\n"
        "uncertainty (a handful of clusters barely constrains a variance-of-variances\n"
        "estimator). They are reported for context only. The scene-resampled bootstrap\n"
        "interval computed in section B is a directly observable quantity from the same\n"
        "300 episodes and is the headline result of this analysis, not these derived\n"
        "ICC/DEFF/n_eff figures."
    )

    # -------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    tags = PRIORITY
    ypos = np.arange(len(tags))
    offset = 0.15
    for i, tag in enumerate(tags):
        d = fig_data[tag]
        y_naive = ypos[i] + offset
        y_scene = ypos[i] - offset
        ax.errorbar(d["sr"], y_naive, xerr=[[d["sr"] - d["wlo"]], [d["whi"] - d["sr"]]],
                    fmt="o", color="#1b1b1b", markersize=3.5, capsize=2.5, lw=1.1,
                    label="naive Wilson (episode-level)" if i == 0 else None)
        ax.errorbar(d["sr"], y_scene, xerr=[[d["sr"] - d["sblo"]], [d["sbhi"] - d["sr"]]],
                    fmt="s", color="#4477aa", markersize=3.5, capsize=2.5, lw=1.1,
                    label="scene-cluster bootstrap" if i == 0 else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t}\n(n=300, k=8 scenes)" for t in tags])
    ax.set_xlabel("success rate (%)")
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(left=0)
    # Legend placed BELOW the axes (not inside the data area) so it never overlaps
    # any marker or interval, including pad's wide naive-Wilson bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_ylim(-0.6, len(tags) - 1 + 0.6)
    fig.tight_layout()
    pdf_path = os.path.join(pc.FIGURES, "scene_uncertainty.pdf")
    png_path = os.path.join(pc.FIGURES, "scene_uncertainty.png")
    os.makedirs(pc.FIGURES, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")

    # -------------------------------------------------------------- comparison to ROBUSTNESS.md
    print("\nCOMPARISON with receipts/analysis/ROBUSTNESS.md section D "
          "(seed 20260822 there, seed 20260911 here):")
    for tag in PRIORITY:
        d = fig_data[tag]
        md_lo, md_hi = ROBUSTNESS_MD_SCENE_CI[tag]
        print(f"  {tag:10s} here=[{d['sblo']:.1f}, {d['sbhi']:.1f}]  "
              f"ROBUSTNESS.md=[{md_lo:.1f}, {md_hi:.1f}]")
    print(
        "\nExpected sources of small differences: (1) a different RNG seed (20260911 vs\n"
        "20260822) drives a different set of 10,000 scene-resample draws -- with only 8\n"
        "clusters, the resample distribution has visible Monte Carlo noise at the 2.5th\n"
        "and 97.5th percentiles even at 10,000 reps; (2) both implementations here and in\n"
        "analyze_robustness.py resample K=len(scenes) scenes per replicate and pool\n"
        "successes/episodes across the resampled scene multiset (unweighted pooled SR, not\n"
        "a mean-of-per-scene-SR) -- same convention, so this is NOT a source of difference.\n"
        "If any pair above differs by more than a percentage point or two, that is larger\n"
        "than plausible seed noise and should be investigated rather than accepted."
    )


if __name__ == "__main__":
    main()
