#!/usr/bin/env python3
"""
analysis4_mde_power.py — a practical planning answer to "how large a difference in
success rate could an experiment of this size realistically detect?"

INPUTS (read-only, via paper_evidence/scripts/paper_common.py):
  - receipts/sweep_measurements/stretchA_300 (p1 = 0.15 anchor: the stretchA SR)
  - receipts/baseline_full_14498 (p1 ~= 0.18 anchor: the full 1077-episode baseline SR,
    computed as pc.load_baseline() success rate, NOT hardcoded)

DESIGN
  Two independent arms, equal n per arm, two-sided two-proportion z-test,
  alpha = 0.05, target power = 0.80. For a chosen baseline p1 and a given n per
  arm, we solve for the minimum absolute difference d = p2 - p1 (p2 > p1) at
  which this test attains power >= 0.80. This is the "minimum detectable
  difference" (MDE) / "detectable effect size" under the stated assumptions --
  referred to only by that name (or "statistical resolution") throughout this
  file, its outputs, and the report built from it -- per the wording mandate,
  the other, more colloquial term for this quantity is intentionally avoided.

EXACT FORMULA (documented once, used everywhere; standard unpooled-variance power
formula for the two-proportion z-test, e.g. Fleiss/Levin/Paik "Statistical Methods
for Rates and Proportions", or equivalently the sample-size formula solved for d):

  Let n = per-arm sample size, p1 = baseline rate, p2 = p1 + d, pbar = (p1+p2)/2.
  Test statistic under H0 uses the POOLED standard error:
      se0 = sqrt(2 * pbar * (1-pbar) / n)
  The estimator's true standard error under H1 uses the UNPOOLED variance:
      se1 = sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
  z_(alpha/2) = Phi^-1(1 - alpha/2)   [Phi = standard normal CDF]
  Power (one-sided approximation to the two-sided test, valid whenever d is not
  tiny relative to its own MDE, which holds everywhere in this table):
      power(d) = Phi( (d - z_(alpha/2) * se0) / se1 )
  MDE(n) is the d > 0 solving power(d) = 0.80, found by bisection
  (scipy.optimize.brentq) rather than the (biased, non-invertible-in-closed-form)
  sample-size formula, because pbar itself depends on d.

  ASSUMPTIONS: independent arms (no episode reuse/pairing across arms), a normal
  approximation to the binomial difference (adequate here: p1 in [0.15, 0.19],
  n >= 50), and the closed-form/pooled-vs-unpooled convention above. This is the
  simplest defensible design for this planning question; it explicitly ignores
  scene clustering (see analysis3_scene_uncertainty.py for that) -- MDE numbers
  here are therefore, if anything, OPTIMISTIC relative to a cluster-aware design.

  A ~10,000-replicate Monte Carlo simulation is run at THREE (n, d) points purely
  as a cross-check of the closed form (see `monte_carlo_check()`); it is reported
  as a validation line only, never as the primary method, per instructions.

BASELINES USED (documented, not hardcoded):
  - p1 = 0.15 (PRIMARY): pinned literally, matching the stretchA SR = 45/300 =
    0.1500 exactly. Chosen as primary because it is the operative rate in the
    transform-sweep regime this paper reports on.
  - p1 = computed full-benchmark baseline SR (pc.load_baseline(), 197/1077 =
    0.18292..., "approximately 0.18" per the task brief) (SECONDARY).

n per arm: 50, 100, 200, 300, 500, 750, 1000, 1500, 2000.

SANITY ANCHOR: at n=300, p1=0.15, MDE should land near 8-10 pp (this project's
pre-registration states a ~9 pp resolution floor at n=300); printed and checked.

SEED: numpy.random.default_rng(20260911), used only for the Monte Carlo cross-check.

OUTPUTS:
  - paper_evidence/tables/mde_power.csv
  - paper_evidence/figures/mde_power.{pdf,png}
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

SEED = 20260911
ALPHA = 0.05
POWER_TARGET = 0.80
N_LIST = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
EXPERIMENT_N = {"transform arms (this study)": 300, "height arms (this study)": 200}


def power_for_d(p1, d, n, alpha=ALPHA):
    """power(d) under the unpooled-variance / pooled-null formula documented above."""
    p2 = p1 + d
    if not (0 < p2 < 1):
        return 0.0
    pbar = (p1 + p2) / 2
    za = stats.norm.ppf(1 - alpha / 2)
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
    z = (d - za * se0) / se1
    return stats.norm.cdf(z)


def mde(p1, n, alpha=ALPHA, power=POWER_TARGET):
    """Minimum d solving power_for_d(p1, d, n) == power, via bisection."""
    f = lambda d: power_for_d(p1, d, n, alpha) - power
    lo, hi = 1e-6, 1 - p1 - 1e-9
    if f(hi) < 0:
        raise RuntimeError(f"no achievable d<{hi} reaches power={power} at n={n}, p1={p1}")
    return brentq(f, lo, hi, xtol=1e-10)


def n_for_d(p1, d, alpha=ALPHA, power=POWER_TARGET):
    """Minimum per-arm n solving power_for_d(p1, d, n) == power, via bisection on n."""
    f = lambda n: power_for_d(p1, d, n, alpha) - power
    lo, hi = 4.0, 1e8
    return brentq(f, lo, hi, xtol=1e-6)


def monte_carlo_check(p1, d, n, alpha=ALPHA, n_sims=10000, seed=SEED):
    """
    Validation-only cross-check of the closed form: simulate n_sims two-arm trials
    with true rates p1, p1+d and n per arm, run the (pooled-SE) two-proportion
    z-test at the given alpha on each, and report the empirical rejection rate
    (= simulated power). NOT used anywhere as the primary MDE method.
    """
    rng = np.random.default_rng(seed)
    p2 = p1 + d
    x1 = rng.binomial(n, p1, size=n_sims)
    x2 = rng.binomial(n, p2, size=n_sims)
    phat1, phat2 = x1 / n, x2 / n
    pbar = (x1 + x2) / (2 * n)
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (phat2 - phat1) / se0
    za = stats.norm.ppf(1 - alpha / 2)
    reject = np.abs(z) >= za
    return reject.mean()


def main():
    p1_stretch = 45 / 300
    baseline_recs = pc.load_baseline()
    p1_baseline = sum(pc.is_success(r) for r in baseline_recs.values()) / len(baseline_recs)
    assert abs(p1_stretch - 0.15) < 1e-9

    p1_defs = [
        ("p1_stretchA_0.15", p1_stretch, "primary",
         "stretchA transform-arm SR (45/300 = 0.1500 exactly) -- the operative rate "
         "in this study's transform-sweep regime"),
        ("p1_baseline_0.18", p1_baseline, "secondary",
         f"computed full-benchmark baseline SR (pc.load_baseline(), "
         f"{sum(pc.is_success(r) for r in baseline_recs.values())}/{len(baseline_recs)} = "
         f"{p1_baseline:.5f}, ~0.18 per the task brief)"),
    ]

    print(f"analysis4_mde_power.py  (alpha={ALPHA}, power={POWER_TARGET}, seed={SEED})\n")
    print("Baselines:")
    for name, val, role, desc in p1_defs:
        print(f"  {name} = {val:.5f} ({role}) -- {desc}")

    rows = []
    print(f"\n{'p1':>8s} {'n/arm':>7s} {'MDE(pp)':>9s} {'p2':>8s} {'rel.effect':>11s}")
    mde_grid = {}
    for name, p1, role, desc in p1_defs:
        mde_grid[name] = {}
        for n in N_LIST:
            d = mde(p1, n)
            p2 = p1 + d
            rel = d / p1
            mde_grid[name][n] = d
            note = ("formula: unpooled-variance two-proportion power/MDE, alpha=0.05, "
                    "power=0.80, solved for d via bisection (see module docstring)")
            rows.append([f"{p1:.5f}", n, ALPHA, POWER_TARGET, f"{100*d:.4f}",
                         f"{p2:.5f}", f"{rel:.4f}", "unpooled-variance z-test (bisection)",
                         f"{name}: {desc}; {note}"])
            print(f"{p1:8.4f} {n:7d} {100*d:9.3f} {p2:8.4f} {rel:11.3f}")

    # ---- sanity anchor ------------------------------------------------------
    d300 = mde_grid["p1_stretchA_0.15"][300]
    print(f"\nSANITY ANCHOR: MDE at n=300, p1=0.15 = {100*d300:.2f} pp "
          f"(pre-registration expects ~8-10 pp; "
          f"{'OK, within expected band' if 8.0 <= 100*d300 <= 10.0 else 'OUT OF EXPECTED BAND -- investigate'}).")

    # ---- MDE at the actual experiment sizes used in this study --------------
    print("\nMDE at the actual per-arm sizes used in this study "
          "(primary baseline p1=0.15):")
    for label, n in EXPERIMENT_N.items():
        d = mde(p1_stretch, n)
        print(f"  {label} (n={n}): MDE = {100*d:.2f} pp")
        rows.append([f"{p1_stretch:.5f}", n, ALPHA, POWER_TARGET, f"{100*d:.4f}",
                     f"{p1_stretch+d:.5f}", f"{d/p1_stretch:.4f}",
                     "unpooled-variance z-test (bisection)",
                     f"actual per-arm n used for {label} in this study"])

    # ---- n required for 5pp and 2pp at p1=0.15 -------------------------------
    print("\nScaling cost at p1=0.15 (primary baseline): n per arm required to "
          "reliably detect a fixed absolute difference:")
    for d_target in [0.05, 0.02]:
        n_req = n_for_d(p1_stretch, d_target)
        print(f"  detect {100*d_target:.0f} pp difference -> n per arm = {n_req:.0f} "
              f"(rounded up: {int(np.ceil(n_req))})")
        rows.append([f"{p1_stretch:.5f}", int(np.ceil(n_req)), ALPHA, POWER_TARGET,
                     f"{100*d_target:.4f}", f"{p1_stretch+d_target:.5f}",
                     f"{d_target/p1_stretch:.4f}",
                     "unpooled-variance z-test (bisection on n)",
                     f"derived block: minimum n per arm to detect a fixed {100*d_target:.0f}pp "
                     "absolute difference at power=0.80, alpha=0.05, p1=0.15"])

    header = ["p1", "n_per_arm", "alpha", "power", "mde_pp", "implied_p2",
              "relative_effect", "method", "notes"]
    out_csv = pc.write_csv(os.path.join(pc.TABLES, "mde_power.csv"), header, rows)
    print(f"\nwrote {out_csv}")

    # ---- Monte Carlo cross-check (validation only) ---------------------------
    print("\nMonte Carlo cross-check of the closed form (validation only, "
          f"10,000 sims each, seed={SEED}):")
    check_points = [
        (p1_stretch, mde_grid["p1_stretchA_0.15"][300], 300),
        (p1_stretch, mde_grid["p1_stretchA_0.15"][750], 750),
        (p1_baseline, mde_grid["p1_baseline_0.18"][300], 300),
    ]
    for p1, d, n in check_points:
        sim_power = monte_carlo_check(p1, d, n)
        print(f"  p1={p1:.4f} d={100*d:.2f}pp n={n}: closed-form power={POWER_TARGET:.2f}, "
              f"simulated power={sim_power:.3f} "
              f"({'OK' if abs(sim_power-POWER_TARGET) < 0.03 else 'CHECK -- gap > 0.03'})")

    # ---- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = {"p1_stretchA_0.15": "#1b1b1b", "p1_baseline_0.18": "#4477aa"}
    labels = {"p1_stretchA_0.15": "p1 = 0.15 (stretchA)",
              "p1_baseline_0.18": f"p1 = {p1_baseline:.3f} (full baseline)"}
    for name, p1, role, desc in p1_defs:
        ys = [100 * mde_grid[name][n] for n in N_LIST]
        ax.plot(N_LIST, ys, marker="o", markersize=3, lw=1.2,
                color=colors[name], label=labels[name])
    ax.set_xscale("log")
    ymax = max(100 * mde_grid[name][N_LIST[0]] for name, *_ in p1_defs) * 1.08
    ax.set_ylim(0, ymax)
    # Both curves sit well above y~9 pp at x=200/300, and the legend sits in the
    # upper-right at y~20+, so anchoring these labels flush to the x-axis (non-rotated,
    # divergent horizontal alignment so the two labels grow away from each other)
    # keeps them clear of both the curves and the legend.
    text_spec = {200: ("height arms\nn=200", "right"),
                 300: ("transform arms\nn=300", "left")}
    for label, n in EXPERIMENT_N.items():
        ax.axvline(n, color="0.75", lw=0.8, ls="--", zorder=0)
        txt, ha = text_spec[n]
        ax.text(n, 0.3, txt, fontsize=6, color="0.35", ha=ha, va="bottom")
    ax.set_xlabel("n per arm")
    ax.set_ylabel("minimum detectable difference (pp)")
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST], rotation=45, ha="right")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    pdf_path = os.path.join(pc.FIGURES, "mde_power.pdf")
    png_path = os.path.join(pc.FIGURES, "mde_power.png")
    os.makedirs(pc.FIGURES, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"\nwrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
