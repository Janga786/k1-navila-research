#!/usr/bin/env python3
"""
analysis2_censoring.py — differential compute (wall-clock) censoring analysis.

INPUTS (read-only):
  receipts/sweep_measurements/{stretchA,stretchB,crop,pad,h060,h078,h095,h110,h130,h150}/*.json
  All access goes through paper_evidence/scripts/paper_common.py (ARMS, load_arm,
  is_success, is_timeout). No receipt file is written to.

OUTPUTS:
  paper_evidence/tables/censoring_summary.csv
  paper_evidence/tables/censoring_bounds.csv
  paper_evidence/tables/censoring_tests.csv
  paper_evidence/figures/censoring_bounds.pdf
  paper_evidence/figures/censoring_bounds.png
  paper_evidence/figures/censoring_timeout_rate.pdf
  paper_evidence/figures/censoring_timeout_rate.png

METHOD:
  Every episode is killed at a FIXED 900 s wall-clock budget
  (paper_common.WALL_CLOCK_BUDGET_S) and recorded as a failure
  (term_reason == "wall_timeout", success/spl forced to 0). This is a compute
  limit, not a navigation judgment. For each of the 10 arms in
  paper_common.ARMS we report:
    - intention-to-treat (ITT) SR: timeouts counted as failures (PREFERRED).
    - a completed-episode SENSITIVITY estimate: SR after dropping timed-out
      episodes (biased upward -- slow episodes are plausibly the harder ones).
    - partial-identification bounds over the full n: lower = ITT SR (every
      timeout assumed a failure, by construction), upper = (succ+timeouts)/n
      (every timeout assumed a would-have-succeeded).
  Fisher's exact test (two-sided) is used for: crop-vs-pad timeout rate
  (primary confound test), stretchA-vs-stretchB timeout rate (a same-condition
  replicate control), and crop-vs-pad SR under ITT. A differential-censoring
  decomposition (d_ITT, d_completed, attenuation fraction) is reported
  descriptively only -- no causal language.

SEED: numpy.random.default_rng(20260911). (No resampling is required for any
  number here -- all quantities are exact counts/Fisher tests -- but the seed
  is fixed per the task spec in case any RNG use is added.)
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import (  # noqa: E402
    ARMS, FIGURES, TABLES, TRANSFORM_ARMS, WALL_CLOCK_BUDGET_S,
    is_success, is_timeout, load_arm, write_csv,
)

SEED = 20260911
RNG = np.random.default_rng(SEED)  # reserved; no stochastic step is currently needed


def arm_counts(tag):
    recs = load_arm(tag)
    n = len(recs)
    succ = sum(1 for r in recs.values() if is_success(r))
    n_timeout = sum(1 for r in recs.values() if is_timeout(r))
    fail = n - succ
    return n, succ, fail, n_timeout


def main():
    print("=" * 78)
    print(f"ANALYSIS 2 — differential compute censoring (wall budget = "
          f"{WALL_CLOCK_BUDGET_S:.0f}s)")
    print("=" * 78)

    per_arm = {}
    for tag in ARMS:
        n, succ, fail, n_timeout = arm_counts(tag)
        n_completed = n - n_timeout
        sr_itt = succ / n
        sr_completed = succ / n_completed if n_completed > 0 else float("nan")
        lower = succ / n
        upper = (succ + n_timeout) / n
        per_arm[tag] = dict(
            n=n, succ=succ, fail=fail, n_timeout=n_timeout, n_completed=n_completed,
            sr_itt=sr_itt, sr_completed=sr_completed, lower=lower, upper=upper,
        )

    # ------------------------------------------------------------------
    # Print + build summary / bounds tables
    # ------------------------------------------------------------------
    summary_rows = []
    bounds_rows = []
    print(f"{'arm':10s} {'n':>4s} {'succ':>5s} {'fail':>5s} {'timeout':>7s} "
          f"{'to%':>6s} {'SR_ITT%':>8s} {'SR_comp%':>9s} {'shift_pp':>8s}")
    for tag, (_, _, label) in ARMS.items():
        d = per_arm[tag]
        timeout_pct = 100 * d["n_timeout"] / d["n"]
        sr_itt_pct = 100 * d["sr_itt"]
        sr_completed_pct = 100 * d["sr_completed"]
        shift_pp = sr_completed_pct - sr_itt_pct
        print(f"{tag:10s} {d['n']:4d} {d['succ']:5d} {d['fail']:5d} {d['n_timeout']:7d} "
              f"{timeout_pct:6.2f} {sr_itt_pct:8.2f} {sr_completed_pct:9.2f} {shift_pp:8.2f}")

        summary_rows.append((
            tag, label, d["n"], d["succ"], d["fail"], d["n_timeout"],
            f"{timeout_pct:.4f}", f"{sr_itt_pct:.4f}", f"{sr_completed_pct:.4f}",
            d["n_completed"], f"{shift_pp:+.4f}",
            "ITT (timeout=failure) is the PREFERRED estimate. sr_completed is a "
            "SENSITIVITY ANALYSIS ONLY: it drops timed-out episodes, which are "
            "plausibly the harder/slower episodes, so dropping them inflates SR "
            "(shift_pp is positive for every arm here, consistent with that bias)."
        ))

        bound_width_pp = 100 * (d["upper"] - d["lower"])
        bounds_rows.append((
            tag, d["n"], d["succ"], d["n_timeout"], f"{sr_itt_pct:.4f}",
            f"{100*d['lower']:.4f}", f"{100*d['upper']:.4f}", f"{bound_width_pp:.4f}",
            "Partial-identification bounds over full n: lower = succ/n assumes every "
            "timed-out episode would have failed; upper = (succ+timeout)/n assumes "
            "every timed-out episode would have succeeded. lower_bound_pct == sr_itt_pct "
            "by construction (both count all timeouts as failures)."
        ))

    write_csv(
        os.path.join(TABLES, "censoring_summary.csv"),
        ["arm", "label", "n", "successes", "failures", "n_timeout", "timeout_pct",
         "sr_itt_pct", "sr_completed_pct", "n_completed", "shift_pp", "notes"],
        summary_rows,
    )
    write_csv(
        os.path.join(TABLES, "censoring_bounds.csv"),
        ["arm", "n", "successes", "n_timeout", "sr_itt_pct", "lower_bound_pct",
         "upper_bound_pct", "bound_width_pp", "notes"],
        bounds_rows,
    )

    # ------------------------------------------------------------------
    # Historical cross-check (ROBUSTNESS.md section B)
    # ------------------------------------------------------------------
    hist_itt = {"stretchA": 15.0, "stretchB": 16.0, "crop": 16.7, "pad": 13.3,
                "h060": 11.5, "h078": 12.5, "h095": 18.5, "h110": 13.0,
                "h130": 15.0, "h150": 14.5}
    hist_timeout_rate = {"stretchA": 6.7, "stretchB": 7.3, "crop": 2.0, "pad": 12.3,
                          "h060": 5.0, "h078": 6.5, "h095": 4.5, "h110": 3.0,
                          "h130": 7.5, "h150": 6.5}
    print()
    print("Cross-check vs ROBUSTNESS.md section B (rounded to 1 dp there):")
    any_mismatch = False
    for tag in ARMS:
        d = per_arm[tag]
        got_itt = round(100 * d["sr_itt"], 1)
        got_to = round(100 * d["n_timeout"] / d["n"], 1)
        ok_itt = abs(got_itt - hist_itt[tag]) < 0.05
        ok_to = abs(got_to - hist_timeout_rate[tag]) < 0.05
        if not (ok_itt and ok_to):
            any_mismatch = True
            print(f"  *** MISMATCH {tag}: ITT got {got_itt} vs hist {hist_itt[tag]}; "
                  f"timeout% got {got_to} vs hist {hist_timeout_rate[tag]} ***")
    if not any_mismatch:
        print("  All 10 arms match ROBUSTNESS.md section B ITT SR and timeout rate.")

    # ------------------------------------------------------------------
    # Fisher tests
    # ------------------------------------------------------------------
    test_rows = []

    def fisher_2x2(a, b, c, d_, label_note):
        table = [[a, b], [c, d_]]
        odds_ratio, p = stats.fisher_exact(table, alternative="two-sided")
        return odds_ratio, p

    # (1) crop vs pad timeout rate: 2x2 = [timeout, not-timeout] x [crop, pad]
    dc, dp = per_arm["crop"], per_arm["pad"]
    a, b = dc["n_timeout"], dc["n"] - dc["n_timeout"]   # crop: timeout, not-timeout
    c, d_ = dp["n_timeout"], dp["n"] - dp["n_timeout"]  # pad:  timeout, not-timeout
    or_ct, p_ct = fisher_2x2(a, b, c, d_, "crop-vs-pad timeout rate")
    risk_diff_ct = 100 * (dc["n_timeout"] / dc["n"] - dp["n_timeout"] / dp["n"])
    print()
    print(f"Fisher exact, crop-vs-pad TIMEOUT RATE: table=[[{a},{b}],[{c},{d_}]] "
          f"OR={or_ct:.4f} p={p_ct:.3e} risk_diff={risk_diff_ct:+.2f}pp")
    if not (p_ct < 1e-5):
        print(f"  *** NOTE: historical p ~= 7e-07; got {p_ct:.3e} ***")
    test_rows.append((
        "timeout_rate_fisher", "crop", "pad", a, b, c, d_,
        f"{or_ct:.6f}", f"{p_ct:.6e}", f"{risk_diff_ct:+.4f}",
        "2x2 cells are [timeout, not-timeout] for arm1 (a,b) then arm2 (c,d); "
        "risk_diff_pp = 100*(timeout_rate(arm1)-timeout_rate(arm2)); "
        "historically p~=7e-07, this is far more significant than any SR difference "
        "in the sweep."
    ))

    # (2) stretchA vs stretchB timeout rate (control: identical config replicate)
    dA, dB = per_arm["stretchA"], per_arm["stretchB"]
    a2, b2 = dA["n_timeout"], dA["n"] - dA["n_timeout"]
    c2, d2 = dB["n_timeout"], dB["n"] - dB["n_timeout"]
    or_ab, p_ab = fisher_2x2(a2, b2, c2, d2, "stretchA-vs-stretchB timeout rate")
    risk_diff_ab = 100 * (dA["n_timeout"] / dA["n"] - dB["n_timeout"] / dB["n"])
    print(f"Fisher exact, stretchA-vs-stretchB TIMEOUT RATE (control, same condition): "
          f"table=[[{a2},{b2}],[{c2},{d2}]] OR={or_ab:.4f} p={p_ab:.4f} "
          f"risk_diff={risk_diff_ab:+.2f}pp")
    test_rows.append((
        "timeout_rate_fisher", "stretchA", "stretchB", a2, b2, c2, d2,
        f"{or_ab:.6f}", f"{p_ab:.6e}", f"{risk_diff_ab:+.4f}",
        "Control comparison: stretchA/B are the identical configuration run twice, "
        "so this Fisher test on timeout rate shows whether the timeout mechanism "
        "itself is stable run-to-run (not a behavioural/config difference)."
    ))

    # (3) crop vs pad SR under ITT
    a3, b3 = dc["succ"], dc["n"] - dc["succ"]
    c3, d3 = dp["succ"], dp["n"] - dp["succ"]
    or_sr, p_sr = fisher_2x2(a3, b3, c3, d3, "crop-vs-pad ITT SR")
    risk_diff_sr = 100 * (dc["succ"] / dc["n"] - dp["succ"] / dp["n"])
    print(f"Fisher exact, crop-vs-pad ITT SR: table=[[{a3},{b3}],[{c3},{d3}]] "
          f"OR={or_sr:.4f} p={p_sr:.4f} risk_diff={risk_diff_sr:+.2f}pp")
    test_rows.append((
        "sr_itt_fisher", "crop", "pad", a3, b3, c3, d3,
        f"{or_sr:.6f}", f"{p_sr:.6e}", f"{risk_diff_sr:+.4f}",
        "2x2 cells are [successes, failures] for arm1 (a,b) then arm2 (c,d) under ITT "
        "(timeout counted as failure). Not statistically significant."
    ))

    # ------------------------------------------------------------------
    # Differential-censoring decomposition (descriptive, not causal)
    # ------------------------------------------------------------------
    d_itt = 100 * (dc["sr_itt"] - dp["sr_itt"])
    d_completed = 100 * (dc["sr_completed"] - dp["sr_completed"])
    attenuation_frac = (d_itt - d_completed) / d_itt if d_itt != 0 else float("nan")
    attenuation_pct = 100 * attenuation_frac

    print()
    print("Differential-censoring decomposition (crop vs pad), DESCRIPTIVE ONLY:")
    print(f"  SR_ITT(crop)       = {100*dc['sr_itt']:.4f}%   "
          f"SR_ITT(pad)       = {100*dp['sr_itt']:.4f}%")
    print(f"  SR_completed(crop) = {100*dc['sr_completed']:.4f}%   "
          f"SR_completed(pad) = {100*dp['sr_completed']:.4f}%")
    print(f"  d_ITT       = SR_ITT(crop) - SR_ITT(pad)             = {d_itt:+.4f} pp")
    print(f"  d_completed = SR_completed(crop) - SR_completed(pad) = {d_completed:+.4f} pp")
    print(f"  attenuation = (d_ITT - d_completed) / d_ITT          = {attenuation_frac:+.4f} "
          f"= {attenuation_pct:+.2f}%")
    print("  Interpretation: this is the share of the observed ITT gap that does not")
    print("  survive restriction to completed episodes. It is NOT a causal decomposition:")
    print(f"  (i) neither gap is statistically significant (crop-vs-pad ITT SR Fisher "
          f"p={p_sr:.4f}); (ii) the completed-episode restriction is itself a biased")
    print("  comparison (drops plausibly-harder slow episodes differentially by arm),")
    print("  so this decomposition is descriptive, not causal.")

    test_rows.append((
        "differential_censoring_decomposition", "crop", "pad", "", "", "", "",
        "", "", f"{d_itt:+.4f}",
        f"d_ITT={d_itt:+.4f}pp (=risk_diff_sr above); d_completed={d_completed:+.4f}pp; "
        f"attenuation=(d_ITT-d_completed)/d_ITT={attenuation_frac:+.6f} "
        f"({attenuation_pct:+.2f}%) = share of the observed ITT gap that does not "
        "survive restriction to completed episodes. NOT a causal claim: neither gap is "
        f"statistically significant (ITT SR Fisher p={p_sr:.4f}), and the "
        "completed-episode restriction is itself a biased comparison (drops "
        "plausibly-harder slow episodes differentially by arm's timeout rate)."
    ))

    write_csv(
        os.path.join(TABLES, "censoring_tests.csv"),
        ["test", "arm1", "arm2", "a", "b", "c", "d", "odds_ratio", "p_fisher",
         "risk_diff_pp", "notes"],
        test_rows,
    )

    # ------------------------------------------------------------------
    # FIGURES
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

    height_arms = [t for t in ARMS if t not in TRANSFORM_ARMS]
    all_arms_order = TRANSFORM_ARMS + height_arms
    labels_all = [ARMS[t][2] for t in all_arms_order]

    # censoring_bounds figure: transform arms only (priority), all 10 stay readable too
    # -> try all 10; fall back logic not needed here since 10 bars is fine at 7.16in width.
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    x = np.arange(len(all_arms_order))
    itt_pts = [100 * per_arm[t]["sr_itt"] for t in all_arms_order]
    lo_pts = [100 * per_arm[t]["lower"] for t in all_arms_order]
    hi_pts = [100 * per_arm[t]["upper"] for t in all_arms_order]
    yerr_lo = [itt - lo for itt, lo in zip(itt_pts, lo_pts)]
    yerr_hi = [hi - itt for itt, hi in zip(itt_pts, hi_pts)]

    colors = ["#2c7fb8" if t in TRANSFORM_ARMS else "#969696" for t in all_arms_order]
    ax.errorbar(x, itt_pts, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="0.3",
                elinewidth=1.0, capsize=3, zorder=1)
    ax.scatter(x, itt_pts, s=22, c=colors, zorder=2, label="ITT SR (point)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_arms_order, rotation=30, ha="right")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, max(hi_pts) * 1.15)
    ax.axvline(3.5, color="0.85", lw=0.8, zorder=0)
    ax.text(1.5, max(hi_pts) * 1.08, "transform arms", ha="center", fontsize=7, color="0.3")
    ax.text(3.5 + (len(all_arms_order) - 3.5) / 2, max(hi_pts) * 1.08, "height arms",
            ha="center", fontsize=7, color="0.3")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"censoring_bounds.{ext}"))
    plt.close(fig)

    # separate timeout-rate figure
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.4))
    to_pct = [100 * per_arm[t]["n_timeout"] / per_arm[t]["n"] for t in all_arms_order]
    ax2.bar(x, to_pct, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_arms_order, rotation=30, ha="right")
    ax2.set_ylabel("wall-timeout rate (%)")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(FIGURES, f"censoring_timeout_rate.{ext}"))
    plt.close(fig2)

    print()
    print("Wrote:")
    for p in ["censoring_summary.csv", "censoring_bounds.csv", "censoring_tests.csv"]:
        print(" ", os.path.join(TABLES, p))
    for p in ["censoring_bounds.pdf", "censoring_bounds.png",
              "censoring_timeout_rate.pdf", "censoring_timeout_rate.png"]:
        print(" ", os.path.join(FIGURES, p))


if __name__ == "__main__":
    main()
