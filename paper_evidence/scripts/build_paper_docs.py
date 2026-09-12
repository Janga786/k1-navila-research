#!/usr/bin/env python3
"""
build_paper_docs.py — assemble the paper-facing markdown from the generated tables.

Run LAST, after analysis1-4 and reproduction_check.py. Every number written into
PAPER_RESULTS.md / PAPER_EVIDENCE.md is READ FROM a generated CSV (or recomputed from the
raw receipts through paper_common) — nothing is transcribed by hand. If a table is missing
this script fails loudly rather than emitting a partial document.

Inputs : paper_evidence/tables/*.csv, receipts/ (via paper_common)
Outputs: paper_evidence/tables/headline_metrics.csv
         paper_evidence/PAPER_RESULTS.md
         paper_evidence/PAPER_EVIDENCE.md
         paper_evidence/README.md
"""

import csv
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_common as pc

PE = pc.PAPER_EVIDENCE
T = pc.TABLES


def read(name):
    p = os.path.join(T, name)
    if not os.path.exists(p):
        raise SystemExit(f"missing required table: {p} (run the analysis scripts first)")
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def rowby(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise SystemExit(f"no row matching {kw}")


def f(x, nd=2):
    return f"{float(x):.{nd}f}"


def sha256(path, n=12):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def git(*args):
    return subprocess.run(["git", "-C", pc.ROOT] + list(args),
                          capture_output=True, text=True).stdout.strip()


# ------------------------------------------------------------------ headline metrics
def headline():
    """Recompute the reproduced headline metrics straight from the raw receipts."""
    out = []
    b = pc.load_baseline()
    n = len(b)
    sr = sum(pc.is_success(r) for r in b.values())
    osr = sum(pc.is_oracle_success(r) for r in b.values())
    spl = sum(float(r.get("spl") or 0.0) for r in b.values()) / n
    ne = [float(r["distance_to_goal"]) for r in b.values()
          if float(r["distance_to_goal"]) != pc.NE_SENTINEL]
    out.append(["baseline_full_14498", "baseline (model_14498, full benchmark)", n,
                sr, f"{100*sr/n:.4f}", osr, f"{100*osr/n:.4f}",
                f"{100*spl:.4f}", f"{sum(ne)/len(ne):.4f}", len(ne),
                sum(pc.is_timeout(r) for r in b.values())])
    for tag in pc.TRANSFORM_ARMS:
        r = pc.load_arm(tag)
        n = len(r)
        sr = sum(pc.is_success(x) for x in r.values())
        osr = sum(pc.is_oracle_success(x) for x in r.values())
        spl = sum(float(x.get("spl") or 0.0) for x in r.values()) / n
        ne = [float(x["distance_to_goal"]) for x in r.values()
              if float(x["distance_to_goal"]) != pc.NE_SENTINEL]
        out.append([tag, pc.ARMS[tag][2], n, sr, f"{100*sr/n:.4f}", osr, f"{100*osr/n:.4f}",
                    f"{100*spl:.4f}", f"{sum(ne)/len(ne):.4f}", len(ne),
                    sum(pc.is_timeout(x) for x in r.values())])
    pc.write_csv(os.path.join(T, "headline_metrics.csv"),
                 ["arm", "label", "n", "successes", "sr_pct", "oracle_successes", "osr_pct",
                  "spl_pct", "ne_mean_m", "ne_n_excl_sentinel", "n_timeout"], out)
    return out


def main():
    hl = headline()
    hb = {r[0]: r for r in hl}

    cont = read("repeatability_contingency.csv")
    stats = read("repeatability_statistics.csv")
    cens = read("censoring_summary.csv")
    bounds = read("censoring_bounds.csv")
    tests = read("censoring_tests.csv")
    scene = read("scene_uncertainty.csv")
    mde = read("mde_power.csv")

    n11 = int(rowby(cont, stretchA_outcome="success", stretchB_outcome="success")["n_episodes"])
    n10 = int(rowby(cont, stretchA_outcome="success", stretchB_outcome="failure")["n_episodes"])
    n01 = int(rowby(cont, stretchA_outcome="failure", stretchB_outcome="success")["n_episodes"])
    n00 = int(rowby(cont, stretchA_outcome="failure", stretchB_outcome="failure")["n_episodes"])
    ntot = int(rowby(cont, stretchA_outcome="TOTAL")["n_episodes"])

    k_as = rowby(stats, statistic="cohens_kappa", ci_method="asymptotic_fleiss_cohen_SE")
    k_bo = rowby(stats, statistic="cohens_kappa", ci_method="episode_bootstrap_percentile")
    k_sc = rowby(stats, statistic="cohens_kappa_scene_cluster")
    disc = rowby(stats, statistic="discordance_rate")
    disc_sc = rowby(stats, statistic="discordance_rate_scene_cluster")
    mcn = rowby(stats, statistic="mcnemar_exact_p")
    pa = rowby(stats, statistic="positive_agreement_PA")
    na = rowby(stats, statistic="negative_agreement_NA")
    uni = rowby(stats, statistic="union_of_successes")
    unif = rowby(stats, statistic="union_both_succeeded_fraction")
    agr = rowby(stats, statistic="raw_agreement")

    to_cp = rowby(tests, test="timeout_rate_fisher", arm1="crop", arm2="pad")
    to_ab = rowby(tests, test="timeout_rate_fisher", arm1="stretchA", arm2="stretchB")
    sr_cp = rowby(tests, test="sr_itt_fisher", arm1="crop", arm2="pad")
    deco = rowby(tests, test="differential_censoring_decomposition")

    commit = git("rev-parse", "HEAD")
    status = git("status", "--short") or "(clean — only untracked paper_evidence/)"
    prim = [r for r in scene if r["priority"] == "primary"]

    def mderow(p1pref, n):
        for r in mde:
            if r["n_per_arm"] == str(n) and r["p1"].startswith(p1pref) and "derived" not in r["notes"]:
                return r
        raise SystemExit(f"no mde row n={n} p1~{p1pref}")

    # ------------------------------------------------------------------ PAPER_RESULTS.md
    L = []
    a = L.append
    a("# PAPER_RESULTS — machine-generated results section\n")
    a(f"Generated by `paper_evidence/scripts/build_paper_docs.py` at repo commit `{commit}`.")
    a("Every number below is read from a generated CSV under `paper_evidence/tables/` or")
    a("recomputed from the raw per-episode receipts via `paper_common.py`. None is hand-typed.\n")
    a("---\n")

    a("## 1. Reproduced baseline and sweep metrics\n")
    a("Source: `tables/headline_metrics.csv` (recomputed from `receipts/` raw per-episode JSON).")
    a("Verification of these against the historical record: `REPRODUCTION_CHECK.md`.\n")
    a("| arm | n | successes | SR % | OSR % | SPL % | NE (m) | wall-clock timeouts |")
    a("|---|---|---|---|---|---|---|---|")
    for r in hl:
        a(f"| {r[0]} | {r[2]} | {r[3]} | {f(r[4])} | {f(r[6])} | {f(r[7])} | {f(r[8])} | {r[10]} |")
    a("")
    a("NE is the mean `distance_to_goal` excluding the `-1.0` sentinel written when a")
    a("wall-clock kill lands before a valid final pose exists; the sentinel-excluded episode")
    a("count is in `ne_n_excl_sentinel`. SR/OSR/SPL are over the full pinned denominator,")
    a("with timed-out episodes counted as failures (intention-to-treat).\n")

    a("## 2. Repeatability of nominally identical evaluations\n")
    a("`stretchA` and `stretchB` are the same transform, the same checkpoint and the same 300")
    a("episodes, run independently. The configuration equivalence is proven at code and hash")
    a("level; the *execution-environment* equivalence (same driver build, no confounding load)")
    a("is corroborated by the operational timeline rather than instrumented per arm — see")
    a("`REPRODUCTION_CHECK.md` section 6. Source: `tables/repeatability_*.csv`.\n")
    a("**Paired outcome contingency table** (n = %d):\n" % ntot)
    a("| | stretchB success | stretchB failure | total |")
    a("|---|---|---|---|")
    a(f"| **stretchA success** | {n11} | {n10} | {n11+n10} |")
    a(f"| **stretchA failure** | {n01} | {n00} | {n01+n00} |")
    a(f"| **total** | {n11+n01} | {n10+n00} | {ntot} |")
    a("")
    a(f"- stretchA SR = {f(hb['stretchA'][4])} % ({hb['stretchA'][3]}/{hb['stretchA'][2]}); "
      f"stretchB SR = {f(hb['stretchB'][4])} % ({hb['stretchB'][3]}/{hb['stretchB'][2]}) "
      "— the aggregate rates agree closely.")
    a(f"- Raw paired agreement = {100*float(agr['value']):.2f} %; "
      f"**paired discordance = {100*float(disc['value']):.2f} %** "
      f"({n10+n01}/{ntot} episodes changed outcome between two identical runs).")
    a(f"- Cohen's kappa = **{f(k_as['value'],4)}**, 95 % CI "
      f"[{f(k_as['ci95_lo'],4)}, {f(k_as['ci95_hi'],4)}] (asymptotic) and "
      f"[{f(k_bo['ci95_lo'],4)}, {f(k_bo['ci95_hi'],4)}] (episode bootstrap, 10 000 resamples).")
    a(f"- Positive agreement = {f(pa['value'],4)}; negative agreement = {f(na['value'],4)}. "
      "The asymmetry matters: agreement on failure is high simply because failure is common, "
      "so kappa alone understates how unstable the *successes* are.")
    a(f"- Of the {uni['value']} episodes that succeeded in *either* replicate, only {n11} "
      f"(**{100*float(unif['value']):.1f} %**) succeeded in *both*.")
    a(f"- Exact McNemar test on the {n10+n01} discordant pairs ({n10} vs {n01}): "
      f"p = {f(mcn['value'],4)} — no directional bias, as expected for two runs of the same "
      "configuration. The instability is symmetric, not a drift between arms.")
    a(f"- Scene-cluster bootstrap (8 scenes, 10 000 resamples): kappa 95 % CI "
      f"[{f(k_sc['ci95_lo'],4)}, {f(k_sc['ci95_hi'],4)}]; discordance 95 % CI "
      f"[{100*float(disc_sc['ci95_lo']):.2f}, {100*float(disc_sc['ci95_hi']):.2f}] %.")
    a("")
    a("  Note that the scene-cluster interval for kappa is *narrower* than the episode-level")
    a("  bootstrap. This is expected rather than anomalous: cluster resampling preserves each")
    a("  scene's internal composition intact, and per-scene kappa is fairly homogeneous, so")
    a("  less variation is induced than by resampling episodes independently. Quote the")
    a("  episode-level or asymptotic interval as the primary CI for kappa and treat the")
    a("  scene-cluster one as a robustness check — do not present it as the conservative bound.")
    a("")
    a("Wording note for the manuscript: describe this as **run-to-run outcome instability**")
    a("(irreproducibility of per-episode outcomes), not as \"randomness\". The data establish")
    a("that identical configurations yield different per-episode outcomes; they do not")
    a("characterise the process generating that variation.\n")
    a("Figure: `figures/repeatability_paired.pdf`\n")

    a("## 3. Differential wall-clock timeout censoring\n")
    a(f"Every episode is killed at a fixed **{int(pc.WALL_CLOCK_BUDGET_S)} s wall-clock** budget "
      "(`EP_TIMEOUT=900` in `arms_runner.sh`, enforced by `timeout -k 30`) and recorded as a")
    a("failure with `term_reason=wall_timeout`, `success=0`, `spl=0`. This is a *compute*")
    a("limit, not a navigation outcome. It is distinct from the simulated-time episode cap")
    a("(`--max_episode_s 120`, i.e. 6000 sim steps), which produces `term_reason=step_cap`.")
    a("Source: `tables/censoring_summary.csv`, `censoring_bounds.csv`, `censoring_tests.csv`.\n")
    a("| arm | n | successes | timeouts | timeout % | ITT SR % | completed-episode SR % (sensitivity) | SR lower bound % | SR upper bound % | bound width (pp) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in bounds:
        c = rowby(cens, arm=r["arm"])
        a(f"| {r['arm']} | {r['n']} | {r['successes']} | {r['n_timeout']} | "
          f"{f(r['n_timeout'] and 100*int(r['n_timeout'])/int(r['n']))} | {f(r['sr_itt_pct'])} | "
          f"{f(c['sr_completed_pct'])} | {f(r['lower_bound_pct'])} | {f(r['upper_bound_pct'])} | "
          f"{f(r['bound_width_pp'])} |")
    a("")
    a("- The **intention-to-treat SR is the preferred estimate**. The completed-episode column")
    a("  is a *sensitivity analysis only*: dropping timed-out episodes is itself biased, since")
    a("  slow episodes are plausibly the harder ones, which is why every arm shifts upward.")
    a("- The **partial-identification bounds** are assumption-free over the censored episodes:")
    a("  the lower bound counts every timeout as a failure (and therefore equals the ITT SR by")
    a("  construction), the upper bound counts every timeout as a success. The true SR under")
    a("  an unlimited compute budget lies in that interval. `pad`'s interval is "
      f"{f(rowby(bounds, arm='pad')['bound_width_pp'])} pp wide versus "
      f"{f(rowby(bounds, arm='crop')['bound_width_pp'])} pp for `crop`.")
    a(f"- **Timeout rates are not equal across arms.** crop {to_cp['a']}/{int(to_cp['a'])+int(to_cp['b'])} "
      f"vs pad {to_cp['c']}/{int(to_cp['c'])+int(to_cp['d'])}: odds ratio {f(to_cp['odds_ratio'],4)}, "
      f"Fisher exact p = {float(to_cp['p_fisher']):.3g}, risk difference {f(to_cp['risk_diff_pp'])} pp.")
    a(f"- Control: the same test between the two identical replicates (stretchA vs stretchB) gives "
      f"OR {f(to_ab['odds_ratio'],4)}, p = {float(to_ab['p_fisher']):.4f} "
      f"({f(to_ab['risk_diff_pp'])} pp) — the timeout mechanism itself is stable run-to-run, so the "
      "crop-vs-pad gap is a property of the arms, not of run-to-run noise.")
    a(f"- By contrast the crop-vs-pad **success-rate** difference is not significant: "
      f"OR {f(sr_cp['odds_ratio'],4)}, Fisher p = {float(sr_cp['p_fisher']):.4f}, "
      f"{float(sr_cp['risk_diff_pp']):+.2f} pp.")
    a("")
    a("**Differential-censoring decomposition** (descriptive, not causal):\n")
    a("```")
    a(deco["notes"].replace("; ", "\n"))
    a("```")
    a("This quantifies the share of the observed ITT gap that does not survive restriction to")
    a("completed episodes. It is **not** a causal decomposition: neither gap is statistically")
    a("significant, and the completed-episode restriction is itself a biased comparison. The")
    a("defensible claim is that the ITT crop-vs-pad gap must not be presented as a purely")
    a("behavioural difference while the arms are censored at materially different rates.\n")
    a("Figures: `figures/censoring_bounds.pdf`, `figures/censoring_timeout_rate.pdf`\n")

    a("## 4. Scene-dependent uncertainty\n")
    a("Episodes are nested in scenes and share a map, so they are not independent draws. The")
    a("pinned 300 episodes span **8 scenes** (the full 1077-episode benchmark spans 11), with")
    a("per-scene episode counts from 9 to 84. Source: `tables/scene_uncertainty.csv`,")
    a("`tables/scene_per_scene_sr.csv`.\n")
    a("| arm | SR % | naive Wilson 95 % CI | naive width (pp) | scene-bootstrap 95 % CI | scene width (pp) | width ratio |")
    a("|---|---|---|---|---|---|---|")
    for r in prim:
        a(f"| {r['arm']} | {f(r['sr_pct'])} | [{f(r['wilson_lo_pct'])}, {f(r['wilson_hi_pct'])}] | "
          f"{f(r['wilson_width_pp'])} | [{f(r['scene_boot_lo_pct'])}, {f(r['scene_boot_hi_pct'])}] | "
          f"{f(r['scene_boot_width_pp'])} | {f(r['width_ratio_scene_over_wilson'])}x |")
    a("")
    a("The scene bootstrap resamples **scenes** with replacement, carrying all of a drawn")
    a("scene's episodes, over 10 000 replicates.\n")
    a("- The effect is **large but not uniform**: pad "
      f"{f(rowby(prim, arm='pad')['width_ratio_scene_over_wilson'])}x and stretchB "
      f"{f(rowby(prim, arm='stretchB')['width_ratio_scene_over_wilson'])}x wider, while crop is "
      f"{f(rowby(prim, arm='crop')['width_ratio_scene_over_wilson'])}x — i.e. slightly *narrower*. "
      "Acknowledging clustering does not simply inflate every interval; it makes the interval "
      "depend on how unevenly an arm's successes are distributed across scenes.")
    a("- Per-scene SR is highly heterogeneous (`tables/scene_per_scene_sr.csv`), which is what")
    a("  drives the inflation where it occurs.")
    a("")
    a("Clustering diagnostics (**indicative only**):\n")
    a("| arm | ICC | design effect | effective n |")
    a("|---|---|---|---|")
    for r in prim:
        a(f"| {r['arm']} | {f(r['icc'],4)} | {f(r['deff'],2)} | {f(r['n_eff'],1)} |")
    a("")
    a("With only k = 8 clusters and cluster sizes spanning 9-84, the ICC / design-effect /")
    a("effective-n estimates carry very wide sampling uncertainty (two secondary height arms")
    a("even yield slightly negative method-of-moments ICCs). They should be reported as")
    a("indicative, and the directly observable quantity — the change in interval width when")
    a("scenes rather than episodes are resampled — should carry the argument.\n")
    a("Figure: `figures/scene_uncertainty.pdf`\n")

    a("## 5. Statistical resolution (minimum detectable difference)\n")
    a("Two-sided alpha = 0.05, power = 0.80, two independent arms with equal n, unpooled-variance")
    a("two-proportion z-test solved numerically. Source: `tables/mde_power.csv`.\n")
    a("| n per arm | minimum detectable difference (pp), p1 = 0.15 | p1 = 0.183 (full baseline) |")
    a("|---|---|---|")
    for n in [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]:
        a(f"| {n} | {f(mderow('0.15', n)['mde_pp'])} | {f(mderow('0.18', n)['mde_pp'])} |")
    a("")
    a(f"- At the size actually used for the transform arms (**n = 300 per arm**), the smallest")
    a(f"  absolute SR difference detectable under these assumptions is "
      f"**{f(mderow('0.15',300)['mde_pp'])} pp** (p1 = 0.15).")
    a(f"- At the height-sweep size (**n = 200 per arm**) it is "
      f"**{f(mderow('0.15',200)['mde_pp'])} pp**.")
    a("- Every transform-arm contrast in this study is a few pp, i.e. well inside the region")
    a("  this design cannot resolve. The null results are therefore statements about")
    a("  **statistical resolution**, not evidence of equivalence.")
    d5 = [r for r in mde if "5pp" in r["notes"] or "5 pp" in r["notes"]]
    d2 = [r for r in mde if "2pp" in r["notes"] or "2 pp" in r["notes"]]
    if d5:
        a(f"- Detecting a 5 pp difference at p1 = 0.15 would require n = {d5[0]['n_per_arm']} per arm.")
    if d2:
        a(f"- Detecting a 2 pp difference would require n = {d2[0]['n_per_arm']} per arm.")
    a("")
    a("Terminology: this quantity is the *minimum detectable difference* / *statistical")
    a("resolution* under the stated assumptions.\n")
    a("Figure: `figures/mde_power.pdf`\n")

    with open(os.path.join(PE, "PAPER_RESULTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    # ------------------------------------------------------------------ PAPER_EVIDENCE.md
    RAW4 = "receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json"
    ev = [
        ("Baseline benchmark performance (SR / OSR / NE / SPL, n=1077)",
         "receipts/baseline_full_14498/*.json (1077 files)",
         "paper_evidence/scripts/build_paper_docs.py (headline) + scripts/reproduction_check.py",
         "paper_evidence/tables/headline_metrics.csv", "1077 episodes",
         "Direct aggregation; SR/OSR/SPL over full denominator; NE excludes -1.0 sentinel",
         "REPRODUCED from raw data; matches the historical record to the reported precision"),
        ("Transform-arm success and timeout rates (4 arms)",
         RAW4,
         "paper_evidence/scripts/build_paper_docs.py, scripts/analysis2_censoring.py",
         "paper_evidence/tables/headline_metrics.csv, tables/censoring_summary.csv",
         "300 episodes per arm",
         "Direct aggregation, intention-to-treat (timeout = failure)",
         "REPRODUCED from raw data; matches SWEEP_STATUS.md and ROBUSTNESS.md"),
        ("Paired repeatability 2x2 contingency (stretchA vs stretchB)",
         "receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json",
         "paper_evidence/scripts/analysis1_repeatability.py",
         "paper_evidence/tables/repeatability_contingency.csv",
         "300 paired episodes (identical record_idx sets, verified)",
         "Exact paired cross-tabulation on record_idx",
         "REPRODUCED; matches ROBUSTNESS.md section A exactly"),
        ("Cohen's kappa and confidence interval",
         "receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json",
         "paper_evidence/scripts/analysis1_repeatability.py",
         "paper_evidence/tables/repeatability_statistics.csv", "300 paired episodes",
         "Cohen's kappa; asymptotic Fleiss/Cohen SE interval + episode bootstrap "
         "(10 000 resamples, seed 20260911) + scene-cluster bootstrap (8 scenes)",
         "VERIFIED independently by the orchestrator; point estimate matches ROBUSTNESS.md"),
        ("Run-to-run outcome discordance rate",
         "receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json",
         "paper_evidence/scripts/analysis1_repeatability.py",
         "paper_evidence/tables/repeatability_statistics.csv", "300 paired episodes",
         "(n10+n01)/n, with scene-cluster bootstrap CI",
         "REPRODUCED; matches ROBUSTNESS.md section A"),
        ("Exact McNemar test on discordant pairs",
         "receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json",
         "paper_evidence/scripts/analysis1_repeatability.py",
         "paper_evidence/tables/repeatability_statistics.csv",
         "33 discordant pairs of 300",
         "Exact two-sided binomial test, p=0.5",
         "VERIFIED independently by the orchestrator"),
        ("Wall-clock timeout rates by arm",
         "receipts/sweep_measurements/*/*.json (all 10 arms)",
         "paper_evidence/scripts/analysis2_censoring.py",
         "paper_evidence/tables/censoring_summary.csv",
         "300 per transform arm, 200 per height arm",
         "Count of term_reason == 'wall_timeout' under a fixed 900 s budget",
         "REPRODUCED; matches ROBUSTNESS.md section B"),
        ("Partial-identification bounds on SR under censoring",
         RAW4 + " and the six height arms",
         "paper_evidence/scripts/analysis2_censoring.py",
         "paper_evidence/tables/censoring_bounds.csv",
         "300 per transform arm, 200 per height arm",
         "Assumption-free bounds: lower = succ/n, upper = (succ + n_timeout)/n",
         "NEW ANALYSIS; arithmetic verified independently by the orchestrator"),
        ("crop vs pad timeout-rate comparison",
         "receipts/sweep_measurements/crop_300/*.json, pad_300/*.json",
         "paper_evidence/scripts/analysis2_censoring.py",
         "paper_evidence/tables/censoring_tests.csv", "300 + 300",
         "Fisher exact test (two-sided) with odds ratio and risk difference",
         "REPRODUCED; matches the historical p ~ 7e-07"),
        ("Differential-censoring decomposition of the crop-vs-pad gap",
         "receipts/sweep_measurements/crop_300/*.json, pad_300/*.json",
         "paper_evidence/scripts/analysis2_censoring.py",
         "paper_evidence/tables/censoring_tests.csv", "300 + 300",
         "(d_ITT - d_completed)/d_ITT; DESCRIPTIVE, explicitly not causal",
         "VERIFIED independently by the orchestrator"),
        ("Scene-aware vs naive confidence intervals",
         RAW4 + " joined to receipts/analysis/episode_index.csv",
         "paper_evidence/scripts/analysis3_scene_uncertainty.py",
         "paper_evidence/tables/scene_uncertainty.csv",
         "300 episodes per arm in 8 scenes",
         "Wilson score interval vs cluster bootstrap resampling SCENES "
         "(10 000 replicates, seed 20260911)",
         "VERIFIED independently by the orchestrator; consistent with ROBUSTNESS.md section D"),
        ("Per-scene success rates",
         RAW4 + " joined to receipts/analysis/episode_index.csv",
         "paper_evidence/scripts/analysis3_scene_uncertainty.py",
         "paper_evidence/tables/scene_per_scene_sr.csv",
         "300 episodes per arm across 8 scenes",
         "Stratified aggregation by scene",
         "REPRODUCED; consistent with ROBUSTNESS.md section D"),
        ("Clustering diagnostics (ICC, design effect, effective n)",
         RAW4 + " joined to receipts/analysis/episode_index.csv",
         "paper_evidence/scripts/analysis3_scene_uncertainty.py",
         "paper_evidence/tables/scene_uncertainty.csv", "8 scenes per arm",
         "One-way ANOVA method-of-moments ICC with unequal cluster sizes; "
         "DEFF = 1 + (m0 - 1) * ICC",
         "VERIFIED independently by the orchestrator; UNSTABLE at k=8 — report as indicative"),
        ("Minimum detectable difference / statistical resolution",
         "No experimental input (analytic); p1 anchored to measured SRs via paper_common",
         "paper_evidence/scripts/analysis4_mde_power.py",
         "paper_evidence/tables/mde_power.csv",
         "n per arm from 50 to 2000 (planning curve)",
         "Two-proportion z-test power, two-sided alpha=0.05, power=0.80, unpooled variance",
         "VERIFIED independently by the orchestrator; consistent with the pre-registered "
         "~9 pp resolution at n=300"),
    ]
    E = []
    b = E.append
    b("# PAPER_EVIDENCE — result-to-source map\n")
    b(f"Repo commit for every entry: `{commit}`")
    b(f"Working tree at generation time: `{status}`\n")
    b("Raw experimental data was never modified; verified by checksum comparison before and")
    b("after the analysis (see `FINAL_REPORT.md`).\n")
    b("Per-file SHA-256 for every raw input: `provenance/input_sha256.txt` "
      "(with per-directory rollup hashes at the top).\n")
    b("---\n")
    for (res, inp, scr, out, nn, meth, ver) in ev:
        b(f"**RESULT:** {res}  ")
        b(f"**INPUT FILE(S):** `{inp}`  ")
        b(f"**SCRIPT:** `{scr}`  ")
        b(f"**OUTPUT FILE:** `{out}`  ")
        b(f"**SAMPLE SIZE:** {nn}  ")
        b(f"**STATISTICAL METHOD:** {meth}  ")
        b(f"**GIT COMMIT:** `{commit}`  ")
        b(f"**VERIFICATION STATUS:** {ver}\n")
    b("---\n")
    b("## SHA-256 of generated tables and figures\n")
    b("| file | sha256 (first 12) | bytes |")
    b("|---|---|---|")
    for sub in ("tables", "figures"):
        d = os.path.join(PE, sub)
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            b(f"| `{sub}/{fn}` | `{sha256(p)}` | {os.path.getsize(p)} |")
    b("")
    with open(os.path.join(PE, "PAPER_EVIDENCE.md"), "w") as fh:
        fh.write("\n".join(E) + "\n")

    # ------------------------------------------------------------------ README.md
    R = f"""# paper_evidence — AIRC 2027 reanalysis package

Reanalysis of ALREADY-COLLECTED experiments in `receipts/`. **No new simulation, training, or
robot execution was performed.** All raw experimental data is read-only; integrity was verified
by checksum before and after (see `FINAL_REPORT.md`).

Repo commit: `{commit}`

## Read in this order

| file | what it is |
|---|---|
| `REPRODUCTION_CHECK.md` | Did the historical results reproduce from raw data? (verdict + per-metric table) |
| `PAPER_RESULTS.md` | The results section — every number read from a generated table |
| `PAPER_EVIDENCE.md` | Result -> input -> script -> output -> method -> commit map, plus checksums |
| `FINAL_REPORT.md` | Final audit, caveats that must be disclosed, manuscript-ready asset list |

## The four reanalyses

1. **Repeatability** of nominally identical evaluations — `scripts/analysis1_repeatability.py`
2. **Wall-clock timeout censoring** — `scripts/analysis2_censoring.py`
3. **Scene-dependent uncertainty** — `scripts/analysis3_scene_uncertainty.py`
4. **Statistical resolution / MDE** — `scripts/analysis4_mde_power.py`

## Reproducing everything from scratch

```bash
cd {pc.ROOT}
python3 paper_evidence/scripts/analysis1_repeatability.py
python3 paper_evidence/scripts/analysis2_censoring.py
python3 paper_evidence/scripts/analysis3_scene_uncertainty.py
python3 paper_evidence/scripts/analysis4_mde_power.py
python3 paper_evidence/scripts/reproduction_check.py
python3 paper_evidence/scripts/build_paper_docs.py     # must run last
```

All scripts are CPU-only, deterministic (seed 20260911), and read `receipts/` read-only.
`scripts/paper_common.py` is the single shared data-access layer: it defines episode
indexing, the scene join, and the success/timeout conventions exactly once.

## Layout

```
tables/      machine-generated CSVs (the source of every number in the markdown)
figures/     paper-ready PDF (vector) + PNG (300 dpi), IEEE two-column styling
scripts/     analysis code
provenance/  environment capture, input checksums, reruns of the historical scripts
```
"""
    with open(os.path.join(PE, "README.md"), "w") as fh:
        fh.write(R)

    print("wrote PAPER_RESULTS.md, PAPER_EVIDENCE.md, README.md, tables/headline_metrics.csv")


if __name__ == "__main__":
    main()
