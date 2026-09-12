# PAPER_EVIDENCE — result-to-source map

Repo commit for every entry: `a6fe40b431353b0eee968c07015dd8d1b1211d53`
Working tree at generation time: `?? paper_evidence/`

Raw experimental data was never modified; verified by checksum comparison before and
after the analysis (see `FINAL_REPORT.md`).

Per-file SHA-256 for every raw input: `provenance/input_sha256.txt` (with per-directory rollup hashes at the top).

---

**RESULT:** Baseline benchmark performance (SR / OSR / NE / SPL, n=1077)  
**INPUT FILE(S):** `receipts/baseline_full_14498/*.json (1077 files)`  
**SCRIPT:** `paper_evidence/scripts/build_paper_docs.py (headline) + scripts/reproduction_check.py`  
**OUTPUT FILE:** `paper_evidence/tables/headline_metrics.csv`  
**SAMPLE SIZE:** 1077 episodes  
**STATISTICAL METHOD:** Direct aggregation; SR/OSR/SPL over full denominator; NE excludes -1.0 sentinel  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED from raw data; matches the historical record to the reported precision

**RESULT:** Transform-arm success and timeout rates (4 arms)  
**INPUT FILE(S):** `receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/build_paper_docs.py, scripts/analysis2_censoring.py`  
**OUTPUT FILE:** `paper_evidence/tables/headline_metrics.csv, tables/censoring_summary.csv`  
**SAMPLE SIZE:** 300 episodes per arm  
**STATISTICAL METHOD:** Direct aggregation, intention-to-treat (timeout = failure)  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED from raw data; matches SWEEP_STATUS.md and ROBUSTNESS.md

**RESULT:** Paired repeatability 2x2 contingency (stretchA vs stretchB)  
**INPUT FILE(S):** `receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis1_repeatability.py`  
**OUTPUT FILE:** `paper_evidence/tables/repeatability_contingency.csv`  
**SAMPLE SIZE:** 300 paired episodes (identical record_idx sets, verified)  
**STATISTICAL METHOD:** Exact paired cross-tabulation on record_idx  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED; matches ROBUSTNESS.md section A exactly

**RESULT:** Cohen's kappa and confidence interval  
**INPUT FILE(S):** `receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis1_repeatability.py`  
**OUTPUT FILE:** `paper_evidence/tables/repeatability_statistics.csv`  
**SAMPLE SIZE:** 300 paired episodes  
**STATISTICAL METHOD:** Cohen's kappa; asymptotic Fleiss/Cohen SE interval + episode bootstrap (10 000 resamples, seed 20260911) + scene-cluster bootstrap (8 scenes)  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator; point estimate matches ROBUSTNESS.md

**RESULT:** Run-to-run outcome discordance rate  
**INPUT FILE(S):** `receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis1_repeatability.py`  
**OUTPUT FILE:** `paper_evidence/tables/repeatability_statistics.csv`  
**SAMPLE SIZE:** 300 paired episodes  
**STATISTICAL METHOD:** (n10+n01)/n, with scene-cluster bootstrap CI  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED; matches ROBUSTNESS.md section A

**RESULT:** Exact McNemar test on discordant pairs  
**INPUT FILE(S):** `receipts/sweep_measurements/stretchA_300/*.json, stretchB_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis1_repeatability.py`  
**OUTPUT FILE:** `paper_evidence/tables/repeatability_statistics.csv`  
**SAMPLE SIZE:** 33 discordant pairs of 300  
**STATISTICAL METHOD:** Exact two-sided binomial test, p=0.5  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator

**RESULT:** Wall-clock timeout rates by arm  
**INPUT FILE(S):** `receipts/sweep_measurements/*/*.json (all 10 arms)`  
**SCRIPT:** `paper_evidence/scripts/analysis2_censoring.py`  
**OUTPUT FILE:** `paper_evidence/tables/censoring_summary.csv`  
**SAMPLE SIZE:** 300 per transform arm, 200 per height arm  
**STATISTICAL METHOD:** Count of term_reason == 'wall_timeout' under a fixed 900 s budget  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED; matches ROBUSTNESS.md section B

**RESULT:** Partial-identification bounds on SR under censoring  
**INPUT FILE(S):** `receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json and the six height arms`  
**SCRIPT:** `paper_evidence/scripts/analysis2_censoring.py`  
**OUTPUT FILE:** `paper_evidence/tables/censoring_bounds.csv`  
**SAMPLE SIZE:** 300 per transform arm, 200 per height arm  
**STATISTICAL METHOD:** Assumption-free bounds: lower = succ/n, upper = (succ + n_timeout)/n  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** NEW ANALYSIS; arithmetic verified independently by the orchestrator

**RESULT:** crop vs pad timeout-rate comparison  
**INPUT FILE(S):** `receipts/sweep_measurements/crop_300/*.json, pad_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis2_censoring.py`  
**OUTPUT FILE:** `paper_evidence/tables/censoring_tests.csv`  
**SAMPLE SIZE:** 300 + 300  
**STATISTICAL METHOD:** Fisher exact test (two-sided) with odds ratio and risk difference  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED; matches the historical p ~ 7e-07

**RESULT:** Differential-censoring decomposition of the crop-vs-pad gap  
**INPUT FILE(S):** `receipts/sweep_measurements/crop_300/*.json, pad_300/*.json`  
**SCRIPT:** `paper_evidence/scripts/analysis2_censoring.py`  
**OUTPUT FILE:** `paper_evidence/tables/censoring_tests.csv`  
**SAMPLE SIZE:** 300 + 300  
**STATISTICAL METHOD:** (d_ITT - d_completed)/d_ITT; DESCRIPTIVE, explicitly not causal  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator

**RESULT:** Scene-aware vs naive confidence intervals  
**INPUT FILE(S):** `receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json joined to receipts/analysis/episode_index.csv`  
**SCRIPT:** `paper_evidence/scripts/analysis3_scene_uncertainty.py`  
**OUTPUT FILE:** `paper_evidence/tables/scene_uncertainty.csv`  
**SAMPLE SIZE:** 300 episodes per arm in 8 scenes  
**STATISTICAL METHOD:** Wilson score interval vs cluster bootstrap resampling SCENES (10 000 replicates, seed 20260911)  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator; consistent with ROBUSTNESS.md section D

**RESULT:** Per-scene success rates  
**INPUT FILE(S):** `receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json joined to receipts/analysis/episode_index.csv`  
**SCRIPT:** `paper_evidence/scripts/analysis3_scene_uncertainty.py`  
**OUTPUT FILE:** `paper_evidence/tables/scene_per_scene_sr.csv`  
**SAMPLE SIZE:** 300 episodes per arm across 8 scenes  
**STATISTICAL METHOD:** Stratified aggregation by scene  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** REPRODUCED; consistent with ROBUSTNESS.md section D

**RESULT:** Clustering diagnostics (ICC, design effect, effective n)  
**INPUT FILE(S):** `receipts/sweep_measurements/{stretchA,stretchB,crop,pad}_300/*.json joined to receipts/analysis/episode_index.csv`  
**SCRIPT:** `paper_evidence/scripts/analysis3_scene_uncertainty.py`  
**OUTPUT FILE:** `paper_evidence/tables/scene_uncertainty.csv`  
**SAMPLE SIZE:** 8 scenes per arm  
**STATISTICAL METHOD:** One-way ANOVA method-of-moments ICC with unequal cluster sizes; DEFF = 1 + (m0 - 1) * ICC  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator; UNSTABLE at k=8 — report as indicative

**RESULT:** Minimum detectable difference / statistical resolution  
**INPUT FILE(S):** `No experimental input (analytic); p1 anchored to measured SRs via paper_common`  
**SCRIPT:** `paper_evidence/scripts/analysis4_mde_power.py`  
**OUTPUT FILE:** `paper_evidence/tables/mde_power.csv`  
**SAMPLE SIZE:** n per arm from 50 to 2000 (planning curve)  
**STATISTICAL METHOD:** Two-proportion z-test power, two-sided alpha=0.05, power=0.80, unpooled variance  
**GIT COMMIT:** `a6fe40b431353b0eee968c07015dd8d1b1211d53`  
**VERIFICATION STATUS:** VERIFIED independently by the orchestrator; consistent with the pre-registered ~9 pp resolution at n=300

---

## SHA-256 of generated tables and figures

| file | sha256 (first 12) | bytes |
|---|---|---|
| `tables/censoring_bounds.csv` | `caea9a1b4798` | 3296 |
| `tables/censoring_summary.csv` | `82e1cb4c78ae` | 3521 |
| `tables/censoring_tests.csv` | `588b62515521` | 1367 |
| `tables/headline_metrics.csv` | `502b5b0709aa` | 487 |
| `tables/mde_power.csv` | `3ab5ceee8cb6` | 6842 |
| `tables/repeatability_contingency.csv` | `cb395861735f` | 207 |
| `tables/repeatability_statistics.csv` | `a41eacb333d1` | 1900 |
| `tables/scene_per_scene_sr.csv` | `e4bb02aec279` | 2537 |
| `tables/scene_uncertainty.csv` | `c53275b4e2b6` | 4748 |
| `figures/censoring_bounds.pdf` | `19e640720a25` | 12572 |
| `figures/censoring_bounds.png` | `40947e85d298` | 63235 |
| `figures/censoring_timeout_rate.pdf` | `d652abcf8c63` | 11618 |
| `figures/censoring_timeout_rate.png` | `7a6e41a71e08` | 41074 |
| `figures/mde_power.pdf` | `6f314f4b2f16` | 12675 |
| `figures/mde_power.png` | `04f5f47a7760` | 98053 |
| `figures/repeatability_paired.pdf` | `0f35e40c5074` | 12650 |
| `figures/repeatability_paired.png` | `5062a82bcd4e` | 75363 |
| `figures/scene_uncertainty.pdf` | `cd3bae5332db` | 12747 |
| `figures/scene_uncertainty.png` | `acb70e08a81a` | 68251 |

