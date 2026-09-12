# FINAL_REPORT — AIRC 2027 reanalysis audit

Compiled 2026-09-11 at repo commit `a6fe40b431353b0eee968c07015dd8d1b1211d53`.

Scope: reanalysis of experiments **already run**. No new simulation, no retraining, no
sim-to-real, no GPU job, no change to benchmark semantics. All numbers below are read from
the generated tables under `paper_evidence/tables/` — see `PAPER_RESULTS.md` for the
results section itself and `PAPER_EVIDENCE.md` for the result-to-source map.

---

## 1. Did the historical results reproduce?

**Yes — 27/27 independently recomputed metric checks MATCH, zero mismatches.**

Recomputed directly from the raw per-episode JSON, independent of the committed scripts:

| metric | claimed | regenerated |
|---|---|---|
| baseline n / SR / OSR / NE / SPL | 1077 / 18.3 % / 30.3 % / 7.59 m / 10.93 % | 1077 / 18.29 % / 30.27 % / 7.591 m / 10.93 % |
| stretchA SR (timeout rate) | 15.0 % (6.67 %) | 15.00 % (6.67 %) |
| stretchB SR (timeout rate) | 16.0 % (7.33 %) | 16.00 % (7.33 %) |
| crop SR (timeout rate) | 16.7 % (2.0 %) | 16.67 % (2.00 %) |
| pad SR (timeout rate) | 13.3 % (12.3 %) | 13.33 % (12.33 %) |
| six height arms (n=200) | 11.5 / 12.5 / 18.5 / 13.0 / 15.0 / 14.5 % | all exact |
| repeatability 2x2 | 30 / 15 / 18 / 237 | 30 / 15 / 18 / 237 |
| Cohen's kappa | ~0.580 | 0.5802 |

Rerunning the two committed scripts (`analyze_sweep.py`, `analyze_robustness.py`) reproduced
`ANALYSIS.md`, `ROBUSTNESS.md`, `contrasts.csv` and `episode_index.csv` **byte-for-byte**.
`arm_summary.csv` differed only past the 10th significant digit (see caveat 10.1). It was
restored; the tracked tree is clean.

## 2. Exact stretchA / stretchB contingency counts

n = 300 paired episodes, paired on `record_idx` (identical episode sets, verified).

| | stretchB success | stretchB failure | total |
|---|---|---|---|
| **stretchA success** | 30 | 15 | 45 |
| **stretchA failure** | 18 | 237 | 255 |
| **total** | 48 | 252 | 300 |

## 3. Cohen's kappa and its CI

kappa = **0.5802**; 95 % CI **[0.4515, 0.7088]** (asymptotic Fleiss/Cohen SE) and
**[0.4410, 0.6979]** (episode bootstrap, 10 000 resamples, seed 20260911).
Scene-cluster bootstrap: [0.4937, 0.6610] — narrower, for the reason given in caveat 10.4.

Report alongside kappa, not instead of it: positive agreement **0.645**, negative agreement
**0.935**. Kappa is prevalence-sensitive, and high agreement on failure is largely a
consequence of failure being common.

## 4. Discordance rate

**11.00 %** (33/300 episodes changed outcome between two nominally identical runs); raw
paired agreement 89.00 %. Scene-cluster 95 % CI [7.53 %, 14.67 %].

The sharper statement for the paper: of the **63** episodes that succeeded in *either*
replicate, only **30 (47.6 %)** succeeded in *both*.

## 5. What McNemar's test shows

Exact two-sided binomial test on the 33 discordant pairs (15 success-to-fail vs 18
fail-to-success): **p = 0.7283**. No directional bias — the instability is symmetric, as
expected for two runs of the same configuration. McNemar shows the arms are not *shifted*
relative to each other; it does **not** show the outcomes are stable. That is precisely the
point: aggregate rates reproduce while per-episode outcomes do not.

## 6. Timeout rates by arm (fixed 900 s wall-clock budget)

stretchA 20/300 = 6.67 % | stretchB 22/300 = 7.33 % | crop 6/300 = 2.00 % | pad 37/300 = 12.33 %
height arms: h060 5.00 %, h078 6.50 %, h095 4.50 %, h110 3.00 %, h130 7.50 %, h150 6.50 %.

crop vs pad: odds ratio 0.1451, **Fisher exact p = 7.04e-07**, risk difference -10.33 pp.
Control (stretchA vs stretchB, identical configuration): OR 0.9026, p = 0.8731, -0.67 pp —
the timeout mechanism itself is stable run-to-run, so the crop-vs-pad gap is a property of
the arms rather than run-to-run noise.

## 7. Censoring bounds

Assumption-free partial identification over the full denominator: lower = all timeouts are
failures (equals the ITT SR by construction), upper = all timeouts are successes.

| arm | timeouts | SR lower % | SR upper % | width (pp) |
|---|---|---|---|---|
| stretchA | 20 | 15.00 | 21.67 | 6.67 |
| stretchB | 22 | 16.00 | 23.33 | 7.33 |
| crop | 6 | 16.67 | 18.67 | 2.00 |
| pad | 37 | 13.33 | 25.67 | 12.33 |

The bounds overlap heavily across arms: pad's [13.33, 25.67] contains crop's entire interval.
No transform-arm ranking survives the censoring uncertainty.

Descriptive decomposition (**not** causal): d_ITT = +3.3333 pp, d_completed = +1.7977 pp,
attenuation = (d_ITT - d_completed)/d_ITT = **46.07 %** — the share of the observed ITT
crop-vs-pad gap that does not survive restriction to completed episodes. Neither gap is
statistically significant (ITT SR Fisher p = 0.3035), and the completed-episode restriction
is itself biased, so this is a caution against reading the ITT gap as purely behavioural —
not an estimate of a causal effect of censoring.

## 8. Scene-aware vs naive intervals

| arm | SR % | naive Wilson 95 % CI | scene-bootstrap 95 % CI | width ratio |
|---|---|---|---|---|
| stretchA | 15.00 | [11.40, 19.48] | [10.49, 19.91] | 1.17x |
| stretchB | 16.00 | [12.29, 20.57] | [8.60, 24.07] | 1.87x |
| crop | 16.67 | [12.88, 21.30] | [12.00, 19.92] | 0.94x |
| pad | 13.33 | [9.95, 17.65] | [5.75, 21.93] | 2.10x |

Resampling **scenes** (not episodes) with replacement, 10 000 replicates. The effect is
large but **not uniform**: pad's interval roughly doubles and stretchB's nearly doubles,
while crop's is marginally narrower. Acknowledging clustering does not inflate every
interval; it makes the interval reflect how unevenly an arm's successes fall across scenes.
Per-scene SR ranges from 0 % to 44 % within a single arm.

Clustering diagnostics are **indicative only** (k = 8 clusters, sizes 9-84): ICC 0.020-0.209,
design effect 1.65-7.94, effective n 38-182 for the four primary arms. Two secondary height
arms yield slightly negative method-of-moments ICCs. Do not lead with these; lead with the
directly observable change in interval width.

## 9. Detectable effect size at the sizes actually used

Two-sided alpha = 0.05, power = 0.80, two-proportion z-test, p1 = 0.15:

- **n = 300 per arm (transform arms): 9.05 pp**
- **n = 200 per arm (height arms): 11.31 pp**

(At p1 = 0.183: 9.62 pp and 11.98 pp.) Detecting 5 pp would need n = 906 per arm; 2 pp would
need n = 5274. Every transform-arm contrast observed here is a few pp — inside the region
this design cannot resolve. The nulls are statements about **statistical resolution**, not
evidence of equivalence.

## 10. Discrepancies and caveats that must be disclosed

**10.1 Floating-point drift in `arm_summary.csv` on rerun.** Rerunning `analyze_sweep.py`
today reproduces every committed output byte-for-byte except `arm_summary.csv`, which differs
in the 10th-plus significant digit of some CI endpoints (numpy 1.26.4/scipy 1.15.3 originally
vs numpy 2.4.6/scipy 1.17.1 now). No reported value changes at the precision anything is
quoted to. The file was restored to the committed state. Disclose as a reproducibility note.

**10.2 stretchA's raw invocation log does not survive.** `sweep_run_log.txt` begins after
stretchA had already completed; the earlier `/tmp` log was lost in the 2026-07-28..31 power
outage. stretchA's configuration is proven from `arms_runner.sh`'s single unconditional code
path, the identical `ARMS` array entry, the identical checkpoint hash
(`8e7559b5...`, mtime predating both runs) and the identical 300-episode set, and is
corroborated by the contemporaneous `SWEEP_STATUS.md` line — but unlike stretchB there is no
surviving byte-level invocation record. Describe the pair as "identically configured per the
runner's single code path", not "byte-identical invocations verified from logs".

**10.3 Execution-environment equivalence is asserted, not instrumented.** Both arms are
believed to run on driver 580.173.02 from the upgrade/apt-hold timeline; no per-arm
`nvidia-smi` capture exists, and no concurrent-load or thermal equivalence is logged. This
slightly weakens, but does not invalidate, the "same-driver replicate" framing.

**10.4 The scene-cluster CI for kappa is narrower than the episode-level CI.** This is
expected, not a bug (independently reproduced): cluster resampling preserves each scene's
internal composition, and per-scene kappa is fairly homogeneous. Quote the asymptotic or
episode-bootstrap interval as the primary CI for kappa; present the scene-cluster one as a
robustness check, never as the conservative bound.

**10.5 Only 8 scenes, unevenly sized.** The pinned 300 episodes cover 8 of the benchmark's
11 scenes, with counts 9-84 and one scene alone contributing 28 %. The subset is *not*
scene-representative. ICC / design-effect / effective-n are unstable at k = 8 and must be
labelled indicative.

**10.6 The baseline spans a driver change.** The n=1077 baseline ran on driver 580.159.03
(since purged); every sweep arm ran on 580.173.02. Baseline-vs-sweep comparisons cross that
change. stretchA-vs-stretchB does not, which is why the repeatability claim rests on it.

**10.7 Two distinct budgets — do not conflate.** `--max_episode_s 120` is *simulated* time
(6000 sim steps, `term_reason=step_cap`); `EP_TIMEOUT=900` is the *wall-clock* kill
(`term_reason=wall_timeout`). Only the latter is the censoring mechanism analysed here.

**10.8 Nine records carry a schema anomaly.** `max_episode_steps == -1` / `ended_at_step == -1`
in stretchA (3), stretchB (4), h078 (1), h130 (1), from a runner-side wall-timeout stub. These
are **the same records** that carry the `-1.0` distance sentinel — one phenomenon, not two.
Scoring is unaffected; any analysis of `ended_at_step` must exclude them.

**10.9 A prior claim in older project documents is contradicted by the n=300 data.**
`FINAL_RESULTS_full1077.md` and `SLIDE_FACTS.md` state, from an n=10 pilot, that crop reaches
the goal but fails to stop (a peripheral-FOV arrival-recognition mechanism). At n=300 the
effect runs the other way (`ROBUSTNESS.md` section C). That mechanism must not be reused in
the manuscript.

**10.10 Terminology.** Call the run-to-run effect *outcome instability* / *irreproducibility*,
not "randomness". Call the MDE *minimum detectable difference* or *statistical resolution*,
never a "noise floor". Note that the *historical* scripts and their captured stdout under
`provenance/` do use the older term; those captures are verbatim evidence and were left
unedited, but the manuscript must not inherit the wording from them.

## 11. Manuscript-ready figures and tables

All under `paper_evidence/`. Figures are vector PDF plus 300 dpi PNG, IEEE two-column styling.

**Figures**
| file | use |
|---|---|
| `figures/repeatability_paired.pdf` | paired outcome categories (identical-run instability) |
| `figures/censoring_bounds.pdf` | SR partial-identification bounds per arm (priority) |
| `figures/censoring_timeout_rate.pdf` | timeout rate per arm (companion) |
| `figures/scene_uncertainty.pdf` | naive vs scene-aware 95 % CIs |
| `figures/mde_power.pdf` | minimum detectable difference vs n per arm |

**Tables**
| file | use |
|---|---|
| `tables/headline_metrics.csv` | reproduced baseline + transform-arm metrics |
| `tables/repeatability_contingency.csv` | raw 2x2 counts |
| `tables/repeatability_statistics.csv` | kappa, CIs, McNemar, agreement indices |
| `tables/censoring_summary.csv` | per-arm ITT and completed-episode SR |
| `tables/censoring_bounds.csv` | partial-identification bounds |
| `tables/censoring_tests.csv` | Fisher tests, OR, decomposition |
| `tables/scene_uncertainty.csv` | naive vs scene-bootstrap CIs, ICC/DEFF/n_eff |
| `tables/scene_per_scene_sr.csv` | per-scene SR by arm |
| `tables/mde_power.csv` | MDE curve |

## 12. Audit checklist

| check | result |
|---|---|
| Full pipeline rerun from scratch (tables/ and figures/ deleted first) | PASS — all 9 tables + 10 figure files regenerated |
| Outputs deterministic across runs | PASS — all 9 CSVs byte-identical on re-run (seed 20260911) |
| No raw experimental file modified | PASS — 3,498 `receipts/` files byte-identical to the pre-work SHA-256 snapshot |
| `git status` / `git diff` | PASS — tracked tree clean; only untracked `paper_evidence/` |
| All paper numbers machine-generated | PASS — `build_paper_docs.py` reads every value from a generated CSV |
| Correct episode pairing | PASS — paired on `record_idx`; `paired_episodes()` hard-fails on asymmetry; sets verified identical |
| Scene bootstrap resamples scenes, not episodes | PASS — draws indices into the scene-key list only; asserts scene count; independently reproduced by the orchestrator |
| Timeout bounds mathematically correct | PASS — lower = succ/n, upper = (succ+n_timeout)/n; independently recomputed |
| No large GPU experiment launched | PASS — CPU only; the sole GPU process is the pre-existing idle VLM bridge (PID 10831, started 2026-08-04, 0 % util), untouched |
| Analysis scripts cannot write to `receipts/` | PASS — every write path resolves under `paper_evidence/tables` or `figures` |
