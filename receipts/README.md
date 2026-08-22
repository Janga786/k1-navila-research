# Evidence index — what the paper claims, and where the proof lives

Entry point for the sim-to-real VLN + embodiment-effects paper. Every number in the paper
should be traceable to something in this directory. Written 2026-08-22, after the ten-arm
sweep completed.

**Read `KNOWN_GAPS.md` first.** It lists, adversarially, the places this evidence base does
*not* hold up. Two paper-facing mechanistic claims are currently unsupported; the primary
results are fully backed.

---

## Reproduce every reported number

```bash
python receipts/analysis/analyze_sweep.py        # -> ANALYSIS.md, arm_summary.csv, contrasts.csv
python receipts/analysis/analyze_robustness.py   # -> ROBUSTNESS.md, episode_index.csv
```

Both read only the committed per-episode receipts in this directory. They need no GPU, no
simulator, and no data outside this repository — a fresh clone reproduces every table.

---

## Primary data

| path | contents |
|---|---|
| `sweep_measurements/<arm>/` | 2,400 per-episode records, ten arms. Verified byte-identical to the eval's own output. 0 missing episodes in every arm. |
| `baseline_full_14498/` | 1,077 per-episode records for the June baseline (SR 18.3% headline). |
| `determinism_probe_data/`, `driver_smoke_580173/` | the 6-episode determinism probe and the post-driver-change smoke test. |

Arm tags: `stretchA_300`, `stretchB_300`, `crop_300`, `pad_300` (transform ablation, 300
episodes each) and `h060_200` … `h150_200` (camera heights 0.60–1.50 m, 200 episodes each).

## Analysis

| path | contents |
|---|---|
| `analysis/ANALYSIS.md` | The pre-registered analysis: per-arm SR/OS/SPL/NE/ONE with Wilson and Clopper-Pearson CIs, unpaired two-proportion contrasts with Holm correction, the Cochran-Armitage height-trend test, the resolution floor, and `ended_at_step` by termination reason. |
| `analysis/ROBUSTNESS.md` | Not pre-registered, and where the interesting qualifications live: run-to-run outcome instability, differential compute censoring, the reach/arrival-recognition split, and scene-clustered CIs. |
| `analysis/episode_index.csv` | list order → `episode_id` → record index → scene → goal radius. This is what makes "episodes 0-299" mean anything to an outsider. |

## Design and provenance

| path | contents |
|---|---|
| `PRE_REGISTRATION.md` | The plan, fixed in advance, plus the determinism verdict, the driver confound, and **Amendments 1 and 2** (the McNemar → unpaired switch, and the adjudication of the reproducibility gate). |
| `KNOWN_GAPS.md` | The adversarial gap register. Read this before writing anything. |
| `provenance/ENVIRONMENT.md` | Host, GPU, both driver versions, Python/torch/CUDA, IsaacLab commit, and the PhysX buffer modification with its open question. |
| `provenance/pip_freeze_vlnce-isaac.txt` | Full package list (395 packages). |
| `provenance/checkpoint_sha256.txt` | Weight hashes. The evaluated policy is `model_14498.pt`; weights are gitignored for size, so the hash is what pins them. |
| `provenance/isaaclab_simulation_cfg.patch` | The one local simulator modification, previously unversioned anywhere. |
| `provenance/sweep_run_log.txt` | The raw runner log for all ten arms, rescued from volatile `/tmp`. |

---

## Headline results

Ten arms, 2,400 episodes, ~400 GPU-hours, 2026-07-26 → 08-17.

| arm | n | SR % | 95% CI | OS % |
|---|---|---|---|---|
| stretchA | 300 | 15.0 | [11.4, 19.5] | 29.0 |
| stretchB | 300 | 16.0 | [12.3, 20.6] | 31.7 |
| crop | 300 | 16.7 | [12.9, 21.3] | 24.0 |
| pad | 300 | 13.3 | [9.9, 17.6] | 29.0 |
| h060 (0.60 m) | 200 | 11.5 | [7.8, 16.7] | 27.5 |
| h078 (0.78 m, native) | 200 | 12.5 | [8.6, 17.8] | 28.0 |
| h095 (0.95 m) | 200 | 18.5 | [13.7, 24.5] | 31.0 |
| h110 (1.10 m) | 200 | 13.0 | [9.0, 18.4] | 22.0 |
| h130 (1.30 m) | 200 | 15.0 | [10.7, 20.6] | 24.5 |
| h150 (1.50 m) | 200 | 14.5 | [10.3, 20.0] | 25.5 |
| *baseline, pinned 300* | 300 | 17.7 | [13.8, 22.4] | 31.7 |

**What the sweep supports.**

1. **The transform ablation is null.** No contrast among stretch / crop / pad survives
   correction. This was the pre-registered expected outcome.
2. **The height sweep is null**, on every pairwise contrast and on the monotone trend test.
   The h095 = 18.5% peak sits inside run-to-run noise and must not be reported as an optimum
   without a replication arm at that height.
3. **The benchmark reproduced across a forced NVIDIA driver change** (580.159.03 →
   580.173.02, the old version purged and unobtainable). All contrasts against the baseline
   are null. This bounds driver effects and run-to-run noise *combined* — an upper bound on
   reproducibility, not a clean noise floor.
4. **Aggregate-stable, episode-unstable.** Two identical runs agree to 1.0 pp on aggregate SR
   but disagree on 11.0% of individual episodes (kappa = 0.58); only 48% of ever-successful
   episodes succeed in both runs. This is the sharpest form of the non-determinism finding
   and the reason paired testing was abandoned.
5. **A compute confound qualifies the transform arms.** Episodes are killed at a fixed 900 s
   wall-clock and scored as failures. That rate differs sharply by arm (crop 2.0% vs pad
   12.3%, p ≈ 7e-7) and accounts for roughly half the crop-vs-pad SR gap.

**What it does not support:** the sim-vs-VLM attribution of nondeterminism (no receipts —
`KNOWN_GAPS.md` G1) and the peripheral-FOV arrival-recognition mechanism (refuted by the
sweep itself — G6). Both currently appear in `FINAL_RESULTS_full1077.md` and `SLIDE_FACTS.md`,
which now carry correction notices.

**Design limitation to state in Methods, not to apologise for later:** at these sample sizes
the minimum detectable difference is ~8.7 pp (n=300 vs 300) and ~10.7 pp (n=200 vs 200) at
α=0.05, power=0.80. Every observed contrast is far below that. These arms can exclude large
effects and nothing else.
