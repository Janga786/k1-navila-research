# Reproduction check — data provenance & independent recomputation

Compiled 2026-09-11. Every number below is copied verbatim from the stdout of
`paper_evidence/scripts/reproduction_check.py` (Task B, independent recomputation) or from
the rerun of the two committed scripts (Task A). Nothing here is hand-typed from the claimed
values — the claimed values are inputs to the script, the regenerated values are its output.

Full stdout: `paper_evidence/provenance/rerun_analyze_sweep.stdout.txt`,
`paper_evidence/provenance/rerun_analyze_robustness.stdout.txt`. The reproduction-check script
itself is `paper_evidence/scripts/reproduction_check.py`; run it with
`/home/boosterk1/miniconda3/bin/python3 paper_evidence/scripts/reproduction_check.py`.

## Verdict up front

- **27 / 27 independently recomputed metric checks MATCH** the historically claimed values
  (baseline SR/OSR/NE/SPL, four transform-arm SR%/timeout%, six height-arm SR%, and the
  7-way stretchA-vs-stretchB repeatability check: four 2x2 cell counts, n, discordance%,
  kappa). Zero mismatches.
- Rerunning the two committed analysis scripts (`analyze_sweep.py`, `analyze_robustness.py`)
  reproduced **5 of 6** regenerated files byte-for-byte identical to the committed versions
  (`ANALYSIS.md`, `contrasts.csv`, `ROBUSTNESS.md`, `episode_index.csv`, and trivially the
  scripts themselves). **`arm_summary.csv` differed** — at floating-point noise level only
  (~1e-10 to 1e-15 relative magnitude, i.e. differences appear only after the 10th+
  significant digit; every number changes by less than 1e-11 in absolute terms). Root cause
  identified: numpy/scipy version drift (committed run: numpy 1.26.4 / scipy 1.15.3, per
  `receipts/provenance/ENVIRONMENT.md`; this rerun: numpy 2.4.6 / scipy 1.17.1) changes
  floating-point summation order in a couple of derived columns. **No number changes at the
  precision anything is reported to** (all paper-facing values are quoted to 1-2 decimal
  places or fewer). The tree was restored to the committed state with
  `git checkout -- receipts/analysis/arm_summary.csv`; see the "Byte-for-byte reproduction"
  section below for the full diff.
- Data-integrity checks (episode counts, no duplicate/non-canonical filenames, identical
  record_idx sets across the four transform arms and across the six height arms, sentinel
  counts, the documented schema anomaly) all check out exactly as documented in
  `ROBUSTNESS.md` §E and `KNOWN_GAPS.md`.
- The stretchA-vs-stretchB "byte-identical configuration" claim is **partially proven, partially
  asserted** — see the dedicated section below. Nothing here contradicts the claim, but one
  leg of it (stretchA's raw invocation log line) cannot be independently verified because the
  underlying log was not preserved across the July power outage.
- **No discrepancy was found that changes any paper-facing number.** One genuine (documented)
  data-integrity oddity is worth restating for the paper: the four episodes with a `-1.0`
  `distance_to_goal` sentinel in stretchA and stretchB, and the one each in h078/h130, are
  **exactly** the same episodes flagged by the `max_episode_steps == -1` schema anomaly — i.e.
  the sentinel and the schema anomaly are the same underlying runner-stub records, not two
  independent phenomena. This is consistent with, and slightly sharper than, what
  `ROBUSTNESS.md` §E already says.

---

## 1. Baseline (`receipts/baseline_full_14498`, n=1077)

| metric | expected | regenerated | diff | tol | source | script | verdict |
|---|---|---|---|---|---|---|---|
| n | 1077 | 1077 | 0 | 0 | `receipts/baseline_full_14498/*.json` | reproduction_check.py | MATCH |
| SR % | 18.30 | 18.29 | 0.01 | 0.05 pp | `receipts/baseline_full_14498` | reproduction_check.py | MATCH |
| OSR % | 30.30 | 30.27 | 0.03 | 0.05 pp | `receipts/baseline_full_14498` | reproduction_check.py | MATCH |
| NE (m) | 7.590 | 7.591 | 0.001 | 0.01 m | `receipts/baseline_full_14498` | reproduction_check.py | MATCH |
| SPL % | 10.93 | 10.93 | 0.00 | 0.05 pp | `receipts/baseline_full_14498` | reproduction_check.py | MATCH |

Baseline carries **0** `distance_to_goal == -1.0` sentinels (NE computed over all 1077 of
1077). Tolerances: 0.05 percentage points for any rate quoted to 1 decimal place in the
claimed values (half the last-digit rounding step, plus slack for the internal
Wilson/summation float noise seen in Task A); 0.01 m for NE, quoted to 2 decimal places.

## 2. Transform arms (n=300 each): SR% and timeout rate

| arm | metric | expected | regenerated | diff | tol | source | script | verdict |
|---|---|---|---|---|---|---|---|---|
| stretchA | SR % | 15.00 | 15.00 | 0.00 | 0.05 pp | `sweep_measurements/stretchA_300` | reproduction_check.py | MATCH |
| stretchA | timeout % | 6.67 | 6.67 | 0.00 | 0.05 pp | `sweep_measurements/stretchA_300` | reproduction_check.py | MATCH |
| stretchB | SR % | 16.00 | 16.00 | 0.00 | 0.05 pp | `sweep_measurements/stretchB_300` | reproduction_check.py | MATCH |
| stretchB | timeout % | 7.33 | 7.33 | 0.00 | 0.05 pp | `sweep_measurements/stretchB_300` | reproduction_check.py | MATCH |
| crop | SR % | 16.70 | 16.67 | 0.03 | 0.05 pp | `sweep_measurements/crop_300` | reproduction_check.py | MATCH |
| crop | timeout % | 2.00 | 2.00 | 0.00 | 0.05 pp | `sweep_measurements/crop_300` | reproduction_check.py | MATCH |
| pad | SR % | 13.30 | 13.33 | 0.03 | 0.05 pp | `sweep_measurements/pad_300` | reproduction_check.py | MATCH |
| pad | timeout % | 12.30 | 12.33 | 0.03 | 0.05 pp | `sweep_measurements/pad_300` | reproduction_check.py | MATCH |

crop and pad's SR expected values (16.7, 13.3) are 1-decimal roundings of the true 16.667%
and 13.333%; the 0.03 pp "diff" is exactly that rounding, not a data disagreement.

## 3. Height arms (n=200 each), vs `SWEEP_STATUS.md`

| arm | expected SR % | regenerated SR % | diff | tol | source | script | verdict |
|---|---|---|---|---|---|---|---|
| h060 | 11.50 | 11.50 | 0.00 | 0.05 pp | `sweep_measurements/h060_200` | reproduction_check.py | MATCH |
| h078 | 12.50 | 12.50 | 0.00 | 0.05 pp | `sweep_measurements/h078_200` | reproduction_check.py | MATCH |
| h095 | 18.50 | 18.50 | 0.00 | 0.05 pp | `sweep_measurements/h095_200` | reproduction_check.py | MATCH |
| h110 | 13.00 | 13.00 | 0.00 | 0.05 pp | `sweep_measurements/h110_200` | reproduction_check.py | MATCH |
| h130 | 15.00 | 15.00 | 0.00 | 0.05 pp | `sweep_measurements/h130_200` | reproduction_check.py | MATCH |
| h150 | 14.50 | 14.50 | 0.00 | 0.05 pp | `sweep_measurements/h150_200` | reproduction_check.py | MATCH |

## 4. Repeatability 2x2 (stretchA vs stretchB, `ROBUSTNESS.md` §A)

| metric | expected | regenerated | diff | tol | source | script | verdict |
|---|---|---|---|---|---|---|---|
| n paired | 300 | 300 | 0 | 0 | stretchA ∩ stretchB | reproduction_check.py | MATCH |
| A success ∩ B success | 30 | 30 | 0 | 0 | stretchA/stretchB | reproduction_check.py | MATCH |
| A success, B failure | 15 | 15 | 0 | 0 | stretchA/stretchB | reproduction_check.py | MATCH |
| A failure, B success | 18 | 18 | 0 | 0 | stretchA/stretchB | reproduction_check.py | MATCH |
| A failure ∩ B failure | 237 | 237 | 0 | 0 | stretchA/stretchB | reproduction_check.py | MATCH |
| discordance % | 11.00 | 11.00 | 0.00 | 0.05 pp | stretchA/stretchB | reproduction_check.py | MATCH |
| Cohen's kappa | 0.5800 | 0.5802 | 0.0002 | 0.001 | stretchA/stretchB | reproduction_check.py | MATCH |

kappa tolerance 0.001 (the claimed value carries an explicit "~" and is quoted to 3 decimal
places; 0.0002 is within that rounding).

---

## 5. Data-integrity findings

All from `reproduction_check.py`'s "DATA INTEGRITY" section, computed directly from the
filesystem and JSON contents (no dependency on `analyze_sweep.py`/`analyze_robustness.py`).

**Episode counts, no missing/duplicate files.** Every one of the ten sweep arms has exactly
its pinned count on disk (stretchA/stretchB/crop/pad = 300, h060..h150 = 200), the baseline
has exactly 1077, and no arm or the baseline has a duplicate-index filename (e.g. `"07.json"`
and `"7.json"` both parsing to record_idx 7) or any filename that fails to round-trip through
`int()`. All eleven directories: **OK**.

**Transform-arm episode-set identity.** stretchA, stretchB, crop, and pad cover the
**identical** 300-element record_idx set (pairwise symmetric-difference = 0 for all four).
This is what makes the stretchA-vs-stretchB pairing in §4 above, and every transform contrast
in `ANALYSIS.md`, valid.

**Height-arm episode-set identity.** h060 through h150 cover the **identical** 200-element
record_idx set (pairwise symmetric-difference = 0 for all six), and that 200-element set is a
**subset** of the 300-element transform set.

**`-1.0` `distance_to_goal` sentinel counts, per arm:**

| arm | sentinel count / n |
|---|---|
| stretchA | 3 / 300 |
| stretchB | 4 / 300 |
| crop | 0 / 300 |
| pad | 0 / 300 |
| h060 | 0 / 200 |
| h078 | 1 / 200 |
| h095 | 0 / 200 |
| h110 | 0 / 200 |
| h130 | 1 / 200 |
| h150 | 0 / 200 |
| baseline | 0 / 1077 |

**Schema anomaly (`max_episode_steps == -1`, `ended_at_step == -1`), per arm, vs the counts
documented in `ROBUSTNESS.md` §E:**

| arm | expected count | regenerated count | co-occurs with `ended_at_step=-1`? | verdict |
|---|---|---|---|---|
| stretchA | 3 | 3 | yes | MATCH |
| stretchB | 4 | 4 | yes | MATCH |
| crop | 0 | 0 | yes | MATCH |
| pad | 0 | 0 | yes | MATCH |
| h060 | 0 | 0 | yes | MATCH |
| h078 | 1 | 1 | yes | MATCH |
| h095 | 0 | 0 | yes | MATCH |
| h110 | 0 | 0 | yes | MATCH |
| h130 | 1 | 1 | yes | MATCH |
| h150 | 0 | 0 | yes | MATCH |
| **total (all 10 arms)** | **9** | **9** | — | MATCH |

**New observation (not previously stated this precisely):** for every arm, the schema-anomaly
count equals the sentinel count exactly (stretchA 3=3, stretchB 4=4, h078 1=1, h130 1=1, and 0
everywhere else). These are not two independent phenomena — the `max_episode_steps=-1`
runner-side stub rows are the *same* records that carry the `-1.0` distance sentinel. This
sharpens, and does not contradict, `ROBUSTNESS.md` §E's statement that "scoring is unaffected...
the distance sentinel is handled identically."

**Scene coverage** (`receipts/analysis/episode_index.csv`, 1077 rows, one per baseline
episode):

- Full 1077-episode benchmark spans **11 scenes**.
- The pinned 300-episode transform set spans **8 scenes** (all present in the index; 0
  record_idx from the pinned set are missing from `episode_index.csv`).
- Episodes-per-scene for the pinned 300 (sums to 300):

| scene | episodes in pinned 300 |
|---|---|
| zsNo4HB9uLZ | 84 |
| 2azQ1b91cZZ | 81 |
| QUCTc6BB5sX | 57 |
| TbHJrupSAjP | 24 |
| EU6Fwq7SyZv | 18 |
| X7HyMhZNoso | 15 |
| x8F5xyUWy9e | 12 |
| Z6MFQCViBuw | 9 |

This confirms `ROBUSTNESS.md` §D's point that the pinned 300 is not scene-representative (3 of
the 11 full-benchmark scenes are absent from the pinned set entirely, and the surviving 8 are
reweighted — one scene alone, zsNo4HB9uLZ, is 28% of the pinned 300).

---

## 6. stretchA vs stretchB configuration equivalence — proven vs. asserted

This is the paper-critical claim: that stretchA and stretchB are the same configuration run
independently, so their comparison isolates run-to-run outcome instability under a fixed
configuration and driver. Evidence, exactly as
printed by `reproduction_check.py`:

**PROVEN (directly checkable from committed files):**

1. **Identical launch parameters in `arms_runner.sh`'s `ARMS` array.** The array entries are
   `"stretchA stretch 300 "` and `"stretchB stretch 300 "` — identical transform (`stretch`),
   identical episode count (`300`), identical (empty) extra-flags field; only the name/tag
   token differs. Both arms are executed by the **same unconditional loop body** in the same
   script, using a single `$CKPT` variable and a single `$RUNNER` invocation template — the
   transform/episode-range/checkpoint-path are not arm-specific, so this is a code-level proof,
   not an inference from logs.
2. **Identical 300-record_idx set** (§5 above, symmetric difference = 0).
3. **Identical checkpoint file.** Both used `checkpoints/model_14498.pt`
   (SHA-256 `8e7559b5075fe6c90d4ce81fb1eb66d69ef852b2ed3718b5416b6a0fd2bdad4b`, from
   `receipts/provenance/checkpoint_sha256.txt`), and that file's mtime
   (`2026-06-10T19:06:47`) predates both arm launches — it was never rewritten between the
   June training run and either sweep-arm run.
4. **stretchB's raw invocation line is directly present** in
   `receipts/provenance/sweep_run_log.txt`:
   `out_tag=stretchB_300  episodes=[0,300)  flags: --closed_loop --max_episode_s 120 --vlm_transform stretch`

**ASSERTED (documented, but not independently re-derivable from a raw log by this script):**

1. **stretchA's raw invocation line is NOT in `sweep_run_log.txt`.** That file begins
   `[08-04 15:32] SWEEP RUN (re)started` — i.e. it only starts capturing **after** stretchA had
   already completed (`ARM DONE: stretchA` is timestamped `[07-26 17:10]` in `SWEEP_STATUS.md`)
   and partway through stretchB (`have 233`). The earlier `/tmp` log covering stretchA's
   original run (2026-07-24 → 07-26) was not preserved — it was written before the 2026-07-28
   → 31 power-loss outage and evidently lost with it. stretchA's flags are corroborated by (a)
   `arms_runner.sh`'s single code path (point 1 above) and (b) `SWEEP_STATUS.md`'s own
   contemporaneous line `[07-24 12:42] ARM START: stretchA  transform=stretch  eps 0-300
   extra='none'`, but there is no surviving byte-level invocation record for stretchA the way
   there is for stretchB. This is a real, if narrow, provenance gap — flag it in the paper as
   "corroborated by the runner script and the status log, not by a raw per-invocation log."
2. **Same NVIDIA driver version (580.173.02) for both arms.** This rests on the upgrade/hold
   timeline — driver bumped 2026-07-24 06:45 (`PRE_REGISTRATION.md` "DRIVER-CHANGE CONFOUND"),
   auto-upgrades disabled and ~200 nvidia/cuda/kernel packages held that same day
   (`AWAY_STATUS.md`), `arms_runner.sh` started 2026-07-24 12:42 (after the bump) — not on a
   per-arm `nvidia-smi` capture. No file in this repository records driver version at either
   arm's start or end individually.
3. **No other concurrent GPU load / thermal / clock-throttling equivalence** is logged for
   either arm; equivalence here is assumed, not measured.

**Bottom line:** the *configuration* (transform, episode range, checkpoint, code path) is
proven identical by source-code and hash inspection. The *execution environment* (driver
version across both runs, absence of confounding load) is asserted from the operational
timeline and is plausible, but not independently instrumented per-arm. Relevant paths/hashes:
`arms_runner.sh` (repo root), `SWEEP_STATUS.md` (repo root),
`receipts/provenance/sweep_run_log.txt`, `receipts/provenance/checkpoint_sha256.txt`,
`receipts/provenance/ENVIRONMENT.md`, `receipts/PRE_REGISTRATION.md`, `AWAY_STATUS.md`.

---

## 7. Byte-for-byte reproduction of the committed analysis scripts (Task A)

Both scripts were rerun with `/home/boosterk1/miniconda3/bin/python3` after checksumming every
file in `receipts/analysis/` before and after:

```
python3 receipts/analysis/analyze_sweep.py
python3 receipts/analysis/analyze_robustness.py
```

Result: `ANALYSIS.md`, `contrasts.csv`, `ROBUSTNESS.md`, and `episode_index.csv` were
**byte-identical** to the committed versions (SHA-256 unchanged; `git status --short` showed
no change for these four files after the rerun). **`arm_summary.csv` differed** — `git diff
--stat` reported 11 changed lines (all 11 data rows; the header was unchanged). Every changed
value differs only beyond the ~10th significant digit, e.g.:

```
- ...19.551644068402698...    (committed)
+ ...19.551644068402695...    (rerun today)
- ...12.038914792910116...
+ ...12.038914792910111...
```

All such diffs are on the order of 1e-11 to 1e-15 in absolute value — no reported number
(all quoted to 1-2 decimal places in every paper-facing table) is affected. Cause: the
committed file was produced with numpy 1.26.4 / scipy 1.15.3 (`receipts/provenance/
ENVIRONMENT.md`); this rerun used numpy 2.4.6 / scipy 1.17.1 (`paper_evidence/provenance/
environment.txt`), and cross-version floating-point summation/CI-inversion order differs at
the ULP level. Per the task's authorized narrow exception, the committed file was restored
with `git checkout -- receipts/analysis/arm_summary.csv` immediately after capturing the diff;
`git status --short` on `receipts/` was empty again afterward (verified: pre- and post-rerun
SHA-256 lists over `receipts/analysis/*` are identical after the restore).

**Conclusion:** the committed summaries are reproducible for all practical (reported)
purposes; they are not bit-for-bit reproducible across a numpy/scipy version bump in one
CSV's derived-statistics columns. State this precisely in the paper's reproducibility
statement rather than claiming exact bitwise reproducibility.

---

## 8. Discrepancies

**None that affect any reported number.** Summarizing what was and was not found:

- No MISMATCH among the 27 independently recomputed metrics (§1-4).
- No missing or duplicate episode files in any of the eleven per-arm/baseline directories.
- No asymmetry in the transform-arm or height-arm record_idx sets.
- The schema-anomaly counts (9 total: stretchA 3, stretchB 4, h078 1, h130 1) match
  `ROBUSTNESS.md` §E exactly, and are shown here to coincide exactly, row-for-row, with the
  `-1.0` NE sentinel rows in those same four arms (see §5) — a clarification, not a
  contradiction, of the existing documentation.
- One genuine, pre-existing provenance gap was confirmed rather than newly discovered:
  stretchA's raw per-invocation log line does not survive in `sweep_run_log.txt` (§6). This
  does not contradict the "same configuration" claim (the code-level proof in §6 point 1
  stands on its own), but it means the claim rests on script inspection + a status-log line
  for stretchA, not on a captured raw invocation record the way stretchB's does. Recommend the
  paper's Methods/reproducibility section say exactly this, rather than implying symmetric
  log evidence for both arms.
- Rerunning the committed analysis scripts changed one file's least-significant floating-point
  digits (§7) — not a discrepancy in content, but worth stating plainly as a bitwise-vs-
  numerically reproducible distinction rather than silently claiming exact reproducibility.

---

## Files produced by this task

- `paper_evidence/REPRODUCTION_CHECK.md` (this file)
- `paper_evidence/scripts/reproduction_check.py`
- `paper_evidence/provenance/environment.txt`
- `paper_evidence/provenance/input_sha256.txt`
- `paper_evidence/provenance/rerun_analyze_sweep.stdout.txt`
- `paper_evidence/provenance/rerun_analyze_robustness.stdout.txt`
