# AIRC 2027 reanalysis

Reproduces every number, table and figure in "Run-to-Run Variability and Single-Run Camera
Comparisons in Closed-Loop Vision-Language Navigation on a Simulated Humanoid" from the
committed per-episode receipts and the small data files in `data/`. Nothing here runs the
simulator or the model.

## Setup
- Place this folder at `paper_evidence/airc2027_reanalysis/` in k1-navila-research
  (the scripts read `../../receipts`), or set `K1_REPO` to the repository root.
- Put the VLN-CE-Isaac episode file `vln_ce_isaac_v1.json.gz` (released with NaVILA-Bench;
  SHA-256 ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec) next to the
  scripts, or set `VLNCE_DATASET`.
- Python with numpy, pandas, scipy, matplotlib and Pillow. The shipped tables were computed with
  Python 3.11 and the versions in `requirements.txt`. The last floating-point digit of some values
  can differ between CPUs and library versions (seen with these versions on an Intel Xeon Gold 6128,
  and with Python 3.10, numpy 2.2.6, pandas 2.3.3 and scipy 1.15.3); `check_tables.py` allows for that.

## Data beyond the receipts (`data/`, `fig1_assets/`)
All of it was derived on the sweep workstation (2026-10-01) from files that stay there (videos, scans and
full logs are in the private NaVILA-Complete-Archive, `12_airc2027_materials/`). The post-sweep tests are
published next to this folder in `../airc2027_post_sweep/`.

| File | What it is |
| --- | --- |
| `data/relabelled_wall_timeouts.csv` | The 34 records labelled `wall_timeout` whose episodes had finished: the evaluator's SIGTERM handler stayed armed after the result was written and relabelled the record while the episode's output (its video) was being written. These are the only `wall_timeout` records whose episode video exists, and a video is written only after the episode loop ends. `common.py` treats them as completed episodes. |
| `data/gray_frames_start.csv` | For all 3,360 episode videos (10 sweep runs and J): the share of the robot-camera half of video frames 0 and 1 whose pixels are all within ±6 of RGB 53 (frame 0 = initial frame after the reset, which enters every model query; frame 1 = the frame recorded at the first query). A frame counts as gray at a share ≥ 0.10. |
| `data/gray_frames_full_length/` | The same measure over every frame of every video of all 11 runs (decoded at reduced size). The robot-camera half of a video is refreshed only when a frame is added to the model's frame history (every 25 control steps, 0.5 s), so gray video frames come in blocks of five, each one history frame; the 8 warm-up frames added before the first query are not in the videos. |
| `data/probe_ticks/` | Per-query logs (step, pose, reply) of two pre-sweep probe pairs that ran six episodes twice each: `exta`/`extb` (driver 580.159.03) and `smoke173a`/`smoke173b` (580.173.02). |
| `fig1_assets/` | Re-renders of the start of episode 0 (dataset entry 0) made with the unmodified evaluation code, robot model, scene and render settings (`../airc2027_post_sweep/rendering_diagnosis/images/as_evaluated/`), used in Fig. 1: the robot camera's initial frame at each of the six camera heights (lossless 1280x720 RTX frames resized to 450x253; each matches frame 0 of its run's episode-0 video to within H.264 noise, mean absolute difference 1.0-1.8 of 255) and a path-traced view of the K1 from behind after the reset warm-up (central 1080x1080 crop resized to 612x612). `SOURCES.json` lists each source file and its SHA-256. |

## Run (about three minutes)
| Script | Output | Paper content |
| --- | --- | --- |
| `paper_numbers.py` | `tables/paper_numbers.json` | Table I, rerun agreement, bit-identity by length, variance, Fisher tests, interruption counts and timing |
| `paper_numbers_v2.py` | `tables/paper_numbers_v2.json` | falls, kappa (all and diverged episodes), J agreement, near-goal interruptions, crop–pad interruption McNemar |
| `audit_v3.py` | `tables/audit_v3.json` | Table II (route-clustered paired tests, Holm), planned McNemar and trend tests, discordance, SDs and power, diverged-subset SRs, degenerate route, interruption bounds (`table3`) |
| `kappa_div_ci.py` | `tables/kappa_div_diff.json` | kappa differences on diverged episodes with route-resampled intervals |
| `extras.py` | `tables/extras.json` | J's identical pairs, handler-written interruptions, step-cap share |
| `paired_tests.py` | `tables/paired_tests.json` | why the within-episode permutation test was replaced (variance check, type-I simulation) |
| `corrections_and_gray.py` | `tables/corrections_and_gray.json` | relabelled records and true interruptions (Sections III-D, V-C); gray-frame counts at the start and over whole episodes for every run, route consistency and outcome subsets (Section V-B); gray frames in identical and diverged rerun pairs (computed, not reported in the paper) |
| `probe_divergence.py` | `tables/probe_divergence.json` | where divergence starts (Section V-A) |
| `make_fig_system.py` | `fig_system.pdf` (Fig. 1) | set `FIG_DIR` for the output folder |
| `make_figure_v3.py` | `fig_results.pdf` (Fig. 2) | set `FIG_DIR` for the output folder |

All scripts use fixed seeds and give identical tables on every run (PDF metadata aside). The scripts
overwrite `tables/`, so to check a run, copy `tables/` first and then run
`python check_tables.py <copy> tables`: integers and strings must match exactly and floats to a relative
1e-9.

Change, 2026-10-06: `paper_numbers.py` and `paper_numbers_v2.py` now read the `pad` column as `w["pad"]`.
Under pandas 2.x, `w.pad` is the `DataFrame.pad` method, so those two scripts gave wrong interruption
overlaps or crashed; the shipped tables were computed under pandas 3, where the column was read, and
are unchanged.

## Run labels
`base` = J (`baseline_full_14498`), `A` = D1 (`stretchA_300`), `B` = D2 (`stretchB_300`),
`h078` = D3 (`h078_200`); the other labels match the receipt folders. Camera heights are the camera's
height above the floor in dataset entry 0 with the robot standing after the 1-s reset warm-up (trunk
origin 0.497 m above the floor, measured on the workstation) plus the camera offset above the trunk
origin: `h060` 0.57 m (offset 0.07 m), `h078` 0.75 m (0.25 m, the default), `h095` 0.92 m (0.42 m),
`h110` 1.07 m (0.57 m), `h130` 1.27 m (0.77 m), `h150` 1.47 m (0.97 m). During 2 s of walking the
camera rode 0.03 m higher on average. Heights above the floor were not measured in other episodes.

## Notes
- The cause of the gray region (Section V-B; `../airc2027_post_sweep/rendering_diagnosis/DIAGNOSIS.md`),
  the frame-replay test (Sections V-A and VI; `../airc2027_post_sweep/frame_replay/RESULTS.md`) and the
  walking check (Section III-A; `../airc2027_post_sweep/walking_check/RESULTS.md`) come from post-sweep
  tests on the sweep workstation; this package does not recompute them.
- An interruption in every script means a `wall_timeout` record that is not in
  `data/relabelled_wall_timeouts.csv` (117 of the 151 labelled records).
- `paper_numbers.json` and `paper_numbers_v2.json` also hold earlier permutation-test p-values
  (`configs`, `june_vs_reruns`, `h095_with_june`). The paper reports the route-clustered paired
  tests in `audit_v3.json`; `paired_tests.py` shows why the permutation test was dropped.
- Wall time per 1,000 simulated steps (Section V-C) = run duration from the ARM START/ARM DONE
  timestamps in `SWEEP_STATUS.md` divided by the run's summed `ended_at_step`; D2 uses only its
  second, 67-episode session.
