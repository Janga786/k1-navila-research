# paper_evidence — AIRC 2027 reanalysis package

Reanalysis of ALREADY-COLLECTED experiments in `receipts/`. **No new simulation, training, or
robot execution was performed.** All raw experimental data is read-only; integrity was verified
by checksum before and after (see `FINAL_REPORT.md`).

Repo commit: `a6fe40b431353b0eee968c07015dd8d1b1211d53`

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
cd /home/boosterk1/Projects/k1_research
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

## AIRC 2027 paper
- `airc2027_reanalysis/`: reproduces every number, table and figure of the AIRC 2027 paper from
  `receipts/` (see its README).
- `airc2027_post_sweep/`: the post-sweep tests the paper cites (rendering diagnosis, frame-replay
  test, walking check); see its README.
