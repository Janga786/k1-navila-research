# K1 NaVILA — Full Benchmark Results (VLN-CE-Isaac, R2R val-unseen, n=1077)

**Run:** `full_14498` · completed 2026-06-22 · **48.3 h** of GPU compute (290 episodes run, 787
resumed-skip, 10.0 min/ep) · lab RTX 3090.
**Config:** policy `model_14498` (feet_distance fine-tune; circumduction- and foot-clip-fixed,
all sim gates passed) · VLM `navila-llama3-8b-8f` **int8** · **raw** frames (no clean_render/bright)
· `--vlm_transform stretch` · `--closed_loop` · `--max_episode_s 120`. Code @ repo
`Janga786/k1-navila-research`.

## Headline (paper-comparable, denominator = full 1077)

| Robot | Policy | NE↓ | OS↑ | SR↑ | SPL↑ |
|-------|--------|-----|-----|-----|------|
| Go2 (paper, blind) | — | 6.03 | 49.0 | 36.2 | 33.3 |
| Go2 (paper, vision) | — | 5.49 | 58.7 | 50.2 | 45.5 |
| H1  (paper, blind) | — | 7.67 | 33.3 | 24.4 | 21.0 |
| H1  (paper, vision) | — | 5.86 | 54.6 | 45.3 | 40.3 |
| **K1 (ours, vision)** | **model_14498** (n=1050/1077) | **7.64** | **29.2** | **17.7** | **10.7** |
| K1 — prior stack (reference) | v3 model_16000 | — | ~10 | 0.76 | — |

**SR 0.76% → 17.7% — a ~23× improvement.** NE is essentially tied with the paper's blind-H1
(7.64 vs 7.67); OS/SR/SPL sit a notch below it — the embodiment gap analyzed below.

## Decomposition (where the episodes go)
- **191 successes / 1077 = SR 17.7%** (18.2% over the 1050 present).
- **315 reaches / 1077 = OS 29.2%** — got within the 3.0 m goal radius at some point.
- **Stop-given-reached = 191/315 = 61%** — of episodes that reached, 61% recognized arrival and
  stopped. (Paper blind-H1 implies ~73%; ours is lower but far from the old ~0% — the camera/render
  fixes earned this.)
- **Reached-but-didn't-stop = 124 (11.5% of 1077)** — the stop-failure class; the lever here is
  arrival recognition (FOV/height), not locomotion.
- **Didn't reach = 70.8%** — the dominant failure mode, traced to exploration cost (below).
- **27 missing (2.5%)** — persistent timeouts (93 timeout events total, mostly recovered on retry);
  honestly counted as failures, so 17.7% is a conservative floor. (Retry pass pending.)

## The scientific finding: the camera-height / FOV embodiment tax
The same frozen NaVILA VLM drives H1 (blind) to SR 24.4 / OS 33.3 and K1 to SR 17.7 / OS 29.2. The
gap is **embodiment, not the policy or the brain** (the locomotion passed every sim gate; the VLM is
identical and was shown stop-capable). Two quantified mechanisms:
1. **Exploration cost (the OS/SR gap).** K1's eye sits at **0.78 m** vs H1's ~1.5 m, so goals behind
   furniture aren't visible — the robot must explore to find them. **SPL among successes is 60%**
   (≈1.7× the optimal path) vs blind-H1's ~86%; successful K1 runs wander before arriving.
   Mean path length 22.0 m. NE among the 191 successes is **1.53 m** (it stops *well* inside the
   radius when it does arrive — the stop logic is sound; the issue is reaching).
2. **Arrival recognition needs peripheral FOV (controlled).** A same-episode crop-vs-stretch A/B:
   stretch SR 30 vs crop 0 (crop *reached* but never stopped). Cropping improves approach geometry
   but blinds the "I have arrived" judgment.

Supporting controls already in hand: **int8 vs fp16 action drift 7.1%** (below the model's own 14.9%
sensitivity to an imperceptible JPEG re-encode; 100% agreement on success episodes — int8 is not a
confound); and the **renderer result** — the same stack on a 5090 (Isaac 4.1 lacks Blackwell support,
~17% frame speckle) scores **SR 0/10** on episodes where this 3090 (clean render) scores 30/10.

## Caveats (stated plainly)
- **n = 1 robot, single VLM.** A deployment/embodiment result, not a broad empirical claim.
- **Sim-only.** No real-robot navigation number yet (deployment networking solved — `--net eno1` /
  GetMode verified — but the robot has not walked under NaVILA; camera reliability pending).
- **27 missing** counted as failures (conservative); retry pending.
- **int8** (to co-fit the VLM with Isaac on 24 GB) — controlled and shown ≈fp16 at the action level.

## Provenance
Measurements: `NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_full_14498/measurements/`
(1050 JSON). Aggregator: `aggregate_k1_vision_results.py` (NE = final distance_to_goal; missing =
failures over 1077). Full project record + paper analysis: `~/Desktop/PROJECT_DOSSIER_AND_PAPER_ANALYSIS.md`.
Method/controls: `RESULTS_2026-06-11_smoke.md`, `REPLAY_ARBITER_2026-06-11.md`.
