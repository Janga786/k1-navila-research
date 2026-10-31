# K1 NaVILA — Full Benchmark Results (VLN-CE-Isaac, R2R val-unseen, n=1077)

> [!IMPORTANT]
> **CORRECTION NOTICE — 2026-08-22, revised 2026-10-30.** This document originally claimed a
> causal "camera-height / FOV embodiment tax" with two mechanisms. Both causal claims are
> withdrawn, and the body below has now been edited to match the evidence (the original text
> is preserved in git history):
> - **Mechanism 2** ("arrival recognition needs peripheral FOV": crop *reached* but never
>   stopped, from an n=10 pilot) **did not replicate**. At n=300 per arm the effect runs the
>   OPPOSITE way: crop converts **69.4%** of goal-reaches into stops versus stretch's
>   **51.1%** (Fisher p = 0.008), while reaching *less* often (OS 24.0% vs 29.0/31.7%).
>   No transform shows a significant SR difference at n=300. See
>   `receipts/analysis/ROBUSTNESS.md` §C and `receipts/KNOWN_GAPS.md` G6.
> - **Mechanism 1 as a causal camera-height claim did not survive the controlled sweep**:
>   across the pre-specified height ladder (n=200/arm) there is no monotone SR trend
>   (z = +0.828, p = 0.408; `receipts/analysis/ANALYSIS.md`). The viewpoint-gap numbers
>   below are kept as description, not cause.
>
> The 2,400-episode sweep's strongest result is about evaluation reliability itself —
> see `paper_evidence/PAPER_RESULTS.md` and `receipts/KNOWN_GAPS.md`.

**Run:** `full_14498` · completed 2026-06-22 · **COMPLETE: 1077/1077 scored, 0 missing** (the 27
straggler timeouts were recovered on a retry pass with a doubled wall-clock guard — they were slow
scenes, not hung). · ~54 h total GPU compute · lab RTX 3090.
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
| **K1 (ours, vision)** | **model_14498** (n=1077/1077) | **7.59** | **30.3** | **18.3** | **10.9** |
| K1 — prior stack (reference) | v3 model_16000 | — | ~10 | 0.76* | — |

\* Recorded value; the raw run directory was not retained (`EVIDENCE.md` claim 7) — treat as
"under 1%". The first complete pipeline scored under 1%; the final stack scores 18.3%. NE is
essentially tied with the paper's blind-H1 (7.59 vs 7.67); OS/SR/SPL sit a notch below it —
the viewpoint gap is described (not causally attributed) below.

## Decomposition (where the episodes go) — complete n=1077
- **197 successes = SR 18.3%.**
- **326 reaches = OS 30.3%** — got within the 3.0 m goal radius at some point.
- **Stop-given-reached = 197/326 = 60%** — of episodes that reached, 60% recognized arrival and
  stopped. (Paper blind-H1 implies ~73%.)
- **Reached-but-didn't-stop = 129 (12.0%)** — the stop-failure class: reached the goal radius but
  did not stop. What drives it is unresolved; the FOV mechanism proposed earlier reversed at
  n=300 (see correction notice).
- **Didn't reach = 69.7%** — the dominant observed failure class (exploration; described below).
- **0 missing** — all 1077 scored. (The first pass left 27 as persistent timeouts; doubling the
  harness wall-clock guard 900→1800 s recovered every one — they were slow scenes, not hung.
  Under measured nondeterminism a retry is a second draw; `receipts/PRE_REGISTRATION.md` discusses
  the caveat and why the bias is negligible here.)

## The viewpoint gap, described (causal claims withdrawn — see correction notice)
The same frozen NaVILA VLM drives H1 (blind) to SR 24.4 / OS 33.3 and K1 to SR 18.3 / OS 30.3
(the locomotion passed every sim gate; the VLM is identical and was shown stop-capable).
Descriptively:
- **Both failure classes contribute.** K1's eye sits at **0.78 m** vs H1's ~1.5 m (derived).
  **SPL among successes is 60%** (≈1.7× the optimal path) vs blind-H1's ~86%; successful K1
  runs wander before arriving. Mean path length 22.1 m. NE among the 197 successes is
  **1.53 m** (it stops well inside the radius when it does stop). Against blind-H1, the
  stop-given-reached shortfall (60% vs ~73%) and the lower reach rate both contribute to the
  SR gap; within K1's own episodes, never-reached (69.7%) is the largest class.
- **Whether the lower viewpoint *causes* the gap is unresolved.** The pre-specified
  camera-height ladder (n=200/arm) found no monotone SR trend, so this run's descriptive gap
  cannot be attributed to camera height on this stack's own evidence.

Supporting controls: **int8 vs fp16 action drift 7.1%** (below the model's own 14.9%
sensitivity to an imperceptible JPEG re-encode; 100% agreement on success episodes — int8 is
not a confound). The **5090 was disqualified as an eval machine** after visible frame
corruption (Isaac 4.1 lacks Blackwell support, ~17% frame speckle) and an SR 0/10 screen on
episodes where the 3090 (clean render) scored 3/10 — a 10-episode hardware screen, within
run-to-run noise, not a finding.

## Caveats (stated plainly)
- **n = 1 robot, single VLM.** A deployment/embodiment result, not a broad empirical claim.
- **Sim-only numbers.** No real-robot navigation score is claimed (deployment networking solved —
  `--net eno1` / GetMode verified — and the trained policy has not walked on hardware). A live
  NaVILA episode via the early laptop relay (May 2026, Booster's built-in walker executing the
  VLM's velocity commands) is receipted in `receipts/media/LIVE_EPISODE_RECEIPT.md`.
- **int8** (to co-fit the VLM with Isaac on 24 GB) — controlled and shown ≈fp16 at the action level.
- **Known rendering defect (found post-sweep):** robot head/arm/logo geometry drawn at the
  spawn pose can appear in the robot camera
  (`paper_evidence/airc2027_post_sweep/rendering_diagnosis/DIAGNOSIS.md`); it is visible in
  this run's era and its effect on the headline numbers is not quantified.

## Provenance
Measurements: `NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_full_14498/measurements/`
(1077 JSON, re-verified 2026-07-02). Aggregator: `aggregate_k1_vision_results.py` (NE = final distance_to_goal; missing =
failures over 1077). Full project record + paper analysis: offline project dossier (local notes, not in-repo).
Method/controls: `RESULTS_2026-06-11_smoke.md`, `REPLAY_ARBITER_2026-06-11.md`.
