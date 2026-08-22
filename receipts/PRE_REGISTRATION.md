# Pre-registration — transform ablation + camera-height sweep (K1 NaVILA)
Written 2026-07-23, BEFORE any arm runs. Fixed in advance; not to be revised after seeing arm data.

## Baseline (free anchor, from full_14498 stretch, episodes 0-299)
| subset | n | SR | OS | NE | mean path | frac>20m |
|---|---|---|---|---|---|---|
| **0-299 (pinned arm set)** | 300 | **17.7%** | **31.7%** | **7.22 m** | 22.7 m | 62% |
| 0-99 (old pilot slice) | 100 | 17.0% | 31.0% | 8.17 m | 23.8 m | 69% |
| 0-1076 (global headline) | 1077 | 18.3% | 30.3% | 7.59 m | 22.1 m | 59% |

**The comparison baseline for the stretch arm is the 0-299 subset (17.7 / 31.7 / 7.22), NOT
the global 18.3%** (different episode population). 0-299 is representative: SR/OS within
~1.4 pp of global; skew is mild and documented here.

## Reproducibility receipt (pre-registered)
The fresh **stretch @ 300 (retried protocol)** run is pre-registered as a reproducibility
check. Expected: SR ~17.7 / OS ~31.7 / NE ~7.22, matching the full_14498 0-299 subset.
- **If it reproduces** (within Monte-Carlo/nondeterminism tolerance from the probe): a
  reproducibility receipt, worth its ~2.4 days.
- **If it does NOT reproduce**: STOP — indicates code drift or nondeterminism since June;
  escalate before interpreting any ablation. Do not proceed.

## Hypothesis & analysis plan (fixed in advance)
- **Design:** 3 transforms {stretch, crop, pad} x 300 episodes, single run each (no seeds —
  seed is a no-op and the pipeline is greedy/no_grad; determinism verdict from the probe).
  Same pinned episode set 0-299 across all arms (paired).
- **Primary test:** McNemar on paired per-episode binary success across transform arms
  (episode-aligned by index). NOT a two-sample rate test.
- **Primary wandering metric:** `ended_at_step` (sim steps — deterministic, load-independent),
  reported as a distribution grouped by `term_reason` per arm.
- **Camera sweep:** 6 ground-referenced heights {0.60, 0.78(native), 0.95, 1.10, 1.30, 1.50 m}
  x 200 episodes, same pinned set. Heights above the robot's mountable range labeled
  "virtual viewpoint." 1.50 m anchored to H1's ~1.55 m camera in this codebase.

## Pre-registered NULL (point 5)
Stretch is 18.3% at n=1077; the earlier n=10 pilot (stretch 0/10) is statistically consistent
with noise. **A null result on the transform ablation is an expected and acceptable outcome.**
The n=10 "crop 0 vs stretch 30" effect has already NOT replicated at n=100 (stretch 12.0% over
100). No script, log, or summary asserts an expected direction. We report what lands. The
analysis plan above is fixed; no post-hoc reinterpretation.

## Determinism gate on the retry pass (point 4) — HARD GATE
The two-pass protocol (900 record + 1800 straggler retry, matching full_14498) is only valid
if the pipeline is deterministic:
- **Deterministic** -> a 1800s retry is a clean re-execution of the same episode; protocol sound.
- **NON-deterministic** -> a retry is a second independent draw; retried episodes get two
  chances at success while others get one = upward bias. Worse, transforms with higher timeout
  rates (see Q4: stretch 14 vs crop 10 over 0-49) would get MORE retries = differential bias =
  a confound to the ablation. NOTE: full_14498 itself used a retry pass (27 stragglers), so its
  baseline carries the same bias — comparable only if retry rates match across arms.
- **RULE: do not launch the two-pass design until the determinism probe lands. If it shows
  nondeterminism, STOP and escalate before running any retry.**

## Wall-timeout handling (points 3 + B)
Wall-timeout JSONs: force ONLY `success=0` and `spl=0` (both require a stop). `oracle_success`,
`oracle_navigation_error`, `path_length` take their real partial values when present (they do
NOT require a stop — a wall-timeout can legitimately have OS=1), sentinel only when the kill
lands before the first step. Distance sentinel = `-1.0` (real distances are >=0; 0.0 would
read as "at the goal"). Aggregator computes NE/ONE over `distance_to_goal >= 0` only.

## Probe coverage limit (accepted)
The determinism probe cannot diff true wall-timeouts (no JSON). Once the wall-timeout fix is
in, the two longest episodes become diffable; add them to a follow-up probe and record here.

## DETERMINISM VERDICT (2026-07-23) — NON-DETERMINISTIC. HARD STOP.
Extended probe: 6 episodes {3,8,16,50,20,99} run twice under byte-identical settings
(model_14498, int8, stretch, closed_loop, max_episode_s 120), no --seed, no variation.
**Result: 6/6 episodes DIFFERED between the two runs; 2/6 FLIPPED the success outcome.**
- ep 11 (idx-mapped): run A path 21.1m dtg 7.30 success=0  |  run B path 14.0m dtg 1.37 success=1
- ep144: run A dtg 2.57 success=1  |  run B dtg 11.76 success=0   (flip, opposite direction)
- ep 16/50/77: macroscopic trajectory divergence (e.g. path 19.3m vs 28.9m same episode).
Trajectories diverge chaotically: tiny run-to-run differences (GPU physics and/or non-
deterministic CUDA in the greedy VLM forward, flipping knife-edge scanning decisions) amplify
through the closed VLM->policy->physics loop into different outcomes.

CONSEQUENCES (pre-registered gate point 4 fires):
1. Two-pass 1800 retry is INVALID as designed — a retry is a 2nd independent draw; timed-out
   episodes would get two chances at success, differentially across transforms. DO NOT RUN IT.
2. Broader: single-run-per-arm + McNemar-on-paired-outcomes is undermined — the per-episode
   "pair" is not a stable outcome (ep 11/144 flip). Paired testing tests noise.
3. NOT invalidated: the aggregate 18.3% over n=1077 (and 17.7% over 0-299) — an aggregate rate
   over large N is a valid estimate; per-episode noise averages out in the rate. full_14498's
   27 straggler retries (2.5%) got 2nd draws -> negligible bias on 18.3%.
STATUS: STOPPED before any launch. Awaiting design decision (replication vs larger-N-unpaired
vs determinism-hardening). Nothing edited in the eval; no runs launched.

## DRIVER-CHANGE CONFOUND (2026-07-24) — METHODS-CRITICAL, record now
An unattended apt upgrade on 2026-07-24 06:45 replaced the NVIDIA driver 580.159.03 ->
580.173.02 and (via a driver/module mismatch) took CUDA down. 580.159.03 has been PURGED
from the Ubuntu archive and is unobtainable, so it cannot be restored. Recovery: rebuilt the
580.173.02 kernel module from source with gcc-12 (DKMS); machine now runs driver 580.173.02.
CONSEQUENCES FOR THE PAPER (state both explicitly in Methods):
1. full_14498 (SR 18.3%; 0-299 subset 17.7%) ran on driver **580.159.03**. Every NEW arm
   (stretch/crop/pad @300, height sweep) runs on **580.173.02**. The stretch@300 vs
   full_14498[0-299] comparison therefore SPANS A DRIVER CHANGE: it is an UPPER BOUND on
   reproducibility (driver-version + run-to-run nondeterminism combined), NOT a clean
   run-to-run noise floor. Label it that way in every table/figure/caption.
2. Methods must report BOTH driver versions and which results came from which: 580.159.03 =
   the 18.3% headline + all June/July results through 2026-07-24; 580.173.02 = every arm
   from the 2026-07-24 sweep onward. Combined with the nondeterminism finding, single-run SR
   in this benchmark is not exactly reproducible; report it as an estimate with a CI.

## ROOT-CAUSE OF NONDETERMINISM (2026-07-24, driver 580.173.02)
- 5b (VLM determinism), STRENGTHENED: 50 real VLM queries sampled across 10 episodes
  (turn/scan/move phases), each fed twice, token-diffed. **50/50 IDENTICAL (100%), including
  10/10 turn/decision-phase queries.** The VLM forward pass is deterministic even on the
  knife-edge scanning/turning cases where this project previously found the VLM unstable.
- CONCLUSION: the chaotic run-to-run divergence originates SIM-SIDE (Isaac physics and/or
  rendering), NOT in the VLM. Publishable as "chaotic divergence originating in the
  simulator, not the policy/VLM."
- 5a (physics-vs-render split): see note below.

## 5a (render/physics determinism) — RESOLVED (2026-07-24)
Same episode-3 start pose rendered in two separate processes -> frames DIFFER:
max pixel |diff| 21/255, mean 0.30, 25.68% of channels nonzero. The simulator produces
NON-bit-identical observations for the same commanded state across runs. Combined with 5b
(VLM 50/50 identical): the closed loop is seeded by sim-side render/physics nondeterminism,
which the sensitive VLN policy amplifies into chaotic trajectory divergence. Precise claim:
"VLM deterministic; simulator observations vary run-to-run (~26% of pixels, <=8% magnitude);
policy amplifies -> chaotic divergence." Physics-vs-render split not separated (the test
conflates settle+render); not pursued further (timeboxed).

---

## AMENDMENT 1 (2026-07-27) — primary test changed McNemar -> unpaired two-proportion
Recorded BEFORE any sweep arm data was inspected; the first arm (stretchA) completed
2026-07-26 but its numbers were not read into the analysis design. Logged in HANDOFF.md the
same day.

**What changed.** The "Hypothesis & analysis plan" section above names *McNemar on paired
per-episode binary success* as the primary test. That test is INVALID here, by this
document's own determinism gate (§"DETERMINISM VERDICT", §"CONSEQUENCES" point 2): with 6/6
probe episodes differing run-to-run and 2/6 flipping outcome, a per-episode "pair" is not a
stable unit and paired testing would test noise rather than treatment.

**Replacement (fixed in advance, not revised after seeing data):**
- Primary test: **unpaired two-proportion** comparison of SR and OS between arms, reported
  with the exact (Fisher) p-value as primary and the normal-approximation z-test alongside.
- Interval estimates: **Wilson** 95% CIs on every rate, with Clopper-Pearson reported too.
- Multiplicity: **Holm-Bonferroni within each contrast family x metric** (noise floor /
  transform / height / driver), since the sweep runs many contrasts.
- Height sweep additionally gets a **Cochran-Armitage trend test** over camera height. It
  tests monotone trend only; a non-monotone optimum is not something this design can claim.
- Resolution floor at these n is ~8.7 pp (n=300 vs 300) and ~10.7 pp (n=200 vs 200) at
  alpha=0.05, power=0.80. Effects smaller than that were never detectable here.

**Note on pairing.** All arms are in fact episode-aligned (verified in ANALYSIS.md §0), so the
design is *structurally* paired. We deliberately do NOT exploit that, for the reason above.
Using unpaired tests on episode-aligned data is conservative — it discards a variance
reduction we cannot trust — and that direction of error is the right one here.

## AMENDMENT 2 (2026-08-22) — reproducibility-receipt branch RESOLVED: it reproduced
The "Reproducibility receipt (pre-registered)" section above set a hard branch: if the fresh
stretch@300 run did not reproduce the full_14498 0-299 anchor (SR ~17.7 / OS ~31.7 / NE ~7.22),
**STOP and escalate before interpreting any ablation.**

**Outcome: it reproduced. The gate does not fire; the ablation is interpretable.**

| run | driver | SR | OS | NE |
|---|---|---|---|---|
| full_14498 [pinned 300] | 580.159.03 | 17.7% | 31.7% | 7.22 m |
| stretchA @300 | 580.173.02 | 15.0% | 29.0% | 7.22 m |
| stretchB @300 | 580.173.02 | 16.0% | 31.7% | 7.06 m |

No contrast against the anchor is significant (SR: p=0.44 and p=0.66; OS: p=0.53 and p=1.00 —
see ANALYSIS.md §3.4). A second independent probe at the native camera height (h078 vs the
same baseline on the 200-episode set) is likewise null. This holds ACROSS the 580.159.03 ->
580.173.02 driver change, so it is an upper bound on reproducibility — it bounds driver-change
effects and run-to-run nondeterminism *combined*, and remains consistent with zero.

**Caveat that must travel with this receipt:** "reproduced" here means *no detectable
difference at this power*, not *equivalence*. With a ~8.7 pp resolution floor, a real effect of
a few pp would have been invisible. Do not upgrade this to an equivalence claim; report it as
a null with the CI attached.
