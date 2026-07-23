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
