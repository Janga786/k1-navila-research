# Locomotion training-run ledger (primary source: run dirs + pipeline logs)

Disk census: **31 run directories** (`booster_train/logs/rsl_rl/k1_navila/` = 28,
`k1_navila_t1/` = 3). Below: every substantive named run. GPU-hours from measured
iteration times (6.6–7.4 s/iter @ 4096 envs) or wall-clock in pipeline logs.

> **FLAG:** if a slide says "~9 training runs," that undercounts. Honest options:
> 22 substantive runs (below), or 31 run dirs on disk, or "3 runs on the winning
> path + 19 documented dead ends/diagnostics."

## Original lineage (June) — produced model_14498
| Run dir | Date | Config | Iters | ~GPU-h | Outcome |
|---|---|---|---|---|---|
| (6 exploratory dirs 06-03→06-05) | 06-03..05 | setup/backbone iterations | var | ~15 | superseded |
| 2026-06-05_05-48-43 | 06-05 | backbone recipe → model_8700 | ~8.7k | ~16 | KEPT then superseded — walks, circumducts |
| 2026-06-08_16-09-29 | 06-08 | round-1: +5 terms at once | ~11k | ~20 | REJECTED — regressions (fell, over-turned) |
| 2026-06-09_13-52-48 | 06-09 | round-2: +swing_height −20, +hip_dev | 12k | ~22 | **KEPT → model_11999** (circ −33%) |
| 2026-06-10_14-30-20 | 06-10 | round-3: fine-tune 11999 + feet_distance | +2.5k | ~5 | **KEPT → model_14498** (0 foot contacts; benchmark policy) |

## Ablation sweep (06-26→29) — diagnostic
| 5 dirs (06-26..28) | baseline / −swing_height / −feet_distance / −hip_dev / swing−30 | 6.5k each | ~58 total | evidence base (weekend_sweep/WEEKEND_RESULTS.md) |

## Gait quest (July) — produced model_14200
| Run dir | Date | Config | Iters | ~GPU-h | Outcome |
|---|---|---|---|---|---|
| 2026-07-13_14-08-34 | 07-13 | round-4 FT 14498 (swing −30/hip −0.2) | +3k | 5.3 | REJECTED — gait degraded monotonically |
| k1_navila_t1/2026-07-14_08-39-37 | 07-14 | T1-faithful v1 (unported clip_actions) | 12k | 21.7 | COLLAPSED — Mean reward 0.00 throughout |
| k1_navila_t1/2026-07-15_06-28-20 | 07-15 | v2 fixed structure (no terminate_vel) | →3.26k + 80 resume | ~6.5 | COLLAPSED — critic divergence (1.4e7) |
| k1_navila_t1/2026-07-15_12-48-27 | 07-15 | v3 + terminate_vel guard | →12k | 17.3 | REJECTED by user — trots; undershoots speed |
| (v4, dir removed) | 07-16 | v3 + stronger tracking | killed ~2.8k | ~5 | KILLED per user (line abandoned) |
| 2026-07-16_09-03-22 | 07-16 | r5 FT 14498 + rails −10 | +3k | 5.8 | REJECTED — fell on turns |
| 2026-07-16_15-00-02 | 07-16 | r5a rails −5 turn-gated, no floor | +3k | 6.2 | REJECTED — first zero-waiver tracking pass, but 2.8% foot clipping |
| 2026-07-16_21-16-50 | 07-16 | r5b rails + floor | +3k | 6.2 | REJECTED — clipping 9% |
| (r5c, dir removed) | 07-17 | rails + floor, from scratch | cancelled 3k/12k | ~6 | CANCELLED per user |
| 2026-07-17_09-46-56 | 07-17 | r6: original + POSITIVE bands | 12k | 24.5 | REJECTED — reward farming (hip 28°) |
| 2026-07-18_10-24-12 | 07-18 | r6b: original + penalty feet band | 12k | 24.3 | REJECTED — curriculum effect (hip 20°) |
| 2026-07-19_10-50-11 | 07-19 | **r7: FT model_11999 + floor + penalty band** | +2.5k | 5.2 | **KEPT → model_14200 @ iter 14200** (checkpoint sweep; all 9 gates) |

Totals: ~22 substantive runs, ≈ **270 GPU-hours** June+July combined
(≈170 h in the July quest alone). Winning path: 11999 → 14498 → 14200-r7.
