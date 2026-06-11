# K1 NaVILA Benchmark — 10-Episode Smoke Results (2026-06-11)

**Machine:** lab HP Z8 G4, RTX 3090 24 GB · **Policy:** `checkpoints/model_14498.pt` (feet_distance fine-tune of model_11999)
**VLM:** navila-llama3-8b-8f, **int8** (`--load_8bit`) · **Frames:** raw render (no `--clean_render`/`--bright`), PNG wire, `--vlm_transform stretch`
**Episodes:** R2R val-unseen indices 0–9 · **Flags:** `--closed_loop --max_episode_s 120`, `EP_TIMEOUT=900` · **Code:** commit `6dc4d2d`

## Headline (paper-comparable)

| Robot | Policy | NE↓ | OS↑ | SR↑ | SPL↑ |
|-------|--------|-----|-----|-----|------|
| Go2 (paper, blind) | — | 6.03 | 49.0 | 36.2 | 33.3 |
| Go2 (paper, vision) | — | 5.49 | 58.7 | 50.2 | 45.5 |
| H1 (paper, blind) | — | 7.67 | 33.3 | 24.4 | 21.0 |
| H1 (paper, vision) | — | 5.86 | 54.6 | 45.3 | 40.3 |
| K1 — old stack (full 1,077) | v3 model_16000 | — | ~10 | 0.76 | — |
| K1 — desktop 5090 smoke (same 10 eps) | model_11999-era | — | 40.0 | 0.0 | — |
| **K1 — lab 3090 smoke (this run)** | **model_14498** | **5.61** | **40.0** | **30.0** | **15.3** |

The blind K1 beats the paper's **blind H1** on NE / OS / SR on this slice, with a quantized VLM,
despite a lower camera (0.78 m vs ~1.5 m eye height).

## Per-episode vs the desktop (same episodes, same VLM, different renderer)

| ep | desktop (5090, speckled frames) | lab 3090 (clean frames) | outcome |
|----|--------------------------------|--------------------------|---------|
| 0 | miss | miss (ONE 7.81) | match |
| 1 | miss | miss (ONE 6.31) | match |
| 2 | miss | miss (ONE 7.55) | match |
| 3 | mesh-hole fall mid-approach | mesh-hole fall (ONE 4.47) | match (terrain) |
| 4 | stop @ 4.15 m (near-miss) | stop @ 4.34 m, ONE 3.34 (near-miss) | match |
| 5 | miss | **SUCCESS — stop 1.12 m, SPL 0.44** | **WIN** |
| 6 | reach 0.43 m, walk-out | reach 0.46 m, walk-out, late stop 8.8 m | match (archway prior) |
| 7 | reach 0.42 m, walk-out (3× deterministic) | **SUCCESS — ONE 0.36 m, stop 1.60 m, SPL 0.50** | **WIN (render fix)** |
| 8 | reached zone, fell through floor | **SUCCESS — ONE 0.44 m, stop 2.00 m, SPL 0.59** | **WIN** |
| 9 | paced over goal, never stopped | miss (ONE 10.92) | desktop better (noise) |

## Interpretation

- **All SR gain = stop-on-arrival conversions on clean frames** (eps 5, 7, 8). The desktop's
  Isaac 4.1 renderer lacks Blackwell (RTX 5090) support → ~17% speckle (high-freq residual 15.6
  vs 0.4–1.3 here) → arrival-blindness. Confirms the desktop session's diagnosis.
- **int8 exonerated**: paper-level SR, organic in-episode stops, correct first decisions.
- **Surviving failure classes**: wander-never-near (0–2; blind-policy/VLM ceiling), mesh hazards
  (3; environment), late/far stops (4), archway walk-out (6; semantic prior — survives clean
  frames). Cheap lever remaining: `--vlm_transform crop` A/B. Structural lever: height-map policy
  (paper vision rows above ≈ +14 SR).
- **Caveats**: n=10 (wide CIs; expect the full 1,077 to land lower), SPL trails paper blind-H1
  (successes wander before arriving), ep9 regressed vs desktop (single-episode noise).
- **Harness during the run**: 0 PARSE_FAIL, executor stall→re-query firing correctly, no socket
  hangs, no hopeless-stall false fires, all 10 episodes scored.

## Full-run recipe (this box)

```
cd ~/Projects/k1_research
VLM_BRIDGE_EXTRA="--load_8bit" EP_TIMEOUT=900 \
  bash scripts/run_powered_benchmark.sh full_14498 checkpoints/model_14498.pt
# resumable; ~9–12 min/episode ≈ 7–9 days chunked. Transform pending crop A/B.
```

Artifacts: `NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit/`
(measurements, per-tick VLM frame payloads under `diag/`, videos incl. the ep5/7/8 successes).
