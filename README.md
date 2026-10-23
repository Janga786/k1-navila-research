# K1 NaVILA — vision-language navigation on a Booster K1 humanoid

Research code and evidence for running **NaVILA** (an existing 8B vision-language navigation
model) on the **Booster K1** humanoid, and for measuring how reliable that evaluation is.
Fort Lewis College, summer 2026. Advisor: Dr. Yiyan Li. Paper in preparation for AIRC 2027.

## What's here

- **Two-tier stack.** NaVILA issues mid-level commands at ~1 Hz; an RL locomotion policy I trained
  in Isaac Sim / Isaac Lab tracks them at 50 Hz (235-dim observation, 12-DoF action).
- **Full benchmark.** The K1 integrated into VLN-CE-Isaac (R2R val-unseen) and run on all
  **1,077 episodes**: **SR 18.3% · OS 30.3% · NE 7.59 m · SPL 10.9**, up from under 1% (recorded
  as 0.76%; raw run not retained, see `EVIDENCE.md` claim 7) for the first full pipeline, over 22 training runs (~270 GPU-hours) and pipeline fixes. Simulation results.
- **Evaluation reliability (the paper).** In 2,400 preregistered ablation episodes, two
  byte-identical 300-episode runs matched in aggregate (15.0% vs 16.0% SR) but disagreed on
  **11.0%** of episode outcomes (**Cohen's kappa 0.58**), and a fixed wall-clock limit censored
  conditions unequally (2.0% vs 12.3% timeouts). A 10-episode pilot effect did not replicate
  at n = 300; the correction is kept on the record.
- **Physical robot.** A three-machine relay (RTX 5090 VLM inference, ~330 ms) ran NaVILA on the
  real K1 for two natural-language episodes with zero teleoperation, using Booster's built-in
  walker (the trained policy was evaluated in simulation).

## Start here

| File | What it is |
|---|---|
| [`FINAL_RESULTS_full1077.md`](FINAL_RESULTS_full1077.md) | Full 1,077-episode results vs. the NaVILA paper's Go2/H1 numbers |
| [`paper_evidence/README.md`](paper_evidence/README.md) | AIRC 2027 reanalysis package: repeatability, timeout censoring, scene uncertainty, power |
| [`receipts/PRE_REGISTRATION.md`](receipts/PRE_REGISTRATION.md) | Preregistration for the 2,400-episode ablation |
| [`receipts/KNOWN_GAPS.md`](receipts/KNOWN_GAPS.md) | Open issues and corrections, stated plainly |
| [`EVIDENCE.md`](EVIDENCE.md) | Claim-by-claim verification against primary sources |
| [`ROBOT_DEPLOY_RUNBOOK.md`](ROBOT_DEPLOY_RUNBOOK.md) | How the physical-robot relay is wired and run |
| [`checkpoints/`](checkpoints/README.md) | Final locomotion policy checkpoints (`.pt`, TorchScript, ONNX) |

## Layout

- `checkpoints/` — deployment-bound policy weights and their provenance
- `paper_evidence/` — CPU-only, deterministic reanalysis scripts, tables, and figures
- `receipts/` — run ledgers, preregistration, determinism probes, raw metric outputs
- `scripts/` — benchmark and analysis helpers
- Top-level `*_STATUS.md`, `HANDOFF.md`, `RUN_STATUS.md` — lab-notebook status logs kept for the record

## Credits

Built on [NaVILA](https://navila-bot.github.io/) and the VLN-CE-Isaac benchmark. Robot: Booster K1 EDU.
