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
  identically configured 300-episode runs matched in aggregate (15.0% vs 16.0% SR) but disagreed on
  **11.0%** of episode outcomes (**Cohen's kappa 0.58**), and a fixed wall-clock limit censored
  conditions unequally (2.0% vs 12.3% timeouts). A 10-episode pilot effect did not replicate
  at n = 300; the correction is kept on the record.
- **Physical robot.** In May 2026, NaVILA drove the real K1 **live** through the early laptop
  relay: VLM-issued velocity commands (no teleoperation, ~350 ms/inference) executed by
  Booster's built-in walking controller, walking the robot toward a volleyball on instruction.
  Receipts — a screen recording of the live relay plus a dated third-person clip — are in
  [`receipts/media/`](receipts/media/LIVE_EPISODE_RECEIPT.md); no navigation score is claimed
  for the live session. The later three-machine deploy relay is wired and documented
  (`ROBOT_DEPLOY_RUNBOOK.md`) with print/dry/live modes and safety interlocks. **The trained
  RL policy was evaluated in simulation only and has not walked on hardware.**

## Start here

| File | What it is |
|---|---|
| [`FINAL_RESULTS_full1077.md`](FINAL_RESULTS_full1077.md) | Full 1,077-episode results vs. the NaVILA paper's Go2/H1 numbers |
| [`paper_evidence/README.md`](paper_evidence/README.md) | AIRC 2027 reanalysis package: repeatability, timeout censoring, scene uncertainty, power |
| [`receipts/PRE_REGISTRATION.md`](receipts/PRE_REGISTRATION.md) | Preregistration for the 2,400-episode ablation |
| [`receipts/KNOWN_GAPS.md`](receipts/KNOWN_GAPS.md) | Open issues and corrections, stated plainly |
| [`EVIDENCE.md`](EVIDENCE.md) | Claim-by-claim verification against primary sources |
| [`ROBOT_DEPLOY_RUNBOOK.md`](ROBOT_DEPLOY_RUNBOOK.md) | How the physical-robot relay is wired and run |
| [`receipts/media/LIVE_EPISODE_RECEIPT.md`](receipts/media/LIVE_EPISODE_RECEIPT.md) | The May 2026 live NaVILA episode: relay screen recording + third-person video |
| [`checkpoints/`](checkpoints/README.md) | Final locomotion policy checkpoints (`.pt`, TorchScript, ONNX) |

## Layout

- `checkpoints/` — deployment-bound policy weights and their provenance
- `paper_evidence/` — CPU-only, deterministic reanalysis scripts, tables, and figures
- `receipts/` — run ledgers, preregistration, determinism probes, raw metric outputs
- `scripts/` — benchmark and analysis helpers
- `archive/` — lab-notebook status logs and slide-prep ledgers (`HANDOFF.md`, `AWAY_STATUS.md`,
  `SLIDE_FACTS.md`, …) kept for the record, annotated where later evidence corrected an entry

## Credits and provenance

Built on [NaVILA](https://navila-bot.github.io/) and the VLN-CE-Isaac benchmark. Robot: Booster K1 EDU.
The experiment designs, decisions, and claims are mine; substantial parts of the implementation and
the unattended sweep operations were carried out with AI coding agents under my direction (the
handoff notes in `archive/` record that process).
