# K1 NaVILA — vision-language navigation on a Booster K1 humanoid

Research code and evidence for running the pretrained **NaVILA** model (8B parameters), a vision-language
navigation model, on the **Booster K1** humanoid, and for measuring how reliable that evaluation is.
Fort Lewis College, full-time research from May 2026. Advisor: Dr. Yiyan Li.

## What's here

- **Two-tier stack.** NaVILA issues mid-level commands at ~1 Hz; an RL locomotion policy I trained
  in Isaac Sim / Isaac Lab tracks them at 50 Hz (235-dim observation, 12-DoF action).
- **Full benchmark.** The K1 integrated into VLN-CE-Isaac (R2R val-unseen) and run on all
  **1,077 episodes**: **SR 18.3% · OS 30.3% · NE 7.59 m · SPL 10.9**, up from under 1% (recorded
  as 0.76%; raw run not retained, see `EVIDENCE.md` claim 7) for the first full pipeline, over 22 training runs (~270 GPU-hours) and pipeline fixes. Simulation results.
- **Evaluation reliability.** In 2,400 preregistered ablation episodes, two
  identically configured 300-episode runs matched in aggregate (15.0% vs 16.0% SR) but disagreed on
  **11.0%** of episode outcomes (**Cohen's kappa 0.58**), and a fixed wall-clock limit censored
  conditions unequally (2.0% vs 12.3% timeouts). A 10-episode pilot effect did not replicate
  at n = 300; the correction is kept on the record.
- **Physical robot.** In the May 2026 session, NaVILA drove the real K1 **live** through the
  early laptop relay (RTX 5090 inference, ~350 ms/inference; Booster SDK velocity control):
  VLM-issued velocity commands executed by Booster's built-in walking controller, with zero
  teleoperation. Two live natural-language episodes were run ("walk to the volleyball and
  stop", "walk to the volleyball and turn left"); one is captured on video, the second is on
  the project record. Hosted recordings (the mp4 files are not tracked in git):
  - [Relay screen recording + robot](https://janga786.github.io/media/k1-relay-live.mp4) — the
    relay in `[LIVE]` mode streaming VLM velocity commands over the robot's own head camera.
  - [Walk to the volleyball](https://janga786.github.io/media/k1-walk-to-volleyball.mp4) —
    third-person clip of the K1 walking toward the volleyball during the live episode.

  Details and claim boundary: [`receipts/media/LIVE_EPISODE_RECEIPT.md`](receipts/media/LIVE_EPISODE_RECEIPT.md);
  no navigation score is claimed for the live session. The later three-machine deploy relay is wired and documented
  (`ROBOT_DEPLOY_RUNBOOK.md`) with print/dry/live modes and safety interlocks. **Both live
  episodes used Booster's built-in walker; my trained RL walking policy has not yet worked
  outside simulation.**

## Start here

| File | What it is |
|---|---|
| [`FINAL_RESULTS_full1077.md`](FINAL_RESULTS_full1077.md) | Full 1,077-episode results vs. the published NaVILA Go2/H1 baselines |
| [`paper_evidence/README.md`](paper_evidence/README.md) | Reanalysis package: repeatability, timeout censoring, scene uncertainty, power |
| [`receipts/PRE_REGISTRATION.md`](receipts/PRE_REGISTRATION.md) | Preregistration for the 2,400-episode ablation |
| [`receipts/KNOWN_GAPS.md`](receipts/KNOWN_GAPS.md) | Open issues and corrections, stated plainly |
| [`EVIDENCE.md`](EVIDENCE.md) | Claim-by-claim verification against primary sources |
| [`ROBOT_DEPLOY_RUNBOOK.md`](ROBOT_DEPLOY_RUNBOOK.md) | How the physical-robot relay is wired and run |
| [`receipts/media/LIVE_EPISODE_RECEIPT.md`](receipts/media/LIVE_EPISODE_RECEIPT.md) | The May 2026 live NaVILA episode: relay screen recording + third-person video |
| [`checkpoints/`](checkpoints/README.md) | Final locomotion policy checkpoints (`.pt`, TorchScript, ONNX) |

## Layout

- `checkpoints/` — deployment-bound policy weights and their provenance
- `paper_evidence/` — CPU-only, deterministic reanalysis scripts, tables, and figures (folder
  names such as `paper_evidence/airc2027_*` are historical labels for the reanalysis evidence,
  kept so internal links and hashes stay valid)
- `receipts/` — run ledgers, preregistration, determinism probes, raw metric outputs
- `scripts/` — benchmark and analysis helpers
- `archive/` — lab-notebook status logs and slide-prep ledgers (`HANDOFF.md`, `AWAY_STATUS.md`,
  `SLIDE_FACTS.md`, …) kept for the record, annotated where later evidence corrected an entry

## Credits and provenance

Built on [NaVILA](https://navila-bot.github.io/) and the VLN-CE-Isaac benchmark. Robot: Booster K1 EDU.
The experiment designs, decisions, and claims are mine; substantial parts of the implementation and
the unattended sweep operations were carried out with AI coding agents under my direction (the
handoff notes in `archive/` record that process).
