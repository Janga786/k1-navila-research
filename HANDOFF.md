# HANDOFF — NaVILA-on-K1 experiment sweep (compiled 2026-07-27)

You are picking up an unattended, month-long experiment on a lab workstation. The user
(Janga786 / jangarabliss@gmail.com) is AWAY from 2026-07-25 until 2026-08-23. Do NOT touch
the physical robot. The machine must stay awake and reachable the whole time.

## THE ONE THING THAT MATTERS: a sweep is running in tmux
A transform + camera-height ablation of the NaVILA vision-language-navigation model on the
Booster K1 humanoid, in Isaac Sim. Runner: `~/Projects/k1_research/arms_runner.sh`, running
inside tmux session **`sweep`** (reattach: `tmux attach -t sweep`, detach: Ctrl-b then d).
Log: `/tmp/sweep.log`. Status file (pushed to GitHub after each arm): `SWEEP_STATUS.md`.

### Arms, in order (single-pass @900s wall-clock, NO retry, guaranteed per-episode record):
1. `stretchA_300` — vlm_transform=stretch, eps 0-299 — **DONE: SR 15.0% / OS 29.0%, 0 missing**
2. `stretchB_300` — stretch (SAME-DRIVER REPLICATE = the noise floor) — **RUNNING (~74/300 as of 07-27 06:36)**
3. `crop_300` — vlm_transform=crop, eps 0-299
4. `pad_300` — vlm_transform=pad, eps 0-299
5. `h060_200` .. `h150_200` — 6-height camera sweep, stretch, eps 0-199, `--cam_z` = 0.07/0.25/0.42/0.57/0.77/0.97 (ground height 0.53+cam_z = 0.60/0.78/0.95/1.10/1.30/1.50 m; ≥0.9 m = "virtual viewpoint")

**ETA ~Aug 11-12** at ~11.4 min/episode; ~11-day buffer before the user returns. The runner
is RESUMABLE: re-running `arms_runner.sh` skips completed arms and completed episodes.

## YOUR JOB while it runs
1. Let it run. Surface each arm's completion (a background monitor watches SWEEP_STATUS.md
   for "ARM DONE"; re-arm with:
   `tail -n +1 -f ~/Projects/k1_research/SWEEP_STATUS.md | grep --line-buffered -E 'ARM DONE|ALL ARMS COMPLETE'`
   as a run_in_background Bash command).
2. If the sweep dies: relaunch it in tmux (it resumes):
   `tmux new-session -d -s sweep 'bash ~/Projects/k1_research/arms_runner.sh 2>&1 | tee /tmp/sweep.log'`
   The runner auto-relaunches the VLM bridge if it's down.
3. Health check pattern (use after any session gap):
   `tmux ls | grep sweep; pgrep -af navila_eval_v3; nvidia-smi; tail -5 SWEEP_STATUS.md`
   and confirm CUDA: `conda activate vlnce-isaac; python -c "import torch;print(torch.cuda.is_available())"`.

## HARD RULES (from the user, non-negotiable)
- **NEVER touch the physical robot** — no `deploy.py` without `--mujoco`, no SDK/robot commands.
- **No destructive git** (no reset --hard, checkout ., clean, stash, rebase, force-push).
- **Report facts, not assumptions.** Read the code/logs before claiming anything. A wrong
  "yes" costs the paper.
- **Do NOT change benchmark semantics.** No scoring/termination changes. Additive logging only.
- **STOP and ask** before any hard-to-reverse or system-level change.
- The user gives explicit launch/decision gates. Don't launch multi-day work without a go.
- Sudo: the user authorized it for THIS machine's upkeep. Password is in the session
  transcript (it was exposed; the user was told to change it). Don't echo it needlessly.

## KEY FACTS / PRIOR FINDINGS (so you don't re-derive them)
- **Headline result (the paper's anchor):** full_14498 benchmark = SR **18.3%** over n=1077
  (0-299 subset = 17.7%; single-pass-equivalent 16.7%). Ran on driver **580.159.03**.
- **DRIVER CHANGE CONFOUND:** an auto-upgrade broke CUDA; recovered by rebuilding the module
  with gcc-12. Machine now runs **580.173.02** (580.159 is PURGED from Ubuntu, unrecoverable).
  So every sweep arm is on 580.173; the baseline was 580.159. The stretch-vs-baseline compare
  spans a driver change (upper bound on reproducibility). stretchB is the same-driver replicate.
- **The pipeline is NON-DETERMINISTIC (a finding, not a bug):** identical settings → different
  trajectories/outcomes (chaotic divergence). ROOT CAUSE localized: 5b showed the VLM is
  deterministic (50/50 real queries incl. turn phases); 5a showed the simulator renders
  non-identical pixels for the same pose. So chaos is SIM-SIDE (render/physics), amplified by
  the sensitive policy. → This killed the retry design (dropped) and the paired-McNemar plan.
- **Analysis plan (pre-registered, fixed):** unpaired two-proportion tests, binomial CIs,
  ~9 pp resolution floor at n=300. A NULL on the transform ablation is expected/acceptable.
  Primary wandering metric = `ended_at_step` (deterministic), NOT wall-clock. See
  `receipts/PRE_REGISTRATION.md` — READ IT before interpreting any arm.
- **wall_timeout recording:** Isaac overrides the eval's SIGTERM handler, so the RUNNER writes
  a guaranteed stub JSON (term_reason=wall_timeout, distances=-1.0 sentinel) if the episode
  produced no record. Aggregator excludes the -1.0 sentinel from NE/ONE, counts as failure.
- **Camera flags added to navila_eval_v3.py:** `--cam_z/--cam_height/--cam_width/--cam_aperture`
  (offset-only, isolation-verified: only z moves, FOV/res fixed). Ground height = 0.53 + cam_z.
- Robot deploy policy (separate from the benchmark): model_14200 (round-5 lateral-band gait)
  is installed in booster_deploy but the user rejected earlier gaits; the FIRST real-robot walk
  happened 07-23 (vibrated → traced to a mode-switch/dual-controller issue, fixed a deploy-side
  low-pass filter + verified ChangeMode). This is DONE and robot-gated; not part of the sweep.

## INFRASTRUCTURE (all set up, all survive reboot)
- **Remote:** `ssh boosterk1@100.116.68.68` (Tailscale; key valid to Jan 2027). sshd+tailscaled
  enabled. Box is on WiFi (SSID "Robotz", autoconnect on, powersave disabled).
- **Never sleeps:** sleep/suspend/hibernate targets MASKED; logind idle/lid ignored; GNOME
  no-sleep/no-blank.
- **Auto-upgrades DISABLED** (apt-daily timers off; 200 nvidia/kernel packages held) so the
  driver can't break again mid-run. **RELEASE WHEN USER IS BACK** (in AWAY_STATUS.md):
  `sudo apt-mark unhold $(apt-mark showhold); sudo systemctl enable --now apt-daily.timer apt-daily-upgrade.timer`
- **Blind CUDA-recovery steps** (if it breaks again): AWAY_STATUS.md has the gcc-12 DKMS rebuild.

## REPOS (all pushed; user = Janga786, no co-author trailer)
- `github.com/Janga786/k1-navila-research` (~/Projects/k1_research) — the sweep, receipts,
  EVIDENCE.md, PRE_REGISTRATION.md, SLIDE_FACTS.md, AWAY_STATUS.md, HANDOFF.md.
- `github.com/Janga786/NaVILA-Complete-Archive` (~/Desktop/NAVILA_COMPLETE_ARCHIVE_2026-07-13)
  — master report + 1077 result JSONs + paper assets (outline, reproducibility, references,
  figures, VLM input frames, training scalars).
- `Janga786/{booster_train, booster_deploy, k1-vlm-navigation}` — training, deploy, robot stack.
- eval_results is GITIGNORED in k1_research → per-arm JSONs are copied to
  `receipts/sweep_measurements/<tag>/` and pushed from there.

## WHAT THE USER WANTS AT THE END (Aug 23)
The paper needs: per-arm SR/OS with binomial CIs (unpaired), the stretchB-vs-stretchA
same-driver noise floor, the height-sweep trend (SR/OS vs camera height, virtual-viewpoint
labeled), all with the driver-change confound stated. Everything feeds a sim-to-real VLN +
embodiment-effects paper. Paper-asset scaffolding is in the archive's `11_paper_assets/`.

## FILES TO READ FIRST (in this order)
1. `~/Projects/k1_research/SWEEP_STATUS.md` — live progress
2. `~/Projects/k1_research/receipts/PRE_REGISTRATION.md` — the fixed analysis plan + all findings
3. `~/Projects/k1_research/AWAY_STATUS.md` — infra + recovery
4. `~/Projects/k1_research/arms_runner.sh` — the runner
5. This file.
