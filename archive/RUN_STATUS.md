# RUN_STATUS — powered crop-vs-stretch A/B (100 episodes/arm) + remote-access check
Launched 2026-07-22 09:02 MDT on the lab **RTX 3090** (renderer-valid; NEVER move these
runs to a Blackwell GPU). Checkpoint: `model_14498` (matches the June 10-episode A/B).

## Exact launch (already running, detached via setsid — survives logout; tmux not installed)
```bash
cd ~ && setsid bash Projects/k1_research/run_ab100.sh > /tmp/ab100_chain.log 2>&1 &
```
`run_ab100.sh` chains the two arms sequentially (one GPU):
1. `TRANSFORM=stretch bash scripts/run_powered_benchmark.sh stretch100 "" 0 100`
2. `TRANSFORM=crop    bash scripts/run_powered_benchmark.sh crop100    "" 0 100`
with `VLM_BRIDGE_EXTRA=--load_8bit EP_TIMEOUT=900`. The int8 VLM bridge auto-launched
and stays up across both arms (log: `/tmp/vlm_server_vlnce_v3.log`).

## Logs & results
- Chain log: `/tmp/ab100_chain.log` · per-arm logs: `/tmp/powered_stretch100.log`, `/tmp/powered_crop100.log`
- Results: `NaVILA-Bench/eval_results/k1_matterport_vision_loco_{stretch100,crop100}/measurements/`
  (one JSON per scored episode; the harness prints a paper-table aggregate at each arm's end)

## Check progress (either)
```bash
ls ~/Projects/k1_research/NaVILA-Bench/eval_results/k1_matterport_vision_loco_stretch100/measurements | wc -l   # of 100
tail -5 /tmp/powered_stretch100.log    # shows "[powered] progress: ... ETA" every 10 eps
```

## Resume after any crash/reboot (safe to run anytime — skips scored episodes)
```bash
cd ~ && setsid bash Projects/k1_research/run_ab100.sh > /tmp/ab100_chain.log 2>&1 &
```

## Timing
- At the instructed planning rate **9–12 min/episode**: 15–20 h/arm → arm 1 finishes
  late Wed night–Thu morning; arm 2 finishes **Thu afternoon–Fri ~01:00** (tight at the
  12-min bound). NOTE: the recorded full-1077 average was ~3 min/episode; at that rate
  both arms finish **Wed evening**. Resume-safety covers any overrun; check Thu morning.
- Aggregate when done: the harness prints it, or
  `python aggregate_k1_vision_results.py --measurements-dir <arm>/measurements --total-episodes 100 --paper-table`

## ⚠️ STEP 4 — REMOTE ACCESS: **NOT REACHABLE OFF-NETWORK** (loud flag)
- Tailscale: **not installed**. sshd: **inactive** (both `ssh`/`sshd` units). Only LAN
  IP `10.218.0.139` on host `booster`.
- **The August remote plan does not work against this machine as it stands.** Options
  (require sudo + account decisions, deliberately NOT done today): install Tailscale and
  enable sshd (`sudo apt install tailscale openssh-server && sudo tailscale up &&
  sudo systemctl enable --now ssh`), or change the August plan to on-site/other host.

## Straggler protocol (observed: ep 0 hit the 900 s guard on arm 1 — known slow scene)
Timeouts leave no JSON and are retried by simply rerunning the launcher. After both arms
complete, sweep stragglers exactly like the full run did (900→1800 s):
`EP_TIMEOUT=1800 bash Projects/k1_research/run_ab100.sh` (resume-safe; only unscored episodes run).
