# Frame-replay test: do rendering differences cause reruns to diverge? — RESULTS

Run 2026-10-02 on the sweep workstation (RTX 3090, driver 580.173.02), following `PLAN.md` (pre-registered,
private archive commit `5ede302ecd1fdea980821468f92444d91f6485d5`, unchanged). Departures from the plan are
listed under "Deviations". All numbers below come from files in this folder (`tables/`, `replay/`, `runs/`, `logs/`).

## Answer

**All three predictions held in all eight episodes** — the first row of the plan's outcome table:
*only the rendered images differed; physics and inference repeated exactly when given identical inputs.*

- **H1 held in 8/8.** In every episode A and B first diverged in a model reply (q*). The robot state was
  bit-identical at every control step before the step of q* (reset, 200 warm-up steps and every main-loop step)
  and at the query itself. The two requests at q* differed.
- **H2 held in 8/8 under each condition** (simulator loaded, server alone, restarted server). Each run's q*
  request got that run's own reply in 20 of 20 sends: 16 requests × 3 conditions × 20 = 960 sends, 960
  returned the run's own reply.
- **H3 held in 8/8.** Run R, whose model history was run A's saved frames, reproduced run A exactly: byte-identical
  requests at every query (238 queries in total), identical replies, a bit-identical state after the reset and at
  every step (22,519 control steps in total, warm-up included), and a final record equal in all six fields. No `replay_mismatch.json` was written.
- Rendering itself does not repeat. In R the robot passed through exactly A's states, yet none of R's 912 fresh
  renders equals A's frame at the same point: the median frame differs in 12.6 % of colour channels (range
  4.6–33.5 %), the median maximum difference is 7 levels (largest 63).

Mechanism, as measured: the renderer gives each process slightly different images; every request therefore
differs from the first query on (8/8 pairs, first request difference at query 0). The replies nevertheless
agreed on all 118 queries before q* across the eight pairs, and then one reply flipped on a request that
differed only by render noise. The flip is a deterministic function of the request (H2), physics is
deterministic given the command sequence (H1, H3), so the closed loop carries the flip into a different
trajectory.

## 1. Episodes

Run lengths in steps, end reason, success. Sweep lengths (D1/D2) for reference. q* = query index (step).
"Frames differing at q*": all 8 frames sent at q* differ between A and B in every episode; max / mean absolute
difference is over the 8 frames (levels of 255).

| `episode_idx` (record) | sweep D1/D2 | run A | run B | run R | q* (step) | first state difference | first request difference | frames differing at q* (max / mean abs. diff) | reply A at q* | reply B at q* |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 (6) | 903 / 1493 | 1022, stop, 0 | 903, stop, 0 | 1022, stop, 0 | 3 (109) | none before q* (step 109) | query 0 | 8 of 8 (11 / 0.17) | move forward 25 cm | turn right 15 degree |
| 8 (11) | 1573 / 1573 | 1573, sim_done, 0 | 1776, stop, 1 | 1573, sim_done, 0 | 6 (530) | none before q* (step 530) | query 0 | 8 of 8 (19 / 0.15) | turn right 15 degree | move forward 25 cm |
| 16 (25) | 2640 / 4068 | 6001, step_cap, 0 | 2232, sim_done, 0 | 6001, step_cap, 0 | 20 (1807) | none before q* (step 1807) | query 0 | 8 of 8 (15 / 0.10) | turn right 45 degree | move forward 25 cm |
| 20 (32) | 1563 / 6001 | 1563, sim_done, 0 | 1437, stop, 0 | 1563, sim_done, 0 | 4 (277) | none before q* (step 321) | query 0 | 8 of 8 (12 / 0.16) | turn left 30 degree | turn left 45 degree |
| 50 (77) | 6001 / 6001 | 6001, step_cap, 0 | 6001, step_cap, 0 | 6001, step_cap, 0 | 51 (5482) | none before q* (step 5482) | query 0 | 8 of 8 (65 / 0.25) | turn right 15 degree | turn left 45 degree |
| 99 (144) | 2709 / 2772 | 2677, stop, 0 | 1947, stop, 1 | 2677, stop, 0 | 11 (724) | none before q* (step 724) | query 0 | 8 of 8 (39 / 0.12) | turn right 15 degree | move forward 25 cm |
| 92 (137) | 925 / 926 | 1025, stop, 1 | 926, stop, 1 | 1025, stop, 1 | 10 (665) | none before q* (step 692) | query 0 | 8 of 8 (31 / 0.12) | turn right 15 degree | turn right 30 degree |
| 212 (323) | 1049 / 907 | 1049, stop, 1 | 994, stop, 1 | 1049, stop, 1 | 13 (954) | none before q* (step 986) | query 0 | 8 of 8 (24 / 0.15) | move forward 75 cm | move forward 25 cm |

Replies are quoted without the prefix "The next action is"; full strings in `tables/episodes.csv`.
B differed from A in every episode, so no C or D run was needed (extra-run rule not triggered).
In episodes 20, 92 and 212 the two q* replies give the same velocity command and differ only in duration
(30° vs 45°, 15° vs 30°, 75 cm vs 25 cm), so the states stay identical until the shorter command ends; the first
state difference is therefore 27–44 steps after the step of q*. In the other five it is at the step of q*.
Per-frame detail: `tables/qstar_frames.csv` (all 64 frames differ; 8.2–25.4 % of channels; max abs. diff 3–65).
The 8 frames each run sent at q* and their difference images are in `qstar_frames/ep<idx>/`.

## 2. Hypotheses per episode

| `episode_idx` | H1 | H2 cond. 1 (simulator loaded) | H2 cond. 2 (server alone) | H2 cond. 3 (restarted server) | H3 |
|---|---|---|---|---|---|
| 3 | held | held (20/20, 20/20) | held | held | held |
| 8 | held | held | held | held | held |
| 16 | held | held | held | held | held |
| 20 | held | held | held | held | held |
| 50 | held | held | held | held | held |
| 99 | held | held | held | held | held |
| 92 | held | held | held | held | held |
| 212 | held | held | held | held | held |
| **total** | **8 of 8** | **8 of 8** | **8 of 8** | **8 of 8** | **8 of 8** |

H1 detail (all eight): (a) states before the step of q* identical, (b) state at q* identical, (c) requests at q*
differ — all true. H2: every one of the 48 request×condition cells returned a single reply, the run's own
(`tables/h2_replies.csv`). The q* requests were rebuilt from the saved frames and matched the logged request
SHA-256 in all 16 cases before being replayed. H3: `replay/h3_ep<idx>.json`.

## 3. Requests vs replies before q*

| `episode_idx` | queries before q* | requests differed, replies matched | first request difference |
|---|---|---|---|
| 3 | 3 | 3 | query 0 |
| 8 | 6 | 6 | query 0 |
| 16 | 20 | 20 | query 0 |
| 20 | 4 | 4 | query 0 |
| 50 | 51 | 51 | query 0 |
| 99 | 11 | 11 | query 0 |
| 92 | 10 | 10 | query 0 |
| 212 | 13 | 13 | query 0 |
| total | 118 | 118 | |

Every request differed between A and B (the frames differ from the first render on), but the model gave the same
reply to 118 differing request pairs before the first flip. Per-query timeline: `tables/query_timeline.csv`,
`figures/timeline_request_reply.png`.

## 4. Run R: fresh renders vs run A's frames

R's fresh renders were taken at states bit-identical to A's (H3), so this measures render repeatability across
processes at identical states (`tables/runR_fresh_vs_A.csv`, summary `tables/runR_fresh_vs_A_summary.csv`).

| `episode_idx` | frames | bit-identical | max abs. diff (max / median) | channels differing (median, min–max) |
|---|---|---|---|---|
| 3 | 50 | 0 | 23 / 8 | 17.2 % (14.5–33.5) |
| 8 | 72 | 0 | 29 / 8 | 15.3 % (11.1–26.8) |
| 16 | 250 | 0 | 26 / 7 | 10.9 % (5.7–18.4) |
| 20 | 72 | 0 | 23 / 5.5 | 10.4 % (4.6–19.3) |
| 50 | 250 | 0 | 63 / 9 | 13.8 % (5.2–27.5) |
| 99 | 117 | 0 | 42 / 7 | 11.3 % (8.5–14.4) |
| 92 | 50 | 0 | 29 / 7 | 11.6 % (9.5–14.3) |
| 212 | 51 | 0 | 20 / 8 | 14.3 % (12.8–20.2) |
| all | 912 | 0 | 63 / 7 | 12.6 % (4.6–33.5) |

Most differing values are ±1 level (episode 3 q* frame: 35.4 % of pixels differ by 1, 0.27 % by 2 or more;
`figures/qstar_diff_heatmap_ep3.png`).

## Figures

- `figures/qstar_diff_heatmap_ep3.png` — episode 3 (first diverging episode in table order), current-observation
  frame sent at q*: run A, run B, and |A − B| (max over channels), binned.
- `figures/timeline_request_reply.png` — per pair and query: request hash identical/different, reply
  identical/different, q* marked. The same data as a table: `tables/query_timeline.csv`.

## Exploratory (not pre-registered)

- 6 of the 16 A/B reruns reproduce a July sweep record exactly (six fields and length): ep3 B = D1, ep8 A = D1 =
  D2, ep20 A = D1, ep50 B = D2, ep92 B = D2, ep212 A = D1 (`tables/exploratory_rerun_vs_sweep.json`). This fits
  the measured picture of a small set of deterministic branches selected by which way a knife-edge reply flips.
- The validation run (episode 92, not used above) gave a third trajectory for that episode (913 steps, success).

## Not determined

- Why the renderer differs between processes at identical states (RTX renderer internals; not probed here).
- Which pixels or features of the q* frames flip the reply; how close other queries are to a flip.
- Inference determinism beyond these 16 requests (60 sends each) and physics determinism beyond these
  trajectories (up to 6001 steps, 120 s) on this GPU and driver; other hardware not tested.
- Whether the sweep's own reruns diverged by exactly this mechanism at the same queries: the sweep did not log
  queries; the exploratory matches above are consistent with it but do not show it.

## Deviations from PLAN.md

1. **Attempt 1 stopped at Step 1** (2026-10-01 23:26) by the plan's stop rule: the logging copy's in-process PNG
   reload failed inside Isaac Sim, where `PIL.Image` (Pillow 12.3.0, `~/.local`) is mixed with Kit's prebundled
   Pillow 10.2.0 plugins. Report: `STOP_REPORT.md`. GPU used: 20 s simulator + 216 s server. The evidence is kept
   (`runs/validate_attempt1/`, `logs/*attempt1*`). The walking-check experiment used the GPU in between (23:40–00:10).
2. **Logging copy patched** with your approval (2026-10-02): `PROPOSED_FIX_png_decode.diff` — the two in-process
   PNG decodes use `cv2.imread` instead of PIL; nothing else changed and PLAN §5's descriptions still hold. PNG
   *encoding* under the mix was shown byte-identical to plain Pillow (and Step 1 check 3 confirmed it at all 16
   queries). New SHA-256: copy `cd9f13c034a07606d5cb75c90f4e76174cac28ae9f803f39651a67eb4965e124` (pinned
   `37c0b3eb…`), `evaluator.diff` `344bf26a6a57f0fba520bc9ac0892045811c59ad0c92722e27c67e23cf9f0ce1` (pinned `028f49f4…`).
3. **Offline suite extended** (simulated Kit Pillow mix; A/B fixture pinned to render seeds 11/12 after one unpinned
   run did not diverge, 19/23): final 24/24 (`logs/offline_tests.txt`); new SHA-256
   `981adf873a79d9ca40c3d273f2695d1b956847509c50177cf36b725a9fc2b4a7` (pinned `d16e093f…`).
4. **Budget thresholds** for attempt 2 were reduced by attempt 1's 236 s (8 h 45 min − 236 s, 9 h − 236 s). The guard
   never triggered; all GPU work ended 3 h 49 min after server session 1 started.
5. **Server session 2 idled for 9 h 23 min** (13:51:32–23:14:42). Per the plan it stays up until the results are pushed, but the
   session supervising the batch went idle after Step 2 began (10:16), so the analysis and upload started only at
   23:06. The server held 13.4 GB of GPU memory, idle, from 13:51 until it was stopped after the push. The total
   span from server start to server stop therefore exceeds the 9-hour cap, while the GPU work itself took 3 h 49 min.
   No other job was blocked beyond what GPU_LOCK would have blocked anyway.
6. `scripts/analyze.py` was written before any run data existed (PLAN §12 said after the runs); it only formats
   outputs of the pinned `compare_runs.py`. `scripts/stage_upload.py` (upload selection) was added for Step 6.
7. Attempt 2 reused the planned labels after moving attempt 1's files aside, and started a new server session 1.

## GPU time (`logs/gpu_time.tsv`, `logs/client_sessions.tsv`)

| Item | Wall time |
|---|---|
| Attempt 1: validation run + server session | 20 s + 216 s |
| Attempt 2 simulator runs (33, all rc 0): validate 203 s, A 3481 s, B 2884 s, R 3519 s, condition 1 1295 s | 11 382 s (3.16 h) |
| Condition 2 and 3 client sessions (16, all rc 0; requests run on the server) | 2161 s |
| Server session 1 (attempt 2), 10:02:25–13:33:15 | 12 650 s (3.51 h) |
| Server session 2, 13:33:15–23:14:42 (condition 3 until 13:51:32, then idle until stopped after the push) | 34 887 s (9.69 h; 18 min used) |
| Total span of attempt 2, server start to server stop | 13 h 12 min (10:02:25–23:14:42) |
| GPU work, attempt 2: server start to last replay | 3 h 49 min |

## Files

`PLAN.md` (pre-registered), `STOP_REPORT.md` + `PROPOSED_FIX_png_decode.diff` (attempt 1), `scripts/` (logging copy,
`evaluator.diff`, run/batch scripts, `compare_runs.py` = verdict logic, analysis, offline mock suite), `logs/`,
`runs/<label>/` (per run: `queries.jsonl`, `frames.jsonl`, `state_hashes.txt`, `state_components.tsv`,
`state_layout.json`, `states_at_queries.npz`, `run_meta.json`, `measurements/*.json`), `replay/` (comparisons,
q* request specs, every replay reply, H2/H3 verdict files), `tables/`, `figures/`, `qstar_frames/`.
All other frames, the fresh renders, the videos and the rebuilt request bytes stay on the workstation, listed in
`MANIFEST.csv` (path, bytes, SHA-256).
