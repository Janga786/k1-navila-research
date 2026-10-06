# Frame-replay test: do rendering differences cause reruns to diverge? — PLAN

Pre-registered on 2026-10-01, before any GPU run of this experiment. This file is committed to the private
repo `Janga786/NaVILA-Complete-Archive` as `15_airc2027_frame_replay/PLAN.md` and is not changed afterwards;
any departure from it is reported in a "Deviations" section of `RESULTS.md`.

Workstation: the HP Z8 G4 that ran the NaVILA–K1 sweep (RTX 3090 24 GB, driver 580.173.02 = the sweep's
driver). Work directory: `~/Projects/k1_research/airc2027_replay/`. For the AIRC 2027 paper.

## 1. Background

- Reruns of the same configuration diverge. Only 52 of 300 episode records are identical between D1
  (`stretchA_300`) and D2 (`stretchB_300`) (re-counted today from `NaVILA-Bench-main/eval_results/`: 52 records
  with identical six fields = identical full JSON = byte-identical files).
- In the pre-sweep probe logs (6 episodes, 2 pairs of runs; archive `12_airc2027_materials/probes/`), each of
  the 11 diverging pairs first differed in a model reply, at a query whose logged position, heading and step
  count were identical (as stated in the request; not re-derived here).
- The 2026-10-01 diagnosis (archive `14_airc2027_renders/DIAGNOSIS.md`) found: physics bit-identical across
  fresh processes over at most 1.5 s of walking; images differing by one brightness level in about 10–19 % of
  colour channels; one model input answered identically 60 of 60 times with the model server alone on the GPU.
- Working hypothesis: rendering differences change the images the model sees, which occasionally flips a
  reply, and the closed loop amplifies the flip. This experiment tests that directly; it does not assume it.

## 2. Pre-registered predictions

If rendering differences alone cause the divergence:

- **H1.** In A and B the robot state is bit-identical (a) at every control step before the step of q* and
  (b) at the moment of the query q*; and (c) the requests sent at q* differ.
- **H2.** Each run's request at q* gets that run's own reply in 20 of 20 sends, under every replay condition.
- **H3.** Compared with run A, run R has byte-identical requests at every query; identical replies at every
  query; a bit-identical state at every step; and a final record equal in all six fields: `path_length`,
  `distance_to_goal`, `oracle_navigation_error`, `success`, `spl`, `oracle_success`.

| Outcome | Meaning |
|---|---|
| H1, H2 and H3 all hold | Only the rendered images differed. Physics and inference repeated exactly when given identical inputs. |
| H1 fails because the state differs before q* | A physical or other state difference came before the reply changed. |
| H1 fails because the requests at q* are identical | The replies differed on identical input: inference nondeterminism. |
| H2 fails | Inference is nondeterministic for that input; the distribution of replies is reported. |
| H3 fails | Something besides rendering varies; the first difference is reported. |

Every outcome is reported, including ones that contradict the hypothesis.

## 3. Operational definitions (implemented in `scripts/compare_runs.py`, pinned below by SHA-256)

- **Query**: one call of the evaluator's `sample_images_and_send_to_vlm`. Queries are indexed 0, 1, 2, … in
  order within a run.
- **Request**: the exact bytes sent to the model server for a query (`json.dumps({'images': [8 base64 PNGs],
  'query': instruction}).encode()`). Two requests differ iff their SHA-256 differ.
- **Reply**: the model's decoded reply text (`stream_output`). Two replies differ iff the strings differ.
- **State** after a control step: SHA-256 over the float32 bytes, in this order, of the robot's root state
  (`root_state_w`), joint positions (`joint_pos`), joint velocities (`joint_vel`), all body states
  (`body_state_w`), joint position targets (`joint_pos_target`) and the low-level policy observation (the
  wrapper's 5×47 history buffer `_history_buf`, whose flatten is the 235-dim policy input). Shapes and dtypes
  are recorded per run in `state_layout.json`. Per-part hashes are kept too.
  Labels: `reset` (after `env.reset()`), `w0`…`w199` (the evaluator's 200 warm-up steps), `0`…`N` (main loop,
  label = `num_steps`). "Control step" = every `env.step` call of the evaluator.
- **State at a query**: the same hash computed at the query, before the request is built (it equals the state
  after the previous control step; this equality is checked).
- **q\*** for a pair (A, X): the first query index present in both runs whose replies differ.
  **Step of q\***: `num_steps` of that query (the main-loop step whose `env.step` follows it).
- **Diverging pair**: a pair with a q*. If the replies agree at every common query index there is no q* (even
  if lengths or states differ; any such difference is reported with the first state-hash difference).
- **H1 verdict** for a pair: *held* if (a) every state label before the step of q* is identical in A and X
  (same labels, same hashes; the steps of q* must be equal), (b) the state hashes at q* are equal and (c) the
  request hashes at q* differ; *failed — state differs before q\** if (a) or (b) fails;
  *failed — requests at q\* identical* if (a) and (b) hold and (c) fails.
- **H2 verdict** for an episode under one condition: *held* iff A's q* request got A's logged q* reply in 20 of 20
  sends and X's q* request got X's logged q* reply in 20 of 20 sends (each request's SHA-256 checked per send).
- **H3 verdict** for an episode: *held* iff R and A have the same number of queries with pairwise-equal request
  hashes and replies, identical ordered state logs (same labels and hashes, same length), equal six record
  fields, and run R wrote no `replay_mismatch.json`.
- Totals are reported as "held in n of m", m = number of episodes for which that test ran.

## 4. Episodes

`episode_idx` as passed to the evaluator; record = `episode_id − 1`, checked today against
`vln_ce_isaac_v1.json.gz` (all eight agree). Lengths are `ended_at_step` of the sweep records (all agree with
the request).

| `episode_idx` | record | Why | D1 / D2 length (steps) |
|---|---|---|---|
| 3 | 6 | probe episode | 903 / 1493 |
| 8 | 11 | probe episode (identical in D1/D2) | 1573 / 1573 |
| 16 | 25 | probe episode | 2640 / 4068 |
| 20 | 32 | probe episode | 1563 / 6001 |
| 50 | 77 | probe episode | 6001 / 6001 |
| 99 | 144 | probe episode | 2709 / 2772 |
| 92 | 137 | shortest diverging D1–D2 pair outside the probes, neither run interrupted | 925 / 926 |
| 212 | 323 | second shortest | 1049 / 907 |

## 5. Instrumentation: the logging copy

`NaVILA-Bench-main/scripts/navila_eval_v3.py` was verified to have SHA-256
`9b010715ced0a46efb17b86e918653e933bdc26b5fd765f3d8a0912d7b1281fc` and copied to
`scripts/navila_eval_v3_LOG.py`. Every change is marked `[LOG]` and listed in `scripts/evaluator.diff`.
Sweep files are read-only: nothing in the sweep tree is edited and nothing is written to `eval_results/`
(the copy refuses a `--log_dir` under `eval_results/`).

- `--log_dir DIR` (mandatory in the copy): the record JSON, the video, the SIGTERM record and all logs go
  under DIR.
- Frame log: every frame appended to `image_observations` (initial frame, 8 warm-up frames, every main-loop
  frame) is saved as lossless `frames/f{k:04d}.png` (k = append order); `frames.jsonl` holds k, the label
  (`init`, `w<i>` or `num_steps`) and the SHA-256 of the raw RGB bytes. The first 10 PNGs are reloaded in-process
  and compared with the stored array (the run aborts on a mismatch).
- Query log `queries.jsonl`: query index, `num_steps`, the 8 frame indices sent, the SHA-256 of each of the 8 PNG
  byte strings, the SHA-256 and length of the whole request, the instruction, the raw reply, the parsed command
  and `time_to_go`, `distance_to_goal`, the state hash at the query, the request's wall time.
- State log: after every `env.step` and after `env.reset`, `state_hashes.txt` (`label hash`) and
  `state_components.tsv` (per-part hashes); raw arrays at each query in `states_at_queries.npz` (plus the
  wrapper's last action, gait phase and command, not hashed). The query-time snapshot is taken before the
  evaluator's `torch.inference_mode()` block.
- `--replay_frames_from DIR` (run R): at each append k, `DIR/frames/f{k:04d}.png` is loaded into a uint8 array,
  wrapped as the evaluator wraps fresh frames (`Image.fromarray(arr)`) and stored in the model's history instead
  of the fresh render (its SHA-256 is checked against A's log). The fresh frame is still rendered and saved as
  `fresh_frames/f{k:04d}.png`. While running, R's request hash at every query (checked before sending), reply at
  every query and state hash at every step are compared with A's logs; at the first mismatch the copy writes
  `replay_mismatch.json` (what differed, where, both values) and ends the episode with a partial record
  (`term_reason = replay_mismatch`).
- `--replay_queries FILE` (Step 4, condition 1): builds the environment and resets the episode, sends each request
  in FILE `--replay_n` times (20) through the evaluator's own socket code (moved verbatim into `_vlm_send_bytes`),
  logs every reply, exits without running the episode.
- Outside replay mode nothing changes the simulation, the random state, the control flow or the bytes sent to the
  model: the hooks only copy tensors to the host, hash them and write files. (Wall-clock timing changes; the
  evaluator has no time-dependent logic besides its 120-s socket timeout.)
- Offline verification before this plan was committed (no Isaac, no GPU, no real model; fake `omni.*` modules
  with deterministic fake physics and noisy fake renders, the real policy checkpoint and the real `eval_utils`;
  `scripts/mock/run_offline_tests.sh`, output `logs/offline_tests.txt`): normal runs, validation checks, R → H3
  held; an injected 1-µm physics jitter is caught as a state mismatch at the injected step; a nondeterministic
  fake server is caught as a reply mismatch and as an H2 failure; all three H1 verdicts are produced when their
  conditions are constructed; rebuilt q* requests match the logged hashes; conditions 1 and 2 run; the SIGTERM
  path writes the record and logs under `--log_dir`; the copy refuses `eval_results/` and existing run folders.

## 6. Commands

Model server (env `navila`, cwd NaVILA-Bench), as `run_powered_benchmark.sh` starts it with
`VLM_BRIDGE_EXTRA=--load_8bit` (`arms_runner.sh`), via `scripts/server.sh`:

    python scripts/vlm_server_bridge.py \
      --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f --port 54321 --load_8bit

Evaluator runs (env `vlnce-isaac`, cwd NaVILA-Bench, `OMNI_KIT_ACCEPT_EULA=yes`), via `scripts/run_sim.sh`:

    python <airc2027_replay>/scripts/navila_eval_v3_LOG.py --task=k1_matterport_vision --num_envs=1 \
      --checkpoint=$HOME/Projects/k1_research/checkpoints/model_14498.pt --episode_idx=<i> --gait_phase_init=0.0 \
      --out_tag=fr_<label> --closed_loop --max_episode_s 120 --vlm_transform stretch --headless --enable_cameras \
      --log_dir <airc2027_replay>/runs/<label> [--replay_frames_from … | --replay_queries … --replay_n 20]

These are the sweep's commands, confirmed against `run_powered_benchmark.sh` and `arms_runner.sh` (D1/D2:
`TRANSFORM=stretch`, no extra flags, model_14498, int8 server), changed only in: the script path, `--out_tag`,
the output folder (`--log_dir`) and the new flags. Pre-registered procedural differences from the sweep (not
deviations): the runner's 900-s wall-clock kill is replaced by a 3600-s safety timeout (`timeout -k 30 3600`);
`PYTHONDONTWRITEBYTECODE=1` for the simulator and server processes (no `.pyc` writes into the sweep tree; no
functional effect); the server is started with `exec` so its PID is known, and logs to `logs/`; a fresh server
session rather than the sweep's weeks-old one. The PNG encoder is Pillow 12.3.0 from `~/.local` (installed
2026-07-01, before the sweep, so the same as the sweep's); each run records its library versions in
`run_meta.json`, and request rebuilding uses the same interpreter.

## 7. Run order

GPU rule for every step: `GPU_LOCK` (containing `frame_replay`) must exist; before every simulator run
`nvidia-smi` must show no compute process other than the model server started here; nothing else runs on the
GPU in parallel; every GPU process is logged to `logs/gpu_time.tsv` (label, start, end, wall seconds, return
code, command). Long batches run detached (`setsid nohup`), log to `logs/batch.log`, stop at the first failure.

1. **Step 1 — validation** (`scripts/batch_validate.sh`): start server session 1 (kept for Steps 2, 3 and
   Step 4 conditions 1–2); one normal run of episode_idx 92 (label `validate`, not used in the analysis); checks
   (`scripts/validate_run.py`): every output lands under `airc2027_replay/` (no file elsewhere in
   `~/Projects/k1_research` newer than the run's start marker, except the walking-check session's own
   `airc2027_walking/` folder); every saved PNG reloads to exactly the stored array (all frames); rebuilding every
   query's request from the saved frames reproduces the logged request hash and PNG hashes; the record has the
   sweep's schema (same keys and value types as `stretchA_300` record 137); log consistency. Any failure stops
   the experiment.
2. **Step 2 — runs A and B** (`scripts/batch_main.sh`): for each episode in table order (3, 8, 16, 20, 50, 99,
   92, 212): run A, then run B, each in a fresh process (labels `ep{idx}_A`, `ep{idx}_B`); compare (q* with index
   and step, first state-hash difference, first request-hash difference, six record fields).
3. **Extra-run rule**: if no reply differs between A and B, run C and, if needed, D (normal runs, fresh
   processes) until one differs from A; use the first that differs. If none of B–D differs, record "no divergence
   in 4 runs" for that episode.
4. **Step 3 — run R**: for each episode with a diverging pair, in table order, run R with
   `--replay_frames_from` = run A's folder; H3 check; R's fresh renders compared with A's frames.
5. **Step 4 — model replays at q\***: for each diverging pair, rebuild A's and the other run's q* requests from
   the saved frames and check them against the logged hashes (any mismatch stops the experiment). Then
   condition 1 for every episode (table order), condition 2 for every episode, restart, condition 3 for every
   episode (see 8).
6. **Step 5 — analysis** (CPU) and **Step 6 — upload**; then stop the model server (session 2 stays up, idle,
   until the results are pushed), confirm with `nvidia-smi` that no compute process is left, delete `GPU_LOCK`.

## 8. Replay conditions (Step 4)

In every condition each episode's two q* requests are sent 20 times each, A's request 20 times in a row, then
the other run's 20 times, every reply logged:

1. **Simulator loaded**: the copy with `--replay_queries` on that episode (environment built, episode reset,
   simulator resident on the GPU as in the sweep; no episode run), server session 1.
2. **Server alone**: no simulator process; `scripts/replay_client.py` (standard library only; its socket function
   is identical to the evaluator's), server session 1.
3. **Restarted server**: server session 1 stopped, the server started once more with the same command
   (session 2), then the same as condition 2.

## 9. Stop rules

The batch stops (and the experiment is reported as stopped, without improvising) on: a run with return code
≠ 0 (including the 3600-s timeout), a missing record, an `error.txt`, a record ending in `vlm_error` or
`wall_timeout`, a dead model server, an unexpected GPU compute process, less than 15 GB free disk, a failed
Step 1 check, or a rebuilt q* request that does not match its logged hash. A replay mismatch in run R is a
result (H3 failed), not a failure. `GPU_LOCK` is deleted at the very end, also after an early stop.

## 10. Budget

Expected 5–7 GPU-hours; measured cost model ≈ 85 s + 0.134 s/step per simulator run (sweep timing + logging
overhead), i.e. ≈ 3.5–4 h of simulator time plus server time. Cap: 9 hours, clock starting when server session 1
starts. Before each GPU unit (a Step-2 episode's A+B; each R run; each condition-1 run; the condition-2 block;
the restart + condition-3 block) the batch estimates its time (90 s + 0.14 s × the longer sweep length per
simulator run) and starts it only if it would end within 8 h 45 min; C/D runs of an episode already started are
run if they end within 9 h. Otherwise that unit and everything after it on the GPU is skipped and reported.
Disk: 107 GB free at the start (≥ 30 GB needed), so run R keeps all of its fresh renders.

## 11. What will be reported

`RESULTS.md`, `tables/*.csv`, two figures:
1. One row per episode: lengths and outcomes (success, end reason) of each run; q* (query index, step); first
   state-hash difference (a step, or "none before q*"); first request-hash difference; which of the 8 frames
   differ at q* and by how much (max / mean absolute difference, share of channels); the two replies at q*.
2. H1, H2 (each condition) and H3 per episode — held, failed (with the kind of failure and where) or not tested —
   with totals "held in n of m".
3. Requests vs replies: per pair, the number of queries before q* whose requests differed while the replies matched.
4. Run R: its fresh renders against A's frames (per frame max / mean absolute difference, share of channels).
5. Deviations from this plan, GPU time (sum of simulator runs, server sessions, total span), anything not
   determined.
Figures: (a) heatmap of |A − X| at q* (max over channels) for the current-observation frame (8th frame) of the
first diverging episode in table order; (b) per pair, a timeline of request-hash and reply agreement by query.
Upload (private repo, `15_airc2027_frame_replay/`): this file unchanged, `RESULTS.md`, `scripts/` (with the copy
and `evaluator.diff`), `logs/`, per-run `queries.jsonl`, `frames.jsonl`, `state_hashes.txt` (+ per-part hashes),
`states_at_queries.npz`, record JSON, `replay_mismatch.json` if any, `tables/`, `figures/`, and for each
diverging episode the 8 frames each run sent at q* plus their difference images. All other frames stay on the
workstation, listed in `MANIFEST.csv` (path, size, SHA-256). Never uploaded: Matterport scenes, the VLN-CE-Isaac
dataset file, model weights or checkpoints, robot meshes, conda environments, Claude Code session transcripts.
Every file < 50 MB; staged files are scanned for secrets.

## 12. Code pinned at pre-registration (SHA-256)

Verdict logic lives in `compare_runs.py`; the Step 5 table/figure script is written after the runs and only
formats the outputs of these scripts. Sweep evaluator: `9b010715ced0a46efb17b86e918653e933bdc26b5fd765f3d8a0912d7b1281fc`.

| File | SHA-256 |
|---|---|
| `scripts/navila_eval_v3_LOG.py` | `37c0b3ebe7d0d3d28decb41cbfa922b3c37b3cd5206ba6eccc463384a7ccb037` |
| `scripts/evaluator.diff` | `028f49f4f040eb2a21059dc0c094c48dbdf35f1a068e4b78539f30a06da3ea88` |
| `scripts/fr_common.py` | `edc8237adc41c8c65c85cfced15c792e9325dc719fccf2239aa9bd20920f3dcd` |
| `scripts/compare_runs.py` | `acfce8f2de6b23aea9204ccacb701994a069edfc56d58607102df6ea03439b60` |
| `scripts/validate_run.py` | `cbf202797b8cd78541276fc00f7f3ea470207f2722755c0b9f8cfe6dcf7479bf` |
| `scripts/make_qstar_requests.py` | `61fdfc4d33c77b34a56f2ea7e3c5235557925cac06550c76f6f63ddf8c2772b0` |
| `scripts/replay_client.py` | `da77cfe3eb895661bc21c17e07c6fc9f16f4000c88478530e1e60173dbe09410` |
| `scripts/run_sim.sh` | `f4d159bdfa74234cd837dd37fe63c7dff62b287485b87e68026a285f4eb006f7` |
| `scripts/server.sh` | `05de3d8181555c96647fcc2e20fd7a587209584862893dd619c28345b6a3f0f5` |
| `scripts/batch_validate.sh` | `3d1e3569a07794afff24594adb536101e6e414524b4ab8a1f98c3a4ddf58ee25` |
| `scripts/batch_main.sh` | `9149f2709b35d145fb72fb54a47ae531678a3a2e74843cbcb5d3e9ce826f4710` |
| `scripts/mock/run_offline_tests.sh` | `d16e093fc7da1ea36273f3eab4c438462a29f48e57b6fcf900b133db923332b1` |
