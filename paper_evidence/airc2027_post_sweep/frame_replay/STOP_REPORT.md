# Frame-replay test — STOPPED at Step 1 (2026-10-01 23:26)

The experiment was stopped by the PLAN's stop rule: **the logging copy misbehaved** in the Step 1
validation run. Nothing was improvised; no further GPU run was made. The proposed fix below is **not
applied**.

## Where things stand

| Item | State |
|---|---|
| PLAN.md | pushed, private repo `Janga786/NaVILA-Complete-Archive`, `15_airc2027_frame_replay/PLAN.md`, commit `5ede302ecd1fdea980821468f92444d91f6485d5` |
| Results commit | none (nothing else pushed) |
| Runs | Step 1 `validate` only (episode_idx 92): failed at frame 0, rc 3. Steps 2–6 not run. |
| GPU time | 20 s simulator (`validate`) + 216 s model server (session 1) = 236 s (`logs/gpu_time.tsv`) |
| Model server | stopped (SIGTERM); `nvidia-smi` shows no compute process |
| GPU_LOCK | deleted after this report was written |
| Sweep tree | untouched: no file under `~/Projects/k1_research` outside `airc2027_replay/` (and the walking-check session's `airc2027_walking/`) is newer than the run's start marker; evaluator SHA-256 still `9b0107…281fc` |

## What failed (evidence: `runs/validate/error.txt`, `logs/validate.log`)

At the first frame append, the copy saves `frames/f0000.png` and reloads it to verify it (PLAN §5, "the first 10
PNGs are reloaded in-process"). The reload `np.array(Image.open(path))` raised `AssertionError` inside
`PIL/Image.py`:

```
File ".../.local/lib/python3.10/site-packages/PIL/Image.py", line 874, in tobytes
File ".../isaacsim/extscache/omni.kit.pip_archive/pip_prebundle/PIL/ImageFile.py", line 238, in load
File ".../isaacsim/extscache/omni.kit.pip_archive/pip_prebundle/PIL/PngImagePlugin.py", line 922, in load_prepare
File ".../.local/lib/python3.10/site-packages/PIL/Image.py", line 647, in im
    assert self._im is not None
```

## Diagnosis

- Inside the Isaac Sim 4.1 process the `PIL` package is **mixed**: `PIL.Image` is Pillow **12.3.0** from
  `~/.local` (imported by the evaluator before Kit starts), while `PIL.ImageFile` and `PIL.PngImagePlugin`, first
  imported after Kit starts, come from Kit's prebundled Pillow **10.2.0** (`omni.kit.pip_archive/pip_prebundle`,
  added to `sys.path` by that extension's `[[python.module]] path = "pip_prebundle"`). The same Kit log shows other
  extensions tripping over the mix (`cannot import name 'is_directory' from 'PIL._util'`).
- Reproduced without Kit and without the GPU: importing `PIL.Image` from 12.3.0 and then letting `PIL.ImageFile` /
  `PIL.PngImagePlugin` load from the prebundle gives the identical `AssertionError` on `Image.open(png)`; pure
  12.3.0 decodes the same file fine.
- **PNG encoding is not affected**: the mixed module set encodes PNGs byte-identically to pure 12.3.0 (tested on a
  real 1280×720 robot-camera frame and on the frame written inside Isaac: identical SHA-256 and length). So the
  sweep's request bytes were produced normally, and requests rebuilt outside Isaac (pure 12.3.0) should match the
  evaluator's — Step 1 check 3 would confirm this at every query.
- Only two lines of the copy decode PNGs in-process: the first-10-frames reload check and run R's loading of run
  A's frames (`--replay_frames_from`). Both would fail the same way. The offline mock could not catch this because
  it does not start Kit.

## Proposed fix (not applied): `PROPOSED_FIX_png_decode.diff`

Decode those PNGs with `cv2.imread(path, cv2.IMREAD_UNCHANGED)` + BGR→RGB instead of PIL (cv2 is already imported
by the evaluator; PNG is lossless, so any conforming decoder returns the same pixels), then wrap with
`Image.fromarray(arr)` exactly as before. 14 changed lines; nothing else changes (no effect on the simulation, the
random state, the control flow or the bytes sent to the model; PLAN §5's descriptions remain true).

Offline proof (scratchpad; mock harness now able to simulate the Kit mix with `FAKE_KIT_PIL_MIX=1`):
- pinned copy + simulated mix → rc 3, `AssertionError` (reproduces the Isaac failure);
- patched copy + simulated mix → normal runs rc 0; all Step 1 checks pass (PNG round-trip, request rebuild at
  every query, schema, consistency); in-process reload checks 10/10; run R → H3 held, R's history frames = A's.

## To resume (needs your OK)

1. Apply `PROPOSED_FIX_png_decode.diff` to `scripts/navila_eval_v3_LOG.py`; regenerate `scripts/evaluator.diff`;
   add the `FAKE_KIT_PIL_MIX=1` case to `scripts/mock/run_offline_tests.sh` and rerun it.
2. Record in RESULTS.md → Deviations: the copy was patched after the Step 1 failure (new SHA-256 of the copy and of
   `evaluator.diff`); PLAN.md itself stays unchanged.
3. Move the failed attempt aside (`runs/validate` → `runs/validate_attempt1`, `logs/validate.log` →
   `logs/validate_attempt1.log`; `run_sim.sh` refuses existing run folders) and keep it as evidence.
4. Re-take `GPU_LOCK` once the walking check has released the GPU, then run Step 1 again, then Steps 2–6 as planned.
