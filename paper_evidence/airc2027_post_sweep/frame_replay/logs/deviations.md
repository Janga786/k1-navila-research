# Running log of departures from PLAN.md (commit 5ede302), folded into RESULTS.md → Deviations

1. **Attempt 1 stopped at Step 1** (2026-10-01 23:26). The logging copy's in-process PNG reload failed inside
   Isaac Sim (Pillow mix: `PIL.Image` 12.3.0 from `~/.local`, `PIL.ImageFile`/`PIL.PngImagePlugin` from Kit's
   prebundled 10.2.0). Stopped per PLAN §9; report in `STOP_REPORT.md`. GPU used: 20 s simulator + 216 s
   server = 236 s. Evidence kept: `runs/validate_attempt1/`, `logs/validate_attempt1.log`,
   `logs/server_session1_attempt1.*`, `logs/batch_validate_attempt1.out`, `logs/offline_tests_attempt1.txt`.
   GPU_LOCK released 23:33; the walking check held the GPU 23:40–00:10 and released it after its push (5c1c09d).
2. **Copy patched with the user's approval (2026-10-02 ~09:57)**: `PROPOSED_FIX_png_decode.diff` applied — the
   two in-process PNG decodes (first-10-frames reload check; run R loading run A's frames) use
   `cv2.imread(..., IMREAD_UNCHANGED)` + BGR→RGB instead of `PIL.Image.open`. Nothing else changed; PLAN §5's
   descriptions still hold. The applied file is byte-identical to the version tested offline.
   New SHA-256: `scripts/navila_eval_v3_LOG.py` `cd9f13c034a07606d5cb75c90f4e76174cac28ae9f803f39651a67eb4965e124`
   (pinned: `37c0b3eb…`); `scripts/evaluator.diff` `344bf26a6a57f0fba520bc9ac0892045811c59ad0c92722e27c67e23cf9f0ce1`
   (pinned: `028f49f4…`).
3. **Offline suite extended**: `scripts/mock/run_offline_tests.sh` gained the simulated-Kit-Pillow-mix cases
   (the mock's fake AppLauncher reproduces the mix when `FAKE_KIT_PIL_MIX=1`), and its A/B fixture now uses fixed
   render seeds 11/12 (a pair known to diverge): the unpinned fixture did not diverge in one run, leaving the three
   request tests without input (`logs/offline_tests_attempt2_unpinned_fixture.txt`, 19/23). Final: 24/24
   (`logs/offline_tests.txt`). New SHA-256 `981adf873a79d9ca40c3d273f2695d1b956847509c50177cf36b725a9fc2b4a7` (pinned: `d16e093f…`).
4. **Budget**: attempt 1's 236 s count against the 9-h cap; the batch thresholds for attempt 2 are 8 h 45 min − 236 s
   and 9 h − 236 s (`FR_SOFT_S=31264`, `FR_HARD_S=32164`), clock from attempt 2's server session 1 start.
5. **Run labels / server sessions**: attempt 2 reuses the planned labels (`validate`, `ep*_A`, …) and starts a new
   server session 1; PLAN's session rule applies within attempt 2.
6. `scripts/analyze.py` (Step 5 tables/figures; formatting only, verdicts from the pinned `compare_runs.py`) was
   written before any run data existed (PLAN §12 said "after the runs").
7. GPU_LOCK re-created for attempt 2 at 2026-10-02 10:02:16 (no lock existed; GPU idle since the walking check released it at 00:10).
