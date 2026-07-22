# COMMIT_LEDGER — final-week commit sweep, 2026-07-22 (STEP 1 output)
All commits authored `Janga786 <jangarabliss@gmail.com>`, no co-author trailers.
No force-pushes. Verification command per repo: `git log --author=Janga786 --oneline`.

## booster_train — 4 new commits, 64 files, 6,330 LOC added
Remote (created today, private): **https://github.com/Janga786/booster_train** (remote name `janga`; `origin` remains BoosterRobotics upstream, untouched — we cannot and should not push there)
| Commit | Files | LOC+ | Content |
|---|---|---|---|
| d9ee287 | 21 | 2,379 | K1 NaVILA locomotion task: env cfgs (base/T1/R5/R6), GAE-truncation env, all reward terms, obs-contract docs |
| 6c1e1fe | 13 | 2,375 | Evaluation gates + sim2sim tools (tracking gate, circumduction/contact probes, MuJoCo torque gate, renderer, L0 checks) |
| 8cd84df | 25 | 1,372 | Fail-closed pipelines rounds 4–7, weekend ablation sweep, reports (bulky logs excluded by pre-existing gitignore) |
| 607ea1e | 5 | 204 | K1 getup task |
Verified: `git log --author=Janga786` → 4 commits; working tree clean; `janga/main` = 607ea1e.
Identity fix applied: repo-local user was Gavinw575 → set to Janga786 before committing.

## booster_deploy — 3 new commits, 30 files, 1,620 LOC added (4 total by Janga786 incl. 07-02 k1_navila task)
Remote (created today, private): **https://github.com/Janga786/booster_deploy** (`janga`; `origin` = BoosterRobotics, untouched)
| Commit | Files | LOC+ | Content |
|---|---|---|---|
| 8c93bb6 | 4 | 15 | deploy.py warn-and-skip discovery + deploy.sh navilaenv wrapper + gitignore |
| 32a5e53 | 2 | 1 | k1_navila checkpoint → k1_navila_r5_14200 |
| 37c8cf7 | 24 | 1,604 | arm_scan task suite + validators/runners + math util fix |
NOTE: `tasks/*/models/` is gitignored by the repo's own convention — the r5_14200/14498
deploy weights are NOT in this repo; they are committed in k1-navila-research/checkpoints.

## k1-navila-research (~/Projects/k1_research) — 1 new commit (+1 for this ledger)
Remote: **https://github.com/Janga786/k1-navila-research** (pre-existing, pushed)
| 3d3f35d | 9 | 426 | EVIDENCE.md + receipts/ (recompute script/outputs, reward tables, run ledger, git ledger, media index, fresh tracking regen log) |

## k1-vlm-navigation — 0 new commits; **4 stranded commits pushed**
Remote: **https://github.com/Janga786/k1-vlm-navigation** (pre-existing)
The July deploy-stack commits (watchdogs, odometry, mode-switch verification, 73-test
suite) were local-only since the token expiry; `origin/main` now at 94a760e.

## Also checked, nothing to commit
NAVILA_COMPLETE_ARCHIVE (handled in STEP 2) · no other dirty authored repos under
~/Projects. Working trees verified clean after this sweep except archive (STEP 2).
