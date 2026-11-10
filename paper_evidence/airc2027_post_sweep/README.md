# Post-sweep tests

> [!NOTE]
> No paper is planned. Folder names containing `airc2027` and words such as "paper",
> "manuscript", "Fig." or "reviewer" in these files are left over from an earlier write-up plan
> that was dropped. They refer to this analysis, not to a forthcoming publication.

Three tests run after the sweep on the sweep workstation (RTX 3090, driver 580.173.02, Isaac Sim 4.1,
Isaac Lab fork 4d558ec) for the evaluation-reliability analysis. Files are copied unchanged
from the private NaVILA-Complete-Archive at commit 454c793, except for the withheld images below.

| Folder | Test | Private-archive commits |
| --- | --- | --- |
| `rendering_diagnosis/` | Cause of the gray robot-camera frames; measured camera heights; first repeatability checks of rendering, physics and the model server (Oct 1, 2026). Start at `DIAGNOSIS.md`. | a8fbd73 |
| `frame_replay/` | Whether rendering differences alone make reruns diverge: eight diverging episodes run twice with full logging, a run fed the first run's frames, and repeated model queries (Oct 1–2, 2026). Start at `RESULTS.md`. | plan 5ede302; results 7442611, 454c793 |
| `walking_check/` | The walking policy on the evaluated robot model and on its training robot model, flat ground (Oct 1–2, 2026). Start at `RESULTS.md`. | plan 52ec62c; results 5c1c09d |

The frame-replay and walking-check `PLAN.md` files were each committed to the private archive before
that test's first GPU use (frame replay: plan 2026-10-01 23:25:47 −06:00, first GPU use 23:26:07;
walking check: plan 23:00:19 −06:00) and were not changed afterwards; departures are listed under
"Deviations" in each `RESULTS.md`. The rendering diagnosis had no written plan. The private archive's
commit times cannot be checked publicly.

Withheld: 795 rendered images from `rendering_diagnosis/` and 256 from `frame_replay/` are not
published, as a precaution under the Matterport3D terms of use. Each folder's `WITHHELD_IMAGES.csv`
lists them (path, bytes, SHA-256); some links in the copied Markdown point to them. Published images,
some showing Matterport3D scenes at figure scale as in the analysis: `rendering_diagnosis/images/` (the
re-renders behind Fig. 1 and the corrected-model comparison), `frame_replay/figures/` and
`walking_check/figures/`. Run outputs that never left the workstation are listed in
`frame_replay/MANIFEST.csv`.

Not included: the simulator, the VLN-CE-Isaac dataset file, Matterport3D scenes, model weights, and robot
meshes, URDF or USD files. The numbers in the analysis that come from the per-episode records are
reproduced by `../airc2027_reanalysis/`.
