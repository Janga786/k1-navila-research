# ENVIRONMENT — provenance for the transform + camera-height sweep

Captured 2026-08-22 20:25 UTC on the machine that ran every sweep arm.
Companion files in this directory: `pip_freeze_vlnce-isaac.txt` (full package list),
`checkpoint_sha256.txt` (model weight hashes), `sweep_run_log.txt` (the raw runner log).

## Host
```
OS            : Ubuntu 22.04.5 LTS
Kernel        : 6.8.0-124-generic
CPU           : Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
RAM           : 62 GiB
GPU           : NVIDIA GeForce RTX 3090, 24576 MiB
NVIDIA driver : 580.173.02   <-- ALL sweep arms
              : 580.159.03 (PURGED, unobtainable)     <-- full_14498 baseline only
```

## Python / ML stack
```
conda env     : vlnce-isaac
python        : 3.10.20
torch         : 2.2.2+cu118
torch CUDA    : 11.8
numpy         : 1.26.4
scipy         : 1.15.3
```

## Simulator / benchmark code
```
IsaacLab commit          : 4d558ec83878c4892a46591c85ba91ac9d3c1834
IsaacLab local modifications:
   M source/extensions/omni.isaac.lab/omni/isaac/lab/sim/simulation_cfg.py
  ?? source/extensions/omni.isaac.matterport
  ?? source/extensions/omni.isaac.vlnce
```

`omni.isaac.vlnce` and `omni.isaac.matterport` show as untracked inside the IsaacLab
checkout because they are **symlinks** into `NaVILA-Bench-main/isaaclab_exts/`, which IS
tracked in this repository (61 files). No benchmark code is unversioned.

## Model
```
  4cb84059bf3280f80d4bd2df32d50059f20e0e8eb8b18d4523f5f3cde0dfbdc4  checkpoints/model_11999.pt
  01b01ce13e5151bdcf566b3ed892fbae70f94d5431594b69b13cf2af03f48a77  checkpoints/model_14498_jit.pt
  8e7559b5075fe6c90d4ce81fb1eb66d69ef852b2ed3718b5416b6a0fd2bdad4b  checkpoints/model_14498.pt
  0cc229329f0a3edb172a3e42959cf252afa0512fe14d46acbf4af0baf4dffa13  checkpoints/model_r5_14200_jit.pt
  60674bfe16b0f7bfa47911623660983c9c0a97960491957911e2a3d39fce30e8  checkpoints/model_r5_14200.pt
```
The evaluated policy is `checkpoints/model_14498.pt`. Weight files are gitignored
(size); the SHA-256 above pins exactly which weights produced every number in ANALYSIS.md.

## Methods-critical local simulator modification

`IsaacLab/.../sim/simulation_cfg.py` is modified relative to commit `4d558ec8` and the change
is captured verbatim in `isaaclab_simulation_cfg.patch` (apply it to reproduce). It **reduces
PhysX GPU buffer capacities** so the scene fits in the RTX 3090's 24 GB:

| PhysX buffer | upstream | ours |
|---|---|---|
| `gpu_max_rigid_contact_count` | 2^23 | 2^21 |
| `gpu_found_lost_pairs_capacity` | 2^21 | 2^18 |
| `gpu_found_lost_aggregate_pairs_capacity` | 2^25 | 2^18 |
| `gpu_total_aggregate_pairs_capacity` | 2^21 | 2^18 |
| `gpu_collision_stack_size` | 2^26 | 2^23 |
| `gpu_heap_capacity` | 2^26 | 2^22 |
| `gpu_temp_buffer_capacity` | 2^24 | 2^20 |

These are allocation-capacity knobs, not physics parameters, so they should not alter dynamics
**as long as no buffer overflows**. That caveat is not free: when a PhysX buffer does overflow
it drops contacts rather than failing loudly, which would perturb the simulation. State this
modification in Methods, and note the open question below.

**Open question (declare, do not hide):** whether these reduced capacities contribute to the
sim-side nondeterminism documented in `PRE_REGISTRATION.md` (§ROOT-CAUSE, §5a) has NOT been
tested. The root-cause work localised the chaos to the simulator and cleared the VLM, but did
not separate physics from rendering, and did not test buffer pressure as a mechanism. A
reviewer may reasonably ask; the honest answer is that it is untested. The cheap experiment is
to re-run the 5a two-process render probe at upstream buffer sizes on a larger card.

The same buffer settings applied to the `full_14498` baseline and to every sweep arm, so this
is a constant across all reported comparisons, not a differential confound between arms.
Verified by mtime, not assumed: `simulation_cfg.py` was last modified **2026-06-05 11:19**;
the baseline records were written **2026-06-11 → 06-22** and the sweep arms **2026-07-26 →
08-17**. The modification predates both, and nothing touched it in between.
