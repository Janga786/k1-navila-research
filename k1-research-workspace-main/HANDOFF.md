# Handoff Notes — K1 Humanoid Research Workspace

For a future Claude Code session: **start here**.
Last updated: 2026-05-08.

This document records the architecture, what was built recently, what's
pending, and the gotchas that aren't obvious from the code itself.

---

## 1. The big picture

The workspace covers three layers of a NaVILA-driven Booster K1 humanoid stack:

```
NaVILA (vision-language model)
        │ "move forward 75 cm" / "turn left 30 deg" / "stop"
        ▼
Multi-step planner + heading-assist + turn controller
        │ (vx, vy, vyaw) at 50 Hz
        ▼
K1 velocity-tracking policy (PPO, exported TorchScript)
        │ 12 leg joint targets
        ▼
MuJoCo physics @ 200 Hz (decimation 10) — K1 *physically walks*
```

Everything currently runs in **MuJoCo sim2sim**. Real-robot deploy goes
through `booster_deploy/scripts/deploy.py` (no `--mujoco` flag).

---

## 2. Project layout

```
~/Projects/k1_research/
├── HANDOFF.md                       ← THIS FILE
├── README.md                         workspace overview (older)
├── booster/                          upstream BoosterRobotics code (git submodules-ish)
│   ├── booster_assets/               K1/T1 URDF + MJCF + meshes
│   │   └── robots/K1/K1_22dof.xml    main MJCF (used by everything below)
│   ├── booster_deploy/               sim2sim + real-robot deployment
│   │   ├── scripts/deploy.py         entry point: --task=<name> [--mujoco]
│   │   ├── booster_deploy/controllers/mujoco_controller.py
│   │   └── tasks/locomotion/k1_velocity.py    ★ K1 velocity-tracking deploy task
│   ├── booster_train/                Isaac Lab + RSL-RL training
│   │   ├── scripts/rsl_rl/{train,play}.py
│   │   ├── source/booster_train/booster_train/tasks/manager_based/velocity/
│   │   │   └── robots/k1/            ★ Booster-K1-Velocity-v0 env config
│   │   └── logs/rsl_rl/k1_velocity/2026-05-08_14-15-21/
│   │       ├── model_9999.pt         latest training checkpoint
│   │       └── exported/             ★ TorchScript (.pt) + ONNX
│   └── NaVILA/                       NaVILA repo (8B llava model)
│       └── checkpoints/navila-llama3-8b-8f/
├── Documentation/
│   └── k1_velocity_2026-05-08_14-15-21/
│       ├── README.md                 training-run summary
│       ├── scalars_final.md          per-tag final values
│       └── plots/                    7 PNG plots from tensorboard
├── experiments/navila/               ★ closed-loop NaVILA + K1 demos
│   ├── navila_k1_core.py             SHARED brain: SubStep, planner, controllers, VLMRunner
│   ├── navila_k1_walking_loop.py     SIM path: NaVILA → trained policy → MuJoCo
│   ├── navila_k1_realrobot.py        REAL path: NaVILA → built-in walker (B1LocoClient.Move)
│   │                                 — modes: print | dry | live
│   │                                 — sources: zed | mjpeg | dir | static
│   ├── navila_k1_bridge.py           NaVILA loader, parse_action, prompt builder
│   ├── navila_mujoco_loop.py         older sliding demo (no walking)
│   ├── test_navila.py                manual NaVILA smoke test
│   ├── tests/                        58 unit tests (no NaVILA / SDK / MuJoCo deps)
│   ├── VALIDATION.md                 pre-flight pyramid: L0 deps → L4 live deploy
│   └── docs/demo_*.mp4               videos (in the GitHub repo)
└── ...
```

Pushed to GitHub: **https://github.com/Janga786/k1-vlm-navigation** (public,
just the `experiments/navila/` folder + README).

---

## 3. Conda environments

Three relevant envs (`conda env list`):

| env | python | mujoco | torch | use case |
|---|---|---|---|---|
| `isaacLab_311` | 3.11 | — | nightly cu128 | Isaac Lab training (`booster_train` play.py / train.py) |
| `isaacLab` | 3.10 | 3.6.0 | nightly cu128 | booster_deploy (mujoco + walking policy, no NaVILA) |
| `navila` | 3.10 | 3.8.0 | 2.7.1+cu128 | **everything in one process**: NaVILA + mujoco + walking policy |

The `navila` env is the only one that hosts the full closed-loop demo. It
also has the `llava` package via an editable install whose path mapping
points at `/home/janga/Projects/booster/NaVILA/...` (a path that **does not
exist** — the actual NaVILA is at `~/Projects/k1_research/booster/NaVILA`).
The `experiments/navila` scripts work around this by `sys.path.insert(0, ...)`
at the top.

Quick aliases used in the workspace (defined in shell rc):
- `k1env` — activate `isaacLab_311`
- `k1tasks` — list registered booster_train tasks
- `k1train BeyondMimic-K1-v0 --headless --num_envs 4096`

---

## 4. What was built recently (this session)

### 4.1 K1 velocity-tracking deploy task
Created `booster_deploy/tasks/locomotion/k1_velocity.py` and registered it
under the name `k1_velocity` in `booster_deploy/tasks/locomotion/__init__.py`.

The deploy task replicates the **training observation layout exactly**:
```
per-step (47): cmd(3) | gait_phase(2) | gravity(3) | ang_vel(3)
              | joint_pos_rel(12) | joint_vel(12) | last_action(12)
flat (235):    term-major history flatten of the 5 most-recent per-step obs
              (NOT history-major like the older t1_walk policy)
```

Joint order (verified by probing Isaac Lab):
```
[L_Hip_Pitch, R_Hip_Pitch, L_Hip_Roll, R_Hip_Roll,
 L_Hip_Yaw,   R_Hip_Yaw,   L_Knee_Pitch, R_Knee_Pitch,
 L_Ankle_Pitch, R_Ankle_Pitch, L_Ankle_Roll, R_Ankle_Roll]
```

PD gains and effort limits in the deploy cfg match the training
`BOOSTER_K1_LOCOMOTION_CFG` (kp=350 hips/knees, 250 ankles; kd=7.5/5.0;
knee effort 60 Nm — bumped from K1_CFG default 40). With these matched,
the K1 walks; with the older 100/50 gains the K1 falls in ~1.4 s.

### 4.2 Training documentation
`Documentation/k1_velocity_2026-05-08_14-15-21/` — 7 plots (training summary,
reward components, tracking metrics, terminations, optimisation, train
curves, perf), generated from the tensorboard event file. Headline
numbers: mean ep length 1490/1500, time-out termination 98 %, fps 188 k,
mean ep reward 103.

### 4.3 Closed-loop NaVILA + walking K1 demo
`experiments/navila/navila_k1_walking_loop.py`. Single process in `navila`
env. Two threads:

```
main thread:                            VLM thread:
  physics + walking policy at 50 Hz      pull latest 8-frame head buffer
  render head + scene cams at 30 Hz      run NaVILA generate() (~400 ms)
  push head frames into ring buffer  →   parse → (vx, vy, vyaw)
  read shared command each tick      ←   publish under lock
```

Key abstractions in this script:
- **`SubStep` dataclass** — one atomic instruction NaVILA gets, plus how we
  detect it's done: `proximity_target`, `yaw_delta_target`, `time_limit`.
- **`parse_substeps()`** — splits on `|` / `;` / "then"; auto-detects turn
  commands and named scene targets ("red box", etc.).
- **`WalkingSceneController(MujocoController)`** — overrides `__init__` to
  load the MJCF from a string (so we can splice in cameras + targets at
  runtime). Overrides `update_vel_command` to a no-op (the VLM thread sets
  the velocity command, not stdin).
- **`VLMRunner`** — owns NaVILA model + head-frame deque + shared cmd lock.
  `set_instruction()` lets the main thread switch sub-steps mid-flight;
  `clear_stop()` resets the stop event. **Doesn't terminate on "stop"** —
  main loop advances the instruction instead.
- **`build_scene_xml()`** — augments `K1_22dof.xml` with `head_cam` on
  Head_2, `scene_cam` (third-person, tracks Trunk), red target box,
  blue/green distractors, and bumps shadow/MSAA quality. **Important**:
  the K1 MJCF *already* has a `<geom name="ground">` plane with material
  `matplane`. We re-tune its friction (1.0 to match training) and kill
  `matplane`'s `reflectance` (the source of bright white blotches we saw
  in early renders). Don't add a second floor — z-fights catastrophically.

Three controllers ride underneath NaVILA:
1. **Heading assist** (`--heading-assist`, default on) — for proximity
   sub-steps, overlays `vyaw = K_p · (target_bearing − robot_yaw)` on top
   of NaVILA's command. Cancels the K1 policy's natural ~25 % sideways
   drift.
2. **Open-loop turn controller** (`--turn-controller`, default on) — for
   pure-turn sub-steps, **bypasses NaVILA** and feeds
   `vx=vy=0, vyaw=clip(K_p · (yaw_target − yaw_unwrap))`. NaVILA stubbornly
   emits "move forward" even when prompted with "turn right 90 degrees" so
   asking it to turn doesn't work. With `--turn-min-vyaw=0.30` floor and
   `--turn-tolerance-deg=5`, a 90° turn completes in **~2 s** (was 25-s
   timeout under NaVILA-only).
3. **Closest-approach termination** — sub-step done if `min_distance < 1.5 m`
   and current distance has grown by `> 0.25 m`. Catches "walked past the
   target" cases the proximity threshold misses.

Per-tick HUD prints `applied[TURN|HEAD|VLM ]` with the actually-commanded
vx/vy/vyaw and the VLM's raw output, so you can see at a glance which
controller is in charge.

### 4.4 Rendering fixes
The MuJoCo scene initially had ~50 % white-pixel fraction in the third-person
view. Two causes, both fixed:
1. I added a `<geom name="floor">` on top of the existing `<geom name="ground">` →
   coplanar planes z-fight catastrophically. **Fixed**: detect the existing
   plane and re-tune it instead of adding a duplicate.
2. The K1's `matplane` material has `reflectance="0.3"` → bright cubemap
   reflection samples produce white blotches. **Fixed**: override the
   material to `reflectance="0" shininess="0" specular="0"`.
Also: `<visual><quality shadowsize="4096" offsamples="4"/>` for sharper
shadows + 4× MSAA, plus a softer `<headlight>` and a haze color so the
horizon isn't a sharp seam. Result: white-pixel fraction down to **0.1 %**.

---

## 5. Pending work (next-session TODOs)

### 5.1 Make targets solid (collidable)  — TODO
The user's stated next request. Currently the box geoms in
`build_scene_xml()` set `contype="0" conaffinity="0"`, so MuJoCo treats them
as visual-only ghosts (the K1 walks through them). Two-line fix: remove
those two attributes from the box `<geom>` elements (target + distractors).
The K1's collision primitives (cylinders/boxes for trunk/arms/legs/feet)
have default `contype/conaffinity=1`, so collisions will work immediately.

Suggested CLI flag: `--ghost-targets / --no-ghost-targets` (default
`no-ghost-targets`, i.e. solid).

Caveats to test for after the change:
- The K1 walking policy is OOD on obstacles. Hitting a wall may topple it
  (safety fallback at `projected_gravity[2] > -0.5` will then stop the
  policy and the `is_running` check in the main loop exits cleanly).
- The proximity_threshold (default 1.0 m) leaves ~0.6 m clearance from
  box face (box is 0.4 m wide), so heading-assist sub-steps SHOULD stop
  before contact. Verify this on first test.

### 5.2 "Until you cannot see X" termination  — research-y
NaVILA isn't great at negative perception. A robust solution would be a
**second VLM call** (probe): every few seconds ask "Do you see the blue
box in the current view? yes/no" with a different prompt. Advance the
sub-step when answer is "no". Adds ~400 ms latency but only when active.
Currently this case falls through to the time-limit backstop.

### 5.3 Improve NaVILA yaw correction
NaVILA's outputs are biased toward "move forward N cm" even when the
target is off-axis. The heading-assist controller papers over this in sim
but the underlying VLM is the weak link. Options:
- Fine-tune NaVILA on K1-specific scenes
- Replace with a smaller, tighter VLM trained on direct waypoint regression
- Add a "look around → turn → walk" macro-step pre-amble

### 5.4 Real-robot deploy
**Done in principle**: see `experiments/navila/navila_k1_realrobot.py`
and `experiments/navila/VALIDATION.md`. Same NaVILA + planner +
controllers, but uses `B1LocoClient.Move` (the K1's built-in walker)
instead of the trained sim policy. Three modes:
- `--mode print` — no SDK, log only (offline NaVILA validation)
- `--mode dry`   — SDK init, no Move() (pre-flight on the real robot)
- `--mode live`  — real motion (with watchdog + always-kDamping shutdown)

Image sources: `zed | mjpeg | dir | static`.

**Still missing for actual deployment**:
- A `PoseSource` implementation (subscribe to `LowState` for IMU yaw →
  enables the open-loop turn controller on the real robot). Skeleton
  class is in `realrobot.py`. Without it, turn sub-steps fall back to
  NaVILA-only execution.
- A test on the real K1 with the floor cleared — climb the
  `VALIDATION.md` pyramid first (L0 deps → L1 unit tests → L2 MuJoCo
  end-to-end → L3 dry-run on real robot → L4 live).

---

## 6. Common commands

```bash
# === training (Isaac Lab, isaacLab_311 env) ===
conda activate isaacLab_311
cd ~/Projects/k1_research/booster/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-Velocity-v0 --headless --num_envs 4096
python scripts/rsl_rl/play.py  --task=Booster-K1-Velocity-v0-Play --headless --num_envs 1
                                                          # ↑ exports JIT + ONNX into logs/.../exported/

# === sim2sim deploy (isaacLab env) ===
conda activate isaacLab
cd ~/Projects/k1_research/booster/booster_deploy
python scripts/deploy.py -l                               # list registered tasks
python scripts/deploy.py --task=k1_velocity --mujoco      # run K1 in MuJoCo viewer
                                                          # (type "vx vy vyaw" + Enter to drive)

# === closed-loop NaVILA + walking K1 (navila env) ===
/home/janga/miniconda3/envs/navila/bin/python \
  ~/Projects/k1_research/experiments/navila/navila_k1_walking_loop.py \
  --instruction "walk to the red box | turn right 90 deg | walk forward" \
  --per-step-time 25 --save-video out

# pipeline test without burning NaVILA
python navila_k1_walking_loop.py --no-vlm --debug-vx 0.4 --max-sim-seconds 6 --save-video out

# === GitHub ===
gh repo view Janga786/k1-vlm-navigation
gh auth status
```

---

## 7. Gotchas / non-obvious things

- **`MUJOCO_GL=egl`** must be set before the first `import mujoco`. The
  GLFW backend conflicts with torch's CUDA init in the same process. The
  walking loop sets it via `os.environ.setdefault(...)` at the top.
- **Velocity policy obs is term-major history flatten**, not history-major.
  Each obs term keeps its own `(history_length, term_dim)` buffer; flatten
  concatenates `[term1_t0..tN, term2_t0..tN, ...]`. The older `t1_walk`
  policy in the same `LocomotionPolicy` class uses history-major flatten.
  This is why I wrote a separate `K1VelocityPolicy` instead of reusing.
- **Scene XML must use absolute meshdir.** `K1_22dof.xml` has
  `meshdir="meshes"` (relative). When loading from a temp file or string,
  rewrite the compiler's meshdir to absolute, otherwise mesh load fails.
- **NaVILA editable install path is broken.** The `__editable__.vila-1.0.0`
  finder in the `navila` env points at `/home/janga/Projects/booster/NaVILA`
  which doesn't exist. The actual NaVILA repo is at
  `~/Projects/k1_research/booster/NaVILA`. The walking loop adds it to
  `sys.path` explicitly. Don't `import llava` until that path is injected.
- **Sudo is per-tty.** When in a Claude Code session and you ask for sudo,
  the user has to type the password (`! sudo …`). Cache from one tty does
  not transfer to another.
- **Don't add a second floor** to the K1 scene. The MJCF already has
  `<geom name="ground">`. Z-fighting with a duplicate produces 50 % white
  pixels in renders.
- **`matplane` reflectance** is the source of bright white blotches in
  renders. Override to `0` if you build a scene from this MJCF.
- **The K1 deploy MJCF is 22-DoF** but the velocity policy was trained on
  the **12-DoF locomotion URDF** (arms get merged into Trunk by Isaac Lab's
  URDF importer because they have no actuator config). The deploy task
  scatters 12 leg targets into the 22-DoF default pose (arms held).
- **The walking policy needs ≥0.25 rad/s vyaw** to actually rotate the
  base. Smaller commands stall the discrete walking gait. Hence
  `--turn-min-vyaw=0.30`.

---

## 8. Useful inspection scripts

```bash
# probe Isaac Lab joint order for the K1 locomotion asset
/home/janga/miniconda3/envs/isaacLab_311/bin/python /tmp/probe_joints.py --headless

# render plots from a tensorboard event file
/home/janga/miniconda3/envs/isaacLab/bin/python /tmp/make_plots.py

# headless walking validation (no VLM, no viewer)
/home/janga/miniconda3/envs/isaacLab/bin/python /tmp/validate_k1_velocity.py --vx 0.5 --seconds 8
```

(These three scripts in `/tmp/` may have been cleaned up. The relevant
patterns are documented above.)

---

## 9. References

- **Booster-Gym paper** (training recipe): arXiv:2506.15132
- **NaVILA model card**: NVIDIA NaVILA-llama3-8b-8f (8-frame, indoor R2R/RxR)
- **GitHub**: https://github.com/Janga786/k1-vlm-navigation
- **Isaac Lab**: ~/IsaacLab (system install)
- **booster_deploy**: https://github.com/BoosterRobotics/booster_deploy
- **booster_train**: https://github.com/BoosterRobotics/booster_train

---

*If you (the next Claude session) are picking this up: read the
`navila_k1_walking_loop.py` file end-to-end first — it's the most concrete
artifact and exposes all the design choices. Then look at `k1_velocity.py`
in `booster_deploy/tasks/locomotion/` to see how the policy is plumbed.
The pending TODO in §5.1 is the natural next 5-minute task.*
