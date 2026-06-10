# How the K1 NaVILA System Works

A complete reference for the logic, data flow, and architecture of the
real-robot demo, the NaVILA-Bench evaluation pipeline, and the
legged-loco velocity-tracking policy.

---

## 1. SYSTEM OVERVIEW

The system answers one question: **"Given a natural-language instruction
like 'walk to the chair, then turn right 90 degrees, then walk forward',
how does a real humanoid robot (Booster K1) actually do it?"**

The answer is a three-stage stack. Stage 1 is **NaVILA**, an 8-billion
parameter video-language model fine-tuned for navigation. It reads the
robot's head-camera video and the instruction text, and emits short
mid-level commands like `"move forward 75 cm"` or `"turn left 30
degrees"`. Stage 2 is a **velocity planner** that translates those text
commands into Cartesian velocities `(vx, vy, vyaw)` — forward speed,
sideways speed, and yaw rate — clipped to safety caps, and decomposes a
multi-step instruction into sub-steps that have explicit
termination conditions. Stage 3 is the **locomotion controller**: on the
real robot it is the K1's built-in walker (`B1LocoClient.Move`); in the
NaVILA-Bench Isaac Sim benchmark it is a 12-dimensional joint-position
policy that was trained in legged-loco with PPO on rough terrain. The
three stages are connected by TCP because NaVILA needs a beefy GPU
(7-8 GB VRAM, ~400-1000 ms per inference) that you don't put on the
robot, the velocity planner runs near the GPU, and the K1 SDK can only
be reached from a host that is on the robot's Ethernet network.

---

## 2. THE REAL ROBOT PIPELINE (how the demo works)

The real-robot demo is split across three machines: a **desktop** that
runs NaVILA, a **laptop** that talks to the robot over the K1 SDK and
holds the camera, and the **K1 robot** itself, which receives Cartesian
velocity commands and walks. The desktop and laptop talk over Tailscale
(TCP/5555); the laptop and K1 talk over Ethernet.

There is also a sim-only path that runs everything in one process
against a MuJoCo simulation — useful for testing without the robot.

### 2.1 `navila_server.py` — the GPU brain on the desktop

**What it does.** Owns the NaVILA model. Accepts a single TCP client
(the laptop relay), receives the multi-step instruction and JPEG frames,
runs inference, and returns the latest `(vx, vy, vyaw)` plus the raw
NaVILA text on every tick. The model is loaded once at process start
and stays resident across client reconnects.

- **Main loop**: `main()` → `handle_client()` for each accepted socket.
- **Data IN**: TCP messages — `hello`, `set_instruction`, `tick`
  (header + JPEG blob), `shutdown`. Wire format defined in
  `navila_protocol.py`.
- **Data OUT**: TCP `state` messages containing
  `{vx, vy, vyaw, tag, label, raw, step_idx, step_total,
  step_instruction, done_reason, all_done, drain_done, vlm_stop,
  inf_count, inf_ms, buffer_size}`.
- **Key constants**:
  - `DEFAULT_PORT = 5555` (proto).
  - The `Session` class' default `SessionConfig` keeps `per_step_time=25
    s`, `vx_max=0.4`, `vy_max=0.15`, `vyaw_max=0.4`,
    `proximity_threshold=1.0 m`, `drain_seconds=1.5`. These are
    deliberately conservative on real-robot caps (sim uses 0.6).
  - `heading_assist` and `turn_controller` default OFF on this server
    because they need a pose source, which the laptop relay does not
    have.
- **Failure modes**:
  - **Client tries to `tick` before `set_instruction`** → server
    replies with an `error` message; nothing breaks.
  - **JPEG decode failure** → the bad frame is dropped, tick still
    returns the most recent VLM cmd.
  - **NaVILA inference fails inside `VLMRunner._loop`** → prints the
    exception, sleeps 0.5 s, retries; the published command stays at
    its last value (or zero if the loop hasn't published yet).
  - **Socket timeout (default 60 s)** closes the connection; the next
    `hello` from the laptop spawns a fresh `Session`.

### 2.2 `navila_k1_core.py` — the planner + the inference thread

This is the brain of the planner. It is imported by every entry point
that needs to translate "VLM says X" into "send these velocities for Y
seconds and decide when the sub-step is done."

**Components.**

1. **`SubStep` dataclass** — one atomic chunk of an instruction, with
   its own termination criterion: a yaw delta target, a proximity
   target, or just a time limit.
2. **`parse_substeps()`** — splits a free-text instruction on `|`, `;`,
   or `then` (with optional comma); for each chunk it scans for a turn
   phrase (`"turn left 90 deg"`) → sets `yaw_delta_target`, and for the
   last named scene target (`"red box"`, `"blue cube"`, ...) → sets
   `proximity_target` from `DEFAULT_SCENE_TARGETS`. If both are
   detected it keeps the yaw and drops the proximity (turns are
   explicit primitives).
3. **`check_termination()`** — decides whether the current sub-step is
   done. Priority order (first match wins): (1) yaw target reached
   within tolerance, (2) proximity target hit OR closest-approach
   overshoot, (3) NaVILA emitted `"stop"`, (4) time limit elapsed.
4. **`apply_controllers()`** — picks the actual `(vx, vy, vyaw)` to
   send. Three mutually-exclusive cases tagged `TURN`, `HEAD`, `VLM`:
   - **TURN**: pure-turn sub-step + `turn_controller=True` → bypass
     NaVILA, drive `(0, 0, K·(target − unwrap_yaw))` with a min-mag
     floor.
   - **HEAD**: proximity sub-step + `heading_assist=True` + pose
     available → keep NaVILA's vx, but **add** a yaw-correction
     overlay so the robot keeps pointing at the target while NaVILA
     controls forward speed.
   - **VLM**: pass-through, clipped to caps.
5. **`VLMRunner`** — the NaVILA inference thread. Owns the
   episode-long rolling frame buffer, samples 8 frames per inference
   following paper §II-A (see §3 below), runs `model.generate`, parses
   the output, holds the resulting velocity for `duration_s` seconds,
   then zeros the command and starts the next inference. Stretches
   `duration` when the operator's safety cap clips the paper-spec
   speed, so the requested distance/angle is still travelled.

**Key constants.**
- `NUM_FRAMES = 8` (paper-spec for "8f" model). 
- `ACTION_DURATION = 1.5 s` (legacy fallback for unparsed actions).
- `_buf_soft_cap = 500`, `_buf_recent_keep = 50` — memory cap to keep
  long episodes from eating GBs of RAM; compaction preserves the first
  frame and the most recent 50, and uniformly down-samples the middle.

**Failure modes.**
- **NaVILA emits something the parser doesn't understand** →
  `parse_action` returns label `"unparsed -> stop"` with zero velocity.
  Robot stays still until the next inference; sub-step's time limit
  eventually fires.
- **Buffer never gets seeded** → `_loop` sleeps in 50 ms increments
  forever (until `bootstrap_buffer` is called).
- **Unbounded episode** → if you forget to call `set_instruction`, the
  buffer grows; compaction caps it at 500 frames automatically.

### 2.3 `navila_k1_bridge.py` — text → velocity

Two pure-Python helpers that are also re-exported by the core.

- `build_prompt(instruction, num_frames)` — builds the exact prompt
  NaVILA was fine-tuned on (see §3 for the template).
- `parse_action(text)` — regex parser that maps NaVILA's mid-level
  output to **paper-spec fixed velocities**:
  - `move/walk/go/step forward N (cm|m)` → `vx=0.5, dur=N/0.5`
  - `move back N` → `vx=-0.5, dur=N/0.5`
  - `turn left/right N (deg|rad)` → `vyaw=±π/6, dur=N/(π/6)`
  - `stop|halt|done|completed` → `(0,0,0), dur=0, label="stop"`
  - anything else → `(0,0,0), dur=0, label="unparsed -> stop"`

Lazy-imports torch / llava so it's importable in test environments.

### 2.4 `navila_k1_realrobot.py` — single-process entry point

Same logic as `navila_server.py`, but the planner lives in the same
process as the SDK. Splits responsibilities into pluggable classes:

- **`ImageSource`**: `StaticImageSource`, `DirReplayImageSource`,
  `MJPEGImageSource`, `ZEDImageSource` — each is `__call__`-able to
  return a `PIL.Image`.
- **`Actuator`**: `PrintActuator` (no SDK), `DryRunActuator` (SDK
  initialised, `Move()` never called), `LiveActuator` (real motion,
  background sender thread at `SEND_HZ=20`, **watchdog** zeros the
  command if the planner hasn't refreshed it in `1.5 s`).
- **`PoseSource`**: stub by default. Subclass + plug in to enable
  heading-assist / proximity termination on the real robot.

**Main loop**: `main()` runs at ~50 Hz, polls pose (if available),
reads `vlm.get_command()`, runs `check_termination`, advances sub-steps
when done, runs `apply_controllers`, calls `actuator.send(vx, vy,
vyaw)`. Pushes a head frame into the VLM buffer every
`--frame-buffer-period` (default 0.4 s).

**Failure modes.**
- **`pyzed` not installed** → `ZEDImageSource.__init__` raises a clear
  error pointing at the SDK install.
- **`booster_robotics_sdk_python` missing** → `DryRunActuator.init()`
  raises with a message saying "use `--mode print` instead."
- **LiveActuator can't reach the K1** → `Move()` exceptions print to
  stderr; the planner keeps running. Operator is expected to Ctrl-C.

### 2.5 `navila_laptop_relay.py` — the laptop side of the split deploy

When the GPU lives on a desktop and the SDK lives on a laptop, this is
the laptop's process. It connects to the desktop server, streams JPEG
frames over Tailscale, reads back velocity commands, **re-clips them
locally** (defense in depth), and calls `actuator.send(vx, vy, vyaw)`
just like `navila_k1_realrobot.py` would. The HUD (cv2 window) shows
the live camera with overlaid step number, instruction, NaVILA raw
output, applied command, and link age.

- **`RemoteVLMClient`** — owns the TCP socket and a background
  "tick thread" that sends `tick` messages every `--tick-period`
  (default 0.4 s) and stashes the server's response in `self._state`.
- **Main loop** at ~20 Hz: grab a frame, hand it to the remote client,
  read latest `state`, re-clip, `actuator.send`, draw HUD, sleep.

**Failure modes.**
- **Network blip** → tick thread catches `ConnectionError/timeout`,
  sets `connection_lost=True`, and exits. Main loop notices, breaks
  out, sends zero velocity, switches the robot to `kPrepare`.
- **Server stops responding** → `--io-timeout` (5 s default) fires the
  same path.
- **Operator hits 'q' or Ctrl-C** → SIGINT handler sets a stop event;
  finally block zeros the cmd and switches mode.

### 2.6 `navila_protocol.py` — wire format

Each message is framed as:
```
[4 bytes BE uint32: json_len]
[json_len bytes: utf-8 JSON header]
[4 bytes BE uint32: blob_len]
[blob_len bytes: binary blob, may be empty]
```
The header has a `"type"` field (`hello`, `set_instruction`, `tick`,
`shutdown` from laptop; `hello_ack`, `instruction_ack`, `state`,
`shutdown_ack`, `error` from server). JPEG bytes ride in the blob on
`tick` messages. Hard caps: 1 MiB JSON, 16 MiB blob.

### 2.7 `navila_k1_walking_loop.py` — full sim with the trained policy

A single-process MuJoCo loop that wires NaVILA → planner → the
**trained walking policy** from `booster_deploy.controllers.mujoco_controller`.
Same planner code as the real-robot path; the actuator is replaced by
`WalkingSceneController`, which:
1. Loads `K1_22dof.xml` and splices in: a head camera (mounted on
   `Head_2`, FOV 70°, facing body +X), a third-person `scene_cam`
   tracking the trunk, a red navigation target, blue/green
   distractors.
2. Runs the trained 235→12 MLP policy at 50 Hz (decimation 10 ×
   200 Hz physics).
3. Renders two MP4s when `--save-video` is set (head view + scene
   view), with a HUD overlay showing step status, distance, yaw delta.

### 2.8 `navila_mujoco_loop.py` — the older "no walking" sim

Kinematically slides the floating base instead of running a real
policy. The K1 stands in `STANDING_QPOS`; `slide_base()` translates
and rotates the floating base in the body frame at the commanded
velocity. Useful purely for testing NaVILA's perception loop in
isolation; the legs don't move.

### 2.9 End-to-end data flow (split-deploy mode)

```
+----------------------------- DESKTOP (GPU) -----------------------------+
|                                                                        |
|  navila_server.py                                                      |
|      |                                                                 |
|      |  on hello/set_instruction:                                      |
|      |    - construct Session(VLMRunner, SessionConfig)                |
|      |    - parse_substeps() -> list[SubStep]                          |
|      |    - VLMRunner buffer cleared, instruction set                  |
|      |                                                                 |
|      |  on tick(jpeg, optional pose):                                  |
|      |    - decode JPEG -> PIL.Image                                   |
|      |    - VLMRunner._frame_buffer.append(img)                        |
|      |    - update_yaw_unwrap(state, pose_yaw) if pose                 |
|      |    - check_termination(...) -> maybe advance step               |
|      |    - apply_controllers(...) -> (vx, vy, vyaw, tag)              |
|      |    - return state JSON                                          |
|      |                                                                 |
|      |                                                                 |
|      |  VLMRunner inference thread (concurrent):                       |
|      |    while True:                                                  |
|      |      frames = sample_frames(buffer, NUM_FRAMES)  # paper §II-A  |
|      |      raw    = _navila_inference(...)             # generate()   |
|      |      vx,vy,vyaw,dur,label = parse_action(raw)                   |
|      |      clip to caps; stretch duration if clipped                  |
|      |      publish (vx,vy,vyaw); hold dur seconds; publish (0,0,0)    |
|      |      if label=='stop': stop_event.set(); sleep(0.1)             |
|      |                                                                 |
+--------------------+---------------------------------------------------+
                     ^                                              |
                     |  state JSON (5 Hz)                           |  Tailscale TCP/5555
                     |                                              v
+--------------------|---------------------------- LAPTOP ----------|------+
|                    |                                              |     |
|  navila_laptop_relay.py                                                 |
|      ^                                                                  |
|      |  ZED grab (or MJPEG, dir, static)                                |
|      |    PIL.Image (RGB)                                               |
|      |       |                                                          |
|      |       v                                                          |
|      |  encode_jpeg(quality=80)                                         |
|      |       |                                                          |
|      |       v  RemoteVLMClient.set_frame_jpeg()                        |
|      |       v  RemoteVLMClient tick thread ----> [server, see above]   |
|      |       <----- state JSON                                          |
|      |                                                                  |
|      v                                                                  |
|  _clip(state.vx, vx_max); same for vy, vyaw                             |
|  actuator.send(vx, vy, vyaw)                                            |
|      |                                                                  |
|      v                                                                  |
|  LiveActuator background sender (20 Hz):                                |
|      cmd_lock: vx, vy, vyaw = self._cmd                                 |
|      if age > 1.5s: vx=vy=vyaw=0  (watchdog)                            |
|      client.Move(vx, vy, vyaw)                                          |
|                                                                         |
+----------------------+--------------------------------------------------+
                       |
                       |  Booster SDK (Ethernet, ~20 Hz Move() calls)
                       v
+----------------------|--------------------------------- K1 -------------+
|                                                                         |
|  B1LocoClient (Booster's built-in walker firmware):                     |
|    - takes (vx, vy, vyaw)                                               |
|    - produces 22-DoF joint torques                                      |
|    - K1 walks                                                           |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## 3. THE VLM (how NaVILA thinks)

NaVILA is a **video-language model**: it takes 8 RGB frames and a text
instruction, and emits one short navigation command per inference.

### 3.1 The prompt template

Built by `navila_k1_bridge.build_prompt`. Used identically by the real
robot (`navila_server.py`), the MuJoCo loops, and the benchmark
(`vlm_server_bridge.py` builds the same string locally — they must
match byte-for-byte because the model was fine-tuned on it).

```python
def build_prompt(instruction: str, num_frames: int = 8) -> str:
    image_token = "<image>\n"
    history_tokens = image_token * (num_frames - 1)   # 7 <image>\n tokens
    return (
        f"Imagine you are a robot programmed for navigation tasks. "
        f"You have been given a video "
        f"of historical observations {history_tokens}, "
        f"and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        f"Analyze this series of images to decide your next action, "
        f"which could be turning left or right by a specific degree, "
        f"moving forward a certain distance, or stop if the task is "
        f"completed."
    )
```

For an 8-frame model the prompt contains 7 historical `<image>\n`
tokens then 1 current `<image>\n` token, in that exact order. The
`tokenizer_image_token` helper replaces those literal sequences with
`IMAGE_TOKEN_INDEX` placeholders that the model then attends to as
visual tokens. The conversation template `llama_3` wraps the user
question and the assistant response stub before the model generates.

### 3.2 Frame sampling (paper §II-A)

The reference NaVILA repo, `llava/mm_utils.get_frame_from_vcap_vlnce`,
does exactly this:

```python
latest_frame = frames[-1]
sampled_indices = np.linspace(0, len(frames)-1, num=N-1,
                              endpoint=False, dtype=int)
sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
```

`VLMRunner.sample_frames` and `navila_eval.sample_images_and_send_to_vlm`
both implement this. **Why this exact recipe matters:**

- `np.linspace(0, len-1, num=N-1, endpoint=False, dtype=int)` always
  produces an index `0` as its first element (because `endpoint=False`
  means the stop value is excluded but the start is included). This
  guarantees the first frame of the episode is always present in the
  prompt — the paper's invariant.
- The latest frame is appended separately, so the prompt has
  7 "historical" + 1 "current". When the episode has > 8 frames, the
  current is distinct from any historical.
- If the buffer has fewer than 8 frames, the laptop pipeline left-pads
  with the first frame (gentle bootstrap); the benchmark pipeline pads
  with **black frames** at index 0 (harsher but matches
  `vlnce_frame_sampling` in `mm_utils.py`).

**Worked example: 50 accumulated frames, sample 8.**

```
total = 50
indices = np.linspace(0, 49, num=7, endpoint=False, dtype=int)
        = [0, 7, 14, 21, 28, 35, 42]     # uniform across [0, 49)
sampled = frames[0], frames[7], frames[14], frames[21], frames[28],
          frames[35], frames[42], frames[49]
```

The first frame is always in slot 0. The current frame (frame 49) is
in slot 7. The middle six are uniformly stepped across the rest. This
gives NaVILA both long-horizon scene context (start of episode) and
fresh observations (most recent frames).

**Note on the laptop's bootstrap.** The first time you `bootstrap_buffer`,
the code pushes exactly **one** frame to the buffer rather than
NUM_FRAMES copies. That's intentional — the comment in
`VLMRunner.bootstrap_buffer` explains that pre-filling 8 copies would
bias the linspace sampling toward the bootstrap frame for the rest of
the episode. The pad-with-first-frame logic only kicks in when fewer
than NUM_FRAMES are available, so on episode start NaVILA sees
`[frame0]*7 + [frame0]`, and as more frames arrive the linspace samples
naturally spread out.

### 3.3 Parsing the output

NaVILA emits free text. After a short scene description it usually
ends with a command verb. `parse_action` uses four regexes:

```python
_FORWARD  = r"\b(?:move|walk|go|step)\s+(?:forward|ahead)\s+(\d+(?:\.\d+)?)\s*(cm|m|meter|meters|metre|metres)\b"
_BACKWARD = r"\b(?:move|walk|go|step)\s+(?:back|backward|backwards)\s+(\d+(?:\.\d+)?)\s*(cm|m|meter|meters|metre|metres)\b"
_TURN     = r"\bturn\s+(left|right)\s+(\d+(?:\.\d+)?)\s*(deg|degree|degrees|rad|radian|radians)\b"
_STOP     = r"\b(?:stop|halt|done|complete[d]?)\b"
```

Matching order: stop → forward → backward → turn → fall-through.

**Possible outputs and how each is handled.**

| NaVILA says                  | Parsed (vx, vy, vyaw)      | duration                | label                 |
| ---------------------------- | -------------------------- | ----------------------- | --------------------- |
| `move forward 75 cm`         | `(0.5, 0, 0)`              | `0.75 / 0.5 = 1.5 s`    | `forward 0.75m`       |
| `move forward 25 cm`         | `(0.5, 0, 0)`              | `0.5 s`                 | `forward 0.25m`       |
| `walk forward 0.5 m`         | `(0.5, 0, 0)`              | `1.0 s`                 | `forward 0.50m`       |
| `move back 30 cm`            | `(-0.5, 0, 0)`             | `0.6 s`                 | `backward 0.30m`      |
| `turn left 30 degrees`       | `(0, 0, +π/6)`             | `π/6 / (π/6) = 1.0 s`   | `turn left ...rad`    |
| `turn right 90 degrees`      | `(0, 0, -π/6)`             | `3.0 s`                 | `turn right ...rad`   |
| `stop` / `halt` / `done`     | `(0, 0, 0)`                | `0.0 s`                 | `stop` (sets event)   |
| `I see a chair.`             | `(0, 0, 0)`                | `0.0 s`                 | `unparsed -> stop`    |

Anything that hits `unparsed -> stop` is treated by the planner like
"no progress; wait for the next inference or the time limit." It does
**not** trigger `stop_event` (which would end the sub-step). That is
deliberate: a one-frame parse miss shouldn't kill the run.

### 3.4 Inference pipeline

`VLMRunner._navila_inference` (and the identical code in
`vlm_server_bridge.py`) does:

1. **Image preprocessing.** `process_images(frames, image_processor,
   model.config)` runs the model's image processor (CLIP-style
   resize + normalize) and returns a `[8, 3, H, W]` tensor in fp16.
2. **Prompt build.** Construct the string above, wrap with the
   `llama_3` conversation template (`<|begin_of_text|>...
   <|start_header_id|>user<|end_header_id|>... <|eot_id|>
   <|start_header_id|>assistant<|end_header_id|>`).
3. **Tokenize with image splicing.** `tokenizer_image_token` replaces
   each `<image>\n` with the `IMAGE_TOKEN_INDEX` sentinel
   (-200), splits the prompt into chunks around those sentinels, and
   produces a 1-D LongTensor of token IDs.
4. **Generate.** `model.generate` with `do_sample=False, temperature=0,
   max_new_tokens=256` (greedy decoding so the same frames produce the
   same output), with `KeywordsStoppingCriteria([stop_str], ...)` that
   halts as soon as the assistant turn delimiter appears.
5. **Decode + trim.** `tokenizer.batch_decode` and strip the stop
   string. Result is a single short sentence like `"The next action is
   move forward 75 cm."`

The 256-token cap is a recent fix: the original reference script used
64, which silently truncated before the action verb on some prompts
and forced the parser into `unparsed -> stop` (the robot froze).

---

## 4. THE VELOCITY MAPPING (how text becomes movement)

The mapping is **paper-spec fixed-speed**: NaVILA's discrete set
`{forward, turn left, turn right, stop}` is cast to fixed velocities
`{0.5 m/s, π/6 rad/s, -π/6 rad/s, 0}`. Distance/angle changes the
**duration** the velocity is held, not the speed.

### 4.1 Forward example

NaVILA emits `"move forward 75 cm"`.

```
n  = 75
u  = "cm"
d  = 75 / 100 = 0.75 m

vx        = FORWARD_SPEED  = 0.5 m/s
vy        = 0.0
vyaw      = 0.0
duration  = |d| / FORWARD_SPEED = 0.75 / 0.5 = 1.5 s
distance  = vx * duration = 0.5 * 1.5 = 0.75 m   ✓
```

Other forwards: 25 cm → 0.5 s, 50 cm → 1.0 s, 1 m → 2.0 s. Speed is
always 0.5 m/s.

### 4.2 Turn example

NaVILA emits `"turn left 30 degrees"`.

```
n  = 30
u  = "deg"
a  = radians(30) = π/6 ≈ 0.524 rad

vx        = 0.0
vy        = 0.0
vyaw      = +TURN_SPEED = +π/6 ≈ +0.524 rad/s   (left = +)
duration  = |a| / TURN_SPEED = (π/6) / (π/6) = 1.0 s
angle     = vyaw * duration = π/6 ≈ 30°         ✓
```

`turn right 90°` is `vyaw=-π/6, dur=3.0 s`.

### 4.3 The vyaw safety clip + duration-stretch fix

The real-robot caps default to `vyaw_max=0.4 rad/s`, **below** the
paper-spec π/6 ≈ 0.524 rad/s. Without the fix, a 30° turn would be
0.4 × 1.0 = 0.4 rad = 23°, undershooting by 7°.

The fix lives in `VLMRunner._loop` right after clipping:

```python
if duration > 0.0:
    scale = 1.0
    for req, applied in ((req_vx, vx), (req_vy, vy), (req_vyaw, vyaw)):
        if applied != 0.0 and abs(applied) < abs(req):
            scale = max(scale, abs(req) / abs(applied))
    duration *= scale
```

For the 30° example with `vyaw_max=0.4`:
- `req_vyaw = π/6 ≈ 0.524`, `applied_vyaw = 0.4`.
- `scale = 0.524 / 0.4 ≈ 1.31`.
- `duration = 1.0 * 1.31 = 1.31 s`.
- Angle covered = `0.4 * 1.31 ≈ 0.524 rad = 30°` ✓.

The only deviation from the paper is a slightly longer execution time,
which is exactly what you want under conservative caps.

### 4.4 What happens between actions

After holding the commanded velocity for `duration` seconds, the
`VLMRunner._loop` does:

```python
self._publish(0.0, 0.0, 0.0, f"{label} (settle)", raw, 0.0)
time.sleep(0.1)  # 100 ms settle so the K1 comes to rest
```

The settle prevents the next NaVILA inference from observing motion
blur and prevents the discrete walking gait from carrying momentum
into the next command. The status tag becomes `"<action> (settle)"` so
the HUD shows what's happening.

### 4.5 What happens on `stop`

When `parse_action` returns label `"stop"`:

1. `_publish(0, 0, 0, "stop", raw, inf_ms)` — zero velocity is sent.
2. `self.stop_event.set()` — the planner's `check_termination` will
   pick this up on the next tick and advance the sub-step.
3. `time.sleep(0.1)` — avoid busy-spinning while the planner advances.

If the sub-step that just finished was the **last** one, the planner
sets `drain_deadline = now + drain_seconds`, during which it sends
`(0, 0, 0)` regardless of what NaVILA says. After the drain expires
the plan is complete.

---

## 5. THE BENCHMARK PIPELINE (how we get paper metrics)

The benchmark answers a different question: **"On the public R2R-CE
Matterport dataset, what success rate does the system achieve?"** This
needs a controlled simulator (Isaac Sim 4.1 + Matterport scenes),
deterministic episodes with ground-truth paths, and the standard VLN
metrics (SR, SPL, NE, ONE, OSR, PL).

### 5.1 `run_benchmark.py` — the outer driver

```python
for i in range(start_idx, end_idx):
    subprocess.run(['python', 'scripts/navila_eval.py',
                     f'--episode_idx={i}', ...],
                    timeout=300)
```

It iterates the R2R-CE episodes loaded from
`vln_ce_isaac_v1.json.gz` and launches one `navila_eval.py` subprocess
per episode. The subprocess boundary is necessary because Isaac Sim
leaks GPU memory across episodes; killing the process between runs is
the supported workaround. The 300 s timeout catches Isaac Sim cold-start
hangs.

### 5.2 `navila_eval.py` — one episode

Pseudocode of the main loop:

```python
# Setup
episode    = read_episodes(r2r_data_path)[args_cli.episode_idx]
env_cfg    = parse_env_cfg(args_cli.task)
reset_start_pos_rot(env_cfg, args_cli, episode)
env        = gym.make(args_cli.task, cfg=env_cfg)
env        = RslRlVecEnvHistoryWrapper(env, history_length=9)  # for go2
ppo_runner = OnPolicyRunner(env, agent_cfg)
ppo_runner.load(resume_path)  # checkpoint from legged-loco
policy     = ppo_runner.get_inference_policy()
env        = VLNEnvWrapper(env, policy, ...)
obs, infos = env.reset()

# Warmup: 8 frames at zero velocity so the VLM doesn't get black-pads
for warm_i in range(8 * steps_per_image):
    obs, _, done, infos = env.step(zero_cmd)
    if (warm_i + 1) % steps_per_image == 0:
        image_observations.append(Image.fromarray(curr_frame))

# Main loop
while simulation_app.is_running():
    if num_steps == target_steps:
        stream_output = sample_images_and_send_to_vlm(
            image_observations, vlm_host, vlm_port, instruction.instruction_text)
        vlm_vel_commands, time_to_go = get_vel_command(stream_output)
        env_steps_to_go = int(time_to_go / (dt * decimation))
        target_steps = num_steps + env_steps_to_go

    obs, _, done, infos = env.step(torch.tensor(vlm_vel_commands))
    if done or env.is_stop_called or num_steps > max_episode_steps: break

    # camera obs accumulates one frame every 0.5 s of sim time
    if num_steps % steps_per_image == 0:
        image_observations.append(Image.fromarray(curr_frame))

    num_steps += 1
    if env_steps_to_go == 0: env.set_stop_called(True)

# Save measurements + video
```

Three things to notice:

1. **Frames accumulate at one per 0.5 sim seconds** (`steps_per_image
   = 0.5 / 0.02 = 25 sim steps`). This matches the rate NaVILA was
   trained at.
2. **The VLM is only called at `target_steps`** — once a command has
   been chosen, the environment steps for the full `env_steps_to_go`
   ticks before asking again. While stepping, the velocity command in
   the proprioceptive observation does **not** change.
3. **The black-frame warmup** at the top is the "Fix B" mentioned in
   the comment: without it the first call to NaVILA would pad the
   buffer with black frames, which biases NaVILA into hallucinating
   "turn left 45°" for the first ~10 commands.

### 5.3 `vlm_server_bridge.py` — the VLM JSON socket

A standalone server that hosts NaVILA in the `booster` conda env
(which has `llava` installed), separate from `vlnce-isaac` (which has
Isaac Sim 4.1 + RTX 5090 wiring but no `llava`). The wire format here
is **different** from the relay protocol:

```
[8 bytes BE size_header]
[size bytes UTF-8 JSON payload]

Request payload:  {"images": ["<b64-JPEG>", ...], "query": "<instruction>"}
Response payload: "<navila output as bare JSON string>"
```

`navila_eval.sample_images_and_send_to_vlm` does the frame sampling
(same `np.linspace` recipe as the laptop pipeline), encodes each as
base64 JPEG, sends, and waits for the bare-string response.

### 5.4 `VLNEnvWrapper` — text → velocity → joint actions

The wrapper sits between the NaVILA action and the low-level policy.
Its `step(action)` does:

```python
def step(self, action) -> ...:
    self.update_command(action)              # inject vx,vy,vyaw into obs
    low_level_action = self.low_level_policy(self.low_level_obs)
    low_level_obs, reward, done, info = self.env.step(low_level_action)
    self.low_level_obs = low_level_obs
    ...
    return obs, reward, done, info
```

`update_command(command)` overwrites the `velocity_commands` slot of
the proprioceptive observation:

```python
if isinstance(self.env, RslRlVecEnvHistoryWrapper):
    self.low_level_obs[:, 6:9] = command
    self.env.proprio_obs_buf[:, -1, 6:9] = command
else:
    self.low_level_obs[:, 9:12] = command   # K1: vel_commands at indices 9..11
```

The K1 vision config's PolicyCfg places `velocity_commands` at indices
9..11: the first nine dims are `base_lin_vel (3) + base_ang_vel (3) +
projected_gravity (3)`. So writing `(0.5, 0, 0)` into indices 9..11
makes the policy "see" a forward velocity command of 0.5 m/s.

### 5.5 The exact command flow

```
NaVILA text                                 example: "move forward 75 cm"
   |
   v
get_vel_command(text)                      [0.5, 0, 0], 1.5
   |  (substring-match: "75" in "move forward")
   v
vlm_vel_commands = [0.5, 0, 0],  time_to_go = 1.5
env_steps_to_go  = int(1.5 / 0.02) = 75
target_steps     = num_steps + 75
   |
   v
env.step(torch.tensor([0.5, 0, 0]))  -- happens 75 times in a row
   |
   v
VLNEnvWrapper.step:
   update_command([0.5,0,0]) -> low_level_obs[:, 9:12] = [0.5,0,0]
   low_level_action = policy(low_level_obs)   # 12-dim joint targets
   env.step(low_level_action)
   |
   v
IsaacLab applies joint targets via implicit PD actuators
(scale 0.5, plus K1_DEFAULT_JOINT_POS offset)
   |
   v
Physics steps 4 times at dt=0.005 = 0.02 sim s per policy step
   |
   v
K1 walks 75 * 0.02 = 1.5 s at 0.5 m/s -> ~0.75 m of forward motion
```

`get_vel_command` is intentionally simple (substring matching on
NaVILA's output, looking for "75"/"50"/"25"/"45"/"30"/"15") — the
benchmark assumes NaVILA emits in this discrete set. Numbers outside
the recognized set fall to the bucket's default duration.

### 5.6 Metrics

Defined in `omni/isaac/vlnce/utils/measures.py`:

- **PathLength (PL)** — cumulative Euclidean distance walked.
- **DistanceToGoal (NE)** — for the current position, distance to the
  closest ground-truth waypoint plus the remaining waypoint chain to
  the goal. Uses a KDTree over `episode["gt_locations"]`.
- **Success (SR)** — `is_stop_called AND DistanceToGoal <
  episode["goals"][0]["radius"]`. The success radius for R2R-CE is
  **3 m** (configured per-episode in the dataset).
- **SPL** — `SR * (start_end_episode_distance / max(start_end_episode_distance,
  agent_episode_distance))`. Penalises long paths to the goal.
- **OracleNavigationError (ONE)** — minimum DistanceToGoal observed at
  any point along the trajectory.
- **OracleSuccess (OSR)** — `ONE < success_radius`.

`is_stop_called` is set to True at two points:
1. Inside `navila_eval`'s main loop when `env_steps_to_go == 0` (i.e.
   the latest VLM output was `"stop"` or anything that
   `get_vel_command` mapped to a 0 s duration).
2. Inside `VLNEnvWrapper` if the robot stayed in the same position for
   1000 consecutive sim steps.

So **success = NaVILA said stop AND we're within 3 m of the goal**.

---

## 6. THE LOCOMOTION POLICY (how the robot walks)

The locomotion policy is the K1 vision policy trained in legged-loco
(`k1_vision_rough`). It is a feedforward MLP `actor_hidden_dims=[512,
256, 128]` with ELU activations, trained with PPO. The policy's job
is: given proprioception + height scan + a target velocity command,
produce 12 joint-position offsets that make the K1 walk at that
velocity over rough terrain.

### 6.1 Observations (vision policy)

The observation vector is the concatenation, in this exact order
(`PolicyCfg` in `k1_low_vision_cfg.py`):

| slice    | term                 | dim  | meaning                                                                    |
| -------- | -------------------- | ---: | -------------------------------------------------------------------------- |
| `[0:3]`  | `base_lin_vel`       |    3 | Trunk linear velocity in body frame (m/s).                                 |
| `[3:6]`  | `base_ang_vel`       |    3 | Trunk angular velocity in body frame (rad/s).                              |
| `[6:9]`  | `projected_gravity`  |    3 | Gravity vector projected into body frame. Tells the policy "which way is down". |
| `[9:12]` | `velocity_commands`  |    3 | The target (vx, vy, vyaw). This is what NaVILA writes into.                |
| `[12:24]`| `joint_pos`          |   12 | Joint positions relative to default (radians).                             |
| `[24:36]`| `joint_vel`          |   12 | Joint velocities (rad/s).                                                  |
| `[36:48]`| `actions`            |   12 | The previous policy output (last action), passed back for stability.       |
| `[48:208]`| `height_scan`       |  160 | A 16x10 grid of (sensor_z - hit_z) for points beneath the robot, clipped to ±1.0 m. See §6.6. |

**Total: 208 dims** for the vision policy. (The blind variant
`k1_base` drops `height_scan`, giving 48 dims.)

The corruption flag adds uniform noise at runtime:
`base_lin_vel ±0.1`, `base_ang_vel ±0.2`, `gravity ±0.05`,
`joint_pos ±0.01`, `joint_vel ±1.5`. The actor sees noisy obs (sim2real),
but the critic gets the clean version.

### 6.2 Actions

```python
ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )
```

The policy outputs 12 numbers in (roughly) `[-1, 1]`. Each is scaled
by 0.5 and added to the joint's default position to get the target
position fed to the implicit PD actuator. So the policy can move each
joint up to ±0.5 rad around its default.

The 12 joints are the K1's six leg DoFs per leg, after `merge_fixed_joints=True`
collapsed the arms and head into Trunk:

```
Left  Leg: Left_Hip_Pitch,  Left_Hip_Roll,  Left_Hip_Yaw,
           Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll
Right Leg: same, with Right_ prefix
```

Actuators are split into two groups:
- **hips_knees**: `effort_limit=200 Nm, velocity_limit=20 rad/s,
  stiffness=350, damping=10`.
- **ankles**: `effort_limit=100 Nm, velocity_limit=20 rad/s,
  stiffness=250, damping=5`.

Default standing pose: `Hip_Pitch=-0.15, Knee_Pitch=+0.30,
Ankle_Pitch=-0.15` (slight squat for stability).

### 6.3 Rewards (`K1Rewards` extends Isaac Lab's `RewardsCfg`)

| Term                           | Weight     | What it does                                                                                       |
| ------------------------------ | ---------: | -------------------------------------------------------------------------------------------------- |
| `termination_penalty`          |   `-200.0` | Big negative when the episode terminates (fall, time-out). The "don't die" signal.                |
| `track_lin_vel_xy_exp`         |   `+1.0`   | `exp(-(cmd_vxy - actual_vxy)^2 / 0.5^2)` in yaw frame. Main reward for matching the commanded forward speed. |
| `track_ang_vel_z_exp`          |   `+1.0`   | Same shape but for yaw rate in world frame.                                                       |
| `feet_air_time` (biped)        |   `+0.25`  | Rewards each foot spending close to 0.4 s in the air when the robot is commanded to move. Keeps the gait alive instead of shuffling. |
| `feet_slide`                   |   `-0.25`  | Penalises foot velocity when in contact. Discourages floor-skating.                                |
| `dof_pos_limits` (ankles)      |   `-1.0`   | Penalises ankles hitting their joint limits.                                                       |
| `joint_deviation_hip` (yaw/roll)| `-0.2`     | Penalises Hip_Yaw and Hip_Roll drifting from default. Without this the policy splays the hips outward to "ride low". |
| `action_rate_l2`               |  `-0.005`  | Penalises jittery actions. Encourages smooth motion.                                              |
| `flat_orientation_l2`          |   `-1.0`   | Penalises trunk roll/pitch (keep the body upright).                                               |
| `dof_acc_l2`                   | `-1.25e-7` | Tiny penalty on joint acceleration. Smoothness.                                                   |
| `dof_torques_l2`               |    `0.0`   | Disabled (set to 0 in `__post_init__`).                                                            |
| `undesired_contacts`           |   `None`   | Disabled.                                                                                          |
| `lin_vel_z_l2`                 |   `None`   | Disabled (no z-vel penalty — would prevent jumping over obstacles).                                |

The `track_lin_vel_xy_exp` term uses `std=0.5`, so the reward halves
when the velocity error is 0.5 m/s. The `track_ang_vel_z_exp` uses
`std=0.5` rad/s.

### 6.4 Termination conditions

```python
TerminationsCfg:
    time_out      = DoneTerm(time_out=True)             # episode_length_s = 20.0
    base_contact  = DoneTerm(illegal_contact,
                              params={body_names="Trunk", threshold=1.0})
    bad_orientation = DoneTerm(bad_orientation,
                                 params={limit_angle=1.3})  # ~75°
```

So an episode ends when the timer expires (20 s), the trunk takes >1 N
of contact force (it touched the ground or a wall), or trunk roll/pitch
exceeds 1.3 rad (~75°). The 1.3 rad threshold is intentionally relaxed
from 0.8 because 0.8 rad fired during normal walking strides and was
killing 61% of training episodes inside 2 s.

### 6.5 Domain randomization (`EventCfg`)

| Term                            | When     | What                                                                                |
| ------------------------------- | -------- | ----------------------------------------------------------------------------------- |
| `physics_material`              | startup  | Sets foot-ground friction (static 0.8, dynamic 0.6). Combined with terrain's 1.0 via `max` rule. |
| `add_base_mass`                 | startup  | Originally ±2 kg on Trunk; **disabled** in `__post_init__` (`= None`) to stabilise early training. |
| `base_external_force_torque`    | reset    | Zero by default; widening this knob is the typical perturbation curriculum.        |
| `reset_base`                    | reset    | Spawns inside a flat patch on the current terrain tile, random yaw `[-π, π]`. Linear-vel jitter `±0.5 m/s`, angular `±0.5 rad/s` (or zero in `__post_init__` override). |
| `reset_robot_joints`            | reset    | Joint positions scaled by `[1.0, 1.0]` (so always at the default).                  |

Crucially, only **friction**, **spawn jitter**, and **noise on the
observations** (set in `PolicyCfg.__post_init__` with
`enable_corruption=True`) are active during training. Mass and external
forces are disabled. This is intentional — start with a stable training
setup and add randomization once the gait is solid.

### 6.6 Terrain (`ROUGH_TERRAINS_CFG`)

A 10×20 grid of 8×8 m terrain tiles, each randomly drawn from:

| Tile type                | Proportion | Parameters                                                                |
| ------------------------ | ----------: | ------------------------------------------------------------------------- |
| `pyramid_stairs`         | 20%         | Step height 5-20 cm, step width 30 cm, 3 m platform.                      |
| `pyramid_stairs_inv`     | 20%         | Same, inverted (going down).                                              |
| `boxes` (random grid)    | 20%         | Grid 0.45 m, box height 5-15 cm. Random elevation.                        |
| `random_rough` (HF)      | 20%         | Heightfield noise ±2-8 cm.                                                |
| `hf_pyramid_slope`       | 10%         | Slopes 0-30%, 2 m platform.                                               |
| `init_pos` (obstacles)   | 10%         | Up to 10 fixed-height (1.5 m) discrete obstacles, width 0.3-1.5 m. The "walk past pillars" tile. |

Each sub-terrain registers a `FlatPatchSamplingCfg` so `reset_base` can
spawn the robot on a flat patch within the tile. The `max_init_terrain_level=5`
limit + curriculum decides how difficult the spawned tile is.

### 6.7 The RayCaster height scan

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Trunk",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    mesh_prim_paths=["/World/ground"],   # or "/World/matterport" at eval time
)
```

A virtual sensor that sits 20 m above the trunk and casts a 16 × 10
grid of rays straight down (16 along x, 10 along y, 10 cm spacing → a
1.6 m × 1.0 m patch centered on the robot's xy). Each ray returns the
height of the ground beneath that point. `attach_yaw_only=True` means
the grid stays world-aligned for x/y, only rotating with yaw —
otherwise pitch/roll would smear the grid as the body tilts.

The observation `height_scan` returns `(sensor_z - hit_z)` for each of
the 160 grid cells, clipped to ±1.0 m. From the policy's perspective:
"how far below me is the floor right here? if 0, flat; if 0.2, a 20 cm
drop; if -0.2, a 20 cm step up I need to step over."

This is the "what does the policy see?" question for terrain. It does
**not** see the camera. The RGB camera in the benchmark scene exists
only for NaVILA — the locomotion policy is blind to it.

---

## 7. KEY CONSTANTS CHEAT SHEET

| Constant                  | Value                       | Lives in                              | Meaning                                                                   |
| ------------------------- | --------------------------- | -------------------------------------- | ------------------------------------------------------------------------- |
| `NUM_FRAMES`              | `8`                         | `navila_k1_bridge.py`                  | Frames per NaVILA inference (paper-spec "8f" model).                       |
| `FORWARD_SPEED`           | `0.5 m/s`                   | `navila_k1_bridge.py`                  | Paper-spec forward velocity (§II-B).                                      |
| `TURN_SPEED`              | `π/6 rad/s ≈ 0.524`         | `navila_k1_bridge.py`                  | Paper-spec turn rate.                                                     |
| `ACTION_DURATION`         | `1.5 s`                     | `navila_k1_bridge.py`                  | Legacy fallback for unparsed actions.                                     |
| `SEND_HZ`                 | `20 Hz`                     | `navila_k1_bridge.py`, LiveActuator    | Move() send rate on the K1.                                               |
| `DEFAULT_PORT`            | `5555`                      | `navila_protocol.py`                   | TCP port for desktop-laptop link.                                         |
| `MAX_JSON_BYTES`          | `1 MiB`                     | `navila_protocol.py`                   | Header size cap.                                                          |
| `MAX_BLOB_BYTES`          | `16 MiB`                    | `navila_protocol.py`                   | JPEG blob size cap.                                                       |
| `vx_max` / `vy_max` / `vyaw_max` | `0.4 / 0.15 / 0.4`   | `SessionConfig`, real-robot CLI       | Conservative on-robot safety caps.                                        |
| Sim caps                  | `0.6 / 0.3 / 0.6`           | `walking_loop` CLI                     | Looser caps in sim where the floor is clean.                              |
| `per_step_time`           | `25 s`                      | `SessionConfig`                        | Per-sub-step time limit.                                                  |
| `proximity_threshold`     | `1.0 m`                     | `SessionConfig`                        | "Close enough" for a proximity sub-step.                                  |
| `closest_approach_min`    | `1.5 m`                     | `check_termination`                    | Closest-approach termination only fires after we've been at least this close. |
| `closest_approach_margin` | `0.25 m`                    | `check_termination`                    | Then fires when distance grows by 0.25 m past the minimum.                |
| `turn_tolerance_deg`      | `5°`                        | `check_termination` / walking_loop      | Yaw target hit if we're within this many degrees.                         |
| `turn_kp`                 | `2.0`                       | `apply_controllers`                    | P gain for the turn controller (rad/s per rad).                           |
| `turn_min_vyaw`           | `0.30 rad/s`                | `apply_controllers`                    | Floor on the turn controller's vyaw. Below ~0.25 the gait doesn't rotate. |
| `heading_kp`              | `1.5`                       | `apply_controllers`                    | P gain for heading assist.                                                |
| `drain_seconds`           | `1.5 s`                     | `SessionConfig`                        | Send zero velocity for this long after the last sub-step finishes.        |
| `tick-period`             | `0.4 s`                     | `navila_laptop_relay` CLI              | How often the laptop pushes a frame + polls for state.                    |
| `watchdog_seconds`        | `1.5 s`                     | LiveActuator                            | Zero the cmd if planner update is older than this.                        |
| `frame_buffer_period`     | `0.4 s`                     | real-robot / sim CLIs                  | How often a head frame is appended to the VLM buffer.                     |
| `steps_per_image`         | `25` (= 0.5 / 0.02)         | `navila_eval.py`                       | Sim ticks between image observations during eval (NaVILA was trained at 0.5 s cadence). |
| `_buf_soft_cap`           | `500 frames`                | `VLMRunner`                            | Memory cap on the rolling buffer.                                         |
| `_buf_recent_keep`        | `50 frames`                 | `VLMRunner`                            | Most-recent frames preserved when compacting.                             |
| `max_new_tokens`          | `256`                       | `_navila_inference`                    | Generation budget. Was 64; got bumped because 64 truncated before the verb. |
| `success_radius`          | `3.0 m` (per episode)       | dataset / `Success` measure            | R2R-CE success criterion: stop-called AND DistanceToGoal < 3 m.           |
| Joint default pose        | `Hip_Pitch=-0.15, Knee=0.30, Ankle_P=-0.15` | k1 cfgs                | Slight squat for stability.                                               |
| Action scale              | `0.5`                       | k1 ActionsCfg                          | Policy output is multiplied by 0.5 before being added to the joint default. |
| Decimation                | `4`                         | k1 cfgs                                | 4 physics steps per policy step.                                          |
| `sim.dt`                  | `0.005 s`                   | k1 cfgs                                | 200 Hz physics → 50 Hz control.                                           |
| `episode_length_s` (training) | `20.0 s`                | k1 cfgs                                | Training episode length.                                                  |
| `bad_orientation`         | `limit_angle=1.3 rad`       | k1 cfgs                                | ~75° trunk tilt termination.                                              |
| `base_contact` threshold  | `1.0 N`                     | k1 cfgs                                | Trunk contact force termination.                                          |
| Height scan grid          | `1.6 × 1.0 m, 0.1 m spacing → 16×10 = 160 rays` | k1 cfgs    | What the vision policy sees of the floor.                                  |
| Camera FOV                | horizontal aperture `54.0` mm, 512×512 RGB | k1 NaVILA-Bench cfg   | What NaVILA sees.                                                         |
| PPO hyperparams           | `lr=1e-3 (adaptive), γ=0.99, λ=0.95, clip=0.2, entropy=0.005, n_envs=4096, steps_per_env=32, kl_target=0.01, max_iter=2000` | k1 cfg | Standard locomotion config.                                              |

---

## 8. WHAT CAN GO WRONG (DEBUGGING GUIDE)

### Robot doesn't move

- **NaVILA emits `unparsed -> stop`.** Check the server log for the
  `raw=...` field. If it's "I see a hallway." with no command verb,
  the model is hallucinating descriptions. Common causes: (a) bad
  camera feed (see "camera frames black" below), (b) prompt mismatch
  (verify `build_prompt` byte-for-byte against `run_navigation.py`),
  (c) `max_new_tokens` too small (256 should be fine).
- **`drain_deadline` already fired.** Server is in drain; `state.tag
  == "DRAIN"`. Means all sub-steps completed but you didn't notice.
  Look at `all_done` / `drain_done`.
- **LiveActuator watchdog tripped.** `cmd_set_at` is older than 1.5 s.
  Cause: planner stopped updating. Check the relay's `link_age` in the
  HUD.
- **Robot is in `kPrepare` not `kWalking`.** Only LiveActuator switches
  to `kWalking`; PrintActuator and DryRunActuator don't. Confirm
  `--mode live`.
- **Caps clipped everything to ~zero.** Check `vx_max`, `vyaw_max`.

### Robot walks but in the wrong direction

- **Coordinate convention.** Body +X is forward, +Y is left, yaw left
  is positive. If the K1 yawed when you said "forward", check that
  NaVILA actually emitted `move forward N` and not `turn left N`. The
  HUD's `raw` field will tell you.
- **Heading assist overlay is too aggressive.** With
  `heading_assist=True` and a target off to the side, vyaw can fight
  vx. Try `--heading-kp 0.5` or turn it off.
- **Scene targets coordinate frame mismatch.** Real robot scene
  targets only matter if you also have a pose source; otherwise they
  do nothing.
- **Turn controller chose the wrong direction.** `turn_kp` sign is
  derived from `yaw_unwrap` accumulation. If yaw is being measured in
  the opposite handedness, the controller chases the wrong way. Check
  `yaw_from_quat` results against the actual heading.

### Robot walks but never stops

- **Time limit too generous.** `per_step_time=25 s` per sub-step. For
  short hops drop it to 10.
- **NaVILA never emits `stop`.** The model decides this. If the
  proximity / yaw / time-limit terminators are all disabled (no pose,
  no scene target, no turn target), only NaVILA's stop or the time
  limit can end the sub-step.
- **`closest_approach_min` not yet reached.** The closest-approach
  trigger requires the robot to get within 1.5 m before it can fire.
  If `proximity_threshold` is 0.3 m, you may have walked past at 0.5
  m and the trigger never fires until `closest_approach_margin`
  detects retreat.
- **In the benchmark, `is_stop_called` never gets set.** Check that
  `get_vel_command` is mapping the NaVILA text to `time_to_go=0`
  (only "stop" does). Otherwise `env_steps_to_go` will be `>0` every
  cycle.

### Robot falls over immediately

- **Init pose collides with the world.** In the benchmark, K1 spawns
  at `z=start_position_z + 0.55` (matches the K1 standing height). If
  the scene has a step there, the robot drops into a wall. Check
  `episode["start_position"]`.
- **`bad_orientation` fires too early.** If the trunk tilts > 1.3 rad
  it terminates. Originally the eval used 0.8 rad which was firing
  mid-stride — both training and eval now use 1.3 rad. If you see
  early terminations, double-check both.
- **`base_contact` on a merged-arm body.** `merge_fixed_joints=True`
  collapses arms into Trunk, so if the merged-arm geometry clips a
  wall the contact sensor fires base_contact and the env resets in a
  loop. The k1_matterport vision config notes this and does NOT add
  base_contact for that reason. If you re-enabled it and see
  resets-in-place, that's why.
- **PD gains too aggressive for the morphology.** Check actuator
  stiffness — hips_knees=350, ankles=250. Lowering damping by 50%
  often helps if the policy is twitchy.
- **Bad weights.** The 235→12 K1 sim2real policy (used by the
  walking_loop demo) needs a specific training run. Confirm
  `load_run=...` points at a converged checkpoint, not the zero-action
  stub mentioned in `k1_matterport_base_cfg.py`.

### NaVILA outputs garbage

- **Prompt mismatch.** The model was fine-tuned on a specific prompt
  string. The conformance test `test_prompt_matches_reference_run_navigation`
  checks byte-for-byte against `run_navigation.py`.
- **Frame order broken.** If you accidentally feed `current` first and
  `historical` last, the model loses its temporal cue. The reference
  is "first 7 historical + 1 current at the end".
- **First frame missing from the buffer.** Paper §II-A invariant — if
  the `np.linspace(..., endpoint=False)` is replaced by `endpoint=True`
  or some other recipe, frame 0 disappears. Check
  `VLMRunner.sample_frames`.
- **Black-frame pollution.** When the buffer has fewer than 8 frames,
  the benchmark's `sample_images_and_send_to_vlm` pads with black at
  the front. The "Fix B" warmup loop in `navila_eval.py` steps the
  env for 8 image-cadence cycles at zero velocity before the main
  loop starts, so the buffer never has fewer than 8 real frames when
  we ask NaVILA. If you removed the warmup, expect "turn left 45°"
  for the first 10 commands.
- **Camera FOV wrong.** `horizontal_aperture=54.0` in
  `PinholeCameraCfg`. Wide-angle (100+) and very narrow (≤20)
  distributions are out-of-distribution for NaVILA's fine-tuning data.
- **Frames flipped.** Look at the saved `head_view.mp4`. If the image
  is upside-down (common after `merge_fixed_joints` because the camera
  rotation `xyaxes` in MJCF doesn't match Isaac Sim's), NaVILA sees
  the floor as the ceiling.

### Camera frames are black

- **ZED not connected / wrong resolution.** `pyzed` raises in
  `ZEDImageSource.__init__`. Test with `--image-source static` first.
- **MJPEG URL wrong.** `MJPEGImageSource` looks for `\xff\xd8 ...
  \xff\xd9` (JPEG SOI/EOI). If the URL serves something else
  (multipart boundary not found), it raises.
- **Isaac Sim camera not enabled.** Pass `--enable_cameras` to
  navila_eval (the benchmark runner does).
- **Headless render with no display server.** Set `MUJOCO_GL=egl`
  (the loops do this automatically) and `__GL_PLATFORM=egl`.
- **In sim, camera prim mounted on a body that doesn't exist.** The
  K1 vision benchmark cfg mounts on `Trunk/rgb_camera` (because the
  head was merged). If you didn't `merge_fixed_joints` the URDF, the
  path doesn't resolve.

### Benchmark SR is 0%

- **`is_stop_called` never fires.** See "Robot walks but never stops".
- **Robot terminates by fall before reaching goal.** Check the
  measurements JSON — `success=0` with high `path_length` suggests
  fall. Watch the videos in `eval_results/.../videos/`.
- **K1 policy is the zero-stub.** `k1_matterport_base_cfg.py`
  explicitly mentions a zero-action stub for early development;
  swapping it for a real trained policy is required.
- **Goal radius too tight.** R2R-CE uses 3 m; some episodes may have
  goals inside furniture or behind a wall the robot can't path around.
- **Camera mounted wrong / occluded.** Inspect the recorded
  `output_<idx>.mp4` — if the head camera is pointing at the floor or
  ceiling, NaVILA can't see anything useful.

### Training reward doesn't improve

- **Termination penalty dominating.** `-200` for any termination is
  large. If the policy is hitting `bad_orientation` every episode,
  it's stuck. Lower the limit angle gradually or temporarily disable
  termination_penalty.
- **Commands ranges too wide.** The K1 cfg `__post_init__` overrides
  `lin_vel_x = (0.0, 1.0), lin_vel_y = (0.0, 0.0), ang_vel_z = (-1.0,
  1.0)`. If you accidentally allowed backward walking or large lateral
  commands, the policy never finds a stable basin.
- **Standing reward absent.** The `rel_standing_envs=0.02` of
  `UniformVelocityCommandCfg` means 2% of envs are commanded zero
  velocity. If you increase this without adding a "stand still"
  reward, the agent learns to stand and never starts walking.
- **Action rate too lax.** `action_rate_l2 = -0.005`. Jittery policies
  rack up zero progress. If reward is flat and actions look like
  noise, double the penalty.

### Training reward plateaus early

- **No curriculum.** `terrain_levels` is configurable. If
  `terrain_generator.curriculum=False` (default in
  `K1VisionRoughEnvCfg`), the policy never sees harder terrain. Turn
  it on once the gait is stable.
- **Domain randomization disabled.** `add_base_mass=None` is the
  current default. Once basic walking works, re-enable mass perturbation
  and `base_external_force_torque` to break the policy out of brittle
  local optima.
- **Reward shaping in conflict.** `feet_air_time` rewards swinging
  feet, `feet_slide` penalises sliding contacts. If the gait happens
  to swing-and-slide simultaneously the two cancel. Inspect per-term
  rewards in the run's logs.
- **Velocity tracking std too narrow.** `std=0.5` halves the reward
  at 0.5 m/s error. If `lin_vel_x` ranges are (0, 1.0) the agent only
  ever sees up to 0.5 m/s reward for tracking — manageable, but if you
  widen the cmd range without widening `std`, the policy learns to
  stand still.

---

## 9. WHAT TO IMPROVE NEXT (RANKED)

1. **Train a real K1 vision policy and stop using the zero-stub in
   benchmark eval.** *Limitation:* `k1_matterport_base_cfg.py`'s
   docstring explicitly mentions a zero-action stub.
   `k1_matterport_vision_cfg.py` is wired for a real policy but the
   default `experiment_name="k1_vision_rough"` checkpoint may not be
   converged. *Improvement:* Run `train.py --task k1_vision` to ~2000
   iterations (the configured `max_iterations`), evaluate in
   `legged-loco/play.py`, then point `navila_eval` at the new
   checkpoint. *Effort:* days (training time-bound). *Relevant:*
   the NaVILA paper Table IV "Vision" row; the H1 vision config this
   was cloned from is known to converge in ~12 hours on a 5090.

2. **Add a pose source so heading-assist and proximity termination
   work on the real robot.** *Limitation:* Real-robot pose is None by
   default; only "stop" and time-limit can end a sub-step. *Improvement:*
   Subclass `PoseSource` to either (a) subscribe to K1 SDK
   `LowStateMsg` for IMU yaw, (b) integrate ZED visual odometry, or
   (c) plug in an external SLAM stack. *Effort:* 1-2 days for SDK
   yaw; 1-2 weeks for VO/SLAM. *Relevant:* ORB-SLAM3 or DROID-SLAM
   for monocular VO; ZED SDK exposes positional tracking natively
   (`Camera.enable_positional_tracking`).

3. **Replace substring matching in `get_vel_command` with `parse_action`.**
   *Limitation:* The benchmark's `eval_utils.get_vel_command` does
   crude `"75" in text` matching that confuses "75 cm" with "175 cm"
   and ignores any number not in {15,25,30,45,50,75}. *Improvement:*
   Import `parse_action` from `navila_k1_bridge` and use the same
   regex. *Effort:* 1 hour. *Relevant:* the unit tests in
   `test_action_parser.py` already verify the regex against the
   paper-spec.

4. **End-to-end episode replay for offline debugging.** *Limitation:*
   When a real-robot run misbehaves, the ZED frames + planner state
   are gone. *Improvement:* Write JPEGs + a state-trace JSONL to disk
   from the laptop relay; replay them through the planner in
   `navila_k1_realrobot.py --image-source dir` to reproduce the bug
   without the robot. *Effort:* half a day. *Relevant:* `DirReplayImageSource`
   already exists; we just need the recording counterpart.

5. **NaVILA inference latency optimisation.** *Limitation:* 400-1000
   ms per inference at the desktop means the planner updates the cmd
   every ~0.5-1.0 s, but the K1 walks at 0.5 m/s — that's 25-50 cm of
   travel between decisions. *Improvement:* (a) batch inference if
   multiple sessions, (b) quantize NaVILA to INT8 (vLLM/AWQ), (c) prune
   the LLaMA-3-8B backbone via VILA-Lite. *Effort:* 1-2 weeks for
   quantization. *Relevant:* VILA-1.5 8-bit; LLaVA-NeXT INT4.

6. **Closed-loop integration with the trained policy on the real
   robot.** *Limitation:* The real-robot path uses Booster's built-in
   walker (`B1LocoClient.Move`). The trained sim2real policy lives in
   booster_deploy but isn't wired into the laptop relay. *Improvement:*
   Build a `PolicyActuator` that runs the 235→12 MLP in the laptop
   process (or on the K1 onboard PC), feeds joint targets via SDK.
   *Effort:* 1 week. *Relevant:* `MujocoController.policy_step` shows
   the obs construction.

7. **Better scene grounding than `DEFAULT_SCENE_TARGETS`.** *Limitation:*
   The current proximity termination relies on a hardcoded
   `{"red box": (3.0, 0.0, 0.30), ...}` mapping. Real homes don't have
   colored cubes at known coordinates. *Improvement:* Replace with a
   live object detector (Grounding-DINO or OWL-ViT) querying NaVILA's
   instruction for nouns, projecting hits into the robot's frame via
   ZED depth. *Effort:* 1 week. *Relevant:* OWL-ViT, GroundingDINO,
   the FoundationPose-style RGB-D grounding line.

8. **Add a "rotate-in-place to align" sub-step before "walk forward"
   when a target is named.** *Limitation:* On `"walk to the red box"`,
   if the K1 starts facing away, NaVILA emits forwards and the planner
   blindly walks the wrong way until the time-limit fires. *Improvement:*
   When `parse_substeps` sees a proximity target and pose is available,
   inject a synthetic `"turn ... to face X"` sub-step at the front,
   targeting `atan2(ty-ry, tx-rx) − yaw`. *Effort:* 1 day. *Relevant:*
   the existing heading-assist controller already does this within a
   sub-step; we'd be doing it as its own sub-step.

9. **Train with the same scene generator as the benchmark
   (Matterport).** *Limitation:* The k1_vision policy is trained on
   procedurally-generated rough terrain (stairs, slopes, boxes), but
   evaluated on Matterport interiors which have walls, doorways, and
   tight corners the training distribution never showed. *Improvement:*
   Add a Matterport-mixed training regime — randomly load a Matterport
   scene as terrain for some envs. *Effort:* 2-3 weeks (need the
   physics material + collision optimization Matterport USD assets
   for many envs). *Relevant:* the VLN-CE-Isaac dataset organisation;
   benchmarks usually train on a mix of synthetic and real-scene
   terrain.

10. **Memory-aware long-horizon prompting.** *Limitation:* NaVILA's 8
    frame context is mostly fresh observations; once the episode runs
    longer than ~5 s, the "historical" frames are too dense in time to
    capture room-scale memory. *Improvement:* Move to the 16-frame
    NaVILA variant (or fine-tune one), or implement an
    episodic-memory token (CLIP embedding of a key past frame) and
    splice it into the prompt. *Effort:* 1-2 weeks if the 16f
    checkpoint exists; months if we have to fine-tune. *Relevant:*
    the NaVILA paper's Table IX shows the 8f/16f trade-off; LLaVA-NeXT-Video
    uses up to 32 frames.

---

## 10. THINGS I NOTICED

**Dead code / inconsistencies.**

- `navila_k1_bridge.py:36` — `DEFAULT_CKPT = Path.home() /
  "Projects/booster/NaVILA/checkpoints/..."` (missing `k1_research`
  segment). Other files use the correct path. The bridge's own
  `main()` would crash on a fresh clone if you ever ran it.
- `navila_mujoco_loop.py:57-58` — `K1_XML` and `DEFAULT_CKPT` likewise
  point at `Projects/booster/...` instead of `Projects/k1_research/booster/...`.
  This file is the older "no walking" sim; the walking_loop replaced
  it but the bug wasn't backported.
- `demo_planner.py` is a planner stub left over from the
  go2_matterport baseline. It computes velocity from an expert
  trajectory with a PID, not from NaVILA. Unused by the K1 path —
  consider deleting or moving to a `legacy/` folder so it doesn't
  confuse readers.
- `wrappers.py:174` — `elif "h1" or "g1" in self.task_name:` is a
  Python truthy bug: `"h1"` is always truthy, so this branch always
  matches when not `"go2"`. `K1` falls into the 200-step warmup
  bucket by accident. Should be `elif "h1" in self.task_name or "g1"
  in self.task_name:`.
- `navila_eval.py:323` — `max_episode_steps = 100 * 0.5 / (dt *
  decimation)` evaluates to `100 * 0.5 / 0.02 = 2500` sim steps, i.e.
  50 sim seconds. Reads at first as "100 steps" — the multiplier
  hides what the bound actually is.

**Hardcoded values that should be configurable.**

- `DEFAULT_SCENE_TARGETS` in `navila_k1_core.py` has very specific
  coordinates. Fine for the MuJoCo demo, but a future per-scene YAML
  config would be cleaner.
- `head_buffer_period = 0.4` in walking_loop is hardcoded inside
  `main()` (not exposed via CLI). The real-robot path exposes
  `--frame-buffer-period`.
- The benchmark's `get_vel_command` has six magic numbers
  (15/25/30/45/50/75) baked in. These match `parse_action`'s
  paper-spec but are independently maintained — easy to drift.
- VLNEnvWrapper warmup_steps (50 default, 100 for go2, 200 for h1) is
  picked by string match on the task name. The K1 falls into the 200
  bucket by accident (see "wrappers.py:174" note); but even with the
  bug fixed, the value should live in the task cfg.

**Potential bugs not yet found.**

- `_navila_inference` returns the full assistant turn including any
  preamble. The parser only finds the **first** match for each verb,
  so `"turn left 30 degrees, then turn right 60 degrees"` would
  silently execute only the left turn. If NaVILA ever emits chained
  actions in one response (it usually doesn't, but it can with rare
  prompts), the planner ignores everything after the first verb.
- `LiveActuator.shutdown()` switches to `kDamping` (collapse) in the
  test mock but `kPrepare` (stay standing) in the production code
  (`navila_k1_realrobot.py:353`). Different behavior on test vs prod
  cleanup. The test description says `kDamping`, production says
  `kPrepare` — operator preference per the comment. This is fine but
  worth flagging: a test that passes does not prove the right cleanup
  mode is used in production.
- The benchmark's `sample_images_and_send_to_vlm` does
  `int(i * (num_images - 1) / 7)` for sampling, which gives **exactly**
  the same indices as `np.linspace(0, num_images-1, num=7,
  endpoint=False, dtype=int) + [last]` only when `num_images` is large
  enough. For `num_images=8`, `int(i*7/7) for i in range(7)` gives
  `[0,1,2,3,4,5,6]`, which matches `np.linspace(0,7,7,endpoint=False)=[0,1,2,3,4,5,6]`.
  But for `num_images=10`, `int(i*9/7)` gives `[0,1,2,3,5,6,7]`, vs.
  `np.linspace(0,9,7,endpoint=False)=[0,1,2,3,5,6,7]`. They coincide
  by luck — if the formula or `num_frames` changes, they may diverge.
  Worth replacing with the explicit linspace call to make the
  invariant obvious.
- `RslRlVecEnvHistoryWrapper.get_observations` builds a uniform
  history by stacking the current proprio copy N times
  (`torch.cat([proprio_obs.unsqueeze(1)] * self.history_length, dim=1)`).
  This means immediately after reset, the policy sees a perfectly
  static history — fine for go2 but probably hurts H1/K1 (which need
  temporal cues for stability).

**TODO/FIXME-like comments in the codebase.**

- `k1_matterport_base_cfg.py:14-17` — Notes that a benchmark-style K1
  policy "needs to be trained separately" and that the current
  checkpoint dir contains a zero-action stub. This is the single
  biggest "future work" tag in the repo.
- `navila_k1_walking_loop.py:540-552` — A long inline comment explains
  why pure-turn sub-steps bypass NaVILA: "NaVILA stubbornly emits
  'move forward' most of the time even when asked to turn." Suggests
  the turn-controller workaround is masking a fine-tuning bug in
  NaVILA itself (or a domain shift between the training distribution
  and the K1's head camera).
- `wrappers.py:98` — A commented-out `# print("============== Height
  Map ==============")` left in the step method. Harmless but worth
  removing.
- The `__pycache__` folders are checked in (`booster/NaVILA/llava/...`).
  Probably from a one-time `pip install -e .`; usually a `.gitignore`
  miss.

**Surprising choices that are deliberate.**

- The same `Image.fromarray` round-trip happens on every frame on the
  laptop side (PIL → encode → server → decode → PIL). Pre-encoded
  JPEGs would save a couple of ms per tick but add complexity to the
  image-source abstraction. Probably the right call.
- The desktop server treats `tick` as one bidirectional message
  pair, not a streaming push. This caps the effective frame rate at
  1/(tick latency). The current `tick-period=0.4s` works because
  NaVILA only needs frames at ~2-3 Hz, but if you want >5 Hz frame
  pushes you'd need to split the protocol into a frame stream + a
  separate command poll.
- The settle delay is `time.sleep(0.1)` *plus* the publish of zeros.
  That makes the actual inter-action gap a bit over 100 ms, not
  exactly 100 ms. Combined with NaVILA's variable inference time, the
  cadence is "as fast as the model can run" rather than a fixed
  schedule. This means under heavier inference load the K1 walks
  bursty. A future steadier-cadence mode would help debugging.
- `navila_eval.py`'s `same_pos_count >= 1000` threshold is 1000
  steps × 0.02 s = 20 s of zero motion. Lenient; useful because the
  walking policy sometimes shifts weight without moving the COM, and
  you don't want to bail prematurely.
