# K1 Real-Robot NaVILA Deploy — Runbook (built-in walking controller)

Rewritten 2026-07-02 after the sim2real readiness audit (supersedes the 2026-06-11
version, which still used the dead-end StartVisionService camera path).
Topology: **this lab box runs everything** — VLM server (GPU) + relay (Booster SDK
over the wired LAN, this box = 192.168.10.10 on `eno1`, robot = 192.168.10.102).
Robot drives via `B1LocoClient.Move()` (Booster's built-in controller) — the trained
policy is NOT in this path.

GPU note: the benchmark is COMPLETE (2026-06-22), so the GPU is free for the
fp16 VLM (~17 GB). `--load-8bit` (~9 GB) is available if anything else needs VRAM.

---
## 0. Camera (one command, after any robot boot)

**Doctrine corrected 2026-07-01/06** (supersedes the old "/booster_video_stream,
no unlock" guidance — evidence: `~/k1_qr_nav_ws/CAMERA.md` +
`CAMERA_STEREONET_ISSUE.md` + on-robot `K1_camera_fix_notes.md`; verified live
2026-07-09 at ~29.5 fps). The X5 head camera **boots into a low-rate idle mode
(~0.4 fps)** — it must be unlocked with **StartVisionService, face detection
OFF** (`noface`; the face-YOLO otherwise starves the Jetson → the old mid-run
"decay"), then bridge the **raw topic `/boostercamera/head/raw/rgb`**
(~28–30 fps, 100% unique, 544×448). `/booster_video_stream` sits behind the
broken StereoNet rectifier and republishes stale JPEGs (10 msg/s but ~0.4
content fps) — do not bridge it.

```bash
bash ~/k1_qr_nav_ws/camera_up.sh   # unlock (noface) + bridge raw topic + verify
# "✅ CAMERA LIVE ... msg_rate ~29.5/s" → done.
# Deploy pulls http://192.168.10.102:8080/frame.jpg (544×448;
#  NAVILA_UNDISTORT=1 + stretch validated on this geometry).
```

Health tiers (measured): **27–30 fps** = ideal fresh unlock · **~10 fps** =
repeated-unlock tier, still fine for NaVILA's ~1 Hz decisions · **<8 fps** =
sick → reboot the robot and **charge the battery** (brownout silently degrades
the camera). camera_up.sh already skips the unlock if the feed is live
(repeated unlocks degrade the X5) and auto-escalates (perception restart →
re-unlock → re-bridge) if the feed is in the decayed state.

| camera symptom | fix |
|---|---|
| feed "frozen"/~0.4 fps content | the per-boot unlock wasn't run → `bash ~/k1_qr_nav_ws/camera_up.sh` (#1 cause) |
| `/frame.jpg` 503 / connection refused | HTTP bridge not running (only the camera service auto-starts) → `camera_up.sh` |
| rate fine, then decayed mid-session | camera_up.sh escalation handles it; if it recurs, charge the battery |
| image garbled | NV12 decode issue — camera_up.sh/bringup pass `--force-type image` for `boostercamera/*` topics |

---
## 1. VLM server (this box, terminal 1)

`prep_frame` (undistort + squarify) runs **server-side**, so the alignment env
vars go on THIS process, not the relay:

```bash
cd ~/robots/k1/workspace/k1-vlm-navigation
NAVILA_UNDISTORT=1 NAVILA_VLM_TRANSFORM=stretch \
~/miniconda3/envs/navila/bin/python navila_server.py \
  --bind 0.0.0.0 \
  --model-path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f
# fp16, loads in ~1-5 min. Add --load-8bit if VRAM is contended (benchmark-blessed).
# (env python directly, NOT `conda run` — conda run buffers stdout so the server
#  looks frozen/silent even while it's loading and serving fine.)
```

Optional preflight without the robot: `conda run -n navila python preflight_vlm.py`.

---
## 2. Relay — escalate print → dry → live (this box, terminal 2)

`--net` is the LOCAL interface NAME (`eno1`), never an IP — passing the robot IP
recreates the FastDDS "whitelist filtered out" blocker.

```bash
cd ~/robots/k1/workspace/k1-vlm-navigation

# A) PRINT — nothing moves; verifies camera → VLM → parser end to end:
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode print \
  --image-source mjpeg --mjpeg-url http://192.168.10.102:8080/frame.jpg \
  --server localhost \
  --instruction "walk to the boxes on the floor and stop"

# B) DRY — SDK connects; Move() calls are LOGGED, not sent. Add --pose odom to
#    also validate odometry (watch the printed pose while you push/rotate the
#    robot base gently — x,y,theta should track):
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode dry \
  --image-source mjpeg --mjpeg-url http://192.168.10.102:8080/frame.jpg \
  --server localhost --net eno1 --pose odom \
  --instruction "walk to the boxes on the floor and stop"

# C) LIVE — ROBOT WALKS. Floor cleared, robot standing in Prepare (remote), hand on q:
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode live \
  --image-source mjpeg --mjpeg-url http://192.168.10.102:8080/frame.jpg \
  --server localhost --net eno1 --pose odom \
  --instruction "walk to the boxes on the floor and stop"
```

Multi-step plans chain with `|`:
`--instruction "walk to the chair | turn right 90 deg | walk forward and stop at the wall"`

### Odometry ladder (do this once, in the first live session)
1. Run live with `--pose odom` but WITHOUT `--turn-controller` (default off).
2. During a straight walk, confirm the HUD/console pose x advances ≈ distance
   walked, and a LEFT turn increases theta (+yaw = left).
3. Only after that checks out, enable `--turn-controller` next run — it closes
   the loop on turns (lands 90° at ±5°) instead of open-loop timing.

---
## Safety (live mode) — what protects you now

- **`q` or Ctrl-C = e-stop → robot returns to kPrepare and STAYS STANDING** (never
  kDamping — a biped goes limp and falls).
- Velocity caps default conservative (vx 0.4 / vy 0.15 / vyaw 0.4); leave them.
- **Server-link watchdog (relay-side, 1.5 s):** if the VLM server stops answering,
  the command is zeroed within `--watchdog-seconds` (the actuator-level watchdog
  alone could not fire while the relay loop lives — fixed 2026-07-02).
- **Camera watchdog (3 s):** if no fresh frame arrives (`--camera-stale-seconds`),
  the command is zeroed — the VLM never navigates a frozen world.
- **Stall-abort (needs `--pose odom`):** commanded motion with < 0.25 m and < 30°
  net progress over 10 s ends the sub-step instead of grinding against furniture
  (was: pushed a wall for up to 25 s). Without odometry this backstop is OFF —
  keep a clear lane and e-stop early.
- **Turns now execute to completion:** "turn right 90 degrees" holds the turn
  ~3.9 s at 0.4 rad/s (was: ~1 s → only ~15° executed). Expect visibly longer,
  deliberate turns.
- Watchdog inside the actuator additionally zeroes Move() if the planner thread
  itself stops updating.

## First-session instruction ladder

1. "Walk forward and stop at the wall." / "Turn left and walk to the door."
2. "Walk to the chair and stop in front of it."   ← the arrival-recognition money test
3. "Walk past the desk, turn left, and stop near the doorway."
4. Probes: "Walk to the doorway and stop inside it." / "Walk toward me and stop."
Avoid: backward motion, >3–4 m per clause, small/high/reflective targets.
Give each instruction 30–60 s; ~1 decision/sec; some milling is normal.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `/frame.jpg` 503 forever | `booster-video-stream` service down → restart it on the robot (see §0; NO StartVisionService, NO reboot) |
| bridge `frames: 0` | wrong topic → must be `/booster_video_stream` |
| VLM buffer frozen / robot ignores scene | URL must end `/frame.jpg`, never `/stream`; also check the new stale-frame guard messages in the relay log |
| relay can't reach SDK / "whitelist filtered out" | `--net eno1` (interface NAME); wired LAN: this box 192.168.10.10 ↔ robot 192.168.10.102 must ping |
| robot won't enter walking | put it in Prepare via remote first; check battery in LUI |
| live mode runs, decisions stream, but robot never moves | robot wasn't STANDING in Prepare when live launched → kWalking silently refused. Since 2026-07-09 the launcher verifies via GetMode and aborts with instructions; bring the robot to Prepare (remote) and relaunch |
| camera decays mid-session after ~40 min, all software fixes fail | battery brownout — charge the robot; next boot: `camera_up.sh` once (its escalation now waits up to 180 s after a perception restart before re-unlocking) |
| pose stays `(?, ?)` with `--pose odom` | odometry subscriber got no message — SDK link problem; re-check `--net eno1` and that the actuator init succeeded first |
| turns overshoot/oscillate with `--turn-controller` | odom theta sign/scale not validated yet — go back to the odometry ladder step 2 |
