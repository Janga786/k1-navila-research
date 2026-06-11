# K1 Real-Robot NaVILA Deploy — Runbook (built-in walking controller)

Verified end-to-end 2026-06-11 (camera path live-tested; robot walked: pending first live session).
Topology: **this lab box runs everything** — VLM server (GPU) + relay (Booster SDK over the wired
LAN, this box = 192.168.10.10, robot = 192.168.10.102). Robot drives via `B1LocoClient.Move()`
(Booster's controller) — the trained policy is NOT in this path.

⚠️ GPU note: the VLM server needs ~17 GB fp16 — do NOT run while Isaac/benchmark is using the GPU.

---
## 0. AFTER EVERY ROBOT REBOOT (or camera returns 503 forever)

The camera publishes NOTHING until the vision service is started manually:

```bash
# from this box (robot wifi IP; pw 123456):
ssh booster@10.218.0.138 'source /opt/booster/BoosterRos2/install/setup.bash && \
  source /opt/booster/BoosterRos2Interface/install/setup.bash && \
  python3 /home/booster/vision_service_call.py start'
# expect:  status=0   (a "local_setup.bash not found" warning above it is benign)
```

Then start the HTTP camera bridge (topic MUST be the mipi head cam — it only appears
after the call above; the bridge's built-in default topic is wrong for this robot):

```bash
ssh booster@10.218.0.138 'printf "%s\n" "#!/bin/bash" \
  "source /opt/booster/BoosterRos2Interface/install/setup.bash" \
  "exec python3 \$HOME/robot_video_bridge.py --topic /boostercamera/head/raw/rgb --port 8080" \
  > /tmp/start_bridge.sh'
ssh booster@10.218.0.138 'setsid bash /tmp/start_bridge.sh > /tmp/bridge.log 2>&1 < /dev/null & echo launched'

# verify (frames should be nonzero, age < 0.2 s):
ssh booster@10.218.0.138 'curl -s http://localhost:8080/status'
curl -s -o /tmp/f.jpg -w 'HTTP %{http_code}\n' http://10.218.0.138:8080/frame.jpg
```

Known-good state: ~10 fps, frame 544×448 (note: calibration was for 1280×720 — undistort is
approximate at this resolution; acceptable. Full-res hunt: /home/booster/x5_camera_rpc.py).

---
## 1. VLM server (this box, terminal 1)

```bash
conda run -n navila python ~/robots/k1/workspace/k1-vlm-navigation/navila_server.py \
  --bind 0.0.0.0 \
  --model-path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f
# fp16, loads in ~1-5 min (weights disk-cached). Leave running.
```

---
## 2. Relay — escalate print → dry → live (this box, terminal 2)

```bash
cd ~/robots/k1/workspace/k1-vlm-navigation

# A) PRINT — nothing moves; verifies camera -> VLM -> parser end to end:
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode print \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost \
  --instruction "walk to the boxes on the floor and stop"

# B) DRY — SDK connects to the robot; Move() calls are LOGGED, not sent:
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode dry \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost --net 192.168.10.102 \
  --instruction "walk to the boxes on the floor and stop"

# C) LIVE — ROBOT WALKS. Floor cleared, robot standing in Prepare (remote), hand on q:
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode live \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost --net 192.168.10.102 \
  --instruction "walk to the boxes on the floor and stop"
```

Multi-step plans chain with `|`:
`--instruction "walk to the chair | turn right 90 deg | walk forward and stop at the wall"`

---
## Safety (live mode)

- **`q` or Ctrl-C = e-stop → robot returns to kPrepare and STAYS STANDING (does not go limp).**
- Velocity caps default conservative (vx 0.4 m/s); leave them.
- No on-robot stall-abort yet: if it wedges against furniture it pushes for up to 25 s
  before re-querying. Keep a clear lane; e-stop early.
- Watchdog: if the link drops, the actuator zeroes the command within ~1.5 s.

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
| `/frame.jpg` 503 forever | vision service not started → step 0 magic call |
| bridge `frames: 0` | wrong topic → must be `/boostercamera/head/raw/rgb` |
| VLM buffer frozen / robot ignores scene | URL must end `/frame.jpg`, never `/stream` |
| relay can't reach SDK | wired LAN: this box 192.168.10.10 ↔ robot 192.168.10.102 must ping |
| robot won't enter walking | put it in Prepare via remote first; check battery in LUI |
