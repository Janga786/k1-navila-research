# Test NaVILA on the physical K1 — step by step (2026-06-11)

Robot uses **Booster's built-in walking controller** (not the trained policy). Everything
runs on the lab box: the NaVILA VLM server (GPU) + the relay (talks to robot over the
wired LAN). You only ever touch this box + the robot's e-stop.

Helper scripts live in `~/robots/k1/workspace/k1-vlm-navigation/`:
- `reboot_robot.sh`   — reboot robot, wait until back (fresh camera window)
- `bringup_camera.sh` — unlock + start camera, **measure fps**, tell you go/no-go

Use **two terminals** on the lab box.

---
## Terminal 1 — VLM server (leave it running the whole session)

Check if it's already up:
```bash
python3 -c "import socket; socket.create_connection(('localhost',5555),timeout=2).close(); print('VLM server UP')" || echo "DOWN — start it below"
```
If DOWN, start it (loads in ~2–5 min; wait for `listening on 0.0.0.0:5555`):
```bash
cd ~/robots/k1/workspace/k1-vlm-navigation
conda run -n navila python navila_server.py --bind 0.0.0.0 \
  --model-path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f
```

---
## Terminal 2 — everything else

### Step 1 — Camera (this is the finicky part; verify fps before trusting it)

The head camera runs ~10 fps for about an hour after boot, then decays. Bring it up and
**measure**:
```bash
cd ~/robots/k1/workspace/k1-vlm-navigation
bash bringup_camera.sh
```
- Prints `✅ CAMERA OK: <fps>` → continue to Step 2.
- Prints `❌ CAMERA TOO SLOW/DEAD` → reboot and retry:
  ```bash
  bash reboot_robot.sh        # ~2 min
  bash bringup_camera.sh      # should now show ~10 fps
  ```
- Sanity-check the saved frame shows your room:
  ```bash
  xdg-open /tmp/k1_cam_check.jpg
  ```

### Step 2 — PRINT test (NOTHING MOVES — verifies camera → VLM → parser)

```bash
cd ~/robots/k1/workspace/k1-vlm-navigation
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode print \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost \
  --instruction "walk to the boxes on the floor and stop"
```
You should see one decision per second, e.g. `raw='The next action is move forward 75 cm.'`,
and the `vx/vyaw` it parses to. Point the robot at something it can act on (open space / a
landmark) and confirm the decisions make sense. **Ctrl-C to stop.**

### Step 3 — DRY test (SDK connects to robot; commands LOGGED, not sent)

Same command, `--mode dry` and add `--net 192.168.10.102`:
```bash
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode dry \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost --net 192.168.10.102 \
  --instruction "walk to the boxes on the floor and stop"
```
Look for `B1LocoClient initialised`. (A `whitelist interfaces were filtered out` warning may
print — see Troubleshooting; the SDK still inits.) **Ctrl-C to stop.**

### Step 4 — LIVE test (ROBOT WALKS)

Before running: **clear a lane**, get the robot **standing** (Booster remote / LUI app → it
should be upright and balancing, not sitting), and keep a hand on the e-stop.
```bash
~/envs/navilaenv/bin/python navila_laptop_relay.py --mode live \
  --image-source mjpeg --mjpeg-url http://10.218.0.138:8080/frame.jpg \
  --server localhost --net 192.168.10.102 \
  --instruction "walk to the boxes on the floor and stop"
```
**E-STOP: press `q` in the HUD window, or Ctrl-C.** Either zeroes the command and returns the
robot to `kPrepare` — it **stays standing, does not go limp.**

Safety notes:
- Speed caps default conservative (vx ~0.4 m/s). Leave them.
- No on-robot stall guard yet: if it wedges against furniture it may push for up to ~25 s
  before re-deciding. Keep the lane clear; e-stop early.
- If the link drops, the actuator watchdog zeroes the command within ~1.5 s.

Multi-step plans chain with `|`:
`--instruction "walk to the chair | turn right 90 deg | walk forward and stop at the wall"`

---
## Good first instructions (easy → harder)

1. "Walk forward and stop at the wall."
2. "Walk to the chair and stop in front of it."      ← arrival test (the money shot)
3. "Walk past the desk, turn left, and stop near the doorway."
4. "Walk toward me and stop."  /  "Walk to the doorway and stop inside it."

Use **large floor-standing** targets (camera is at ~0.78 m — tabletop items are out of
frame). Avoid backward motion, >3–4 m per clause, and small/shiny targets. Give each
instruction 30–60 s; some milling before a clean stop is normal.

---
## Troubleshooting

| symptom | fix |
|---|---|
| `bringup_camera.sh` shows 0 fps | `bash reboot_robot.sh` then re-run bringup. If still 0 right after reboot, wait 30 s (services starting) and re-run bringup. |
| frame.jpg blank / 503 | camera not unlocked → re-run `bringup_camera.sh` (it calls StartVisionService). |
| robot ignores the scene / VLM repeats one action | the camera URL must end `/frame.jpg` (never `/stream`). bringup serves the right one. |
| print mode prints no decisions | VLM server not up (Terminal 1) or wrong `--server`. |
| **live mode: relay logs Move but robot doesn't walk** | DDS networking. This box's wired IP is `192.168.10.10`; the SDK targets `192.168.10.102`. Check link: `~/envs/navilaenv/bin/python -c "import socket; socket.create_connection(('192.168.10.102',22),timeout=3); print('wired OK')"`. If the `whitelist filtered out` error blocks commands, the FastDDS profile (`~/fastdds_k1.xml`, env `FASTRTPS_DEFAULT_PROFILES_FILE`) needs this box's wired interface/subnet whitelisted. |
| robot won't stand / enter walking | put it in Prepare/stand via the Booster remote or LUI first; check battery. |

Robot: `ssh booster@10.218.0.138` (pw 123456, sudo same). Wired: robot `192.168.10.102`,
this box `192.168.10.10`. VLM server: this box `:5555`.
