# Live NaVILA-on-K1 episode — receipt (May 2026)

Two recordings of the same session document NaVILA driving the **real K1** in live mode
through the early laptop relay. Added to the repository 2026-10-30; this resolves the
"volleyball clip not found" gap in `../media_index.md` and `../../EVIDENCE.md` §18.

## Hosted recordings

The `.mp4` files are git-ignored and are not in this repository; they are hosted with the
portfolio:

- <https://janga786.github.io/media/k1-relay-live.mp4> — relay screen recording + robot: the
  relay in `[LIVE]` mode streaming VLM velocity commands over the robot's own head camera.
- <https://janga786.github.io/media/k1-walk-to-volleyball.mp4> — third-person clip of the K1
  walking toward the volleyball during the live episode.

Two live natural-language episodes were run in the May 2026 session ("walk to the volleyball
and stop", "walk to the volleyball and turn left"). One is captured on video (above); the
second is on the project record. Both used Booster's built-in walker through the relay, with
zero teleoperation. The trained RL policy has never worked outside simulation.

## `relay_live_volleyball_2026-05.mp4` (screen recording, 17.5 s)

A screen capture of the **"NaVILA Laptop Relay"** window during the episode:

- Mode banner: `[LIVE] walk to the volley ball and turn 90 degrees` (step 1/1).
- NaVILA's action text is shown per inference (e.g. "The next action is move forward 75 cm.",
  "…turn left 45 degree.", "…move forward 25 cm.").
- The command line shows the **VLM as the command source**: `out: vx=+0.15 vy=+0.00
  vyaw=+0.00 [VLM]`, with live telemetry (`vlm: inf#15 … inf#46`, ~352–353 ms per inference,
  `buf=8`, `link_age≈0.2–0.4 s`) and the emergency-stop hint.
- The video pane is the robot's own head camera, moving through the lab as the commands
  stream — the loop camera → VLM → velocity command → motion is visible in one frame.

## `thirdperson_volleyball_2026-05-15.mp4` (phone video, 9.9 s)

A third-person clip of the K1 walking toward the volleyball in the lab during the session
(original file dated **2026-05-15**, Fort Lewis College lab; device metadata stripped for
publication). The robot is on a safety/power tether.

## Claim boundary (read before citing)

- These recordings support: **NaVILA issued velocity commands to the real K1 in live mode,
  with no teleoperation, and the robot walked toward the instructed goal** under Booster's
  **built-in walking controller** executing those commands, via the early laptop relay (the
  relay telemetry shows a remote VLM link; this deployment predates the deploy relay
  documented in `../../ROBOT_DEPLOY_RUNBOOK.md`).
- One episode is captured in these recordings; a second instruction episode from the same
  session is on the project record but was not recorded.
- They do **not** show: the trained RL policy on hardware (it was evaluated in simulation
  only and has not walked on hardware), an untethered run, or a scored navigation episode
  (no success metric is claimed for the live session).
