"""FAKE model server for the offline harness (same TCP framing as vlm_server_bridge.py). NOT the VLM.
Reply = function of coarse statistics of the decoded frames, so +-1 render noise flips it now and then
(a knife edge), 'stop' once the last frame is bright (the fake robot walked ~4 m).
  --port P   --nondet p   (p = probability of a random reply flip, emulating inference nondeterminism)
"""
import argparse, base64, io, json, os, random, socket
import numpy as np
from PIL import Image

REPLIES = ["The next action is move forward 75 cm.", "The next action is move forward 75 cm.",
           "The next action is move forward 50 cm.", "The next action is turn left 15 degrees.",
           "The next action is move forward 75 cm.", "The next action is turn right 30 degrees."]
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=54399); ap.add_argument("--nondet", type=float, default=0.0)
a = ap.parse_args()
rnd = random.Random(os.urandom(8))
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("localhost", a.port)); s.listen(1)
print(f"[fake] listening on localhost:{a.port}", flush=True)
while True:
    c, _ = s.accept()
    n = b""
    while len(n) < 8: n += c.recv(8 - len(n))
    size = int.from_bytes(n, "big"); buf = b""
    while len(buf) < size: buf += c.recv(min(4096, size - len(buf)))
    req = json.loads(buf.decode())
    ims = [np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"), dtype=np.float64) for b in req["images"]]
    m = ims[-1].mean()
    if m > 160: text = "The next action is stop."
    else:
        text = REPLIES[int(np.floor(m * 40 + ims[0].mean() * 7)) % len(REPLIES)]
        if a.nondet > 0 and rnd.random() < a.nondet: text = REPLIES[(REPLIES.index(text) + 3) % len(REPLIES)]
    print(f"[fake] mean {m:.4f} -> {text!r}", flush=True)
    out = json.dumps(text).encode(); c.sendall(len(out).to_bytes(8, "big")); c.sendall(out); c.close()
