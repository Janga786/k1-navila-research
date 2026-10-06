"""Step 6 client: send one saved 8-frame payload N times to the running model server and record every reply.
Payload: NaVILA-Bench-main/eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit/diag/ep7/frames/tick005/
(the payload of the July two-query check). Primary request = what the evaluator sends: each frame as a lossless PNG of
the stored image (no resize; transform 'stretch' is a no-op), base64, query = that episode's instruction (dataset
index 7). Reference request = the July check byte-for-byte (384x384 PNG, query 'Walk to the goal and stop.').
Usage: python step6_client.py <session_label> <n_primary> <n_july> <out_jsonl>"""
import base64
import glob
import gzip
import hashlib
import io
import json
import os
import socket
import sys
import time

from PIL import Image

KR = os.path.expanduser("~/Projects/k1_research")
TICK = f"{KR}/NaVILA-Bench/eval_results/k1_matterport_vision_loco_smoke14498_lab3090_raw8bit/diag/ep7/frames/tick005"
EPS = json.load(gzip.open(f"{KR}/NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz", "rt"))["episodes"]


def build(resize, query):
    enc = []
    for f in sorted(glob.glob(f"{TICK}/frame_*.jpg"))[:8]:
        im = Image.open(f).convert("RGB")
        if resize:
            im = im.resize((384, 384))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        enc.append(base64.b64encode(buf.getvalue()).decode())
    return json.dumps({"images": enc, "query": query}).encode()


def ask(req):
    s = socket.socket()
    s.connect(("localhost", 54321))
    s.settimeout(120)
    s.sendall(len(req).to_bytes(8, "big"))
    s.sendall(req)
    n = b""
    while len(n) < 8:
        n += s.recv(8 - len(n))
    ln = int.from_bytes(n, "big")
    d = b""
    while len(d) < ln:
        d += s.recv(ln - len(d))
    s.close()
    return d.decode()


def main():
    label, n1, n2, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    instr = EPS[7]["instruction"]["instruction_text"]
    reqs = {"primary_evaluator_encoding_ep7_instruction": build(False, instr),
            "july_5b_exact_384png_generic_query": build(True, "Walk to the goal and stop.")}
    with open(out, "a") as fh:
        for kind, n in (("primary_evaluator_encoding_ep7_instruction", n1), ("july_5b_exact_384png_generic_query", n2)):
            for i in range(n):
                t0 = time.time()
                r = ask(reqs[kind])
                rec = {"session": label, "request": kind, "i": i, "reply": r, "reply_sha256": hashlib.sha256(r.encode()).hexdigest(),
                       "request_sha256": hashlib.sha256(reqs[kind]).hexdigest(), "seconds": round(time.time() - t0, 3)}
                fh.write(json.dumps(rec) + "\n")
                print(label, kind[:7], i, repr(r[:80]), rec["seconds"], flush=True)


if __name__ == "__main__":
    main()
