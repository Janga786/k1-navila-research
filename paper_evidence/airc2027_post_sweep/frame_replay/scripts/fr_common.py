"""Shared helpers for the frame-replay scripts (CPU only; no simulator, no GPU).

Run directories are written by navila_eval_v3_LOG.py:
  frames/f{k:04d}.png, frames.jsonl, queries.jsonl, state_hashes.txt, state_components.tsv,
  states_at_queries.npz, run_meta.json, measurements/<record>.json, videos/output_<record>.mp4
  (run R also: fresh_frames/f{k:04d}.png and possibly replay_mismatch.json)

build_request() reproduces the evaluator's request encoding exactly for --vlm_transform stretch
without --clean_render/--bright (the sweep configuration): each sent frame is Image.fromarray of
the stored uint8 array, saved as PNG, base64-encoded; request = json.dumps({'images', 'query'}).
Run it with the interpreter the evaluator uses (vlnce-isaac python) so the PNG encoder is the same.
"""
import base64
import hashlib
import io
import json
import os

import numpy as np
from PIL import Image

KR = os.path.expanduser("~/Projects/k1_research")
R = os.path.join(KR, "airc2027_replay")
NB = os.path.join(KR, "NaVILA-Bench-main")
DATASET = os.path.join(NB, "isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz")
SIX = ("path_length", "distance_to_goal", "oracle_navigation_error", "success", "spl", "oracle_success")
# (episode_idx, record, D1 length, D2 length) in table order
EPISODES = [(3, 6, 903, 1493), (8, 11, 1573, 1573), (16, 25, 2640, 4068), (20, 32, 1563, 6001),
            (50, 77, 6001, 6001), (99, 144, 2709, 2772), (92, 137, 925, 926), (212, 323, 1049, 907)]
RECORD = {e[0]: e[1] for e in EPISODES}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_states(run_dir):
    """Ordered [(label, hash)] from state_hashes.txt."""
    out = []
    with open(os.path.join(run_dir, "state_hashes.txt")) as fh:
        for line in fh:
            if line.strip():
                lab, hx = line.split()
                out.append((lab, hx))
    return out


def load_components(run_dir):
    """{label: {part: hash}} from state_components.tsv."""
    path = os.path.join(run_dir, "state_components.tsv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            out[f[0]] = dict(zip(header[1:], f[1:]))
    return out


def load_record(run_dir):
    md = os.path.join(run_dir, "measurements")
    files = sorted(os.listdir(md)) if os.path.isdir(md) else []
    if len(files) != 1:
        return None
    with open(os.path.join(md, files[0])) as fh:
        return json.load(fh)


def load_run(run_dir):
    run = {"dir": run_dir, "name": os.path.basename(run_dir.rstrip("/")),
           "queries": load_jsonl(os.path.join(run_dir, "queries.jsonl")),
           "frames": load_jsonl(os.path.join(run_dir, "frames.jsonl")),
           "states": load_states(run_dir) if os.path.exists(os.path.join(run_dir, "state_hashes.txt")) else [],
           "record": load_record(run_dir)}
    mp = os.path.join(run_dir, "run_meta.json")
    run["meta"] = json.load(open(mp)) if os.path.exists(mp) else None
    mm = os.path.join(run_dir, "replay_mismatch.json")
    run["mismatch"] = json.load(open(mm)) if os.path.exists(mm) else None
    return run


def load_frame(run_dir, k, sub="frames"):
    return np.array(Image.open(os.path.join(run_dir, sub, f"f{k:04d}.png")))


def build_request(arrays, query):
    """Request bytes exactly as navila_eval_v3.sample_images_and_send_to_vlm builds them for the
    sweep configuration. arrays: the 8 uint8 HxWx3 frames in send order. Returns (bytes, png_sha256s)."""
    enc, pngs = [], []
    for arr in arrays:
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        pngs.append(sha256(buf.getvalue()))
        enc.append(base64.b64encode(buf.getvalue()).decode())
    return json.dumps({'images': enc, 'query': query}).encode(), pngs


def rebuild_query_request(run_dir, q, sub="frames"):
    """Rebuild one logged query's request from the saved frames."""
    arrays = []
    for k in q["indices"]:
        if k < 0:  # black padding frame (never happens with >= 9 frames in the history)
            ref = load_frame(run_dir, 0, sub)
            arrays.append(np.zeros_like(ref))
        else:
            arrays.append(load_frame(run_dir, k, sub))
    return build_request(arrays, q["instruction"])


def img_diff_stats(a, b):
    """max / mean absolute difference and share of channels that differ (uint8 arrays)."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {"max_abs_diff": int(d.max()), "mean_abs_diff": float(d.mean()),
            "share_channels_differ": float((d > 0).mean()),
            "share_pixels_differ": float((d.max(axis=-1) > 0).mean())}


def main_loop_label(lab):
    """True for main-loop step labels (integers), False for 'reset' / 'w<i>'."""
    return lab.lstrip("-").isdigit()
