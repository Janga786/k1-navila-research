"""Compare lossless simulator frames with the sweep's video frames (CPU).

Each comparison: sim PNG (robot camera, 1280x720, no text) vs the robot-camera half of a sweep video frame. The
evaluator draws the instruction text on the robot-camera frame before writing the video, so the same overlay is
drawn on a copy of the sim frame (eval_utils.add_instruction_on_img, imported read-only) before comparing.
Reported: gray fraction of the sim frame (raw, and with overlay) and of the video frame; mean / 99th-percentile
absolute difference sim+overlay vs video (the video is H.264, so ~2-4 levels is the codec floor); and a
side-by-side PNG.

Usage: python compare_to_video.py <out_json> <spec> [<spec> ...]
  spec = <label>|<sim_png>|<run_tag>|<record>|<video_frame>|<episode_idx>
"""
import gzip
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

KR = os.path.expanduser("~/Projects/k1_research")
sys.path.insert(0, os.path.join(KR, "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils"))
from eval_utils import add_instruction_on_img  # noqa: E402

ER = os.path.join(KR, "NaVILA-Bench-main/eval_results")
EPS = json.load(gzip.open(os.path.join(KR, "NaVILA-Bench-main/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz"), "rt"))["episodes"]
G = lambda a: float((np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6).mean())  # noqa: E731


def video_frame(tag, rec, f):
    v = f"{ER}/k1_matterport_vision_loco_{tag}/videos/output_{rec}.mp4"
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", v, "-vf", f"select=eq(n\\,{f})", "-vsync", "0",
                        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8)[: 2000 * 720 * 3].reshape(720, 2000, 3)[:, :1280]


def main():
    out = sys.argv[1]
    res = []
    for spec in sys.argv[2:]:
        label, sim_png, tag, rec, f, idx = spec.split("|")
        sim = np.array(Image.open(sim_png).convert("RGB"))
        ov = sim.copy()
        add_instruction_on_img(ov, EPS[int(idx)]["instruction"]["instruction_text"])
        vid = video_frame(tag, int(rec), int(f))
        d = np.abs(ov.astype(np.int16) - vid.astype(np.int16))
        r = {"label": label, "sim_png": os.path.relpath(sim_png, os.path.dirname(os.path.abspath(out))), "video": f"{tag}/output_{rec}.mp4",
             "video_frame": int(f), "episode_idx": int(idx), "record": int(EPS[int(idx)]["episode_id"]) - 1,
             "gray_sim_raw": G(sim), "gray_sim_with_overlay": G(ov), "gray_video": G(vid),
             "mean_abs_diff_sim+overlay_vs_video": float(d.mean()), "p99_abs_diff": float(np.percentile(d.max(2), 99))}
        assert r["record"] == int(rec), (r["record"], rec)
        res.append(r)
        Image.fromarray(np.concatenate([ov, vid], 1)).resize((1280, 360), Image.LANCZOS).save(
            os.path.join(os.path.dirname(os.path.abspath(out)), f"{label}_sim_vs_video.png"))
        print(f"{label:28s} gray sim {r['gray_sim_raw']:.4f} (+text {r['gray_sim_with_overlay']:.4f}) video {r['gray_video']:.4f}  "
              f"MAE {r['mean_abs_diff_sim+overlay_vs_video']:.2f}  p99 {r['p99_abs_diff']:.0f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
