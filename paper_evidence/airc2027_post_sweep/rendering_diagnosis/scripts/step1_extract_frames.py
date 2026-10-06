"""Step 1 (CPU): decode sweep episode videos with ffmpeg and save the robot-camera half (left 1280 px)
and the chase-camera half (right 720 px) of chosen frames as PNG, plus the gray fraction of each half at
full resolution.

Gray measure (same as 12_airc2027_materials/analysis): a pixel is gray when every channel is within +-6 of
RGB 53; a frame "is gray" when >= 10 % of the robot-camera half is gray.

Video frame bookkeeping (navila_eval_v3.py): frame 0 = initial frame after the 1-s reset warm-up; frame
k >= 1 is written at main-loop step 5(k-1); its robot-camera half is the history frame captured at the
last step that is a multiple of 25.

Usage: python step1_extract_frames.py <out_dir> <csv_out> <run_tag>:<record>:<f1,f2,...> [...]
Read-only on eval_results/.
"""
import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ER = os.path.expanduser("~/Projects/k1_research/NaVILA-Bench-main/eval_results")
W, H = 2000, 720


def gray_frac(a):
    return float((np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6).mean())


def decode(video, frames):
    """Decode only the requested frame indices (exact, via select filter)."""
    expr = "+".join(f"eq(n\\,{f})" for f in frames)
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", video, "-vf", f"select={expr}",
                        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    n = buf.size // (W * H * 3)
    out = buf[: n * W * H * 3].reshape(n, H, W, 3)
    got = sorted(frames)[:n]
    return dict(zip(got, out))


def main():
    out_dir, csv_out = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for spec in sys.argv[3:]:
        tag, rec, fl = spec.split(":")
        frames = sorted({int(x) for x in fl.split(",")})
        video = f"{ER}/k1_matterport_vision_loco_{tag}/videos/output_{rec}.mp4"
        dec = decode(video, frames)
        for f in frames:
            if f not in dec:
                rows.append([tag, rec, f, "", "", "missing"])
                continue
            a = dec[f]
            robot, chase = a[:, :1280], a[:, 1280:]
            base = f"{tag}_rec{rec}_f{f:04d}"
            Image.fromarray(robot).save(f"{out_dir}/{base}_robotcam.png")
            Image.fromarray(chase).save(f"{out_dir}/{base}_chasecam.png")
            step = "init" if f == 0 else 5 * (f - 1)
            hist_step = "init" if f == 0 else 25 * ((5 * (f - 1)) // 25)
            rows.append([tag, rec, f, step, hist_step, round(gray_frac(robot), 4), round(gray_frac(chase), 4)])
    new = not os.path.exists(csv_out)
    with open(csv_out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "record", "video_frame", "main_loop_step", "robotcam_history_step",
                        "robotcam_gray_frac", "chasecam_gray_frac"])
        w.writerows(rows)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
