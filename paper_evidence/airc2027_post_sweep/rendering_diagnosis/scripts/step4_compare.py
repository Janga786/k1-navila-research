"""Step 4 (CPU): compare fresh-process captures from sim_capture.py pixel by pixel and pose by pose.
For each frame tag present in all given process directories: largest absolute channel difference, mean absolute
difference, share of channels that differ, gray fraction per process, and whether the Trunk/root state and all body
states are bit-identical (float32 bytes) to process 1.
Usage: python step4_compare.py <out_json> <proc_dir_1> <proc_dir_2> [...]"""
import json
import os
import sys

import numpy as np


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    metas = [json.load(open(f"{d}/capture.json")) for d in dirs]
    tags = [t for t in metas[0]["frames"] if all(t in m["frames"] for m in metas)]
    res = {"processes": dirs, "frames": {}}
    for t in tags:
        A = np.load(f"{dirs[0]}/{t}_robotcam.npy").astype(np.int16)
        ra = np.load(f"{dirs[0]}/{t}_root_state.npy")
        ba = np.load(f"{dirs[0]}/{t}_body_states.npy")
        rec = {"gray_frac_per_process": [m["frames"][t]["robotcam_gray_frac"] for m in metas], "vs_process_1": []}
        for d in dirs[1:]:
            B = np.load(f"{d}/{t}_robotcam.npy").astype(np.int16)
            diff = np.abs(A - B)
            rb, bb = np.load(f"{d}/{t}_root_state.npy"), np.load(f"{d}/{t}_body_states.npy")
            rec["vs_process_1"].append({
                "process": d, "max_abs_diff": int(diff.max()), "mean_abs_diff": float(diff.mean()),
                "share_channels_differ": float((diff > 0).mean()), "share_pixels_any_channel_differ": float((diff.max(2) > 0).mean()),
                "root_state_bit_identical": bool(ra.tobytes() == rb.tobytes()),
                "all_body_states_bit_identical": bool(ba.tobytes() == bb.tobytes()),
                "max_abs_body_state_diff": float(np.abs(ba.astype(np.float64) - bb).max())})
        res["frames"][t] = rec
        print(t, "gray", [round(g, 4) for g in rec["gray_frac_per_process"]],
              [(r["max_abs_diff"], round(r["mean_abs_diff"], 4), round(r["share_channels_differ"], 4), r["root_state_bit_identical"],
                r["all_body_states_bit_identical"]) for r in rec["vs_process_1"]], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
