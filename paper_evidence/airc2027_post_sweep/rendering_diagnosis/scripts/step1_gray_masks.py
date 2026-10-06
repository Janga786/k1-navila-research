"""Step 1.2 helper (CPU): for a set of robot-camera PNGs, draw the gray mask (every channel within +-6 of 53)
in red over the image, and report per-pair statistics: gray fraction of each, pixels gray in A only / B only,
and the mean colour of the region that is gray in A but not in B (what B rendered there instead), plus the
mean absolute pixel difference between A and B outside both masks.
Usage: python step1_gray_masks.py <out_png> <A.png> <B.png> [<A2.png> <B2.png> ...]"""
import sys

import numpy as np
from PIL import Image


def mask(a):
    return np.abs(a.astype(np.int16) - 53).max(axis=2) <= 6


def overlay(a, m):
    o = a.copy()
    o[m] = (0.4 * o[m] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    return o


def main():
    out = sys.argv[1]
    files = sys.argv[2:]
    rows = []
    for i in range(0, len(files), 2):
        A = np.array(Image.open(files[i]).convert("RGB"))
        B = np.array(Image.open(files[i + 1]).convert("RGB"))
        mA, mB = mask(A), mask(B)
        onlyA, onlyB = mA & ~mB, mB & ~mA
        neither = ~mA & ~mB
        diff = np.abs(A.astype(np.int16) - B.astype(np.int16))
        print(f"A={files[i].rsplit('/',1)[-1]}  B={files[i+1].rsplit('/',1)[-1]}")
        print(f"   grayA={mA.mean():.4f} grayB={mB.mean():.4f} onlyA={onlyA.mean():.4f} onlyB={onlyB.mean():.4f}")
        if onlyA.any():
            print(f"   mean RGB of A-only region: in A {A[onlyA].mean(0).round(1)}  in B {B[onlyA].mean(0).round(1)}")
        if onlyB.any():
            print(f"   mean RGB of B-only region: in A {A[onlyB].mean(0).round(1)}  in B {B[onlyB].mean(0).round(1)}")
        print(f"   mean |A-B| outside both masks: {diff[neither].mean():.2f}; max {diff[neither].max()}; "
              f"share of pixels with |A-B|>20 outside masks: {(diff[neither].max(1) > 20).mean():.4f}")
        rows.append(np.concatenate([overlay(A, mA), overlay(B, mB)], axis=1))
    sheet = np.concatenate(rows, axis=0)
    im = Image.fromarray(sheet)
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    main()
