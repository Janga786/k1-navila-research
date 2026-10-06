"""Step 6: choose what goes to the private archive, copy it into the archive checkout under
15_airc2027_frame_replay/, and write MANIFEST.csv (path, bytes, SHA-256) for everything that stays on the
workstation. Prints the explicit list of paths to `git add` (never -A).

  stage_upload.py <archive_checkout> [--dry-run]

Uploaded: PLAN.md (must be byte-identical to the pre-registered file), RESULTS.md, STOP_REPORT.md,
PROPOSED_FIX_png_decode.diff, MANIFEST.csv, scripts/ (incl. the copy, evaluator.diff, mock/), logs/, tables/,
figures/, qstar_frames/, replay/ (minus the request .bin files, which are rebuilt from qstar_frames), and per run:
queries.jsonl, frames.jsonl, state_hashes.txt, state_components.tsv, state_layout.json, states_at_queries.npz,
run_meta.json, measurements/*.json, replay_mismatch.json, error.txt, replay_queries.jsonl.
Kept on the workstation (listed in MANIFEST.csv): every frames/ and fresh_frames/ PNG, videos, request .bin files.
Refuses anything >= 50 MB, any file type on the never-upload list, and any file matching a secret pattern.
"""
import hashlib
import os
import re
import shutil
import sys

R = os.path.realpath(os.path.expanduser("~/Projects/k1_research/airc2027_replay"))
DEST = "15_airc2027_frame_replay"
MAX = 50 * 1024 * 1024
RUN_FILES = {"queries.jsonl", "frames.jsonl", "state_hashes.txt", "state_components.tsv", "state_layout.json",
             "states_at_queries.npz", "run_meta.json", "replay_mismatch.json", "error.txt", "replay_queries.jsonl"}
TOP_FILES = ["PLAN.md", "RESULTS.md", "STOP_REPORT.md", "PROPOSED_FIX_png_decode.diff"]
TOP_DIRS = ["scripts", "logs", "tables", "figures", "qstar_frames", "replay"]
NEVER_EXT = {".usd", ".usda", ".usdc", ".glb", ".obj", ".stl", ".dae", ".pt", ".pth", ".ckpt", ".safetensors",
             ".bin", ".gz", ".urdf", ".mp4"}
NEVER_NAME = re.compile(r"(^\.env|\.env$|transcript|\.jsonl\.claude|vln_ce_isaac)", re.I)
SECRET = re.compile(rb"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
                    rb"-----BEGIN [A-Z ]*PRIVATE KEY|https://[^/\s:@]+:[^/\s@]+@github\.com|hf_[A-Za-z0-9]{30,}|"
                    rb"(?i:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token)\s*[=:]\s*['\"]?[A-Za-z0-9/+_\-]{8,})")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wanted(rel):
    parts = rel.split(os.sep)
    if "__pycache__" in parts:
        return False
    if parts[0] == "runs":
        return parts[-1] in RUN_FILES or (len(parts) >= 3 and parts[-2] == "measurements" and rel.endswith(".json"))
    if parts[0] == "replay":
        return not rel.endswith(".bin")
    return parts[0] in TOP_DIRS or rel in TOP_FILES


def main():
    dest_root, dry = sys.argv[1], "--dry-run" in sys.argv
    up, keep, problems = [], [], []
    for dp, dns, fns in os.walk(R):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, R)
            if rel == "MANIFEST.csv":
                continue
            (up if wanted(rel) else keep).append(rel)
    for rel in up:
        p = os.path.join(R, rel)
        ext = os.path.splitext(rel)[1].lower()
        if os.path.getsize(p) >= MAX:
            problems.append(f"too large: {rel} ({os.path.getsize(p)} B)")
        if ext in NEVER_EXT or NEVER_NAME.search(os.path.basename(rel)):
            problems.append(f"never-upload type: {rel}")
        if ext not in (".png", ".npz"):
            with open(p, "rb") as fh:
                m = SECRET.search(fh.read())
            if m:
                problems.append(f"secret pattern in {rel}: {m.group(0)[:12]!r}...")
    if problems:
        print("REFUSED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    # manifest of everything kept on the workstation
    keep.sort()
    with open(os.path.join(R, "MANIFEST.csv"), "w") as fh:
        fh.write("path,bytes,sha256\n")
        for rel in keep:
            p = os.path.join(R, rel)
            fh.write(f"{rel},{os.path.getsize(p)},{sha(p)}\n")
    up.append("MANIFEST.csv")
    total = sum(os.path.getsize(os.path.join(R, rel)) for rel in up)
    print(f"upload: {len(up)} files, {total / 1e6:.1f} MB; kept on workstation (MANIFEST.csv): {len(keep)} files, "
          f"{sum(os.path.getsize(os.path.join(R, k)) for k in keep) / 1e9:.2f} GB")
    if dry:
        return
    for rel in sorted(up):
        d = os.path.join(dest_root, DEST, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(R, rel), d)
    with open(os.path.join(dest_root, ".frame_replay_add_list"), "w") as fh:
        for rel in sorted(up):
            fh.write(os.path.join(DEST, rel) + "\n")
    print("copied; explicit add list:", os.path.join(dest_root, ".frame_replay_add_list"))


if __name__ == "__main__":
    main()
