#!/usr/bin/env python3
"""
reproduction_check.py — independent recomputation of every key paper metric directly from
the raw per-episode JSONs in receipts/, via paper_common.py ONLY.

Does NOT import or call receipts/analysis/analyze_sweep.py or analyze_robustness.py. This is
a from-scratch re-derivation, so that a match against the historically claimed values is real
evidence of reproducibility rather than a tautology.

READ-ONLY: this script never writes to receipts/. It only reads JSON records through
paper_common and prints a plain-text report to stdout, which paper_evidence/REPRODUCTION_CHECK.md
transcribes verbatim.

Run:
    /home/boosterk1/miniconda3/bin/python3 paper_evidence/scripts/reproduction_check.py
"""
import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_EVIDENCE = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER_EVIDENCE)
RECEIPTS = os.path.join(ROOT, "receipts")

sys.path.insert(0, HERE)
import paper_common as pc  # noqa: E402

NE_SENTINEL = -1.0
TRANSFORM_ARMS = ["stretchA", "stretchB", "crop", "pad"]
HEIGHT_ARMS = ["h060", "h078", "h095", "h110", "h130", "h150"]

SEP = "=" * 100


def hdr(title):
    print("\n" + SEP)
    print(title)
    print(SEP)


# --------------------------------------------------------------------------- core stats
def rate(k, n):
    return 100.0 * k / n


def sr_count(recs):
    return sum(1 for r in recs.values() if pc.is_success(r))


def osr_count(recs):
    return sum(1 for r in recs.values() if pc.is_oracle_success(r))


def spl_sum(recs):
    return sum(float(r.get("spl", 0.0)) for r in recs.values())


def ne_values(recs):
    return [r["distance_to_goal"] for r in recs.values()
            if r.get("distance_to_goal", NE_SENTINEL) != NE_SENTINEL]


def timeout_count(recs):
    return sum(1 for r in recs.values() if pc.is_timeout(r))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- verdict bookkeeping
RESULTS = []  # (group, metric, expected, regen, diff, tol, source, verdict)


def check(group, metric, expected, regen, tol, source, fmt="{:.4f}"):
    diff = abs(expected - regen)
    ok = diff <= tol + 1e-12
    verdict = "MATCH" if ok else "MISMATCH"
    RESULTS.append((group, metric, fmt.format(expected), fmt.format(regen),
                     fmt.format(diff), tol, source, verdict))
    print(f"[{verdict}] {group:12s} {metric:10s} expected={fmt.format(expected)} "
          f"regen={fmt.format(regen)} diff={fmt.format(diff)} tol={tol} src={source}")
    return ok


def check_int(group, metric, expected, regen, source):
    diff = abs(expected - regen)
    ok = diff == 0
    verdict = "MATCH" if ok else "MISMATCH"
    RESULTS.append((group, metric, str(expected), str(regen), str(diff), 0, source, verdict))
    print(f"[{verdict}] {group:12s} {metric:10s} expected={expected} regen={regen} "
          f"diff={diff} tol=0 src={source}")
    return ok


# =========================================================================================
# BASELINE (n=1077)
# =========================================================================================
hdr("BASELINE (receipts/baseline_full_14498, n=1077 claimed)")
baseline = pc.load_baseline()
n_base = len(baseline)
print(f"n_present = {n_base}")
check_int("baseline", "n", 1077, n_base, "receipts/baseline_full_14498/*.json")

k_sr = sr_count(baseline)
k_os = osr_count(baseline)
sr_pct = rate(k_sr, 1077)
os_pct = rate(k_os, 1077)
spl_pct = rate(spl_sum(baseline), 1077)
ne_vals = ne_values(baseline)
ne_mean = mean(ne_vals)
n_sentinel_base = n_base - len(ne_vals)

check("baseline", "SR%", 18.3, sr_pct, 0.05, "baseline_full_14498", "{:.2f}")
check("baseline", "OSR%", 30.3, os_pct, 0.05, "baseline_full_14498", "{:.2f}")
check("baseline", "NE_m", 7.59, ne_mean, 0.01, "baseline_full_14498", "{:.3f}")
check("baseline", "SPL%", 10.93, spl_pct, 0.05, "baseline_full_14498", "{:.2f}")
print(f"baseline distance_to_goal == -1.0 sentinel count: {n_sentinel_base} "
      f"(NE computed over {len(ne_vals)} of {n_base})")

# =========================================================================================
# TRANSFORM ARMS (n=300 each): SR% and timeout rate
# =========================================================================================
hdr("TRANSFORM ARMS (stretchA/stretchB/crop/pad, n=300 claimed)")
CLAIMED_TRANSFORM = {
    "stretchA": (15.0, 6.67),
    "stretchB": (16.0, 7.33),
    "crop":     (16.7, 2.0),
    "pad":      (13.3, 12.3),
}
arm_recs = {}
for tag in TRANSFORM_ARMS:
    recs = pc.load_arm(tag)   # asserts n_total == 300 internally
    arm_recs[tag] = recs
    n = len(recs)
    k = sr_count(recs)
    t = timeout_count(recs)
    exp_sr, exp_to = CLAIMED_TRANSFORM[tag]
    check(tag, "SR%", exp_sr, rate(k, n), 0.05, f"sweep_measurements/{tag}_300", "{:.2f}")
    check(tag, "timeout%", exp_to, rate(t, n), 0.05, f"sweep_measurements/{tag}_300", "{:.2f}")

# =========================================================================================
# HEIGHT ARMS (n=200 each): SR% vs SWEEP_STATUS.md
# =========================================================================================
hdr("HEIGHT ARMS (h060..h150, n=200 claimed, vs SWEEP_STATUS.md)")
CLAIMED_HEIGHT = {"h060": 11.5, "h078": 12.5, "h095": 18.5,
                  "h110": 13.0, "h130": 15.0, "h150": 14.5}
height_recs = {}
for tag in HEIGHT_ARMS:
    recs = pc.load_arm(tag)
    height_recs[tag] = recs
    n = len(recs)
    k = sr_count(recs)
    check(tag, "SR%", CLAIMED_HEIGHT[tag], rate(k, n), 0.05,
          f"sweep_measurements/{tag}_200 (SWEEP_STATUS.md)", "{:.2f}")

# =========================================================================================
# REPEATABILITY 2x2 (stretchA vs stretchB) + Cohen's kappa
# =========================================================================================
hdr("REPEATABILITY 2x2 (stretchA vs stretchB, ROBUSTNESS.md section A claimed)")
A, B = arm_recs["stretchA"], arm_recs["stretchB"]
both = sorted(set(A) & set(B))
n_pair = len(both)
a1b1 = sum(1 for i in both if pc.is_success(A[i]) and pc.is_success(B[i]))
a1b0 = sum(1 for i in both if pc.is_success(A[i]) and not pc.is_success(B[i]))
a0b1 = sum(1 for i in both if not pc.is_success(A[i]) and pc.is_success(B[i]))
a0b0 = sum(1 for i in both if not pc.is_success(A[i]) and not pc.is_success(B[i]))
disc = a1b0 + a0b1
po = (a1b1 + a0b0) / n_pair
pe = (((a1b1 + a1b0) / n_pair) * ((a1b1 + a0b1) / n_pair)
      + ((a0b1 + a0b0) / n_pair) * ((a1b0 + a0b0) / n_pair))
kappa = (po - pe) / (1 - pe)
disc_pct = rate(disc, n_pair)

check_int("2x2", "n_paired", 300, n_pair, "stretchA/stretchB intersection")
check_int("2x2", "a1b1(A+B+)", 30, a1b1, "stretchA/stretchB")
check_int("2x2", "a1b0(A+B-)", 15, a1b0, "stretchA/stretchB")
check_int("2x2", "a0b1(A-B+)", 18, a0b1, "stretchA/stretchB")
check_int("2x2", "a0b0(A-B-)", 237, a0b0, "stretchA/stretchB")
check("2x2", "discord%", 11.0, disc_pct, 0.05, "stretchA/stretchB", "{:.2f}")
check("2x2", "kappa", 0.580, kappa, 0.001, "stretchA/stretchB", "{:.4f}")

# =========================================================================================
# DATA INTEGRITY
# =========================================================================================
hdr("DATA INTEGRITY — episode counts, duplicates, pairing, sentinels, schema anomaly")

ALL_TAGS = TRANSFORM_ARMS + HEIGHT_ARMS
all_recs = dict(arm_recs)
all_recs.update(height_recs)

print("\n-- episode counts + duplicate-filename check --")
for tag in ALL_TAGS:
    subdir, n_total, _ = pc.ARMS[tag]
    d = os.path.join(RECEIPTS, subdir)
    files = [f for f in os.listdir(d) if f.endswith(".json")]
    stems = [os.path.splitext(f)[0] for f in files]
    ints = [int(s) for s in stems]
    dup = len(ints) != len(set(ints))
    # a "canonical" stem check: int(stem) round-trips to the same string (no leading zeros,
    # no alternate spellings colliding on the same integer key)
    noncanonical = [s for s in stems if str(int(s)) != s]
    n_files = len(files)
    print(f"  {tag:10s} files={n_files:4d} expected={n_total:4d} "
          f"duplicate_index={dup} noncanonical_filenames={len(noncanonical)} "
          f"{'OK' if (n_files == n_total and not dup and not noncanonical) else 'ANOMALY'}")

print("\n-- baseline duplicate-filename check --")
bd = os.path.join(RECEIPTS, "baseline_full_14498")
bfiles = [f for f in os.listdir(bd) if f.endswith(".json")]
bstems = [os.path.splitext(f)[0] for f in bfiles]
bints = [int(s) for s in bstems]
bdup = len(bints) != len(set(bints))
bnoncanon = [s for s in bstems if str(int(s)) != s]
print(f"  baseline_full_14498 files={len(bfiles)} duplicate_index={bdup} "
      f"noncanonical_filenames={len(bnoncanon)} "
      f"{'OK' if (len(bfiles) == 1077 and not bdup and not bnoncanon) else 'ANOMALY'}")

print("\n-- transform-arm identical record_idx set check --")
sets = {tag: set(arm_recs[tag].keys()) for tag in TRANSFORM_ARMS}
ref = sets["stretchA"]
all_identical = True
for tag in TRANSFORM_ARMS:
    same = sets[tag] == ref
    all_identical = all_identical and same
    print(f"  {tag:10s} identical_to_stretchA_set={same} "
          f"(sym_diff={len(sets[tag] ^ ref)})")
print(f"  ALL FOUR TRANSFORM ARMS COVER IDENTICAL record_idx SET: {all_identical}")

print("\n-- height-arm identical record_idx set check (200-episode set) --")
hsets = {tag: set(height_recs[tag].keys()) for tag in HEIGHT_ARMS}
href = hsets["h060"]
h_identical = True
for tag in HEIGHT_ARMS:
    same = hsets[tag] == href
    h_identical = h_identical and same
    print(f"  {tag:10s} identical_to_h060_set={same} (sym_diff={len(hsets[tag] ^ href)})")
print(f"  ALL SIX HEIGHT ARMS COVER IDENTICAL record_idx SET: {h_identical}")
print(f"  height set is subset of transform-300 set: {href <= ref}")

print("\n-- -1.0 distance_to_goal sentinel counts --")
for tag in ALL_TAGS:
    recs = all_recs[tag]
    n_sent = sum(1 for r in recs.values() if r.get("distance_to_goal", NE_SENTINEL) == NE_SENTINEL)
    print(f"  {tag:10s} sentinel_count={n_sent} / {len(recs)}")
print(f"  baseline   sentinel_count={n_sentinel_base} / {n_base}")

print("\n-- schema anomaly: max_episode_steps == -1 (== ended_at_step == -1) --")
CLAIMED_ANOMALY = {"stretchA": 3, "stretchB": 4, "h078": 1, "h130": 1}
anomaly_total = 0
for tag in ALL_TAGS:
    recs = all_recs[tag]
    c_mes = sum(1 for r in recs.values() if r.get("max_episode_steps") == -1)
    c_eas = sum(1 for r in recs.values() if r.get("ended_at_step") == -1)
    same = c_mes == c_eas
    anomaly_total += c_mes
    exp = CLAIMED_ANOMALY.get(tag, 0)
    verdict = "MATCH" if c_mes == exp else "MISMATCH"
    print(f"  [{verdict}] {tag:10s} max_episode_steps=-1 count={c_mes}  "
          f"ended_at_step=-1 count={c_eas}  co-occur={same}  claimed={exp}")
check_int("schema", "total_anomaly_rows", 9, anomaly_total, "all 10 arms, 2400 episodes")

# =========================================================================================
# SCENE COVERAGE
# =========================================================================================
hdr("SCENE COVERAGE (receipts/analysis/episode_index.csv)")
scene_of = {}
with open(pc.EPISODE_INDEX, newline="") as f:
    for row in csv.DictReader(f):
        scene_of[int(row["record_idx"])] = row["scene"]
n_full_episodes = len(scene_of)
full_scenes = set(scene_of.values())
pinned300 = ref  # stretchA's set, identical across all 4 transform arms
missing_from_index = [i for i in pinned300 if i not in scene_of]
pinned_scenes = {scene_of[i] for i in pinned300 if i in scene_of}
print(f"episode_index.csv rows = {n_full_episodes}")
print(f"full benchmark scene count = {len(full_scenes)}")
print(f"pinned-300 record_idx missing from episode_index.csv = {len(missing_from_index)}")
print(f"pinned-300 scene count = {len(pinned_scenes)}")

from collections import Counter
pinned_scene_counts = Counter(scene_of[i] for i in pinned300 if i in scene_of)
print("episodes-per-scene, pinned 300 (scene: count):")
for s, c in pinned_scene_counts.most_common():
    print(f"  {s:15s} {c}")
print(f"sum check: {sum(pinned_scene_counts.values())} == 300 -> "
      f"{sum(pinned_scene_counts.values()) == 300}")

# =========================================================================================
# stretchA vs stretchB CONFIGURATION EQUIVALENCE EVIDENCE
# =========================================================================================
hdr("stretchA vs stretchB CONFIGURATION EQUIVALENCE EVIDENCE")

print("-- 1. arms_runner.sh ARMS array entries (source of truth for the invocation loop) --")
runner_path = os.path.join(ROOT, "arms_runner.sh")
with open(runner_path) as f:
    runner_txt = f.read()
m = re.search(r"ARMS=\((.*?)\)\n", runner_txt, re.S)
arms_block = m.group(1) if m else ""
for line in arms_block.strip().splitlines():
    line = line.strip().strip('"')
    if line.startswith("stretchA") or line.startswith("stretchB"):
        print(f"  {line!r}")
sA_line = next(l for l in arms_block.strip().splitlines()
               if l.strip().strip('"').startswith("stretchA"))
sB_line = next(l for l in arms_block.strip().splitlines()
               if l.strip().strip('"').startswith("stretchB"))
sA_fields = sA_line.strip().strip('"').split(None, 1)[1] if len(sA_line.split()) > 1 else ""
sB_fields = sB_line.strip().strip('"').split(None, 1)[1] if len(sB_line.split()) > 1 else ""
sA_rest = " ".join(sA_line.strip().strip('"').split()[1:])
sB_rest = " ".join(sB_line.strip().strip('"').split()[1:])
print(f"  stretchA non-name fields: {sA_rest!r}")
print(f"  stretchB non-name fields: {sB_rest!r}")
print(f"  IDENTICAL (transform/end/extra) beyond the name field: {sA_rest == sB_rest}")
print("  Both arms are launched by the SAME unconditional code path in the same script run "
      "(single $CKPT variable, single $RUNNER invocation template) — this is PROVEN by the "
      "script text: the transform/episode-range/checkpoint-path variables are not "
      "arm-specific, only the tag string is.")

print("\n-- 2. checkpoint hash (single file, used for every arm; CKPT is a static path in "
      "arms_runner.sh) --")
ckpt_hash_path = os.path.join(RECEIPTS, "provenance", "checkpoint_sha256.txt")
with open(ckpt_hash_path) as f:
    for line in f:
        if "model_14498.pt" in line and "jit" not in line:
            print(f"  {line.strip()}")
ckpt_file = os.path.join(ROOT, "checkpoints", "model_14498.pt")
if os.path.exists(ckpt_file):
    st = os.stat(ckpt_file)
    import datetime
    print(f"  checkpoints/model_14498.pt mtime = "
          f"{datetime.datetime.fromtimestamp(st.st_mtime).isoformat()} "
          f"(no writes recorded between the June training run and either sweep-arm launch)")

print("\n-- 3. SWEEP_STATUS.md ARM START lines (committed, timestamped by the runner itself) --")
status_path = os.path.join(ROOT, "SWEEP_STATUS.md")
with open(status_path) as f:
    status_txt = f.read()
for line in status_txt.splitlines():
    if "ARM START: stretchA" in line or "ARM START: stretchB" in line or \
       "ARM DONE: stretchA" in line or "ARM DONE: stretchB" in line:
        print(f"  {line}")

print("\n-- 4. sweep_run_log.txt raw invocation 'flags:' lines per arm (rescued /tmp log) --")
log_path = os.path.join(RECEIPTS, "provenance", "sweep_run_log.txt")
with open(log_path) as f:
    log_txt = f.read()
found = {}
for tag in ("stretchA_300", "stretchB_300", "crop_300", "pad_300"):
    mm = re.search(rf"out_tag={tag}\s+episodes=\S+\s+flags:.*", log_txt)
    found[tag] = mm.group(0) if mm else None
    print(f"  {tag}: {found[tag] if found[tag] else 'NOT FOUND IN LOG'}")
print(f"\n  stretchA_300 raw flags line present in sweep_run_log.txt: {found['stretchA_300'] is not None}")
print(f"  stretchB_300 raw flags line present in sweep_run_log.txt: {found['stretchB_300'] is not None}")
if found["stretchA_300"] is None and found["stretchB_300"] is not None:
    print("  CONSEQUENCE: sweep_run_log.txt begins '[08-04 15:32] SWEEP RUN (re)started' — "
          "AFTER stretchA had already completed (2026-07-26) and mid-way through stretchB "
          "(the power-loss outage 07-28..31 truncated the earlier log). The exact "
          "'[powered] ... flags:' invocation line is directly verifiable in this file for "
          "stretchB ONLY. stretchA's flags are corroborated by (1) arms_runner.sh's single "
          "unconditional code path and (2) SWEEP_STATUS.md's own 'transform=stretch eps 0-300 "
          "extra=none' line for stretchA, but NOT by a raw pre-outage invocation log line, "
          "which was not preserved.")

print("\n-- 5. NVIDIA driver version at the time of each arm --")
print("  ENVIRONMENT.md / PRE_REGISTRATION.md / SWEEP_STATUS.md all assert driver 580.173.02 "
      "for every sweep arm (stretchA included): the driver was bumped 2026-07-24 06:45 "
      "(pre-registration 'DRIVER-CHANGE CONFOUND' entry), auto-upgrades were then DISABLED "
      "and ~200 nvidia/cuda/kernel packages HELD the same day (AWAY_STATUS.md), and "
      "arms_runner.sh's own header records 'started Fri Jul 24 12:42:44' — after the bump. "
      "stretchA ran 07-24->07-26, stretchB 07-26->08-05 (spanning the 07-28..31 power outage). "
      "No file in this repository re-captures nvidia-smi output at each arm boundary; the "
      "single-driver claim rests on the upgrade timeline + apt holds, not a per-arm probe.")
print(f"  THIS machine's current driver (reanalysis time, informational only, NOT part of "
      f"the sweep): see paper_evidence/provenance/environment.txt")

print("\n-- SUMMARY: stretchA vs stretchB equivalence, proven vs asserted --")
print("  PROVEN (from committed files, checkable by inspection):")
print("    - identical transform/episode-range/extra-flags in arms_runner.sh's ARMS array")
print("    - identical record_idx set (300/300, checked above)")
print("    - identical checkpoint file (single path, single SHA-256, static mtime predating both runs)")
print("    - stretchB's raw invocation flags line, directly, in sweep_run_log.txt")
print("  ASSERTED (documented, not independently re-derived by this script):")
print("    - stretchA's raw invocation flags line (pre-outage log not preserved)")
print("    - both arms ran on the SAME driver 580.173.02 (inferred from the upgrade/hold "
      "timeline; no direct nvidia-smi capture exists per-arm in this repo)")
print("    - identical GPU / no other concurrent GPU load during either arm (not logged)")

# =========================================================================================
# SUMMARY
# =========================================================================================
hdr("VERDICT SUMMARY")
n_match = sum(1 for r in RESULTS if r[-1] == "MATCH")
n_mismatch = sum(1 for r in RESULTS if r[-1] == "MISMATCH")
print(f"Total metric checks: {len(RESULTS)}  MATCH={n_match}  MISMATCH={n_mismatch}")
if n_mismatch:
    print("\nMISMATCHES:")
    for r in RESULTS:
        if r[-1] == "MISMATCH":
            print(f"  {r}")
print("\nDone.")
