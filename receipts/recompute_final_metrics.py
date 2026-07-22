#!/usr/bin/env python3
"""Recompute the headline K1 benchmark numbers from per-episode measurement JSONs.

Primary data: NaVILA-Bench/eval_results/k1_matterport_vision_loco_full_14498/measurements/
(1,077 files, one per episode; keys incl. distance_to_goal, success, oracle_success, spl,
path_length, oracle_navigation_error).

Regenerate:  python3 receipts/recompute_final_metrics.py
Expected:    SR 18.29 / OS 30.27 / NE 7.59 / SPL 10.93  (n=1077, 0 missing)
Also prints the failure decomposition (never-reached / reached-no-stop / success)
and SPL-among-successes used in the paper-evidence chain.
"""
import json, glob, os, sys

MDIR = os.path.expanduser(
    "~/Projects/k1_research/NaVILA-Bench/eval_results/"
    "k1_matterport_vision_loco_full_14498/measurements")
TOTAL = 1077

files = sorted(glob.glob(os.path.join(MDIR, "*.json")))
n = len(files)
recs = [json.load(open(f)) for f in files]

suc = sum(r.get("success", 0.0) for r in recs)
osu = sum(r.get("oracle_success", 0.0) for r in recs)
spl = sum(r.get("spl", 0.0) for r in recs)
ne  = [r["distance_to_goal"] for r in recs if "distance_to_goal" in r]
pl  = [r.get("path_length", 0.0) for r in recs]

sr_pct  = 100.0 * suc / TOTAL
os_pct  = 100.0 * osu / TOTAL
spl_pct = 100.0 * spl / TOTAL
ne_mean = sum(ne) / len(ne)

n_success   = int(round(suc))
n_reached   = int(round(osu))
n_reach_nostop = n_reached - n_success
n_never     = TOTAL - n_reached

ne_suc = [r["distance_to_goal"] for r in recs if r.get("success", 0.0) > 0.5]

print(f"files present         : {n} / {TOTAL}  (missing counted as failures)")
print(f"SR   = {sr_pct:.2f}%   ({n_success}/{TOTAL})")
print(f"OS   = {os_pct:.2f}%   ({n_reached}/{TOTAL})")
print(f"SPL  = {spl_pct:.2f}%")
print(f"NE   = {ne_mean:.2f} m  (final distance_to_goal, present episodes)")
print(f"mean path length      : {sum(pl)/len(pl):.2f} m")
print("--- failure decomposition ---")
print(f"never reached         : {n_never}  ({100.0*n_never/TOTAL:.1f}%)")
print(f"reached, no stop      : {n_reach_nostop}  ({100.0*n_reach_nostop/TOTAL:.1f}%)")
print(f"success               : {n_success}  ({100.0*n_success/TOTAL:.1f}%)")
print(f"stop-given-reached    : {n_success}/{n_reached} = {100.0*n_success/max(n_reached,1):.1f}%")
print(f"NE among successes    : {sum(ne_suc)/max(len(ne_suc),1):.2f} m")
print("--- derived (paper chain) ---")
print(f"SPL-among-successes (K1)      : {spl_pct:.2f}/{sr_pct:.2f} = {spl_pct/sr_pct:.3f}")
print(f"SPL-among-successes (H1 blind): 21.0/24.4 = {21.0/24.4:.3f}  (paper Table values)")
print(f"H1-blind stop-conversion @1077: {round(0.244*TOTAL)}/{round(0.333*TOTAL)} = "
      f"{0.244/0.333*100:.1f}%  (derived from paper SR 24.4 / OS 33.3)")
sys.exit(0)
