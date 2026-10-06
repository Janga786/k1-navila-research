"""Step 1 checks for one normal-mode run of navila_eval_v3_LOG.py (CPU only).

  validate_run.py <run_dir> --marker <file> --sweep-record <json> [--out <report.json>]

 1 outputs: the run directory holds every expected output, and no file or directory under
   ~/Projects/k1_research outside airc2027_replay/ is newer than --marker (so nothing was written
   to eval_results/ or anywhere else in the sweep tree).
 2 PNG round-trip: every frames/f*.png decodes to exactly the array whose SHA-256 frames.jsonl
   logged (all frames), and the in-process reload checks of the first frames passed.
 3 request rebuild: rebuilding every query's request from the saved frames reproduces the logged
   request SHA-256 and each logged PNG SHA-256.
 4 schema: the record JSON has the same keys and value types as the sweep's record.
 Also (consistency): state lines = reset + w0..w199 + 0..N contiguous; each query's state hash
 equals the latest logged state; frame indices contiguous; no error.txt.
Exit code 0 only if every check passes.
"""
import argparse
import json
import os
import subprocess
import sys

import fr_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--marker", required=True)
    ap.add_argument("--sweep-record", required=True)
    ap.add_argument("--out")
    ap.add_argument("--replay-root", default=C.R, help="experiment root (offline tests use a scratch root)")
    a = ap.parse_args()
    rd = os.path.realpath(a.run_dir)
    run = C.load_run(rd)
    rep = {"run_dir": rd}

    # 1 outputs
    rec_idx = run["meta"]["record"] if run["meta"] else None
    expected = ["frames", "frames.jsonl", "queries.jsonl", "state_hashes.txt", "state_components.tsv",
                "states_at_queries.npz", "run_meta.json", "state_layout.json",
                f"measurements/{rec_idx}.json", f"videos/output_{rec_idx}.mp4"]
    missing = [e for e in expected if not os.path.exists(os.path.join(rd, e))]
    # Files (not directory mtimes: git's transient lock files touch .git/ directories) newer than
    # the marker anywhere in ~/Projects/k1_research, except this experiment's folder and the
    # walking-check session's own folder (it prepares its code there while waiting for GPU_LOCK).
    replay_root = os.path.realpath(a.replay_root)
    real_root = os.path.realpath(C.R)
    walking_root = os.path.realpath(os.path.join(C.KR, "airc2027_walking"))
    newer = subprocess.run(["find", os.path.realpath(C.KR), "-path", real_root, "-prune", "-o",
                            "-path", replay_root, "-prune", "-o", "-path", walking_root, "-prune", "-o",
                            "-type", "f", "-newer", a.marker, "-print"],
                           capture_output=True, text=True).stdout.split()
    rep["1_outputs"] = {"ok": (not missing) and (not newer) and rd.startswith(replay_root + os.sep),
                        "missing": missing, "files_written_outside_airc2027_replay": newer[:50],
                        "n_files_written_outside": len(newer),
                        "excluded_from_check": sorted({real_root, replay_root, walking_root})}

    # 2 PNG round-trip
    bad, n = [], 0
    for f in run["frames"]:
        arr = C.load_frame(rd, f["k"])
        n += 1
        if C.sha256(arr.tobytes()) != f["sha256"] or list(arr.shape) != f["shape"] or str(arr.dtype) != f["dtype"]:
            bad.append(f["k"])
    inproc = [f.get("png_reload_verified") for f in run["frames"] if "png_reload_verified" in f]
    ks = [f["k"] for f in run["frames"]]
    rep["2_png_roundtrip"] = {"ok": n > 0 and not bad and len(inproc) == min(10, n) and all(inproc),
                              "frames_checked": n, "mismatching_frames": bad,
                              "in_process_checks": len(inproc), "in_process_all_true": all(inproc),
                              "k_contiguous": ks == list(range(len(ks)))}

    # 3 request rebuild
    bad_q = []
    for q in run["queries"]:
        if q.get("request_sha256") is None:
            bad_q.append({"i": q["query_index"], "why": "no request logged"})
            continue
        data, pngs = C.rebuild_query_request(rd, q)
        if C.sha256(data) != q["request_sha256"] or pngs != q["png_sha256"] or len(data) != q["request_len"]:
            bad_q.append({"i": q["query_index"], "rebuilt": C.sha256(data), "logged": q["request_sha256"]})
    rep["3_request_rebuild"] = {"ok": len(run["queries"]) > 0 and not bad_q,
                                "queries_checked": len(run["queries"]), "failures": bad_q}

    # 4 schema
    sweep = json.load(open(a.sweep_record))
    rec = run["record"] or {}
    tdiff = {k: [type(sweep[k]).__name__, type(rec[k]).__name__] for k in sweep
             if k in rec and type(sweep[k]) is not type(rec[k])}
    rep["4_schema"] = {"ok": set(rec) == set(sweep) and not tdiff,
                       "missing_keys": sorted(set(sweep) - set(rec)), "extra_keys": sorted(set(rec) - set(sweep)),
                       "type_differences": tdiff, "record": rec, "sweep_record": sweep}

    # consistency
    labels = [lab for lab, _ in run["states"]]
    n_main = len(labels) - 201
    exp_labels = ["reset"] + [f"w{i}" for i in range(200)] + [str(i) for i in range(max(n_main, 0))]
    last_by_label = dict(run["states"])
    q_ok = all(q["state_hash"] == last_by_label.get(q["state_label_before"]) for q in run["queries"])
    q_prev_ok = all((q["state_label_before"] == ("w199" if q["num_steps"] == 0 else str(q["num_steps"] - 1)))
                    for q in run["queries"])
    end_ok = rec.get("ended_at_step") is not None and n_main in (rec["ended_at_step"] + 1, rec["ended_at_step"])
    rep["consistency"] = {"ok": labels == exp_labels and q_ok and q_prev_ok and end_ok and ks == list(range(len(ks)))
                          and not os.path.exists(os.path.join(rd, "error.txt")),
                          "state_labels_as_expected": labels == exp_labels, "n_state_lines": len(labels),
                          "main_loop_lines": n_main, "ended_at_step": rec.get("ended_at_step"),
                          "query_state_hash_equals_previous_step": q_ok, "query_previous_label_ok": q_prev_ok,
                          "error_txt": os.path.exists(os.path.join(rd, "error.txt"))}
    rep["all_ok"] = all(rep[k]["ok"] for k in ("1_outputs", "2_png_roundtrip", "3_request_rebuild",
                                                  "4_schema", "consistency"))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=2)
    for k in ("1_outputs", "2_png_roundtrip", "3_request_rebuild", "4_schema", "consistency"):
        print(f"{k}: {'PASS' if rep[k]['ok'] else 'FAIL'}  " +
              json.dumps({kk: vv for kk, vv in rep[k].items() if kk not in ("record", "sweep_record")})[:600])
    print("ALL_OK" if rep["all_ok"] else "VALIDATION_FAILED")
    sys.exit(0 if rep["all_ok"] else 1)


if __name__ == "__main__":
    main()
