"""Step 4 input: rebuild the two requests at q* (run A's and the other run's) from the saved frames,
check each against the logged request SHA-256 (and per-PNG SHA-256), and write
  <out_dir>/requests/<run>_q<i>.bin  and  <out_dir>/requests.json
Exit code 1 if any rebuilt request does not match its logged hash.

  make_qstar_requests.py <runA> <runX> <out_dir>
"""
import json
import os
import sys

import fr_common as C
from compare_runs import compare_pair


def main():
    run_a, run_x, out = sys.argv[1], sys.argv[2], sys.argv[3]
    A, X = C.load_run(run_a), C.load_run(run_x)
    res = compare_pair(A, X)
    if not res["diverged"]:
        print("no q* (replies never differ)")
        sys.exit(1)
    qs = res["qstar"]["i"]
    os.makedirs(os.path.join(out, "requests"), exist_ok=True)
    spec = {"qstar": qs, "step_A": res["qstar"]["step_A"], "step_X": res["qstar"]["step_X"], "requests": []}
    ok = True
    for run in (A, X):
        q = run["queries"][qs]
        data, pngs = C.rebuild_query_request(run["dir"], q)
        sha = C.sha256(data)
        match = sha == q["request_sha256"] and pngs == q["png_sha256"]
        ok &= match
        rel = f"requests/{run['name']}_q{qs:03d}.bin"
        with open(os.path.join(out, rel), "wb") as fh:
            fh.write(data)
        spec["requests"].append({"label": f"{run['name']}_q{qs:03d}", "run": run["name"], "path": rel,
                                 "sha256": sha, "logged_request_sha256": q["request_sha256"],
                                 "rebuilt_matches_log": match, "logged_reply": q["reply"],
                                 "num_steps": q["num_steps"], "indices": q["indices"], "bytes": len(data)})
        print(f"{run['name']} q{qs}: rebuilt {sha[:16]} logged {q['request_sha256'][:16]} match={match}")
    spec["requests_identical"] = spec["requests"][0]["sha256"] == spec["requests"][1]["sha256"]
    with open(os.path.join(out, "requests.json"), "w") as fh:
        json.dump(spec, fh, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
