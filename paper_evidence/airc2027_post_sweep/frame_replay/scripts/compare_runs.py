"""Compare two logged runs of one episode (CPU only).

  compare_runs.py pair <runA> <runX> [--out F] [--images]   A vs B/C/D: q*, first differences, H1
  compare_runs.py h3   <runA> <runR> [--out F]              run R vs run A: H3

Definitions (pre-registered in PLAN.md):
  reply      = the model's decoded reply text; two replies differ if the strings differ.
  request    = the request bytes; two requests differ if their SHA-256 differ.
  state      = SHA-256 over the float32 bytes of root state, joint positions, joint velocities,
               all body states, joint position targets and the policy observation history
               (state_hashes.txt; labels: reset, w0..w199 = warm-up steps, 0..N = main loop).
  q*         = the first query index i (present in both runs) whose replies differ.
  step of q* = num_steps of query q* (the main-loop step whose env.step follows the query).
H1 for a pair: (a) every state before the step of q* is identical (same labels, same hashes);
  (b) the state hash logged at query q* is identical; (c) the requests at q* differ.
  held = a and b and c; failed_state_before_qstar = not (a and b);
  failed_requests_identical = a and b and not c.
"""
import argparse
import json
import sys

import fr_common as C


def _first_state_diff(sa, sx):
    """First position where the ordered state logs differ (label or hash), or None."""
    for i in range(max(len(sa), len(sx))):
        if i >= len(sa):
            return {"pos": i, "label": sx[i][0], "kind": "only_in_X (A ended)"}
        if i >= len(sx):
            return {"pos": i, "label": sa[i][0], "kind": "only_in_A (X ended)"}
        if sa[i][0] != sx[i][0]:
            return {"pos": i, "label": sa[i][0], "label_X": sx[i][0], "kind": "label"}
        if sa[i][1] != sx[i][1]:
            return {"pos": i, "label": sa[i][0], "kind": "hash"}
    return None


def _label_before_step(lab, step):
    """True if state label `lab` is a control step before main-loop step `step`."""
    if lab == "reset" or lab.startswith("w"):
        return True
    return int(lab) < step


def _summary(run):
    rec = run["record"] or {}
    return {"name": run["name"], "ended_at_step": rec.get("ended_at_step"), "term_reason": rec.get("term_reason"),
            "success": rec.get("success"), "n_queries": len(run["queries"]), "n_frames": len(run["frames"]),
            "n_state_lines": len(run["states"]), "fields": {f: rec.get(f) for f in C.SIX}}


def compare_pair(A, X, images=False):
    qa, qx = A["queries"], X["queries"]
    n = min(len(qa), len(qx))
    timeline, qstar, first_req = [], None, None
    for i in range(n):
        req_same = qa[i]["request_sha256"] == qx[i]["request_sha256"]
        rep_same = qa[i]["reply"] == qx[i]["reply"]
        timeline.append({"i": i, "step_A": qa[i]["num_steps"], "step_X": qx[i]["num_steps"],
                         "request_same": req_same, "reply_same": rep_same,
                         "state_at_query_same": qa[i]["state_hash"] == qx[i]["state_hash"]})
        if first_req is None and not req_same:
            first_req = i
        if qstar is None and not rep_same:
            qstar = i
    lim = qstar if qstar is not None else n
    out = {"A": _summary(A), "X": _summary(X), "n_common_queries": n,
           "query_counts_differ": len(qa) != len(qx),
           "first_request_diff": None if first_req is None else
           {"i": first_req, "step_A": qa[first_req]["num_steps"], "step_X": qx[first_req]["num_steps"]},
           "n_requests_differ_replies_match_before_qstar": sum(1 for t in timeline[:lim] if not t["request_same"]),
           "n_queries_before_qstar": lim,
           "first_state_diff": _first_state_diff(A["states"], X["states"]),
           "records_equal_six_fields": all((A["record"] or {}).get(f) == (X["record"] or {}).get(f) for f in C.SIX),
           "timeline": timeline, "qstar": None, "H1": None}
    if qstar is None:
        out["diverged"] = False
        return out
    out["diverged"] = True
    a, x = qa[qstar], qx[qstar]
    s_a, s_x = a["num_steps"], x["num_steps"]
    fsd = out["first_state_diff"]
    states_before_same = (s_a == s_x) and (fsd is None or not _label_before_step(fsd["label"], s_a))
    state_at_q_same = a["state_hash"] == x["state_hash"]
    req_differ = a["request_sha256"] != x["request_sha256"]
    frames = []
    for j, (ka, kx) in enumerate(zip(a["indices"], x["indices"])):
        fa = next((f for f in A["frames"] if f["k"] == ka), None)
        fx = next((f for f in X["frames"] if f["k"] == kx), None)
        fr = {"slot": j, "k_A": ka, "k_X": kx, "label_A": fa and fa["label"], "label_X": fx and fx["label"],
              "frame_sha_same": bool(fa and fx and fa["sha256"] == fx["sha256"]),
              "png_sha_same": a["png_sha256"][j] == x["png_sha256"][j]}
        if images and ka >= 0 and kx >= 0:
            fr.update(C.img_diff_stats(C.load_frame(A["dir"], ka), C.load_frame(X["dir"], kx)))
        frames.append(fr)
    out["qstar"] = {"i": qstar, "step_A": s_a, "step_X": s_x, "reply_A": a["reply"], "reply_X": x["reply"],
                    "cmd_A": a["parsed_cmd"], "cmd_X": x["parsed_cmd"],
                    "request_sha_A": a["request_sha256"], "request_sha_X": x["request_sha256"],
                    "state_hash_A": a["state_hash"], "state_hash_X": x["state_hash"],
                    "frames": frames, "n_frames_differ": sum(1 for f in frames if not f["frame_sha_same"])}
    if states_before_same and state_at_q_same:
        verdict = "held" if req_differ else "failed_requests_identical"
    else:
        verdict = "failed_state_before_qstar"
    out["H1"] = {"verdict": verdict, "a_states_before_step_identical": states_before_same,
                 "b_state_at_qstar_identical": state_at_q_same, "c_requests_at_qstar_differ": req_differ,
                 "first_state_diff_relative": ("none before q*" if states_before_same else fsd)}
    return out


def check_h3(A, Rr):
    """H3: R vs A — byte-identical requests and identical replies at every query, bit-identical
    state at every step, final record equal in the six fields."""
    qa, qr = A["queries"], Rr["queries"]
    req = len(qa) == len(qr) and all(a["request_sha256"] == r["request_sha256"] for a, r in zip(qa, qr))
    rep = len(qa) == len(qr) and all(a["reply"] == r["reply"] for a, r in zip(qa, qr))
    st = A["states"] == Rr["states"]
    rec_a, rec_r = A["record"] or {}, Rr["record"] or {}
    fields = {f: {"A": rec_a.get(f), "R": rec_r.get(f), "equal": rec_a.get(f) == rec_r.get(f)} for f in C.SIX}
    rec_eq = all(v["equal"] for v in fields.values())
    first_q = next((i for i, (a, r) in enumerate(zip(qa, qr))
                    if a["request_sha256"] != r["request_sha256"] or a["reply"] != r["reply"]), None)
    frames_from_A = all(f.get("sha256") == f.get("A_sha256") for f in Rr["frames"]) if Rr["frames"] else False
    return {"A": _summary(A), "R": _summary(Rr),
            "requests_identical_all": req, "replies_identical_all": rep, "states_identical_all": st,
            "record_six_fields_equal": rec_eq, "fields": fields,
            "replay_mismatch": Rr["mismatch"], "first_query_diff": first_q,
            "first_state_diff": _first_state_diff(A["states"], Rr["states"]),
            "R_history_frames_equal_A_frames": frames_from_A,
            "H3": "held" if (req and rep and st and rec_eq and Rr["mismatch"] is None) else "failed"}


def check_h2(requests_json, replies_jsonl, n_expected=20):
    """H2 for one episode under one replay condition: each run's request at q* gets that run's own
    (logged) reply in n_expected of n_expected sends. Reports the full reply distribution."""
    spec = json.load(open(requests_json))
    rows = C.load_jsonl(replies_jsonl)
    per = []
    for req in spec["requests"]:
        mine = [r for r in rows if r["label"] == req["label"]]
        dist = {}
        for r in mine:
            dist[r["reply"]] = dist.get(r["reply"], 0) + 1
        own = dist.get(req["logged_reply"], 0)
        per.append({"label": req["label"], "logged_reply": req["logged_reply"], "sends": len(mine),
                    "own_reply_count": own, "distribution": dist,
                    "request_sha_ok": all(r["request_sha256"] == req["sha256"] for r in mine)})
    held = all(p["sends"] == n_expected and p["own_reply_count"] == n_expected and p["request_sha_ok"] for p in per)
    return {"requests": per, "requests_identical": spec.get("requests_identical"), "H2": "held" if held else "failed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "h3", "h2"])
    ap.add_argument("run_a", help="pair/h3: run A dir; h2: requests.json")
    ap.add_argument("run_x", help="pair: other run; h3: run R; h2: replies .jsonl")
    ap.add_argument("--out")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--n", type=int, default=20, help="h2: sends per request (20 in the experiment)")
    a = ap.parse_args()
    if a.mode == "h2":
        res = check_h2(a.run_a, a.run_x, n_expected=a.n)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=2)
        json.dump(res, sys.stdout, indent=1)
        print()
        return
    A, X = C.load_run(a.run_a), C.load_run(a.run_x)
    res = compare_pair(A, X, images=a.images) if a.mode == "pair" else check_h3(A, X)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
    short = {k: v for k, v in res.items() if k != "timeline"}
    if a.mode == "pair" and res.get("qstar"):
        short["qstar"] = {k: v for k, v in res["qstar"].items() if k != "frames"}
    json.dump(short, sys.stdout, indent=1)
    print()
    if a.mode == "pair":
        print(f"DIVERGED={int(res['diverged'])}")


if __name__ == "__main__":
    main()
