#!/usr/bin/env python3
"""fp16-vs-int8 replay arbiter for the NaVILA benchmark VLM.

Replays the exact 8-frame payloads saved by --diag (diag/epN/frames/tickNNN/)
through a running vlm_server_bridge (fp16 or int8) and records the answers.
Then `compare` quantifies:
  - int8-replay vs int8-original  -> determinism + (jpg-save vs png-wire) effect
  - fp16-replay vs int8-original  -> the headline "did quantization change the run"
  - fp16-replay vs int8-replay    -> pure precision difference on identical inputs

Usage:
  replay_vlm_arbiter.py run --diag-root <...>/diag --out /tmp/replay_fp16.json [--port 54321]
  replay_vlm_arbiter.py compare --original --a /tmp/replay_int8.json --b /tmp/replay_fp16.json \
      --diag-root <...>/diag --report /tmp/replay_report.md
"""
import argparse
import base64
import glob
import gzip
import importlib.util
import json
import os
import socket
import sys
import time
import types

ASSETS_EP = ("/home/boosterk1/Projects/k1_research/NaVILA-Bench-main/isaaclab_exts/"
             "omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz")
EVAL_UTILS = ("/home/boosterk1/Projects/k1_research/NaVILA-Bench-main/isaaclab_exts/"
              "omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py")


def ask(port, images_b64, query, timeout=300.0):
    req = json.dumps({"images": images_b64, "query": query}).encode()
    s = socket.create_connection(("localhost", port), timeout=15.0)
    s.settimeout(timeout)
    s.sendall(len(req).to_bytes(8, "big"))
    s.sendall(req)
    hdr = b""
    while len(hdr) < 8:
        c = s.recv(8 - len(hdr))
        if not c:
            raise ConnectionError("eof in header")
        hdr += c
    n = int.from_bytes(hdr, "big")
    buf = b""
    while len(buf) < n:
        c = s.recv(min(4096, n - len(buf)))
        if not c:
            break
        buf += c
    s.close()
    return json.loads(buf.decode())


def load_parser():
    for m in ("cv2", "attr"):
        if m not in sys.modules:
            stub = types.ModuleType(m)
            if m == "attr":
                stub.s = lambda *a, **k: (lambda c: c)
                stub.ib = lambda *a, **k: None
            sys.modules[m] = stub
    spec = importlib.util.spec_from_file_location("eu", EVAL_UTILS)
    eu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eu)
    return eu.get_vel_command


def action_key(text, parser):
    cmd, ttg = parser(text)
    return (round(cmd[0], 3), round(cmd[2], 3), round(ttg, 3))


def cmd_run(args):
    eps = json.loads(gzip.open(ASSETS_EP, "rt", encoding="utf-8").read())["episodes"]
    out = {}
    total = done = errs = 0
    ep_dirs = sorted(glob.glob(os.path.join(args.diag_root, "ep*")))
    for d in ep_dirs:
        total += sum(1 for _ in open(os.path.join(d, "ticks.jsonl")))
    print(f"[replay] starting: {len(ep_dirs)} episodes, {total} ticks, port {args.port}",
          flush=True)
    t0 = time.time()
    for d in ep_dirs:
        idx = int(os.path.basename(d)[2:])
        instr = eps[idx]["instruction"]["instruction_text"]
        rows = []
        for line in open(os.path.join(d, "ticks.jsonl")):
            r = json.loads(line)
            fdir = os.path.join(d, r["frames_dir"])
            frames = sorted(glob.glob(os.path.join(fdir, "frame_*.jpg")),
                            key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
            if len(frames) != 8:
                rows.append({"tick": r["tick"], "original": r["vlm_raw"],
                             "replay": None, "note": f"{len(frames)} frames"})
                continue
            b64 = [base64.b64encode(open(f, "rb").read()).decode() for f in frames]
            ans = None
            for attempt in range(3):  # one flaky socket must not kill the pass
                try:
                    ans = ask(args.port, b64, instr)
                    break
                except Exception as e:
                    print(f"[replay] {os.path.basename(d)} tick {r['tick']} "
                          f"attempt {attempt + 1} failed: {e!r}", flush=True)
                    time.sleep(3 + 5 * attempt)
            if ans is None:
                errs += 1
            rows.append({"tick": r["tick"], "original": r["vlm_raw"], "replay": ans})
            done += 1
            if done % 25 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[replay] {done}/{total} ticks  ({rate:.2f}/s, "
                      f"eta {int((total - done) / max(rate, 1e-9))}s)", flush=True)
        out[os.path.basename(d)] = rows
        json.dump(out, open(args.out, "w"))  # checkpoint after every episode
    print(f"[replay] DONE {done}/{total} ({errs} unanswered) -> {args.out}", flush=True)


def agree(a, b):
    return a is not None and b is not None and a.strip() == b.strip()


def cmd_compare(args):
    parser = load_parser()
    A = json.load(open(args.a))   # int8 replay
    B = json.load(open(args.b))   # fp16 replay
    lines = ["# fp16 replay arbiter — results\n"]
    tot = {"n": 0, "i8r_vs_orig": 0, "fp16_vs_orig": 0, "fp16_vs_i8r": 0,
           "act_fp16_vs_i8r": 0, "act_fp16_vs_orig": 0}
    mism = []
    lines.append("| ep | ticks | int8-replay==orig | fp16==orig | fp16==int8-replay | "
                 "fp16 action==int8 action |")
    lines.append("|---|---|---|---|---|---|")
    for ep in sorted(A.keys(), key=lambda e: int(e[2:])):
        ra, rb = A[ep], B[ep]
        n = len(ra)
        c1 = sum(agree(x["replay"], x["original"]) for x in ra)
        c2 = sum(agree(y["replay"], y["original"]) for y in rb)
        c3 = sum(agree(x["replay"], y["replay"]) for x, y in zip(ra, rb))
        ca = 0
        cao = 0
        for x, y in zip(ra, rb):
            try:
                if x["replay"] and y["replay"] and \
                   action_key(x["replay"], parser) == action_key(y["replay"], parser):
                    ca += 1
                if y["replay"] and \
                   action_key(y["replay"], parser) == action_key(x["original"], parser):
                    cao += 1
                elif y["replay"]:
                    mism.append((ep, x["tick"], x["original"], y["replay"]))
            except Exception:
                pass
        lines.append(f"| {ep} | {n} | {c1}/{n} | {c2}/{n} | {c3}/{n} | {ca}/{n} |")
        tot["n"] += n
        tot["i8r_vs_orig"] += c1
        tot["fp16_vs_orig"] += c2
        tot["fp16_vs_i8r"] += c3
        tot["act_fp16_vs_i8r"] += ca
        tot["act_fp16_vs_orig"] += cao
    n = max(tot["n"], 1)
    lines.append("")
    lines.append(f"**TOTALS over {tot['n']} ticks:**")
    lines.append(f"- int8-replay == int8-original (determinism + jpg-vs-png): "
                 f"{tot['i8r_vs_orig']}/{n} = {100 * tot['i8r_vs_orig'] / n:.1f}%")
    lines.append(f"- fp16-replay == int8-original (verbatim): "
                 f"{tot['fp16_vs_orig']}/{n} = {100 * tot['fp16_vs_orig'] / n:.1f}%")
    lines.append(f"- fp16-replay == int8-replay (verbatim, identical inputs): "
                 f"{tot['fp16_vs_i8r']}/{n} = {100 * tot['fp16_vs_i8r'] / n:.1f}%")
    lines.append(f"- fp16 vs int8 ACTION-level match (what the robot would do): "
                 f"{tot['act_fp16_vs_i8r']}/{n} = {100 * tot['act_fp16_vs_i8r'] / n:.1f}%")
    lines.append(f"- fp16 vs ORIGINAL action-level match: "
                 f"{tot['act_fp16_vs_orig']}/{n} = {100 * tot['act_fp16_vs_orig'] / n:.1f}%")
    if mism:
        lines.append(f"\n## Action-level disagreements (fp16 vs original), {len(mism)} total")
        for ep, t, o, r in mism[:40]:
            lines.append(f"- {ep} tick {t}: int8 {o!r}  ->  fp16 {r!r}")
        if len(mism) > 40:
            lines.append(f"- ... and {len(mism) - 40} more")
    report = "\n".join(lines)
    open(args.report, "w").write(report + "\n")
    print(report)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--diag-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--port", type=int, default=54321)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True, help="int8 replay json")
    c.add_argument("--b", required=True, help="fp16 replay json")
    c.add_argument("--report", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
