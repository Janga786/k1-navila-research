"""Step 4, conditions 2 and 3: send saved requests to the running model server with no simulator
process. Each request in requests.json is sent N times in a row (file order) and every reply is
logged. _vlm_send_bytes is a verbatim copy of the function in navila_eval_v3_LOG.py (itself the
evaluator's socket code moved into a function); test_offline.py checks that the two are identical.
Imports only the standard library: this process never touches the GPU.

  replay_client.py <requests.json> <N> <out.jsonl> <condition_label> [--port 54321]
"""
import hashlib
import json
import os
import socket
import sys
import time


def _sha256(b):
    return hashlib.sha256(b).hexdigest()


def _vlm_send_bytes(data_bytes, vlm_host, vlm_port, log=None):
    """[LOG] The evaluator's socket exchange, moved verbatim out of sample_images_and_send_to_vlm
    so that --replay_queries sends saved requests through the very same code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((vlm_host, vlm_port))
        s.settimeout(120.0)  # bound every recv: a VLM crash/OOM must not hang the eval forever (rank 4)
        s.sendall(len(data_bytes).to_bytes(8, 'big'))
        s.sendall(data_bytes)
        size_data = b''                       # recv(8) may return <8 bytes on a partial read (rank 12)
        while len(size_data) < 8:
            chunk = s.recv(8 - len(size_data))
            if not chunk:
                raise ConnectionError("VLM closed connection before sending the size header")
            size_data += chunk
        size = int.from_bytes(size_data, 'big')
        response_data = b''
        while len(response_data) < size:
            packet = s.recv(4096)
            if not packet:
                break
            response_data += packet
        if log is not None:  # [LOG]
            log["response_sha256"] = _sha256(response_data)
            log["response_len"] = len(response_data)
        return json.loads(response_data.decode())


def main():
    args = sys.argv[1:]
    port = 54321
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i:i + 2]
    spec_path, n, out_path, cond = args[0], int(args[1]), args[2], args[3]
    spec = json.load(open(spec_path))
    base = os.path.dirname(os.path.abspath(spec_path))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", buffering=1) as out:
        for req in spec["requests"]:
            with open(os.path.join(base, req["path"]), "rb") as fh:
                data = fh.read()
            sha = _sha256(data)
            if sha != req["sha256"]:
                raise SystemExit(f"request {req['label']}: sha256 {sha} != {req['sha256']}")
            for i in range(n):
                log = {}
                t0 = time.time()
                reply = _vlm_send_bytes(data, "localhost", port, log=log)
                out.write(json.dumps({"condition": cond, "label": req["label"], "i": i, "request_sha256": sha,
                                      "reply": reply, "response_sha256": log.get("response_sha256"),
                                      "wall_s": round(time.time() - t0, 4), "t_unix": t0}) + "\n")
                print(cond, req["label"], i, repr(reply), flush=True)


if __name__ == "__main__":
    main()
