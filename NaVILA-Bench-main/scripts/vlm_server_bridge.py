"""Drop-in replacement for scripts/vlm_server.py that uses our existing
NaVILA install (in the `booster` conda env, NOT vlnce-isaac).

The benchmark's vlm_server.py imports `llava` from the NaVILA repo. We
already have llava working in the `booster` env, so this bridge runs
*there*. The TCP wire format matches the benchmark's expectations:

    [8 bytes BE size][JSON utf-8 payload]
    payload = {"images": [<base64-JPEG>, ...], "query": "<instruction>"}

Response (same framing): {"text": "<navila output>"}  — actually
the benchmark sends back a bare string, so this server does the same.

Why a separate process: vlnce-isaac env's torch / cuda stack works with
Isaac Sim 4.1 + RTX 5090 but doesn't have llava installed. We don't want
to mix the two installs. Run navila_eval inside vlnce-isaac; run this
inside `booster`. They talk over TCP.

Usage:

    conda activate booster
    python scripts/vlm_server_bridge.py \\
        --model_path ~/Projects/k1_research/booster/NaVILA/checkpoints/navila-llama3-8b-8f \\
        --port 54321
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Bring the NaVILA repo on sys.path so `import llava` resolves.
NAVILA_REPO = Path.home() / "Projects/k1_research/booster/NaVILA"
sys.path.insert(0, str(NAVILA_REPO))
EXPERIMENTS = Path.home() / "Projects/k1_research/experiments/navila"
sys.path.insert(0, str(EXPERIMENTS))


def load_navila(model_path: str, load_8bit: bool = False):
    """Load NaVILA via llava's load_pretrained_model.

    load_8bit: bitsandbytes int8 — drops the 8B VLM ~17 GB -> ~9 GB so it can
    co-reside with the Isaac render on a 24 GB card (lab 3090). Near-lossless
    for short greedy action decoding; default off (32 GB boxes run fp16).
    """
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    disable_torch_init()
    print(f"[bridge] loading NaVILA from {model_path}", flush=True)
    name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base=None, model_name=name, attn_implementation="sdpa",
        load_8bit=load_8bit,
    )
    print("[bridge] NaVILA ready.", flush=True)
    return tokenizer, model, image_processor


def run_inference(tokenizer, model, image_processor, frames, instruction,
                  max_new_tokens=1024, num_video_frames=8):
    """One NaVILA forward pass. Matches the reference run_navigation.py
    prompt format byte-for-byte."""
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria, process_images, tokenizer_image_token,
    )
    images_tensor = process_images(frames, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    image_token = "<image>\n"
    qs = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video "
        f"of historical observations {image_token * (num_video_frames - 1)}, "
        f"and current observation <image>\n. "
        f'Your assigned task is: "{instruction}" '
        "Analyze this series of images to decide your next action, which could be "
        "turning left or right by a specific degree, moving forward a certain "
        "distance, or stop if the task is completed."
    )
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    with torch.inference_mode():
        t0 = time.perf_counter()
        out_ids = model.generate(
            input_ids,
            images=[images_tensor.half().to(model.device)],
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
    out = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if out.endswith(stop_str):
        out = out[: -len(stop_str)].strip()
    print(f"[bridge] inf {dt_ms:5.0f} ms  -> {out!r}", flush=True)
    return out


def decode_images(b64_list):
    out = []
    for b in b64_list:
        try:
            out.append(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))
        except Exception as e:
            print(f"[bridge] base64 decode error: {e!r}", flush=True)
            out.append(Image.new("RGB", (224, 224), (0, 0, 0)))
    return out


def serve(host, port, tokenizer, model, image_processor, num_video_frames):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"[bridge] listening on {host}:{port}", flush=True)

    while True:
        conn, addr = s.accept()
        try:
            # 8-byte BE size header
            size_data = b""
            while len(size_data) < 8:
                chunk = conn.recv(8 - len(size_data))
                if not chunk:
                    break
                size_data += chunk
            if len(size_data) != 8:
                continue
            size = int.from_bytes(size_data, "big")

            buf = b""
            while len(buf) < size:
                chunk = conn.recv(min(4096, size - len(buf)))
                if not chunk:
                    break
                buf += chunk
            req = json.loads(buf.decode("utf-8"))
            images = decode_images(req["images"])
            text = run_inference(
                tokenizer, model, image_processor, images, req["query"],
                num_video_frames=num_video_frames,
            )

            # Response: bare string (vlm_server.py returns outputs directly).
            resp_bytes = json.dumps(text).encode("utf-8")
            conn.sendall(len(resp_bytes).to_bytes(8, "big"))
            conn.sendall(resp_bytes)
        except (ConnectionError, OSError) as e:
            print(f"[bridge] client error: {e!r}", flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=54321)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--num_video_frames", type=int, default=8)
    ap.add_argument("--load_8bit", action="store_true",
                    help="int8 quantized load (fits 24 GB alongside Isaac; see load_navila)")
    args = ap.parse_args()

    tokenizer, model, image_processor = load_navila(args.model_path, load_8bit=args.load_8bit)
    serve(args.host, args.port, tokenizer, model, image_processor,
          args.num_video_frames)


if __name__ == "__main__":
    main()
