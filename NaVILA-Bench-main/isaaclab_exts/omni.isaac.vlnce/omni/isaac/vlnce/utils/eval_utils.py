from typing import List, Optional

import cv2
import attr
import textwrap
import gzip
import json
import re
import numpy as np


@attr.s(auto_attribs=True)
class InstructionData:
    instruction_text: str
    instruction_tokens: Optional[List[str]] = None


def skip(*args, **kwargs):
    pass


def read_episodes(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    return data["episodes"]


def add_instruction_on_img(img: np.ndarray, text: str, start_y=0) -> None:
    font_size = 0.6
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(
        text, width=int((img.shape[1] - 15) / char_size[0])
    )
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = start_y
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def get_vel_command(text):
    # Angular rate is fixed at wz = pi/6 rad/s = 30 deg/s; the turn ANGLE is encoded
    # purely in time_to_go (= degrees / 30). NaVILA emits arbitrary angles (e.g. "90 deg"),
    # so we EXTRACT the number rather than bucketing to {15,30,45} — the old buckets
    # silently executed any other angle (60/90/120/180) as 15 deg (audit rank 3).
    t = text.lower()
    if "turn left" in t or "turn right" in t:
        sign = 1.0 if "turn left" in t else -1.0
        m = re.search(r"(\d+)", t)                       # first integer = the angle in deg
        degrees = int(m.group(1)) if m else 45           # numberless turn -> 45 deg default
        time_to_go = degrees / np.degrees(np.pi / 6.0)   # = degrees / 30.0
        return [0.0, 0.0, sign * np.pi / 6.0], time_to_go
    elif "move forward" in t:                            # NaVILA quantizes forward to 25/50/75 cm
        if "75" in t:
            return [0.5, 0.0, 0.0], 1.5
        elif "50" in t:
            return [0.5, 0.0, 0.0], 1.0
        elif "25" in t:
            return [0.5, 0.0, 0.0], 0.5
        return [0.5, 0.0, 0.0], 0.5
    elif "stop" in t:
        return [0.0, 0.0, 0.0], 0.0
    else:
        # Unrecognized output (incl. "do not move", "turn around", truncated text):
        # HOLD and re-query — do NOT blindly walk forward 0.25 m (audit ranks 7, 8).
        print(f"[PARSE_FAIL] unrecognized VLM output: {text!r}", flush=True)
        return [0.0, 0.0, 0.0], 0.5