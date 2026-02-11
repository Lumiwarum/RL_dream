from __future__ import annotations

import os
from typing import List

import imageio.v2 as imageio
import numpy as np


def write_mp4(frames: List[np.ndarray], out_path: str, fps: int = 30) -> None:
    """
    Write a list of RGB frames (H,W,3 uint8) to an MP4 using imageio-ffmpeg.
    """
    if len(frames) == 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Ensure uint8
    frames_u8 = []
    for f in frames:
        if f is None:
            continue
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        frames_u8.append(f)
    if len(frames_u8) == 0:
        return
    imageio.mimsave(out_path, frames_u8, fps=int(fps))
