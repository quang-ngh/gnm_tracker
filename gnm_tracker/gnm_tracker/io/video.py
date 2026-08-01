"""Video decode + frame extraction (Section 4, io/).

Uses imageio/ffmpeg to read RGB frames and source fps. One face per clip is
assumed (Section 1, not-in-scope: multi-face). Frames are returned as a single
``(T, H, W, 3)`` uint8 RGB array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_video(
    path: str | Path,
    max_frames: int | None = None,
    long_side: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return ``(frames (T, H, W, 3) uint8 RGB, fps)``.

    Args:
      path: video file.
      max_frames: optionally cap the number of frames read.
      long_side: if set, downscale so the longer image side equals this (speed).
    """
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))

    frames: list[np.ndarray] = []
    for i, frame in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break
        frame = np.asarray(frame)
        if frame.ndim == 2:  # grayscale -> RGB
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:  # RGBA -> RGB
            frame = frame[..., :3]
        if long_side is not None:
            frame = _resize_long_side(frame, long_side)
        frames.append(frame)
    reader.close()

    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8), fps


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return frame
    new = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(frame, new, interpolation=cv2.INTER_AREA)


def write_video(path: str | Path, frames, fps: float = 25.0) -> None:
    """Write a list/array of ``(H, W, 3)`` uint8 RGB frames to an mp4."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
