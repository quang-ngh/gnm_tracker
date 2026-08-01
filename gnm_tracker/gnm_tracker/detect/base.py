"""Detector-agnostic landmark container.

Holds one frame's 2D landmarks plus per-landmark confidence, so the rest of the
tracker never depends on MediaPipe types directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# MediaPipe canonical indices (subject/anatomical sides, verified in the gist).
MP_EYE_OUTER_RIGHT = 33
MP_EYE_OUTER_LEFT = 263
MP_MOUTH_CORNER_RIGHT = 61
MP_MOUTH_CORNER_LEFT = 291


@dataclasses.dataclass
class FrameLandmarks:
    """Detected 2D landmarks for a single frame.

    Attributes:
      xy: ``(L, 2)`` landmark positions in pixels.
      z: ``(L,)`` normalized depth (MediaPipe, roughly image-width units).
      presence: ``(L,)`` in [0, 1] (landmark is present / not occluded).
      visibility: ``(L,)`` in [0, 1] (landmark is visible / in-frame).
      image_hw: ``(H, W)`` of the source frame.
      transform: ``(4, 4)`` facial transformation matrix, or None.
      blendshapes: optional dict of the 52 blendshape scores (weak init only).
      detected: whether a face was found at all.
    """

    xy: np.ndarray
    z: np.ndarray
    presence: np.ndarray
    visibility: np.ndarray
    image_hw: tuple[int, int]
    transform: np.ndarray | None = None
    blendshapes: dict[str, float] | None = None
    detected: bool = True

    @property
    def num_landmarks(self) -> int:
        return int(self.xy.shape[0])

    def confidence(self) -> np.ndarray:
        """Per-landmark confidence ``c_i`` for the reprojection loss (Section 6).

        Combines presence and visibility; both default to 1 when MediaPipe does
        not populate them (it often leaves them at 0 for the face mesh, in which
        case we treat all landmarks as confident).
        """
        p = self.presence
        v = self.visibility
        if np.all(p == 0) and np.all(v == 0):
            return np.ones(self.num_landmarks, dtype=np.float32)
        return (np.minimum(np.where(p > 0, p, 1.0), np.where(v > 0, v, 1.0))
                .astype(np.float32))

    def interocular_px(self) -> float:
        """Outer-eye-corner distance in pixels — the normalization length scale."""
        d = np.linalg.norm(self.xy[MP_EYE_OUTER_RIGHT] - self.xy[MP_EYE_OUTER_LEFT])
        return float(max(d, 1e-6))

    @classmethod
    def empty(cls, num_landmarks: int, image_hw: tuple[int, int]) -> "FrameLandmarks":
        """A 'no face detected' placeholder (confidence 0 everywhere)."""
        z = np.zeros(num_landmarks, dtype=np.float32)
        return cls(
            xy=np.zeros((num_landmarks, 2), dtype=np.float32),
            z=z,
            presence=z.copy(),
            visibility=z.copy(),
            image_hw=image_hw,
            detected=False,
        )
