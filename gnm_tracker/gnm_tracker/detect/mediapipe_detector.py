"""MediaPipe Face Landmarker (Tasks API) wrapper.

Produces 468 canonical landmarks + per-landmark presence/visibility + the face
transform matrix, and optionally the 52 blendshapes. Per the plan (Section 3),
blendshapes are a *weak initializer only*, never treated as ground-truth
expression (Guardrail: do not treat the 52 blendshapes as ground truth).

MediaPipe's mesh returns 478 points (468 canonical + 10 iris); we keep the first
468 to match the vendored `head_dense_468.txt` correspondence.

The `.task` model bundle must be present. Download once:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
and pass its path (or drop it at data/mediapipe/face_landmarker.task).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gnm_tracker.config import resolve_path
from gnm_tracker.detect.base import FrameLandmarks

NUM_CANONICAL = 468
_DEFAULT_MODEL = "data/mediapipe/face_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class MediaPipeDetector:
    """Detects 468 face landmarks per frame."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        num_faces: int = 1,
        output_blendshapes: bool = True,
        running_mode: str = "video",
    ):
        # Import here so importing this module never requires mediapipe.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = resolve_path(model_path or _DEFAULT_MODEL)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {model_path}.\n"
                f"Download it once with:\n  curl -Lo {model_path} {_MODEL_URL}"
            )

        self._vision = mp_vision
        mode_map = {
            "image": mp_vision.RunningMode.IMAGE,
            "video": mp_vision.RunningMode.VIDEO,
        }
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mode_map[running_mode],
            num_faces=num_faces,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mode = running_mode

    # ------------------------------------------------------------------
    def _to_frame_landmarks(self, result, image_hw: tuple[int, int]) -> FrameLandmarks:
        h, w = image_hw
        if not result.face_landmarks:
            return FrameLandmarks.empty(NUM_CANONICAL, image_hw)

        lms = result.face_landmarks[0][:NUM_CANONICAL]
        xy = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        z = np.array([lm.z for lm in lms], dtype=np.float32)
        presence = np.array([getattr(lm, "presence", 0.0) or 0.0 for lm in lms], np.float32)
        visibility = np.array([getattr(lm, "visibility", 0.0) or 0.0 for lm in lms], np.float32)

        transform = None
        if getattr(result, "facial_transformation_matrixes", None):
            transform = np.asarray(result.facial_transformation_matrixes[0], np.float32)

        blendshapes = None
        if getattr(result, "face_blendshapes", None):
            blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

        return FrameLandmarks(
            xy=xy, z=z, presence=presence, visibility=visibility,
            image_hw=image_hw, transform=transform, blendshapes=blendshapes,
            detected=True,
        )

    def _mp_image(self, rgb: np.ndarray):
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    # ------------------------------------------------------------------
    def detect_image(self, rgb: np.ndarray) -> FrameLandmarks:
        """Single frame (IMAGE mode)."""
        result = self._landmarker.detect(self._mp_image(rgb))
        return self._to_frame_landmarks(result, rgb.shape[:2])

    def detect_video(self, frames: np.ndarray, fps: float) -> list[FrameLandmarks]:
        """Sequence (VIDEO mode). frames: ``(T, H, W, 3)`` uint8 RGB."""
        out: list[FrameLandmarks] = []
        dt_ms = 1000.0 / max(fps, 1e-6)
        for i, frame in enumerate(frames):
            ts = int(round(i * dt_ms))
            result = self._landmarker.detect_for_video(self._mp_image(frame), ts)
            out.append(self._to_frame_landmarks(result, frame.shape[:2]))
        return out

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def stack_detections(dets: list[FrameLandmarks]) -> dict[str, np.ndarray]:
    """Stack a list of FrameLandmarks into batched arrays for the fitter.

    Returns dict with:
      xy ``(T, L, 2)``, conf ``(T, L)``, valid ``(T,)`` bool,
      interocular ``(T,)`` pixels.
    """
    xy = np.stack([d.xy for d in dets]).astype(np.float32)
    conf = np.stack([d.confidence() for d in dets]).astype(np.float32)
    detected = np.array([d.detected for d in dets], dtype=bool)
    interocular = np.array(
        [d.interocular_px() if d.detected else 1.0 for d in dets], dtype=np.float32
    )
    # Zero the confidence of undetected frames so they contribute no gradient.
    conf = conf * detected[:, None]
    return {"xy": xy, "conf": conf, "detected": detected, "interocular": interocular}
