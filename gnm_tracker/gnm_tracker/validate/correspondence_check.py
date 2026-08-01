"""Correspondence validation — the eyelid gotcha (Section 10).

Before trusting `head_dense_468.txt`, run the vendored gist's self-check on OUR
GNM build. It asserts that structures like the palpebral aperture, mouth width
and inner-lip gap remain resolvable — a landmark embedding can be numerically
valid yet silently collapse upper/lower eyelid landmarks onto the same triangle
(aperture ~0.5 mm instead of ~6 mm), breaking blink signal with no error raised.

If this fails, regenerate the correspondence with its build script against our
topology (`data/correspondence/build_mediapipe_correspondence.py`).

Runs on GNM's NumPy backend — no torch required, so it is cheap to wire into CI.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from gnm_tracker.config import resolve_path

_VENDORED = "data/correspondence/gnm_mediapipe_landmarks.py"


def _load_vendored():
    path = resolve_path(_VENDORED)
    spec = importlib.util.spec_from_file_location("gnm_mp_468", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gnm_neutral_vertices() -> np.ndarray:
    """GNM neutral template vertices ``(17821, 3)`` via the NumPy backend."""
    from gnm.shape import gnm_numpy

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD
    )
    return np.asarray(gnm.template_vertex_positions)


def run_correspondence_self_check() -> bool:
    """Run the vendored self-check on GNM's neutral mesh. Returns True if it passes."""
    mod = _load_vendored()
    vertices = gnm_neutral_vertices()
    return bool(mod.self_check(vertices))
