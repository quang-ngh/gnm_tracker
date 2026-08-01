"""The correspondence self-check must pass on our GNM build (Section 10).

Runs on GNM's NumPy backend (no torch). This is the eyelid-gotcha guard.
"""

import numpy as np
import pytest

pytest.importorskip("gnm")

from gnm_tracker.config import resolve_path  # noqa: E402
from gnm_tracker.validate import (  # noqa: E402
    gnm_neutral_vertices,
    run_correspondence_self_check,
)


def test_self_check_passes():
    assert run_correspondence_self_check() is True


def test_dense_468_shape_and_weights():
    data = np.loadtxt(resolve_path("data/correspondence/head_dense_468.txt"))
    assert data.shape == (468, 6)
    weights = data[:, 1::2]
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-3)


def test_landmarks_land_in_face_region():
    verts = gnm_neutral_vertices()
    data = np.loadtxt(resolve_path("data/correspondence/head_dense_468.txt"))
    idx, w = data[:, ::2].astype(int), data[:, 1::2]
    lm = (verts[idx] * w[..., None]).sum(axis=1)
    # inter-ocular distance is a few cm on the neutral template (meters).
    interocular = np.linalg.norm(lm[33] - lm[263])
    assert 0.03 < interocular < 0.12
