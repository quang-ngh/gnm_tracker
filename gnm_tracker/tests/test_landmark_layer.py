import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.config import resolve_path  # noqa: E402
from gnm_tracker.model.landmark_layer import BarycentricLandmarks  # noqa: E402


def test_barycentric_matches_manual():
    idx = np.array([[0, 1, 2], [1, 2, 3]])
    w = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    layer = BarycentricLandmarks(idx, w)
    verts = torch.randn(2, 5, 3)
    out = layer(verts)
    expected = (verts[:, idx] * torch.as_tensor(w, dtype=torch.float32)[..., None]).sum(dim=2)
    assert out.shape == (2, 2, 3)
    assert torch.allclose(out, expected, atol=1e-5)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        BarycentricLandmarks(np.array([[0, 1, 2]]), np.array([[0.2, 0.2, 0.2]]))


def test_load_dense_468(cfg):
    layer = BarycentricLandmarks.from_txt(resolve_path(cfg.model.dense_468_file))
    assert layer.num_landmarks == 468
