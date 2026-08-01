"""Export schema round-trips and pose layout (Section 7). NumPy only."""

import types

import numpy as np

from gnm_tracker.export.schema import ClipRecord


def _fake_result(t=5, n_id=8, n_expr=6, n_joints=4):
    params = {
        "shape": np.random.randn(n_id).astype(np.float32),
        "expression": np.random.randn(t, n_expr).astype(np.float32),
        "rotations": np.random.randn(t, n_joints, 3).astype(np.float32),
        "translation": np.random.randn(t, 3).astype(np.float32),
    }
    return types.SimpleNamespace(
        params=params,
        neutral=np.zeros(n_expr, np.float32),
        valid=np.array([True, True, False, True, True]),
        reproj_median=0.02,
        psi_cleanliness={"mean_drift": 0.01},
        psi_clean_pass=True,
    )


def test_pose_layout(cfg):
    r = _fake_result()
    rec = ClipRecord.from_sequence_result("clip0", r, cfg)
    # pose = J*3 rotations + 3 translation
    assert rec.pose.shape == (5, 4 * 3 + 3)
    assert rec.shape.shape == (8,)          # single identity vector, not per-frame
    assert rec.expr.shape == (5, 6)
    assert rec.num_valid == 4


def test_npz_roundtrip(tmp_path, cfg):
    r = _fake_result()
    rec = ClipRecord.from_sequence_result("clip0", r, cfg)
    p = tmp_path / "rec.npz"
    np.savez_compressed(p, **rec.to_npz_dict())
    data = np.load(p, allow_pickle=True)
    assert data["expr"].shape == rec.expr.shape
    assert data["pose"].shape == rec.pose.shape
    assert bool(data["valid"][2]) is False
    assert "audio_feat" not in data.files  # deferred, not written in v1
