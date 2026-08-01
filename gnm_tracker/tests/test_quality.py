"""Per-frame quality mask logic (Section 9). compute_validity is NumPy-only."""

import numpy as np
import pytest

from gnm_tracker.quality.mask import compute_validity


def _metrics(t, **overrides):
    m = {
        "reproj_med": np.zeros(t),
        "conf_mean": np.ones(t),
        "jerk": np.zeros(t),
        "yaw_deg": np.zeros(t),
        "pitch_deg": np.zeros(t),
        "detected": np.ones(t, bool),
    }
    m.update(overrides)
    return m


def test_all_pass(cfg):
    valid, _ = compute_validity(_metrics(10), cfg)
    assert valid.all()


def test_high_reproj_dropped(cfg):
    reproj = np.zeros(10)
    reproj[4] = 1.0  # far above threshold
    valid, reasons = compute_validity(_metrics(10, reproj_med=reproj), cfg)
    assert not valid[4]
    assert not reasons["reproj"][4]


def test_extreme_pose_dropped(cfg):
    yaw = np.zeros(10)
    yaw[7] = 89.0
    valid, _ = compute_validity(_metrics(10, yaw_deg=yaw), cfg)
    assert not valid[7]


def test_short_valid_run_dropped(cfg):
    # frames 0,1 valid; frame 2 invalid; frames 3..9 valid (run of 7).
    reproj = np.zeros(10)
    reproj[2] = 1.0
    valid, _ = compute_validity(_metrics(10, reproj_med=reproj), cfg)
    assert not valid[0] and not valid[1]     # run of 2 < min_valid_run -> dropped
    assert valid[3:].all()                    # run of 7 kept


def test_undetected_dropped(cfg):
    det = np.ones(6, bool)
    det[3] = False
    valid, _ = compute_validity(_metrics(6, detected=det), cfg)
    assert not valid[3]


def test_head_pose_zero_is_frontal():
    torch = pytest.importorskip("torch")
    from gnm_tracker.quality.mask import head_pose_angles_deg

    rotations = torch.zeros(4, 4, 3)
    yaw, pitch = head_pose_angles_deg(rotations)
    assert torch.allclose(yaw, torch.zeros(4), atol=1e-4)
    assert torch.allclose(pitch, torch.zeros(4), atol=1e-4)
