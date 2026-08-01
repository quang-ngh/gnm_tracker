"""Design invariants (Section 2), enforced structurally."""

import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.model import ClipParams  # noqa: E402


def _params(t=4, n_id=8, n_expr=6, n_joints=4):
    return ClipParams(t, n_id, n_expr, n_joints)


def test_shape_is_single_vector_not_per_frame():
    p = _params()
    assert p.shape.shape == (8,)              # ONE identity per clip
    assert p.expression.shape == (4, 6)       # psi is per-frame
    assert p.rotations.shape == (4, 4, 3)     # pose is per-frame
    assert p.translation.shape == (4, 3)


def test_set_trainable_freezes_the_rest():
    p = _params()
    p.set_trainable(["expression", "rotation", "translation"])
    assert not p.shape.requires_grad          # shape frozen in Stage C
    assert p.expression.requires_grad
    assert p.rotations.requires_grad


def test_freeze_shape():
    p = _params()
    p.set_trainable(["shape"])
    assert p.shape.requires_grad
    p.freeze_shape()
    assert not p.shape.requires_grad


def test_rigid_stage_only_pose():
    p = _params()
    p.set_trainable(["rotation", "translation"])
    assert not p.shape.requires_grad and not p.expression.requires_grad
    assert p.rotations.requires_grad and p.translation.requires_grad


def test_subtract_neutral():
    p = _params()
    p.expression.data.fill_(1.0)
    p.subtract_neutral(torch.full((6,), 0.25))
    assert torch.allclose(p.expression.data, torch.full((4, 6), 0.75))
