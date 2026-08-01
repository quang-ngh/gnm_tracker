"""psi-cleanliness metric behaviour (Section 9). Needs the real GNM model."""

import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.quality.psi_cleanliness import passes, psi_cleanliness_score  # noqa: E402


def test_identical_identity_has_zero_drift(gnm_model):
    a_shape = 0.4 * torch.randn(gnm_model.identity_dim)
    b_shapes = a_shape[None].repeat(3, 1)     # identical identities
    psi = 0.2 * torch.randn(4, gnm_model.expression_dim)
    score = psi_cleanliness_score(gnm_model, psi, a_shape, b_shapes=b_shapes)
    assert score["mean_drift"] == pytest.approx(0.0, abs=1e-5)
    assert passes(score, threshold=1e-3)


def test_different_identities_give_finite_drift(gnm_model):
    a_shape = 0.4 * torch.randn(gnm_model.identity_dim)
    psi = 0.2 * torch.randn(4, gnm_model.expression_dim)
    score = psi_cleanliness_score(gnm_model, psi, a_shape, num_pairs=4)
    assert score["mean_drift"] >= 0.0
    assert score["per_identity"].shape == (4,)
