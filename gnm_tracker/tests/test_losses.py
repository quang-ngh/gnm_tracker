import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.fit import losses  # noqa: E402


def test_landmark_loss_zero_when_equal():
    uv = torch.randn(3, 10, 2)
    conf = torch.ones(3, 10)
    assert losses.landmark_reprojection_loss(uv, uv.clone(), conf, delta=3.0).item() == 0.0


def test_landmark_subset_selects():
    pred = torch.zeros(1, 5, 2)
    target = torch.zeros(1, 5, 2)
    target[0, 0] = 100.0  # only landmark 0 is wrong
    conf = torch.ones(1, 5)
    subset = torch.tensor([1, 2, 3, 4])  # excludes the wrong landmark
    assert losses.landmark_reprojection_loss(pred, target, conf, 3.0, subset).item() == 0.0


def test_temporal_second_diff():
    ramp = torch.arange(10, dtype=torch.float32)[:, None]  # linear -> 0 curvature
    assert losses.temporal_second_diff(ramp).item() == pytest.approx(0.0, abs=1e-6)
    assert losses.temporal_second_diff(torch.randn(10, 3)).item() > 0.0


def test_weighted_total_skips_zero_weight():
    terms = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
    total = losses.weighted_total(terms, {"a": 0.0, "b": 2.0})
    assert total.item() == pytest.approx(6.0)


def test_expression_prior_l2():
    prior = losses.ExpressionPrior(mode="l2")
    psi = torch.tensor([[1.0, 2.0, 2.0]])  # sum sq = 9
    assert prior.loss(psi).item() == pytest.approx(9.0)
