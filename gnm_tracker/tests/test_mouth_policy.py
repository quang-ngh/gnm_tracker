import types

import pytest

torch = pytest.importorskip("torch")

from gnm_tracker.mouth import MouthPolicy  # noqa: E402


def test_loss_only_penalizes_tongue_dims():
    tongue = torch.tensor([1, 3])
    policy = MouthPolicy(tongue, mode="pin", pin_weight=1.0)
    # psi nonzero only outside tongue dims -> zero mouth loss.
    psi = torch.zeros(2, 5)
    psi[:, [0, 2, 4]] = 5.0
    assert policy.loss(psi).item() == pytest.approx(0.0)
    # psi nonzero on tongue dims -> positive.
    psi2 = torch.zeros(2, 5)
    psi2[:, [1, 3]] = 1.0
    assert policy.loss(psi2).item() == pytest.approx(2.0)  # 2 dims * 1 mean over frames


def test_hard_zero_clamps_tongue():
    tongue = torch.tensor([1, 3])
    policy = MouthPolicy(tongue, mode="pin", hard_zero=True)
    params = types.SimpleNamespace(expression=torch.nn.Parameter(torch.ones(2, 5)))
    policy.apply_hard_zero(params)
    assert torch.all(params.expression.data[:, [1, 3]] == 0)
    assert torch.all(params.expression.data[:, [0, 2, 4]] == 1)


@pytest.mark.parametrize("mode", ["viseme", "audio"])
def test_deferred_modes_raise(mode):
    with pytest.raises(NotImplementedError):
        MouthPolicy(torch.tensor([1]), mode=mode)
