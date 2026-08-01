"""Make etils array-normalization autograd-safe for PyTorch.

GNM's ``gnm_xnp`` methods are decorated with ``@enp.check_and_normalize_arrays``.
That decorator normalizes every array argument through ``DType.asarray``, which
for the torch backend ends up calling ``torch.asarray(tensor, dtype=...)``.
``torch.asarray`` returns a **new leaf** tensor and drops the autograd graph
(``requires_grad`` becomes ``False``), so gradients from the fitted params
(rotation / translation / expression / shape) never reach the loss and
``loss.backward()`` raises "element 0 of tensors does not require grad".

``torch.as_tensor`` performs the same (zero-copy when possible) conversion but
*preserves* the autograd history. We wrap ``DType.asarray`` so that whenever the
input already is a grad-carrying torch tensor and the normalized output has been
detached, we re-derive it with ``.to(dtype/device)`` (a graph-preserving op).

Idempotent and process-wide; installed on import (once).
"""

from __future__ import annotations

from etils import enp
from etils.enp.array_types import dtypes

_PATCHED_FLAG = "_gnm_tracker_grad_safe"


def install() -> None:
    if getattr(dtypes.DType.asarray, _PATCHED_FLAG, False):
        return

    _orig_asarray = dtypes.DType.asarray

    def asarray(self, array_like, *, xnp, casting=dtypes.Casting.ALL):
        # For a grad-carrying torch tensor the original path calls
        # ``torch.asarray(t, dtype=...)`` which (torch>=2.11) DROPS the autograd
        # history and, for a same-dtype tensor, even flips ``requires_grad`` to
        # False *in place* on the input. Both break gradient flow to the fitted
        # params. Normalize such tensors with a graph-preserving, non-mutating
        # ``.to()`` instead: run the original on a detached view only to learn
        # the target dtype/device, then cast the real tensor.
        if enp.lazy.is_torch(array_like) and array_like.requires_grad:
            probe = _orig_asarray(self, array_like.detach(), xnp=xnp, casting=casting)
            return array_like.to(dtype=probe.dtype, device=probe.device)
        return _orig_asarray(self, array_like, xnp=xnp, casting=casting)

    setattr(asarray, _PATCHED_FLAG, True)
    dtypes.DType.asarray = asarray


install()
