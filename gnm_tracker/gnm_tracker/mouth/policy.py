"""Mouth-interior (tongue/teeth) policy — decide, don't drift (Section 8).

The tongue/teeth expression dimensions (GNM dims 350..381) are **not observable
from a single camera**. Left free they take image-driven garbage; so a policy is
mandatory (Guardrail: never leave tongue/teeth dims free).

Modes:
  * ``pin`` (default, v1): strongly regularize (or hard-zero) the tongue dims via
    ``L_mouth`` so they never absorb pixel noise. The only runnable mode with no
    audio.
  * ``viseme`` (deferred, needs audio): drive tongue dims from a phoneme->viseme
    prior. Raises until audio is introduced.
  * ``audio`` (deferred, research): condition tongue dims on audio features
    inside the fit. Raises until audio is introduced.

``L_mouth`` must always keep these dims bounded whenever they are not actively
driven — even in ``pin`` mode with ``hard_zero``, the penalty stays on.
"""

from __future__ import annotations

import torch


class MouthPolicy:
    def __init__(
        self,
        tongue_dims: torch.Tensor,
        mode: str = "pin",
        pin_weight: float = 50.0,
        hard_zero: bool = False,
    ):
        if mode in ("viseme", "audio"):
            raise NotImplementedError(
                f"mouth.mode='{mode}' needs audio, which is deferred in v1 "
                "(Section 8). Use mode='pin'."
            )
        if mode != "pin":
            raise ValueError(f"unknown mouth.mode '{mode}'")
        self.mode = mode
        self.pin_weight = float(pin_weight)
        self.hard_zero = bool(hard_zero)
        self.register_dims(tongue_dims)

    def register_dims(self, tongue_dims: torch.Tensor) -> None:
        self.tongue_dims = tongue_dims.to(torch.long)

    def loss(self, psi: torch.Tensor) -> torch.Tensor:
        """L_mouth = pin_weight * mean_t || psi[:, tongue_dims] ||^2."""
        tongue = psi[:, self.tongue_dims.to(psi.device)]
        return self.pin_weight * (tongue**2).sum(dim=-1).mean()

    @torch.no_grad()
    def apply_hard_zero(self, params) -> None:
        """Clamp tongue dims to exactly zero (used each step if hard_zero)."""
        if self.hard_zero:
            params.expression.data[:, self.tongue_dims.to(params.expression.device)] = 0.0

    @classmethod
    def from_config(cls, cfg, tongue_dims: torch.Tensor) -> "MouthPolicy":
        return cls(
            tongue_dims=tongue_dims,
            mode=cfg.mouth.mode,
            pin_weight=float(cfg.mouth.pin_weight),
            hard_zero=bool(cfg.mouth.hard_zero),
        )
