"""Staged optimizer (Section 6 schedule).

A :class:`Fitter` bundles the model, camera, clip params, detection targets and
regularizers, and exposes :meth:`run_stage` which the single-frame (Tier 1) and
sequence (Tier 2) fitters drive:

  * Stage A — rigid pose (beta=0, psi=0), stable-landmark subset, landmark only.
  * Stage B — shared shape over K frames + per-frame pose + coarse psi.
  * Stage C — freeze shape; per-frame psi + pose on all frames (+photometric,
    +temporal, +mouth).

Design invariants are enforced structurally: shape is a single vector (params),
and :meth:`ClipParams.freeze_shape` is called before Stage C so psi can never
absorb shape error (invariant #3). Temporal smoothness is computed on the whole
clip's params (never across clip boundaries — one Fitter == one clip).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gnm_tracker.fit import losses


@dataclasses.dataclass
class Targets:
    uv: torch.Tensor          # (T, L, 2) detected landmark pixels
    conf: torch.Tensor        # (T, L) per-landmark confidence
    interocular: torch.Tensor  # (T,) pixels, for normalization
    images: torch.Tensor | None = None  # (T, H, W, 3) for photometric (optional)


class Fitter:
    def __init__(
        self,
        model,
        camera,
        params,
        targets: Targets,
        cfg,
        mouth,
        expr_prior,
        renderer=None,
        rigid_subset: torch.Tensor | None = None,
    ):
        self.model = model
        self.camera = camera
        self.params = params
        self.targets = targets
        self.cfg = cfg
        self.mouth = mouth
        self.expr_prior = expr_prior
        self.renderer = renderer          # optional differentiable renderer (photometric)
        self.rigid_subset = rigid_subset  # landmark indices for Stage A
        self.device = params.shape.device
        self.history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    def _forward(self, idx: torch.Tensor):
        p = self.params
        verts, lmk3d = self.model.forward_landmarks(
            p.shape, p.expression[idx], p.rotations[idx], p.translation[idx]
        )
        uv, z = self.camera.project(lmk3d)
        return verts, uv, z

    def _compute_terms(self, idx, weights, stage_cfg, iter_i, use_rigid_subset):
        p = self.params
        verts, uv, _ = self._forward(idx)
        target_uv = self.targets.uv[idx]
        conf = self.targets.conf[idx]

        subset = self.rigid_subset if use_rigid_subset else None
        delta = float(self.cfg.fit.huber_delta)
        terms: dict[str, torch.Tensor] = {}
        terms["lmk"] = losses.landmark_reprojection_loss(uv, target_uv, conf, delta, subset)
        terms["psi"] = self.expr_prior.loss(p.expression[idx])
        terms["beta"] = losses.identity_prior_loss(p.shape)
        terms["mouth"] = self.mouth.loss(p.expression[idx])
        # Temporal smoothness on the WHOLE clip's params (within-clip only).
        terms["temp"] = losses.temporal_loss(p.expression, p.rotations, p.translation)
        terms["col"] = losses.collision_loss(verts)

        # Photometric only in stages that ask for it, after landmarks settle, and
        # only when a differentiable renderer is available (else no-op on Mac).
        pho_from = int(stage_cfg.get("photometric_from_iter", 0)) if hasattr(stage_cfg, "get") else 0
        if float(weights.get("pho", 0.0)) > 0 and self.renderer is not None and iter_i >= pho_from:
            terms["pho"] = self._photometric(idx, verts)
        else:
            terms["pho"] = verts.new_zeros(())
        return terms

    def _photometric(self, idx, verts) -> torch.Tensor:
        """L_pho = || M_face ⊙ (render - I) ||_1  (Section 6)."""
        images = self.targets.images[idx]  # (B, H, W, 3) in [0,1]
        rendered, mask = self.renderer(
            verts, self.model.triangles, self.model.face_vertex_mask, images.shape[1:3]
        )
        m = mask.unsqueeze(-1)
        return (m * (rendered - images).abs()).sum() / (m.sum() * 3 + 1e-8)

    # ------------------------------------------------------------------
    def run_stage(
        self,
        name: str,
        stage_cfg,
        frame_pool: np.ndarray | torch.Tensor,
        batch_size: int | None = None,
        use_rigid_subset: bool = False,
    ) -> list[dict]:
        """Optimize one stage; returns its per-iteration loss history."""
        p = self.params
        p.set_trainable(list(stage_cfg.params), camera=self.camera)
        weights = stage_cfg.weights.to_dict() if hasattr(stage_cfg.weights, "to_dict") else dict(stage_cfg.weights)

        opt = torch.optim.Adam(p.trainable_parameters(self.camera), lr=float(stage_cfg.lr))
        pool = torch.as_tensor(np.asarray(frame_pool), dtype=torch.long, device=self.device)
        hist: list[dict] = []
        for it in range(int(stage_cfg.iters)):
            if batch_size is not None and batch_size < len(pool):
                sel = pool[torch.randperm(len(pool), device=self.device)[:batch_size]]
            else:
                sel = pool
            opt.zero_grad(set_to_none=True)
            terms = self._compute_terms(sel, weights, stage_cfg, it, use_rigid_subset)
            total = losses.weighted_total(terms, weights)
            if self.camera_focal_trainable():
                total = total + float(self.cfg.camera.focal_log_reg) * self.camera.focal_log_reg()
            total.backward()
            opt.step()
            self.mouth.apply_hard_zero(p)
            hist.append({k: float(v.detach()) for k, v in terms.items()} | {"total": float(total.detach())})
        self.history[name] = hist
        return hist

    def camera_focal_trainable(self) -> bool:
        lf = getattr(self.camera, "log_focal", None)
        return isinstance(lf, torch.nn.Parameter) and lf.requires_grad

    # ------------------------------------------------------------------
    @torch.no_grad()
    def normalized_reproj_error(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Median per-landmark reprojection error / interocular (for gates & mask)."""
        if idx is None:
            idx = torch.arange(self.params.num_frames, device=self.device)
        _, uv, _ = self._forward(idx)
        dist = torch.linalg.norm(uv - self.targets.uv[idx], dim=-1)  # (B, L)
        return dist / self.targets.interocular[idx][:, None]
