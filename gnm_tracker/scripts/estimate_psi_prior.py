#!/usr/bin/env python
"""Estimate an expression prior precision matrix for the Mahalanobis L_psi (Sec 6).

Samples many expressions from GNM's shipped ExpressionSampler (a conditional VAE),
estimates their covariance, and writes ``data/groups/psi_precision.npz`` used by
``fit.losses.ExpressionPrior`` when ``fit.regularizers.psi_prior: mahalanobis``.
Optional — without it the fitter falls back to a strong L2 prior.

Requires TensorFlow (the sampler's decoder is a Keras model).

    python scripts/estimate_psi_prior.py --per-class 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _common
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--out", default="data/groups/psi_precision.npz")
    args = ap.parse_args()

    from gnm.shape.semantic_sampler import Expression, ExpressionSampler

    sampler = ExpressionSampler()
    rng = np.random.default_rng(0)

    samples = []
    for cls in Expression:
        samples.append(sampler.sample_expression(int(cls), num_samples=args.per_class, rng=rng))
    expr = np.concatenate(samples, axis=0).astype(np.float64)
    print(f"sampled {expr.shape[0]} expressions of dim {expr.shape[1]}")

    mean = expr.mean(axis=0)
    cov = np.cov(expr, rowvar=False)
    precision = np.linalg.inv(cov + args.eps * np.eye(cov.shape[0]))

    from gnm_tracker.config import resolve_path

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, precision=precision.astype(np.float32), mean=mean.astype(np.float32))
    print(f"wrote {out}  (set fit.regularizers.psi_prior: mahalanobis to use it)")


if __name__ == "__main__":
    main()
