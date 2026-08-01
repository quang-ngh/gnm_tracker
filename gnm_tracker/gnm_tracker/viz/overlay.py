"""Overlays: fitted mesh + reprojected-vs-detected landmarks (Section 11).

The #1 debug tools. A bad pose/shape is obvious the instant the mesh is drawn on
the frame, and per-landmark residual segments show immediately which landmark is
pulling the fit wrong. Colors are RGB (frames stay RGB end-to-end).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

_EDGE_CACHE: dict[int, np.ndarray] = {}


def _get_cmap(name: str):
    """matplotlib colormap lookup that works old and new (>=3.9 dropped cm.get_cmap)."""
    try:
        from matplotlib import colormaps

        return colormaps[name]
    except Exception:  # pragma: no cover - old matplotlib
        from matplotlib import cm

        return cm.get_cmap(name)


def _edges(model) -> np.ndarray:
    key = id(model)
    if key not in _EDGE_CACHE:
        tris = model.triangles.cpu().numpy()
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
        e = np.sort(e, axis=1)
        _EDGE_CACHE[key] = np.unique(e, axis=0)
    return _EDGE_CACHE[key]


@torch.no_grad()
def projected_vertices(fitter, idx: int) -> tuple[np.ndarray, np.ndarray]:
    p = fitter.params
    verts = fitter.model(
        p.shape, p.expression[idx : idx + 1], p.rotations[idx : idx + 1],
        p.translation[idx : idx + 1],
    )[0]
    uv, z = fitter.camera.project(verts)
    return uv.cpu().numpy(), z.cpu().numpy()


@torch.no_grad()
def projected_landmarks(fitter, idx: int) -> np.ndarray:
    p = fitter.params
    _, lmk = fitter.model.forward_landmarks(
        p.shape, p.expression[idx : idx + 1], p.rotations[idx : idx + 1],
        p.translation[idx : idx + 1],
    )
    uv, _ = fitter.camera.project(lmk[0])
    return uv.cpu().numpy()


def draw_mesh_overlay(image: np.ndarray, fitter, idx: int, cfg) -> np.ndarray:
    """Draw the fitted GNM wireframe over the frame (semi-transparent)."""
    img = image.copy()
    uv, _ = projected_vertices(fitter, idx)
    edges = _edges(fitter.model)
    stride = max(1, int(cfg.viz.wireframe_stride))
    edges = edges[::stride]
    h, w = img.shape[:2]
    layer = img.copy()
    p0 = uv[edges[:, 0]]
    p1 = uv[edges[:, 1]]
    for a, b in zip(p0, p1):
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        cv2.line(layer, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 (0, 255, 0), 1, cv2.LINE_AA)
    alpha = float(cfg.viz.wireframe_alpha)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def draw_landmark_residuals(image: np.ndarray, fitter, idx: int, cfg) -> np.ndarray:
    """Reprojected (model) vs detected (MediaPipe) landmarks, joined by residual
    segments colored by error magnitude."""
    img = image.copy()
    pred = projected_landmarks(fitter, idx)
    det = fitter.targets.uv[idx].cpu().numpy()
    inter = float(fitter.targets.interocular[idx].cpu())
    err = np.linalg.norm(pred - det, axis=-1) / max(inter, 1e-6)
    cmap = _get_cmap(cfg.viz.residual_cmap)
    norm = np.clip(err / max(float(cfg.quality.reproj_error_max) * 2, 1e-6), 0, 1)
    for i in range(len(pred)):
        if not (np.isfinite(pred[i]).all() and np.isfinite(det[i]).all()):
            continue
        r, g, b, _ = cmap(float(norm[i]))
        color = (int(r * 255), int(g * 255), int(b * 255))
        pd = (int(det[i, 0]), int(det[i, 1]))
        pp = (int(pred[i, 0]), int(pred[i, 1]))
        cv2.line(img, pd, pp, color, 1, cv2.LINE_AA)
        cv2.circle(img, pd, 1, (255, 255, 255), -1)   # detected = white
        cv2.circle(img, pp, 1, (255, 0, 0), -1)        # reprojected = red
    return img
