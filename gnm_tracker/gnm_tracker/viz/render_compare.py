"""Shaded mesh rendering for debug previews (Section 11).

Reuses GNM's own rendering ingredients as much as possible (the machine here is
headless with no EGL/OSMesa, so `gnm_pyrender.render`'s GL path can't run):

  * per-vertex colors from GNM's ``visualization.vertex_colors.get_vertex_colors``
    (blue skin + bright scleras + dark irises + teeth) — so the **eyes are
    visible**, exactly like GNM's demo;
  * per-vertex normals from GNM's ``compute_vertex_normals``;
  * a camera head-light, matching ``gnm_pyrender``'s DirectionalLight-on-camera.

Only the final rasterization is ours: a small **z-buffered** software rasterizer
(Gouraud color + depth test) so eyelids/eyeballs/teeth occlude correctly. NOT
differentiable — eyeballing only.
"""

from __future__ import annotations

import numpy as np
import torch

# GNM's demo skin color (50,156,237) -> the familiar blue head.
_GNM_SKIN = (50 / 255.0, 156 / 255.0, 237 / 255.0)
_GRAY_BG = (239, 239, 239)
_AMBIENT = 0.30

_VCOL_CACHE: dict[int, np.ndarray] = {}


def _gnm_vertex_colors(model) -> np.ndarray:
    """Per-vertex colors (V,3 in [0,1]) via GNM's shipped vertex_colors module."""
    key = id(model)
    if key not in _VCOL_CACHE:
        skin = np.array(_GNM_SKIN, np.float32)
        try:
            from gnm.shape.visualization import vertex_colors as vc

            cols = np.asarray(vc.get_vertex_colors(gnm_np=model.gnm), np.float32)
            # GNM leaves non-highlighted groups (eye sockets/interior, ears, gums)
            # at black; fill those with the skin color so eyes read as eyeballs
            # (bright sclera + dark iris) rather than black holes.
            cols[cols.sum(axis=1) == 0] = skin
        except Exception:  # fallback: uniform skin
            cols = np.tile(skin, (model.num_vertices, 1))
        _VCOL_CACHE[key] = cols
    return _VCOL_CACHE[key]


def _zraster(uv, depth, vcol, tris, image_hw, bg_color) -> np.ndarray:
    """Z-buffered Gouraud rasterizer. vcol is per-vertex RGB in [0,255]."""
    h, w = image_hw
    img = np.zeros((h, w, 3), np.float32)
    if bg_color is not None:
        img[:] = np.asarray(bg_color, np.float32)
    zbuf = np.full((h, w), np.inf, np.float32)

    x, y = uv[:, 0], uv[:, 1]
    for i0, i1, i2 in tris:
        x0, y0, x1, y1, x2, y2 = x[i0], y[i0], x[i1], y[i1], x[i2], y[i2]
        minx = int(max(0, np.floor(min(x0, x1, x2))))
        maxx = int(min(w - 1, np.ceil(max(x0, x1, x2))))
        miny = int(max(0, np.floor(min(y0, y1, y2))))
        maxy = int(min(h - 1, np.ceil(max(y0, y1, y2))))
        if maxx < minx or maxy < miny:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue
        ys, xs = np.mgrid[miny : maxy + 1, minx : maxx + 1]
        xs = xs + 0.5
        ys = ys + 0.5
        a = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        b = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        z = a * depth[i0] + b * depth[i1] + c * depth[i2]
        sub_z = zbuf[miny : maxy + 1, minx : maxx + 1]
        sel = inside & (z < sub_z)
        if not sel.any():
            continue
        col = a[..., None] * vcol[i0] + b[..., None] * vcol[i1] + c[..., None] * vcol[i2]
        sub_img = img[miny : maxy + 1, minx : maxx + 1]
        sub_img[sel] = col[sel]
        sub_z[sel] = z[sel]
    return np.clip(img, 0, 255).astype(np.uint8)


def _shaded_vertex_colors(model, n_cam: np.ndarray) -> np.ndarray:
    """GNM per-vertex colors modulated by a camera head-light -> RGB [0,255]."""
    base = _gnm_vertex_colors(model)                       # (V,3) [0,1]
    intensity = np.clip(-n_cam[:, 2], 0.0, 1.0)            # headlight: front is bright
    shade = _AMBIENT + (1.0 - _AMBIENT) * intensity
    return base * shade[:, None] * 255.0


@torch.no_grad()
def _posed(model, camera, params, idx: int):
    """Return numpy (uv, depth, cam-space normals) for frame ``idx``."""
    verts = model(
        params.shape, params.expression[idx : idx + 1],
        params.rotations[idx : idx + 1], params.translation[idx : idx + 1],
    )[0]
    n_world = model.gnm.compute_vertex_normals(verts)      # GNM's own normals
    n_cam = (n_world @ camera.r_wc.T).cpu().numpy()
    cam = camera.to_camera(verts).cpu().numpy()
    uv, _ = camera.project(verts)
    return uv.cpu().numpy(), cam[:, 2], n_cam, cam


def _front_faces(cam: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Keep triangles whose geometric normal faces the camera (cull the rest)."""
    v = cam[tris]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return tris[fn[:, 2] < 0]


def render_posed(fitter, idx: int, image_hw: tuple[int, int]) -> np.ndarray:
    """GNM-colored, shaded, z-buffered render on a light background (demo-like)."""
    m = fitter.model
    uv, depth, n_cam, cam = _posed(m, fitter.camera, fitter.params, idx)
    tris = _front_faces(cam, m.triangles.cpu().numpy())
    vcol = _shaded_vertex_colors(m, n_cam)
    return _zraster(uv, depth, vcol, tris, image_hw, _GRAY_BG)


def side_by_side(image: np.ndarray, fitter, idx: int) -> np.ndarray:
    """Input frame | GNM-colored fit render (Section 11)."""
    rendered = render_posed(fitter, idx, image.shape[:2])
    return np.concatenate([image, rendered], axis=1)


def overlay_shaded(frame: np.ndarray, fitter, idx: int, color=None, alpha: float = 0.55) -> np.ndarray:
    """Semi-transparent GNM-colored mesh over the frame (front faces only)."""
    m = fitter.model
    uv, depth, n_cam, cam = _posed(m, fitter.camera, fitter.params, idx)
    tris = _front_faces(cam, m.triangles.cpu().numpy())
    vcol = _shaded_vertex_colors(m, n_cam)
    rgb = _zraster(uv, depth, vcol, tris, frame.shape[:2], bg_color=None)
    mask = (rgb.sum(-1) > 0)[..., None]
    out = frame.astype(np.float32)
    out = np.where(mask, (1 - alpha) * out + alpha * rgb.astype(np.float32), out)
    return out.astype(np.uint8)


@torch.no_grad()
def render_ortho(model, verts3d: torch.Tensor, image_hw=(256, 256), margin=0.12) -> np.ndarray:
    """Orthographic front view of a GNM-colored mesh (psi-cleanliness viz)."""
    v = verts3d.detach().cpu().numpy()
    tris = model.triangles.cpu().numpy()
    n_world = model.gnm.compute_vertex_normals(verts3d).cpu().numpy()
    h, w = image_hw
    xy = v[:, :2].copy()
    xy[:, 1] = -xy[:, 1]
    lo, hi = xy.min(0), xy.max(0)
    span = (hi - lo).max() * (1 + 2 * margin)
    center = (hi + lo) / 2
    scale = min(h, w) / max(span, 1e-6)
    uv = (xy - center) * scale + np.array([w / 2, h / 2])
    # Orthographic camera looks along -Z here (front = larger z toward viewer).
    n_cam = n_world.copy()
    n_cam[:, 1] *= -1
    n_cam[:, 2] *= -1
    cam_flip = v.copy()
    cam_flip[:, 1] *= -1
    cam_flip[:, 2] *= -1
    tris_front = _front_faces(cam_flip, tris)
    vcol = _shaded_vertex_colors(model, n_cam)
    return _zraster(uv, -v[:, 2], vcol, tris_front, image_hw, _GRAY_BG)
