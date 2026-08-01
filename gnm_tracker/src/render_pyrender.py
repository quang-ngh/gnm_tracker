#!/usr/bin/env python
"""Render a fitted clip with GNM's OWN pyrender pipeline — for Linux/GPU debug.

This is the "use GNM's render codebase directly" path. It calls
``gnm.shape.visualization.gnm_pyrender.render`` (proper GL: per-part meshes,
MetallicRoughness material, DirectionalLight, MSAA) instead of the portable
software rasterizer in ``gnm_tracker/viz/render_compare.py``.

It needs a working offscreen GL (OSMesa or EGL), which is available on Linux but
NOT on macOS (Apple dropped OpenGL). Run it on your Linux box via the bash
wrapper ``scripts/render_pyrender.sh`` (or directly):

    # OSMesa (CPU):  apt-get install libosmesa6 libosmesa6-dev
    pip install pyrender pyopengl

    scripts/render_pyrender.sh outputs/clips/<id>.npz <clip>.mp4 \
        outputs/pyrender_<id>.mp4 osmesa
    # or directly (EGL on a GPU box):
    python src/render_pyrender.py --record outputs/clips/<id>.npz \
        --video <clip>.mp4 --out outputs/pyrender_<id>.mp4 --platform egl

Output is a side-by-side ``input | GNM render`` mp4.

Camera bridge: the tracker uses an OpenCV pinhole (cam = R_wc @ world, u=f·x/z+cx)
with the head positioned by GNM's translation in world space, so the extrinsic is
rotation-only. We convert intrinsics+extrinsics to OpenGL with GNM's own
``camera_conversions`` helpers, and reuse GNM's ``get_vertex_colors`` /
``compute_vertex_normals`` / ``triangles_group`` — nothing about the mesh is
re-derived here.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="clip .npz produced by fit_sequence")
    ap.add_argument("--video", required=True, help="source clip")
    ap.add_argument("--out", default="outputs/pyrender_side_by_side.mp4")
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="auto",
                    help="auto (CUDA if available, else CPU), or cpu/cuda/mps")
    ap.add_argument("--platform", default="osmesa", choices=["osmesa", "egl"],
                    help="offscreen GL backend (Linux)")
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=10.0)
    ap.add_argument("--msaa", type=int, default=2, help="multisample anti-aliasing factor")
    args = ap.parse_args()

    # Must be set BEFORE importing pyrender / gnm_pyrender.
    os.environ["PYOPENGL_PLATFORM"] = args.platform

    import numpy as np
    import torch

    from gnm.shape.visualization import camera_conversions, gnm_pyrender
    from gnm.shape.visualization import vertex_colors as vc
    from gnm_tracker.config import load_config
    from gnm_tracker.io import read_video, write_video
    from gnm_tracker.model import ClipParams, GnmModel, PerspectiveCamera

    cfg = load_config(args.config)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    print(f"device: {device}")
    model = GnmModel(cfg, device)
    gnm = model.gnm

    frames, fps = read_video(args.video, max_frames=args.max_frames)
    h, w = frames[0].shape[:2]

    # -- load fitted params (add neutral back so the mesh matches the image) --
    data = np.load(args.record, allow_pickle=True)
    j = model.num_joints
    rot = data["pose"][:, : j * 3].reshape(-1, j, 3)
    trans = data["pose"][:, j * 3 :]
    expr = data["expr"] + data["neutral_expr"][None]
    n = min(len(frames), len(expr))
    frames = frames[:n]

    params = ClipParams(n, model.identity_dim, model.expression_dim, j, device)
    with torch.no_grad():
        params.shape.copy_(torch.as_tensor(data["shape"], device=device))
        params.expression.copy_(torch.as_tensor(expr[:n], device=device))
        params.rotations.copy_(torch.as_tensor(rot[:n], device=device))
        params.translation.copy_(torch.as_tensor(trans[:n], device=device))

    # -- world vertices + normals for every frame (GNM's own normals) --------
    with torch.no_grad():
        verts = model(params.shape, params.expression, params.rotations, params.translation)
        normals = gnm.compute_vertex_normals(verts)
    verts_np = verts.cpu().numpy()[:, None]        # (N, 1, V, 3)
    normals_np = normals.cpu().numpy()[:, None]    # (N, 1, V, 3)

    # -- per-vertex colors (GNM's; fill unset groups with skin so eyes show) --
    cols = np.asarray(vc.get_vertex_colors(gnm_np=gnm), np.float32)
    cols[cols.sum(1) == 0] = np.array(vc.DEFAULT_COLOR, np.float32)
    colors_np = np.broadcast_to(cols[None, None], (n, 1, gnm.num_vertices, 3)).copy()

    # -- camera: OpenCV pinhole -> OpenGL via GNM's converters ---------------
    cam = PerspectiveCamera.from_config(cfg, (h, w))
    f = float(cam.focal.detach())
    cx, cy = (float(x) for x in cam.principal_point)
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    w2c_cv = np.eye(4)
    w2c_cv[:3, :3] = cam.r_wc.cpu().numpy()        # rotation-only extrinsic
    cam_to_image = camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
        k, w, h, args.near, args.far
    )
    w2c_gl = camera_conversions.opencv_extrinsics_to_opengl(w2c_cv)
    cam_to_image = np.broadcast_to(cam_to_image, (n, 4, 4)).copy()
    w2c_gl = np.broadcast_to(w2c_gl, (n, 4, 4)).copy()

    # -- mesh topology per component + uvs -----------------------------------
    triangles = {c: gnm.triangles_group(c) for c in gnm.mesh_component_names}
    vertex_uvs = np.asarray(gnm.vertex_uvs, np.float32)

    background = np.full((n, h, w, 3), 0.94, np.float32)  # light-gray render panel
    rendered = gnm_pyrender.render(
        vertices=verts_np,
        triangles=triangles,
        world_to_camera=w2c_gl,
        camera_to_image=cam_to_image,
        vertex_normals=normals_np,
        vertex_uvs=vertex_uvs,
        vertex_colors=colors_np,
        image_size=(w, h),
        multisample_antialiasing=args.msaa,
        background_color=background,
        verbose=True,
    )
    rendered = (np.clip(rendered, 0, 1) * 255).astype(np.uint8)   # (N, H, W, 3)

    sbs = [np.concatenate([frames[i], rendered[i]], axis=1) for i in range(n)]
    write_video(args.out, sbs, fps=fps)
    print(f"wrote {args.out}  ({n} frames, {args.platform})")


if __name__ == "__main__":
    main()
