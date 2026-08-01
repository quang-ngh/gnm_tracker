"""Regenerate the MediaPipe-468 <-> GNM correspondence from scratch.

Use this instead of the shipped `head_dense_468.txt` if GNM's topology changes in
a future release, or if you want to adjust which anatomy each landmark is allowed
to snap to.

    pip install numpy scipy
    curl -LO https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
    python build_mediapipe_correspondence.py --canonical canonical_face_model.obj --out head_dense_468.txt

Method
------
1. Compute GNM's shipped 68 sparse landmarks in 3D.
2. Map MediaPipe's canonical 468 mesh to those 68 via the standard iBUG ordering,
   and solve a Umeyama similarity transform. The mirrored variant is also tried
   and the lower-residual one wins, which catches left/right ordering surprises.
3. Project every canonical vertex onto the GNM surface, CONSTRAINED to triangles
   touching an anatomically appropriate vertex group, and store the barycentric
   anchor.

The region constraint in step 3 is the part that matters. Unconstrained
nearest-surface projection puts landmarks on whatever geometry happens to be
closest, which is wrong in exactly the places you care about — see the eyelid
note below.

Unofficial / community-built. Not affiliated with Google.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gnm.shape import gnm_landmarks, gnm_numpy

# MediaPipe canonical 468 -> iBUG 68 (jaw, brows, nose, eyes, lips).
MP68 = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]

# ── landmark groups used to constrain projection ─────────────────────────────
MP_UPPER_LIP = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308}
MP_LOWER_LIP = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308}
MP_CORNERS = {61, 291, 78, 308}

# Eyelid MARGINS, split upper/lower. THIS SPLIT IS LOAD-BEARING: if these
# landmarks are projected onto the orbital SKIN (or onto a single undivided
# socket band), the upper and lower lid land on the same triangle and the
# palpebral aperture collapses from ~6 mm to ~0.5 mm. Nothing errors; blink
# detection, eye-closure and every ocular measure just quietly stop working.
MP_LID_UPPER = {157, 158, 159, 160, 161, 246, 384, 385, 386, 387, 388, 466}
MP_LID_LOWER = {144, 145, 153, 154, 163, 7, 373, 374, 380, 381, 390, 249}

MP_EYE_R = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246}
MP_EYE_L = {362, 249, 390, 373, 374, 380, 381, 382, 398, 384, 385, 386, 387, 388, 466, 263}
MP_BROW_R = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46}
MP_BROW_L = {336, 296, 334, 293, 300, 276, 283, 282, 295, 285}


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (a ** 2).sum() * len(src)
    t = md - s * R @ ms
    rms = float(np.sqrt(((s * src @ R.T + t - dst) ** 2).sum(1).mean()))
    return s, R, t, rms


def mirror_68(idx_list):
    m = list(range(68))
    m[0:17] = list(range(16, -1, -1))
    m[17:22], m[22:27] = list(range(26, 21, -1)), list(range(21, 16, -1))
    m[31:36] = list(range(35, 30, -1))
    m[36:42] = [45, 44, 43, 42, 47, 46]
    m[42:48] = [39, 38, 37, 36, 41, 40]
    m[48:55] = [54, 53, 52, 51, 50, 49, 48]
    m[55:60] = [59, 58, 57, 56, 55]
    m[60:65] = [64, 63, 62, 61, 60]
    m[65:68] = [67, 66, 65]
    return [idx_list[i] for i in m]


def closest_point_barycentric(p, tri):
    """Closest point on a triangle, as barycentric coordinates (clamped)."""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return np.array([1 - v, v, 0.0])
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return np.array([1 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.array([0.0, 1 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return np.array([1 - v - w, v, w])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", type=Path, required=True,
                    help="MediaPipe canonical_face_model.obj")
    ap.add_argument("--out", type=Path, default=Path("head_dense_468.txt"))
    args = ap.parse_args()

    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    neutral = np.asarray(gnm(None, None, None, None))

    canon = np.array([[float(x) for x in ln.split()[1:4]]
                      for ln in args.canonical.read_text().splitlines()
                      if ln.startswith("v ")])
    assert canon.shape == (468, 3), f"expected 468 canonical verts, got {canon.shape}"

    lm = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    gnm68 = (neutral[lm.indices] * lm.weights[..., None]).sum(axis=1)

    fits = {tag: umeyama(canon[mp], gnm68)
            for tag, mp in (("normal", MP68), ("mirrored", mirror_68(MP68)))}
    best = min(fits, key=lambda k: fits[k][3])
    s, R, t, rms = fits[best]
    print(f"umeyama: normal {fits['normal'][3]*1000:.2f} mm, "
          f"mirrored {fits['mirrored'][3]*1000:.2f} mm -> using {best}")
    canon_g = s * canon @ R.T + t

    def group(name):
        return set(np.asarray(gnm.vertex_group_indices(name)).tolist()) \
            if name in gnm.vertex_group_names else set()

    skin = group("skin_exterior") or group("skin")
    upper_lip = group("upper_lip") | group("upper_lip_region")
    lower_lip = group("lower_lip") | group("lower_lip_region")
    orbital_l, orbital_r = group("left_orbital_region"), group("right_orbital_region")
    brow_l, brow_r = group("left_brow_region"), group("right_brow_region")
    exclude = (group("eyes") | group("teeth") | group("gums")
               | group("tongue") | group("mouth_sock"))
    generic = skin - exclude

    # Split the eye socket into upper/lower lid bands, per eye — see the note by
    # MP_LID_UPPER above for why this is not optional.
    socket = np.asarray(sorted(group("eye_sockets")), dtype=int)
    lid_upper, lid_lower = set(), set()
    for side in (socket[neutral[socket, 0] >= 0], socket[neutral[socket, 0] < 0]):
        if not side.size:
            continue
        cy = neutral[side, 1].mean()
        lid_upper |= set(side[neutral[side, 1] >= cy].tolist())
        lid_lower |= set(side[neutral[side, 1] < cy].tolist())

    regions = {
        "corners": upper_lip | lower_lip, "upper_lip": upper_lip, "lower_lip": lower_lip,
        "lid_upper": lid_upper or generic, "lid_lower": lid_lower or generic,
        "eye_l": orbital_l or generic, "eye_r": orbital_r or generic,
        "brow_l": brow_l | orbital_l, "brow_r": brow_r | orbital_r, "generic": generic,
    }

    def region_for(i):
        if i in MP_CORNERS: return "corners"
        if i in MP_LID_UPPER: return "lid_upper"
        if i in MP_LID_LOWER: return "lid_lower"
        if i in MP_UPPER_LIP: return "upper_lip"
        if i in MP_LOWER_LIP: return "lower_lip"
        if i in MP_EYE_L: return "eye_l"
        if i in MP_EYE_R: return "eye_r"
        if i in MP_BROW_L: return "brow_l"
        if i in MP_BROW_R: return "brow_r"
        return "generic"

    tris = np.asarray(gnm.triangles_group("~eye_exteriors")).astype(np.int64)
    cache, rows, errs = {}, [], []
    for i in range(468):
        key = region_for(i)
        if key not in cache:
            sub = tris[np.isin(tris, list(regions[key])).any(axis=1)]
            cache[key] = (sub, cKDTree(neutral[sub].mean(axis=1)))
        sub, tree = cache[key]
        _, cand = tree.query(canon_g[i], k=min(24, len(sub)))
        best_d, best_row = np.inf, None
        for ti in np.atleast_1d(cand):
            tri = sub[ti]
            bary = closest_point_barycentric(canon_g[i], neutral[tri])
            d = np.linalg.norm(canon_g[i] - (neutral[tri] * bary[:, None]).sum(0))
            if d < best_d:
                best_d, best_row = d, (tri, bary)
        rows.append(best_row)
        errs.append(best_d)

    with open(args.out, "w") as f:
        for tri, bary in rows:
            f.write(f"{tri[0]} {bary[0]:.6f} {tri[1]} {bary[1]:.6f} "
                    f"{tri[2]} {bary[2]:.6f}\n")

    e = np.asarray(errs) * 1000
    print(f"wrote {args.out}: projection distance mean {e.mean():.2f} mm, "
          f"p95 {np.percentile(e, 95):.2f} mm, max {e.max():.2f} mm")

    # Sanity: the aperture must survive. This is the check that catches the
    # eyelid-collapse failure mode described above.
    p = np.array([(neutral[tri] * bary[:, None]).sum(0) for tri, bary in rows])
    ap_r = np.linalg.norm(p[159] - p[145]) * 1000
    ap_l = np.linalg.norm(p[386] - p[374]) * 1000
    print(f"palpebral aperture: R {ap_r:.2f} mm, L {ap_l:.2f} mm "
          f"({'ok' if min(ap_r, ap_l) > 3 else 'COLLAPSED — check the lid split'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
