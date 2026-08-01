#!/usr/bin/env python
"""Regenerate data/groups/expression_groups.json from the GNM model.

The mouth policy (Section 8) needs the *indices* of the tongue/teeth expression
dimensions exposed as a named group. Rather than hard-code them, derive them
from the loaded model so the file stays correct if GNM's expression basis
changes. Also emits mesh-orientation diagnostics used to set the camera's
base rotation, and the vertex groups that make up the mouth-interior photometric
mask.

    python src/dump_gnm_groups.py            # writes data/groups/...

Uses GNM's NumPy backend (no torch required).
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
from gnm.shape import gnm_landmarks, gnm_numpy

OUT = Path(__file__).resolve().parents[1] / "data" / "groups" / "expression_groups.json"

# Vertex groups that make up the mouth interior + eye interior. These are
# excluded from the photometric loss mask (M_face in Section 6): they are hidden
# geometry whose pixels are unreliable. Names verified against the v3.0 model.
MOUTH_INTERIOR_GROUPS = [
    "mouth_sock", "teeth", "gums", "upper_teeth_and_gums",
    "lower_teeth_and_gums", "tongue",
]
EYE_INTERIOR_GROUPS = ["eye_interiors", "scleras", "irises", "pupils"]


def _prefix(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def main() -> None:
    gnm = gnm_numpy.GNM.from_local(
        version=gnm_numpy.GNMMajorVersion.V3,
        variant=gnm_numpy.GNMVariant.HEAD,
    )
    names = list(gnm.expression_names)

    # Group expression dims by name prefix (e.g. "tongue", "left_eye_region").
    groups: "collections.OrderedDict[str, list[int]]" = collections.OrderedDict()
    for i, n in enumerate(names):
        groups.setdefault(_prefix(n), []).append(i)
    group_ranges = {p: [min(ix), max(ix)] for p, ix in groups.items()}

    # Mouth-interior expression dims (not observable from a single camera).
    interior = sorted(groups.get("tongue_mean", []) + groups.get("tongue", []))

    # ---- orientation diagnostics (drives camera.base_rotation_axis_angle) ----
    v = np.asarray(gnm.template_vertex_positions)  # (V, 3), meters
    cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
    lm = (v[cfg.indices] * cfg.weights[..., None]).sum(axis=1)  # (68, 3)
    ibug = dict(nose_tip=30, chin=8, brow_mid=27, eye_r_outer=36, eye_l_outer=45)
    orient = {k: [round(float(x), 4) for x in lm[i]] for k, i in ibug.items()}
    interocular_m = float(np.linalg.norm(lm[36] - lm[45]))

    doc = {
        "gnm_version": str(gnm.version),
        "gnm_variant": str(gnm.variant),
        "num_vertices": int(gnm.num_vertices),
        "num_joints": int(gnm.num_joints),
        "joint_names": list(gnm.joint_names),
        "identity_dim": int(gnm.identity_dim),
        "expression_dim": int(gnm.expression_dim),
        "expression_group_ranges": group_ranges,
        "mouth_interior_expr_dims": interior,
        "mouth_interior_vertex_groups": [
            g for g in MOUTH_INTERIOR_GROUPS if g in set(gnm.vertex_group_names)
        ],
        "eye_interior_vertex_groups": [
            g for g in EYE_INTERIOR_GROUPS if g in set(gnm.vertex_group_names)
        ],
        "template_bbox_min": [round(float(x), 4) for x in v.min(0)],
        "template_bbox_max": [round(float(x), 4) for x in v.max(0)],
        "template_landmark_positions_m": orient,
        "template_interocular_m": round(interocular_m, 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT}")
    print(json.dumps(
        {k: doc[k] for k in (
            "mouth_interior_expr_dims", "expression_group_ranges",
            "template_landmark_positions_m", "template_interocular_m")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
