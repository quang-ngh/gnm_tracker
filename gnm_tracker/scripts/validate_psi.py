#!/usr/bin/env python
"""Validation gates (Sections 9 & 10) — CI entry point.

    python scripts/validate_psi.py --correspondence-only   # eyelid self-check
    python scripts/validate_psi.py                         # + psi-cleanliness

Exits non-zero if any requested check fails, so it can gate CI.
"""

from __future__ import annotations

import argparse
import sys

import _common


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--correspondence-only", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from gnm_tracker.config import load_config
    from gnm_tracker.validate import run_correspondence_self_check

    cfg = load_config(args.config)
    ok = True

    print("== correspondence self-check (eyelid gotcha, Section 10) ==")
    corr_ok = run_correspondence_self_check()
    print(f"correspondence self-check: {'PASS' if corr_ok else 'FAIL'}")
    ok &= corr_ok

    if not args.correspondence_only:
        print("\n== psi-cleanliness gate (Section 9) ==")
        from gnm_tracker.validate.psi import run_psi_cleanliness_check

        device = _common.get_device(cfg, args.device)
        res = run_psi_cleanliness_check(cfg, device=device)
        print(f"mean drift {res['score']['mean_drift']:.4f} "
              f"(threshold {res['threshold']}) -> {'PASS' if res['pass'] else 'FAIL'}")
        ok &= res["pass"]

    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
