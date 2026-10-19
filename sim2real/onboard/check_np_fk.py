#!/usr/bin/env python3
"""check_np_fk.py -- np_fk.NumpyKinematics vs reach_io.G1Kinematics (pinocchio) on random poses.

  python3 sim2real/onboard/check_np_fk.py [--urdf ...] [--n 2000]
Prints the largest position (m) and rotation (entry) difference over all frames; > 1e-9 fails.
Also times both (per frames() call).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import reach_io as R  # noqa: E402
from np_fk import NumpyKinematics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=R.URDF_DEFAULT)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    pin_k = R.G1Kinematics(a.urdf)
    np_k = NumpyKinematics(a.urdf, R)
    rng = np.random.default_rng(1)
    dp = dr = 0.0
    for _ in range(a.n):
        w = rng.uniform(-0.6, 0.6, 3)
        q = rng.uniform(R.ARM_TARGET_LO, R.ARM_TARGET_HI)
        A, B = pin_k.frames(w, q), np_k.frames(w, q)
        for f in ("palm_pos", "palm_ref", "torso_pos", "shoulder_pos", "camera_pos"):
            dp = max(dp, float(np.max(np.abs(getattr(A, f) - getattr(B, f)))))
        for f in ("palm_R", "torso_R", "camera_R", "palm_normal"):
            dr = max(dr, float(np.max(np.abs(getattr(A, f) - getattr(B, f)))))
    t0 = time.perf_counter()
    for _ in range(500):
        pin_k.frames(w, q)
    t1 = time.perf_counter()
    for _ in range(500):
        np_k.frames(w, q)
    t2 = time.perf_counter()
    print(f"{a.n} random poses: max |position diff| {dp:.2e} m, max |rotation diff| {dr:.2e}; "
          f"frames() pinocchio {(t1 - t0) * 2:.3f} ms, numpy {(t2 - t1) * 2:.3f} ms")
    ok = dp < 1e-9 and dr < 1e-9
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
