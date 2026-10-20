#!/usr/bin/env python3
"""Keep only the poses of an arm-pose library (scripts/build_arm_pose_library.py) whose joints stay inside
given bounds -- e.g. the robot's right shoulder roll <= +0.0524 rad (AppleArmPose-v4: the screw holding the
hand wiring, onboard/arm_mpc_acados.py THETA_LIMITS). Every per-pose array is filtered alike.

    python3 scripts/filter_arm_pose_library.py --src envs/assets/arm_pose_library.npz \\
        --out envs/assets/arm_pose_library_roll.npz --joint right_shoulder_roll_joint --max 0.0524
"""
import argparse

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", required=True)
p.add_argument("--out", required=True)
p.add_argument("--joint", default="right_shoulder_roll_joint")
p.add_argument("--min", type=float, default=-np.inf)
p.add_argument("--max", type=float, default=np.inf)
a = p.parse_args()
L = dict(np.load(a.src))
names = [str(n) for n in L["joint_names"]]
j = names.index(a.joint)
M = len(L["q"])
keep = (L["q"][:, j] >= a.min) & (L["q"][:, j] <= a.max)
out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == M else v) for k, v in L.items()}
np.savez_compressed(a.out, **out)
print(f"{a.src}: kept {int(keep.sum())} of {M} poses ({a.joint} in [{a.min}, {a.max}]) -> {a.out}")
