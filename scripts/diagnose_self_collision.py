#!/usr/bin/env python3
"""
Find exactly which collision-sphere pairs are overlapping at the robot's
start pose, so we stop guessing which self_collision_ignore entries to add.

Usage (run in your Isaac Sim / cuRobo python environment -- doesn't need
Isaac Sim itself, just the curobo package):
    python diagnose_self_collision.py
"""
import torch
import yaml

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path

ROBOT_CFG_PATH = join_path(get_robot_configs_path(), "g1_dex3_right_arm.yml")

# The actual start pose being checked: right arm at its retract_config /
# env default. Locked joints (legs/waist/left arm/hands) come from the yaml
# itself automatically -- CudaRobotModel applies lock_joints internally.
START_Q = [0.3, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]


def main():
    tensor_args = TensorDeviceType()

    with open(ROBOT_CFG_PATH) as f:
        config_file = yaml.safe_load(f)["robot_cfg"]

    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    q = torch.tensor([START_Q], **tensor_args.as_torch_dict())

    spheres_batch = kin_model.get_robot_as_spheres(q)
    spheres = spheres_batch[0] if isinstance(spheres_batch, list) and len(spheres_batch) and isinstance(spheres_batch[0], list) else spheres_batch

    print(f"Got {len(spheres)} spheres. Inspecting the first one to confirm attribute names...")
    s0 = spheres[0]
    print(f"  type: {type(s0)}")
    print(f"  repr: {s0}")
    print(f"  vars: {vars(s0) if hasattr(s0, '__dict__') else 'n/a'}")
    print()

    ignore = config_file["kinematics"].get("self_collision_ignore", {}) or {}

    def is_ignored(a, b):
        return b in ignore.get(a, []) or a in ignore.get(b, [])

    def get_name(s):
        for attr in ("name", "link_name"):
            if hasattr(s, attr):
                return getattr(s, attr)
        return None

    def get_pos(s):
        for attr in ("position", "pose"):
            if hasattr(s, attr):
                val = getattr(s, attr)
                return list(val[:3]) if val is not None else None
        return None

    def get_radius(s):
        for attr in ("radius", "r"):
            if hasattr(s, attr):
                return getattr(s, attr)
        return None

    names_ok = all(get_name(s) is not None for s in spheres[:5])
    if not names_ok:
        print("Could not auto-detect sphere name/position/radius attributes.")
        print("Paste the 'vars:' output above and I'll adjust the script.")
        return

    violations = []
    n = len(spheres)
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = spheres[i], spheres[j]
            n1, n2 = get_name(s1), get_name(s2)
            if n1 == n2:
                continue
            if is_ignored(n1, n2):
                continue
            p1, p2 = get_pos(s1), get_pos(s2)
            r1, r2 = get_radius(s1), get_radius(s2)
            if p1 is None or p2 is None or r1 is None or r2 is None:
                continue
            d = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            if d < (r1 + r2):
                violations.append((n1, n2, d, r1 + r2))

    violations.sort(key=lambda x: x[2] - x[3])  # worst penetration first
    print(f"Found {len(violations)} colliding sphere pairs NOT already in self_collision_ignore:")
    for a, b, d, thresh in violations[:30]:
        print(f"  {a:30s} <-> {b:30s}  dist={d:.4f}  sum_radii={thresh:.4f}  penetration={thresh - d:.4f}")


if __name__ == "__main__":
    main()
