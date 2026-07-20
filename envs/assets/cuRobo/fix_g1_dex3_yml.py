"""
Sanitize a cuRobo robot config YAML against the actually-installed cuRobo
version's accepted kinematics fields.

Newer cuRobo configs (e.g. ones built for humanoid retargeting on `main`)
include fields like `tool_frames` that older pinned commits' constructors
don't accept. This script:

  1. Introspects CudaRobotGeneratorConfig.__init__ to find the real,
     currently-supported field set.
  2. Drops any key in the YAML's `kinematics` section that isn't accepted.
  3. Fills in a handful of commonly-required fields (seen in the ur5e.yml
     template) with safe defaults if they're missing entirely.
  4. Leaves collision_spheres, self_collision_ignore, self_collision_buffer,
     collision_link_names, cspace, urdf_path, asset_root_path, base_link,
     ee_link untouched.

Run this against your unitree_g1.yml (already copied into cuRobo's
content/configs/robot/ directory) before loading it with MotionGenConfig.
"""

import inspect
import sys

import yaml

from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig


# Defaults matching the ur5e.yml template style, only used to fill in
# fields that are accepted by this cuRobo version but missing from the
# G1 YAML entirely.
TEMPLATE_DEFAULTS = {
    "usd_path": "",
    "usd_robot_root": "/robot",
    "isaac_usd_path": "",
    "usd_flip_joints": {},
    "usd_flip_joint_limits": [],
    "link_names": None,
    "lock_joints": None,
    "extra_links": None,
    "mesh_link_names": None,
    "use_usd_kinematics": False,
}


def main(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if "robot_cfg" not in data:
        print("ERROR: file does not have a top-level 'robot_cfg' key. "
              "Run the earlier wrap-fix script first.")
        sys.exit(1)

    kin = data["robot_cfg"]["kinematics"]

    # 1. Find the real accepted field set for this cuRobo install.
    sig = inspect.signature(CudaRobotGeneratorConfig.__init__)
    accepted = set(sig.parameters.keys()) - {"self", "tensor_args"}

    # 2. Drop unsupported keys, report what was removed.
    present_keys = set(kin.keys())
    unsupported = present_keys - accepted
    if unsupported:
        print(f"Removing unsupported keys: {sorted(unsupported)}")
        for k in unsupported:
            del kin[k]
    else:
        print("No unsupported keys found.")

    # 3. Fill in commonly-required fields if missing (and accepted).
    added = []
    for key, default_val in TEMPLATE_DEFAULTS.items():
        if key in accepted and key not in kin:
            kin[key] = default_val
            added.append(key)
    if added:
        print(f"Added missing default fields: {added}")
    else:
        print("No default fields needed adding.")

    # 4. Sanity check the fields we most care about are intact.
    for critical_key in ["collision_spheres", "self_collision_ignore",
                          "collision_link_names", "cspace",
                          "urdf_path", "asset_root_path",
                          "base_link", "ee_link"]:
        status = "OK" if critical_key in kin else "MISSING"
        print(f"  {critical_key}: {status}")

    out_path = path  # overwrite in place
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nSanitized config written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_g1_dex3_yml.py /path/to/unitree_g1.yml")
        sys.exit(1)
    main(sys.argv[1])