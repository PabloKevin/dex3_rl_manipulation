#!/usr/bin/env python3
"""
Run this INSIDE your Isaac Sim / cuRobo Python environment.

Unlike check_robot_yaml_compat.py (which only checked the top-level
kinematics keys), this checks EVERY nested config class that
CudaRobotGeneratorConfig.__post_init__ builds from your yaml:
  - kinematics.*            -> CudaRobotGeneratorConfig
  - kinematics.cspace.*     -> CSpaceConfig
  - kinematics.extra_links.*.* -> LinkParams   (if extra_links is present)

It strips anything unsupported, prints what it removed and why, and
writes a compatible copy. Re-run it after each edit if you're not sure
you got everything -- it's idempotent.

Usage:
    python fix_robot_yaml_full.py g1_dex3_right_arm.yml
"""
import sys
import dataclasses
import yaml


def valid_field_names(cls):
    return {f.name for f in dataclasses.fields(cls)}


def strip_unsupported(d: dict, cls, label: str) -> list:
    """Remove keys from dict d that aren't fields of dataclass cls. Returns removed keys."""
    valid = valid_field_names(cls)
    unsupported = [k for k in list(d.keys()) if k not in valid]
    for k in unsupported:
        print(f"  [{label}] removing unsupported key: {k!r}")
        del d[k]
    return unsupported


def main(path: str):
    from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
    from curobo.types.state import JointState  # noqa: F401  (sanity import)

    try:
        from curobo.types.base import CSpaceConfig
    except ImportError:
        from curobo.cuda_robot_model.cuda_robot_generator import CSpaceConfig  # older layout fallback

    try:
        from curobo.cuda_robot_model.kinematics_parser import LinkParams
    except ImportError:
        LinkParams = None

    with open(path) as f:
        data = yaml.safe_load(f)

    kin = data["robot_cfg"]["kinematics"]

    print("Checking kinematics (CudaRobotGeneratorConfig) ...")
    # cspace and extra_links are nested dataclasses -- check them separately,
    # don't let the top-level check delete them for "not matching a scalar field"
    nested_keys = {"cspace", "extra_links"}
    top_level_check = {k: v for k, v in kin.items() if k not in nested_keys}
    strip_unsupported(top_level_check, CudaRobotGeneratorConfig, "kinematics")
    for k in list(kin.keys()):
        if k not in nested_keys and k not in top_level_check:
            del kin[k]

    if "cspace" in kin:
        print("Checking kinematics.cspace (CSpaceConfig) ...")
        strip_unsupported(kin["cspace"], CSpaceConfig, "cspace")

    if "extra_links" in kin and LinkParams is not None:
        print("Checking kinematics.extra_links.* (LinkParams) ...")
        for link_name, link_def in kin["extra_links"].items():
            strip_unsupported(link_def, LinkParams, f"extra_links.{link_name}")

    if path.endswith((".yml", ".yaml")):
        base, ext = path.rsplit(".", 1)
        out_path = f"{base}_compat.{ext}"
    else:
        out_path = path + "_compat.yml"

    with open(out_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"\nWrote compatible copy to: {out_path}")
    print("If this still errors, paste the traceback -- it'll point at yet another nested class we haven't covered.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <robot_yaml_path>")
    main(sys.argv[1])
