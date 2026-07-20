#!/usr/bin/env python3
"""
Run this INSIDE your Isaac Sim / cuRobo Python environment (the same
interpreter that runs run_g1_curobo.py) -- it needs to import curobo.

It inspects your installed cuRobo version to see exactly which
`kinematics.*` keys CudaRobotGeneratorConfig accepts, strips out anything
your version doesn't support, warns about anything required that's
missing, and writes a guaranteed-compatible copy next to the original.
This avoids the trial-and-error of hitting one unsupported key per run.

Usage:
    python check_robot_yaml_compat.py g1_dex3_right_arm.yml
"""
import sys
import dataclasses
import yaml

try:
    from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
except ImportError as e:
    raise SystemExit(
        "Could not import CudaRobotGeneratorConfig from its usual location "
        "(curobo.cuda_robot_model.cuda_robot_generator). Your cuRobo version "
        "may keep it elsewhere -- open curobo/types/robot.py and check the "
        f"top-of-file import line for the real path. Original error: {e}"
    )


def main(path: str):
    valid_fields = {f.name for f in dataclasses.fields(CudaRobotGeneratorConfig)}

    with open(path) as f:
        data = yaml.safe_load(f)

    kin = data["robot_cfg"]["kinematics"]
    present = set(kin.keys())

    unsupported = present - valid_fields
    required_missing = [
        f.name
        for f in dataclasses.fields(CudaRobotGeneratorConfig)
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in present
        and f.name != "tensor_args"  # passed separately by load_from_robot_config
    ]

    print(f"Installed cuRobo CudaRobotGeneratorConfig accepts {len(valid_fields)} kinematics fields.")

    if unsupported:
        print(f"Removing unsupported keys found in your yaml: {sorted(unsupported)}")
        for k in unsupported:
            del kin[k]
    else:
        print("No unsupported keys found -- your yaml already matches this cuRobo version.")

    if required_missing:
        print(f"WARNING: these required fields have no default and aren't in your yaml: {required_missing}")
        print("You'll need to add them by hand.")

    if path.endswith((".yml", ".yaml")):
        base = path.rsplit(".", 1)[0]
        ext = path.rsplit(".", 1)[1]
        out_path = f"{base}_compat.{ext}"
    else:
        out_path = path + "_compat.yml"

    with open(out_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote a guaranteed-compatible copy to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <robot_yaml_path>")
    main(sys.argv[1])
