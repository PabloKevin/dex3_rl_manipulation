import os, yaml
from curobo.util_file import get_robot_configs_path, get_assets_path, join_path

# 1. Which yml is actually found by name lookup?
cfg_path = join_path(get_robot_configs_path(), "g1_dex3_right_arm.yml")
print("Config file used:     ", cfg_path)
print("Config file exists:   ", os.path.exists(cfg_path))

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
k = cfg["robot_cfg"]["kinematics"]
print("urdf_path field:      ", k["urdf_path"])
print("use_external_assets:  ", k.get("use_external_assets"))
print("external_asset_path:  ", k.get("external_asset_path"))

# 2. Resolve it the way curobo would
base = k.get("external_asset_path") or get_assets_path()
resolved = join_path(base, k["urdf_path"])
print("Resolved URDF path:   ", resolved)
print("Resolved URDF exists: ", os.path.exists(resolved))

# 3. If it exists, inspect it directly (no curobo needed)
if os.path.exists(resolved):
    import xml.etree.ElementTree as ET
    root = ET.parse(resolved).getroot()
    joint_names = [j.get("name") for j in root.findall("joint")]
    print("Total joints in URDF: ", len(joint_names))
    print("Has left_hip_pitch_joint: ", "left_hip_pitch_joint" in joint_names)
    print("Has right_hand_palm_link as a link:",
          any(l.get("name") == "right_hand_palm_link" for l in root.findall("link")))
    print("First 15 joint names:", joint_names[:15])