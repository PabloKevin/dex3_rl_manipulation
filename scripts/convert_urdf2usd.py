import os
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# 1. Get absolute paths so there is no guessing
base_dir = os.path.expanduser("~")
urdf_path = os.path.join(base_dir, "unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf")
output_dir = os.path.join(base_dir, "dex3_rl_manipulation/assets")

# 2. Check if the URDF actually exists before trying
if not os.path.exists(urdf_path):
    print(f"ERROR: Cannot find URDF at {urdf_path}")
    exit()

# 3. Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Converting: {urdf_path}")
print(f"Outputting to: {output_dir}")

# 4. Configure and Run
cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    usd_dir=output_dir,
    fix_base=True,
    make_instanceable=True,
)

try:
    UrdfConverter(cfg)
    print("SUCCESS: Conversion complete!")
except Exception as e:
    print(f"FAILURE: {e}")