"""One-off probe: local-frame offset from right_hand_palm_link to the
mid-finger links, so reach_rewards' grasp point can be moved from the palm
origin toward where the fingers/pressure pads actually are.

Run:
  isaaclab.sh -p probe_palm_offset.py --headless
"""
import sys
sys.path.insert(0, "/home/pablo_kevin/dex3_rl_manipulation")

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import envs  # noqa: registers AppleGrasp-*
from isaaclab.utils.math import matrix_from_quat

from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY

cfg = AppleGraspEnvCfg_PLAY()
env = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
names = robot.data.body_names
palm_id = names.index("right_hand_palm_link")

targets = ["right_hand_middle_1_link", "right_hand_index_1_link", "right_hand_thumb_2_link"]

palm_pos = robot.data.body_pos_w[0, palm_id, :]
palm_quat = robot.data.body_quat_w[0, palm_id, :]
R = matrix_from_quat(palm_quat.unsqueeze(0))[0]   # world <- local

print("\n" + "=" * 60)
print(f"palm world pos: {palm_pos.tolist()}")
for t in targets:
    if t not in names:
        print(f"  {t}: NOT FOUND")
        continue
    tid = names.index(t)
    tpos = robot.data.body_pos_w[0, tid, :]
    d_world = tpos - palm_pos
    d_local = torch.matmul(R.T, d_world)   # world -> local
    print(f"{t}:")
    print(f"  world offset from palm: {d_world.tolist()}")
    print(f"  LOCAL offset from palm: {d_local.tolist()}  <-- use this")
print("=" * 60)

env.close()
simulation_app.close()
