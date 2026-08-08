"""One-off probe: shoulder/elbow/wrist/palm world positions, to bound the
apple-spawn workspace by the arm's actual reachable radius."""
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
import envs  # noqa

from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY

cfg = AppleGraspEnvCfg_PLAY()
env = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
names = robot.data.body_names
origin0 = env.scene.env_origins[0, :]

chain = [
    "torso_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_palm_link",
]

print("\n" + "=" * 70)
positions = {}
for n in chain:
    if n not in names:
        print(f"{n}: NOT FOUND")
        continue
    idx = names.index(n)
    pos = robot.data.body_pos_w[0, idx, :] - origin0
    positions[n] = pos
    print(f"{n:28s} env-local pos: {[round(v, 4) for v in pos.tolist()]}")

print("-" * 70)
segs = [
    ("shoulder->elbow", "right_shoulder_pitch_link", "right_elbow_link"),
    ("elbow->wrist_roll", "right_elbow_link", "right_wrist_roll_link"),
    ("wrist_roll->palm", "right_wrist_roll_link", "right_hand_palm_link"),
    ("shoulder->palm (current pose, NOT max reach)", "right_shoulder_pitch_link", "right_hand_palm_link"),
]
total = 0.0
for label, a, b in segs:
    if a in positions and b in positions:
        d = torch.norm(positions[b] - positions[a]).item()
        print(f"{label:45s} {d:.4f} m")
        if "->" in label and "current" not in label:
            total += d
print(f"{'sum of rigid segment lengths (upper bound on max reach)':45s} {total:.4f} m")
print("=" * 70)

env.close()
simulation_app.close()
