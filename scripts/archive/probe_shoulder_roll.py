"""One-off probe: which SIGN of right_shoulder_roll_joint (and right_elbow_joint)
moves the elbow away from the torso (good) vs across/into it (bad, self-collision
prone), so posture_box's lower/upper bounds can be set with the correct sign."""
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

from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY, RIGHT_ARM_JOINTS

cfg = AppleGraspEnvCfg_PLAY()
env = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
env.reset()

robot = env.scene["robot"]
names = robot.data.body_names
torso_id = names.index("torso_link")
elbow_id = names.index("right_elbow_link")
shoulder_id = names.index("right_shoulder_pitch_link")
joint_names = list(robot.data.joint_names)
roll_jid = joint_names.index("right_shoulder_roll_joint")

origin0 = env.scene.env_origins[0, :]
torso_y = (robot.data.body_pos_w[0, torso_id, :] - origin0)[1].item()
shoulder_y = (robot.data.body_pos_w[0, shoulder_id, :] - origin0)[1].item()
print(f"\ntorso_y={torso_y:.4f}  shoulder_y={shoulder_y:.4f}  (more negative == further right/away from body)")

for test_roll in (-0.6, -0.3, 0.0, 0.3, 0.6):
    q = robot.data.default_joint_pos.clone()
    dq = torch.zeros_like(robot.data.default_joint_vel)
    q[:, roll_jid] = test_roll
    robot.write_joint_state_to_sim(q, dq)
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.reset()
    zero_actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    for _ in range(5):
        env.step(zero_actions)
    elbow_y = (robot.data.body_pos_w[0, elbow_id, :] - origin0)[1].item()
    print(f"shoulder_roll={test_roll:+.2f}  ->  elbow_y={elbow_y:+.4f}  "
          f"({'toward torso/crossing' if elbow_y > shoulder_y else 'away from torso'})")

env.close()
simulation_app.close()
