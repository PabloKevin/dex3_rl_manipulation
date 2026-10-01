#!/usr/bin/env python3
"""Look at the reach scene: several envs, arm held at its default pose, and
every env re-randomized (apple position/height, branch yaw, tree) every
--reset_s seconds so the domain-randomization range is visible.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/view_scene.py --num_envs 9
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--reset_s", type=float, default=4.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg  # noqa: E402


def main():
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = args_cli.reset_s
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    cfg.viewer.eye = (2.2, -2.6, 2.2)
    cfg.viewer.lookat = (0.3, -0.3, 1.1)
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    while app.is_running():
        env.step(zero)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
