#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Spawn a small marker sphere for every one of cuRobo's collision spheres,
at the robot's current start pose, directly in the Isaac Sim viewport.
Red = genuinely colliding with another link's sphere and NOT already
covered by self_collision_ignore. Gray = fine.

This also prints the same violation list to the terminal, but with real
link names this time (attributed from the yaml's own collision_spheres
section, in collision_link_names order -- the previous diagnostic script's
names were generic per-sphere IDs, not link names, so its ignore-list
filtering was silently matching nothing).

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/visualize_curobo_spheres.py
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize cuRobo collision spheres")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import yaml
import gymnasium as gym
from omni.isaac.core.objects import VisualSphere

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY, ROBOT_ROOT_HEIGHT

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path

ROBOT_CFG_PATH = join_path(get_robot_configs_path(), "g1_dex3_right_arm.yml")
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def build_sphere_to_link_map(config_file: dict):
    k = config_file["kinematics"]
    link_order = k["collision_link_names"]
    cs = k["collision_spheres"]
    sphere_to_link = []
    for link in link_order:
        spheres = cs.get(link, [])
        sphere_to_link.extend([link] * len(spheres))
    return sphere_to_link


def main():
    # --- Build the env so we have the real robot to compare against ---
    cfg = AppleGraspEnvCfg_PLAY()
    env = gym.make("AppleGrasp-Play-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()
    robot = env.scene["robot"]
    device = env.device

    arm_joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINT_NAMES, preserve_order=True)
    current_arm_q = robot.data.joint_pos[0, arm_joint_ids].clone()

    # --- cuRobo FK + collision spheres at this exact pose ---
    tensor_args = TensorDeviceType()
    with open(ROBOT_CFG_PATH) as f:
        config_file = yaml.safe_load(f)["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)

    q = current_arm_q.unsqueeze(0).to(**tensor_args.as_torch_dict())
    spheres = kin_model.get_robot_as_spheres(q)[0]  # list[Sphere], length 674

    sphere_to_link = build_sphere_to_link_map(config_file)
    assert len(sphere_to_link) == len(spheres), (
        f"Sphere count mismatch: yaml says {len(sphere_to_link)}, cuRobo returned {len(spheres)}. "
        "collision_link_names order in the yaml may not match cuRobo's internal build order -- "
        "tell me this happened and we'll find another way to attribute names."
    )

    ignore = config_file["kinematics"].get("self_collision_ignore", {}) or {}

    def is_ignored(a, b):
        return b in ignore.get(a, []) or a in ignore.get(b, [])

    # --- Find genuine violations (proper link names this time) ---
    n = len(spheres)
    violating_idx = set()
    violations = []
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = sphere_to_link[i], sphere_to_link[j]
            if li == lj or is_ignored(li, lj):
                continue
            p1, p2 = spheres[i].position, spheres[j].position
            r1, r2 = spheres[i].radius, spheres[j].radius
            d = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            if d < (r1 + r2):
                violating_idx.add(i)
                violating_idx.add(j)
                violations.append((li, lj, d, r1 + r2))

    violations.sort(key=lambda x: x[2] - x[3])
    print(f"\nFound {len(violations)} genuinely colliding LINK pairs (not in self_collision_ignore):")
    seen_pairs = set()
    for a, b, d, thresh in violations:
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        print(f"  {a:30s} <-> {b:30s}  penetration={thresh - d:.4f} m")
    print()

    # --- Spawn a marker for every sphere, in WORLD frame ---
    # cuRobo spheres are in the robot's base_link (pelvis) frame; the pelvis
    # itself sits at world (0, 0, ROBOT_ROOT_HEIGHT).
    robot_base_pos = torch.tensor([0.0, 0.0, ROBOT_ROOT_HEIGHT])
    for i, s in enumerate(spheres):
        world_pos = torch.tensor(s.position) + robot_base_pos
        is_bad = i in violating_idx
        VisualSphere(
            prim_path=f"/World/CuroboSphere_{i:04d}",
            name=f"curobo_sphere_{i}",
            position=world_pos.numpy(),
            radius=float(s.radius),
            color=(torch.tensor([1.0, 0.0, 0.0]) if is_bad else torch.tensor([0.4, 0.4, 0.4])).numpy(),
        )

    print(f"Spawned {n} marker spheres ({len(violating_idx)} shown in red).")
    print("Look at the viewport -- red spheres show exactly where the overlap is.")
    print("Ctrl+C to exit.")

    while simulation_app.is_running():
        env.step(torch.zeros((1, env.action_space.shape[1]), device=device))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
