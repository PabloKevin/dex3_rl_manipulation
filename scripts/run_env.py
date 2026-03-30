#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 1 — View the Apple Grasp environment.

Usage:
    # Basic viewer (1 env, interactive UI):
    python scripts/run_env.py

    # Multiple envs (check spacing):
    python scripts/run_env.py --num_envs 4

    # Print all robot link names (to verify contact sensor paths):
    python scripts/run_env.py --debug_links

    # Print robot joint names (to verify actuator name patterns):
    python scripts/run_env.py --debug_joints

    # Random actions instead of zero:
    python scripts/run_env.py --random_actions

Notes:
    - Run from the repo root: `python scripts/run_env.py`
    - In distrobox, make sure your Isaac Lab conda env is active:
          conda activate isaaclab
    - Set G1_DEX3_USD_PATH env var or edit apple_grasp_env_cfg.py directly.
    - DDS channel: this script does NOT use DDS (standalone viewer).
"""

import argparse
import sys
import os

# ── Isaac Lab app must be set up BEFORE any omni/isaacsim imports ────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apple Grasp Environment Viewer")
parser.add_argument("--num_envs",      type=int,  default=1,     help="Number of parallel envs")
parser.add_argument("--debug_links",   action="store_true",       help="Print all robot link (body) names and exit")
parser.add_argument("--debug_joints",  action="store_true",       help="Print all robot joint names and exit")
parser.add_argument("--random_actions",action="store_true",       help="Send random actions instead of zeros")
parser.add_argument("--headless",      action="store_true",       help="Run without GUI")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── Now safe to import Isaac Lab / torch ─────────────────────────────────────
import torch
import gymnasium as gym

# Add repo root to path so `envs` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, AppleGraspEnvCfg_PLAY


def main():
    # Use play config (no randomisation) for single-env viewing
    if args_cli.num_envs == 1:
        cfg = AppleGraspEnvCfg_PLAY()
    else:
        cfg = AppleGraspEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
    obs, _ = env.reset()

    # ── Debug: print link and joint names then exit ───────────────────────
    robot: Articulation = env.scene["robot"]

    if args_cli.debug_links:
        print("\n" + "="*60)
        print("ROBOT BODY (LINK) NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.body_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        print("\nUse these names to set:")
        print("  - HEAD_CAMERA_LINK in apple_grasp_env_cfg.py")
        print("  - contact_forces prim_path regex")
        print("  - body_names in SceneEntityCfg for wrist link")
        env.close()
        return

    if args_cli.debug_joints:
        print("\n" + "="*60)
        print("ROBOT JOINT NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.joint_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        print("\nUse these names to set actuator joint_names_expr patterns")
        print("and the action space joint_names in ActionsCfg.")
        env.close()
        return

    # ── Main loop ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("APPLE GRASP VIEWER — Phase 1")
    print(f"  Envs:        {cfg.scene.num_envs}")
    print(f"  Physics dt:  {cfg.sim.dt*1000:.1f} ms  ({1/cfg.sim.dt:.0f} Hz)")
    print(f"  Control dt:  {cfg.sim.dt*cfg.decimation*1000:.1f} ms  ({1/(cfg.sim.dt*cfg.decimation):.0f} Hz)")
    print(f"  Episode len: {cfg.episode_length_s} s")
    print(f"  Action dim:  {env.action_space.shape}")
    print(f"  Obs dim:     {obs['policy'].shape}")
    print("="*60)
    print("  Viewer: click and drag to orbit, scroll to zoom.")
    print("  Press Ctrl+C to exit.")
    print("="*60 + "\n")

    step = 0
    apple: RigidObject  = env.scene["apple"]
    sensor: ContactSensor = env.scene.get("contact_forces", None)

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.random_actions:
                actions = env.action_space.sample()
                actions = torch.tensor(actions, device=env.device, dtype=torch.float32)
            else:
                # Zero actions → robot holds its default joint positions
                actions = torch.zeros(env.action_space.shape, device=env.device)

            obs, reward, terminated, truncated, info = env.step(actions)
            step += 1

            # Print status every 100 steps
            if step % 100 == 0:
                apple_z  = apple.data.root_pos_w[0, 2].item()
                rew_val  = reward[0].item() if reward.numel() > 0 else 0.0
                print(f"  step={step:6d}  apple_z={apple_z:.3f}m  reward={rew_val:.4f}", end="")

                if sensor is not None:
                    total_force = sensor.data.net_forces_w[0].norm(dim=-1).sum().item()
                    print(f"  total_contact_N={total_force:.3f}", end="")

                print()

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
