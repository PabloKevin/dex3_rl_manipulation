#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 1 — View the Apple Grasp environment.

Usage:
    # Basic viewer (holds sim, listens for targets from tune_arm.py):
    ~/IsaacLab/isaaclab.sh -p scripts/run_env.py

    # While sim is running, in a second terminal:
    python3 scripts/tune_arm.py

    # Multiple envs:
    ~/IsaacLab/isaaclab.sh -p scripts/run_env.py --num_envs 4

    # Print all robot link names:
    ~/IsaacLab/isaaclab.sh -p scripts/run_env.py --debug_links

    # Print robot joint names:
    ~/IsaacLab/isaaclab.sh -p scripts/run_env.py --debug_joints
"""

import argparse
import sys
import os
import threading
import json

# ── Isaac Lab app must be set up BEFORE any omni/isaacsim imports ────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Apple Grasp Environment Viewer")
parser.add_argument("--num_envs",       type=int,  default=1,    help="Number of parallel envs")
parser.add_argument("--debug_links",    action="store_true",      help="Print all robot link names and exit")
parser.add_argument("--debug_joints",   action="store_true",      help="Print all robot joint names and exit")
parser.add_argument("--random_actions", action="store_true",      help="Send random actions instead of zeros")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab / torch ─────────────────────────────────────
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, AppleGraspEnvCfg_PLAY

# ── DDS subscriber for live arm targets from tune_arm.py ─────────────────────
# Receives JSON: {"arm": [7 floats], "hand": [7 floats]}
# Published by tune_arm.py on channel rt/arm_tune/targets
_targets_lock = threading.Lock()
_arm_targets  = None   # None = hold default pose (zero actions)
_hand_targets = None

def _start_dds_subscriber():
    """Start DDS subscriber in a background thread. Non-fatal if unavailable."""
    try:
        from unitree_sdk2py.core.channel import (
            ChannelSubscriber, ChannelFactoryInitialize
        )
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        ChannelFactoryInitialize(1)   # channel 1 — same as tune_arm.py

        def _callback(msg: String_):
            global _arm_targets, _hand_targets
            try:
                data = json.loads(msg.data)
                with _targets_lock:
                    _arm_targets  = data.get("arm",  None)
                    _hand_targets = data.get("hand", None)
            except Exception as e:
                print(f"[DDS] parse error: {e}")

        sub = ChannelSubscriber("rt/arm_tune/targets", String_)
        sub.Init(_callback, 1)
        print("[DDS] Subscribed to rt/arm_tune/targets — tune_arm.py can now connect.")
    except Exception as e:
        print(f"[DDS] Subscriber unavailable ({e}). Running with zero actions only.")


def _get_actions(device, action_shape):
    """Return action tensor from latest DDS targets, or zeros if none received."""
    with _targets_lock:
        arm  = list(_arm_targets)  if _arm_targets  is not None else [0.0] * 7
        hand = list(_hand_targets) if _hand_targets is not None else [0.0] * 7
    combined = arm + hand
    return torch.tensor([combined], dtype=torch.float32, device=device)


def main():
    _start_dds_subscriber()

    if args_cli.num_envs == 1:
        cfg = AppleGraspEnvCfg_PLAY()
    else:
        cfg = AppleGraspEnvCfg()
        cfg.scene.num_envs = args_cli.num_envs

    env      = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
    obs, _   = env.reset()

    from envs.mdp.tree_utils import StemManager
    stem = StemManager(env)

    robot: Articulation = env.scene["robot"]

    if args_cli.debug_links:
        print("\n" + "="*60)
        print("ROBOT BODY (LINK) NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.body_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        env.close()
        return

    if args_cli.debug_joints:
        print("\n" + "="*60)
        print("ROBOT JOINT NAMES:")
        print("="*60)
        for i, name in enumerate(robot.data.joint_names):
            print(f"  [{i:3d}]  {name}")
        print("="*60)
        env.close()
        return

    print("\n" + "="*60)
    print("APPLE GRASP VIEWER")
    print(f"  Physics dt : {cfg.sim.dt*1000:.1f} ms  ({1/cfg.sim.dt:.0f} Hz)")
    print(f"  Control dt : {cfg.sim.dt*cfg.decimation*1000:.1f} ms  ({1/(cfg.sim.dt*cfg.decimation):.0f} Hz)")
    print(f"  Action dim : {env.action_space.shape}")
    print(f"  Obs dim    : {obs['policy'].shape}")
    print("="*60)
    print("  Run  python3 scripts/tune_arm.py  in a second terminal to control joints.")
    print("  Without tune_arm.py the robot holds its default pose.")
    print("="*60 + "\n")

    step  = 0
    apple: RigidObject = env.scene["apple"]
    try:
        sensor: ContactSensor = env.scene["contact_forces"]
    except Exception:
        sensor = None

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.random_actions:
                raw     = env.action_space.sample()
                actions = torch.tensor(raw, device=env.device, dtype=torch.float32)
            else:
                actions = _get_actions(env.device, env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(actions)
            stem.update(env)
            step += 1

            if terminated.any() or truncated.any():
                stem.reset(env)

            if step % 100 == 0:
                apple_z = apple.data.root_pos_w[0, 2].item()
                rew_val = reward[0].item() if reward.numel() > 0 else 0.0
                print(f"  step={step:6d}  apple_z={apple_z:.3f}m  reward={rew_val:.4f}", end="")
                if sensor is not None:
                    total_f = sensor.data.net_forces_w[0].norm(dim=-1).sum().item()
                    print(f"  contact_N={total_f:.3f}", end="")
                if any(stem.is_broken()):
                    print("  STEM BROKEN 🍎", end="")
                print()

    env.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"\n>>> CRASH: {e}")
    finally:
        simulation_app.close()