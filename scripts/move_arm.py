#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Phase 1 — Scripted reach + grasp motion (NO policy, NO RL).

Stages:
  1. HOME  → wait 1 s at default pose
  2. REACH → smoothly interpolate right arm toward apple over 3 s
  3. CLOSE → smoothly close Dex3 hand over 2 s
  4. LIFT  → raise arm 10 cm over 2 s
  5. HOLD  → hold for 2 s
  6. OPEN  → open hand and return to home over 2 s

Purpose: validate that
  - Joint names / actuator config are correct
  - Arm kinematics can reach the apple position
  - Dex3 fingers close correctly on the sphere
  - Contact sensors fire during grasp
  - The apple is lifted (rigid body follows the hand)

Usage:
    python scripts/move_arm.py

    # Slow-motion (physics slower, easier to observe):
    python scripts/move_arm.py --slow

    # Print contact forces during grasp:
    python scripts/move_arm.py --verbose

TODO: After running, check the printed "apple_z" during LIFT stage.
      If it does not increase, the grasp is not working — tune Dex3
      joint targets or stiffness in apple_grasp_env_cfg.py.
"""

import argparse
import sys
import os
#import math
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted Reach + Grasp")
parser.add_argument("--slow",         action="store_true", help="Half-speed for easier observation")
parser.add_argument("--show_contacts", action="store_true", help="Print contact forces each step")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # registers gym tasks

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

from envs.apple_grasp_env_cfg import AppleGraspEnvCfg_PLAY


# ─────────────────────────────────────────────────────────────────────────────
#  JOINT TARGETS  — tune these to match your URDF coordinate convention
# ─────────────────────────────────────────────────────────────────────────────
# All values in radians, relative to the joint's default (neutral) position.
# Run `python scripts/run_env.py --debug_joints` to confirm joint names,
# then iterate on these values until the arm visually reaches the sphere.

# Right arm: 7 joints
#   [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#    wrist_roll, wrist_pitch, wrist_yaw]
ARM_HOME    = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]   # neutral
ARM_REACH   = [-np.pi/2, -0.3,  0.1,  0.9,  0.0,  0.2,  0.1]   # reaching forward-down
ARM_LIFT    = [0.4, -0.3,  0.1,  0.7,  0.0,  0.0,  0.1]   # lifted 10 cm higher

# Right Dex3 hand: 7 joints
# TODO: Replace with values that fully close YOUR hand URDF.
# From g1_dex3_example.cpp, mid-range = (max+min)/2 which varies per joint.
# Start with all-zeros (open) → all-ones-normalised (close) and iterate.
HAND_OPEN   = [0.0] * 7   # fully open
HAND_CLOSE  = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]  # ← tune per finger


def slerp(a: list, b: list, t: float) -> torch.Tensor:
    """Smooth 5th-order interpolation between joint target vectors.
    t in [0, 1].  Uses smoothstep: 6t^5 - 15t^4 + 10t^3
    """
    t = max(0.0, min(1.0, t))
    s = 6*t**5 - 15*t**4 + 10*t**3   # smooth-step
    out = [a_i + (b_i - a_i) * s for a_i, b_i in zip(a, b)]
    return out


def build_action(arm_targets: list, hand_targets: list, device) -> torch.Tensor:
    """Concatenate arm + hand into a [1, 14] action tensor."""
    combined = arm_targets + hand_targets
    return torch.tensor([combined], dtype=torch.float32, device=device)


def main():
    cfg = AppleGraspEnvCfg_PLAY()
    speed = 0.5 if args_cli.slow else 1.0

    env = gym.make("AppleGrasp-v0", cfg=cfg).unwrapped
    env.reset()

    # Hold apple to branch with breakable stem joint
    from envs.mdp.tree_utils import create_stem_joints, check_stem_broken
    stem_joints = create_stem_joints(env)

    robot:  Articulation  = env.scene["robot"]
    apple:  RigidObject   = env.scene["apple"]
    try:
        sensor: ContactSensor = env.scene["contact_forces"]
    except Exception:
        sensor = None
    dev = env.device

    ctrl_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    def secs(t): return int(t * ctrl_hz * speed)

    stages = [
        ("HOME",  secs(1.0), ARM_HOME,  HAND_OPEN),
        ("REACH", secs(3.0), ARM_REACH, HAND_OPEN),
        ("CLOSE", secs(2.0), ARM_REACH, HAND_CLOSE),
        ("LIFT",  secs(2.0), ARM_LIFT,  HAND_CLOSE),
        ("HOLD",  secs(2.0), ARM_LIFT,  HAND_CLOSE),
        ("OPEN",  secs(2.0), ARM_HOME,  HAND_OPEN),
    ]

    print("\n" + "="*60)
    print("SCRIPTED REACH + GRASP")
    print(f"  Control rate: {ctrl_hz:.1f} Hz")
    print(f"  Speed: {'slow (0.5x)' if args_cli.slow else 'normal'}")
    print("="*60)
    for name, n, _, _ in stages:
        print(f"  {name:<8}  {n} steps  ({n/ctrl_hz:.1f} s)")
    print("="*60 + "\n")

    # ── State machine variables ───────────────────────────────────────────
    # We step ONE action per render loop iteration so the renderer never blocks.
    stage_idx   = 0
    stage_step  = 0          # step within current stage
    global_step = 0
    prev_arm    = list(ARM_HOME)
    prev_hand   = list(HAND_OPEN)
    done        = False

    stage_name, n_steps, arm_target, hand_target = stages[stage_idx]
    print(f"── Stage: {stage_name} ──────────────")

    with torch.inference_mode():
        while simulation_app.is_running() and not done:

            # ── Compute interpolated action for this step ─────────────────
            t       = (stage_step + 1) / n_steps
            arm     = slerp(prev_arm,  arm_target,  t)
            hand    = slerp(prev_hand, hand_target, t)
            actions = build_action(arm, hand, dev)

            # ── Step the environment (physics + render) ───────────────────
            _, _, terminated, truncated, _ = env.step(actions)
            global_step += 1
            stage_step  += 1

            # ── Periodic print ────────────────────────────────────────────
            if global_step % 25 == 0:
                apple_z = apple.data.root_pos_w[0, 2].item()
                print(f"  [{stage_name}] step={global_step:5d}  apple_z={apple_z:.4f} m", end="")
                if sensor is not None and args_cli.show_contacts:
                    norms = sensor.data.net_forces_w[0].norm(dim=-1)
                    print(f"  contacts={norms.tolist()}", end="")
                print()

            # ── Unexpected episode end ────────────────────────────────────
            if terminated.any() or truncated.any():
                print(f"\n  ⚠  Episode ended at step {global_step}!")
                done = True
                continue

            # ── Advance to next stage when current one finishes ───────────
            if stage_step >= n_steps:
                prev_arm  = arm_target
                prev_hand = hand_target
                stage_idx += 1

                if stage_idx >= len(stages):
                    done = True
                    print("\n" + "="*60)
                    print("Scripted motion complete.")
                    print("Check the viewer: did the apple lift during LIFT stage?")
                    print("  YES → grasp config is working, move to Phase 2 (RL).")
                    print("  NO  → tune HAND_CLOSE targets or Dex3 stiffness in env cfg.")
                    print("="*60)
                else:
                    stage_name, n_steps, arm_target, hand_target = stages[stage_idx]
                    stage_step = 0
                    print(f"── Stage: {stage_name} ──────────────")

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