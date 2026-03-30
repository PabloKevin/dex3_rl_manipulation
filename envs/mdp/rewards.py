# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Custom reward functions for the Apple Grasp task.

Each function signature: fn(env, **params) -> torch.Tensor  [num_envs]
Scalar per environment per step. Multiplied by weight in RewardsCfg.

Curriculum design:
  Phase 1 active:  reaching, action_penalty
  Phase 2 active:  + contact
  Phase 3 active:  + lift, hold
  Phase 4 active:  + place (bring to basket)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Reaching reward
# ─────────────────────────────────────────────

def reward_reaching(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.15,
) -> torch.Tensor:
    """Gaussian-shaped reward for end-effector proximity to apple.

    r = exp(−dist² / (2·std²))
    Ranges in [0, 1]. std controls how "peaked" the reward is.

    Args:
        robot_cfg: SceneEntityCfg with body_names=[wrist_link_name]
        object_cfg: SceneEntityCfg for the apple RigidObject
        std: width of the gaussian in metres
    """
    robot: Articulation = env.scene[robot_cfg.name]
    apple: RigidObject  = env.scene[object_cfg.name]

    # Wrist link position in world frame [num_envs, 3]
    wrist_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    apple_pos = apple.data.root_pos_w  # [num_envs, 3]

    dist = torch.norm(apple_pos - wrist_pos, dim=-1)  # [num_envs]
    return torch.exp(-dist**2 / (2 * std**2))


# ─────────────────────────────────────────────
#  Contact reward
# ─────────────────────────────────────────────

def reward_fingertip_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Reward proportional to number of fingertips in contact with apple.

    r = (num_fingers_in_contact / num_total_fingers)
    Ranges in [0, 1].

    Args:
        threshold: minimum contact force (N) to count as "touching"
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # net_forces_w: [num_envs, num_bodies, 3]
    forces = sensor.data.net_forces_w
    force_norms = torch.norm(forces, dim=-1)          # [num_envs, num_bodies]
    in_contact   = (force_norms > threshold).float()  # [num_envs, num_bodies]
    num_fingers  = force_norms.shape[-1]
    return in_contact.sum(dim=-1) / num_fingers       # [num_envs]


def reward_grasp_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    min_fingers: int = 3,
    max_force: float = 10.0,
) -> torch.Tensor:
    """Reward for a stable, balanced grasp.

    Combines:
      - At least min_fingers in contact
      - Balanced force distribution (low std dev across fingers)
      - Not crushing (force below max_force)

    Use this as a replacement for reward_fingertip_contact in Phase 3+.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w                    # [N, F, 3]
    force_norms = torch.norm(forces, dim=-1)             # [N, F]

    # Enough fingers in contact
    in_contact = (force_norms > 0.1).sum(dim=-1).float()  # [N]
    enough_contact = (in_contact >= min_fingers).float()   # [N]

    # Balance: low coefficient of variation across fingers
    mean_force = force_norms.mean(dim=-1, keepdim=True) + 1e-6
    cv = force_norms.std(dim=-1) / mean_force.squeeze()    # [N]
    balance = torch.exp(-cv)                               # high when balanced

    # Not crushing
    max_f = force_norms.max(dim=-1).values                # [N]
    not_crushing = (max_f < max_force).float()

    return enough_contact * balance * not_crushing


# ─────────────────────────────────────────────
#  Lift reward
# ─────────────────────────────────────────────

def reward_lift(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    init_height: float,
    target_height: float,
) -> torch.Tensor:
    """Reward for lifting the apple above init height, shaped toward target.

    r = clamp((z - init) / (target - init), 0, 1)
    Ranges in [0, 1]. Zero until apple is above init_height.

    Args:
        init_height:   z of apple at reset (world frame)
        target_height: z at which reward saturates
    """
    apple: RigidObject = env.scene[object_cfg.name]
    apple_z = apple.data.root_pos_w[:, 2]  # [num_envs]

    height_delta = target_height - init_height
    progress = (apple_z - init_height) / (height_delta + 1e-6)
    return progress.clamp(0.0, 1.0)


def reward_hold(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_height: float,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Reward for keeping the apple at a target height (stable hold).

    r = exp(−|z - target|² / tolerance²)

    Args:
        target_height: desired z position to hold at
        tolerance: height window in metres
    """
    apple: RigidObject = env.scene[object_cfg.name]
    apple_z = apple.data.root_pos_w[:, 2]
    return torch.exp(-((apple_z - target_height) ** 2) / (tolerance**2))


# ─────────────────────────────────────────────
#  Place reward (Phase 4)
# ─────────────────────────────────────────────

def reward_place(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float],
    std: float = 0.10,
) -> torch.Tensor:
    """Gaussian reward for bringing apple close to a goal position (basket).

    Args:
        target_pos: (x, y, z) world position of basket centre
        std: gaussian width in metres
    """
    apple: RigidObject = env.scene[object_cfg.name]
    goal = torch.tensor(target_pos, device=env.device).unsqueeze(0)  # [1, 3]
    dist = torch.norm(apple.data.root_pos_w - goal, dim=-1)          # [N]
    return torch.exp(-dist**2 / (2 * std**2))


# ─────────────────────────────────────────────
#  Penalty terms
# ─────────────────────────────────────────────

def penalty_joint_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 0.9,
) -> torch.Tensor:
    """Penalty for joints approaching their limits.

    Returns negative scalar; weight should be negative in RewardsCfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    pos    = asset.data.joint_pos[:, asset_cfg.joint_ids]

    lower = limits[..., 0]
    upper = limits[..., 1]
    margin = (upper - lower) * (1.0 - soft_ratio) / 2.0

    violated_lower = (pos < lower + margin).float()
    violated_upper = (pos > upper - margin).float()
    return (violated_lower + violated_upper).sum(dim=-1)  # [N]


def penalty_action_rate(
    env: ManagerBasedRLEnv,
    action_key: str = "right_arm",
) -> torch.Tensor:
    """Penalty for large changes in actions between steps (jerk penalty).

    Requires env.action_manager to store previous actions.
    """
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    delta = curr - prev
    return (delta**2).sum(dim=-1)  # [N]
