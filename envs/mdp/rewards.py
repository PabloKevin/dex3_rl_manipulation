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

import math

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
    lift_height: float = 0.10,
) -> torch.Tensor:
    """Reward for lifting the apple above ITS per-episode spawn height,
    shaped toward init_height + lift_height.

    r = clamp((z - init) / lift_height, 0, 1)
    Ranges in [0, 1]. Zero until apple is above its own spawn height.

    init_height is read per-env from env._stem_anchor_pos (set every reset by
    randomize_tree_and_apple to the exact height the apple was placed at that
    episode) rather than a fixed nominal constant — domain randomization
    varies spawn height across envs/episodes, so a fixed reference would give
    a spurious reward purely from the random spawn height, unrelated to any
    actual lifting.

    Args:
        lift_height: metres above spawn height at which reward saturates.
    """
    apple: RigidObject = env.scene[object_cfg.name]
    apple_z = apple.data.root_pos_w[:, 2]  # [num_envs]

    if hasattr(env, "_stem_anchor_pos"):
        init_height = env._stem_anchor_pos[:, 2]
    else:
        init_height = apple_z  # before the first reset has ever run

    progress = (apple_z - init_height) / (lift_height + 1e-6)
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

# ─────────────────────────────────────────────
#  Palm pose reward (position below apple + cupping orientation)
# ─────────────────────────────────────────────

def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product. q: [N,4] wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_angle_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (rad) between batched quats [N,4] wxyz."""
    dot = (q1 * q2).sum(-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(dot)


def _euler_xyz_deg_to_quat(roll_deg, pitch_deg, yaw_deg, device) -> torch.Tensor:
    """Single non-batched target quat [4] wxyz. Same convention as
    reach_apple_curobo.py's euler_xyz_deg_to_quat (intrinsic X, Y, Z)."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    qx = torch.tensor([cr, sr, 0.0, 0.0], device=device)
    qy = torch.tensor([cp, 0.0, sp, 0.0], device=device)
    qz = torch.tensor([cy, 0.0, 0.0, sy], device=device)
    return _quat_mul(_quat_mul(qx.unsqueeze(0), qy.unsqueeze(0)), qz.unsqueeze(0)).squeeze(0)


def reward_palm_below_apple(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,   # body_names=["right_hand_palm_link"]
    object_cfg: SceneEntityCfg,  # apple
    offset_z: float = 0.08,
    std: float = 0.05,
) -> torch.Tensor:
    """Reward palm being `offset_z` metres directly below the apple centre."""
    robot: Articulation = env.scene[robot_cfg.name]
    apple: RigidObject = env.scene[object_cfg.name]

    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    target = apple.data.root_pos_w.clone()
    target[:, 2] -= offset_z

    dist = torch.norm(target - palm_pos, dim=-1)
    return torch.exp(-dist**2 / (2 * std**2))


def reward_palm_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,   # body_names=["right_hand_palm_link"]
    torso_cfg: SceneEntityCfg,   # body_names=["torso_link"]
    target_roll_deg: float = 90.0,
    target_pitch_deg: float = 0.0,
    target_yaw_deg: float = 0.0,
    std_rad: float = 0.3,
) -> torch.Tensor:
    """Reward palm orientation matching the torso-frame cupping pose
    (fingers horizontal, palm normal facing up toward the apple) —
    same ROLL/PITCH/YAW convention as reach_apple_curobo.py / orient_tune.py.
    Tune those two scripts first to confirm the exact angles for YOUR urdf,
    then hardcode them here.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    palm_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    torso_quat = robot.data.body_quat_w[:, torso_cfg.body_ids[0], :]

    if not hasattr(env, "_palm_target_local_quat"):
        env._palm_target_local_quat = _euler_xyz_deg_to_quat(
            target_roll_deg, target_pitch_deg, target_yaw_deg, env.device
        )
    target_local = env._palm_target_local_quat.unsqueeze(0).expand(torso_quat.shape[0], -1)
    target_world = _quat_mul(torso_quat, target_local)

    ang_err = _quat_angle_diff(palm_quat, target_world)
    return torch.exp(-(ang_err**2) / (2 * std_rad**2))