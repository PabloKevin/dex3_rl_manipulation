# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Custom observation functions for the Apple Grasp task.

Each function signature: fn(env, **params) -> torch.Tensor  [num_envs, dim]
They are referenced by ObservationTermCfg(func=...) in the env cfg.
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
#  Joint state observations
# ─────────────────────────────────────────────

def joint_pos_selected(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Joint positions (radians) for selected joints, relative to default pose.

    Shape: [num_envs, len(joint_ids)]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # joint_ids resolved at runtime from joint_names in SceneEntityCfg
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - \
           asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_selected(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Joint velocities (rad/s) for selected joints.

    Shape: [num_envs, len(joint_ids)]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_effort_selected(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Applied joint torques (N·m) for selected joints.

    Shape: [num_envs, len(joint_ids)]
    Note: Isaac Lab exposes this as `applied_torque` on the articulation data.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


# ─────────────────────────────────────────────
#  Contact / tactile observations
# ─────────────────────────────────────────────

def fingertip_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flattened contact force vectors for all tracked finger bodies.

    Shape: [num_envs, num_bodies * 3]
    Each 3-vector is the net contact force (x, y, z) in world frame.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # net_forces_w: [num_envs, num_bodies, 3]
    forces = sensor.data.net_forces_w
    return forces.reshape(forces.shape[0], -1)


def fingertip_contact_normals(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 norm of contact force per fingertip (scalar per finger).

    Useful as a compact tactile signal.
    Shape: [num_envs, num_bodies]
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    return torch.norm(forces, dim=-1)  # [num_envs, num_bodies]


# ─────────────────────────────────────────────
#  Object position observations
# ─────────────────────────────────────────────

def object_pos_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object root position in world frame.

    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w


def object_pos_relative_to_body(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apple position expressed in the robot body link's local frame.

    Gives the policy a translation-invariant signal regardless of env offset.
    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # World positions
    apple_pos_w = obj.data.root_pos_w  # [num_envs, 3]

    # Body positions in world frame [num_envs, num_bodies, 3]
    body_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    # Take first matched body (wrist link)
    wrist_pos_w = body_pos_w[:, 0, :]  # [num_envs, 3]

    # Also subtract the env origin so relative pos is env-independent
    env_origins = env.scene.env_origins  # [num_envs, 3]

    apple_local  = apple_pos_w  - env_origins
    wrist_local  = wrist_pos_w  - env_origins

    return apple_local - wrist_local  # [num_envs, 3]


def object_vel_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object root linear velocity in world frame.

    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w
