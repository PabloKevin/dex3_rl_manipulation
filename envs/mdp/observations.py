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
from isaaclab.utils.math import matrix_from_quat, quat_conjugate, quat_mul

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
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Apple position relative to a point on the robot body link (world-axis
    aligned, just translated — NOT rotated into the body's own frame despite
    the name; translation-invariant w.r.t. env origin is the only thing this
    buys over a raw world-frame difference).

    `local_offset` (in the body link's OWN local frame, same convention as
    reach_rewards.PALM_LOCAL_OFFSET) shifts the reference point off the link
    origin — e.g. from right_hand_palm_link's origin (near the wrist) out to
    where the fingers/contact pads actually are — so this observation lines
    up with whatever point reach_rewards._palm_ref_pos is scoring against.
    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # World positions
    apple_pos_w = obj.data.root_pos_w  # [num_envs, 3]

    # Body position in world frame (first matched body, e.g. palm link)
    body_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]  # [num_envs, 3]

    if any(o != 0.0 for o in local_offset):
        quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
        rot = matrix_from_quat(quat)  # [num_envs, 3, 3], local -> world
        offset = torch.tensor(local_offset, device=env.device, dtype=body_pos_w.dtype)
        body_pos_w = body_pos_w + torch.matmul(rot, offset)

    # Also subtract the env origin so relative pos is env-independent
    env_origins = env.scene.env_origins  # [num_envs, 3]

    apple_local = apple_pos_w - env_origins
    body_local = body_pos_w - env_origins

    return apple_local - body_local  # [num_envs, 3]


def body_quat_relative_to_body(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,      # body to report orientation of, e.g. right_hand_palm_link
    reference_cfg: SceneEntityCfg,  # reference frame body, e.g. torso_link
) -> torch.Tensor:
    """One robot body's orientation expressed relative to another (e.g. palm
    relative to torso) — same torso-relative frame reward_palm_orientation
    scores against, so the policy can observe directly what that reward wants.

    Shape: [num_envs, 4] (w, x, y, z)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    ref_quat = robot.data.body_quat_w[:, reference_cfg.body_ids[0], :]
    return quat_mul(quat_conjugate(ref_quat), body_quat)


def object_vel_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object root linear velocity in world frame.

    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w
