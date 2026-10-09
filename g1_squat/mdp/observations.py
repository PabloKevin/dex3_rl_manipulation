"""Squat-specific observations (critic-only unless noted)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .commands import sole_centres_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _squat(env, command_name):
    return env.command_manager.get_term(command_name)


def pelvis_height_error(env: ManagerBasedRLEnv, command_name: str = "squat") -> torch.Tensor:
    sq = _squat(env, command_name)
    return (sq.pelvis_height() - sq.ref_height).unsqueeze(-1)


def whole_body_com_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Whole-body centre of mass, world frame (N, 3). Masses cached after startup randomisation."""
    robot: Articulation = env.scene[asset_cfg.name]
    cache = getattr(env, "_g1_squat_masses", None)
    if cache is None:
        cache = robot.root_physx_view.get_masses().to(env.device)  # (N, B)
        env._g1_squat_masses = cache
    com = robot.data.body_com_pos_w  # (N, B, 3)
    return (com * cache.unsqueeze(-1)).sum(dim=1) / cache.sum(dim=1, keepdim=True)


def com_offset_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
) -> torch.Tensor:
    """CoM minus the support centre (mid-point of the two soles), horizontal, yaw frame (N, 2)."""
    robot: Articulation = env.scene[asset_cfg.name]
    support = sole_centres_w(robot, asset_cfg.body_ids).mean(dim=1)
    delta = whole_body_com_w(env, SceneEntityCfg(asset_cfg.name)) - support
    delta[:, 2] = 0.0
    return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), delta)[:, :2]


def feet_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] / 100.0


def arm_targets(env: ManagerBasedRLEnv, command_name: str = "arm_motion") -> torch.Tensor:
    return env.command_manager.get_term(command_name).command


def hand_loads(env: ManagerBasedRLEnv, command_name: str = "arm_motion") -> torch.Tensor:
    return env.command_manager.get_term(command_name).load_forces / 10.0


def torso_projected_gravity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
    return quat_apply_inverse(quat, robot.data.GRAVITY_VEC_W)
