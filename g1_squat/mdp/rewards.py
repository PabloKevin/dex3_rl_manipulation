"""Squat reward terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .observations import com_offset_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _squat(env, command_name):
    return env.command_manager.get_term(command_name)


# -- task -------------------------------------------------------------------------------
def pelvis_height_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "squat") -> torch.Tensor:
    sq = _squat(env, command_name)
    err = sq.pelvis_height() - sq.ref_height
    return torch.exp(-torch.square(err / std))


def posture_tracking(env: ManagerBasedRLEnv, std: float, command_name: str = "squat") -> torch.Tensor:
    """exp(-mean squared deviation of the 15 leg/waist joints from q_ref(s) / std^2)."""
    sq = _squat(env, command_name)
    robot: Articulation = env.scene["robot"]
    q = robot.data.joint_pos[:, sq.joint_ids]
    return torch.exp(-torch.mean(torch.square(q - sq.ref_joint_pos), dim=1) / std**2)


def com_over_support(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """exp(-|CoM - support centre|^2 / std^2), horizontal."""
    d = com_offset_b(env, asset_cfg)
    return torch.exp(-torch.sum(torch.square(d), dim=1) / std**2)


def stand_settled(env: ManagerBasedRLEnv, std: float, command_name: str = "squat") -> torch.Tensor:
    """Bonus for being still at the stand pose once standing is commanded and the ramp is done.

    This is the hand-over state to the velocity policy, so it should be quiet and exact.
    """
    sq = _squat(env, command_name)
    robot: Articulation = env.scene["robot"]
    standing = (sq.target < 0.5) & (sq.phase <= 0.0)
    q_err = torch.mean(torch.square(robot.data.joint_pos[:, sq.joint_ids] - sq.q_stand), dim=1)
    vel = torch.mean(torch.square(robot.data.joint_vel[:, sq.joint_ids]), dim=1)
    return standing.float() * torch.exp(-q_err / std**2 - 0.1 * vel)


# -- balance / posture shaping ------------------------------------------------------------
def body_roll_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    """Sideways tilt of a body (the squat leans forward, never sideways)."""
    robot: Articulation = env.scene[asset_cfg.name]
    g = quat_apply_inverse(robot.data.body_link_quat_w[:, asset_cfg.body_ids[0]], robot.data.GRAVITY_VEC_W)
    return torch.square(g[:, 1])


def feet_flat_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    quat = robot.data.body_link_quat_w[:, asset_cfg.body_ids]  # (N, 2, 4)
    g = robot.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, quat.shape[1], -1)
    g_foot = quat_apply_inverse(quat.reshape(-1, 4), g.reshape(-1, 3)).reshape(g.shape)
    return torch.sum(torch.square(g_foot[..., :2]), dim=(1, 2))


def feet_contact_lost(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """Number of feet without ground contact (both feet must stay planted)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2].abs().max(dim=1)[0]
    return torch.sum((fz < threshold).float(), dim=1)


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2]), dim=1)


def mirror_symmetry_l2(env: ManagerBasedRLEnv, pairs: list[tuple[str, str, float]]) -> torch.Tensor:
    """sum (q_left - sign * q_right)^2; sign +1 for pitch joints, -1 for roll/yaw joints."""
    robot: Articulation = env.scene["robot"]
    cache = getattr(env, "_g1_squat_mirror", None)
    if cache is None:
        left = robot.find_joints([p[0] for p in pairs], preserve_order=True)[0]
        right = robot.find_joints([p[1] for p in pairs], preserve_order=True)[0]
        sign = torch.tensor([p[2] for p in pairs], device=env.device)
        cache = (left, right, sign)
        env._g1_squat_mirror = cache
    left, right, sign = cache
    q = robot.data.joint_pos
    return torch.sum(torch.square(q[:, left] - sign * q[:, right]), dim=1)


def effort_near_limit(
    env: ManagerBasedRLEnv, ratio: float = 0.8, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """sum over joints of relu(|tau| / tau_max - ratio): keeps every motor (esp. waist pitch) off saturation."""
    robot: Articulation = env.scene[asset_cfg.name]
    tau = robot.data.computed_torque[:, asset_cfg.joint_ids].abs()
    limit = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.relu(tau / limit - ratio), dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids] * robot.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )
