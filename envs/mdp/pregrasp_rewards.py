# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Rewards that make the pregrasp reach (AppleReachPregrasp-v2) arrive, stop
and STAY at the pregrasp point instead of drifting past it jerkily.

Both read the same goal the part-1 success term uses: the palm point
`standoff` from the apple surface along the palm normal
(reach_rewards._palm_target_dist), and the palm point's world linear speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import SceneEntityCfg

from . import reach_rewards as rr

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _palm_speed(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N] linear speed (m/s) of the palm link."""
    robot = env.scene[robot_cfg.name]
    return torch.norm(robot.data.body_lin_vel_w[:, robot_cfg.body_ids[0]], dim=-1)


def reward_settled_at_goal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    pos_std: float = 0.03,
    v_std: float = 0.05,
) -> torch.Tensor:
    """[N] in (0, 1]: close to the goal AND slow, every step. Unlike the
    binary hold it has a gradient on both factors, so drifting through the
    neighbourhood at speed pays little and parking in it pays fully."""
    d = rr._palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    v = _palm_speed(env, robot_cfg)
    return torch.exp(-((d / pos_std) ** 2)) * torch.exp(-((v / v_std) ** 2))


def penalty_palm_speed_near_goal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    near: float = 0.20,
    v_ref: float = 0.15,
) -> torch.Tensor:
    """[N] (v / v_ref)^2 ramped in over the last `near` metres (0 at `near`,
    1 at the goal): the palm's own speed, in task space, where the joint-space
    brake only sees joint velocities (a small joint motion at the wrist and a
    large one at the shoulder move the palm very differently)."""
    d = rr._palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    v = _palm_speed(env, robot_cfg)
    ramp = (1.0 - d / near).clamp(0.0, 1.0)
    return (v / v_ref) ** 2 * ramp**2


def penalty_target_push(env: "ManagerBasedRLEnv", deadband: float = 0.005, ref: float = 0.02) -> torch.Tensor:
    """[N] how far the TARGET is pushed off where it hangs (m beyond deadband,
    / ref), while its stem holds -- tree_reach.penalty_neighbour_push for
    entity 0. Measured by the env (fruit position), not by any input."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    T = env._tree
    rows = torch.arange(env.num_envs, device=env.device)
    spot = (T.occupant == 0).float().argmax(dim=1)
    rest = T.spot_pos[rows, spot] + env.scene.env_origins
    q = T.objs[0].data.root_pos_w
    hanging = (~T.target_stem.broken & T.target_stem.active).float()
    return ((q - rest).norm(dim=-1) - deadband).clamp(min=0.0) / ref * hanging


def palm_body_reach(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, palm_local_offset: Sequence[float],
                    body_name: str = "torso_link") -> torch.Tensor:
    """[N] horizontal distance (m) of the palm reference point from the body
    (torso_link origin): how far the hand is held out from the robot."""
    robot = env.scene[robot_cfg.name]
    b = robot.data.body_names.index(body_name)
    p = rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    return (p[:, :2] - robot.data.body_pos_w[:, b, :2]).norm(dim=-1)


def penalty_advance_before_height(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    body_reach: float = 0.30,
    reach_ref: float = 0.10,
    tol: float = 0.04,
    z_ref: float = 0.08,
) -> torch.Tensor:
    """[N] rise close to the body first, then go out: the hand held out beyond
    `body_reach` from the torso ((excess / reach_ref)^2), weighted by how far
    the palm still is from the goal's height (0 within `tol`, 1 from tol +
    z_ref). Free once level with the fruit, so the last part is a straight
    horizontal approach instead of a sweep up through the cluster.
    (Self-collision while hugging the body is charged by self_collision.)"""
    t = rr._grasp_target_pos(env, object_cfg, standoff)
    p = rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    w = (((p[:, 2] - t[:, 2]).abs() - tol) / z_ref).clamp(0.0, 1.0)
    out = (palm_body_reach(env, robot_cfg, palm_local_offset) - body_reach).clamp(min=0.0) / reach_ref
    return w * out**2


def reward_contact_tiered(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    palm_links: Sequence[str],
    base_links: Sequence[str],
    good_min: float = 0.5,
    good_max: float = 5.0,
    palm_only: float = 1.0,
    palm_and_base: float = 0.7,
    base_only: float = 0.4,
) -> torch.Tensor:
    """[N] which part of the hand touches the target (each in (good_min,
    good_max] N): palm alone 1.0 > palm + a base phalanx 0.7 > base phalanx
    alone 0.4; nothing else (tips: 0)."""
    from .observations import apple_force_per_link
    palm = apple_force_per_link(env, sensor_cfg, palm_links)
    base = apple_force_per_link(env, sensor_cfg, base_links)
    p = ((palm > good_min) & (palm <= good_max)).any(-1)
    b = ((base > good_min) & (base <= good_max)).any(-1)
    z = torch.zeros(env.num_envs, device=env.device)
    return torch.where(p & ~b, z + palm_only, torch.where(p & b, z + palm_and_base, torch.where(b, z + base_only, z)))
