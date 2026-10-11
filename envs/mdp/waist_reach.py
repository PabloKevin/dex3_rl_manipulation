# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Reach with the waist yaw (envs/apple_reach_tree_waist_env_cfg.py): the
observation that has to change when the torso turns, and the rewards that
the first run (2026-10-10_05-01-43) showed were missing.

The tree reach observes the palm orientation relative to torso_link. With
the waist yaw driven, that frame turns with the torso, while the rewards
(palm normal horizontal and toward the stem) are world-frame: the same
torso-relative reading would mean a different world orientation at every
yaw. palm_quat_torso_unyawed removes the waist yaw from the reference frame,
so it reads exactly like the tree reach's input at yaw 0 (a warm start
keeps its meaning) and does not change when only the torso turns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_from_euler_xyz, quat_mul

from . import reach_rewards as rr

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def palm_quat_torso_unyawed(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,      # e.g. right_hand_palm_link
    reference_cfg: SceneEntityCfg,  # torso_link
    pelvis_cfg: SceneEntityCfg,     # pelvis (waist_yaw_joint's parent)
    waist_cfg: SceneEntityCfg,      # waist_yaw_joint
) -> torch.Tensor:
    """[N, 4] (w, x, y, z) palm orientation relative to the torso frame with
    the waist yaw taken out.

    waist_yaw_joint is the first joint from the pelvis to the torso and turns
    about the pelvis z axis, so torso = pelvis * Rz(yaw) * rest. The
    reference is pelvis * rest = (pelvis * Rz(-yaw) * pelvis^-1) * torso.
    On the robot: palm relative to the pelvis by FK, times the constant rest
    (the locked waist roll/pitch)."""
    robot = env.scene[robot_cfg.name]
    q_palm = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    q_torso = robot.data.body_quat_w[:, reference_cfg.body_ids[0]]
    q_pelvis = robot.data.body_quat_w[:, pelvis_cfg.body_ids[0]]
    yaw = robot.data.joint_pos[:, waist_cfg.joint_ids[0]]
    zero = torch.zeros_like(yaw)
    unyaw = quat_mul(quat_mul(q_pelvis, quat_from_euler_xyz(zero, zero, -yaw)), quat_conjugate(q_pelvis))
    q_ref = quat_mul(unyaw, q_torso)
    return quat_mul(quat_conjugate(q_ref), q_palm)


def reward_reach_hold_heading(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    heading_full_rad: float,
    heading_zero_rad: float,
    pos_tol: float = 0.04,
    tilt_tol_rad: float | None = None,
    standoff: float = 0.015,
    palm_normal_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_cfg: SceneEntityCfg | None = None,
    pressure_min: float = 0.5,
    pressure_max: float | None = 5.0,
) -> torch.Tensor:
    """reach_rewards.reward_reach_hold, weighted by how far the palm normal is
    from the stem direction (the angle the success gate checks): full up to
    heading_full_rad, fading linearly to 0 at heading_zero_rad.

    The first waist run touched central apples gently but from the side
    (palm 51-67 deg off the stem, gate 45 deg): the plain hold paid for that
    every step, so it never turned the palm.

    Returns: [num_envs]
    """
    hold = rr.reward_reach_hold(
        env, robot_cfg, object_cfg, pos_tol=pos_tol, tilt_tol_rad=tilt_tol_rad, standoff=standoff,
        palm_normal_axis=palm_normal_axis, palm_local_offset=palm_local_offset, sensor_cfg=sensor_cfg,
        pressure_min=pressure_min, pressure_max=pressure_max)
    theta = rr._palm_normal_angle(env, robot_cfg, palm_normal_axis, object_cfg)
    fade = ((heading_zero_rad - theta) / (heading_zero_rad - heading_full_rad)).clamp(0.0, 1.0)
    return hold * fade


def waist_yaw_hint(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    standoff: float,
    target_az_rad: float,
    limit_rad: float,
    pelvis_body: str = "pelvis",
    n_grid: int = 91,
) -> torch.Tensor:
    """[N] waist yaw that turns the torso so the grasp target sits at
    target_az_rad from the right shoulder, in the torso's own frame (the
    azimuth where the arm-only reach already succeeds), clamped to
    +-limit_rad. Grid search over the yaw: the shoulder swings around the
    waist axis (through the pelvis origin) as the torso turns."""
    robot = env.scene["robot"]
    if not hasattr(env, "_hint_pelvis_id"):
        env._hint_pelvis_id = robot.data.body_names.index(pelvis_body)
        env._hint_grid = torch.linspace(-limit_rad, limit_rad, n_grid, device=env.device)
    pelvis = robot.data.body_pos_w[:, env._hint_pelvis_id, :2]
    # fixed base, pelvis yaw 0: the env frame is the pelvis frame
    s0 = (env.scene.env_origins[:, :2] + env._shoulder_local_xyz[:2]) - pelvis      # shoulder at yaw 0
    a = rr._grasp_target_pos(env, object_cfg, standoff)[:, :2] - pelvis
    psi = env._hint_grid.view(1, -1)                                                  # [1, G]
    c, s = torch.cos(psi), torch.sin(psi)
    ax = c * a[:, :1] + s * a[:, 1:2] - s0[:, :1]                                     # Rz(-psi) a - s0
    ay = -s * a[:, :1] + c * a[:, 1:2] - s0[:, 1:2]
    err = torch.remainder(torch.atan2(ay, ax) - target_az_rad + math.pi, 2 * math.pi) - math.pi
    return env._hint_grid[err.abs().argmin(dim=1)]


def reward_waist_toward_target(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    waist_cfg: SceneEntityCfg,
    standoff: float,
    target_az_rad: float,
    limit_rad: float,
    std_rad: float = 0.25,
) -> torch.Tensor:
    """Gaussian reward for the waist yaw near waist_yaw_hint: turn the torso
    to present the right shoulder to the fruit. A hint for exploration (the
    first run's waist barely explored); keep its weight below the heading /
    success terms so they decide when the two disagree -- turning left also
    swings the stem outward in the torso frame.

    Returns: [num_envs]
    """
    yaw = env.scene[waist_cfg.name].data.joint_pos[:, waist_cfg.joint_ids[0]]
    hint = waist_yaw_hint(env, object_cfg, standoff, target_az_rad, limit_rad)
    return torch.exp(-((yaw - hint) ** 2) / (2.0 * std_rad ** 2))


#: Head camera (D435 on the torso); torso_link's origin if the articulation
#: does not expose it as a body.
CAMERA_BODY = "d435_link"


def camera_bearing_error(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N] horizontal angle (rad, >= 0) between where the torso faces and the
    line from the head camera to the apple: 0 = apple straight ahead of the
    camera. Yaw only (the camera pitch is fixed on the torso)."""
    robot = env.scene["robot"]
    if not hasattr(env, "_face_ids"):
        names = robot.data.body_names
        torso = names.index("torso_link")
        env._face_ids = (torso, names.index(CAMERA_BODY) if CAMERA_BODY in names else torso)
    torso_id, cam_id = env._face_ids
    q = robot.data.body_quat_w[:, torso_id]
    fwd = torch.stack([1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2), 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])], -1)
    to_apple = env.scene[object_cfg.name].data.root_pos_w[:, :2] - robot.data.body_pos_w[:, cam_id, :2]
    ang = torch.atan2(to_apple[:, 1], to_apple[:, 0]) - torch.atan2(fwd[:, 1], fwd[:, 0])
    return (torch.remainder(ang + math.pi, 2 * math.pi) - math.pi).abs()


def _facing(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, std_rad: float) -> torch.Tensor:
    return torch.exp(-camera_bearing_error(env, object_cfg) ** 2 / (2.0 * std_rad ** 2))


def reward_face_apple(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, std_rad: float = 0.25) -> torch.Tensor:
    """[N] dense: the torso (head camera) faces the apple. Small weight: a
    reason to explore turning, not a reason to give up the reach."""
    return _facing(env, object_cfg, std_rad)


def reward_face_apple_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    heading_full_rad: float,
    heading_zero_rad: float,
    face_std_rad: float = 0.25,
    pos_tol: float = 0.04,
    tilt_tol_rad: float | None = None,
    standoff: float = 0.015,
    palm_normal_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_cfg: SceneEntityCfg | None = None,
    pressure_min: float = 0.5,
    pressure_max: float | None = 5.0,
) -> torch.Tensor:
    """[N] reward_reach_hold_heading x facing: facing pays only while the
    reach itself is good, so it is never worth missing the touch for."""
    hold = reward_reach_hold_heading(
        env, robot_cfg, object_cfg, heading_full_rad, heading_zero_rad, pos_tol=pos_tol,
        tilt_tol_rad=tilt_tol_rad, standoff=standoff, palm_normal_axis=palm_normal_axis,
        palm_local_offset=palm_local_offset, sensor_cfg=sensor_cfg, pressure_min=pressure_min,
        pressure_max=pressure_max)
    return hold * _facing(env, object_cfg, face_std_rad)


def reward_face_apple_success(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg,
                              std_rad: float = 0.25) -> torch.Tensor:
    """[N] reach_rewards.reward_reach_success x facing: extra on a success
    made while looking at the apple."""
    return rr.reward_reach_success(env) * _facing(env, object_cfg, std_rad)
