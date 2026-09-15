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
#  Tactile observations (grasp phase) — deliberately calibration-agnostic.
#
#  tactile_layout.SIM_FORCE_TO_RAW_GAIN is an admitted, unverified placeholder
#  (see that module's docstring), so these observations avoid depending on
#  absolute force magnitude wherever possible: tactile_slip (derivative) and
#  tactile_finger_balance (ratio) are both invariant to a uniform scale, and
#  training additionally randomizes an INDEPENDENT per-taxel gain every
#  episode (domain_randomization.randomize_tactile_gain) so the policy can't
#  lean on a shared scale factor either. tactile_force_level is the one
#  deliberate exception — needed so a command-conditioned grip-force setpoint
#  (see the grasp env cfg's env._target_grip_level) has something to track;
#  domain randomization still applies to it, so tracking is necessarily
#  approximate, matching how an operator tuning a real, uncalibrated hand
#  could only achieve approximate tracking too.
#
#  Reward functions must NOT reuse _sensed_tactile_forces (or these public
#  observation functions) directly — rewards use ground-truth force straight
#  from the ContactSensor (see envs/mdp/grasp_rewards.py's own
#  _per_finger_group_force), since privileged, uncalibrated info is fine for
#  a training-only reward signal but would defeat the point if fed to the
#  policy as an observation.
# ─────────────────────────────────────────────

def _sensed_tactile_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apple-filtered, per-taxel force magnitude, WITH the per-env per-taxel
    calibration gain applied (ground truth * env._tactile_gain).

    Uses force_matrix_w (filtered specifically to the Apple via
    ContactSensorCfg.filter_prim_paths_expr), not net_forces_w — same
    reasoning as reach_rewards.penalty_apple_contact_force: self-collision or
    tree contact must not leak into a tactile reading.

    Shape: [num_envs, 9] (tactile_layout.TAXELS order).
    """
    import tactile_layout

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if not hasattr(env, "_taxel_body_ids"):
        env._taxel_body_ids = [
            sensor.body_names.index(t.link_name) for t in tactile_layout.TAXELS
        ]

    forces = sensor.data.force_matrix_w          # [N, B, 1, 3] -- filtered to Apple
    force_norm = torch.norm(forces, dim=-1).squeeze(-1)   # [N, B]
    ground_truth = force_norm[:, env._taxel_body_ids]      # [N, 9]

    if not hasattr(env, "_tactile_gain"):
        env._tactile_gain = torch.ones(env.num_envs, ground_truth.shape[-1], device=env.device)

    return ground_truth * env._tactile_gain


def _taxel_finger_groups(env: ManagerBasedRLEnv) -> dict[str, list[int]]:
    """Lazily cache, per env, the taxel indices belonging to each finger
    group (thumb/index/middle) — shared by the balance/level observations
    below and by grasp_rewards.py's ground-truth equivalents."""
    import tactile_layout

    if not hasattr(env, "_taxel_finger_idx"):
        fingers = ("thumb", "index", "middle")
        env._taxel_finger_idx = {
            f: [i for i, t in enumerate(tactile_layout.TAXELS) if t.finger == f]
            for f in fingers
        }
    return env._taxel_finger_idx


def tactile_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-taxel temporal slip signal: (force_now - force_prev) / step_dt,
    calibration-gained. The real physical signature of slip (a sudden force
    change despite a steady grip command) — a derivative, so a UNIFORM
    calibration gain cancels out of it entirely; only the independent
    per-taxel gain randomization actually stresses this.

    Shape: [num_envs, 9]
    """
    sensed = _sensed_tactile_forces(env, sensor_cfg)

    if not hasattr(env, "_prev_tactile_force") or env._prev_tactile_force.shape != sensed.shape:
        env._prev_tactile_force = sensed.clone()

    slip = (sensed - env._prev_tactile_force) / env.step_dt
    # Episode boundary: don't score a spurious spike against the previous
    # episode's stale reading (same guard reach_rewards.reward_reach_progress
    # and terminations_and_events.reach_success already use).
    slip = torch.where(
        (env.episode_length_buf <= 1).unsqueeze(-1), torch.zeros_like(slip), slip
    )

    env._prev_tactile_force = sensed.clone()
    return slip


def tactile_finger_balance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-finger-group (thumb/index/middle) mean force, normalized by
    deviation from the cross-finger mean — deliberately NOT absolute
    magnitude, so a uniform miscalibration cancels and only genuine
    imbalance shows up.

    Shape: [num_envs, 3]
    """
    sensed = _sensed_tactile_forces(env, sensor_cfg)
    idx = _taxel_finger_groups(env)
    per_finger = torch.stack(
        [sensed[:, idx[f]].mean(dim=-1) for f in ("thumb", "index", "middle")], dim=-1
    )  # [N, 3]
    mean_f = per_finger.mean(dim=-1, keepdim=True)
    return (per_finger - mean_f) / (mean_f.abs() + 1e-3)


def tactile_force_level(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-finger-group mean of the calibration-gained absolute force level.

    The one deliberate exception to "no raw magnitude" in this module: a
    command-conditioned grip-force setpoint (env._target_grip_level) needs
    SOME absolute-level signal to know when it has reached a commanded
    target — slip/balance alone can't provide that. Scoped narrowly to a
    per-finger-group mean (not all 9 raw taxels) rather than adding a wider
    raw-magnitude observation than this specific need requires.

    Shape: [num_envs, 3]
    """
    sensed = _sensed_tactile_forces(env, sensor_cfg)
    idx = _taxel_finger_groups(env)
    return torch.stack(
        [sensed[:, idx[f]].mean(dim=-1) for f in ("thumb", "index", "middle")], dim=-1
    )  # [N, 3]


def target_grip_level(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The commanded grip-force setpoint itself, exposed directly as an
    observation input — a COMMAND, not a measurement, so no calibration gain
    applies (same as a commanded velocity input to a locomotion policy).

    Shape: [num_envs, 1]
    """
    if not hasattr(env, "_target_grip_level"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return env._target_grip_level.unsqueeze(-1)


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
