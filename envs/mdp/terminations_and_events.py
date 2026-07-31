# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Termination conditions and reset/randomisation events."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import reset_root_state_uniform  # built-in Isaac Lab

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Terminations
# ─────────────────────────────────────────────

def apple_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height: float = -0.1,
) -> torch.Tensor:
    """True if apple falls below min_height (dropped / out of reach).

    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    return apple.data.root_pos_w[:, 2] < min_height


def apple_lifted_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_height: float,
    hold_time: float = 1.0,
) -> torch.Tensor:
    """True when apple has been held above target_height for hold_time seconds.

    Uses env.episode_length_buf to measure elapsed time.
    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    above = apple.data.root_pos_w[:, 2] >= target_height  # [N] bool

    # How many steps is hold_time?
    hold_steps = int(hold_time / (env.cfg.sim.dt * env.cfg.decimation))

    # We need a persistent counter — store on env if not present
    if not hasattr(env, "_hold_counter"):
        env._hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._hold_counter[above] += 1
    env._hold_counter[~above] = 0

    return env._hold_counter >= hold_steps


def reach_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_tol: float = 0.04,
    ori_tol_rad: float = 0.30,
    offset_z: float = 0.08,
    palm_normal_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    hold_time: float = 0.5,
) -> torch.Tensor:
    """Terminate once the palm has held within [pos_tol, ori_tol_rad] of the
    grasp point for `hold_time` continuous seconds — the same neighbourhood
    check as reach_rewards.reward_reach_hold's per-step bonus, counted here
    toward a one-time terminal success instead. Mirrors g1_arm_env's
    hold-to-succeed / R_success design (config_right.yaml: hold.time=0.5,
    R_success=700).

    IsaacLab computes terminations BEFORE rewards each step (see
    ManagerBasedRLEnv.step: termination_manager.compute() then
    reward_manager.compute()), so this caches the result on env._reach_success
    for reach_rewards.reward_reach_success to read the same step it fires —
    that reward function must NOT recompute the check itself, or a policy
    could see the bonus without this termination also firing (or vice versa).

    Shape: [num_envs]  bool
    """
    from .reach_rewards import _palm_target_dist, _palm_normal_angle

    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)
    theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis)
    in_neighbourhood = (d < pos_tol) & (theta < ori_tol_rad)

    if not hasattr(env, "_reach_hold_counter") or env._reach_hold_counter.shape != d.shape:
        env._reach_hold_counter = torch.zeros_like(d)

    # Episode boundary: don't carry a stale count into a fresh episode.
    env._reach_hold_counter = torch.where(
        env.episode_length_buf <= 1, torch.zeros_like(env._reach_hold_counter), env._reach_hold_counter
    )
    env._reach_hold_counter = torch.where(
        in_neighbourhood, env._reach_hold_counter + 1, torch.zeros_like(env._reach_hold_counter)
    )

    hold_steps_required = max(1, round(hold_time / env.step_dt))
    success = env._reach_hold_counter >= hold_steps_required

    env._reach_success = success   # read by reach_rewards.reward_reach_success
    return success


def apple_placed_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float],
    tolerance: float = 0.06,
) -> torch.Tensor:
    """True when apple is within tolerance metres of the target (basket).

    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    goal = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    dist = torch.norm(apple.data.root_pos_w - goal, dim=-1)
    return dist < tolerance


# ─────────────────────────────────────────────
#  Events (reset / randomisation)
# ─────────────────────────────────────────────

def reset_apple_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    base_pos: tuple[float, float, float],
    pos_range_x: tuple[float, float] = (-0.05, 0.05),
    pos_range_y: tuple[float, float] = (-0.05, 0.05),
    pos_range_z: tuple[float, float] = (-0.02, 0.02),
) -> None:
    """Reset apple to base_pos + uniform noise within the given ranges.

    Called at the start of each episode for the specified env_ids.
    """
    apple: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)

    # Sample position noise
    dx = torch.empty(n, device=env.device).uniform_(*pos_range_x)
    dy = torch.empty(n, device=env.device).uniform_(*pos_range_y)
    dz = torch.empty(n, device=env.device).uniform_(*pos_range_z)

    # Build new root state [n, 13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    root_state = apple.data.default_root_state[env_ids].clone()  # [n, 13]

    # Override position (add env origins so position is in world frame)
    env_origins = env.scene.env_origins[env_ids]   # [n, 3]
    base = torch.tensor(base_pos, device=env.device).unsqueeze(0)  # [1, 3]
    noise = torch.stack([dx, dy, dz], dim=-1)       # [n, 3]

    root_state[:, :3] = env_origins + base + noise

    # Zero velocity
    root_state[:, 7:] = 0.0

    # Write back
    apple.write_root_state_to_sim(root_state, env_ids=env_ids)
