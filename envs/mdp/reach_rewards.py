# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach-phase reward terms.

Save as:  envs/mdp/reach_rewards.py

Self-contained on purpose: nothing here imports from or modifies rewards.py,
so your existing grasp/lift/place terms stay untouched and you can switch
phases by swapping which RewardsCfg you use.

Design notes (ported from envs/g1_arm_policies/g1_arm_env.py):

  reward_reach_progress      <- delta_con      (improvement shaping)
  penalty_reach_residual     <- beta_residual  (linear residual penalty)
  reward_reach_gaussian      <- constellation  (exp bonus, use at 2 scales)
  reward_reach_hold          <- R_hold_step    (reason to STOP moving)
  penalty_near_goal_velocity <- lambda_brake   (deceleration near target)
  penalty_action_jerk        <- lambda_jerk    (in action units, well-scaled)
  penalty_joint_jerk         <- lambda_jerk    (true rad/s^3, normalised)
  penalty_joint_deviation_weighted <- lambda_excursion + excursion_weights
  penalty_joint_range_box    <- (new) soft preferred-window posture penalty

Every function returns [num_envs]. Sign convention: functions named
reward_* return a positive quantity and take a POSITIVE weight; functions
named penalty_* return a positive quantity and take a NEGATIVE weight.

IMPORTANT: any function taking a per-joint `weights` list must be given a
SceneEntityCfg with preserve_order=True, otherwise SceneEntityCfg resolves
joint_ids into articulation order and your weights land on the wrong joints.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═════════════════════════════════════════════════════════════════════════
#  Shared helper
# ═════════════════════════════════════════════════════════════════════════

def _palm_target_dist(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float,
) -> torch.Tensor:
    """Distance from the palm body to the grasp point.

    The grasp point is `offset_z` metres BELOW the apple centre, i.e. where
    the palm sits when cupping the apple from underneath.

    Returns: [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    apple: RigidObject = env.scene[object_cfg.name]

    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    target = apple.data.root_pos_w.clone()
    target[:, 2] -= offset_z
    return torch.norm(target - palm_pos, dim=-1)


def _palm_normal_angle(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float],
) -> torch.Tensor:
    """Angle (rad) between the palm normal and world +Z (up).

    `palm_normal_axis` is the palm normal expressed in the palm link's OWN
    local frame. Run the probe in play_debug_patch.py to find it for your USD.

    Returns: [num_envs], 0 = palm facing straight up, pi = facing straight down.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]   # [N, 4] wxyz
    rot = matrix_from_quat(quat)                                  # [N, 3, 3]

    n_local = torch.tensor(palm_normal_axis, device=env.device, dtype=quat.dtype)
    n_local = n_local / (n_local.norm() + 1e-9)

    n_world = torch.matmul(rot, n_local)                          # [N, 3]
    cos_up = n_world[:, 2].clamp(-1.0, 1.0)                       # dot with [0,0,1]
    return torch.acos(cos_up)


# ═════════════════════════════════════════════════════════════════════════
#  Position: three scales so there is gradient at every distance
# ═════════════════════════════════════════════════════════════════════════

def reward_reach_progress(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.08,
) -> torch.Tensor:
    """Potential-based shaping: reward for REDUCING palm->target distance.

    r_t = d_{t-1} - d_t   (metres closed this step)

    Unlike a Gaussian, the gradient is identical at 0.5 m and at 0.02 m, so
    the policy gets a usable learning signal from the very first iteration.
    Being a potential difference it telescopes over an episode to d_0 - d_T,
    so it cannot be farmed by oscillating toward and away from the target.

    This is the single term that fixes "position is never learned".
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)

    if not hasattr(env, "_prev_palm_dist") or env._prev_palm_dist.shape != d.shape:
        env._prev_palm_dist = d.clone()

    r = env._prev_palm_dist - d
    # First step after a reset: prev is stale (belongs to the previous
    # episode), so zero the term rather than paying out a teleport.
    r = torch.where(env.episode_length_buf <= 1, torch.zeros_like(r), r)

    env._prev_palm_dist = d.clone()
    return r


def penalty_reach_residual(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.08,
) -> torch.Tensor:
    """Linear distance penalty. Use a NEGATIVE weight.

    Guarantees a non-zero pull even if progress shaping stalls, and makes
    "hover far away doing nothing" strictly worse than approaching.
    """
    return _palm_target_dist(env, robot_cfg, object_cfg, offset_z)


def reward_reach_gaussian(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.08,
    std: float = 0.05,
) -> torch.Tensor:
    """exp(-d^2 / 2 std^2), in [0, 1].

    Register this TWICE: once coarse (std ~0.25) for a broad basin, once
    fine (std ~0.04) for a sharp peak at the goal. A single std=0.05 term
    is numerically zero beyond ~0.15 m -- exp(-18) = 1.5e-8 at d = 0.30 m --
    which is why the original palm_position term never trained anything.
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)
    return torch.exp(-(d ** 2) / (2.0 * std ** 2))


# ═════════════════════════════════════════════════════════════════════════
#  Orientation: palm normal -> world up, yaw UNCONSTRAINED
# ═════════════════════════════════════════════════════════════════════════

def reward_palm_normal_up(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg | None = None,
    palm_normal_axis: Sequence[float] = (0.0, 0.0, 1.0),
    std_rad: float = 0.5,
    gate_dist: float = 0.0,
    offset_z: float = 0.08,
) -> torch.Tensor:
    """Reward the palm normal pointing at the sky. Fingers free to yaw.

    This constrains ONE rotational DOF instead of three. The previous
    full-quaternion target (roll 90 / pitch 0 / yaw 0) pinned the wrist yaw
    too, removing redundancy the arm needs to satisfy the position target --
    the two rewards were fighting each other.

    Args:
        palm_normal_axis: palm normal in the palm link's LOCAL frame.
        std_rad: Gaussian width on the normal-vs-up angle. 0.5 rad ~ 29 deg.
        gate_dist: if > 0, scale the reward linearly to 0 as the palm gets
            further than gate_dist from the grasp point. Without this, the
            orientation reward is collectable from ANYWHERE in the workspace
            and the policy will bank it instead of reaching.
        offset_z: only used when gating.
    """
    theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis)
    r = torch.exp(-(theta ** 2) / (2.0 * std_rad ** 2))

    if gate_dist > 0.0:
        if object_cfg is None:
            raise ValueError("gate_dist > 0 requires object_cfg")
        d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)
        r = r * (1.0 - d / gate_dist).clamp(0.0, 1.0)

    return r


def reward_reach_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_tol: float = 0.04,
    ori_tol_rad: float = 0.30,
    offset_z: float = 0.08,
    palm_normal_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Per-step bonus (0 or 1) for sitting inside the success neighbourhood.

    g1_arm_env's R_hold_step = 14.0. Without a term like this the policy has
    no incentive to settle -- every reward it can see is already maxed while
    buzzing, so it buzzes. Pair with the brake penalty below.
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)
    theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis)
    return ((d < pos_tol) & (theta < ori_tol_rad)).float()


def reward_reach_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Large one-time bonus on the step terminations_and_events.reach_success
    fires (palm held in the success neighbourhood for hold_time). g1_arm_env's
    R_success = 700, by far its largest reward term.

    Does NOT recompute the neighbourhood/hold check itself — reads
    env._reach_success, cached by reach_success. IsaacLab computes
    terminations before rewards each step (see ManagerBasedRLEnv.step), so
    that cache is always fresh here. Wire reach_success as a TerminationTermCfg
    with the SAME pos_tol/ori_tol_rad/offset_z/palm_normal_axis/hold_time as
    this reward's weight is meant to pay out on, or the two will disagree.
    """
    if not hasattr(env, "_reach_success"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._reach_success.float()


# ═════════════════════════════════════════════════════════════════════════
#  Smoothness / anti-vibration
# ═════════════════════════════════════════════════════════════════════════

def penalty_action_jerk(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Second difference of the command = jerk, in action units.

    jerk_t = a_t - 2 a_{t-1} + a_{t-2}

    g1_arm_env's lambda_jerk, but expressed on the command instead of on q.
    True joint jerk is O(1e3) rad/s^3 and squares to O(1e6), which needs
    absurd weights; in action units this is O(1) and stays tunable.

    Use with mdp.action_rate_l2 (first difference), not instead of it:
    rate suppresses fast motion, jerk suppresses direction reversals --
    the reversals are what you actually see as vibration.
    """
    curr = env.action_manager.action
    prev = env.action_manager.prev_action

    if not hasattr(env, "_prev_prev_action") or env._prev_prev_action.shape != curr.shape:
        env._prev_prev_action = torch.zeros_like(curr)

    jerk = curr - 2.0 * prev + env._prev_prev_action
    out = (jerk ** 2).sum(dim=-1)
    out = torch.where(env.episode_length_buf <= 2, torch.zeros_like(out), out)

    env._prev_prev_action = prev.clone()
    return out


def penalty_joint_jerk(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    weights: Sequence[float] | None = None,
    ref: float = 1000.0,
) -> torch.Tensor:
    """True joint jerk |d(ddq)/dt|^2, normalised by `ref` rad/s^3.

    Optional -- penalty_action_jerk is usually enough and better conditioned.
    Enable this if you need to report a physical jerk metric in the paper, or
    if you want per-joint weighting like config_right.yaml's
    jerk_weights: [1, 1, 5, 5, 8, 12, 15] (heavier on distal joints).

    Requires preserve_order=True on asset_cfg when passing weights.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]

    if not hasattr(env, "_prev_joint_acc") or env._prev_joint_acc.shape != acc.shape:
        env._prev_joint_acc = torch.zeros_like(acc)

    jerk = (acc - env._prev_joint_acc) / env.step_dt / ref

    if weights is not None:
        w = torch.tensor(weights, device=env.device, dtype=acc.dtype)
        out = (w * jerk ** 2).sum(dim=-1)
    else:
        out = (jerk ** 2).sum(dim=-1)

    out = torch.where(env.episode_length_buf <= 2, torch.zeros_like(out), out)
    env._prev_joint_acc = acc.clone()
    return out


def penalty_near_goal_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.08,
    brake_dist: float = 0.15,
) -> torch.Tensor:
    """g1_arm_env's lambda_brake: joint-velocity penalty that ramps in as the
    palm closes on the grasp point.

    penalty = ||dq||^2 * (1 - d/brake_dist)^2   for d < brake_dist, else 0

    This is the most effective single cure for arrive-and-buzz, because it
    costs nothing during the approach (where you WANT speed) and grows
    quadratically exactly where the vibration happens.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dq = asset.data.joint_vel[:, asset_cfg.joint_ids]

    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z)
    frac = (1.0 - d / brake_dist).clamp(0.0, 1.0)

    return (dq ** 2).sum(dim=-1) * frac ** 2


# ═════════════════════════════════════════════════════════════════════════
#  Posture
# ═════════════════════════════════════════════════════════════════════════

def penalty_joint_deviation_weighted(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted ||q - q_default||^2.

    g1_arm_env's lambda_excursion, with excursion_weights taken straight from
    config_right.yaml: [1.0, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0]
                        pitch roll  yaw  elbow  --- wrists free ---

    Sign-agnostic: it simply pulls shoulders and elbow toward the home pose
    whenever the task does not require leaving it. That penalises BOTH
    "elbow to the sky" and "shoulder roll too far inside" without needing to
    know which sign of which joint produces them.

    Requires preserve_order=True on asset_cfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dev = (asset.data.joint_pos[:, asset_cfg.joint_ids]
           - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    w = torch.tensor(weights, device=env.device, dtype=dev.dtype)
    return (w * dev ** 2).sum(dim=-1)


def penalty_joint_range_box(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    lower: Sequence[float],
    upper: Sequence[float],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Quadratic penalty ONLY outside a preferred per-joint window.

    penalty = sum_i w_i * max(0, lo_i - q_i, q_i - hi_i)^2

    Zero cost anywhere inside [lo, hi], so every IK solution in the good
    region stays free -- unlike a blanket joint_deviation_l1, which taxes
    every reach. Use this to forbid specific bad postures once you have
    identified them from the pose print in play_debug_patch.py.

    Requires preserve_order=True on asset_cfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    lo = torch.tensor(lower, device=env.device, dtype=q.dtype)
    hi = torch.tensor(upper, device=env.device, dtype=q.dtype)

    excess = (lo - q).clamp(min=0.0) + (q - hi).clamp(min=0.0)
    if weights is not None:
        excess = excess * torch.tensor(weights, device=env.device, dtype=q.dtype)

    return (excess ** 2).sum(dim=-1)
