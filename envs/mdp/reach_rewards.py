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

def _palm_ref_pos(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """World position of the actual grasp reference point on the hand.

    right_hand_palm_link's own origin sits back near the wrist, not out at
    the finger pads / contact-sensor area — probed empirically
    (probe_palm_offset.py): right_hand_middle_1_link and
    right_hand_index_1_link both sit at local-frame offset
    ~(+0.123, ~0, +-0.028) from the palm link, i.e. local +X is "toward the
    fingers" and the finger pads themselves are a bit further out than that
    first knuckle. `palm_local_offset` (in the palm link's OWN local frame,
    same convention as palm_normal_axis) lets reward/termination/observation
    code all agree on where "the hand" actually is instead of the wrist.

    Returns: [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]

    offset = torch.tensor(palm_local_offset, device=env.device, dtype=palm_pos.dtype)
    if torch.all(offset == 0.0):
        return palm_pos

    quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]
    rot = matrix_from_quat(quat)                      # [N, 3, 3], local -> world
    offset_world = torch.matmul(rot, offset)           # [N, 3]
    return palm_pos + offset_world


def _palm_target_dist(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Distance from the palm's grasp reference point to the grasp target.

    The grasp point is `offset_z` metres BELOW the apple centre, i.e. where
    the hand sits when cupping the apple from underneath.

    Returns: [num_envs]
    """
    apple: RigidObject = env.scene[object_cfg.name]

    palm_pos = _palm_ref_pos(env, robot_cfg, palm_local_offset)
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
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Potential-based shaping: reward for REDUCING palm->target distance.

    r_t = d_{t-1} - d_t   (metres closed this step)

    Unlike a Gaussian, the gradient is identical at 0.5 m and at 0.02 m, so
    the policy gets a usable learning signal from the very first iteration.
    Being a potential difference it telescopes over an episode to d_0 - d_T,
    so it cannot be farmed by oscillating toward and away from the target.

    This is the single term that fixes "position is never learned".
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)

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
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Linear distance penalty. Use a NEGATIVE weight.

    Guarantees a non-zero pull even if progress shaping stalls, and makes
    "hover far away doing nothing" strictly worse than approaching.
    """
    return _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)


def reward_reach_gaussian(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float = 0.08,
    std: float = 0.05,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """exp(-d^2 / 2 std^2), in [0, 1].

    Register this TWICE: once coarse (std ~0.25) for a broad basin, once
    fine (std ~0.04) for a sharp peak at the goal. A single std=0.05 term
    is numerically zero beyond ~0.15 m -- exp(-18) = 1.5e-8 at d = 0.30 m --
    which is why the original palm_position term never trained anything.
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)
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
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
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
        d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)
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
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
    sensor_cfg: SceneEntityCfg | None = None,
    pressure_min: float = 0.5,
    pressure_max: float | None = 5.0,
) -> torch.Tensor:
    """Per-step bonus (0 or 1) for sitting inside the success neighbourhood.

    g1_arm_env's R_hold_step = 14.0. Without a term like this the policy has
    no incentive to settle -- every reward it can see is already maxed while
    buzzing, so it buzzes. Pair with the brake penalty below.

    sensor_cfg (optional, default None = old position/orientation-only
    behaviour, fully backward compatible): when given, ALSO requires palm
    contact force in (pressure_min, pressure_max] -- i.e. "arrived AND
    gently touching", not just "arrived". Kept in sync with terminations_
    and_events.reach_success's own optional touch-gate deliberately: if
    this per-step reward paid out for "nearby" alone while the terminal
    success bonus required actual contact, the policy could collect most of
    the reward (hold pays every step) without ever closing the last few cm
    to touch -- the same reward/gate mismatch that stalled the grasp
    phase's first training run, avoided here by construction.
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)
    theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis)
    in_neighbourhood = (d < pos_tol) & (theta < ori_tol_rad)

    if sensor_cfg is not None:
        from .observations import palm_contact_force
        force = palm_contact_force(env, sensor_cfg).squeeze(-1)
        touching = force > pressure_min
        if pressure_max is not None:
            touching = touching & (force <= pressure_max)
        in_neighbourhood = in_neighbourhood & touching

    return in_neighbourhood.float()


def reward_gentle_touch(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    good_min: float = 0.5,
    good_max: float = 5.0,
) -> torch.Tensor:
    """Flat, per-step bonus (0 or 1) for palm+thumb-base contact force
    landing in [good_min, good_max] N -- a direct, position-independent
    stepping stone toward touching gently, separate from reward_reach_hold's
    combined position+orientation+force gate. Without an isolated reward
    like this, the policy only ever sees the force-related signal bundled
    with the (much harder to simultaneously satisfy) full success
    neighbourhood, which is a much sparser target to stumble into --
    same "give it an easier, more isolated gradient" reasoning as the
    grasp phase's own reward_finger_approach.

    Below good_min: not really touching, no bonus. Above good_max: handled
    by penalty_excess_palm_force below, not by shrinking this term back to
    zero -- keeping this flat rather than peaked avoids fighting that
    penalty over exactly where the "good enough" boundary sits.
    """
    from .observations import palm_contact_force
    force = palm_contact_force(env, sensor_cfg).squeeze(-1)
    return ((force >= good_min) & (force <= good_max)).float()


def penalty_excess_palm_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    start: float = 5.0,
    cap: float = 20.0,
) -> torch.Tensor:
    """Quadratically-growing penalty for palm+thumb-base contact force above
    `start` N, normalized to reach exactly 1.0 (times whatever weight
    RewardsCfg gives this) at `cap` N and CLAMPED there beyond `cap` --
    force above the cap is unambiguously bad and gets the max penalty, but
    doesn't keep growing without bound.

    The clamp matters for the same reason it did in the previous version of
    this function: an empirical trace of an early reach checkpoint showed
    the palm taking a 58 N impact. An UNCLAMPED quadratic penalty at that
    force would be enormous (tens of times larger than the entire
    reach_success bonus of 95) -- a reward outlier that extreme doesn't
    teach "be gentler", it teaches "never go anywhere near this region
    again", which is consistent with what was observed: the policy touched
    once, then retreated and never approached again for the rest of the
    rollout. Capping at `cap` keeps this a clear, dominant-but-bounded
    safety penalty instead.
    """
    from .observations import palm_contact_force
    force = palm_contact_force(env, sensor_cfg).squeeze(-1)
    span = cap - start
    excess = (force - start).clamp(min=0.0, max=span)
    return (excess / span) ** 2


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


def penalty_apple_contact_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    ref: float = 0.8,
) -> torch.Tensor:
    """Squared-norm penalty on contact force between the hand and the apple.

    Apple-filtered, not net_forces_w -- the latter is the TOTAL contact
    force on each hand body against ANYTHING (self-collision, the tree,
    accidental floor contact), which would give a false "pressing the apple"
    signal for contact that has nothing to do with the apple. Read through
    observations.apple_force_per_link, which goes to the per-link,
    Apple-filtered sensors (AppleGraspSceneCfg's contact_forces_* block), so
    this only ever penalises force actually applied to the apple. Do NOT go
    back to reading force_matrix_w off the multi-body `contact_forces`
    sensor -- that silently reports zeros, see that block's comment.

    Squared and normalised by `ref` newtons: a light touch (~ref or below)
    costs little, pressing hard costs much more than linearly. Not a binary
    contact/no-contact term -- a light graze near a successful reach isn't
    penalised much, only genuinely pressing into the apple is.

    Deliberately NOT exposed as an observation (see ObservationsCfg in
    apple_grasp_env_cfg.py) -- the policy doesn't need to measure the force
    to learn not to apply it, and reward functions have full access to sim
    state regardless of what's wired into the obs group, so there's no
    reason to spend an observation dimension on this.
    """
    from .observations import RIGHT_HAND_CONTACT_LINKS, apple_force_per_link
    force_norm = apple_force_per_link(env, sensor_cfg, RIGHT_HAND_CONTACT_LINKS)
    return ((force_norm / ref) ** 2).sum(dim=-1)


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
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
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

    d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)
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


# ═════════════════════════════════════════════════════════════════════════
#  Arm-vs-torso self-collision (task space)
# ═════════════════════════════════════════════════════════════════════════
#
# The robot spawns with enabled_self_collisions=False (see ArticulationCfg
# in apple_grasp_env_cfg.py), so PhysX will happily let the upper arm and
# forearm pass straight through the torso — nothing in the sim stops it and
# nothing in the reward noticed it either. The existing guard,
# RewardsCfg.posture_box, boxes right_shoulder_roll_joint at +0.35 rad,
# which is a JOINT-space proxy for "elbow swung in toward the chest". That
# catches the one failure mode it was tuned against, but it cannot catch the
# general case: with the apple close to the robot, shoulder_pitch/yaw and
# elbow flexion can put the upper arm inside the torso at shoulder_roll
# values the box considers perfectly legal.
#
# So this term works in TASK space instead, on the actual link poses.
#
# Geometry is NOT guessed — both the torso capsule and the arm sphere radii
# below are read off the project's own cuRobo collision model,
# envs/assets/cuRobo/g1_29dof_rev_1_0_spherized.yml, which is the same
# spherization the CasADi/cuRobo planners in scripts/ already plan against.
# Keeping the RL penalty and the planner on one collision model means a pose
# the policy learns is clear is also a pose the planner considers clear.
#
# torso_link's three spheres in that file sit exactly on the link's local Z
# axis (x = y = 0) at z = 0.0930 / 0.2149 / 0.3919 with radii 0.1045 /
# 0.1170 / 0.1014 — i.e. a vertical capsule of radius ~0.117 (the largest,
# taken as the conservative single radius) spanning that z range.

#: Torso collision capsule, in torso_link's OWN local frame. From
#: g1_29dof_rev_1_0_spherized.yml's torso_link spheres (see above).
TORSO_CAPSULE_P0 = (0.0, 0.0, 0.0930)
TORSO_CAPSULE_P1 = (0.0, 0.0, 0.3919)
TORSO_CAPSULE_RADIUS = 0.1170

#: Right-arm collision spheres as (link name, centre in that link's LOCAL
#: frame, radius), lifted verbatim from the same spherized model.
#:
#: Deliberately starts at right_shoulder_yaw_link (the upper arm) and does
#: NOT include right_shoulder_pitch/roll_link: those two sit at the shoulder
#: itself, roughly 0.10 m off the torso centreline, so they are inside or
#: touching the torso capsule in EVERY pose including the home pose. Adding
#: them would contribute a near-constant offset with no usable gradient
#: while making the term's magnitude impossible to read on a training curve.
#: The links kept here are exactly the ones reported as colliding: the upper
#: arm (shoulder_yaw) and the forearm (elbow), plus the wrist for free.
ARM_COLLISION_SPHERES = [
    ("right_shoulder_yaw_link", ( 0.0123,  0.0020, -0.0773), 0.0458),
    ("right_shoulder_yaw_link", (-0.0046,  0.0000, -0.0114), 0.0431),
    ("right_elbow_link",        ( 0.0843,  0.0000, -0.0101), 0.0415),
    ("right_elbow_link",        ( 0.0391,  0.0000, -0.0085), 0.0415),
    ("right_wrist_roll_link",   ( 0.0325,  0.0000,  0.0000), 0.0395),
]


def _point_segment_distance(
    p: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Distance from points `p` [N, S, 3] to the segment a->b ([N, 3] each).

    Returns: [N, S]
    """
    ab = (b - a).unsqueeze(1)                                   # [N, 1, 3]
    ap = p - a.unsqueeze(1)                                     # [N, S, 3]
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-12)              # [N, 1]
    t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)         # [N, S]
    closest = a.unsqueeze(1) + t.unsqueeze(-1) * ab             # [N, S, 3]
    return torch.norm(p - closest, dim=-1)                      # [N, S]


def penalty_torso_self_collision(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    margin: float = 0.02,
    ref: float = 0.05,
    capsule_radius: float = TORSO_CAPSULE_RADIUS,
) -> torch.Tensor:
    """Quadratic penalty on the right arm intruding into the torso capsule.

    For each arm collision sphere s (ARM_COLLISION_SPHERES):

        depth_s = max(0, capsule_radius + r_s + margin - dist(c_s, torso axis))
        penalty = sum_s (depth_s / ref)^2

    Zero cost whenever the whole arm is clear, so every non-colliding IK
    branch stays completely free — this only ever pushes the arm out of the
    torso, it never expresses a preference between two clear postures (that
    is home_excursion's job). Because it is a smooth function of link pose
    rather than a joint box, it gives a real gradient that grows as the arm
    goes deeper, which is what actually teaches "swing the elbow out" rather
    than "this joint value is forbidden".

    `margin` is clearance required ON TOP of the two collision radii, so the
    arm is pushed to a visibly clear posture rather than to exact tangency.
    `ref` normalises the depth so a `ref`-deep intrusion on one sphere costs
    exactly 1.0 before the weight — pick the weight against that scale.

    Both defaults are in metres. Raise `margin` if the arm still brushes the
    torso mesh (the capsule is an approximation of a non-convex mesh, and
    the shoulder end of the torso is where it fits worst).

    `robot_cfg` selects the articulation only; the bodies come from
    ARM_COLLISION_SPHERES / the torso capsule constants, so no body_names
    are needed on it and none are read.

    Returns: [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env, "_torso_collision_ids"):
        names = robot.data.body_names
        env._torso_collision_ids = {
            "torso": names.index("torso_link"),
            "arm": [names.index(link) for link, _, _ in ARM_COLLISION_SPHERES],
        }
        env._arm_sphere_offsets = torch.tensor(
            [c for _, c, _ in ARM_COLLISION_SPHERES],
            device=env.device, dtype=torch.float32,
        )                                                        # [S, 3]
        env._arm_sphere_radii = torch.tensor(
            [r for _, _, r in ARM_COLLISION_SPHERES],
            device=env.device, dtype=torch.float32,
        )                                                        # [S]

    ids = env._torso_collision_ids

    # Torso capsule endpoints, local -> world.
    torso_pos = robot.data.body_pos_w[:, ids["torso"], :]                  # [N, 3]
    torso_rot = matrix_from_quat(robot.data.body_quat_w[:, ids["torso"], :])  # [N, 3, 3]
    p0 = torch.tensor(TORSO_CAPSULE_P0, device=env.device, dtype=torso_pos.dtype)
    p1 = torch.tensor(TORSO_CAPSULE_P1, device=env.device, dtype=torso_pos.dtype)
    a = torso_pos + torch.matmul(torso_rot, p0)                            # [N, 3]
    b = torso_pos + torch.matmul(torso_rot, p1)                            # [N, 3]

    # Arm sphere centres, local -> world. Each sphere rides its own link, so
    # index the link poses per-sphere (a link with two spheres appears twice).
    link_pos = robot.data.body_pos_w[:, ids["arm"], :]                     # [N, S, 3]
    link_rot = matrix_from_quat(
        robot.data.body_quat_w[:, ids["arm"], :].reshape(-1, 4)
    ).reshape(link_pos.shape[0], len(ids["arm"]), 3, 3)                    # [N, S, 3, 3]
    offsets = env._arm_sphere_offsets.to(link_pos.dtype)                   # [S, 3]
    centres = link_pos + torch.matmul(
        link_rot, offsets.unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)                                                          # [N, S, 3]

    dist = _point_segment_distance(centres, a, b)                          # [N, S]
    clearance = capsule_radius + env._arm_sphere_radii.to(dist.dtype) + margin
    depth = (clearance.unsqueeze(0) - dist).clamp(min=0.0)                 # [N, S]

    return ((depth / ref) ** 2).sum(dim=-1)
