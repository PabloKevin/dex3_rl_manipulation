# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp-phase reward terms.

Self-contained on purpose, mirroring reach_rewards.py's own convention: this
module doesn't modify reach_rewards.py/rewards.py, so the reach task's
config is unaffected by anything here.

Sign convention (same as reach_rewards.py): reward_* returns a positive
quantity and takes a POSITIVE weight; penalty_* returns a positive quantity
and takes a NEGATIVE weight.

Ground-truth vs. observation split (deliberate, throughout this module):
these are REWARD functions, so they read force straight from the
ContactSensor (via _per_finger_group_force / _finger_contact_state) with NO
calibration gain applied — rewards are training-time-only and never
deployed, so calibration-robustness doesn't apply to them. Only the tactile
OBSERVATIONS in observations.py apply env._tactile_gain. Don't cross the two.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import math
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ═════════════════════════════════════════════════════════════════════════
#  Shared ground-truth helpers
# ═════════════════════════════════════════════════════════════════════════

def _per_finger_group_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """[N, 3] ground-truth mean force per finger group (thumb/index/middle),
    apple-filtered (force_matrix_w, not net_forces_w — same reasoning as
    reach_rewards.penalty_apple_contact_force).

    Shared by reward_grasp_balance / reward_grasp_setpoint_tracking /
    terminations_and_events.fruit_damaged, so they all agree on the same
    underlying quantity — the ground-truth counterpart of
    observations.tactile_force_level (that one is calibration-gained, this
    one is not, by design).
    """
    import tactile_layout

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if not hasattr(env, "_taxel_body_ids"):
        env._taxel_body_ids = [
            sensor.body_names.index(t.link_name) for t in tactile_layout.TAXELS
        ]
    if not hasattr(env, "_taxel_finger_idx"):
        fingers = ("thumb", "index", "middle")
        env._taxel_finger_idx = {
            f: [i for i, t in enumerate(tactile_layout.TAXELS) if t.finger == f]
            for f in fingers
        }

    forces = sensor.data.force_matrix_w                    # [N, B, 1, 3]
    force_norm = torch.norm(forces, dim=-1).squeeze(-1)     # [N, B]
    ground_truth = force_norm[:, env._taxel_body_ids]        # [N, 9]

    idx = env._taxel_finger_idx
    return torch.stack(
        [ground_truth[:, idx[f]].mean(dim=-1) for f in ("thumb", "index", "middle")], dim=-1
    )  # [N, 3]


# ═════════════════════════════════════════════════════════════════════════
#  Force closure (Lynch & Park, 2017, Ch. 12) — cheap geometric proxy
# ═════════════════════════════════════════════════════════════════════════

# Distal-most link per finger group, used as an approximate contact-point
# LOCATION for the checks below. ContactSensor gives per-link FORCE, not
# exact contact position, so this is a documented approximation, not
# measured contact geometry.
_FINGER_TIP_LINKS = {
    "thumb": "right_hand_thumb_2_link",
    "index": "right_hand_index_1_link",
    "middle": "right_hand_middle_1_link",
}


def _finger_contact_state(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    contact_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-finger (thumb/index/middle) approximate contact point (world),
    force vector (apple-filtered, ground truth), and in-contact mask.

    Returns: (points [N,3,3], forces [N,3,3], in_contact [N,3])
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if not hasattr(env, "_finger_tip_body_ids"):
        env._finger_tip_body_ids = {
            f: sensor.body_names.index(link) for f, link in _FINGER_TIP_LINKS.items()
        }
    if not hasattr(env, "_finger_tip_robot_ids"):
        env._finger_tip_robot_ids = {
            f: robot.data.body_names.index(link) for f, link in _FINGER_TIP_LINKS.items()
        }

    fingers = ("thumb", "index", "middle")
    forces_raw = sensor.data.force_matrix_w  # [N, B, 1, 3]

    points = torch.stack(
        [robot.data.body_pos_w[:, env._finger_tip_robot_ids[f], :] for f in fingers], dim=1
    )  # [N, 3, 3]
    forces = torch.stack(
        [forces_raw[:, env._finger_tip_body_ids[f], 0, :] for f in fingers], dim=1
    )  # [N, 3, 3]
    in_contact = torch.norm(forces, dim=-1) > contact_threshold  # [N, 3]
    return points, forces, in_contact


def _force_closure_ok(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    mu: float = 0.5,
    contact_threshold: float = 0.05,
    collinearity_eps: float = 1e-5,
) -> torch.Tensor:
    """Cheap geometric proxy for 3-finger force closure: >=3 non-collinear
    contacts, each contact force lying within its friction cone of the
    (spherical-apple) surface normal at that point. A full wrench-cone/LP
    feasibility test is more rigor than a per-step reward/termination needs.

    Approximations documented, not hidden:
    - contact "points" are the finger-tip LINK positions (_FINGER_TIP_LINKS),
      not true measured contact locations (ContactSensor gives force, not
      contact position).
    - mu is an ASSUMED constant (0.5, a typical skin-on-fruit-skin order-of-
      magnitude estimate) — no PhysX material is explicitly configured on
      the apple in this scene to read a real coefficient from.
    - the apple is a sphere (APPLE_RADIUS), so the surface normal at any
      point is exactly the radial direction from the apple centre — exact
      for this asset, not a general mesh-normal approximation.

    Shape: [num_envs] bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    points, forces, in_contact = _finger_contact_state(env, sensor_cfg, robot_cfg, contact_threshold)

    enough_contact = in_contact.sum(dim=-1) >= 3

    apple_center = apple.data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
    radial = points - apple_center                      # [N, 3, 3]
    normals = radial / (torch.norm(radial, dim=-1, keepdim=True) + 1e-6)

    # Non-collinearity: cross product of the two vectors spanning the 3
    # contact points, thresholded away from zero.
    v1 = points[:, 1, :] - points[:, 0, :]
    v2 = points[:, 2, :] - points[:, 0, :]
    cross = torch.cross(v1, v2, dim=-1)
    non_collinear = torch.norm(cross, dim=-1) > collinearity_eps

    # Friction-cone alignment per contact: angle between the measured force
    # direction and the INWARD surface normal (-radial) must be <= atan(mu)
    # (the standard Coulomb friction-cone half-angle). Only meaningful for
    # contacts actually engaged; a non-contacting finger can't be part of a
    # closed grasp, so it doesn't veto on angle alone but fails the overall
    # "3 engaged and aligned" requirement via enough_contact instead.
    force_dirs = forces / (torch.norm(forces, dim=-1, keepdim=True) + 1e-6)
    cos_angle = (force_dirs * (-normals)).sum(dim=-1).clamp(-1.0, 1.0)  # [N, 3]
    angle = torch.acos(cos_angle)
    half_angle = math.atan(mu)
    engaged_aligned = torch.where(in_contact, angle <= half_angle, torch.ones_like(angle, dtype=torch.bool))
    all_engaged_aligned = engaged_aligned.all(dim=-1)

    return enough_contact & non_collinear & all_engaged_aligned


# ═════════════════════════════════════════════════════════════════════════
#  Rewards
# ═════════════════════════════════════════════════════════════════════════

def reward_grasp_balance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    min_force: float = 0.05,
) -> torch.Tensor:
    """Cross-finger force balance: exp(-CV) across the 3 finger groups,
    scaled by the FRACTION of fingers actually in contact (0 with none, the
    full balance value once all 3 are) — not an all-or-nothing cliff at
    exactly 3. A hard gate here gives literally zero gradient toward "get a
    second/third finger on" for a policy that's only found 1-2 so far,
    which is exactly the kind of unshaped reward that stalled the reach
    phase's own original reward set (see apple_grasp_env_cfg.py's
    RewardsCfg docstring) until it was replaced with gradient-at-every-
    distance shaping. Orthogonal to reward_grasp_setpoint_tracking below
    (evenness across fingers, not absolute level).
    """
    per_finger = _per_finger_group_force(env, sensor_cfg)  # [N, 3]
    in_contact = per_finger > min_force
    n_contact = in_contact.sum(dim=-1).float()
    mean_f = per_finger.mean(dim=-1)
    cv = per_finger.std(dim=-1) / (mean_f + 1e-6)
    balance = torch.exp(-cv)
    return balance * (n_contact / 3.0)


def reward_finger_approach(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    std: float = 0.05,
) -> torch.Tensor:
    """Dense shaping: exp(-mean_fingertip_to_apple_dist^2 / 2std^2) across
    the 3 finger-tip links (_FINGER_TIP_LINKS) — gradient toward closing the
    gap to the apple even before ANY contact registers. Without this,
    grasp_balance/grasp_hold/grasp_setpoint_tracking are all exactly zero
    (or, for setpoint_tracking, actively favor near-zero force on low-target
    episodes — see that function's own note) until contact happens at all,
    so a policy that hasn't discovered contact yet has NOTHING pulling it
    toward the apple. Same "gradient at every distance" reasoning as
    reach_rewards.reward_reach_gaussian — this is the grasp-phase
    equivalent of that term.
    """
    apple: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_finger_tip_robot_ids"):
        env._finger_tip_robot_ids = {
            f: robot.data.body_names.index(link) for f, link in _FINGER_TIP_LINKS.items()
        }
    fingers = ("thumb", "index", "middle")
    points = torch.stack(
        [robot.data.body_pos_w[:, env._finger_tip_robot_ids[f], :] for f in fingers], dim=1
    )  # [N, 3, 3]
    dist = torch.norm(points - apple.data.root_pos_w.unsqueeze(1), dim=-1)  # [N, 3]
    mean_dist = dist.mean(dim=-1)
    return torch.exp(-(mean_dist ** 2) / (2.0 * std ** 2))


def reward_grasp_setpoint_tracking(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    std: float = 3.0,
    min_force: float = 0.05,
) -> torch.Tensor:
    """Gaussian reward tracking the per-episode commanded env._target_grip_level
    (domain theory: a command-conditioned grip-force setpoint, sampled every
    episode strictly between a friction-based minimum-to-not-slip estimate
    and the randomized fruit-damage-force threshold — see
    domain_randomization.randomize_fruit_damage_force). Ground-truth force
    (a reward, not an observation) compared against the per-env command.

    Gated to zero with NO contact at all (n_contact==0): the commanded
    target can be as low as ~1 N (the friction-based minimum), and on those
    low-target episodes doing literally nothing (force=0) already scores
    close to the Gaussian's maximum — an unintended incentive to not grip at
    all that this gate removes, without needing the same full "all 3
    fingers" requirement reward_grasp_balance uses (some contact evidence
    is enough here; evenness is that function's job, not this one's).

    This supersedes a separate fixed "not crushing" cap: tracking a target
    that's sampled strictly under the damage threshold already keeps force
    below it as a side effect, and reward_grasp_balance handles evenness
    separately. fruit_damaged (terminations_and_events.py) remains as a hard
    backstop in case tracking overshoots.
    """
    per_finger = _per_finger_group_force(env, sensor_cfg)  # [N, 3]
    level = per_finger.mean(dim=-1)
    has_contact = (per_finger > min_force).any(dim=-1).float()
    if not hasattr(env, "_target_grip_level"):
        return torch.zeros(env.num_envs, device=env.device)
    err = level - env._target_grip_level
    return torch.exp(-(err ** 2) / (2.0 * std ** 2)) * has_contact


def reward_grasp_hold(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    mu: float = 0.5,
    contact_threshold: float = 0.05,
) -> torch.Tensor:
    """Per-step bonus (0 or 1) for being in force closure right now — reads
    the SAME cache terminations_and_events.grasp_success computes
    (env._force_closure_ok) when available, so the reward and the
    termination never disagree about the same step; falls back to computing
    it directly if the termination hasn't run yet this step (defensive only
    — TerminationsCfg is always computed before RewardsCfg in
    ManagerBasedRLEnv.step, same guarantee reach_success/reward_reach_success
    already rely on).
    """
    if hasattr(env, "_force_closure_ok"):
        return env._force_closure_ok.float()
    return _force_closure_ok(
        env, sensor_cfg, robot_cfg, object_cfg, mu=mu, contact_threshold=contact_threshold
    ).float()


def reward_grasp_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Large one-time bonus on the step terminations_and_events.grasp_success
    fires. Does NOT recompute the check itself — reads env._grasp_success,
    cached by that termination (mirrors reach_rewards.reward_reach_success's
    pattern exactly, including the "terminations before rewards" ordering
    guarantee it relies on).
    """
    if not hasattr(env, "_grasp_success"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._grasp_success.float()


def penalty_joint_deviation_from_handoff(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted ||q - q_handoff||^2, where q_handoff is the ARM pose the
    reach-autoplay hand-off left it at (env._grasp_handoff_arm_pos) — NOT the
    articulation's default/parked pose, unlike reach_rewards.
    penalty_joint_deviation_weighted. The reach-converged pose near the apple
    is the right reference here.

    Complementary to penalty_action_rate_weighted: that one discourages FAST
    proximal motion between consecutive steps, this one discourages
    DRIFTING far from the hand-off pose over the course of the episode,
    whichever direction. Requires preserve_order=True on asset_cfg (same as
    any per-joint-weight reward in this codebase). Zero before the first
    hand-off has ever run (defensive only).
    """
    if not hasattr(env, "_grasp_handoff_arm_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - env._grasp_handoff_arm_pos
    w = torch.tensor(weights, device=env.device, dtype=dev.dtype)
    return (w * dev ** 2).sum(dim=-1)


def penalty_action_rate_weighted(
    env: "ManagerBasedRLEnv",
    weights: Sequence[float],
) -> torch.Tensor:
    """Per-joint-weighted first-difference penalty over the FULL concatenated
    action vector (arm 7 + hand 7 = 14-dim). Heavy weight on the 4 proximal
    joints (shoulder x3, elbow), light weight on the 10 distal joints
    (wrist x3, hand x7) — the primary mechanism implementing "mostly
    fingers/wrist, arm only as a light correction for an imperfect reach
    hand-off" (the user's explicit action-space requirement for this task).

    `weights` order MUST match ActionsCfg's action-term concatenation order
    (right_arm then right_hand, i.e. env.action_manager.active_terms) — this
    is NOT auto-verified here; confirm it interactively (see the grasp env
    cfg's own comment on RIGHT_HAND_JOINTS ordering) before trusting a
    hardcoded weights list.
    """
    curr = env.action_manager.action
    prev = env.action_manager.prev_action
    if prev is None:
        return torch.zeros(env.num_envs, device=env.device)
    w = torch.tensor(weights, device=env.device, dtype=curr.dtype)
    delta = curr - prev
    return (w * delta ** 2).sum(dim=-1)
