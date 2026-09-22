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
ContactSensor (via _per_finger_group_force / apple_force_vec_per_link) with NO
calibration gain applied — rewards are training-time-only and never
deployed, so calibration-robustness doesn't apply to them. Only the tactile
OBSERVATIONS in observations.py apply env._tactile_gain. Don't cross the two.

Reach/grasp phase gating: every reward here is multiplied by
_grasp_mask(env), i.e. exactly zero while the frozen reach policy is still
driving (see grasp_autoplay.py's module docstring) -- the trainee's actions
are ignored then, so it must not be credited or charged for anything.
Gating lives inside each function rather than in a generic wrapper on
purpose: Isaac Lab's Hydra round-trip re-resolves every term func from its
"module:name" string, which would silently swap a functools.wraps wrapper
back to the ungated original.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import math
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _grasp_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] float: 1.0 in the grasp phase, 0.0 while the reach policy drives."""
    from .grasp_autoplay import grasp_phase_active
    return grasp_phase_active(env).float()


# ═════════════════════════════════════════════════════════════════════════
#  Shared ground-truth helpers
# ═════════════════════════════════════════════════════════════════════════

def _per_finger_group_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """[N, 3] ground-truth mean force per finger group (thumb/index/middle),
    apple-filtered (via observations.apple_force_per_link, not net_forces_w —
    same reasoning as reach_rewards.penalty_apple_contact_force).

    Shared by reward_grasp_balance / reward_grasp_setpoint_tracking /
    terminations_and_events.fruit_damaged, so they all agree on the same
    underlying quantity — the ground-truth counterpart of
    observations.tactile_force_level (that one is calibration-gained, this
    one is not, by design).
    """
    import scripts.tactile_layout as tactile_layout
    from .observations import apple_force_per_link

    if not hasattr(env, "_taxel_finger_idx"):
        fingers = ("thumb", "index", "middle")
        env._taxel_finger_idx = {
            f: [i for i, t in enumerate(tactile_layout.TAXELS) if t.finger == f]
            for f in fingers
        }

    ground_truth = apple_force_per_link(
        env, sensor_cfg, [t.link_name for t in tactile_layout.TAXELS]
    )  # [N, 9]

    idx = env._taxel_finger_idx
    return torch.stack(
        [ground_truth[:, idx[f]].mean(dim=-1) for f in ("thumb", "index", "middle")], dim=-1
    )  # [N, 3]


# ═════════════════════════════════════════════════════════════════════════
#  Grasp closure — palm-inclusive enclosure proxy
# ═════════════════════════════════════════════════════════════════════════

# Distal-most link per finger, used by reward_finger_approach as the
# fingertip location.
_FINGER_TIP_LINKS = {
    "thumb": "right_hand_thumb_2_link",
    "index": "right_hand_index_1_link",
    "middle": "right_hand_middle_1_link",
}

# Contact groups for the closure check: the palm plus EVERY segment of each
# digit, not just the fingertips.
_CLOSURE_GROUPS = {
    "palm": ["right_hand_palm_link"],
    "thumb": ["right_hand_thumb_0_link", "right_hand_thumb_1_link", "right_hand_thumb_2_link"],
    "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
    "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"],
}


def _force_closure_ok(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
    max_mean_push: float = 0.5,
) -> torch.Tensor:
    """Is the hand ENCLOSING the apple right now: >= 3 contact groups
    (_CLOSURE_GROUPS) each pressing with > contact_threshold N, AND those
    pushes coming from opposing sides rather than all from one side.

    "Opposing sides" is measured without any contact POSITIONS: d_g is the
    unit push group g exerts ON the apple (the negated force on the hand
    link, see _palmar_dorsal_force in reach_rewards for the sign
    convention), and the check is ||mean_g d_g|| < max_mean_push. All
    groups pushing from the same side gives ~1.0; palm underneath plus a
    horizontal thumb-vs-fingers pinch gives ~0.33; a perfectly balanced
    enclosure gives 0.

    Replaces a fingertip-only friction-cone proxy (3 distal links in
    contact, each force within atan(mu) of the radial through the link
    ORIGIN). Driving every digit to every combination of its joint limits
    from the reach hand-off pose never once got all three fingertips onto
    the apple at the same time -- with the apple resting on the palm, the
    Dex3's reachable grasp on an 8 cm fruit is a power grasp (palm + finger
    segments + thumb), not a fingertip grasp -- so grasp_hold, grasp_success
    and the apple's release from its pin could never fire. The same sweep
    reaches this criterion for 100+ steps in several combinations. The
    link-origin contact points were also a poor stand-in for real contact
    locations (the palm link origin sits back near the wrist), which is why
    this version uses force directions only.

    Shape: [num_envs] bool
    """
    mag, n_contact, mean_push = _closure_state(env, sensor_cfg, contact_threshold)
    return (n_contact >= 3) & (mean_push < max_mean_push)


def _closure_state(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared by _force_closure_ok and reward_grasp_enclosure, so the hard
    criterion and the reward that shapes toward it can never disagree.

    Returns (per-group force magnitude [N, 4], number of groups in contact
    [N], ||mean unit push on the apple|| [N]).
    """
    from .observations import apple_force_vec_per_link

    force_on_hand = torch.stack(
        [apple_force_vec_per_link(env, sensor_cfg, links).sum(dim=1) for links in _CLOSURE_GROUPS.values()],
        dim=1,
    )  # [N, 4, 3]
    mag = torch.norm(force_on_hand, dim=-1)                    # [N, 4]
    in_contact = mag > contact_threshold
    n_contact = in_contact.sum(dim=-1)
    push_on_apple = -force_on_hand / (mag.unsqueeze(-1) + 1e-6)
    mean_push = (push_on_apple * in_contact.unsqueeze(-1)).sum(dim=1).norm(dim=-1) / n_contact.clamp(min=1)
    return mag, n_contact, mean_push


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
    return balance * (n_contact / 3.0) * _grasp_mask(env)


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
    return torch.exp(-(mean_dist ** 2) / (2.0 * std ** 2)) * _grasp_mask(env)


def reward_gentle_contact_groups(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    good_min: float = 0.5,
    good_max: float = 5.0,
    decay: float = 5.0,
) -> torch.Tensor:
    """Mean over the 4 contact groups (palm/thumb/index/middle, see
    _CLOSURE_GROUPS) of a trapezoid on that group's own contact force:
    ramps 0 -> 1 over [0, good_min], stays 1 across the gentle band
    [good_min, good_max], then decays back to 0 over `decay` N above it.

    The grasp-phase counterpart of reach_rewards.reward_gentle_touch, and
    added for the same reason it was there: it pays PER GROUP, so getting a
    second and a third part of the hand onto the apple is rewarded
    incrementally, instead of everything hinging on the all-or-nothing
    closure test. Counting the palm as a group matters -- the apple rests
    on it in the hand-off pose, so it is the one contact the policy starts
    with and should not be taught to give up.

    The decay above good_max is the anti-crush half: past the gentle band
    squeezing harder pays LESS, a dense gradient the hard fruit_damaged
    cutoff (and penalty_excess_hand_force, which only ramps near this
    episode's damage threshold) can't provide. Motivated by the run where
    the fingers learned to press hard enough to bruise the apple in 72% of
    episodes while never once enclosing it.

    Shape: [num_envs], in [0, 1]
    """
    mag, _, _ = _closure_state(env, sensor_cfg, contact_threshold=good_min)
    ramp_up = (mag / good_min).clamp(max=1.0)
    ramp_down = (1.0 - (mag - good_max) / decay).clamp(min=0.0, max=1.0)
    return (ramp_up * ramp_down).mean(dim=-1) * _grasp_mask(env)


def reward_grasp_enclosure(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
) -> torch.Tensor:
    """Continuous progress toward the closure criterion itself: how many
    contact groups are engaged (capped at the 3 closure needs) TIMES how
    much their pushes oppose each other (1 - ||mean unit push||).

    _force_closure_ok is a step function -- 3 groups AND opposing -- so on
    its own it gives a policy that has 2 groups on, or 3 groups all pushing
    the same way, exactly zero gradient toward fixing either. This is that
    gradient, and it is computed from the same _closure_state, so climbing
    it leads to the real criterion rather than beside it.

    One-sided contact (everything pushing the same way, e.g. all fingers
    pressing down onto the palm) scores ~0 however hard it presses; a
    thumb-vs-fingers pinch with the palm underneath scores ~0.67.

    Shape: [num_envs], in [0, 1]
    """
    _, n_contact, mean_push = _closure_state(env, sensor_cfg, contact_threshold)
    contact_frac = (n_contact.float() / 3.0).clamp(max=1.0)
    opposition = (1.0 - mean_push).clamp(min=0.0)
    return contact_frac * opposition * _grasp_mask(env)


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
    return torch.exp(-(err ** 2) / (2.0 * std ** 2)) * has_contact * _grasp_mask(env)


def reward_grasp_hold(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
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
        return env._force_closure_ok.float() * _grasp_mask(env)
    return _force_closure_ok(env, sensor_cfg, contact_threshold=contact_threshold).float() * _grasp_mask(env)


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


def penalty_excess_hand_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    start_frac: float = 0.6,
    cap_frac: float = 0.9,
) -> torch.Tensor:
    """Quadratically-growing penalty for finger-group contact force
    approaching THIS EPISODE's own randomized bio-damage threshold
    (env._fruit_damage_force) -- scaled per-episode rather than a fixed N
    band, because the commanded grip setpoint (env._target_grip_level)
    legitimately ranges anywhere from ~1 N up to just under that threshold
    (domain_randomization.randomize_fruit_damage_force), and a fixed cap
    would fight grasp_setpoint_tracking on high-target episodes.

    Zero below start_frac * threshold, grows to 1.0 (times RewardsCfg's
    weight) at cap_frac * threshold, clamped beyond -- same "capped, not
    unbounded" reasoning as reach_rewards.penalty_excess_palm_force (an
    unclamped penalty on a rare, huge outlier teaches outright avoidance,
    not "be gentler").

    Added after the first real grasp-phase training run: with NO force
    shaping between "gentle" and terminations_and_events.fruit_damaged's
    hard, instantaneous cutoff, an empirical trace of the resulting policy
    showed 21-76 N transient spikes on first finger contact, and
    fruit_damaged fired in 93.75% of episodes at a median of 11 control
    steps (0.22 s) into the grasp phase -- so close to the hand-off that
    grasp_hold/grasp_success never once registered nonzero across 1407
    iterations. fruit_damaged's own hard-cutoff design is intentional
    (bio-yield is a near-instantaneous threshold effect, not something to
    average -- see its own docstring), so this doesn't touch that; it adds
    the missing DENSE, WITHIN-episode gradient toward backing off before
    the cutoff, mirroring how the reach phase solved the identical problem
    for the palm rather than relying solely on "the episode ended early"
    (a sparse, delayed signal PPO had 1407 iterations to learn nothing
    from).
    """
    if not hasattr(env, "_fruit_damage_force"):
        return torch.zeros(env.num_envs, device=env.device)
    force = _per_finger_group_force(env, sensor_cfg).max(dim=-1).values
    threshold = env._fruit_damage_force
    start = start_frac * threshold
    span = (cap_frac - start_frac) * threshold
    excess = (force - start).clamp(min=0.0)
    excess = torch.minimum(excess, span)
    return ((excess / span) ** 2) * _grasp_mask(env)


def penalty_joint_deviation_from_handoff(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    weights: Sequence[float],
) -> torch.Tensor:
    """Weighted ||q - q_handoff||^2, where q_handoff is the ARM pose the
    reach phase left it at (env._grasp_handoff_joint_pos, a full-articulation
    snapshot indexed by asset_cfg.joint_ids) — NOT the
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
    if not hasattr(env, "_grasp_handoff_joint_pos"):
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - env._grasp_handoff_joint_pos[:, asset_cfg.joint_ids]
    w = torch.tensor(weights, device=env.device, dtype=dev.dtype)
    return (w * dev ** 2).sum(dim=-1) * _grasp_mask(env)


def hand_motion_monitor(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean |hand joint velocity| (rad/s) -- NOT real reward shaping, purely
    a diagnostic so "are the fingers actually moving during training" is
    visible in the console printout / TensorBoard as Episode_Reward/
    hand_motion_monitor, without needing to stop training and run play.py.

    Give this an intentionally small (not literally zero) weight in
    RewardsCfg: Isaac Lab's RewardManager SKIPS computing AND logging any
    term with weight==0.0 entirely (confirmed against
    managers/reward_manager.py's compute() -- a weight-0 term never even
    calls its function), so exactly-zero would make this useless for
    monitoring. A small nonzero weight (e.g. 0.05) keeps its contribution
    to the actual training gradient negligible next to the real
    grasp-quality terms while still producing a legible, nonzero number
    whenever the hand is genuinely moving.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dq = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return dq.abs().mean(dim=-1) * _grasp_mask(env)


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
    # Only once BOTH actions are the trainee's own: on the first grasp step
    # `prev` is an ignored reach-phase action, so the difference means
    # nothing. prev_mask is last step's phase, cached here each call.
    mask = _grasp_mask(env)
    prev_mask = getattr(env, "_action_rate_prev_mask", None)
    if prev_mask is None or prev_mask.shape != mask.shape:
        prev_mask = torch.zeros_like(mask)
    both = mask * prev_mask * (env.episode_length_buf > 1).float()
    env._action_rate_prev_mask = mask.clone()
    return (w * delta ** 2).sum(dim=-1) * both


def penalty_joint_pos_limits(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Isaac Lab's mdp.joint_pos_limits, gated to the grasp phase (the reach
    policy's own joint-limit excursions aren't the trainee's doing)."""
    import isaaclab.envs.mdp as mdp
    return mdp.joint_pos_limits(env, asset_cfg) * _grasp_mask(env)
