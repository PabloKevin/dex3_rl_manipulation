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


# Side-on fruit (commercial fruiting-wall tree, see apple_grasp_env_cfg.py's
# tree block). The reach ends with the apple centred on the palm and gently
# touching it (_palm_target_dist, defined in the palm's own frame, so the
# policy can see it from the apple position relative to the palm). Which side
# the palm comes from and where the fingers point are NOT constrained: any
# side gives a graspable apple. The stem direction (env._stem_dir, apple ->
# branch tip) only defines the IDEAL pose used for placement and the
# workspace probe (_grasp_target_pos / _approach_dir); it is not observed
# and not rewarded.
#: Fallback approach origin when a scene has no env._stem_dir.
APPROACH_BODY = "right_shoulder_pitch_link"

#: Palm-link local axis that points from the wrist toward the fingertips
#: (probe_palm_offset.py: knuckles at local +X). Diagnostics only.
PALM_FINGER_AXIS = (1.0, 0.0, 0.0)

#: Fallback when the scene has no per-env apple radius (env._apple_radius).
DEFAULT_APPLE_RADIUS = 0.04


def _approach_dir(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Horizontal unit approach direction: the stem direction (apple ->
    branch tip, env._stem_dir). The palm normal should point along it, and
    the grasp target sits on the opposite side of the apple.

    Falls back to the horizontal shoulder -> apple direction where no stem
    direction has been set (scenes without randomize_tree_and_apple).

    If env._approach_from_body is set (single_reach.set_approach_origin), the
    stem is not used at all: the horizontal direction from that body to the
    apple (robot-knowable, unlike the stem).

    Returns: [num_envs, 3] (z = 0)
    """
    origin = getattr(env, "_approach_from_body", None)
    if origin is not None:
        robot: Articulation = env.scene["robot"]
        apple: RigidObject = env.scene[object_cfg.name]
        d = apple.data.root_pos_w - robot.data.body_pos_w[:, origin, :]
        d[:, 2] = 0.0
        return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    stem = getattr(env, "_stem_dir", None)
    if isinstance(stem, torch.Tensor):
        d = stem.clone()
    else:
        d = torch.zeros(env.num_envs, 3, device=env.device)
    missing = d[:, :2].norm(dim=-1) < 0.5
    if missing.any():
        robot: Articulation = env.scene["robot"]
        if not hasattr(env, "_approach_body_id"):
            env._approach_body_id = robot.data.body_names.index(APPROACH_BODY)
        apple: RigidObject = env.scene[object_cfg.name]
        fallback = apple.data.root_pos_w - robot.data.body_pos_w[:, env._approach_body_id, :]
        d = torch.where(missing.unsqueeze(-1), fallback, d)
    d[:, 2] = 0.0
    return d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _grasp_target_pos(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    standoff: float,
) -> torch.Tensor:
    """Where the palm's grasp reference point sits in the IDEAL pose:
    `standoff` metres beyond the apple SURFACE, back along the stem, i.e. on
    the side of the fruit opposite the stem, at the apple's own height.
    Placement (domain_randomization) and the workspace probe use it; the
    rewards use the palm-relative _palm_target_dist instead. Uses the per-env
    apple radius (domain_randomization.randomize_apple_scale) when present.

    Returns: [num_envs, 3]
    """
    apple: RigidObject = env.scene[object_cfg.name]
    radius = getattr(env, "_apple_radius", None)
    if not isinstance(radius, torch.Tensor):
        radius = torch.full((env.num_envs,), DEFAULT_APPLE_RADIUS, device=env.device)
    back = (radius + standoff).unsqueeze(-1)
    return apple.data.root_pos_w - back * _approach_dir(env, object_cfg)


def _palm_target_dist(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """How far the apple is from sitting centred on the palm, just touching:
    distance between the apple centre and the point (radius + standoff)
    out from the palm reference point along the palm normal. Expressed in
    the palm's own frame, so it is observable (apple position relative to
    the palm) and independent of the stem, so any approach side scores the
    same. At the ideal (along-the-stem) pose it equals the distance to
    _grasp_target_pos.

    Returns: [num_envs]
    """
    apple: RigidObject = env.scene[object_cfg.name]
    radius = getattr(env, "_apple_radius", None)
    if not isinstance(radius, torch.Tensor):
        radius = torch.full((env.num_envs,), DEFAULT_APPLE_RADIUS, device=env.device)
    n_world = _palm_normal_world(env, robot_cfg, palm_normal_axis)
    seat = _palm_ref_pos(env, robot_cfg, palm_local_offset) + (radius + standoff).unsqueeze(-1) * n_world
    return torch.norm(apple.data.root_pos_w - seat, dim=-1)


def _palm_normal_angle(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float],
    object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Angle (rad) between the palm normal and the approach direction
    (_approach_dir, the stem), i.e. how squarely the palm faces the apple.

    `palm_normal_axis` is the palm normal expressed in the palm link's OWN
    local frame. object_cfg defaults to the scene's "apple".

    Returns: [num_envs], 0 = palm facing the apple, pi = facing away.
    """
    if object_cfg is None:
        object_cfg = SceneEntityCfg("apple")
    n_world = _palm_normal_world(env, robot_cfg, palm_normal_axis)
    if getattr(env, "_heading_horizontal", False):
        # single_reach.set_reach_goal_v2: the HEADING only (the normal's
        # horizontal part); the tilt is the tilt terms' business
        n_world = n_world.clone()
        n_world[:, 2] = 0.0
        n_world = n_world / n_world.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos = (n_world * _approach_dir(env, object_cfg)).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


def _fingers_up_angle(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    finger_axis: Sequence[float] = PALM_FINGER_AXIS,
) -> torch.Tensor:
    """Angle (rad) between the palm's wrist->fingertip axis and world +Z.

    Returns: [num_envs], 0 = fingers straight up.
    """
    f_world = _palm_normal_world(env, robot_cfg, finger_axis)
    return torch.acos(f_world[:, 2].clamp(-1.0, 1.0))


def _palm_normal_world(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float],
) -> torch.Tensor:
    """Unit palm normal in WORLD frame, pointing OUT of the palm face.

    Returns: [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    quat = robot.data.body_quat_w[:, robot_cfg.body_ids[0], :]   # [N, 4] wxyz
    rot = matrix_from_quat(quat)                                  # [N, 3, 3]

    n_local = torch.tensor(palm_normal_axis, device=env.device, dtype=quat.dtype)
    n_local = n_local / (n_local.norm() + 1e-9)
    return torch.matmul(rot, n_local)                             # [N, 3]


def _palmar_dorsal_force(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float],
    link_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split apple contact force on each link into the part pressing on the
    PALM side of the hand and the part pressing on the BACK of it.

    Returns: (palmar [N, L], dorsal [N, L]), both non-negative, in newtons.

    Why this exists
    ---------------
    A ContactSensor reports contact on a whole rigid BODY, not on a patch of
    its surface: right_hand_palm_link registers a contact identically whether
    the apple touched the palm face or the back of the hand. So a reward that
    reads only |force| cannot tell the two apart, and
    reward_gentle_touch paid exactly the same for either -- which is the
    degenerate optimum a policy trained against it actually found: touch the
    apple with the BACK of the hand, collect the touch reward, and never
    bother rotating the palm up at all.

    The fix does not need any new sensing, just the sign of the force. A
    compressive contact on a surface whose outward normal is n_hat pushes the
    body it acts on along -n_hat (this is what "compressive" means, and it is
    why net_forces_w on a foot standing on the ground points UP). So with
    n_hat the OUTWARD palm normal:

        dot(F, n_hat) < 0  ->  something is pressing on the palm face
        dot(F, n_hat) > 0  ->  something is pressing on the back of the hand

    and the magnitudes are just the two clamped halves of that projection.
    No extra geometry has to be measured: palm_normal_axis is already known
    and already used by reward_palm_normal_up.

    The palm's own normal is used as the reference direction for the digit
    links too. That is an approximation -- a curled finger's pad normal is
    not exactly the palm normal -- but it is the RIGHT approximation here,
    because what this has to distinguish is "the apple is on the grasping
    side of the hand" from "the apple is on the back of it", and for any
    cupping pose the whole grasping side faces within well under 90 deg of
    the palm normal. The sign is what matters, not the exact angle.
    """
    from .observations import apple_force_vec_per_link

    n_world = _palm_normal_world(env, robot_cfg, palm_normal_axis)    # [N, 3]
    forces = apple_force_vec_per_link(env, sensor_cfg, link_names)    # [N, L, 3]
    f_n = (forces * n_world.unsqueeze(1)).sum(dim=-1)                 # [N, L]
    return (-f_n).clamp(min=0.0), f_n.clamp(min=0.0)


# ═════════════════════════════════════════════════════════════════════════
#  Position: three scales so there is gradient at every distance
# ═════════════════════════════════════════════════════════════════════════

def reward_reach_progress(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float = 0.015,
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
    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset)

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
    standoff: float = 0.015,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Linear distance penalty. Use a NEGATIVE weight.

    Guarantees a non-zero pull even if progress shaping stalls, and makes
    "hover far away doing nothing" strictly worse than approaching.
    """
    return _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset)


def reward_reach_gaussian(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float = 0.015,
    std: float = 0.05,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """exp(-d^2 / 2 std^2), in [0, 1].

    Register this TWICE: once coarse (std ~0.25) for a broad basin, once
    fine (std ~0.04) for a sharp peak at the goal. A single std=0.05 term
    is numerically zero beyond ~0.15 m -- exp(-18) = 1.5e-8 at d = 0.30 m --
    which is why the original palm_position term never trained anything.
    """
    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset)
    return torch.exp(-(d ** 2) / (2.0 * std ** 2))


def _palm_tilt(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Angle (rad) between the palm normal and the horizontal plane:
    0 = palm facing sideways (any heading), pi/2 = facing straight up/down.

    Returns: [num_envs]
    """
    n_world = _palm_normal_world(env, robot_cfg, palm_normal_axis)
    return torch.asin(n_world[:, 2].abs().clamp(max=1.0))


def reward_palm_lateral(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
    std_rad: float = 0.3,
    gate_dist: float = 0.55,
    standoff: float = 0.015,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Reward the palm facing SIDEWAYS (normal horizontal, any heading), so
    the apple is touched from a lateral and the harvest motion can be
    horizontal. Gaussian on _palm_tilt, scaled linearly to 0 beyond
    gate_dist from the touch pose so it cannot be farmed instead of reaching.
    """
    tilt = _palm_tilt(env, robot_cfg, palm_normal_axis)
    r = torch.exp(-(tilt ** 2) / (2.0 * std_rad ** 2))
    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    return r * (1.0 - d / gate_dist).clamp(0.0, 1.0)


def reward_palm_toward_stem(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
    std_rad: float = 0.5,
    gate_dist: float = 0.55,
    standoff: float = 0.015,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Soft pull of the palm normal toward the stem direction (apple ->
    branch tip, env._stem_dir), i.e. touching the apple from the side
    opposite the stem: keeps the hand off the stem/branch and out of
    neighbouring fruit. The stem is NOT observed, so the policy can only aim
    at the typical stem direction for an apple at that position; a wide
    std_rad makes "similar" good enough. Gated to 0 beyond gate_dist.
    """
    theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis, object_cfg)
    r = torch.exp(-(theta ** 2) / (2.0 * std_rad ** 2))
    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    return r * (1.0 - d / gate_dist).clamp(0.0, 1.0)


def reach_neighbourhood(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_tol: float,
    tilt_tol_rad: float | None,
    standoff: float,
    palm_normal_axis: Sequence[float],
    palm_local_offset: Sequence[float],
    stem_tol_rad: float | None = None,
) -> torch.Tensor:
    """Apple centred on the palm (_palm_target_dist within pos_tol); if
    tilt_tol_rad is given, the palm normal within it of horizontal
    (_palm_tilt); if stem_tol_rad is given, within it of the stem direction
    (_palm_normal_angle). Shared by reward_reach_hold,
    terminations_and_events.reach_success and the grasp task's converged
    hand-off so they cannot disagree.

    If env._enclosure_goal is set (single_reach.set_reach_goal_v2), the
    position test is single_reach.enclosed instead (apple in the hand's
    pocket, pushed in at most a little, the target not dragged off its spot);
    pos_tol / standoff are then unused.

    Returns: [num_envs] bool
    """
    if getattr(env, "_enclosure_goal", None) is not None:
        from .single_reach import enclosed
        inside = enclosed(env, robot_cfg, object_cfg, palm_local_offset, palm_normal_axis)
    else:
        inside = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis) < pos_tol
    if tilt_tol_rad is not None:
        inside = inside & (_palm_tilt(env, robot_cfg, palm_normal_axis) < tilt_tol_rad)
    if stem_tol_rad is not None:
        inside = inside & (_palm_normal_angle(env, robot_cfg, palm_normal_axis, object_cfg) < stem_tol_rad)
    return inside


def reward_reach_hold(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_tol: float = 0.04,
    tilt_tol_rad: float | None = None,
    standoff: float = 0.015,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
    stem_tol_rad: float | None = None,
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
    in_neighbourhood = reach_neighbourhood(
        env, robot_cfg, object_cfg, pos_tol, tilt_tol_rad, standoff,
        palm_normal_axis, palm_local_offset, stem_tol_rad,
    )

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
    robot_cfg: SceneEntityCfg,
    good_min: float = 0.5,
    good_max: float = 5.0,
    link_names: Sequence[str] | None = None,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Flat, per-step bonus (0 or 1) for contact force on `link_names`
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

    `link_names` (default: observations._PALM_CONTACT_LINKS, the palm) exists so RewardsCfg can
    register this term MORE THAN ONCE over different parts of the hand at
    different weights -- "touching with the palm or thumb" is a better
    hand-off pose for the grasp phase than "touching with an index or
    middle finger", and a single term summing all four links cannot express
    that preference: it pays exactly the same for either. Splitting it also
    gives one TensorBoard curve per group, so which part of the hand the
    policy actually settles on is directly readable instead of inferred.

    Note the band applies to the SUM over `link_names`, so a group is a set
    of links that should be touching together, not an arbitrary bag.

    Only the PALMAR component of the contact force counts (see
    _palmar_dorsal_force). Touching the apple with the BACK of the hand pays
    nothing here, no matter how clean the force magnitude looks. This is not
    a refinement -- it is load-bearing. Without it this term is maximised by
    backhanding the apple, which needs no wrist rotation at all and is
    therefore much easier to find than cupping: a policy trained against the
    magnitude-only version converged straight onto it, took the touch reward
    every step, and left reward_palm_normal_up sitting at ~0 for the whole
    run because nothing ever required the palm to face anywhere in
    particular.
    """
    from .observations import _PALM_CONTACT_LINKS
    if link_names is None:
        link_names = _PALM_CONTACT_LINKS
    palmar, _ = _palmar_dorsal_force(
        env, robot_cfg, sensor_cfg, palm_normal_axis, link_names
    )
    force = palmar.sum(dim=-1)
    return ((force >= good_min) & (force <= good_max)).float()


def penalty_back_of_hand_contact(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    palm_normal_axis: Sequence[float] = (0.0, 1.0, 0.0),
    cap: float = 5.0,
    link_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Penalise the apple pressing on the BACK of the hand, anywhere, at any
    distance. Use a NEGATIVE weight.

    Dorsal contact is never part of a correct reach: there is no pose in
    which the apple should be resting against the back of the hand or the
    back of the thumb. So unlike penalty_premature_contact this needs no
    distance gate -- the contact is wrong on its own terms, not wrong
    because of when it happened, and gating it would just leave a region
    where backhanding is free again.

    Making reward_gentle_touch palmar-only already removes the INCENTIVE to
    backhand the apple; this adds an active cost, so a trajectory that
    sweeps the back of the hand through the apple on its way in is worse
    than one that goes around, rather than merely no better.

    Force is clamped per link at `cap`, same reasoning as
    penalty_excess_palm_force and penalty_premature_contact: a fast clip
    spikes far above any steady contact, and an unbounded penalty on a
    transient outlier teaches avoidance of the whole region instead of
    teaching a cleaner approach.

    Returns [num_envs], 0 when nothing is touching the back of the hand.
    """
    from .observations import RIGHT_HAND_CONTACT_LINKS
    if link_names is None:
        link_names = RIGHT_HAND_CONTACT_LINKS
    _, dorsal = _palmar_dorsal_force(
        env, robot_cfg, sensor_cfg, palm_normal_axis, link_names
    )
    return (dorsal.clamp(max=cap) / cap).sum(dim=-1)


def penalty_apple_contact_links(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    link_names: Sequence[str],
    cap: float = 5.0,
) -> torch.Tensor:
    """Penalise ANY apple contact on `link_names`, at any distance. Use a
    NEGATIVE weight.

    Used for the thumb during the reach: the reach ends with the apple
    resting gently on the palm, the thumb open below it (on the wrist side),
    so a thumb touch is always a misplaced hand -- typically the observed
    "side of the thumb pressed against the side of the apple" pose, which
    the old palm+digit-base touch group even paid for. Per-link force
    clamped at `cap` (same reasoning as penalty_back_of_hand_contact).

    Returns [num_envs], 0 when none of the links touch the apple.
    """
    from .observations import apple_force_per_link
    force = apple_force_per_link(env, sensor_cfg, link_names)
    return (force.clamp(max=cap) / cap).sum(dim=-1)


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


def penalty_premature_contact(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    gate_dist: float = 0.05,
    ramp: float = 0.03,
    cap: float = 5.0,
    standoff: float = 0.015,
    palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
    link_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Penalise apple contact ANYWHERE on the hand while the palm is still
    too far from the grasp point for that contact to be the intended gentle
    touch. Use a NEGATIVE weight.

    This is the "collided on the way in" term. The observed failure is the
    BACK of the thumb clipping the apple during the approach: the hand
    arrives from a direction that sweeps a dorsal surface through the
    apple's volume before the palm ever gets in front of it. Nothing else in
    the reward set objects -- reward_gentle_touch only looks at whether the
    force is in band (and the dorsal links aren't even in its group),
    penalty_excess_palm_force only fires above 5 N, and
    penalty_apple_contact_force's weight (-0.05, ref 5 N) makes a 3 N scrape
    cost ~0.018, i.e. nothing. So a trajectory that bumps its way in scores
    essentially the same as one that approaches cleanly.

    Rather than trying to identify the dorsal surface geometrically (which
    would need a pad-normal axis per thumb link that nobody has measured for
    this USD), this uses the fact that makes the contact wrong in the first
    place: WHEN it happens. In the intended trajectory the hand lines up in
    front of the apple first and only then closes the last couple of
    centimetres, so ANY apple contact while the palm reference point is
    still `gate_dist` away is premature by construction, whichever surface
    it lands on. Contact inside the neighbourhood -- the gentle touch this
    whole phase is trying to produce -- costs exactly zero.

    gate is a smooth ramp, not a step: zero at d <= gate_dist, 1.0 at
    d >= gate_dist + ramp. A hard step would put a discontinuity right where
    the policy is trying to settle.

    Force is CLAMPED at `cap` before normalising, for the same reason
    penalty_excess_palm_force clamps: an approach-speed clip can spike far
    above any steady contact, and an unbounded penalty on a transient
    outlier teaches "never go near the apple" rather than "go around".

    Returns [num_envs] in [0, 1] (per link summed, so a multi-link clip can
    exceed 1 -- that is intended, hitting with more of the hand is worse).
    """
    from .observations import RIGHT_HAND_CONTACT_LINKS, apple_force_per_link
    if link_names is None:
        link_names = RIGHT_HAND_CONTACT_LINKS

    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset)
    gate = ((d - gate_dist) / max(ramp, 1e-6)).clamp(0.0, 1.0)      # [N]

    force = apple_force_per_link(env, sensor_cfg, link_names)        # [N, L]
    normalised = (force.clamp(max=cap) / cap).sum(dim=-1)            # [N]
    return gate * normalised


def reward_reach_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Large one-time bonus on the step terminations_and_events.reach_success
    fires (palm held in the success neighbourhood for hold_time). g1_arm_env's
    R_success = 700, by far its largest reward term.

    Does NOT recompute the neighbourhood/hold check itself — reads
    env._reach_success, cached by reach_success. IsaacLab computes
    terminations before rewards each step (see ManagerBasedRLEnv.step), so
    that cache is always fresh here. Wire reach_success as a TerminationTermCfg
    with the SAME pos_tol/tilt_tol_rad/standoff/palm_normal_axis/hold_time as
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
    standoff: float = 0.015,
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

    d = _palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset)
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

# Arm-vs-body self-collision lives in self_collision.py (full-body sphere
# model, also used by scripts/probe_side_reach.py).
