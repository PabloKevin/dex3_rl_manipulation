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
from isaaclab.utils.math import quat_apply

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

# Contact groups for the closure check: the palm plus each digit. These are
# exactly the links that carry a REAL taxel (scripts/tactile_layout.py's
# TAXELS: 3 palm, 2 thumb, 2 index, 2 middle), because everything this
# check reads has to be reproducible on the real hand -- see _pad_state.
# right_hand_thumb_0_link is absent for that reason: no pad sits on it.
_CLOSURE_GROUPS = {
    "palm": ["right_hand_palm_link"],
    "thumb": ["right_hand_thumb_1_link", "right_hand_thumb_2_link"],
    "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
    "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"],
}


# Flexion joints per finger, and which way each one CLOSES. Measured from
# the articulation's own limits (soft_joint_pos_limits, read at runtime --
# no hardcoded ranges): index/middle close toward their UPPER limit
# (0 -> ~1.5/1.7 rad), the thumb's two flexion joints toward their LOWER
# one. right_hand_thumb_0_joint is deliberately absent: it is the thumb's
# rotation/opposition axis, not flexion -- reward_grasp_enclosure is what
# scores the thumb opposing the fingers.
_FLEX_JOINTS = {
    "thumb": {"right_hand_thumb_1_joint": -1, "right_hand_thumb_2_joint": -1},
    "index": {"right_hand_index_0_joint": +1, "right_hand_index_1_joint": +1},
    "middle": {"right_hand_middle_0_joint": +1, "right_hand_middle_1_joint": +1},
}


def _finger_flexion(
    env: "ManagerBasedRLEnv",
    full_credit_flex: float = 0.3,
) -> torch.Tensor:
    """[N, 3] how curled each finger (thumb/index/middle) is, in [0, 1].

    Per joint: fraction of the way from its OPEN limit to its CLOSED one.
    Per finger: the MINIMUM across its joints, so "every part of the finger
    is at least a little closed" is what scores -- a finger with a bent
    knuckle but an otherwise straight, open span gets ~0, which is the
    exact pose that was collecting contact reward before (apple touching
    the BASE of a fully open index/middle).

    Saturates at full_credit_flex (default 30% of the range), so a light,
    natural curl already earns full credit and nothing here pushes toward
    crushing the apple with a maximally clenched fist -- past contact,
    further flexion only raises force, which reward_gentle_contact_groups
    already decays and penalty_excess_hand_force/fruit_damaged punish.

    Order matches _CLOSURE_GROUPS' finger entries (thumb, index, middle).
    """
    robot: Articulation = env.scene["robot"]
    if not hasattr(env, "_flex_joint_cache"):
        ids, signs = [], []
        for joints in _FLEX_JOINTS.values():
            row_i, row_s = [], []
            for name, sign in joints.items():
                row_i.append(robot.data.joint_names.index(name))
                row_s.append(float(sign))
            ids.append(row_i); signs.append(row_s)
        env._flex_joint_cache = (
            torch.tensor(ids, device=env.device),                      # [3, 2]
            torch.tensor(signs, device=env.device),                    # [3, 2]
        )
    ids, signs = env._flex_joint_cache

    q = robot.data.joint_pos[:, ids]                                   # [N, 3, 2]
    lo = robot.data.soft_joint_pos_limits[:, ids, 0]                   # [N, 3, 2]
    hi = robot.data.soft_joint_pos_limits[:, ids, 1]
    # sign +1 closes toward hi, -1 closes toward lo
    closed = torch.where(signs > 0, hi, lo)
    opened = torch.where(signs > 0, lo, hi)
    frac = (q - opened) / (closed - opened + 1e-6)
    per_joint = (frac / full_credit_flex).clamp(min=0.0, max=1.0)
    return per_joint.min(dim=-1).values                                # [N, 3]


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


# Outward normal of each taxel pad, in ITS OWN LINK's frame -- a fixed
# geometric property of the hand, so the real robot gets these in the world
# frame from its own joint encoders + URDF, with no perception and no
# calibration. Measured in sim (scratch probe, ~23k contact samples over 40+
# scripted grasp poses): for each link, mean of the unit contact force it
# received, negated (a compressive contact pushes the link along -n_out).
# Angular spread around these means was 2-10 deg median, and the palm's
# (+0.18, +0.98, +0.12) independently reproduces apple_grasp_env_cfg's
# PALM_NORMAL_AXIS = (0, 1, 0), which is a good sanity check on the method.
#
# middle_0 never registered a single contact in any probed pose (nor did
# thumb_1 in the second sweep), so it carries its own finger's distal-link
# normal as a documented fallback -- it is very nearly unused in practice.
_PAD_NORMAL_LOCAL = {
    "right_hand_palm_link":     (0.176, 0.977, 0.121),
    "right_hand_thumb_1_link":  (0.986, 0.000, -0.169),
    "right_hand_thumb_2_link":  (0.937, 0.176, -0.301),
    "right_hand_index_0_link":  (-0.088, 0.996, 0.018),
    "right_hand_index_1_link":  (0.129, 0.990, 0.050),
    "right_hand_middle_0_link": (0.222, 0.401, 0.889),   # fallback: middle_1's
    "right_hand_middle_1_link": (0.222, 0.401, 0.889),
}


def _pad_state(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per contact group: sensed force magnitude [N, 4] and the unit
    direction that group pushes the fruit [N, 4, 3] -- BOTH computed only
    from signals the real Dex3-1 also has.

    Deliberately NOT the simulator's contact-force vectors, which the real
    hand cannot measure. What a real pad reports is a single scalar
    pressure (unitree_hg PressSensorState_, 9 taxels: 3 palm, 2 per
    finger -- see scripts/tactile_layout.py), so:

    - magnitude: the UNFILTERED net contact force on each taxel's link
      (what a pad feels, whatever touched it -- not the apple-filtered
      sensor, which is simulator-only knowledge), scaled per taxel by
      env._tactile_gain so the same calibration randomization the tactile
      OBSERVATIONS get also perturbs this, then averaged per group.
    - direction: from forward kinematics, not from force. Each pad's
      outward normal is fixed in its own link frame (_PAD_NORMAL_LOCAL),
      so rotating it by that link's orientation gives the direction the pad
      presses -- joint encoders and the URDF, both of which the real robot
      has. No perception estimate is needed. The per-group direction is its
      taxels' directions, magnitude-weighted.

      A pad-origin-to-fruit-centre proxy was tried first and rejected on
      measurement: link origins sit well off the real pads (the palm's is
      back at the wrist), the directions came out far too parallel to each
      other, and closure reachability across a 16-pose curl sweep fell from
      13/16 poses to 2/16. Link-frame normals have 2-10 deg spread and keep
      it at 12/16.
    """
    import scripts.tactile_layout as tactile_layout

    robot: Articulation = env.scene["robot"]
    sensor = env.scene[sensor_cfg.name]

    if not hasattr(env, "_pad_cache"):
        taxels = tactile_layout.TAXELS
        group_names = list(_CLOSURE_GROUPS.keys())
        # taxel -> its group, and the link ids it reads (sensor + articulation)
        sensor_bodies = list(sensor.body_names)
        t_group = [group_names.index("palm" if t.finger == "palm" else t.finger) for t in taxels]
        t_sensor_id = [sensor_bodies.index(t.link_name) for t in taxels]
        t_body_id = [robot.data.body_names.index(t.link_name) for t in taxels]
        n = torch.tensor([_PAD_NORMAL_LOCAL[t.link_name] for t in taxels], device=env.device)
        env._pad_normals = n / n.norm(dim=-1, keepdim=True)
        env._pad_cache = (
            torch.tensor(t_group, device=env.device),
            torch.tensor(t_sensor_id, device=env.device),
            torch.tensor(t_body_id, device=env.device),
            len(group_names),
        )
    t_group, t_sensor_id, t_body_id, n_groups = env._pad_cache

    # -- magnitude per taxel: unfiltered net force on its link, gained --
    net = sensor.data.net_forces_w[:, t_sensor_id, :]               # [N, T, 3]
    mag_t = torch.norm(net, dim=-1)                                  # [N, T]
    if hasattr(env, "_tactile_gain") and env._tactile_gain.shape[-1] == mag_t.shape[-1]:
        mag_t = mag_t * env._tactile_gain

    # -- direction per taxel: its pad normal, rotated by FK --
    quats = robot.data.body_quat_w[:, t_body_id, :]                  # [N, T, 4]
    n_local = env._pad_normals                                       # [T, 3]
    dir_t = quat_apply(
        quats.reshape(-1, 4), n_local.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3)
    ).reshape(quats.shape[0], quats.shape[1], 3)

    # -- pool taxels into groups --
    N = env.num_envs
    mag_g = torch.zeros(N, n_groups, device=env.device)
    cnt_g = torch.zeros(N, n_groups, device=env.device)
    dir_g = torch.zeros(N, n_groups, 3, device=env.device)
    idx = t_group.unsqueeze(0).expand(N, -1)
    mag_g.scatter_add_(1, idx, mag_t)
    cnt_g.scatter_add_(1, idx, torch.ones_like(mag_t))
    dir_g.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, 3), dir_t * mag_t.unsqueeze(-1))
    mag_g = mag_g / cnt_g.clamp(min=1)
    dir_g = dir_g / (dir_g.norm(dim=-1, keepdim=True) + 1e-9)
    return mag_g, dir_g


def _closure_state(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
    min_flex: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared by _force_closure_ok and reward_grasp_enclosure, so the hard
    criterion and the reward that shapes toward it can never disagree.

    Everything here is real-robot computable (see _pad_state): sensed taxel
    pressure, joint encoders, and a perceived fruit centre. That matters
    because this drives env._grasp_ever_closed, which is not only a reward
    input but an OBSERVATION (grip_established) and an interlock on the arm
    -- both of which deployment has to reproduce from real signals. The
    SAFETY terms deliberately do not go through here: fruit_damaged and
    penalty_excess_hand_force keep reading true, apple-filtered force
    (_per_finger_group_force), because "am I bruising the fruit" must not
    depend on an uncalibrated sensor.

    A FINGER group only counts as a contact if it is also curled at least
    min_flex (as _finger_flexion scores it, so 0.5 means ~15% of the joint
    range). The palm always counts. Without this, the apple merely resting
    against the straight base of a fully open index and middle counted as
    two of the three groups closure needs -- a pose a policy converged on,
    and one that cannot hold anything once the apple is released.

    Returns (per-group sensed magnitude [N, 4], number of groups in contact
    [N], ||mean unit push on the fruit|| [N]).
    """
    mag, push_on_apple = _pad_state(env, sensor_cfg)
    curled = _finger_flexion(env) >= min_flex                  # [N, 3]
    counts = torch.cat([torch.ones_like(curled[:, :1]), curled], dim=-1)
    thresh = contact_threshold * getattr(env, "_closure_thresh_scale", 1.0)
    if isinstance(thresh, torch.Tensor):
        thresh = thresh.unsqueeze(-1)
    in_contact = (mag > thresh) & counts
    n_contact = in_contact.sum(dim=-1)
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
    full_credit_flex: float = 0.3,
) -> torch.Tensor:
    """Mean over the 3 FINGERS (thumb/index/middle, see _CLOSURE_GROUPS) of
    a trapezoid on that finger's own contact force, scaled by how curled it
    is:
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
    band = ramp_up * ramp_down                                         # [N, 4]

    # FINGERS ONLY (columns 1..3 of _CLOSURE_GROUPS), each scaled by how
    # curled that finger is. Two things this rules out, both observed:
    # contact paying the same whether the finger was wrapped around the
    # apple or lying straight against it (a policy converged on the apple
    # resting against the base of a fully open index and middle, thumb
    # closing on nothing); and the palm alone -- which touches the apple
    # from the moment of hand-off, for free -- carrying this term. Keeping
    # the apple seated on the palm is still rewarded, by reward_finger_curl
    # (gated on palm contact) and by reward_grasp_enclosure (which counts
    # the palm as one of its groups).
    flex = _finger_flexion(env, full_credit_flex)                      # [N, 3]
    return (band[:, 1:] * flex).mean(dim=-1) * _grasp_mask(env)


def reward_finger_curl(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    full_credit_flex: float = 0.3,
    palm_contact_min: float = 0.5,
) -> torch.Tensor:
    """Mean finger curl (_finger_flexion) while the apple is actually in the
    hand, i.e. gated on the PALM being in contact with it.

    The gradient that gets a finger from "open" to "curled enough to be
    paid for touching": with reward_gentle_contact_groups now scaled by
    flexion, an open finger earns nothing for touching, and closing a
    finger that is not yet touching earns nothing either -- this bridges
    that gap. The palm-contact gate is what stops it being farmed by
    curling the fingers in mid-air with the apple nowhere near the hand.

    Shape: [num_envs], in [0, 1]
    """
    mag, _, _ = _closure_state(env, sensor_cfg, contact_threshold=palm_contact_min)
    apple_in_hand = (mag[:, 0] > palm_contact_min).float()             # palm group
    return _finger_flexion(env, full_credit_flex).mean(dim=-1) * apple_in_hand * _grasp_mask(env)


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


def reward_apple_centered(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    center_local: Sequence[float],
    std: float = 0.03,
) -> torch.Tensor:
    """exp(-d^2 / 2 std^2) on the distance from the fruit centre to the
    hand's GRASP POCKET -- a fixed point in the palm link's frame
    (center_local), i.e. "is the apple in the middle of the hand".

    Nothing else in this reward set asks for this. reward_finger_approach
    is a mean over fingertip distances, so one finger close to the apple
    offsets two far from it; reward_grasp_balance only pays once all three
    are already touching. Both are satisfied by the pose a policy actually
    converged on: the apple caught off to one side on the index while the
    middle and thumb close on nothing.

    center_local is measured, not chosen (see the grasp env cfg's
    GRASP_CENTER_LOCAL): across every scripted curl combination that
    latched a grip AND still held the apple a second after the stem let go,
    the fruit centre sat at the same point in the palm frame to within
    4-6 mm. The reach hand-off leaves it ~3.3 cm off that point, so this
    term is what asks the fingers to draw the apple into the pocket --
    which the compliant stem now lets them do.

    Real-robot computable: palm pose from FK, fruit pose from the same
    perception estimate the reach policy already consumes as apple_pos_palm.

    Shape: [num_envs], in [0, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_cfg.body_ids[0]
    quat = robot.data.body_quat_w[:, body_id]
    pocket = robot.data.body_pos_w[:, body_id] + quat_apply(
        quat, torch.tensor(center_local, device=env.device, dtype=quat.dtype).expand(env.num_envs, 3)
    )
    d = torch.norm(obj.data.root_pos_w - pocket, dim=-1)
    return torch.exp(-(d ** 2) / (2.0 * std ** 2)) * _grasp_mask(env)


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
    arm_dims: int = 0,
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
    if arm_dims:
        # While the arm is locked (see ReachHandoffJointPositionAction's
        # unlock_with_grip) its actions do nothing, so charging for their
        # jitter is a penalty the policy cannot act on -- and this term was
        # the single largest negative in the run where that noise was left
        # unpriced. Arm columns only start counting once the arm can move.
        from .grasp_autoplay import grip_established
        arm_gate = grip_established(env).float().unsqueeze(-1)
        gate = torch.cat([arm_gate.expand(-1, arm_dims),
                          torch.ones(env.num_envs, delta.shape[-1] - arm_dims, device=env.device)], dim=-1)
        delta = delta * gate

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
