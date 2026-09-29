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

    Used by reward_grasp_setpoint_tracking (and the legacy
    reward_grasp_balance) — the ground-truth counterpart of
    observations.tactile_force_level (that one is calibration-gained and
    pad-only, this one is neither, by design). The safety terms use
    _max_link_apple_force instead: a per-finger MEAN halves a force one
    segment carries alone.
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


def _max_link_apple_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """[N] largest apple-filtered force on ANY single hand link, on ANY face
    of it -- the privileged "is the fruit being hurt" quantity, shared by
    penalty_excess_hand_force and terminations_and_events.fruit_damaged.

    Every link in observations.RIGHT_HAND_CONTACT_LINKS: palm, all three
    thumb segments (thumb_0, the opposition knuckle, has no taxel but can
    still press on the fruit), and both segments of index and middle.
    Bruising does not care which face pressed, so no pad weighting here.

    Per-LINK max rather than the per-finger MEAN _per_finger_group_force
    gives (which halved a force carried by one segment alone): bruising is
    local, so the most heavily loaded contact is what counts.

    Edge contacts are not scaled up for their smaller area on purpose: the
    damage threshold (env._fruit_damage_force) is a texture-analyzer PROBE
    force from the literature, i.e. already a concentrated load on a small
    tip, so it is conservative for a flat pad and comparable for a pad
    edge.
    """
    from .observations import apple_force_per_link, RIGHT_HAND_CONTACT_LINKS

    return apple_force_per_link(env, sensor_cfg, RIGHT_HAND_CONTACT_LINKS).max(dim=-1).values


def _finger_segment_contacts(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Privileged per-SEGMENT contact state of the three fingers.

    Returns (force [N, 3, 2], quality [N, 3, 2]), fingers (thumb, index,
    middle) x segments (proximal, distal) -- contact_geometry.
    FINGER_PAD_LINKS order:
      force   -- apple-filtered force magnitude on the segment, any face, N.
      quality -- how squarely it lands on the pad, in [0, 1]
                 (contact_geometry.pad_face_weight over that finger's
                 GRIP_BAND_DEG): 1 on the pad (or, for index/middle, the
                 pad's inner edge), 0 on the side or back.
    Ground truth, no calibration gain -- a reward input, not an observation.
    """
    from .contact_geometry import FINGER_PAD_LINKS, GRIP_BAND_DEG, apple_contact_faces, pad_face_weight

    fingers = ("thumb", "index", "middle")
    links = [l for f in fingers for l in FINGER_PAD_LINKS[f]]
    force, cos_pad = apple_contact_faces(env, sensor_cfg, links)          # [N, 6, 3], [N, 6]
    band = torch.tensor([GRIP_BAND_DEG[f] for f in fingers for _ in range(2)], device=env.device)  # [6, 2]
    quality = pad_face_weight(cos_pad, band[:, 0], band[:, 1])
    return force.norm(dim=-1).view(-1, 3, 2), quality.view(-1, 3, 2)


def _finger_grip(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per finger: (total apple force on its pad links [N, 3], force-weighted
    mean contact quality [N, 3]).

    Quality is kept as a separate MULTIPLIER rather than folded into the
    force because the terms that use it saturate in force (bands, contact
    thresholds): a thumb pressing its side into the apple at 7 N with
    quality 0.25 still has 1.75 N of "quality-weighted force", enough to
    pass any contact threshold -- which gives no gradient toward turning
    the pad onto the fruit. Multiplying the saturated score by quality
    does, continuously in the contact angle.
    """
    force, quality = _finger_segment_contacts(env, sensor_cfg)
    total = force.sum(dim=-1)
    q_mean = (force * quality).sum(dim=-1) / total.clamp(min=1e-6)
    return total, q_mean


def _finger_grip_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """[N, 3] privileged GRIPPING force per finger: segment force times its
    contact quality, summed per finger (see _finger_segment_contacts)."""
    force, quality = _finger_segment_contacts(env, sensor_cfg)
    return (force * quality).sum(dim=-1)


def _back_contact_force(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    full_deg: float = 120.0,
    start_deg: float = 95.0,
) -> torch.Tensor:
    """[N, L] privileged apple force on the BACK of each padded link (palm,
    thumb_1/2, index_0/1, middle_0/1): counted from start_deg off the pad
    normal, in full past full_deg. The grasp-phase, per-link-normal
    counterpart of reach_rewards._palmar_dorsal_force (which uses the palm
    normal for every link)."""
    from .contact_geometry import PAD_NORMAL_LOCAL, apple_contact_faces, pad_face_weight

    links = list(PAD_NORMAL_LOCAL.keys())
    force, cos_pad = apple_contact_faces(env, sensor_cfg, links)
    # Mirror the pad weighting: -cos_pad is the cosine off the BACK normal.
    back = pad_face_weight(-cos_pad, 180.0 - full_deg, 180.0 - start_deg)
    return force.norm(dim=-1) * back


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

    - magnitude: exactly what the tactile OBSERVATIONS see (observations.
      _sensed_tactile_forces): apple force on the taxel's link, counted
      only as far as it lands on the PAD (a side-edge or back contact reads
      partly or not at all, as on the real pads -- see contact_geometry),
      with the same per-episode gain and footprint randomization, then
      averaged per group. This used to be the unfiltered net force on the
      whole link, which read a contact on any face in full.
    - direction: from forward kinematics, not from force. Each pad's
      outward normal is fixed in its own link frame (contact_geometry.
      PAD_NORMAL_LOCAL, geometric -- see there for why the earlier
      contact-averaged middle-finger normal was the side face),
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
    from .contact_geometry import PAD_NORMAL_LOCAL
    from .observations import _sensed_tactile_forces

    robot: Articulation = env.scene["robot"]

    if not hasattr(env, "_pad_cache"):
        taxels = tactile_layout.TAXELS
        group_names = list(_CLOSURE_GROUPS.keys())
        # taxel -> its group, and the articulation body it rotates with
        t_group = [group_names.index("palm" if t.finger == "palm" else t.finger) for t in taxels]
        t_body_id = [robot.data.body_names.index(t.link_name) for t in taxels]
        n = torch.tensor([PAD_NORMAL_LOCAL[t.link_name] for t in taxels], device=env.device)
        env._pad_normals = n / n.norm(dim=-1, keepdim=True)
        env._pad_cache = (
            torch.tensor(t_group, device=env.device),
            torch.tensor(t_body_id, device=env.device),
            len(group_names),
        )
    t_group, t_body_id, n_groups = env._pad_cache

    # -- magnitude per taxel: what the pad itself reads --
    mag_t = _sensed_tactile_forces(env, sensor_cfg)                  # [N, T]

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
    penalty_excess_hand_force keep reading true, apple-filtered force on
    every face of every link (_max_link_apple_force), because "am I
    bruising the fruit" must not depend on an uncalibrated sensor or on
    whether the contact happened to land on a taxel.

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


def reward_tripod_balance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    min_force: float = 0.3,
) -> torch.Tensor:
    """Tripod grasp: thumb opposing the GAP between index and middle, which
    load the fruit about equally. Privileged (_finger_grip_force: ground
    truth, pad and pad-edge contacts only).

        (fingers gripping / 3) * min(F_index, F_middle) / max(F_index, F_middle)

    Replaces reward_grasp_balance's coefficient of variation across all
    three fingers, which could not tell the tripod from the pinch it was
    meant to prevent: a tripod's thumb carries roughly the sum of the two
    fingers' pushes (it opposes both), CV(2, 1, 1) -> 0.70, while the
    observed index-thumb pinch with the middle finger as a light base
    scored about the same, CV(1, 1, 0.3) -> 0.65. Index-vs-middle symmetry
    is the property that actually separates the two: 1.0 when both carry
    the same load, 0 when one carries it all. The thumb is left to
    grasp_enclosure (it has to oppose) and gentle_contact_groups (it has to
    touch).

    The (fingers gripping / 3) factor keeps partial credit on the way in,
    same reasoning as reward_grasp_balance: no cliff before all three
    touch. Each finger counts as ramp(force / min_force) TIMES its contact
    quality (_finger_grip), not a hard count: with a hard count, a thumb
    pressing its side into the apple counted as a full third of the
    tripod, and the trained policy did exactly that.

    Shape: [num_envs], in [0, 1]
    """
    total, q = _finger_grip(env, sensor_cfg)                           # [N, 3], [N, 3]
    gripping = (total / min_force).clamp(max=1.0) * q
    g_i, g_m = total[:, 1] * q[:, 1], total[:, 2] * q[:, 2]
    sym = torch.minimum(g_i, g_m) / torch.maximum(g_i, g_m).clamp(min=min_force)
    return (gripping.sum(dim=-1) / 3.0) * sym * _grasp_mask(env)


def reward_finger_wrap(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    full_force: float = 0.5,
) -> torch.Tensor:
    """Each finger wrapped around the fruit: BOTH its proximal and distal
    segment on it, pad-side (_finger_segment_contacts). Per finger,

        min over its 2 segments of  ramp(force / full_force) * quality

    averaged over thumb, index and middle.

    Nothing else asks for this: gentle_contact_groups and tripod_balance
    read a finger's TOTAL force, so a fingertip-only pinch scores the same
    as a finger wrapped around the fruit. A trained policy settled on
    exactly that -- index/middle distal segments on the apple 84-90% of the
    time, proximal ones 3%, fingertips curled to ~1.4 rad -- and a thumb
    touching with one segment only. Base-plus-tip contact spreads the load
    along the finger and holds the fruit from more directions, which is
    what should survive the tugs of harvesting.

    Reachable, measured: scripted closures with index/middle at ~0.5 rad
    proximal / ~0.9 rad distal put both segments on the apple (22-37 deg
    off the pad, the tripod's inner-edge contact), as did thumb_1 ~0,
    thumb_2 ~-0.5 for the thumb.

    min() on purpose: one segment alone earns nothing here (it is already
    paid by gentle_contact_groups); this term is only the second one.

    Shape: [num_envs], in [0, 1]
    """
    force, quality = _finger_segment_contacts(env, sensor_cfg)         # [N, 3, 2]
    seg = (force / full_force).clamp(max=1.0) * quality
    return seg.min(dim=-1).values.mean(dim=-1) * _grasp_mask(env)


def penalty_back_contact(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    cap: float = 3.0,
) -> torch.Tensor:
    """Apple force on the BACK of any padded link (palm or finger segment),
    each link capped at `cap` N and normalised, summed over links. Use a
    NEGATIVE weight. Privileged: the real pads cannot feel the back of the
    hand at all, which is exactly why the policy needs telling in training.

    Grasp-phase counterpart of reach_rewards' dorsal-contact penalty, but
    measured against each link's OWN pad normal (the reach version uses the
    palm normal for every link, which misreads a curled finger).

    Shape: [num_envs], in [0, L]
    """
    back = _back_contact_force(env, sensor_cfg)                        # [N, L]
    return (back.clamp(max=cap) / cap).sum(dim=-1) * _grasp_mask(env)


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

    Force is PRIVILEGED (_finger_grip): the finger's total apple force (any
    face -- the band's upper edge is about not squeezing too hard, and a
    side contact squeezes too), and the band score is multiplied by the
    contact quality, so only pad (and, for index/middle, pad-edge) contact
    is paid. It used to be the grip detector's sensed magnitude, which
    (before contact_geometry) counted a contact on any face in full -- so a
    finger lying with its side against the apple was paid like one gripping
    it with its pad.

    Shape: [num_envs], in [0, 1]
    """
    mag, quality = _finger_grip(env, sensor_cfg)                       # [N, 3], [N, 3]
    ramp_up = (mag / good_min).clamp(max=1.0)
    ramp_down = (1.0 - (mag - good_max) / decay).clamp(min=0.0, max=1.0)
    band = ramp_up * ramp_down * quality                               # [N, 3]

    # FINGERS ONLY, each scaled by how curled that finger is. Two things
    # this rules out, both observed: contact paying the same whether the
    # finger was wrapped around the apple or lying straight against it (a
    # policy converged on the apple resting against the base of a fully
    # open index and middle, thumb closing on nothing); and the palm alone
    # -- which touches the apple from the moment of hand-off, for free --
    # carrying this term. Keeping the apple seated on the palm is still
    # rewarded, by reward_finger_curl (gated on palm contact) and by
    # reward_grasp_enclosure (which counts the palm as one of its groups).
    flex = _finger_flexion(env, full_credit_flex)                      # [N, 3]
    return (band * flex).mean(dim=-1) * _grasp_mask(env)


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

    "In the hand" is the palm touching it OR at least two fingers gripping
    it (_finger_grip_force): a tripod often lifts the fruit off the palm
    entirely, and a palm-only gate then withdrew this term exactly when the
    grasp got better (measured on scripted closures: 0.39 for tripods vs
    1.37 for pinches that kept the apple on the palm).

    Shape: [num_envs], in [0, 1]
    """
    mag, _, _ = _closure_state(env, sensor_cfg, contact_threshold=palm_contact_min)
    fingers_gripping = (_finger_grip_force(env, sensor_cfg) > palm_contact_min).sum(dim=-1) >= 2
    apple_in_hand = ((mag[:, 0] > palm_contact_min) | fingers_gripping).float()
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
    normal_axis_local: Sequence[float],
    std_perp: float = 0.02,
    std_along: float = 0.03,
) -> torch.Tensor:
    """"Is the apple in the middle of the hand" -- distance from the fruit
    centre to the measured grasp point (center_local, a fixed point in the
    palm's local frame), split into the component ALONG the palm's normal
    axis (normal_axis_local, e.g. PALM_NORMAL_AXIS -- depth into the hand)
    and the component PERPENDICULAR to it (how far off to one side), each
    with its OWN tolerance.

    Nothing else in this reward set asks for this at all. reward_
    finger_approach is a mean over fingertip distances, so one finger close
    to the apple offsets two far from it; reward_grasp_balance only pays
    once all three are already touching. Both are satisfied by the pose a
    policy actually converged on: the apple caught off to one side on the
    index while the middle and thumb close on nothing.

    With std_perp == std_along this is EXACTLY the isotropic 3D Gaussian
    this function's first version used (distance-to-a-point splits into
    along/perp components by Pythagoras regardless of which axis you
    decompose against, so the split alone changes nothing) -- what actually
    matters is making them UNEQUAL. std_along > std_perp on purpose: how
    deep the apple sits in the hand matters much less than how far it sits
    off to one side, which is what directly determines whether the fingers
    close on it symmetrically (the reported failure: caught on the index,
    middle and thumb closing on air).

    Note on hand orientation: this reward is computed entirely in the
    palm's own rotating local frame (both center_local and the axis
    decomposition), so it does not need the palm to point straight up, or
    to hold any particular roll around its own normal, to be satisfiable --
    only that the apple sits at the same point RELATIVE TO THE HAND. The
    reach hand-off's own tolerance (ori_tol_rad=0.30 rad) permits some
    orientation spread episode to episode; center_local's own measurement
    (4-6mm spread across real hand-off episodes at that tolerance) is the
    empirical check that this local-frame target still held across it.

    center_local is measured, not chosen (see the grasp env cfg's
    GRASP_CENTER_LOCAL): across every scripted curl combination that
    latched a grip AND still held the apple a second after the stem let go,
    the fruit centre sat at the same point to within a few mm.

    Real-robot computable: palm pose from FK, fruit pose from the same
    perception estimate the reach policy already consumes as apple_pos_palm.

    Shape: [num_envs], in [0, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = robot_cfg.body_ids[0]
    quat = robot.data.body_quat_w[:, body_id]
    ref = robot.data.body_pos_w[:, body_id]

    n_local = torch.tensor(normal_axis_local, device=env.device, dtype=quat.dtype)
    n_local = n_local / (n_local.norm() + 1e-9)
    n_world = quat_apply(quat, n_local.expand(env.num_envs, 3))

    target_world = ref + quat_apply(
        quat, torch.tensor(center_local, device=env.device, dtype=quat.dtype).expand(env.num_envs, 3)
    )
    err = obj.data.root_pos_w - target_world
    along_err = (err * n_world).sum(dim=-1)
    perp_dist = torch.norm(err - along_err.unsqueeze(-1) * n_world, dim=-1)

    along = torch.exp(-(along_err ** 2) / (2.0 * std_along ** 2))
    across = torch.exp(-(perp_dist ** 2) / (2.0 * std_perp ** 2))
    return along * across * _grasp_mask(env)


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
    """1.0 on every step terminations_and_events.grasp_success's check
    holds. With that termination's terminate=True this is a one-time bonus
    (the episode ends on that step); with terminate=False (the grasp env's
    setting) it is a PER-STEP reward for keeping the fruit lifted in a
    closed grasp -- weight it like one. Does NOT recompute the check itself — reads env._grasp_success,
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

    Reads _max_link_apple_force (every link, every face, per-link max), the
    same quantity fruit_damaged terminates on.
    """
    if not hasattr(env, "_fruit_damage_force"):
        return torch.zeros(env.num_envs, device=env.device)
    force = _max_link_apple_force(env, sensor_cfg)
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
    arm_free_dims: int = 0,
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
        # PROXIMAL arm columns (arm_dims - arm_free_dims, e.g. shoulder+
        # elbow) are locked pre-grip (see ReachHandoffJointPositionAction's
        # unlock_with_grip), so their actions do nothing then and charging
        # for their jitter is a penalty the policy cannot act on -- this
        # term was the single largest negative in the run where that noise
        # was left unpriced. DISTAL arm columns (the trailing arm_free_dims,
        # e.g. the wrist) stay free even pre-grip (pre_grip_free_dims), so
        # they always count -- same reasoning as the hand columns below.
        from .grasp_autoplay import grip_established
        arm_gate = grip_established(env).float().unsqueeze(-1)
        locked_dims = arm_dims - arm_free_dims
        gate = torch.cat([arm_gate.expand(-1, locked_dims),
                          torch.ones(env.num_envs, delta.shape[-1] - locked_dims, device=env.device)], dim=-1)
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
