# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Custom observation functions for the Apple Grasp task.

Each function signature: fn(env, **params) -> torch.Tensor  [num_envs, dim]
They are referenced by ObservationTermCfg(func=...) in the env cfg.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_conjugate, quat_mul

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
#  Apple-filtered contact force: per-link accessors
#
#  IsaacLab's filtered contact reporting ("force_matrix_w") only works when
#  the ContactSensor's prim_path resolves to a SINGLE body per environment:
#  "the reporting of filtered contact forces is only possible as one-to-many
#  [...] only one sensor body in an environment can be filtered against
#  multiple bodies in that environment" (isaaclab/sensors/contact_sensor/
#  contact_sensor.py). A multi-body sensor with a filter does not raise — it
#  silently reports zeros forever, which is exactly how every force-based
#  reward term in this project ended up flat at 0.0.
#
#  So the scene declares one single-body, Apple-filtered sensor per hand
#  link, named "<base>_<short link name>" (see AppleGraspSceneCfg's
#  contact_forces_* block). Everything that wants apple-filtered force goes
#  through the two accessors below instead of touching force_matrix_w, so
#  the naming convention and the [N, 1, 1, 3] indexing live in one place.
#
#  The unfiltered whole-hand sensor (`contact_forces` itself) still exists
#  and is still the right thing for net_forces_w consumers — see
#  fingertip_contact_forces/_normals below, tree_utils and run_env.py.
# ─────────────────────────────────────────────

#: Every right-hand link that has its own Apple-filtered contact sensor.
RIGHT_HAND_CONTACT_LINKS = [
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
]


def link_sensor_name(base: str, link_name: str) -> str:
    """Scene key of the single-body, Apple-filtered sensor for `link_name`.

    `base` is the sensor-group name the config passes around as
    SceneEntityCfg("contact_forces"), so callers keep taking a sensor_cfg
    and it keeps meaning something: it selects the sensor FAMILY, and this
    picks the member of it that watches one link.

    "contact_forces" + "right_hand_thumb_0_link" -> "contact_forces_thumb_0"
    """
    short = link_name.removeprefix("right_hand_").removesuffix("_link")
    return f"{base}_{short}"


def _apple_contact_sensors(env: ManagerBasedRLEnv, base: str) -> dict[str, ContactSensor]:
    """Lazily cache {link_name: ContactSensor} for one sensor family."""
    cache = getattr(env, "_apple_contact_sensors", None)
    if cache is None:
        cache = env._apple_contact_sensors = {}
    group = cache.get(base)
    if group is None:
        group = cache[base] = {
            link: env.scene[link_sensor_name(base, link)]
            for link in RIGHT_HAND_CONTACT_LINKS
        }
    return group


def apple_force_vec_per_link(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    link_names: Sequence[str],
) -> torch.Tensor:
    """Apple-filtered contact force VECTOR (world frame, N) per link.

    Shape: [num_envs, len(link_names), 3], rows in `link_names` order.
    """
    group = _apple_contact_sensors(env, sensor_cfg.name)
    # Each sensor is single-body and has exactly one filter (the Apple),
    # so force_matrix_w is [N, 1, 1, 3].
    return torch.stack(
        [group[link].data.force_matrix_w[:, 0, 0, :] for link in link_names], dim=1
    )


def apple_force_per_link(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    link_names: Sequence[str],
) -> torch.Tensor:
    """Apple-filtered contact force MAGNITUDE (N) per link.

    Shape: [num_envs, len(link_names)], columns in `link_names` order.
    """
    return torch.norm(apple_force_vec_per_link(env, sensor_cfg, link_names), dim=-1)


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
#  Palm contact force (reach phase) — a single raw scalar, NOT calibration-
#  robust like the grasp-phase block below. The reach task doesn't need to
#  generalize across unknown real-hardware calibration the way the grasp
#  phase's multi-finger force-closure comparison does (see that block's own
#  note); this is a simple ground-truth-scale signal, matching what
#  reach_rewards.penalty_excess_palm_force and terminations_and_events.
#  reach_success's optional touch-gate compute against directly.
# ─────────────────────────────────────────────

# The palm only. It used to include the base segment of each digit (thumb_0,
# index_0, middle_0), but then the side of the thumb pressed against the
# side of the apple counted as a "palm touch" and was rewarded; the reach
# goal is the apple resting gently on the palm face itself.
_PALM_CONTACT_LINKS = [
    "right_hand_palm_link",
]


def palm_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Apple-filtered contact force magnitude on the palm link
    (_PALM_CONTACT_LINKS) -- "palm tactile sensing" for the reach phase.
    Deliberately excludes every digit segment: the reach should end with
    the apple resting on the palm face in the middle of an open hand, not
    caught on the thumb or fingers.

    Apple-filtered (via apple_force_per_link, i.e. the per-link filtered
    sensors), same reasoning as reach_rewards.penalty_apple_contact_force:
    self-collision or tree contact must not leak into this reading.

    Shape: [num_envs, 1]
    """
    return apple_force_per_link(env, sensor_cfg, _PALM_CONTACT_LINKS).sum(
        dim=-1, keepdim=True
    )


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
#  policy as an observation. The one sanctioned exception is the grip
#  DETECTOR (grasp_rewards._pad_state): it gates the arm, so it has to be
#  exactly what the real hand could compute, i.e. these sensed values.
#
#  Sensed also means PAD-ONLY: _sensed_tactile_forces models where on the
#  link a real taxel can feel (see contact_geometry), whereas the rewards'
#  safety terms read every face of every link.
# ─────────────────────────────────────────────

def _sensed_tactile_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    all_apples: bool = False,
) -> torch.Tensor:
    """What each real taxel would read: apple-filtered force magnitude on its
    link, times how much of that contact lands on the PAD, times the
    per-env per-taxel calibration gain (env._tactile_gain).

    The pad factor is contact_geometry.pad_face_weight on the contact's
    angle off the pad normal: a real Dex3-1 taxel sits on the pad face, so
    a contact on the link's back reads nothing and one on its side edge
    reads partly or not at all -- where the sim's per-body sensor would
    report both in full. The footprint (full/cut angles) is randomized per
    episode and per taxel (domain_randomization.randomize_tactile_footprint)
    because how far the real pads sense toward their edges is unmeasured;
    this matters most for the tripod grasp, whose index/middle contacts sit
    on the pads' inner edges.

    Apple-filtered via the per-link filtered sensors, not net_forces_w —
    same reasoning as reach_rewards.penalty_apple_contact_force:
    self-collision or tree contact must not leak into a tactile reading.

    Note that several taxels share one sim link (palm_0/1/2 all map to
    right_hand_palm_link), so those columns are identical up to gain and
    footprint — flagged on the monitor plot too (scripts/tactile_monitor.py).

    all_apples: on the multi-apple tree (apple_grasp_tree_env_cfg), the
    per-link sensors are filtered against EVERY apple, and a real pad reads
    any fruit pressing on it -- so sum over them, each contact weighted by
    its own pad factor (a neighbour against the BACK of a finger still
    reads nothing). Default: the target only. Rewards and the grip latch
    (grasp_rewards._pad_state) keep the default.

    Shape: [num_envs, 9] (tactile_layout.TAXELS order).
    """
    import scripts.tactile_layout as tactile_layout
    from .contact_geometry import (
        apple_contact_faces, pad_face_weight, TACTILE_FULL_DEG_DEFAULT, TACTILE_CUT_DEG_DEFAULT,
    )

    links = [t.link_name for t in tactile_layout.TAXELS]
    full = getattr(env, "_tactile_full_deg", TACTILE_FULL_DEG_DEFAULT)
    cut = getattr(env, "_tactile_cut_deg", TACTILE_CUT_DEG_DEFAULT)
    n_filters = 1
    if all_apples:
        n_filters = _apple_contact_sensors(env, sensor_cfg.name)[links[0]].data.force_matrix_w.shape[2]
    sensed = 0.0
    for m in range(n_filters):
        force, cos_pad = apple_contact_faces(
            env, sensor_cfg, links, object_name="apple" if m == 0 else f"apple_{m}", filter_index=m,
        )  # [N, 9, 3], [N, 9]
        sensed = sensed + torch.norm(force, dim=-1) * pad_face_weight(cos_pad, full, cut)

    if not hasattr(env, "_tactile_gain"):
        env._tactile_gain = torch.ones(env.num_envs, sensed.shape[-1], device=env.device)

    sensed = sensed * env._tactile_gain
    # Dead band / noise / dead taxels: only in tasks that run
    # probe_grasp.reset_tactile_realism (AppleGraspSingle-v2); others unchanged.
    if getattr(env, "_tactile_realism", None) is not None:
        from .probe_grasp import apply_tactile_realism
        sensed = apply_tactile_realism(env, sensed)
    return sensed


def _taxel_finger_groups(env: ManagerBasedRLEnv) -> dict[str, list[int]]:
    """Lazily cache, per env, the taxel indices belonging to each finger
    group (thumb/index/middle) — shared by the balance/level observations
    below and by grasp_rewards.py's ground-truth equivalents."""
    import scripts.tactile_layout as tactile_layout

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
    all_apples: bool = False,
) -> torch.Tensor:
    """Per-taxel temporal slip signal: (force_now - force_prev) / step_dt,
    calibration-gained. The real physical signature of slip (a sudden force
    change despite a steady grip command) — a derivative, so a UNIFORM
    calibration gain cancels out of it entirely; only the independent
    per-taxel gain randomization actually stresses this.

    Shape: [num_envs, 9]
    """
    sensed = _sensed_tactile_forces(env, sensor_cfg, all_apples)

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
    all_apples: bool = False,
) -> torch.Tensor:
    """Per-finger-group (thumb/index/middle) mean force, normalized by
    deviation from the cross-finger mean — deliberately NOT absolute
    magnitude, so a uniform miscalibration cancels and only genuine
    imbalance shows up.

    Shape: [num_envs, 3]
    """
    sensed = _sensed_tactile_forces(env, sensor_cfg, all_apples)
    idx = _taxel_finger_groups(env)
    per_finger = torch.stack(
        [sensed[:, idx[f]].mean(dim=-1) for f in ("thumb", "index", "middle")], dim=-1
    )  # [N, 3]
    mean_f = per_finger.mean(dim=-1, keepdim=True)
    return (per_finger - mean_f) / (mean_f.abs() + 1e-3)


def tactile_force_level(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    all_apples: bool = False,
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
    sensed = _sensed_tactile_forces(env, sensor_cfg, all_apples)
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


def object_pos_in_body_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ref_local: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Object position expressed IN the body link's own rotating frame,
    minus ref_local -- i.e. the error vector from a reference point fixed
    in that link to the object.

    Distinct from object_pos_relative_to_body above, which is only
    translated (world-axis aligned) and so mixes the palm's orientation
    into every component. For "is the apple in the middle of the hand" the
    palm-frame vector is the natural signal: it is exactly the quantity
    grasp_rewards.reward_apple_centered scores, so the policy can see the
    error it is being paid to close.

    Shape: [num_envs, 3]
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = robot_cfg.body_ids[0]
    quat = robot.data.body_quat_w[:, body_id]
    delta = obj.data.root_pos_w - robot.data.body_pos_w[:, body_id]
    local = quat_apply(quat_conjugate(quat), delta)
    ref = torch.tensor(ref_local, device=env.device, dtype=local.dtype)
    return local - ref


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
