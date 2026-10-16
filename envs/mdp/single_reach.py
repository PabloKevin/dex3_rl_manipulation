# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Single-policy reach without touch (AppleReachSingle-v1,
envs/apple_reach_single_env_cfg.py): the pieces beyond the pregrasp reach.

  - set_approach_origin: the approach direction every heading / goal term
    uses (reach_rewards._approach_dir) becomes the horizontal line from the
    waist axis to the apple instead of the (unobservable) stem. The waist
    yaw axis goes through the pelvis origin, so the direction does not move
    when the torso turns.
  - reward_fingers_vertical: the hand's long axis (palm local +X, wrist ->
    fingertips) vertical, fingers DOWN (thumb above them) for fruit below
    shoulder + switch_dz, fingers UP (thumb below) above it.
  - penalty_elbow_out: the right elbow out beyond the shoulder, sideways in
    the torso's own frame.
  - apple measurement: the apple centre as the robot's perception gives it,
    taken ONCE at the episode start (true + a per-episode bias: calibration
    / surface->centre radius error, larger along the viewing depth) and held
    for the whole episode, as sim2real/toy_reach.py does (detections sampled
    at home, then self.apple fixed: near the fruit the hand hides it from
    the camera). It does not follow the fruit if it is pushed. The policy
    inputs and the hand schedule read it; rewards / success keep the truth.
    The neighbours too (neighbour_ring_measured): seen in the same detection,
    same bias, held.
  - hand schedule: the hand starts open and closes to a partial cup once the
    palm point is level with the fruit (latched per episode, ramped), with
    the phase in [0, 1] as an observation. Everything it reads (palm point
    from forward kinematics, apple height from the detection) exists on the
    robot too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from . import reach_rewards as rr

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Approach direction
# ─────────────────────────────────────────────

def set_approach_origin(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None, body: str = "pelvis") -> None:
    """mode="startup": reach_rewards._approach_dir measures from `body`
    (horizontal, to the apple centre) instead of along the stem."""
    env._approach_from_body = env.scene["robot"].data.body_names.index(body)


# ─────────────────────────────────────────────
#  Perceived apple position
# ─────────────────────────────────────────────

def _depth_frame(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, origin_body: str):
    """Unit vectors [N, 3] (depth: horizontal origin -> apple, lateral, up)."""
    robot = env.scene["robot"]
    if not hasattr(env, "_meas_origin_id"):
        env._meas_origin_id = robot.data.body_names.index(origin_body)
    d = env.scene[object_cfg.name].data.root_pos_w - robot.data.body_pos_w[:, env._meas_origin_id]
    d[:, 2] = 0.0
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    up = torch.zeros_like(d)
    up[:, 2] = 1.0
    return d, torch.cross(up, d, dim=-1), up


def reset_apple_measurement(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    origin_body: str = "pelvis",
    depth_std: float = 0.010,
    depth_clip: float = 0.025,
    cross_std: float = 0.005,
    cross_clip: float = 0.015,
) -> None:
    """mode="reset", after the tree / apple placement: this episode's
    perceived apple centre, env._apple_meas [N, 3] (world), fixed until the
    next reset = the placed apple + a bias, Gaussian and clipped, depth_std
    along the horizontal origin -> apple direction (the camera's depth;
    where a wrong radius in the surface -> centre step lands), cross_std
    sideways and vertically. No per-step noise: the robot averages its
    detections at home (toy_reach --detect_s) and then keeps one point."""
    N, dev = env.num_envs, env.device
    true = env.scene[object_cfg.name].data.root_pos_w
    if getattr(env, "_apple_meas", None) is None:
        env._apple_meas = true.clone()
    n = len(env_ids)
    bd = (torch.randn(n, device=dev) * depth_std).clamp(-depth_clip, depth_clip)
    bc = (torch.randn(n, 2, device=dev) * cross_std).clamp(-cross_clip, cross_clip)
    d, lat, up = _depth_frame(env, object_cfg, origin_body)
    bias = bd.unsqueeze(-1) * d[env_ids] + bc[:, :1] * lat[env_ids] + bc[:, 1:] * up[env_ids]
    env._apple_meas[env_ids] = true[env_ids] + bias
    # the neighbours, seen in the same detection: positions + hanging mask at
    # the start, shifted by the same bias (one camera calibration), then held
    if hasattr(env, "_tree"):
        from .tree_reach import _neighbour_state

        q, hanging = _neighbour_state(env)
        if getattr(env, "_neigh_meas", None) is None:
            env._neigh_meas = q.clone()
            env._neigh_hanging_meas = hanging.clone()
        env._neigh_meas[env_ids] = q[env_ids] + bias.unsqueeze(1)
        env._neigh_hanging_meas[env_ids] = hanging[env_ids]


def apple_pos_measured(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N, 3] this episode's perceived apple centre (world), or the true
    position before the first reset."""
    m = getattr(env, "_apple_meas", None)
    return env.scene[object_cfg.name].data.root_pos_w if m is None else m


def neighbour_ring_measured(env: "ManagerBasedRLEnv", n_sectors: int = 8, max_range: float = 0.10,
                            shoulder_body: str = "right_shoulder_pitch_link") -> torch.Tensor:
    """[N, n_sectors] tree_reach.neighbour_clearance_ring from this episode's
    perceived target and neighbours (reset_apple_measurement), not the live
    fruit: a neighbour pushed or knocked off later still reads where it was
    seen. Before the first reset: the live ring."""
    from .tree_reach import clearance_ring_from, neighbour_clearance_ring

    q = getattr(env, "_neigh_meas", None)
    if q is None:
        return neighbour_clearance_ring(env, n_sectors, max_range, shoulder_body)
    return clearance_ring_from(env, apple_pos_measured(env, SceneEntityCfg("apple")), q, env._neigh_hanging_meas,
                               n_sectors, max_range, shoulder_body)


def apple_pos_palm_measured(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                            local_offset: Sequence[float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
    """[N, 3] observations.object_pos_relative_to_body with the perceived
    apple centre: apple - palm reference point, world (pelvis) axes."""
    return apple_pos_measured(env, object_cfg) - rr._palm_ref_pos(env, robot_cfg, local_offset)


def apple_pos_pocket_measured(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                              ref_local: Sequence[float] = (0.0, 0.0, 0.0)) -> torch.Tensor:
    """[N, 3] observations.object_pos_in_body_frame with the perceived apple
    centre: in the palm link's own frame, minus ref_local (the grasp task's
    fruit-in-pocket input)."""
    from isaaclab.utils.math import quat_conjugate

    robot = env.scene[robot_cfg.name]
    q = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    delta = apple_pos_measured(env, object_cfg) - robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    return quat_apply(quat_conjugate(q), delta) - torch.tensor(ref_local, device=env.device, dtype=delta.dtype)


# ─────────────────────────────────────────────
#  Hand orientation and arm posture
# ─────────────────────────────────────────────

def finger_sign(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, shoulder_body: str,
                switch_dz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """([N] +1 fingers up (apple centre above shoulder + switch_dz) / -1 down,
    [N] apple centre height above the shoulder)."""
    robot = env.scene["robot"]
    if not hasattr(env, "_fv_shoulder_id"):
        env._fv_shoulder_id = robot.data.body_names.index(shoulder_body)
    dz = env.scene[object_cfg.name].data.root_pos_w[:, 2] - robot.data.body_pos_w[:, env._fv_shoulder_id, 2]
    return torch.where(dz > switch_dz, 1.0, -1.0), dz


def reward_fingers_vertical(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    shoulder_body: str = "right_shoulder_pitch_link",
    switch_dz: float = 0.15,
    band: float = 0.03,
    std_rad: float = 0.4,
    gate_dist: float = 0.55,
    finger_axis: Sequence[float] = rr.PALM_FINGER_AXIS,
    roll_only: bool = False,
) -> torch.Tensor:
    """[N] in [0, 1]: exp(-a^2 / 2 std^2), a = angle between the finger axis
    and world up (fruit above shoulder + switch_dz) or world down (below);
    within +-band of the switch height either way pays. Gated in by the
    palm's distance to the goal like reward_palm_toward_stem (0 beyond
    gate_dist), so it is strongest on the final approach.

    roll_only (AppleReachSingle-v2): a is measured in the plane across the
    approach (lateral, up) only -- the hand's roll about the approach line.
    Leaning the fingers forward / back along the approach is free: the
    tilt that puts all fingertips at one depth (reward_fingertip_depth_equal,
    ~30 deg forward with the cup) must not cost anything here."""
    robot = env.scene[robot_cfg.name]
    q = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    f = quat_apply(q, torch.tensor(finger_axis, device=env.device).expand(env.num_envs, 3))
    if roll_only:
        d = rr._approach_dir(env, object_cfg)
        up = torch.zeros_like(d)
        up[:, 2] = 1.0
        lat = torch.cross(up, d, dim=-1)
        fl, fu = (f * lat).sum(-1).abs(), f[:, 2]
        a_up, a_down = torch.atan2(fl, fu), torch.atan2(fl, -fu)
        r_up = torch.exp(-a_up ** 2 / (2.0 * std_rad ** 2))
        r_down = torch.exp(-a_down ** 2 / (2.0 * std_rad ** 2))
    else:
        fz = f[:, 2].clamp(-1.0, 1.0)
        r_up = torch.exp(-torch.acos(fz) ** 2 / (2.0 * std_rad ** 2))
        r_down = torch.exp(-torch.acos(-fz) ** 2 / (2.0 * std_rad ** 2))
    sign, dz = finger_sign(env, object_cfg, shoulder_body, switch_dz)
    r = torch.where(sign > 0, r_up, r_down)
    r = torch.where((dz - switch_dz).abs() < band, torch.maximum(r_up, r_down), r)
    d = rr._palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    return r * (1.0 - d / gate_dist).clamp(0.0, 1.0)


def penalty_elbow_out(
    env: "ManagerBasedRLEnv",
    elbow_body: str = "right_elbow_link",
    shoulder_body: str = "right_shoulder_pitch_link",
    torso_body: str = "torso_link",
    margin: float = 0.05,
    ref: float = 0.10,
) -> torch.Tensor:
    """[N] the right elbow sideways OUT (torso -y) beyond the shoulder by more
    than `margin`, (excess / ref)^2, in the torso's yaw frame: keeps the arm
    close to the body (shoulder torque, and less arm in the canopy)."""
    out = _elbow_out_dist(env, elbow_body, shoulder_body, torso_body)
    return ((out - margin).clamp(min=0.0) / ref) ** 2


def _elbow_out_dist(env: "ManagerBasedRLEnv", elbow_body: str, shoulder_body: str, torso_body: str) -> torch.Tensor:
    """[N] how far (m) the right elbow sits sideways OUT (torso -y) of the
    shoulder, in the torso's yaw frame; < 0 inward, toward the left arm."""
    robot = env.scene["robot"]
    if not hasattr(env, "_eo_ids"):
        names = robot.data.body_names
        env._eo_ids = (names.index(elbow_body), names.index(shoulder_body), names.index(torso_body))
    e, s, t = env._eo_ids
    qt = robot.data.body_quat_w[:, t]
    y_torso = quat_apply(qt, torch.tensor((0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 3))
    y_torso[:, 2] = 0.0
    y_torso = y_torso / y_torso.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return -((robot.data.body_pos_w[:, e] - robot.data.body_pos_w[:, s]) * y_torso).sum(-1)


def penalty_elbow_in(
    env: "ManagerBasedRLEnv",
    elbow_body: str = "right_elbow_link",
    shoulder_body: str = "right_shoulder_pitch_link",
    torso_body: str = "torso_link",
    margin: float = 0.0,
    ref: float = 0.05,
) -> torch.Tensor:
    """[N] the right elbow sideways IN (torso +y, toward the left arm) beyond
    the shoulder by more than `margin`, (excess / ref)^2, in the torso's yaw
    frame -- penalty_elbow_out's mirror. User, 2026-10-16: the elbow must
    point out of the torso (natural, away from the body: less self-collision);
    model_1999 of the first v2 run held it up to 7 cm inward."""
    inward = -_elbow_out_dist(env, elbow_body, shoulder_body, torso_body)
    return ((inward - margin).clamp(min=0.0) / ref) ** 2


# ─────────────────────────────────────────────
#  Hand schedule: open -> partial cup once level with the fruit
# ─────────────────────────────────────────────

def _hand_ids(env: "ManagerBasedRLEnv", open_targets: dict, closed_targets: dict):
    if not hasattr(env, "_hs_ids"):
        robot = env.scene["robot"]
        names = list(open_targets)
        if set(names) != set(closed_targets):
            raise ValueError("hand schedule: open and closed targets must name the same joints")
        env._hs_ids = torch.tensor([robot.joint_names.index(n) for n in names], device=env.device)
        env._hs_open = torch.tensor([open_targets[n] for n in names], device=env.device)
        env._hs_closed = torch.tensor([closed_targets[n] for n in names], device=env.device)
        env._hand_phase = torch.zeros(env.num_envs, device=env.device)
        env._hand_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._hs_ids


def reset_hand_schedule(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, open_targets: dict,
                        closed_targets: dict) -> None:
    """mode="reset", after hold_default_targets: hand open, phase 0, unlatched."""
    ids = _hand_ids(env, open_targets, closed_targets)
    env._hand_phase[env_ids] = 0.0
    env._hand_latched[env_ids] = False
    robot = env.scene["robot"]
    tgt = robot.data.joint_pos_target[env_ids].clone()
    tgt[:, ids] = env._hs_open
    robot.set_joint_position_target(tgt, env_ids=env_ids)


def step_hand_schedule(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    open_targets: dict,
    closed_targets: dict,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_local_offset: Sequence[float],
    level_tol: float = 0.03,
    ramp_s: float = 0.3,
    write_targets: bool = True,
    trigger: str = "level",
    near: float = 0.10,
) -> None:
    """mode="interval" every step (interval_range_s (0, 0), global time):
    latch once the palm point is within level_tol of the PERCEIVED apple
    centre's height (apple_pos_measured, as on the robot), then ramp the
    hand targets open -> closed over ramp_s. write_targets=False only
    advances the phase (the grasp task: its hand action term reads
    env._hand_phase during the reach phase and owns the targets).
    trigger="distance" (AppleReachSingle-v2): latch once the palm point is
    within `near` (3D) of the perceived apple centre instead."""
    ids = _hand_ids(env, open_targets, closed_targets)
    robot = env.scene[robot_cfg.name]
    p = rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    if trigger == "distance":
        level = (p - apple_pos_measured(env, object_cfg)).norm(dim=-1) < near
    elif trigger == "level":
        level = (p[:, 2] - apple_pos_measured(env, object_cfg)[:, 2]).abs() < level_tol
    else:
        raise ValueError(f"step_hand_schedule: trigger must be 'level' or 'distance', got {trigger!r}")
    env._hand_latched |= level
    env._hand_phase = torch.where(env._hand_latched, (env._hand_phase + env.step_dt / ramp_s).clamp(max=1.0),
                                  env._hand_phase)
    if not write_targets:
        return
    ph = env._hand_phase.unsqueeze(-1)
    tgt = robot.data.joint_pos_target.clone()
    tgt[:, ids] = env._hs_open + ph * (env._hs_closed - env._hs_open)
    robot.set_joint_position_target(tgt)


def hand_phase_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 1] hand schedule phase: 0 open, 1 partial cup (what was commanded,
    known on the robot)."""
    p = getattr(env, "_hand_phase", None)
    if p is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return p.unsqueeze(-1)


# ─────────────────────────────────────────────
#  AppleReachSingle-v2: the apple ENCLOSED in the hand
# ─────────────────────────────────────────────
# User, 2026-10-15: the reach must end with the apple inside the hand (the
# grasp pocket, palm touching it), so the grasp only closes; a push of ~1.5
# cm is acceptable, stopping centimetres short is a failure.

#: fingertip points: link + offset in its own frame (mesh extents of the
#: Dex3 distal links: index / middle along local +x, thumb along local +y)
FINGERTIPS = (
    ("right_hand_index_1_link", (0.048, 0.0, 0.0)),
    ("right_hand_middle_1_link", (0.048, 0.0, 0.0)),
    ("right_hand_thumb_2_link", (0.0, 0.045, 0.0)),
)


def set_reach_goal_v2(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None, touch_gap: float = 0.016,
                      push_in: float = 0.015, short_tol: float = 0.008, lateral_tol: float = 0.02,
                      max_disp: float = 0.02, heading_horizontal: bool = True) -> None:
    """mode="startup": switch reach_rewards.reach_neighbourhood (hold, the
    success termination, the grasp task's converged hand-off) to `enclosed`,
    and reach_rewards._palm_normal_angle to the horizontal heading only (the
    palm may tilt; the tilt terms judge that)."""
    env._enclosure_goal = dict(touch_gap=touch_gap, push_in=push_in, short_tol=short_tol,
                               lateral_tol=lateral_tol, max_disp=max_disp)
    env._heading_horizontal = bool(heading_horizontal)


def target_displacement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] how far (m) the target is from where it hangs (its tree spot); 0
    without a tree. pregrasp_rewards.penalty_target_push without the dead band."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    T = env._tree
    rows = torch.arange(env.num_envs, device=env.device)
    spot = (T.occupant == 0).float().argmax(dim=1)
    rest = T.spot_pos[rows, spot] + env.scene.env_origins
    return (T.objs[0].data.root_pos_w - rest).norm(dim=-1)


def enclosure_state(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
                    palm_local_offset: Sequence[float], palm_normal_axis: Sequence[float],
                    touch_gap: float = 0.016) -> tuple[torch.Tensor, torch.Tensor]:
    """(gap [N], lateral [N]) of the TRUE apple centre against the palm
    point: gap along the palm normal beyond touching (radius + touch_gap: the
    palm collider starts 16 mm out from the palm point), < 0 pressed in;
    lateral = its distance from the normal line (off-centre across the palm).
    radius + touch_gap out along the normal is the grasp pocket
    (GRASP_CENTER_LOCAL, 57 mm for the 40 mm apple)."""
    apple = env.scene[object_cfg.name].data.root_pos_w
    radius = getattr(env, "_apple_radius", None)
    if not isinstance(radius, torch.Tensor):
        radius = torch.full((env.num_envs,), rr.DEFAULT_APPLE_RADIUS, device=env.device)
    n = rr._palm_normal_world(env, robot_cfg, palm_normal_axis)
    rel = apple - rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    along = (rel * n).sum(-1)
    return along - (radius + touch_gap), (rel - along.unsqueeze(-1) * n).norm(dim=-1)


def enclosed(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg,
             palm_local_offset: Sequence[float], palm_normal_axis: Sequence[float]) -> torch.Tensor:
    """[N] bool, env._enclosure_goal's test: the apple in the pocket (gap in
    [-push_in, short_tol], lateral < lateral_tol) and the target not dragged
    more than max_disp from where it hangs."""
    g = env._enclosure_goal
    gap, lat = enclosure_state(env, robot_cfg, object_cfg, palm_local_offset, palm_normal_axis, g["touch_gap"])
    return ((gap >= -g["push_in"]) & (gap <= g["short_tol"]) & (lat < g["lateral_tol"])
            & (target_displacement(env) < g["max_disp"]))


def reward_enclosed(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    touch_gap: float = 0.016,
    push_in: float = 0.015,
    short_std: float = 0.01,
    lateral_std: float = 0.015,
) -> torch.Tensor:
    """[N] in [0, 1], dense: along the normal 1 from touching to push_in
    pressed in, a gaussian (short_std) short of touching -- 1 cm short 0.61,
    2 cm 0.14, 3 cm 0.01 -- and beyond push_in; times a gaussian on the
    lateral offset. True apple (privileged), as every goal term."""
    gap, lat = enclosure_state(env, robot_cfg, object_cfg, palm_local_offset, palm_normal_axis, touch_gap)
    excess = torch.where(gap > 0.0, gap, (-push_in - gap).clamp(min=0.0))
    return torch.exp(-excess ** 2 / (2.0 * short_std ** 2)) * torch.exp(-lat ** 2 / (2.0 * lateral_std ** 2))


def fingertip_points(env: "ManagerBasedRLEnv", tips=FINGERTIPS) -> torch.Tensor:
    """[N, T, 3] world fingertip points (FINGERTIPS)."""
    robot = env.scene["robot"]
    if not hasattr(env, "_tip_ids"):
        env._tip_ids = [robot.data.body_names.index(n) for n, _ in tips]
        env._tip_off = torch.tensor([o for _, o in tips], device=env.device)
    q = robot.data.body_quat_w[:, env._tip_ids]                         # [N, T, 4]
    off = env._tip_off.unsqueeze(0).expand(env.num_envs, -1, -1)
    return robot.data.body_pos_w[:, env._tip_ids] + quat_apply(q.reshape(-1, 4), off.reshape(-1, 3)).view_as(off)


def fingertip_depth_spread(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N] max - min of the fingertips' depth along the horizontal approach
    line (pelvis -> apple): 0 = every tip as far from the torso (and from
    the trunk behind the fruit) as the others."""
    depth = (fingertip_points(env) * rr._approach_dir(env, object_cfg).unsqueeze(1)).sum(-1)
    return depth.amax(-1) - depth.amin(-1)


def reward_fingertip_depth_equal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    std: float = 0.015,
    gate_dist: float = 0.25,
) -> torch.Tensor:
    """[N] in [0, 1]: exp(-(spread / std)^2), fingertip_depth_spread, gated
    in over the last gate_dist to the goal. Measured (pinocchio, URDF): with
    the palm normal level the thumb tip stands 6.9 cm (cup) / 9.6 cm (open)
    ahead of the index / middle tips -- the first thing to hit the trunk;
    leaning the fingers ~30 deg forward along the approach (palm normal
    tilted the same) brings all three to one depth (cup: 0.2 cm)."""
    spread = fingertip_depth_spread(env, object_cfg)
    d = rr._palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    return torch.exp(-(spread / std) ** 2) * (1.0 - d / gate_dist).clamp(0.0, 1.0)


def reward_palm_tilt_band(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    standoff: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    free_rad: float = 0.70,
    std_rad: float = 0.2,
    gate_dist: float = 0.55,
) -> torch.Tensor:
    """[N] reach_rewards.reward_palm_lateral with a free band: 1 up to
    free_rad of tilt (palm normal vs horizontal), a gaussian beyond; gated
    the same way. The equal-fingertip-depth pose needs ~30 deg."""
    tilt = rr._palm_tilt(env, robot_cfg, palm_normal_axis)
    r = torch.exp(-(tilt - free_rad).clamp(min=0.0) ** 2 / (2.0 * std_rad ** 2))
    d = rr._palm_target_dist(env, robot_cfg, object_cfg, standoff, palm_local_offset, palm_normal_axis)
    return r * (1.0 - d / gate_dist).clamp(0.0, 1.0)


def penalty_target_contact_v2(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    tip_links: Sequence[str],
    soft_links: Sequence[str],
    tip_cap: float = 0.5,
    soft_free: float = 2.0,
    soft_ref: float = 2.0,
) -> torch.Tensor:
    """[N] apple contact that is not the enclosing touch: any fingertip
    contact (per link, saturating at tip_cap: a graze costs a little, a touch
    the full weight) + palm / base-phalanx force beyond soft_free N (summed,
    / soft_ref, at most 1). A light palm touch, even a ~1.5 cm push on the
    swinging stem, is free."""
    from .observations import apple_force_per_link
    tips = apple_force_per_link(env, sensor_cfg, tip_links)
    soft = apple_force_per_link(env, sensor_cfg, soft_links).sum(-1)
    return (tips.clamp(max=tip_cap) / tip_cap).sum(-1) + ((soft - soft_free).clamp(min=0.0) / soft_ref).clamp(max=1.0)


def reach_v2_metrics(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    ema: float = 0.05,
) -> dict:
    """Curriculum-term logger (it changes nothing): runs at the start of
    every reset, before the scene resets, so it sees the END state of the
    episodes ending. Running means, split by whether the target was a
    stretch apple (drawn beyond the feasible map, Envelope.stretch_frac):
    success, target knocked off, final gap / lateral (cm), target displaced
    (cm), fingertip depth spread (cm), palm tilt (deg). Logged by Isaac Lab
    as Curriculum/<term>/<key>."""
    st = getattr(env, "_v2_metrics", None)
    if st is None:
        st = env._v2_metrics = {}
    if not hasattr(env, "_tree") or not hasattr(env, "termination_manager") or env.common_step_counter < 1:
        return dict(st)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return dict(st)
    tm = env.termination_manager
    succ = tm.get_term("reach_success")[ids].float() if "reach_success" in tm.active_terms else torch.zeros(len(ids), device=env.device)
    knocked = env._tree.target_stem.broken[ids].float()
    gap, lat = enclosure_state(env, robot_cfg, object_cfg, palm_local_offset, palm_normal_axis)
    vals = {
        "success": succ,
        "knocked": knocked,
        "gap_cm": gap[ids] * 100.0,
        "lateral_cm": lat[ids] * 100.0,
        "displaced_cm": target_displacement(env)[ids] * 100.0,
        "tip_spread_cm": fingertip_depth_spread(env, object_cfg)[ids] * 100.0,
        "tilt_deg": torch.rad2deg(rr._palm_tilt(env, robot_cfg, palm_normal_axis))[ids],
    }
    stretch = getattr(env, "_target_stretch", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))[ids]
    for group, mask in (("feas", ~stretch), ("stretch", stretch)):
        if not bool(mask.any()):
            continue
        for k, v in vals.items():
            m = float(v[mask].mean())
            key = f"{group}_{k}"
            st[key] = m if key not in st else (1.0 - ema) * st[key] + ema * m
    share = float(stretch.float().mean())
    st["stretch_share"] = share if "stretch_share" not in st else (1.0 - ema) * st["stretch_share"] + ema * share
    return dict(st)
