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
) -> torch.Tensor:
    """[N] in [0, 1]: exp(-a^2 / 2 std^2), a = angle between the finger axis
    and world up (fruit above shoulder + switch_dz) or world down (below);
    within +-band of the switch height either way pays. Gated in by the
    palm's distance to the goal like reward_palm_toward_stem (0 beyond
    gate_dist), so it is strongest on the final approach."""
    robot = env.scene[robot_cfg.name]
    q = robot.data.body_quat_w[:, robot_cfg.body_ids[0]]
    f = quat_apply(q, torch.tensor(finger_axis, device=env.device).expand(env.num_envs, 3))
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
    robot = env.scene["robot"]
    if not hasattr(env, "_eo_ids"):
        names = robot.data.body_names
        env._eo_ids = (names.index(elbow_body), names.index(shoulder_body), names.index(torso_body))
    e, s, t = env._eo_ids
    qt = robot.data.body_quat_w[:, t]
    y_torso = quat_apply(qt, torch.tensor((0.0, 1.0, 0.0), device=env.device).expand(env.num_envs, 3))
    y_torso[:, 2] = 0.0
    y_torso = y_torso / y_torso.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    out = -((robot.data.body_pos_w[:, e] - robot.data.body_pos_w[:, s]) * y_torso).sum(-1)
    return ((out - margin).clamp(min=0.0) / ref) ** 2


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
) -> None:
    """mode="interval" every step (interval_range_s (0, 0), global time):
    latch once the palm point is within level_tol of the PERCEIVED apple
    centre's height (apple_pos_measured, as on the robot), then ramp the
    hand targets open -> closed over ramp_s. write_targets=False only
    advances the phase (the grasp task: its hand action term reads
    env._hand_phase during the reach phase and owns the targets)."""
    ids = _hand_ids(env, open_targets, closed_targets)
    robot = env.scene[robot_cfg.name]
    p = rr._palm_ref_pos(env, robot_cfg, palm_local_offset)
    level = (p[:, 2] - apple_pos_measured(env, object_cfg)[:, 2]).abs() < level_tol
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
