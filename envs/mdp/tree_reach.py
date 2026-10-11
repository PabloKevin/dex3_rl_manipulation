# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Reach on a multi-apple tree (envs/apple_reach_tree_env_cfg.py): the
observations, rewards and termination that the single-apple reach task did
not need.

Observations (both available on the real robot):
  - neighbour_clearance_ring: 8 numbers, how much room there is around the
    target apple, by direction -- see its docstring.
  - palm_contact_force_any: palm pressure from ANY contact (the real palm
    sensor cannot tell the target from its neighbour).
Rewards (privileged, sim only):
  - neighbour push / knock-off, from the neighbours' displacement on their
    stems and their stems breaking;
  - smoothness of the COMMANDED arm trajectory -- its acceleration, and how
    far the policy asks beyond what the rate limiter can deliver (the
    bang-bang that made the old reach jerky).
Termination: the target knocked off its stem.

All of them read env._tree (envs/mdp/apple_tree.py).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from . import apple_tree as at

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _neighbour_state(env: "ManagerBasedRLEnv"):
    """Neighbour centres [N, K, 3] (world) and a [N, K] mask of those still
    hanging on their stems (entities apple_1.., i.e. not the target)."""
    T = env._tree
    pos = torch.stack([o.data.root_pos_w for o in T.objs[1:]], dim=1)
    hanging = torch.stack([~s.broken & s.active for s in T.stems[1:]], dim=1)
    return pos, hanging


# ─────────────────────────────────────────────
#  Observations
# ─────────────────────────────────────────────

def neighbour_clearance_ring(
    env: "ManagerBasedRLEnv",
    n_sectors: int = 8,
    max_range: float = 0.10,
    shoulder_body: str = "right_shoulder_pitch_link",
) -> torch.Tensor:
    """[N, n_sectors] room around the target apple, by direction, in [0, 1].

    Looking at the target along the line from the right shoulder to it (the
    way the hand comes in -- robot-knowable, unlike the stem), the plane
    around the target is cut into n_sectors equal sectors, sector 0 centred
    on world UP, counter-clockwise as seen from the robot (sector n/4 =
    left). Each sector holds the surface gap from the target to the nearest
    neighbour whose centre lies in it, clipped to max_range and divided by
    it: 1 = nothing within max_range, 0 = touching or pressed in. Neighbours
    more than one diameter BEHIND the target (deeper into the wall) are left
    out -- the hand cannot reach them on the way in. On the robot this comes
    from the same fruit detections that give the target's position."""
    N = env.num_envs
    if not hasattr(env, "_tree"):
        return torch.ones(N, n_sectors, device=env.device)
    robot = env.scene["robot"]
    if not hasattr(env, "_ring_shoulder_id"):
        env._ring_shoulder_id = robot.data.body_names.index(shoulder_body)
    c = env.scene["apple"].data.root_pos_w                                   # [N, 3]
    q, hanging = _neighbour_state(env)                                       # [N, K, 3]
    r = env._apple_radius.view(N, 1)
    sh = robot.data.body_pos_w[:, env._ring_shoulder_id]
    a = c - sh
    a[:, 2] = 0.0
    a = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)                     # horizontal, robot -> target
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(N, 3)
    left = torch.cross(up, a, dim=-1)                                        # robot's left, seen along a
    d = q - c.unsqueeze(1)
    depth = (d * a.unsqueeze(1)).sum(-1)
    ang = torch.atan2((d * left.unsqueeze(1)).sum(-1), (d * up.unsqueeze(1)).sum(-1))    # 0 = up, +90 deg = left
    sector = torch.floor(torch.remainder(ang + math.pi / n_sectors, 2 * math.pi) / (2 * math.pi / n_sectors)).long()
    sector = sector.clamp(0, n_sectors - 1)
    gap = (d.norm(dim=-1) - 2 * r).clamp(min=0.0)
    valid = hanging & (depth < 2 * r) & (gap < max_range)
    gap = torch.where(valid, gap, torch.full_like(gap, max_range))
    out = torch.full((N, n_sectors), max_range, device=env.device)
    out = out.scatter_reduce(1, sector, gap, reduce="amin")
    return out / max_range


def palm_contact_force_any(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg,
                           link: str = "right_hand_palm_link") -> torch.Tensor:
    """[N, 1] net contact force (N) on the palm link from anything it touches
    -- what the real palm pressure sensor reads. The single-apple reach used
    a reading filtered to the target only."""
    sensor = env.scene[sensor_cfg.name]
    if not hasattr(env, "_palm_any_idx"):
        env._palm_any_idx = sensor.body_names.index(link)
    return sensor.data.net_forces_w[:, env._palm_any_idx].norm(dim=-1, keepdim=True)


# ─────────────────────────────────────────────
#  Rewards
# ─────────────────────────────────────────────

def penalty_neighbour_push(env: "ManagerBasedRLEnv", deadband: float = 0.005, ref: float = 0.02) -> torch.Tensor:
    """[N] how far the neighbours still on the tree are pushed off where they
    hang (summed, m beyond deadband, / ref). Their stems swing back, so this
    is felt only while the hand is pushing."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    T = env._tree
    q, hanging = _neighbour_state(env)
    rest = torch.zeros_like(q)
    for e in range(1, T.n):                                                  # where entity e hangs
        spot = (T.occupant == e).float().argmax(dim=1)
        rest[:, e - 1] = T.spot_pos[torch.arange(env.num_envs, device=env.device), spot] + env.scene.env_origins
    disp = ((q - rest).norm(dim=-1) - deadband).clamp(min=0.0) / ref
    return (disp * hanging.float()).sum(-1)


def penalty_neighbour_knocked(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] neighbours whose stem broke THIS step (pay once per fruit knocked off)."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    broken = torch.stack([s.broken for s in env._tree.stems[1:]], dim=1)
    prev = getattr(env, "_knock_prev", None)
    if prev is None or prev.shape != broken.shape:
        prev = broken.clone()
    fresh = env.episode_length_buf <= 1                                        # new tree: nothing to charge
    newly = broken & ~prev & ~fresh.unsqueeze(-1)
    env._knock_prev = broken.clone()
    return newly.float().sum(-1)


_HAND_POINTS = ["right_hand_palm_link", "right_hand_index_1_link", "right_hand_middle_1_link",
                "right_hand_thumb_2_link"]


def penalty_hand_speed_near_fruit(env: "ManagerBasedRLEnv", near: float = 0.10, v_ref: float = 0.3) -> torch.Tensor:
    """[N] hand speed where it passes close to the NEIGHBOURS: per hand point
    (palm, fingertip links), (speed / v_ref)^2 weighted by proximity to the
    nearest hanging neighbour's surface, 1 at contact falling to 0 at `near`.
    The trained reach still crossed the wall at full speed and pushed fruit
    on the way; slowing down near fruit is what a careful picker does. The
    target itself is left to the v1 brake term."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    if not hasattr(env, "_hand_pt_ids"):
        env._hand_pt_ids = [robot.data.body_names.index(n) for n in _HAND_POINTS]
    p = robot.data.body_pos_w[:, env._hand_pt_ids]                           # [N, H, 3]
    v = robot.data.body_lin_vel_w[:, env._hand_pt_ids].norm(dim=-1)          # [N, H]
    q, hanging = _neighbour_state(env)                                       # [N, K, 3]
    d = (p.unsqueeze(2) - q.unsqueeze(1)).norm(dim=-1) - env._apple_radius.view(-1, 1, 1)
    d = torch.where(hanging.unsqueeze(1), d, torch.full_like(d, 1.0e3))
    w = (1.0 - d.amin(-1).clamp(min=0.0) / near).clamp(min=0.0)               # [N, H]
    return (w * (v / v_ref) ** 2).sum(-1)


def penalty_target_knocked(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] 1 on the step the target's own stem breaks during the reach."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._tree.target_stem.broken.float()


def penalty_commanded_accel(env: "ManagerBasedRLEnv", action_name: str = "right_arm") -> torch.Tensor:
    """[N] mean over joints of (commanded joint acceleration / the rate
    limiter's max_acceleration)^2: 1 = every joint pinned at the limit, the
    bang-bang the frozen v1 reach ran all the way (visible jerk)."""
    term = env.action_manager.get_term(action_name)
    v = term._prev_vel                                                         # this step's commanded velocity
    cache = env.__dict__.setdefault("_cmd_vel_prev", {})                     # per action term
    prev = cache.get(action_name)
    if prev is None or prev.shape != v.shape:
        prev = v.clone()
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    acc = torch.where(fresh, torch.zeros_like(v), (v - prev) / env.step_dt)
    cache[action_name] = v.clone()
    return ((acc / term.cfg.max_acceleration) ** 2).mean(-1)


def penalty_limiter_lead(env: "ManagerBasedRLEnv", action_name: str = "right_arm", ref: float = 0.5) -> torch.Tensor:
    """[N] mean over joints of (how far the policy's target is beyond the
    commanded one, minus one step at max velocity) / ref, squared. The root
    of the jerk: asking for targets the rate limiter cannot reach keeps it
    saturated, accelerating and braking flat out."""
    term = env.action_manager.get_term(action_name)
    desired = term._raw_actions * term._scale + term._offset
    if term.cfg.clip is not None:
        desired = torch.clamp(desired, min=term._clip[:, :, 0], max=term._clip[:, :, 1])
    lead = (desired - term._prev_target).abs() - term.cfg.max_velocity * env.step_dt
    return ((lead.clamp(min=0.0) / ref) ** 2).mean(-1)


# ─────────────────────────────────────────────
#  Termination
# ─────────────────────────────────────────────

def target_knocked_off(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] the target's stem broke (pushed / twisted off during the reach)."""
    if not hasattr(env, "_tree"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._tree.target_stem.broken.clone()
