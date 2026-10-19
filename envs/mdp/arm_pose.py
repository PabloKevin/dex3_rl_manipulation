# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Point-to-point arm policy (AppleArmPose-v1, envs/apple_arm_pose_env_cfg.py):
move the GRASP POCKET (where an enclosed apple's centre sits,
apple_grasp_grasp_env_cfg.GRASP_CENTER_LOCAL in the palm frame) to a goal
point with the palm normal along a goal direction -- and, when asked, the
fingers along a goal direction too -- from wherever the arm is, without the
arm or hand ever entering the body. No tree, no home pose: a caller (the
robot stack, the bench) chooses the goals, e.g. waypoints of an L path into
the fruit wall and then the apple itself.

User, 2026-10-18: "start every episode in a different position, so the
policy learns how to move from a point A to a point B, without the need of
a home position. It is important that the start position do not collide
itself ... the most important thing would be to not selfcollide and to
reach as near as possible and with a good orientation."

Starts and goals come from a pose LIBRARY (scripts/build_arm_pose_library.py
-> envs/assets/arm_pose_library.npz): random arm configurations, teleported,
checked with the full-body sphere model (self_collision.py) at every waist
yaw of a grid, with the reach's cupped hand. Every goal is therefore a pose
the arm can really take without touching the body; position, normal and
finger direction come from that pose, so they are always mutually
consistent. Stored in the TORSO frame (the arm hangs on the torso: the same
at every waist yaw); only the leg clearance changes with the yaw, hence the
per-yaw clearance columns.

An episode holds the waist at one random yaw and chains several goals: a
new goal comes when the current one is reached, or after goal_timeout_s.
So the policy also starts moves from moving, arbitrary arm states, not only
from rest. Goal flags:
  finger_on  the finger direction counts (else free: the inputs are zero);
  via        a pass-through waypoint: reached at a looser tolerance, no stop
             and no hold, then the next goal follows at once -- so a chain
             of waypoints is flown through rather than stop-and-go.

Hold phase (AppleArmPose-v2; off in v1, where the next goal came 0.2 s after
arriving): a stop goal stays for a random extra time after it is reached
(hold_s_range, or for the rest of the episode with hold_forever_prob), paid
by reward_hold for staying inside the tolerance AND still, with the palm's
speed and the commanded acceleration charged while holding; and in some
holds a steady force pushes on the hand (apply_hold_push: mostly back along
the palm normal, as an apple / stem pressed by the cup would). Measured on
the v1 policy (no contact): the position held (0.8 cm, 98% inside 2 cm /
15 deg after 8 s), but the palm never stopped (4.6 cm/s median, 11 p90).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from . import self_collision as scm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PALM = "right_hand_palm_link"
TORSO = "torso_link"
SHOULDER = "right_shoulder_pitch_link"
WAIST = "waist_yaw_joint"


class _Pose:
    def __init__(self, env: "ManagerBasedRLEnv"):
        N, d = env.num_envs, env.device
        z3 = lambda: torch.zeros(N, 3, device=d)
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=d)
        self.target, self.normal, self.finger, self.start = z3(), z3(), z3(), z3()
        self.finger_on, self.via, self.straight_ok, self.success, self.timed_out = zb(), zb(), zb(), zb(), zb()
        self.pending = torch.ones(N, dtype=torch.bool, device=d)
        self.hold = torch.zeros(N, device=d)
        self.t_goal = torch.zeros(N, device=d)
        self.bracket = torch.zeros(N, dtype=torch.long, device=d)
        self.goal_idx = torch.zeros(N, dtype=torch.long, device=d)       # library row of the goal
        self.prev_d = torch.zeros(N, device=d)                            # progress shaping
        # hold phase: reached (and since when), extra time left, the push while holding
        self.arrived, self.in_tol = zb(), zb()
        self.hold_left = torch.zeros(N, device=d)
        self.push = z3()
        self.hold_steps = torch.zeros(N, device=d)
        self.hold_in = torch.zeros(N, device=d)
        self.hold_speed = torch.zeros(N, device=d)
        self.prev_ang = torch.zeros(N, device=d)
        self.yaw = torch.zeros(N, device=d)
        # per-episode counters (metrics)
        self.n_goals = torch.zeros(N, device=d)
        self.n_reached = torch.zeros(N, device=d)
        self.err_sum = torch.zeros(N, device=d)
        self.ang_sum = torch.zeros(N, device=d)
        self.n_timeout = torch.zeros(N, device=d)


def pose_state(env: "ManagerBasedRLEnv") -> _Pose:
    if not hasattr(env, "_arm_pose"):
        env._arm_pose = _Pose(env)
    return env._arm_pose


def _bid(env, name):
    cache = env.__dict__.setdefault("_arm_pose_ids", {})
    if name not in cache:
        cache[name] = env.scene["robot"].data.body_names.index(name)
    return cache[name]


def _jid(env, name):
    cache = env.__dict__.setdefault("_arm_pose_jids", {})
    if name not in cache:
        cache[name] = env.scene["robot"].joint_names.index(name)
    return cache[name]


def hand_frame(env: "ManagerBasedRLEnv", pocket_local) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World pocket [N, 3], palm normal (palm +y) [N, 3], finger direction (palm +x) [N, 3]."""
    robot = env.scene["robot"]
    q = robot.data.body_quat_w[:, _bid(env, PALM)]
    p = robot.data.body_pos_w[:, _bid(env, PALM)]
    N = env.num_envs
    c = torch.tensor(pocket_local, device=env.device).expand(N, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(N, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(N, 3)
    return p + quat_apply(q, c), quat_apply(q, ey), quat_apply(q, ex)


def _torso(env):
    robot = env.scene["robot"]
    return robot.data.body_pos_w[:, _bid(env, TORSO)], robot.data.body_quat_w[:, _bid(env, TORSO)]


# ─────────────────────────────────────────────
#  Library
# ─────────────────────────────────────────────

class _Library:
    """The pose library on the device, with per-yaw-bracket index lists.

    Bracket b covers waist yaws [yaw_grid[b], yaw_grid[b+1]]; a pose counts
    as clear there when it is clear at both ends (the clearance changes
    smoothly with the yaw: only the legs move relative to the arm)."""

    def __init__(self, env, path: str, start_clearance: float, goal_clearance: float, task_filter: dict):
        robot = env.scene["robot"]
        dev = env.device
        lib = np.load(path)
        names = [str(n) for n in lib["joint_names"]]
        self.jids = torch.tensor([robot.joint_names.index(n) for n in names], device=dev)
        self.q = torch.tensor(lib["q"], device=dev, dtype=torch.float32)                # [M, 7]
        self.pocket = torch.tensor(lib["pocket_torso"], device=dev, dtype=torch.float32)
        self.normal = torch.tensor(lib["normal_torso"], device=dev, dtype=torch.float32)
        self.finger = torch.tensor(lib["finger_torso"], device=dev, dtype=torch.float32)
        self.yaw_grid = torch.tensor(lib["yaw_grid"], device=dev, dtype=torch.float32)  # [Y]
        clear = torch.tensor(lib["clearance"], device=dev, dtype=torch.float32)          # [M, Y]
        bclear = torch.minimum(clear[:, :-1], clear[:, 1:])                                # [M, B]
        sh = torch.tensor(lib["shoulder_torso"], device=dev, dtype=torch.float32)
        rel = self.pocket - sh
        n = self.normal
        tf = task_filter
        heading = torch.rad2deg(torch.atan2(n[:, 1], n[:, 0]))
        tilt = torch.rad2deg(torch.asin(n[:, 2].clamp(-1, 1)))
        task = (rel[:, 0] >= tf["min_forward"]) & (rel.norm(dim=-1) <= tf["max_reach"]) \
            & (rel[:, 2] >= tf["dz_range"][0]) & (rel[:, 2] <= tf["dz_range"][1]) \
            & (tilt.abs() <= tf["max_tilt_deg"]) & (heading.abs() <= tf["max_heading_deg"])
        self.sets = {
            "start": self._pack(bclear >= start_clearance),
            "goal_any": self._pack(bclear >= goal_clearance),
            "goal_task": self._pack((bclear >= goal_clearance) & task.unsqueeze(-1)),
        }
        self.n_brackets = bclear.shape[1]
        sizes = {k: v[1].tolist() for k, v in self.sets.items()}
        print(f"[arm_pose] library {path}: {len(self.q)} poses, task-like {int(task.sum())}; "
              f"per yaw bracket: {sizes}")

    @staticmethod
    def _pack(ok: torch.Tensor):
        """[M, B] bool -> (padded index lists [B, Kmax], counts [B])."""
        cols = [ok[:, b].nonzero(as_tuple=True)[0] for b in range(ok.shape[1])]
        counts = torch.tensor([len(c) for c in cols], device=ok.device)
        if int(counts.min()) == 0:
            raise RuntimeError("arm_pose: a waist-yaw bracket has no valid pose in the library")
        pad = torch.zeros(len(cols), int(counts.max()), dtype=torch.long, device=ok.device)
        for b, c in enumerate(cols):
            pad[b, :len(c)] = c
        return pad, counts

    def sample(self, kind: str, bracket: torch.Tensor) -> torch.Tensor:
        pad, counts = self.sets[kind]
        k = (torch.rand(len(bracket), device=bracket.device) * counts[bracket]).long()
        return pad[bracket, k.clamp(max=pad.shape[1] - 1)]


def library(env: "ManagerBasedRLEnv") -> _Library:
    return env._arm_pose_lib


# ─────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────

def reset_arm_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    library_file: str,
    start_clearance: float = 0.015,
    goal_clearance: float = 0.03,
    task_filter: dict | None = None,
    hand_targets: dict | None = None,
) -> None:
    """mode="reset": waist yaw for the episode (random bracket, uniform in
    it, joint + PD target), arm teleported to a library start pose that is
    clear at that yaw (zero velocity, PD target on it), the hand set to its
    held pre-shape at once. The first goal is drawn on the first step
    (advance_goals), once the body poses are fresh."""
    robot = env.scene["robot"]
    dev = env.device
    if not hasattr(env, "_arm_pose_lib"):
        env._arm_pose_lib = _Library(env, library_file, start_clearance, goal_clearance, task_filter)
    L = env._arm_pose_lib
    P = pose_state(env)
    ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
    n = len(ids)
    b = torch.randint(0, L.n_brackets, (n,), device=dev)
    lo, hi = L.yaw_grid[b], L.yaw_grid[b + 1]
    yaw = lo + (hi - lo) * torch.rand(n, device=dev)
    q = L.q[L.sample("start", b)]

    jp = robot.data.joint_pos[ids].clone()
    tgt = robot.data.joint_pos_target[ids].clone()
    jp[:, L.jids] = q
    tgt[:, L.jids] = q
    w = _jid(env, WAIST)
    jp[:, w] = yaw
    tgt[:, w] = yaw
    if hand_targets:
        hj = torch.tensor([robot.joint_names.index(j) for j in hand_targets], device=dev)
        hv = torch.tensor(list(hand_targets.values()), device=dev)
        jp[:, hj] = hv
        tgt[:, hj] = hv
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    robot.set_joint_position_target(tgt, env_ids=ids)

    P.bracket[ids] = b
    P.yaw[ids] = yaw
    P.pending[ids] = True
    P.hold[ids] = 0.0
    P.t_goal[ids] = 0.0
    P.success[ids] = False
    P.timed_out[ids] = False
    for t in (P.n_goals, P.n_reached, P.err_sum, P.ang_sum, P.n_timeout, P.hold_steps, P.hold_in, P.hold_speed,
              P.hold_left):
        t[ids] = 0.0
    P.arrived[ids] = False
    P.push[ids] = 0.0


def _straight_clear(env, ids, a, b, hand_radius: float, margin: float, samples: int = 12) -> torch.Tensor:
    """[n] bool: a sphere of hand_radius around the pocket, moved straight
    from a to b (world [n, 3]), stays margin clear of every body sphere --
    only then does the straight path make sense to reward."""
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    sets = env._self_collision_spheres
    body = scm._world_spheres(env, sets["body"])[ids]                                  # [n, S, 3]
    t = torch.linspace(0.0, 1.0, samples, device=env.device).view(1, -1, 1)
    pts = a.unsqueeze(1) + t * (b - a).unsqueeze(1)                                    # [n, T, 3]
    gap = torch.cdist(pts, body) - sets["body"]["r"].view(1, 1, -1) - hand_radius
    return gap.amin(dim=(-1, -2)) >= margin


def advance_goals(
    env: "ManagerBasedRLEnv",
    env_ids,
    pocket_local,
    task_prob: float = 0.6,
    finger_prob: float = 0.5,
    via_prob: float = 0.25,
    goal_timeout_s: float = 4.0,
    straight_hand_radius: float = 0.06,
    straight_margin: float = 0.02,
    hold_s_range: tuple[float, float] = (0.0, 0.0),
    hold_forever_prob: float = 0.0,
    push_prob: float = 0.0,
    push_max_n: float = 0.0,
) -> None:
    """mode="interval" (every step, after the rewards): a new goal where the
    episode just started (first step), the goal was reached this step, or it
    timed out. Goal = a library pose clear at the episode's yaw (task-like
    -- in front of the shoulder, palm near horizontal, facing forward-ish --
    with task_prob, else anywhere), put in the world through the torso pose.
    The goal pose must differ from where the arm is (> 5 cm or > 30 deg).
    A reached stop goal stays a further hold time (hold_s_range, or the rest
    of the episode with hold_forever_prob; 0 = the next goal at once, v1); a
    stop goal gets a push for its hold with push_prob (apply_hold_push), up
    to push_max_n newtons. The timeout only runs until the goal is reached."""
    P = pose_state(env)
    dt = env.step_dt
    live = ~P.pending
    P.t_goal = P.t_goal + dt * live.float()
    P.hold_left = P.hold_left - dt * P.arrived.float()
    P.timed_out = live & ~P.arrived & ~P.success & (P.t_goal >= goal_timeout_s)
    done = (P.success & P.via) | (P.arrived & (P.hold_left <= 0.0))
    new = (P.pending & (env.episode_length_buf >= 1)) | (live & (done | P.timed_out))
    if not bool(new.any()):
        return
    ids = new.nonzero(as_tuple=True)[0]
    n = len(ids)
    dev = env.device
    L = library(env)
    pocket, normal, _ = hand_frame(env, pocket_local)
    tp, tq = _torso(env)
    b = P.bracket[ids]
    # a few candidates, keep the first one far enough from where the hand is
    idx = None
    for _ in range(4):
        use_task = torch.rand(n, device=dev) < task_prob
        cand = torch.where(use_task, L.sample("goal_task", b), L.sample("goal_any", b))
        tgt = tp[ids] + quat_apply(tq[ids], L.pocket[cand])
        nrm = quat_apply(tq[ids], L.normal[cand])
        far = ((tgt - pocket[ids]).norm(dim=-1) > 0.05) | ((nrm * normal[ids]).sum(-1) < math.cos(math.radians(30)))
        idx = cand if idx is None else torch.where(keep, idx, cand)
        keep = far if _ == 0 else (keep | far)
        if bool(keep.all()):
            break
    # metrics of the goal being replaced (not on the very first goal)
    old = ~P.pending[ids]
    d_old = (P.target[ids] - pocket[ids]).norm(dim=-1)
    a_old = torch.acos((P.normal[ids] * normal[ids]).sum(-1).clamp(-1, 1))
    P.n_goals[ids] += old.float()
    P.n_reached[ids] += (old & (P.arrived[ids] | (P.success[ids] & P.via[ids]))).float()
    P.n_timeout[ids] += (old & P.timed_out[ids]).float()
    P.err_sum[ids] += torch.where(old & P.timed_out[ids], d_old, torch.zeros_like(d_old))
    P.ang_sum[ids] += torch.where(old & P.timed_out[ids], a_old, torch.zeros_like(a_old))

    P.goal_idx[ids] = idx
    P.target[ids] = tp[ids] + quat_apply(tq[ids], L.pocket[idx])
    P.normal[ids] = quat_apply(tq[ids], L.normal[idx])
    P.prev_d[ids] = (P.target[ids] - pocket[ids]).norm(dim=-1)
    P.prev_ang[ids] = torch.acos((P.normal[ids] * normal[ids]).sum(-1).clamp(-1.0, 1.0))
    P.finger[ids] = quat_apply(tq[ids], L.finger[idx])
    P.finger_on[ids] = torch.rand(n, device=dev) < finger_prob
    P.via[ids] = torch.rand(n, device=dev) < via_prob
    P.start[ids] = pocket[ids]
    P.straight_ok[ids] = _straight_clear(env, ids, pocket[ids], P.target[ids], straight_hand_radius, straight_margin)
    P.t_goal[ids] = 0.0
    P.hold[ids] = 0.0
    P.pending[ids] = False
    P.success[ids] = False
    P.timed_out[ids] = False
    P.arrived[ids] = False
    stop = ~P.via[ids]
    extra = hold_s_range[0] + (hold_s_range[1] - hold_s_range[0]) * torch.rand(n, device=dev)
    extra = torch.where(torch.rand(n, device=dev) < hold_forever_prob, torch.full_like(extra, 1.0e9), extra)
    P.hold_left[ids] = torch.where(stop, extra, torch.zeros_like(extra))
    # the push while holding: mostly back along the palm normal (the apple pressing on the cup)
    pdir = -P.normal[ids] + 0.5 * torch.randn(n, 3, device=dev)
    pdir = pdir / pdir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mag = push_max_n * torch.rand(n, device=dev) * ((torch.rand(n, device=dev) < push_prob) & stop).float()
    P.push[ids] = pdir * mag.unsqueeze(-1)


# ─────────────────────────────────────────────
#  Observations
# ─────────────────────────────────────────────

def _live_target(env, pocket_local):
    """Goal pose with the pending first step showing "already there"."""
    P = pose_state(env)
    pocket, normal, finger = hand_frame(env, pocket_local)
    pend = P.pending.unsqueeze(-1)
    return (torch.where(pend, pocket, P.target), torch.where(pend, normal, P.normal),
            torch.where(pend, finger, P.finger), pocket)


def obs_goal_delta(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """[N, 3] goal - pocket, torso frame."""
    tgt, _, _, pocket = _live_target(env, pocket_local)
    _, tq = _torso(env)
    return quat_apply_inverse(tq, tgt - pocket)


def obs_goal_position(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """[N, 3] goal - right shoulder, torso frame: WHERE the goal is relative
    to the body (the delta alone does not say how close to the chest it is)."""
    tgt, _, _, _ = _live_target(env, pocket_local)
    _, tq = _torso(env)
    sh = env.scene["robot"].data.body_pos_w[:, _bid(env, SHOULDER)]
    return quat_apply_inverse(tq, tgt - sh)


def obs_goal_normal(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """[N, 3] goal palm normal in the PALM frame ((0, 1, 0) = there)."""
    _, nrm, _, _ = _live_target(env, pocket_local)
    q = env.scene["robot"].data.body_quat_w[:, _bid(env, PALM)]
    return quat_apply_inverse(q, nrm)


def obs_goal_finger(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """[N, 3] goal finger direction in the PALM frame ((1, 0, 0) = there);
    zeros when the finger direction is free."""
    P = pose_state(env)
    _, _, fin, _ = _live_target(env, pocket_local)
    q = env.scene["robot"].data.body_quat_w[:, _bid(env, PALM)]
    return quat_apply_inverse(q, fin) * P.finger_on.float().unsqueeze(-1)


def obs_goal_flags(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 2] (finger direction counts, pass-through waypoint)."""
    P = pose_state(env)
    return torch.stack([P.finger_on.float(), P.via.float()], dim=-1)


# ─────────────────────────────────────────────
#  Goal tracking (termination slot) and rewards
# ─────────────────────────────────────────────

def errors(env: "ManagerBasedRLEnv", pocket_local):
    """(distance [N], normal angle [N], finger angle [N], live [N])."""
    P = pose_state(env)
    pocket, normal, finger = hand_frame(env, pocket_local)
    d = (P.target - pocket).norm(dim=-1)
    ang = torch.acos((P.normal * normal).sum(-1).clamp(-1.0, 1.0))
    fang = torch.acos((P.finger * finger).sum(-1).clamp(-1.0, 1.0))
    return d, ang, fang, ~P.pending


def track_goal(
    env: "ManagerBasedRLEnv",
    pocket_local,
    pos_tol: float = 0.02,
    ang_tol_deg: float = 15.0,
    finger_tol_deg: float = 25.0,
    speed_tol: float = 0.08,
    hold_time: float = 0.2,
    via_pos_tol: float = 0.03,
    via_ang_tol_deg: float = 25.0,
) -> torch.Tensor:
    """Termination slot that never ends the episode: computes this step's
    goal success before the rewards (the termination manager runs first).
    A goal is reached with the pocket within pos_tol, the normal within
    ang_tol (and the fingers within finger_tol when they count), the palm
    slower than speed_tol, held hold_time; a via goal within via_pos_tol /
    via_ang_tol (fingers likewise), at any speed, no hold."""
    P = pose_state(env)
    d, ang, fang, live = errors(env, pocket_local)
    v = env.scene["robot"].data.body_lin_vel_w[:, _bid(env, PALM)].norm(dim=-1)
    fin_ok = ~P.finger_on | (fang < math.radians(finger_tol_deg))
    stop_ok = live & (d < pos_tol) & (ang < math.radians(ang_tol_deg)) & fin_ok & (v < speed_tol)
    P.hold = torch.where(stop_ok & ~P.via, P.hold + env.step_dt, torch.zeros_like(P.hold))
    via_ok = live & P.via & (d < via_pos_tol) & (ang < math.radians(via_ang_tol_deg)) & fin_ok
    arrive = (P.hold >= hold_time) & ~P.arrived & ~P.via
    P.arrived = P.arrived | arrive
    P.success = arrive | via_ok                      # the event: one step per goal
    P.in_tol = live & (d < pos_tol) & (ang < math.radians(ang_tol_deg)) & fin_ok
    # hold-phase metrics (the step a goal is replaced excluded)
    h = P.arrived & ~arrive
    P.hold_steps += h.float()
    P.hold_in += (h & P.in_tol).float()
    P.hold_speed += v * h.float()
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reward_pos(env: "ManagerBasedRLEnv", pocket_local, std: float) -> torch.Tensor:
    d, _, _, live = errors(env, pocket_local)
    return torch.exp(-0.5 * (d / std) ** 2) * live.float()


def reward_normal(env: "ManagerBasedRLEnv", pocket_local, std_rad: float, gate: float | None = None) -> torch.Tensor:
    d, ang, _, live = errors(env, pocket_local)
    r = torch.exp(-0.5 * (ang / std_rad) ** 2)
    if gate is not None:
        r = r * torch.exp(-0.5 * (d / gate) ** 2)
    return r * live.float()


def reward_finger(env: "ManagerBasedRLEnv", pocket_local, std_rad: float, gate: float | None = None) -> torch.Tensor:
    P = pose_state(env)
    d, _, fang, live = errors(env, pocket_local)
    r = torch.exp(-0.5 * (fang / std_rad) ** 2)
    if gate is not None:
        r = r * torch.exp(-0.5 * (d / gate) ** 2)
    return r * (live & P.finger_on).float()


def reward_progress(env: "ManagerBasedRLEnv", pocket_local, max_speed: float = 1.0,
                    max_ang_speed: float = 3.0, ang_scale: float = 0.2) -> torch.Tensor:
    """[N] potential-based shaping: distance closed this step (m/s, capped
    at +-max_speed) + ang_scale x normal angle closed (rad/s, capped). Over
    a whole move it sums to (start error - end error) x weight, whatever the
    path, so it speeds learning without changing the best policy. The
    previous errors are reset when a goal is issued (advance_goals)."""
    P = pose_state(env)
    d, ang, _, live = errors(env, pocket_local)
    dt = env.step_dt
    v = ((P.prev_d - d) / dt).clamp(-max_speed, max_speed)
    w = ((P.prev_ang - ang) / dt).clamp(-max_ang_speed, max_ang_speed)
    P.prev_d = torch.where(live, d, P.prev_d)
    P.prev_ang = torch.where(live, ang, P.prev_ang)
    return (v + ang_scale * w) * live.float()


def _palm_speed(env):
    return env.scene["robot"].data.body_lin_vel_w[:, _bid(env, PALM)].norm(dim=-1)


def reward_hold(env: "ManagerBasedRLEnv", speed_ref: float = 0.02) -> torch.Tensor:
    """[N] holding a reached stop goal: inside the goal tolerance (track_goal's
    position / normal / finger test, any speed) x exp(-(v / speed_ref)^2 / 2)
    -- still, not buzzing around the goal."""
    P = pose_state(env)
    v = _palm_speed(env)
    return (P.arrived & P.in_tol & ~P.via).float() * torch.exp(-0.5 * (v / speed_ref) ** 2)


def _near_stop_goal(env, pocket_local, near: float) -> torch.Tensor:
    """[N] bool: within `near` of a live stop goal, reached or not. The hold
    penalties are gated on this, NOT on arrival: gated on arrival, the first
    v2 fine-tune learned to stop arriving (under exploration noise the hand
    moves anyway, and only an arrived goal was charged for it: -1.3/s of
    penalties against +0.13/s of hold reward; reached goals 62% -> 37%)."""
    P = pose_state(env)
    d, _, _, live = errors(env, pocket_local)
    return live & ~P.via & (d < near)


def penalty_hold_motion(env: "ManagerBasedRLEnv", pocket_local, near: float = 0.05, ref: float = 0.05,
                        cap: float = 4.0) -> torch.Tensor:
    """[N] (palm speed / ref)^2, capped, within `near` of a stop goal."""
    return ((_palm_speed(env) / ref) ** 2).clamp(max=cap) * _near_stop_goal(env, pocket_local, near).float()


def penalty_hold_commanded_accel(env: "ManagerBasedRLEnv", pocket_local, near: float = 0.05,
                                 action_name: str = "right_arm") -> torch.Tensor:
    """[N] tree_reach.penalty_commanded_accel within `near` of a stop goal
    (its own previous-velocity cache: calling that function twice a step
    would zero its acceleration)."""
    term = env.action_manager.get_term(action_name)
    v = term._prev_vel
    prev = getattr(env, "_arm_pose_cmd_prev", None)
    if prev is None or prev.shape != v.shape:
        prev = v.clone()
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    acc = torch.where(fresh, torch.zeros_like(v), (v - prev) / env.step_dt)
    env._arm_pose_cmd_prev = v.clone()
    return ((acc / term.cfg.max_acceleration) ** 2).mean(-1) * _near_stop_goal(env, pocket_local, near).float()


def apply_hold_push(env: "ManagerBasedRLEnv", env_ids, pocket_local) -> None:
    """mode="interval" (every step, after advance_goals): the hold's push
    (advance_goals) at the pocket, on the palm link, only while the goal is
    held -- zero otherwise."""
    P = pose_state(env)
    robot = env.scene["robot"]
    pocket, _, _ = hand_frame(env, pocket_local)
    f = (P.push * P.arrived.float().unsqueeze(-1)).unsqueeze(1)
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=f, torques=torch.zeros_like(f), positions=pocket.unsqueeze(1),
        body_ids=torch.tensor([_bid(env, PALM)], device=env.device), is_global=True)


def reward_goal_reached(env: "ManagerBasedRLEnv", via_value: float = 0.5) -> torch.Tensor:
    """[N] 1 on the step a stop goal is reached, via_value for a via goal
    (one event per goal: weight = bonus in return units / dt)."""
    P = pose_state(env)
    return P.success.float() * torch.where(P.via, via_value, 1.0)


def penalty_path_deviation(env: "ManagerBasedRLEnv", pocket_local, deadband: float = 0.03,
                           ref: float = 0.05, cap: float = 4.0) -> torch.Tensor:
    """[N] (pocket distance from the straight start->goal segment beyond
    deadband) / ref, squared, capped -- only where that straight segment is
    clear of the body (advance_goals). Waypoints then mean something: the
    hand flies the segments it is given, so an L path chosen to miss the
    neighbours is followed; where the straight line would cross the body,
    the policy goes round it unpaid."""
    P = pose_state(env)
    pocket, _, _ = hand_frame(env, pocket_local)
    a, b = P.start, P.target
    ab = b - a
    t = (((pocket - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
    dev = (pocket - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)
    return (((dev - deadband).clamp(min=0.0) / ref) ** 2).clamp(max=cap) * (~P.pending & P.straight_ok).float()


def penalty_near_goal_speed(env: "ManagerBasedRLEnv", pocket_local, dist: float = 0.10) -> torch.Tensor:
    """[N] palm speed^2 within dist of a STOP goal (via goals fly through)."""
    P = pose_state(env)
    d, _, _, live = errors(env, pocket_local)
    v = env.scene["robot"].data.body_lin_vel_w[:, _bid(env, PALM)].norm(dim=-1)
    return (v ** 2) * ((d < dist) & live & ~P.via).float()


def self_collision(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Termination: any arm/hand sphere really overlapping a body sphere
    (margin 0). Cached for reward_collision."""
    hit = (scm.self_collision_depth(env, margin=0.0).amax(-1) > 0.0) & (env.episode_length_buf >= 1)
    env._arm_pose_hit = hit
    return hit


def reward_collision(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return getattr(env, "_arm_pose_hit", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()


# ─────────────────────────────────────────────
#  Metrics (curriculum slot, changes nothing)
# ─────────────────────────────────────────────

def arm_pose_metrics(env: "ManagerBasedRLEnv", env_ids: Sequence[int], ema: float = 0.05) -> dict:
    """Run before the reset, on the episodes ending: goals per episode,
    share of goals reached, distance / normal angle (cm / deg) left at a
    timed-out goal, share of episodes ended by a self-collision.
    Curriculum/<term>/<key>."""
    st = env.__dict__.setdefault("_arm_pose_metrics", {})
    if not hasattr(env, "_arm_pose") or env.common_step_counter < 1:
        return dict(st)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return dict(st)
    P = env._arm_pose
    goals = P.n_goals[ids].sum().clamp(min=1.0)
    tout = P.n_timeout[ids].sum()
    vals = {
        "goals_per_episode": float(P.n_goals[ids].mean()),
        "reached_share": float(P.n_reached[ids].sum() / goals),
        "timeout_err_cm": float(P.err_sum[ids].sum() / tout.clamp(min=1.0) * 100.0) if tout > 0 else st.get("timeout_err_cm", 0.0),
        "timeout_ang_deg": float(torch.rad2deg(P.ang_sum[ids].sum() / tout.clamp(min=1.0))) if tout > 0 else st.get("timeout_ang_deg", 0.0),
        "hold_in_tol_share": float(P.hold_in[ids].sum() / P.hold_steps[ids].sum().clamp(min=1.0)),
        "hold_speed_cms": float(P.hold_speed[ids].sum() / P.hold_steps[ids].sum().clamp(min=1.0) * 100.0),
        "collision_share": float(getattr(env, "_arm_pose_hit", torch.zeros(env.num_envs, dtype=torch.bool,
                                                                            device=env.device))[ids].float().mean()),
    }
    for k, m in vals.items():
        st[k] = m if k not in st else (1.0 - ema) * st[k] + ema * m
    return dict(st)


# ─────────────────────────────────────────────
#  play.py visuals
# ─────────────────────────────────────────────

class GoalViz:
    """The goal drawn every step (env 0): a sphere at the goal pocket
    (orange = stop, cyan = via), an arrow along the goal palm normal and,
    when it counts, a thinner one along the goal finger direction."""

    def __init__(self, env: "ManagerBasedRLEnv"):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        mat = sim_utils.PreviewSurfaceCfg
        self.env = env
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ArmPoseGoal",
            markers={
                "stop": sim_utils.SphereCfg(radius=0.02, visual_material=mat(diffuse_color=(1.0, 0.5, 0.0))),
                "via": sim_utils.SphereCfg(radius=0.02, visual_material=mat(diffuse_color=(0.0, 0.8, 0.9))),
                "normal": sim_utils.CylinderCfg(radius=0.004, height=0.08, axis="Z",
                                                visual_material=mat(diffuse_color=(0.1, 0.9, 0.1))),
                "finger": sim_utils.CylinderCfg(radius=0.0025, height=0.06, axis="Z",
                                                visual_material=mat(diffuse_color=(0.2, 0.3, 1.0))),
            }))

    @staticmethod
    def _quat_z_to(d: torch.Tensor) -> torch.Tensor:
        z = torch.tensor([0.0, 0.0, 1.0], device=d.device).expand_as(d)
        axis = torch.cross(z, d, dim=-1)
        w = 1.0 + (z * d).sum(-1, keepdim=True)
        q = torch.cat([w, axis], dim=-1)
        flip = w.squeeze(-1) < 1e-6
        q[flip] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=d.device)
        return q / q.norm(dim=-1, keepdim=True)

    def update(self) -> None:
        P = pose_state(self.env)
        i = 0
        t, n, f = P.target[i:i + 1], P.normal[i:i + 1], P.finger[i:i + 1]
        trans = [t, t + 0.04 * n]
        quats = [torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=t.device), self._quat_z_to(n)]
        idx = [1 if bool(P.via[i]) else 0, 2]
        if bool(P.finger_on[i]):
            trans.append(t + 0.03 * f)
            quats.append(self._quat_z_to(f))
            idx.append(3)
        self.markers.visualize(translations=torch.cat(trans), orientations=torch.cat(quats),
                               marker_indices=torch.tensor(idx, device=t.device))


def draw_goal(env: "ManagerBasedRLEnv", env_ids) -> None:
    """mode="interval" event for the PLAY config: GoalViz.update()."""
    if not hasattr(env, "_arm_pose_viz"):
        env._arm_pose_viz = GoalViz(env)
    env._arm_pose_viz.update()
