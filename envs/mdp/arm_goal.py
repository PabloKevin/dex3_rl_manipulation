# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Goal-reaching arm policy for the harvest path (AppleArmGoal-v1,
envs/apple_arm_goal_env_cfg.py): move the GRASP POCKET (where a held apple's
centre sits, apple_grasp_grasp_env_cfg.GRASP_CENTER_LOCAL in the palm frame)
to a goal point with the palm normal along a goal direction, from any arm
pose, with or without an apple's weight in the hand, keeping the arm, hand
and fruit out of the body and the basket.

The harvest bench chains it: after the scripted twist and a short pull,
goal 1 = the SAFE ZONE (straight back along the way the hand came in, so it
clears the fruit wall before moving sideways; stopped short of the body),
goal 2 = the drop pose over the basket (palm down), then release, then
goal 3 = home.

Training episodes draw one goal each:
  SAFE  start at a pick-like pose (palm in the reach envelope, facing the
        wall), goal = the safe-zone point computed from that pose
        (resolve_goals, first step of the episode);
  DROP  start anywhere (mostly pick-like), goal = above a basket placed at
        random within +-basket_range of its nominal place, palm down;
  HOME  start near the drop pose (or anywhere), empty hand, goal = home.
The basket is an analytic obstacle here (basket.py's solid stand + basket,
at the episode's position); the load is the apple's weight applied at the
pocket (no fruit is simulated: holding it is the grasp policy's job).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PALM = "right_hand_palm_link"
TORSO = "torso_link"
SHOULDER = "right_shoulder_pitch_link"
SAFE, DROP, HOME = 0, 1, 2

#: Measured (default arm pose / basket_drop_path waypoints), env-local.
HOME_POCKET = (0.262, -0.095, 0.719)
HOME_NORMAL = (0.0, 1.0, 0.0)
DROP_NORMAL = (0.187, -0.254, -0.949)       # palm (nearly) down at the planned drop pose
#: Full palm orientations (w, x, y, z) at home and at the planned drop pose:
#: the goal also fixes where the FINGERS point (palm-local +x), not only the
#: palm normal (+y). With the normal alone the finger direction was free and
#: the first policies came home with the fingers pointing at the ground.
HOME_QUAT = (0.9552, -0.0001, 0.2959, 0.0001)
DROP_QUAT = (0.5011, -0.5498, 0.3491, -0.5699)


def _axis(q, a):
    """Rotate unit axis a by quaternion q (w, x, y, z), plain floats."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return tuple(sum(R[i][j] * a[j] for j in range(3)) for i in range(3))


HOME_FINGER = _axis(HOME_QUAT, (1.0, 0.0, 0.0))
DROP_FINGER = _axis(DROP_QUAT, (1.0, 0.0, 0.0))
DROP_OFFSET_XY = (-0.019, 0.006)            # pocket vs basket axis at that pose
BASKET_NOMINAL = (-0.05, -0.50, 0.81)       # basket axis x, y and rim height (basket.py)


class _Goal:
    def __init__(self, env: "ManagerBasedRLEnv"):
        N, d = env.num_envs, env.device
        z3 = lambda: torch.zeros(N, 3, device=d)
        self.target, self.normal, self.start, self.basket = z3(), z3(), z3(), z3()
        self.finger = z3()                   # goal finger direction (palm-local +x)
        self.kind = torch.zeros(N, dtype=torch.long, device=d)
        self.carry = torch.zeros(N, dtype=torch.bool, device=d)
        self.mass = torch.zeros(N, device=d)
        self.pending = torch.ones(N, dtype=torch.bool, device=d)
        self.hold = torch.zeros(N, device=d)
        self.success = torch.zeros(N, dtype=torch.bool, device=d)


def goal_state(env: "ManagerBasedRLEnv") -> _Goal:
    if not hasattr(env, "_goal"):
        env._goal = _Goal(env)
    return env._goal


def _ids(env, name):
    cache = env.__dict__.setdefault("_arm_goal_ids", {})
    if name not in cache:
        cache[name] = env.scene["robot"].data.body_names.index(name)
    return cache[name]


def palm_finger(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """World finger direction (palm-local +x) [N, 3]."""
    q = env.scene["robot"].data.body_quat_w[:, _ids(env, PALM)]
    return quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3))


def pocket_and_normal(env: "ManagerBasedRLEnv", pocket_local) -> tuple[torch.Tensor, torch.Tensor]:
    """World pocket position [N, 3] and palm normal [N, 3]."""
    robot = env.scene["robot"]
    pid = _ids(env, PALM)
    q = robot.data.body_quat_w[:, pid]
    p = robot.data.body_pos_w[:, pid]
    N = env.num_envs
    c = torch.tensor(pocket_local, device=env.device).expand(N, 3)
    n = torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(N, 3)
    return p + quat_apply(q, c), quat_apply(q, n)


# ─────────────────────────────────────────────
#  Events
# ─────────────────────────────────────────────

def reset_arm_goal(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    library: str,
    kind_probs: tuple[float, float, float] = (0.35, 0.40, 0.25),
    basket_range: float = 0.15,
    mass_range: tuple[float, float] = (0.10, 0.35),
    drop_height: tuple[float, float] = (0.10, 0.16),
    home_noise: float = 0.15,
    bank_file: str | None = None,
    waist_range: float = 0.0,
) -> None:
    """mode="reset": start pose (arm teleported), goal kind, basket, load.
    bank_file: reach end poses to start pick-like episodes from (default
    reach_end_poses.npz next to the library); with a "waist" key the waist
    yaw starts there too. waist_range: other starts get a waist yaw uniform
    in +-waist_range (0 = untouched, the arm-only task)."""
    G = goal_state(env)
    robot = env.scene["robot"]
    dev = env.device
    if not hasattr(env, "_arm_goal_lib"):
        lib = np.load(library)
        names = [str(n) for n in lib["joint_names"]]
        env._arm_goal_jids = torch.tensor([robot.joint_names.index(n) for n in names], device=dev)
        q = torch.tensor(lib["q"], device=dev, dtype=torch.float32)
        palm = torch.tensor(lib["palm_local"], device=dev, dtype=torch.float32)
        nrm = torch.tensor(lib["palm_normal"], device=dev, dtype=torch.float32)
        sh = robot.data.body_pos_w[0, _ids(env, SHOULDER)] - env.scene.env_origins[0]
        rel = palm - sh
        r = rel.norm(dim=-1)
        # pick-like: palm in the reach envelope in front of the shoulder, facing the wall
        pick = (r > 0.20) & (r < 0.40) & (rel[:, 0] > 0.08) & (nrm[:, 0] > 0.5) & (nrm[:, 2].abs() < 0.5)
        env._arm_goal_lib = (q, pick.nonzero(as_tuple=True)[0])
        wp = np.load(library.replace("harvest_pose_library.npz", "basket_drop_path.npz"))
        order = [list(wp["joint_names"]).index(n) for n in names]
        env._arm_goal_drop_q = torch.tensor(wp["waypoints"][2][order], device=dev, dtype=torch.float32)
        dist = (q - env._arm_goal_drop_q).norm(dim=-1)
        env._arm_goal_near_drop = dist.argsort()[:200]
        # real reach end poses (scripts/record_reach_starts.py), if recorded
        bank = bank_file or library.replace("harvest_pose_library.npz", "reach_end_poses.npz")
        import os
        env._arm_goal_bank = None
        env._arm_goal_bank_waist = None
        if os.path.exists(bank):
            b = np.load(bank)
            border = [list(b["joint_names"]).index(n) for n in names]
            env._arm_goal_bank = torch.tensor(b["q"][:, border], device=dev, dtype=torch.float32)
            if "waist" in b.files:
                env._arm_goal_bank_waist = torch.tensor(b["waist"], device=dev, dtype=torch.float32)
    q_lib, pick_idx = env._arm_goal_lib
    n = len(env_ids)
    u = torch.rand(n, device=dev)
    kind = torch.full((n,), HOME, dtype=torch.long, device=dev)
    kind[u < kind_probs[0] + kind_probs[1]] = DROP
    kind[u < kind_probs[0]] = SAFE
    any_i = torch.randint(0, len(q_lib), (n,), device=dev)
    pick_i = pick_idx[torch.randint(0, len(pick_idx), (n,), device=dev)]
    q = q_lib[pick_i]
    w = (torch.rand(n, device=dev) * 2 - 1) * waist_range
    if env._arm_goal_bank is not None:            # where the reach really leaves the arm (70%)
        bi = torch.randint(0, len(env._arm_goal_bank), (n,), device=dev)
        from_bank = torch.rand(n, device=dev) < 0.7
        q = torch.where(from_bank.unsqueeze(-1), env._arm_goal_bank[bi], q)
        if env._arm_goal_bank_waist is not None:
            w = torch.where(from_bank, env._arm_goal_bank_waist[bi], w)
    use_any = (kind == DROP) & (torch.rand(n, device=dev) < 0.3)
    q = torch.where(use_any.unsqueeze(-1), q_lib[any_i], q)
    # HOME starts near the drop pose: library poses (validated collision-free),
    # not the drop configuration plus raw noise (unchecked -- some started
    # inside the body)
    near_i = env._arm_goal_near_drop[torch.randint(0, len(env._arm_goal_near_drop), (n,), device=dev)]
    home_start = torch.where((torch.rand(n, device=dev) < 0.6).unsqueeze(-1), q_lib[near_i], q_lib[any_i])
    q = torch.where((kind == HOME).unsqueeze(-1), home_start, q)
    lo = robot.data.soft_joint_pos_limits[env_ids][:, env._arm_goal_jids, 0]
    hi = robot.data.soft_joint_pos_limits[env_ids][:, env._arm_goal_jids, 1]
    q = torch.max(torch.min(q, hi), lo)

    jp = robot.data.joint_pos[env_ids].clone()
    jp[:, env._arm_goal_jids] = q
    tgt = robot.data.joint_pos_target[env_ids].clone()
    tgt[:, env._arm_goal_jids] = q
    if waist_range > 0 or env._arm_goal_bank_waist is not None:
        wid = robot.joint_names.index(WAIST)
        jp[:, wid] = w
        tgt[:, wid] = w
    jv = torch.zeros_like(jp)
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    robot.set_joint_position_target(tgt, env_ids=env_ids)

    G.kind[env_ids] = kind
    G.carry[env_ids] = kind != HOME
    G.mass[env_ids] = torch.where(kind != HOME, torch.empty(n, device=dev).uniform_(*mass_range),
                                  torch.zeros(n, device=dev))
    b = torch.tensor(BASKET_NOMINAL, device=dev) + (torch.rand(n, 3, device=dev) * 2 - 1) * basket_range
    G.basket[env_ids] = b
    origin = env.scene.env_origins[env_ids]
    drop = torch.stack([b[:, 0] + DROP_OFFSET_XY[0], b[:, 1] + DROP_OFFSET_XY[1],
                        b[:, 2] + torch.empty(n, device=dev).uniform_(*drop_height)], dim=-1)
    home = torch.tensor(HOME_POCKET, device=dev).expand(n, 3)
    G.target[env_ids] = origin + torch.where((kind == DROP).unsqueeze(-1), drop, home)
    dn = torch.tensor(DROP_NORMAL, device=dev); dn = dn / dn.norm()
    G.finger[env_ids] = torch.where((kind == DROP).unsqueeze(-1), torch.tensor(DROP_FINGER, device=dev).expand(n, 3),
                                    torch.tensor(HOME_FINGER, device=dev).expand(n, 3))
    G.normal[env_ids] = torch.where((kind == DROP).unsqueeze(-1), dn.expand(n, 3),
                                    torch.tensor(HOME_NORMAL, device=dev).expand(n, 3))
    G.pending[env_ids] = True              # start (and the SAFE goal) set on the first step
    G.hold[env_ids] = 0.0
    G.success[env_ids] = False


def resolve_goals(
    env: "ManagerBasedRLEnv",
    env_ids,
    pocket_local,
    retreat_range: tuple[float, float] = (0.10, 0.20),
    body_margin: float = 0.06,
    min_from_shoulder: float = 0.22,
    apple_radius: float = 0.045,
    basket_range: float = 0.15,
    drop_height: tuple[float, float] = (0.10, 0.16),
) -> None:
    """mode="interval" (every step): on the first step of an episode, record
    the start pocket, move the basket off the start pose if it overlaps it
    (_clear_basket) and, for SAFE goals, set the safe-zone point -- see
    safe_zone_point."""
    G = goal_state(env)
    todo = G.pending & (env.episode_length_buf >= 1)
    if not bool(todo.any()):
        return
    ids = todo.nonzero(as_tuple=True)[0]
    pocket, normal = pocket_and_normal(env, pocket_local)
    G.start[ids] = pocket[ids]
    _clear_basket(env, ids, pocket_local, basket_range, drop_height)
    safe = ids[G.kind[ids] == SAFE]
    if len(safe):
        d = torch.empty(len(safe), device=env.device).uniform_(*retreat_range)
        G.target[safe] = safe_zone_point(env, safe, d, pocket_local, body_margin, min_from_shoulder, apple_radius)
        G.normal[safe] = normal[safe]
        G.finger[safe] = palm_finger(env)[safe]
    G.pending[ids] = False


def _clear_basket(env, ids, pocket_local, basket_range, drop_height, apple_radius: float = 0.045,
                  margin: float = 0.03, tries: int = 20) -> None:
    """Starts are validated against the basket at its NOMINAL place; the
    episode's random basket (+-basket_range) can land on the arm. Redraw it
    (analytic obstacle only, so it may move) until arm, hand and held apple
    clear it by margin, and re-aim DROP goals at it."""
    from . import self_collision as scm

    G = goal_state(env)
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    sets = env._self_collision_spheres
    o = env.scene.env_origins[ids].unsqueeze(1)
    arm = scm._world_spheres(env, sets["arm"])[ids] - o
    pocket, _ = pocket_and_normal(env, pocket_local)
    pk = pocket[ids].unsqueeze(1) - o
    nominal = torch.tensor(BASKET_NOMINAL, device=env.device)
    for _ in range(tries):
        b = G.basket[ids]
        bad = (_basket_signed_dist(arm, b) < sets["arm"]["r"].view(1, -1) + margin).any(-1) | \
              ((_basket_signed_dist(pk, b)[:, 0] < apple_radius + margin) & G.carry[ids])
        if not bool(bad.any()):
            break
        k = bad.nonzero(as_tuple=True)[0]
        G.basket[ids[k]] = nominal + (torch.rand(len(k), 3, device=env.device) * 2 - 1) * basket_range
    drop = ids[G.kind[ids] == DROP]
    if len(drop):
        b = G.basket[drop]
        z = b[:, 2] + torch.empty(len(drop), device=env.device).uniform_(*drop_height)
        G.target[drop] = env.scene.env_origins[drop] + torch.stack(
            [b[:, 0] + DROP_OFFSET_XY[0], b[:, 1] + DROP_OFFSET_XY[1], z], dim=-1)


def safe_zone_point(env: "ManagerBasedRLEnv", ids: torch.Tensor, depth: torch.Tensor, pocket_local,
                    body_margin: float = 0.06, min_from_shoulder: float = 0.22, apple_radius: float = 0.045,
                    step: float = 0.01) -> torch.Tensor:
    """World pocket target straight back from where the hand is, along the
    way it came in (the horizontal part of -palm normal; the reach brings
    the palm in along it), up to `depth` -- but never so far that the palm
    or the held apple comes within body_margin of the body (self-collision
    sphere model) or the palm within min_from_shoulder of the shoulder:
    with the fruit 0.25-0.33 m from the shoulder, a fixed 20 cm retreat
    would put the hand against the chest. Shared with the harvest bench."""
    from . import self_collision as scm

    robot = env.scene["robot"]
    pocket, normal = pocket_and_normal(env, pocket_local)
    pocket, normal = pocket[ids], normal[ids]
    palm = robot.data.body_pos_w[ids, _ids(env, PALM)]
    sh = robot.data.body_pos_w[ids, _ids(env, SHOULDER)]
    back = -normal.clone()
    back[:, 2] = 0.0
    n_xy = back.norm(dim=-1, keepdim=True)
    back = torch.where(n_xy > 0.3, back / n_xy.clamp(min=1e-6), torch.tensor([-1.0, 0.0, 0.0], device=env.device))
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    body = scm._world_spheres(env, env._self_collision_spheres["body"])[ids]          # [n, S, 3]
    br = env._self_collision_spheres["body"]["r"]
    best = torch.zeros(len(ids), device=env.device)
    k = int(math.ceil(float(depth.max()) / step))
    for j in range(1, k + 1):
        d = torch.minimum(torch.full_like(depth, j * step), depth)
        shift = back * d.unsqueeze(-1)
        p_apple, p_palm = pocket + shift, palm + shift
        clear_a = ((body - p_apple.unsqueeze(1)).norm(dim=-1) - br - apple_radius).amin(-1)
        clear_p = ((body - p_palm.unsqueeze(1)).norm(dim=-1) - br - 0.03).amin(-1)
        ok = (clear_a >= body_margin) & (clear_p >= body_margin) & ((p_palm - sh).norm(dim=-1) >= min_from_shoulder)
        best = torch.where(ok & (d > best), d, best)
    return pocket + back * best.unsqueeze(-1)


def apply_payload(env: "ManagerBasedRLEnv", env_ids, pocket_local) -> None:
    """mode="interval" (every step): the held apple's weight at the pocket,
    on the palm link (zero when the hand is empty)."""
    G = goal_state(env)
    robot = env.scene["robot"]
    pocket, _ = pocket_and_normal(env, pocket_local)
    f = torch.zeros(env.num_envs, 1, 3, device=env.device)
    f[:, 0, 2] = -9.81 * G.mass * G.carry.float()
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=f, torques=torch.zeros_like(f), positions=pocket.unsqueeze(1),
        body_ids=torch.tensor([_ids(env, PALM)], device=env.device), is_global=True)


# ─────────────────────────────────────────────
#  Observations
# ─────────────────────────────────────────────

WAIST = "waist_yaw_joint"


def waist_yaw(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] waist yaw joint position (rad)."""
    cache = env.__dict__.setdefault("_arm_goal_ids", {})
    if WAIST not in cache:
        cache[WAIST] = env.scene["robot"].joint_names.index(WAIST)
    return env.scene["robot"].data.joint_pos[:, cache[WAIST]]


def _to_torso(env, v, unyaw: bool = False):
    """v in the torso frame -- with unyaw, the torso frame with the waist yaw
    taken out (waist_reach.palm_quat_torso_unyawed's reference: the same
    frame at yaw 0, fixed when only the torso turns)."""
    robot = env.scene["robot"]
    q = robot.data.body_quat_w[:, _ids(env, TORSO)]
    if unyaw:
        from isaaclab.utils.math import quat_conjugate, quat_from_euler_xyz, quat_mul
        qp = robot.data.body_quat_w[:, _ids(env, "pelvis")]
        yaw = waist_yaw(env)
        z = torch.zeros_like(yaw)
        q = quat_mul(quat_mul(quat_mul(qp, quat_from_euler_xyz(z, z, -yaw)), quat_conjugate(qp)), q)
    return quat_apply_inverse(q, v)


def obs_goal_pocket(env: "ManagerBasedRLEnv", pocket_local, unyaw: bool = False) -> torch.Tensor:
    """[N, 3] goal - pocket, torso frame (unyaw: without the waist yaw)."""
    G = goal_state(env)
    pocket, _ = pocket_and_normal(env, pocket_local)
    tgt = torch.where(G.pending.unsqueeze(-1), pocket, G.target)
    return _to_torso(env, tgt - pocket, unyaw)


def obs_goal_finger(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 3] goal finger direction in the PALM frame (1, 0, 0 = there)."""
    G = goal_state(env)
    cur = palm_finger(env)
    tgt = torch.where(G.pending.unsqueeze(-1), cur, G.finger)
    q = env.scene["robot"].data.body_quat_w[:, _ids(env, PALM)]
    return quat_apply_inverse(q, tgt)


def obs_goal_normal(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """[N, 3] goal palm-normal direction in the PALM frame (0, 1, 0 = there)."""
    G = goal_state(env)
    _, normal = pocket_and_normal(env, pocket_local)
    tgt = torch.where(G.pending.unsqueeze(-1), normal, G.normal)
    q = env.scene["robot"].data.body_quat_w[:, _ids(env, PALM)]
    return quat_apply_inverse(q, tgt)


def obs_basket(env: "ManagerBasedRLEnv", pocket_local, unyaw: bool = False) -> torch.Tensor:
    """[N, 3] basket axis at rim height - pocket, torso frame (unyaw: without the waist yaw)."""
    G = goal_state(env)
    pocket, _ = pocket_and_normal(env, pocket_local)
    b = G.basket + env.scene.env_origins
    return _to_torso(env, b - pocket, unyaw)


def obs_carrying(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return goal_state(env).carry.float().unsqueeze(-1)


# ─────────────────────────────────────────────
#  Rewards / termination
# ─────────────────────────────────────────────

def _errors(env, pocket_local):
    G = goal_state(env)
    pocket, normal = pocket_and_normal(env, pocket_local)
    d = (G.target - pocket).norm(dim=-1)
    cos = (G.normal * normal).sum(-1).clamp(-1.0, 1.0)
    return d, torch.acos(cos), ~G.pending


def reward_goal_pos(env: "ManagerBasedRLEnv", pocket_local, std: float) -> torch.Tensor:
    d, _, live = _errors(env, pocket_local)
    return torch.exp(-0.5 * (d / std) ** 2) * live.float()


def reward_goal_normal(env: "ManagerBasedRLEnv", pocket_local, std_rad: float, gate: float | None = None) -> torch.Tensor:
    d, ang, live = _errors(env, pocket_local)
    r = torch.exp(-0.5 * (ang / std_rad) ** 2)
    if gate is not None:
        r = r * torch.exp(-0.5 * (d / gate) ** 2)
    return r * live.float()


def finger_error(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] angle (rad) between the finger direction and the goal's."""
    G = goal_state(env)
    return torch.acos((G.finger * palm_finger(env)).sum(-1).clamp(-1.0, 1.0))


def reward_goal_finger(env: "ManagerBasedRLEnv", pocket_local, std_rad: float, gate: float | None = None) -> torch.Tensor:
    d, _, live = _errors(env, pocket_local)
    r = torch.exp(-0.5 * (finger_error(env) / std_rad) ** 2)
    if gate is not None:
        r = r * torch.exp(-0.5 * (d / gate) ** 2)
    return r * live.float()


def goal_reached(env: "ManagerBasedRLEnv", pocket_local, pos_tol: float = 0.02, ang_tol_deg: float = 20.0,
                 speed_tol: float = 0.15, hold_time: float = 0.3, finger_tol_deg: float = 30.0,
                 home_waist_tol_deg: float | None = None) -> torch.Tensor:
    """Termination (and cached for reward_success): pocket within pos_tol,
    normal within ang_tol, fingers within finger_tol, palm slower than
    speed_tol, held hold_time; HOME goals also with the waist yaw within
    home_waist_tol_deg of 0 when given (every reach starts there)."""
    G = goal_state(env)
    d, ang, live = _errors(env, pocket_local)
    v = env.scene["robot"].data.body_lin_vel_w[:, _ids(env, PALM)].norm(dim=-1)
    ok = live & (d < pos_tol) & (ang < math.radians(ang_tol_deg)) & (v < speed_tol) \
        & (finger_error(env) < math.radians(finger_tol_deg))
    if home_waist_tol_deg is not None:
        ok = ok & ((G.kind != HOME) | (waist_yaw(env).abs() < math.radians(home_waist_tol_deg)))
    G.hold = torch.where(ok, G.hold + env.step_dt, torch.zeros_like(G.hold))
    G.success = G.hold >= hold_time
    return G.success.clone()


def _body_overlap(env, pocket_local, apple_radius: float = 0.045) -> torch.Tensor:
    """[N] bool: arm/hand spheres, or the held apple, actually overlapping
    the body spheres or the stand + basket (margin 0)."""
    from . import self_collision as scm

    G = goal_state(env)
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    sets = env._self_collision_spheres
    arm_body = scm.self_collision_depth(env, margin=0.0).amax(-1) > 0.0
    body = scm._world_spheres(env, sets["body"])
    pocket, _ = pocket_and_normal(env, pocket_local)
    apple_body = (((body - pocket.unsqueeze(1)).norm(dim=-1) - sets["body"]["r"] - apple_radius).amin(-1) < 0.0) & G.carry
    o = env.scene.env_origins.unsqueeze(1)
    arm = scm._world_spheres(env, sets["arm"]) - o
    arm_basket = (_basket_signed_dist(arm, G.basket) < sets["arm"]["r"].view(1, -1)).any(-1)
    apple_basket = (_basket_signed_dist(pocket.unsqueeze(1) - o, G.basket)[:, 0] < apple_radius) & G.carry
    return arm_body | apple_body | arm_basket | apple_basket


def collision(env: "ManagerBasedRLEnv", pocket_local) -> torch.Tensor:
    """Termination: any real collision with the body or the basket ends the
    episode (and costs reward_collision) -- on the robot it would be a mess,
    so it must never pay to cut through, however briefly."""
    hit = _body_overlap(env, pocket_local) & ~goal_state(env).pending
    env._arm_goal_hit = hit
    return hit


def reward_collision(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return getattr(env, "_arm_goal_hit", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()


def reward_waist_home(env: "ManagerBasedRLEnv", std_rad: float = 0.15) -> torch.Tensor:
    """[N] HOME goals: the waist back at yaw 0 (where every reach starts)."""
    G = goal_state(env)
    return torch.exp(-0.5 * (waist_yaw(env) / std_rad) ** 2) * ((G.kind == HOME) & ~G.pending).float()


def reward_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return goal_state(env).success.float()


def penalty_path_deviation(env: "ManagerBasedRLEnv", pocket_local, deadband: float = 0.03, ref: float = 0.05) -> torch.Tensor:
    """[N] (distance of the pocket from the straight start->goal segment,
    beyond deadband) / ref, squared -- SAFE goals only: the way out of the
    fruit wall is straight back. For the basket and home the straight line
    can run through the torso (pick -> basket behind the right shoulder),
    and paying for it taught the first policy to cut through the body."""
    G = goal_state(env)
    pocket, _ = pocket_and_normal(env, pocket_local)
    a, b = G.start, G.target
    ab = b - a
    t = (((pocket - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
    dev = (pocket - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)
    # capped: a wild early policy would otherwise pay more here than everything else together
    return ((((dev - deadband).clamp(min=0.0) / ref) ** 2).clamp(max=4.0)) * ((~G.pending) & (G.kind == SAFE)).float()


def penalty_near_goal_speed(env: "ManagerBasedRLEnv", pocket_local, dist: float = 0.10) -> torch.Tensor:
    d, _, live = _errors(env, pocket_local)
    v = env.scene["robot"].data.body_lin_vel_w[:, _ids(env, PALM)].norm(dim=-1)
    return (v ** 2) * (d < dist).float() * live.float()


def penalty_apple_body(env: "ManagerBasedRLEnv", pocket_local, apple_radius: float = 0.045,
                       margin: float = 0.02, ref: float = 0.05) -> torch.Tensor:
    """[N] the held apple (a sphere at the pocket) into the body spheres."""
    from . import self_collision as scm

    G = goal_state(env)
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    body = scm._world_spheres(env, env._self_collision_spheres["body"])
    pocket, _ = pocket_and_normal(env, pocket_local)
    gap = (body - pocket.unsqueeze(1)).norm(dim=-1) - env._self_collision_spheres["body"]["r"] - apple_radius
    depth = (margin - gap).clamp(min=0.0).amax(-1)
    return ((depth / ref) ** 2) * G.carry.float()


def _basket_signed_dist(p_local: torch.Tensor, basket: torch.Tensor) -> torch.Tensor:
    """Signed distance [N, S] from points (env-local [N, S, 3]) to this env's
    stand + basket, solid up to the rim (basket.py's model, moved)."""
    from . import basket as bk

    dz = (basket[:, 2] - bk.RIM_Z).view(-1, 1)
    rad = torch.hypot(p_local[..., 0] - basket[:, 0:1], p_local[..., 1] - basket[:, 1:2])

    def cyl(radius, top):
        dr, dzz = rad - radius, p_local[..., 2] - (top + dz)
        return torch.hypot(dr.clamp(min=0.0), dzz.clamp(min=0.0)) + torch.maximum(dr, dzz).clamp(max=0.0)

    return torch.minimum(cyl(bk.OUTER_RADIUS, bk.RIM_Z), cyl(bk.STAND_RADIUS, bk.STAND_SIZE[2]))


def penalty_basket(env: "ManagerBasedRLEnv", pocket_local, apple_radius: float = 0.045,
                   margin: float = 0.015, ref: float = 0.05) -> torch.Tensor:
    """[N] arm/hand spheres (and the held apple) into the stand + basket --
    the hand only ever goes over the rim, never in."""
    from . import self_collision as scm

    G = goal_state(env)
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    sets = env._self_collision_spheres
    o = env.scene.env_origins.unsqueeze(1)
    arm = scm._world_spheres(env, sets["arm"]) - o
    pocket, _ = pocket_and_normal(env, pocket_local)
    depth = (sets["arm"]["r"].view(1, -1) + margin - _basket_signed_dist(arm, G.basket)).clamp(min=0.0).amax(-1)
    da = (apple_radius + margin - _basket_signed_dist(pocket.unsqueeze(1) - o, G.basket)).clamp(min=0.0)[:, 0]
    depth = torch.maximum(depth, da * G.carry.float())
    return (depth / ref) ** 2
