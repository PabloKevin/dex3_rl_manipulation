# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Fruit-aware goals for the point-to-point arm policy (AppleArmPose-v5,
envs/apple_arm_pose_v5_env_cfg.py), on top of envs/mdp/arm_pose.py.

Why (sim harvest, 2026-11-13, scripts/play_harvest_robot.py, 16 trees): the
tree-blind reach arrived within 2 cm of where the apple WAS in 15 of 16
tries, but in 4 the arm had swept the hanging fruit 6-16 cm away on the way
("nothing between the fingers"), and in 6 it buzzed around the goal until
the timeout. The policy never knew that a goal is a fruit.

A FRUIT goal is a stop goal with a fruit hanging at the goal point: the
grasp pocket must arrive there with the fruit entering the cupped hand
through its opening -- nothing else of the hand or forearm may pass through
the place the fruit occupies. No physics: a virtual sphere at the goal
(radius per goal, the apples' 32-48 mm) against the arm / hand sphere model
(self_collision.py). A sphere of the hand may be as deep in the fruit as it
would be with the fruit sitting in the pocket (the cup closes round it), so
the goal pose itself and the straight way in along the palm normal are
free; any other way in costs (penalty_fruit_sweep). The policy is told that
the goal is a fruit by one flag -- on the robot: "this goal is the detected
apple", against staging / basket / home goals.

Tracked fruit (a share of the fruit goals): the goal point jumps a few cm
now and then, before and after arrival -- a new detection of a fruit the
hand or a neighbour moved, or the detector's own jitter. The policy has to
follow short corrections and settle again (the v1 policy stalled 6-18 cm
off on short goals: memory note of 2026-11-01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from . import arm_pose as ap
from . import self_collision as scm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _Fruit:
    def __init__(self, env: "ManagerBasedRLEnv"):
        N, d = env.num_envs, env.device
        self.on = torch.zeros(N, dtype=torch.bool, device=d)
        self.tracked = torch.zeros(N, dtype=torch.bool, device=d)
        self.r = torch.full((N,), 0.04, device=d)
        self.excess = torch.zeros(N, device=d)             # this step's deepest excess (m)
        self.worst = torch.zeros(N, device=d)              # deepest excess of the current goal
        # per-episode counters (metrics)
        self.n_fruit = torch.zeros(N, device=d)
        self.n_swept = torch.zeros(N, device=d)
        self.n_fruit_reached = torch.zeros(N, device=d)


def fruit_state(env: "ManagerBasedRLEnv") -> _Fruit:
    if not hasattr(env, "_arm_pose_fruit"):
        env._arm_pose_fruit = _Fruit(env)
    return env._arm_pose_fruit


def reset_fruit(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset": no fruit until the first goal is drawn."""
    F = fruit_state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    F.on[ids] = False
    F.tracked[ids] = False
    for t in (F.excess, F.worst, F.n_fruit, F.n_swept, F.n_fruit_reached):
        t[ids] = 0.0


def assign_fruit(
    env: "ManagerBasedRLEnv",
    env_ids,
    pocket_local,
    fruit_prob: float = 0.7,
    tracked_prob: float = 0.5,
    radius_range: tuple[float, float] = (0.032, 0.048),
    max_tilt_deg: float = 35.0,
    min_forward: float = 0.10,
    finger_free_prob: float = 0.7,
    jump_every_s: float = 2.5,
    jump_range: tuple[float, float] = (0.005, 0.03),
    swept_depth: float = 0.02,
) -> None:
    """mode="interval" (every step, right AFTER arm_pose.advance_goals):
    the goals drawn this step become fruit goals with fruit_prob when they
    are stop goals where fruit hangs for this robot -- in front of the
    shoulder, the palm normal within max_tilt_deg of horizontal (torso
    frame). A fruit goal's fingers are free with finger_free_prob (the
    harvest leaves them free). Then the tracked fruit goals jump: on
    average once per jump_every_s, by jump_range metres in a random
    direction (the progress shaping's previous distance moved with it, so a
    jump pays nothing)."""
    P = ap.pose_state(env)
    F = fruit_state(env)
    dev = env.device
    live = ~P.pending
    new = live & (P.t_goal == 0.0)
    if bool(new.any()):
        ids = new.nonzero(as_tuple=True)[0]
        n = len(ids)
        # the goal that just ended: counted if it was a fruit
        was = F.on[ids]
        F.n_fruit[ids] += was.float()
        F.n_swept[ids] += (was & (F.worst[ids] > swept_depth)).float()
        _, tq = ap._torso(env)
        nrm_t = quat_apply_inverse(tq[ids], P.normal[ids])
        sh = env.scene["robot"].data.body_pos_w[ids, ap._bid(env, ap.SHOULDER)]
        rel = quat_apply_inverse(tq[ids], P.target[ids] - sh)
        tilt_ok = nrm_t[:, 2].abs() <= torch.sin(torch.deg2rad(torch.tensor(max_tilt_deg, device=dev)))
        like = tilt_ok & (rel[:, 0] >= min_forward) & (nrm_t[:, 0] > 0.0) & ~P.via[ids]
        on = like & (torch.rand(n, device=dev) < fruit_prob)
        F.on[ids] = on
        F.tracked[ids] = on & (torch.rand(n, device=dev) < tracked_prob)
        lo, hi = radius_range
        F.r[ids] = lo + (hi - lo) * torch.rand(n, device=dev)
        F.worst[ids] = 0.0
        free = on & (torch.rand(n, device=dev) < finger_free_prob)
        P.finger_on[ids] = P.finger_on[ids] & ~free
        # a fruit is flown round to its front: the straight start -> goal line is not the way in
        P.straight_ok[ids] = P.straight_ok[ids] & ~on
    # a new detection of a tracked fruit
    jump = live & ~new & F.on & F.tracked & (torch.rand(env.num_envs, device=dev) < env.step_dt / jump_every_s)
    if bool(jump.any()):
        ids = jump.nonzero(as_tuple=True)[0]
        n = len(ids)
        pocket, normal, _ = ap.hand_frame(env, pocket_local)
        d0 = (P.target[ids] - pocket[ids]).norm(dim=-1)
        step = torch.randn(n, 3, device=dev)
        step = step / step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        lo, hi = jump_range
        P.target[ids] = P.target[ids] + step * (lo + (hi - lo) * torch.rand(n, 1, device=dev))
        d1 = (P.target[ids] - pocket[ids]).norm(dim=-1)
        P.prev_d[ids] = P.prev_d[ids] + (d1 - d0)


def _excess(env: "ManagerBasedRLEnv", pocket_local, deadband: float) -> torch.Tensor:
    """[N, S] how much deeper each arm / hand sphere is in the fruit (at the
    goal) than it would be with the fruit in the pocket, beyond deadband."""
    P = ap.pose_state(env)
    F = fruit_state(env)
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    arm = env._self_collision_spheres["arm"]
    c = scm._world_spheres(env, arm)                                   # [N, S, 3]
    pocket, _, _ = ap.hand_frame(env, pocket_local)
    need = F.r.unsqueeze(-1) + arm["r"].unsqueeze(0)                   # [N, S]
    in_fruit = (need - (c - P.target.unsqueeze(1)).norm(dim=-1)).clamp(min=0.0)
    in_pocket = (need - (c - pocket.unsqueeze(1)).norm(dim=-1)).clamp(min=0.0)
    on = (F.on & ~P.pending).float().unsqueeze(-1)
    return (in_fruit - in_pocket - deadband).clamp(min=0.0) * on


def penalty_fruit_sweep(env: "ManagerBasedRLEnv", pocket_local, deadband: float = 0.01, ref: float = 0.02,
                        cap: float = 4.0) -> torch.Tensor:
    """[N] sum over the arm / hand spheres of (excess depth in the fruit /
    ref)^2, capped. Zero with the fruit in the pocket and on the straight
    way in along the palm normal (deadband: the fingertips pass ~1 cm
    nearer the fruit's centre on the way in than once it sits in the cup)."""
    F = fruit_state(env)
    e = _excess(env, pocket_local, deadband)
    F.excess = e.amax(dim=-1)
    F.worst = torch.maximum(F.worst, F.excess)
    return ((e / ref) ** 2).sum(dim=-1).clamp(max=cap)


def obs_goal_flags(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 3] (finger direction counts, pass-through waypoint, the goal is a fruit)."""
    P = ap.pose_state(env)
    F = fruit_state(env)
    return torch.stack([P.finger_on.float(), P.via.float(), (F.on & ~P.pending).float()], dim=-1)


def fruit_metrics(env: "ManagerBasedRLEnv", env_ids: Sequence[int], ema: float = 0.05) -> dict:
    """Curriculum slot (logs only): share of the finished episodes' fruit
    goals whose fruit was swept (an arm / hand sphere > swept_depth deeper
    than the cup allows), and the fruit goals per episode."""
    st = env.__dict__.setdefault("_arm_pose_fruit_metrics", {})
    if not hasattr(env, "_arm_pose_fruit") or env.common_step_counter < 1:
        return dict(st)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return dict(st)
    F = env._arm_pose_fruit
    n = F.n_fruit[ids].sum()
    vals = {"fruit_goals_per_episode": float(F.n_fruit[ids].mean())}
    if float(n) > 0:
        vals["swept_share"] = float(F.n_swept[ids].sum() / n)
    for k, m in vals.items():
        st[k] = m if k not in st else (1.0 - ema) * st[k] + ema * m
    return dict(st)


# ─────────────────────────────────────────────
#  The fruit reach inside another task (AppleGraspArmPose-v2's reach phase, the harvest bench)
# ─────────────────────────────────────────────

def approach_normal(u: torch.Tensor, heading: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """[n, 3] the horizontal unit direction u turned by `heading` about the vertical (+ = to the left)
    and raised by `tilt` (+ = the palm looks up)."""
    c, s = torch.cos(heading), torch.sin(heading)
    h = torch.stack([c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1], torch.zeros_like(c)], dim=-1)
    out = h * torch.cos(tilt).unsqueeze(-1)
    out[:, 2] = torch.sin(tilt)
    return out


def fruit_reach_step(
    env: "ManagerBasedRLEnv",
    reaching: torch.Tensor,
    pocket_local,
    goal_offset_axial: tuple[float, float] = (0.0, 0.0),
    goal_offset_cross: float = 0.0,
    heading_deg: tuple[float, float] = (0.0, 0.0),
    tilt_deg: tuple[float, float] = (0.0, 0.0),
    track: dict | None = None,
    follow: bool = True,
    aim_map: str | None = None,
    aim_prob: float = 1.0,
) -> None:
    """Once per control step, before the frozen AppleArmPose-v5 policy's inputs are built: ONE fruit goal --
    the pocket on the PERCEIVED apple (env._apple_meas, else the true one) moved by a calibration-like offset
    (axial along the approach U(goal_offset_axial), sideways / vertical U(+-goal_offset_cross)), the palm
    normal along the horizontal torso -> apple line turned by U(heading_deg) and raised by U(tilt_deg)
    (drawn once per episode) -- or, with aim_map (envs/mdp/aim.py, in aim_prob of the episodes), the
    heading / tilt the aim map chooses for this apple at the episode's waist yaw, as the harvest does --
    fingers free, the fruit flag on. With `follow`, the goal follows the
    perceived apple every step (the camera tracking refreshes env._apple_meas; without tracking it is
    constant anyway). arm_pose.plan_state(env).t_at_goal counts the time since the goal was reached
    (arm_pose.track_goal), for the hand-off."""
    P = ap.pose_state(env)
    S = ap.plan_state(env)
    F = fruit_state(env)
    dev = env.device
    if not hasattr(S, "offset"):
        S.offset = torch.zeros(env.num_envs, 3, device=dev)
    buf = env.episode_length_buf
    fresh = buf <= 0
    if bool(fresh.any()):
        S.wp[fresh] = -1
        S.t_wp[fresh] = 0.0
        S.t_at_goal[fresh] = 0.0
        P.pending[fresh] = True
        P.arrived[fresh] = False
        P.hold[fresh] = 0.0
        F.on[fresh] = False
    meas = getattr(env, "_apple_meas", None)
    apple_all = meas if meas is not None else env.scene["apple"].data.root_pos_w
    todo = (S.wp < 0) & (buf >= 1) & reaching
    if bool(todo.any()):
        ids = todo.nonzero(as_tuple=True)[0]
        n = len(ids)
        apple = apple_all[ids]
        tp, _ = ap._torso(env)
        u = apple - tp[ids]
        u[:, 2] = 0.0
        u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rad = lambda r: torch.deg2rad(r[0] + (r[1] - r[0]) * torch.rand(n, device=dev))
        nrm = approach_normal(u, rad(heading_deg), rad(tilt_deg))
        if aim_map is not None:
            if not hasattr(env, "_aim_map"):
                from . import aim as aim_mod

                env._aim_map = aim_mod.load(aim_map, dev)
            _, tq = ap._torso(env)
            yaw = env.scene["robot"].data.joint_pos[ids, ap._jid(env, ap.WAIST)]
            res = env._aim_map.choose(apple - tp[ids], yaw, turn_waist=False)
            use = torch.rand(n, device=dev) < aim_prob
            nrm = torch.where(use.unsqueeze(-1), quat_apply(tq[ids], res["normal_torso"]), nrm)
        side = torch.stack([-nrm[:, 1], nrm[:, 0], torch.zeros_like(nrm[:, 0])], dim=-1)
        side = side / side.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        up = torch.linalg.cross(nrm, side)
        lo, hi = goal_offset_axial
        ax = lo + (hi - lo) * torch.rand(n, device=dev)
        cr = (2 * torch.rand(n, 2, device=dev) - 1) * goal_offset_cross
        S.offset[ids] = ax.unsqueeze(-1) * nrm + cr[:, :1] * side + cr[:, 1:] * up
        S.goal[ids], S.apple[ids] = apple + S.offset[ids], apple
        P.target[ids], P.normal[ids] = S.goal[ids], nrm
        P.finger[ids] = ap.hand_frame(env, pocket_local)[2][ids]
        P.finger_on[ids] = False
        P.via[ids] = False
        P.start[ids] = ap.hand_frame(env, pocket_local)[0][ids]
        P.pending[ids] = False
        P.arrived[ids] = False
        P.hold[ids] = 0.0
        F.on[ids] = True
        S.wp[ids] = 1
        S.t_wp[ids] = 0.0
    live = reaching & (S.wp >= 0)
    if not bool(live.any()):
        return
    if follow:
        S.apple = torch.where(live.unsqueeze(-1), apple_all, S.apple)
        S.goal = torch.where(live.unsqueeze(-1), apple_all + S.offset, S.goal)
        P.target = torch.where(live.unsqueeze(-1), S.goal, P.target)
    ap.track_goal(env, pocket_local, **(track or {}))
    dt = env.step_dt
    S.t_wp = S.t_wp + dt * live.float()
    at = live & P.arrived & P.in_tol
    S.t_at_goal = torch.where(at, S.t_at_goal + dt, torch.zeros_like(S.t_at_goal))


def set_arm_start_choice(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, joint_names: list[str],
                         poses: list[list[float]], jitter: float = 0.0, clip: dict | None = None) -> None:
    """mode="reset" (after the joint resets): the named joints teleported to one of `poses` (random per env)
    + U(-jitter, jitter) rad, inside `clip` (joint -> (lo, hi)), at rest, PD targets on it."""
    robot = env.scene["robot"]
    dev = env.device
    ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long)
    if not hasattr(env, "_arm_choice"):
        jid = torch.tensor([robot.joint_names.index(j) for j in joint_names], device=dev)
        P = torch.tensor(poses, device=dev, dtype=torch.float32)
        lo = torch.tensor([(clip or {}).get(j, (-1e9, 1e9))[0] for j in joint_names], device=dev)
        hi = torch.tensor([(clip or {}).get(j, (-1e9, 1e9))[1] for j in joint_names], device=dev)
        env._arm_choice = (jid, P, lo, hi)
    jid, P, lo, hi = env._arm_choice
    n = len(ids)
    q = P[torch.randint(0, P.shape[0], (n,), device=dev)] + jitter * (2 * torch.rand(n, P.shape[1], device=dev) - 1)
    q = torch.maximum(torch.minimum(q, hi), lo)
    jp = robot.data.joint_pos[ids].clone()
    jv = robot.data.joint_vel[ids].clone()
    tgt = robot.data.joint_pos_target[ids].clone()
    jp[:, jid] = q
    jv[:, jid] = 0.0
    tgt[:, jid] = q
    robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    robot.set_joint_position_target(tgt, env_ids=ids)


def apple_pos_pocket_until_release(env: "ManagerBasedRLEnv", object_cfg, robot_cfg, ref_local=(0.0, 0.0, 0.0)):
    """[N, 3] single_reach.apple_pos_pocket_measured (the perceived apple in the palm frame, minus the pocket)
    until the stem lets go, zero after: the fruit is then in the hand or lost, and the camera no longer tells."""
    from .grasp_autoplay import stem_released
    from .single_reach import apple_pos_pocket_measured

    v = apple_pos_pocket_measured(env, object_cfg, robot_cfg, ref_local)
    return v * (~stem_released(env)).float().unsqueeze(-1)


def handoff_metrics(env: "ManagerBasedRLEnv", env_ids: Sequence[int], pocket_local, ema: float = 0.05) -> dict:
    """Curriculum slot (logs only), on the episodes ending: share that handed off by ARRIVAL (the fruit goal
    reached and kept) rather than by the watchdog, the hand-off time (s), and the true fruit's distance from
    the pocket at the hand-off (cm) -- how good the frozen reach's hand-offs are."""
    st = env.__dict__.setdefault("_handoff_metrics", {})
    if not hasattr(env, "_grasp_handoff_step") or env.common_step_counter < 1:
        return dict(st)
    rec = env.__dict__.get("_handoff_rec")
    if rec is None:
        return dict(st)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ok = rec["seen"][ids]
    if not bool(ok.any()):
        return dict(st)
    vals = {
        "arrival_share": float(rec["arrived"][ids][ok].float().mean()),
        "handoff_s": float(rec["t"][ids][ok].mean()),
        "fruit_pocket_cm": float(rec["d"][ids][ok].mean() * 100.0),
        "fruit_within_3cm": float((rec["d"][ids][ok] < 0.03).float().mean()),
    }
    rec["seen"][ids] = False
    for k, m in vals.items():
        st[k] = m if k not in st else (1.0 - ema) * st[k] + ema * m
    return dict(st)


def record_handoff(env: "ManagerBasedRLEnv", env_ids, pocket_local) -> None:
    """mode="interval" (every step, after update_reach_grasp_phase): the hand-off's numbers for handoff_metrics."""
    if not hasattr(env, "_grasp_phase_active"):
        return
    N, dev = env.num_envs, env.device
    rec = env.__dict__.get("_handoff_rec")
    if rec is None:
        rec = env._handoff_rec = {"seen": torch.zeros(N, dtype=torch.bool, device=dev),
                                  "prev": torch.zeros(N, dtype=torch.bool, device=dev),
                                  "arrived": torch.zeros(N, dtype=torch.bool, device=dev),
                                  "t": torch.zeros(N, device=dev), "d": torch.zeros(N, device=dev)}
    now = env._grasp_phase_active
    new = now & ~rec["prev"]
    if bool(new.any()):
        ids = new.nonzero(as_tuple=True)[0]
        pocket, _, _ = ap.hand_frame(env, pocket_local)
        rec["seen"][ids] = True
        rec["arrived"][ids] = ap.pose_state(env).arrived[ids]
        rec["t"][ids] = env.episode_length_buf[ids].float() * env.step_dt
        rec["d"][ids] = (env.scene["apple"].data.root_pos_w[ids] - pocket[ids]).norm(dim=-1)
    rec["prev"] = now.clone()


def beyond_handoff(env: "ManagerBasedRLEnv", module: str, term: str, inner: dict) -> torch.Tensor:
    """[N] reward term envs.mdp.<module>.<term>(env, **inner), minus its value at this episode's hand-off,
    clamped at 0, in the grasp phase only: what the TRAINEE added to a cost the frozen reach already left
    (the v5 reach presses the cup onto the fruit: the fruit hangs 1-5 cm off its spot at the hand-off, and
    charging that every step drowned every other term: -11 of a -19 per-second total in the first run)."""
    import importlib

    from .grasp_autoplay import grasp_phase_active

    fn = getattr(importlib.import_module(f"{__package__}.{module}"), term)
    v = fn(env, **inner)
    st = env.__dict__.setdefault("_beyond_handoff", {})
    key = f"{module}.{term}"
    active = grasp_phase_active(env)
    if key not in st:
        st[key] = {"v0": torch.zeros_like(v), "prev": torch.zeros_like(active)}
    s = st[key]
    new = active & ~s["prev"]
    s["v0"] = torch.where(new, v, s["v0"])
    s["prev"] = active.clone()
    return (v - s["v0"]).clamp(min=0.0) * active.float()
