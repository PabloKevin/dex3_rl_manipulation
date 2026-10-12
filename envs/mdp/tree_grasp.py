# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Grasp on the multi-apple tree (envs/apple_grasp_tree_train_env_cfg.py):
the HOLD-UNDER-LOAD stage the single-apple grasp task did not have.

The harvest bench showed the grasp's weak point: the fruit is reached and
closed on, then slips out during the twist (the grip cannot transmit the
torque the stem resists with) or on the way home. Here, once the episode's
stem release comes (grasp_autoplay: deadline after the hand-off), the
target's BreakableStem lets go -- only the hand holds the fruit -- and a
load ramps up on it:
  - a TORQUE about the palm normal (the harvest twist axis), random sign,
    up to load_torque (the tree stems' twist-break range, 0.25-0.40 N*m);
  - a FORCE in a random direction, up to load_force (gravity, the arm's
    accelerations, the stem's pull just before it breaks);
both scaled by a training curriculum (0 until curriculum[0] iterations,
full from curriculum[1]). Rewards pay for keeping the fruit in the grasp
pocket while loaded and charge for the slip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PALM = "right_hand_palm_link"


def _state(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_load"):
        N, d = env.num_envs, env.device
        env._load = dict(
            t0=torch.full((N,), float("nan"), device=d),       # episode time the stem let go
            axis=torch.zeros(N, 3, device=d), fdir=torch.zeros(N, 3, device=d),
            sign=torch.ones(N, device=d), pocket0=torch.zeros(N, 3, device=d),
            mag_f=torch.ones(N, device=d), mag_t=torch.ones(N, device=d),     # this episode's share of the max
            hooked=False,
        )
    return env._load


def _scale(env: "ManagerBasedRLEnv", curriculum: tuple[int, int], steps_per_iter: int) -> float:
    it = env.common_step_counter / max(steps_per_iter, 1)
    a, b = curriculum
    return float(min(max((it - a) / max(b - a, 1), 0.0), 1.0))


def _ramp(env: "ManagerBasedRLEnv", delay: float, ramp: float) -> torch.Tensor:
    L = _state(env)
    t = env.episode_length_buf.float() * env.step_dt - L["t0"] - delay
    return torch.nan_to_num((t / ramp).clamp(0.0, 1.0), nan=0.0)


def reset_load(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset": no load until this episode's stem release."""
    L = _state(env)
    L["t0"][env_ids] = float("nan")
    if "broke" in L:                                   # stem-like twist (release_stem_and_twist)
        L["broke"][env_ids] = False
        L["broke_now"][env_ids] = False
        L["tau"][env_ids] = 0.0
        L["pull"][env_ids] = 0.0


def release_stem_and_load(
    env: "ManagerBasedRLEnv",
    env_ids,
    load_force: float = 6.0,
    load_torque: float = 0.4,
    # (per episode, uniform fractions of these maxima -- see the release)
    delay: float = 0.4,
    ramp: float = 1.0,
    curriculum: tuple[int, int] = (100, 800),
    steps_per_iter: int = 64,
) -> None:
    """mode="interval" (every step), after grasp_autoplay.update_first_closure:
    when env._stem_released turns on, let the target's stem go and arm the
    load (axis = palm normal now, random sign; force direction random). The
    wrench itself is applied every PHYSICS substep (hooked into
    scene.write_data_to_sim, like the stems)."""
    if not hasattr(env, "_tree") or not hasattr(env, "_stem_released"):
        return
    L = _state(env)
    T = env._tree
    robot = env.scene["robot"]
    apple = env.scene["apple"]
    if not hasattr(env, "_palm_id_load"):
        env._palm_id_load = robot.data.body_names.index(PALM)
    pid = env._palm_id_load
    new = env._stem_released & torch.isnan(L["t0"])
    if bool(new.any()):
        ids = new.nonzero(as_tuple=True)[0]
        T.target_stem.release(ids)
        L["t0"][ids] = env.episode_length_buf[ids].float() * env.step_dt
        pq = robot.data.body_quat_w[ids, pid]
        L["axis"][ids] = quat_apply(pq, torch.tensor([[0.0, 1.0, 0.0]], device=env.device).expand(len(ids), 3))
        L["sign"][ids] = torch.where(torch.rand(len(ids), device=env.device) < 0.5, -1.0, 1.0)
        # Each episode its own load, uniform up to the max: harvest loads vary
        # with the fruit and its stem, and some exceed what any grip holds --
        # all at the max left nothing to improve on once it could not hold.
        L["mag_f"][ids] = torch.rand(len(ids), device=env.device)
        L["mag_t"][ids] = torch.rand(len(ids), device=env.device)
        f = torch.randn(len(ids), 3, device=env.device)
        L["fdir"][ids] = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        L["pocket0"][ids] = quat_apply_inverse(pq, apple.data.root_pos_w[ids] - robot.data.body_pos_w[ids, pid])
    if not L["hooked"]:
        orig = env.scene.write_data_to_sim

        def write_with_load():
            s = _scale(env, curriculum, steps_per_iter) * _ramp(env, delay, ramp)        # [N]
            if bool((s > 0).any()):
                force = L["fdir"] * (load_force * s * L["mag_f"]).unsqueeze(-1)
                torque = L["axis"] * (load_torque * s * L["sign"] * L["mag_t"]).unsqueeze(-1)
                apple.instantaneous_wrench_composer.add_forces_and_torques(
                    forces=force.unsqueeze(1), torques=torque.unsqueeze(1), is_global=True)
            orig()

        env.scene.write_data_to_sim = write_with_load
        L["hooked"] = True


def gravity_in_palm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 3] unit gravity direction in the palm link's frame: which way the
    fruit's weight pulls relative to the hand. The one piece of the arm's
    pose that matters for holding, and it means the same thing whatever the
    arm does (twist, carry) -- unlike raw arm joint states. On the robot:
    forward kinematics + the torso IMU."""
    robot = env.scene["robot"]
    if not hasattr(env, "_palm_id_grav"):
        env._palm_id_grav = robot.data.body_names.index(PALM)
    q = robot.data.body_quat_w[:, env._palm_id_grav]
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    return quat_apply_inverse(q, g)


def _pocket_offset(env: "ManagerBasedRLEnv", center_local) -> torch.Tensor:
    robot = env.scene["robot"]
    pid = robot.data.body_names.index(PALM)
    pq = robot.data.body_quat_w[:, pid]
    rel = quat_apply_inverse(pq, env.scene["apple"].data.root_pos_w - robot.data.body_pos_w[:, pid])
    return rel - torch.tensor(center_local, device=env.device)


def reward_hold_under_load(env: "ManagerBasedRLEnv", center_local, tol: float = 0.02,
                           delay: float = 0.4, ramp: float = 1.0) -> torch.Tensor:
    """[N] per step while loaded: 1 with the fruit within tol of the grasp
    pocket (palm frame), fading to 0 at 3*tol; weighted by how much load is
    on (so it only pays for holding against it)."""
    if not hasattr(env, "_load"):
        return torch.zeros(env.num_envs, device=env.device)
    on = _ramp(env, delay, ramp)
    d = _pocket_offset(env, center_local).norm(dim=-1)
    held = (1.0 - ((d - tol) / (2 * tol)).clamp(0.0, 1.0))
    return on * held


def penalty_slip_under_load(env: "ManagerBasedRLEnv", ref: float = 0.02,
                            delay: float = 0.4, ramp: float = 1.0) -> torch.Tensor:
    """[N] (slip / ref)^2 while loaded: how far the fruit has moved in the
    palm since the stem let go."""
    if not hasattr(env, "_load"):
        return torch.zeros(env.num_envs, device=env.device)
    L = env._load
    robot = env.scene["robot"]
    pid = robot.data.body_names.index(PALM)
    pq = robot.data.body_quat_w[:, pid]
    rel = quat_apply_inverse(pq, env.scene["apple"].data.root_pos_w - robot.data.body_pos_w[:, pid])
    on = (_ramp(env, 0.0, 1e-3) > 0).float()          # from the release on
    return on * ((rel - L["pocket0"]).norm(dim=-1) / ref) ** 2


# ─────────────────────────────────────────────
#  Stem-like twist load (AppleGraspSpindle-v2)
# ─────────────────────────────────────────────
# Measured on AppleGraspSpindle-v1 (it 1100): the load above (twist up to
# 0.4 N*m HELD for ~5 s + up to 6 N) is harsher than a harvest -- even the
# policy's own grip, frozen, kept the fruit only 15% of the time at full
# twist -- while at 0.2 N*m the frozen grip held 60% and the acting policy
# 30%: it reacted its grip away. A real stem resists only until it breaks.

def release_stem_and_twist(
    env: "ManagerBasedRLEnv",
    env_ids,
    k_twist: tuple[float, float] = (0.25, 0.35),
    break_twist: tuple[float, float] = (0.25, 0.40),
    twist_deg: float = 90.0,
    twist_s: float = 1.5,
    load_force: float = 2.0,
    delay: float = 0.4,
    ramp: float = 1.0,
    grip_cmd: tuple[float, float] | None = (0.45, 0.75),
    curriculum: tuple[int, int] = (0, 300),
    steps_per_iter: int = 64,
    twist_fail_prob: float = 0.0,
    pull_break: tuple[float, float] | None = None,
    pull_s: float = 0.6,
) -> None:
    """mode="interval" (every step), after grasp_autoplay.update_first_closure,
    in place of release_stem_and_load. When the stem lets go (env.
    _stem_released), `delay` s later a harvest twist starts about the palm
    normal (random sign): the stem's resistance k * angle, the twist going
    twist_deg in twist_s (the bench's), until it reaches this episode's
    break torque -- the stem breaks, the twist load drops to 0 -- or the
    twist ends (unbroken: also 0). A steady carry force in a random
    direction, up to load_force (fruit inertia under the arm's moves), ramps
    in over `ramp` s and stays. k, break torque, force share: per episode.
    grip_cmd: the grip-force command (env._target_grip_level, an input of
    the policy) set to this fraction range of the episode's damage force at
    the release -- what the bench commands for the harvest. Both loads are
    scaled by a curriculum over iterations. Same env._load state as
    release_stem_and_load (rewards read it); + env._load["broke"] [N] bool
    (twist survived to the break) and ["broke_now"] (this step).

    With pull_break (AppleGraspSpindle-v3): when the twist ends with the
    stem still on -- in twist_fail_prob of the episodes the twist is made
    to stop at 80% of the break torque, as the bench's twist that a
    neighbour or the body blocks, or that the fruit slips through -- the
    bench's pull follows: a force on the fruit toward the tree along the
    palm normal (the stem holding it back while the hand pulls away),
    rising to this episode's pull-break force over pull_s, then gone (the
    stem broke). Measured on the bench (spindle setup, 56 apples): stems
    broke by pull 21 times, by twist 10; 10 of 17 drops were in the pull."""
    import math

    if not hasattr(env, "_tree") or not hasattr(env, "_stem_released"):
        return
    L = _state(env)
    T = env._tree
    robot = env.scene["robot"]
    apple = env.scene["apple"]
    N, dev = env.num_envs, env.device
    if "k" not in L:
        L.update(k=torch.zeros(N, device=dev), tb=torch.ones(N, device=dev),
                 broke=torch.zeros(N, dtype=torch.bool, device=dev),
                 broke_now=torch.zeros(N, dtype=torch.bool, device=dev),
                 fail=torch.zeros(N, dtype=torch.bool, device=dev), fb=torch.ones(N, device=dev),
                 pull=torch.zeros(N, device=dev))
    if not hasattr(env, "_palm_id_load"):
        env._palm_id_load = robot.data.body_names.index(PALM)
    pid = env._palm_id_load
    omega = math.radians(twist_deg) / twist_s
    new = env._stem_released & torch.isnan(L["t0"])
    if bool(new.any()):
        ids = new.nonzero(as_tuple=True)[0]
        n = len(ids)
        T.target_stem.release(ids)
        L["t0"][ids] = env.episode_length_buf[ids].float() * env.step_dt
        pq = robot.data.body_quat_w[ids, pid]
        L["axis"][ids] = quat_apply(pq, torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(n, 3))
        L["sign"][ids] = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        L["k"][ids] = torch.empty(n, device=dev).uniform_(*k_twist)
        L["tb"][ids] = torch.empty(n, device=dev).uniform_(*break_twist)
        L["broke"][ids] = False
        L["fail"][ids] = torch.rand(n, device=dev) < twist_fail_prob
        if pull_break is not None:
            L["fb"][ids] = torch.empty(n, device=dev).uniform_(*pull_break)
        L["mag_f"][ids] = torch.rand(n, device=dev)
        f = torch.randn(n, 3, device=dev)
        L["fdir"][ids] = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        L["pocket0"][ids] = quat_apply_inverse(pq, apple.data.root_pos_w[ids] - robot.data.body_pos_w[ids, pid])
        if grip_cmd is not None and hasattr(env, "_target_grip_level"):
            env._target_grip_level[ids] = torch.empty(n, device=dev).uniform_(*grip_cmd) * env._fruit_damage_force[ids]
    # twist torque this control step (the stem's resistance), and the break
    s = _scale(env, curriculum, steps_per_iter)
    t = env.episode_length_buf.float() * env.step_dt - L["t0"] - delay
    t = torch.nan_to_num(t, nan=-1.0)
    twisting = (t >= 0) & (t <= twist_s) & ~L["broke"]
    tau = s * L["k"] * omega * t.clamp(min=0.0)
    tau = torch.where(L["fail"], torch.minimum(tau, 0.8 * s * L["tb"]), tau)      # a blocked / slipping twist
    L["broke_now"] = twisting & (tau >= s * L["tb"]) & (s > 0)
    # the pull after an unbroken twist: up to the pull-break force over pull_s
    if pull_break is not None:
        tp = t - twist_s
        pulling = (tp >= 0) & (tp <= pull_s + 0.5) & ~L["broke"] & (s > 0)
        fpull = s * L["fb"] * (tp / pull_s).clamp(0.0, 1.0)
        brk = pulling & (tp >= pull_s)
        L["broke_now"] = L["broke_now"] | brk
        L["pull"] = torch.where(pulling & ~brk, fpull, torch.zeros_like(fpull))
    L["broke"] |= L["broke_now"]
    L["tau"] = torch.where(twisting & ~L["broke"], tau, torch.zeros_like(tau))
    if not L["hooked"]:
        orig = env.scene.write_data_to_sim

        def write_with_load():
            sc = _scale(env, curriculum, steps_per_iter)
            on = _ramp(env, delay, ramp)
            if sc > 0 and bool((L["tau"] > 0).any() | (on > 0).any()):
                force = L["fdir"] * (load_force * sc * on * L["mag_f"]).unsqueeze(-1) \
                    + L["axis"] * L["pull"].unsqueeze(-1)              # the stem holding back toward the tree
                torque = L["axis"] * (L["tau"] * L["sign"]).unsqueeze(-1)
                apple.instantaneous_wrench_composer.add_forces_and_torques(
                    forces=force.unsqueeze(1), torques=torque.unsqueeze(1), is_global=True)
            orig()

        env.scene.write_data_to_sim = write_with_load
        L["hooked"] = True


def reward_twist_survived(env: "ManagerBasedRLEnv", center_local, tol: float = 0.03) -> torch.Tensor:
    """[N] one-off: the stem broke this step (the twist reached its break
    torque) with the fruit still within tol of the grasp pocket -- the
    harvest's decisive moment."""
    L = getattr(env, "_load", None)
    if L is None or "broke_now" not in L:
        return torch.zeros(env.num_envs, device=env.device)
    return (L["broke_now"] & (_pocket_offset(env, center_local).norm(dim=-1) < tol)).float()
