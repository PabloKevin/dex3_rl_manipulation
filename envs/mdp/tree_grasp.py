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
