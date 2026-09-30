# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Harvest phase building blocks: a breakable stem, the "grasp ready" check
that hands control from the grasp policy to the harvest motion, and the
scripted harvest trajectory (palm poses + differential IK).

Used by scripts/harvest_scripted.py now; meant to be reused by an RL
harvest task later (the stem and the ready check do not depend on who
drives the arm).

Stem model (a first, common-sense approach -- no measured reference yet,
every number is a per-episode randomization range):
  The stem is a short rod rigidly attached to the apple (pedicels stay on
  the fruit), joined to the branch at a PIVOT P a stem length above the
  apple's top. The joint at P is a 6-DoF spring-damper:
    - translation of the rod's end away from P  -> force  (pulling),
    - swing of the apple about P                -> bending moment,
    - rotation of the apple about the stem axis -> twisting moment.
  It breaks the first time any of the three exceeds its threshold, and
  from then on applies nothing. Rotating the fruit about P (the picker's
  lift-and-roll) loads bending, not tension, and bending breaks at a much
  lower hand force than pulling -- which is the point of the motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    normalize,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Breakable stem
# ─────────────────────────────────────────────

@dataclass
class StemParams:
    """Uniform (lo, hi) ranges, sampled per env when the stem is engaged.

    Rough reasoning behind the defaults: apple fruit-removal force by a
    straight pull is commonly quoted around 15-25 N; bending or twisting
    the stem needs far less hand force. Break moments are chosen so that
    bending breaks at ~5 N tangential force on a ~7 cm lever (fruit centre
    to pivot) and twisting at ~0.1 N*m, i.e. ~20-35 deg of stem rotation.
    Replace with measured values when available.
    """
    length: tuple[float, float] = (0.02, 0.04)              # m, apple top -> pivot
    k_pull: tuple[float, float] = (300.0, 600.0)             # N/m, along the stem (stem + branch give)
    k_lateral: tuple[float, float] = (100.0, 300.0)          # N/m, sideways (the branch sways)
    k_bend: tuple[float, float] = (0.6, 1.0)                 # N*m/rad
    k_twist: tuple[float, float] = (0.2, 0.4)                # N*m/rad
    break_pull: tuple[float, float] = (15.0, 25.0)           # N
    break_bend: tuple[float, float] = (0.25, 0.45)           # N*m
    break_twist: tuple[float, float] = (0.08, 0.15)          # N*m
    damping_ratio: float = 0.5                               # of critical, for all 3 springs


class BreakableStem:
    """Per-env breakable stem acting on a RigidObject through its
    instantaneous wrench composer. Call apply() once per PHYSICS substep
    (e.g. from an action term's apply_actions): at 200 Hz a ~1000 N/m
    spring on a 0.15 kg fruit is stable, at the 50 Hz control rate it is
    not."""

    def __init__(self, env: "ManagerBasedRLEnv", object_name: str = "apple",
                 params: StemParams | None = None, apple_radius: float = 0.04):
        self.env = env
        self.obj: RigidObject = env.scene[object_name]
        self.p = params or StemParams()
        self.radius = apple_radius
        n, d = env.num_envs, env.device
        z = lambda: torch.zeros(n, device=d)
        self.active = torch.zeros(n, dtype=torch.bool, device=d)
        self.broken = torch.zeros(n, dtype=torch.bool, device=d)
        self.break_cause = torch.zeros(n, dtype=torch.long, device=d)   # 0 none, 1 pull, 2 bend, 3 twist
        self.pivot = torch.zeros(n, 3, device=d)
        self.axis = torch.zeros(n, 3, device=d)           # unit, apple -> pivot, at rest
        self.end_local = torch.zeros(n, 3, device=d)      # rod end (at P) in apple body frame
        self.quat0 = torch.zeros(n, 4, device=d)
        self.k_pull, self.k_lat, self.k_bend, self.k_twist = z(), z(), z(), z()
        self.c_pull, self.c_bend, self.c_twist = z(), z(), z()
        self.f_break, self.m_bend_break, self.m_twist_break = z(), z(), z()
        # live readouts (last substep)
        self.pull, self.bend, self.twist = z(), z(), z()
        self.bend_deg, self.twist_deg = z(), z()
        self.peak_pull, self.peak_bend, self.peak_twist = z(), z(), z()

    def _u(self, rng, ids):
        return torch.empty(len(ids), device=self.env.device).uniform_(*rng)

    def engage(self, env_ids: torch.Tensor, axis_w: torch.Tensor | None = None,
               pos: torch.Tensor | None = None, quat: torch.Tensor | None = None,
               preload: torch.Tensor | None = None) -> None:
        """Attach the stem to the apple's CURRENT pose (or the given pos /
        quat, e.g. one just written at reset) with no preload: the pivot
        sits `length` beyond the apple surface along axis_w (default world
        +Z, i.e. hanging from above)."""
        ids = env_ids
        if len(ids) == 0:
            return
        pos = self.obj.data.root_pos_w[ids] if pos is None else pos
        quat = self.obj.data.root_quat_w[ids] if quat is None else quat
        radius = getattr(self.env, "_apple_radius", None)
        if isinstance(radius, torch.Tensor):
            self.radius_env = radius
        r = self.radius_env[ids] if hasattr(self, "radius_env") else torch.full((len(ids),), self.radius, device=self.env.device)
        if axis_w is None:
            axis_w = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(len(ids), 3)
        axis_w = normalize(axis_w)
        length = self._u(self.p.length, ids)
        self.pivot[ids] = pos + axis_w * (r + length).unsqueeze(-1)
        self.axis[ids] = axis_w
        self.end_local[ids] = quat_apply(quat_conjugate(quat), self.pivot[ids] - pos)
        self.quat0[ids] = quat
        try:
            m = float(self.obj.cfg.spawn.mass_props.mass)
        except (AttributeError, TypeError):
            m = 0.15
        inertia = 0.4 * m * self.radius ** 2
        zeta = self.p.damping_ratio
        self.k_pull[ids] = self._u(self.p.k_pull, ids)
        self.k_lat[ids] = self._u(self.p.k_lateral, ids)
        self.k_bend[ids] = self._u(self.p.k_bend, ids)
        self.k_twist[ids] = self._u(self.p.k_twist, ids)
        self.c_pull[ids] = 2 * zeta * torch.sqrt(self.k_pull[ids] * m)
        self.c_bend[ids] = 2 * zeta * torch.sqrt(self.k_bend[ids] * inertia)
        self.c_twist[ids] = 2 * zeta * torch.sqrt(self.k_twist[ids] * inertia)
        self.f_break[ids] = self._u(self.p.break_pull, ids)
        self.m_bend_break[ids] = self._u(self.p.break_bend, ids)
        self.m_twist_break[ids] = self._u(self.p.break_twist, ids)
        if preload is not None:
            # Start already applying `preload` (world N): shift the pivot so
            # the springs are stretched by K^-1 * preload at this pose.
            f_ax = (preload * axis_w).sum(-1, keepdim=True) * axis_w
            shift = f_ax / self.k_pull[ids].unsqueeze(-1) + (preload - f_ax) / self.k_lat[ids].unsqueeze(-1)
            self.pivot[ids] = self.pivot[ids] + shift
        self.active[ids] = True
        self.broken[ids] = False
        self.break_cause[ids] = 0
        for t in (self.peak_pull, self.peak_bend, self.peak_twist):
            t[ids] = 0.0

    def release(self, env_ids: torch.Tensor) -> None:
        self.active[env_ids] = False

    def apply(self) -> None:
        on = self.active & ~self.broken
        if not bool(on.any()):
            return
        d = self.obj.data
        pos, quat = d.root_pos_w, d.root_quat_w
        v, w = d.root_lin_vel_w, d.root_ang_vel_w

        # -- pull: rod end vs pivot, applied AT THE CENTRE OF MASS --
        # Applied at the rod end (7 cm from the centre) the pull spring
        # also acts as a rotational spring of k_pull * r^2 ~ 5 N*m/rad (and
        # its damper as c * r^2) on a fruit whose inertia is ~1e-4 kg*m^2:
        # far too stiff for the 5 ms physics step, and it blew up -- 20 deg
        # of spurious rotation and 27 N in one control step with the hand
        # standing still. Rotation is carried by the bend/twist springs
        # instead, which are stable at this step (omega*dt < 0.5).
        r_end = quat_apply(quat, self.end_local)                          # COM -> rod end
        end = pos + r_end
        # Axial (along the rest stem axis) and lateral parts: the stem
        # itself barely stretches, but the spur and branch it hangs from
        # sway sideways several cm under a few N. One isotropic ~1000 N/m
        # spring made a 1 cm IK tracking error worth 10 N, which pulled
        # fruit out of the hand with the stem intact.
        disp = end - self.pivot
        d_ax = (disp * self.axis).sum(-1, keepdim=True) * self.axis
        f_spring = -self.k_pull.unsqueeze(-1) * d_ax - self.k_lat.unsqueeze(-1) * (disp - d_ax)
        f = f_spring - self.c_pull.unsqueeze(-1) * v

        # -- bend / twist: swing-twist split of the rotation since engage,
        #    twist measured about the rest stem axis --
        q_rel = quat_mul(quat, quat_conjugate(self.quat0))
        q_rel = torch.where(q_rel[:, :1] < 0, -q_rel, q_rel)
        s = self.axis
        proj = (q_rel[:, 1:] * s).sum(-1, keepdim=True)
        twist_q = normalize(torch.cat([q_rel[:, :1], proj * s], dim=-1))
        twist_angle = 2.0 * torch.atan2(proj.squeeze(-1), q_rel[:, 0])
        swing_q = quat_mul(q_rel, quat_conjugate(twist_q))
        swing_vec = axis_angle_from_quat(swing_q)                         # world, angle * axis
        w_twist = (w * s).sum(-1, keepdim=True) * s
        w_swing = w - w_twist
        tau_bend = -self.k_bend.unsqueeze(-1) * swing_vec - self.c_bend.unsqueeze(-1) * w_swing
        tau_twist = -(self.k_twist * twist_angle).unsqueeze(-1) * s - self.c_twist.unsqueeze(-1) * w_twist

        self.pull = torch.where(on, f_spring.norm(dim=-1), self.pull)
        self.bend = torch.where(on, self.k_bend * swing_vec.norm(dim=-1), self.bend)
        self.twist = torch.where(on, (self.k_twist * twist_angle).abs(), self.twist)
        self.bend_deg = torch.where(on, torch.rad2deg(swing_vec.norm(dim=-1)), self.bend_deg)
        self.twist_deg = torch.where(on, torch.rad2deg(twist_angle), self.twist_deg)
        self.peak_pull = torch.maximum(self.peak_pull, self.pull * on)
        self.peak_bend = torch.maximum(self.peak_bend, self.bend * on)
        self.peak_twist = torch.maximum(self.peak_twist, self.twist * on)

        cause = torch.zeros_like(self.break_cause)
        cause = torch.where(self.twist > self.m_twist_break, torch.full_like(cause, 3), cause)
        cause = torch.where(self.bend > self.m_bend_break, torch.full_like(cause, 2), cause)
        cause = torch.where(self.pull > self.f_break, torch.full_like(cause, 1), cause)
        newly = on & (cause > 0)
        self.break_cause = torch.where(newly, cause, self.break_cause)
        self.broken |= newly
        on = on & ~newly

        # Cap just above the break limits: a stem breaks there, it never
        # pushes harder -- and it bounds any residual numerical kick.
        f = f * ((1.2 * self.f_break).unsqueeze(-1) / f.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        tau_bend = tau_bend * ((1.2 * self.m_bend_break).unsqueeze(-1)
                               / tau_bend.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        tau_twist = tau_twist * ((1.2 * self.m_twist_break).unsqueeze(-1)
                                 / tau_twist.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)
        mask = on.unsqueeze(-1).float()
        force = f * mask
        torque = (tau_bend + tau_twist) * mask
        self.obj.instantaneous_wrench_composer.add_forces_and_torques(
            forces=force.unsqueeze(1), torques=torque.unsqueeze(1), is_global=True
        )


# ─────────────────────────────────────────────
#  Grasp-ready check (real-robot computable)
# ─────────────────────────────────────────────

class GraspReadyDetector:
    """"The grasp is done, hand over to the harvest": all of
      - grip closure detected (grasp_rewards._force_closure_ok: >= 3 pad
        groups pressing from opposing sides -- taxels + encoders),
      - THUMB pad pressure >= thumb_min N (sensed, pad-only),
      - fingers settled: every finger joint moved less than settle_range
        rad (max - min) over the last hold_s. A velocity threshold never
        passed -- the trained grasp policy keeps the fingers oscillating
        (|dq| 0.3-0.7 rad/s, even low-passed) around a stable grasp; what
        matters for handing over is that they have stopped CLOSING, and
        after the hand-over they hold fixed targets anyway,
    held continuously for hold_s, and at least min_s after the hand-off.
    Everything is available on the real hand."""

    def __init__(self, env: "ManagerBasedRLEnv", sensor_cfg, hand_joint_ids,
                 hold_s: float = 0.5, thumb_min: float = 1.0, settle_range: float = 0.2, min_s: float = 1.0):
        self.env, self.sensor_cfg, self.hand_ids = env, sensor_cfg, hand_joint_ids
        self.hold_steps = max(1, round(hold_s / env.step_dt))
        self.thumb_min, self.settle_range, self.min_s = thumb_min, settle_range, min_s
        self.streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.q_hist = torch.zeros(self.hold_steps, env.num_envs, len(hand_joint_ids), device=env.device)
        self.k = 0

    def reset(self, env_ids) -> None:
        self.streak[env_ids] = 0

    def update(self) -> torch.Tensor:
        from .grasp_rewards import _closure_state, _force_closure_ok
        env = self.env
        robot = env.scene["robot"]
        closure = _force_closure_ok(env, self.sensor_cfg)
        mag, _, _ = _closure_state(env, self.sensor_cfg)
        thumb_ok = mag[:, 1] >= self.thumb_min
        self.q_hist[self.k % self.hold_steps] = robot.data.joint_pos[:, self.hand_ids]
        self.k += 1
        span = (self.q_hist.max(dim=0).values - self.q_hist.min(dim=0).values).max(dim=-1).values
        settled = span < self.settle_range
        since = (env.episode_length_buf - env._grasp_handoff_step).float() * env.step_dt
        ok = env._grasp_phase_active & closure & thumb_ok & settled
        self.last = dict(closure=closure, thumb=thumb_ok, settled=settled,
                         thumb_N=mag[:, 1], max_dq=span)
        self.streak = torch.where(ok, self.streak + 1, torch.zeros_like(self.streak))
        return (self.streak >= self.hold_steps) & (since >= self.min_s)


# ─────────────────────────────────────────────
#  Harvest trajectory + IK
# ─────────────────────────────────────────────

def smoothstep(x: torch.Tensor) -> torch.Tensor:
    """0 -> 1 with zero velocity AND acceleration at both ends (quintic)."""
    x = x.clamp(0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def harvest_palm_pose(
    palm_pos0: torch.Tensor, palm_quat0: torch.Tensor, pivot: torch.Tensor, stem_axis: torch.Tensor,
    swing_axis: torch.Tensor, swing_rad: torch.Tensor, twist_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Palm pose after rotating the whole grasp rigidly about the stem
    PIVOT: first a twist about the stem axis, then a swing about swing_axis
    (both through the pivot). The hand keeps its grip on the fruit, so the
    fruit rolls about its stem attachment instead of being dragged."""
    q_twist = quat_from_angle_axis(twist_rad, stem_axis)
    q_swing = quat_from_angle_axis(swing_rad, swing_axis)
    q = quat_mul(q_swing, q_twist)
    pos = pivot + quat_apply(q, palm_pos0 - pivot)
    quat = quat_mul(q, palm_quat0)
    return pos, quat


def swing_axis_for(mode: str, apple_pos: torch.Tensor, robot_base_pos: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    """Horizontal swing axis for a named direction, per env.
      "side":   the fruit swings SIDEWAYS (perpendicular to the robot->fruit
                direction) -- axis along the horizontal robot->fruit line.
      "toward": the fruit swings TOWARD the robot and up -- axis horizontal
                and perpendicular to that line. Useful when the tree is far
                and a sideways arc runs out of reach.
    sign flips the direction (left/right, toward/away)."""
    radial = apple_pos - robot_base_pos
    radial[:, 2] = 0.0
    radial = normalize(radial)
    up = torch.tensor([0.0, 0.0, 1.0], device=apple_pos.device).expand_as(radial)
    if mode == "side":
        axis = radial
    elif mode == "toward":
        axis = torch.cross(up, radial, dim=-1)
    else:
        raise ValueError(mode)
    return normalize(axis) * sign


class DifferentialIK:
    """Damped-least-squares IK step for one body over a subset of joints,
    using PhysX Jacobians (world frame)."""

    def __init__(self, robot: "Articulation", body_name: str, joint_ids: list[int], damping: float = 0.05):
        self.robot = robot
        self.body_id = robot.data.body_names.index(body_name)
        self.jac_body = self.body_id - 1 if robot.is_fixed_base else self.body_id
        self.jac_cols = list(joint_ids) if robot.is_fixed_base else [j + 6 for j in joint_ids]
        self.joint_ids = list(joint_ids)
        self.damping = damping

    def pose_error(self, pos_des, quat_des):
        pos = self.robot.data.body_pos_w[:, self.body_id]
        quat = self.robot.data.body_quat_w[:, self.body_id]
        dq = quat_mul(quat_des, quat_conjugate(quat))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)
        return pos_des - pos, axis_angle_from_quat(dq)

    def step(self, pos_des: torch.Tensor, quat_des: torch.Tensor, max_dq: float = 0.2,
             base: torch.Tensor | None = None) -> torch.Tensor:
        """Joint position targets (for self.joint_ids) one DLS step toward
        the desired pose, added to `base` -- pass the CURRENTLY COMMANDED
        targets, not the measured angles: under gravity the joint PD sits a
        few centi-radians below its target, and re-commanding the measured
        angles drops the arm by that much again (measured: a ~2 cm drop
        that broke the stem by pulling before the motion began).
        Integrating on the targets acts as the missing integral term."""
        J = self.robot.root_physx_view.get_jacobians()[:, self.jac_body][:, :, self.jac_cols]   # [N, 6, n]
        e_pos, e_rot = self.pose_error(pos_des, quat_des)
        e = torch.cat([e_pos, e_rot], dim=-1).unsqueeze(-1)                                 # [N, 6, 1]
        JT = J.transpose(1, 2)
        lam = (self.damping ** 2) * torch.eye(6, device=J.device).unsqueeze(0)
        dq = (JT @ torch.linalg.solve(J @ JT + lam, e)).squeeze(-1)
        dq = dq.clamp(-max_dq, max_dq)
        if base is None:
            base = self.robot.data.joint_pos[:, self.joint_ids]
        return base + dq


BREAK_CAUSES = {0: "-", 1: "pull", 2: "bend", 3: "twist"}
