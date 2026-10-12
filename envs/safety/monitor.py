# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Safety watchdog for the arm (+ waist): runs beside the policies and never
trusts them. Pure torch, no Isaac Lab: the same code runs in the harvest
bench (fed from PhysX) and, later, as a 500 Hz thread next to the real
robot (fed from LowState: q, dq, tau_est, and a pinocchio model).

Three layers, per env:
  1. COMMAND -- the joint targets about to be sent: velocity, acceleration
     and jerk of the target stream against limits (a policy glitch, a target
     jump), and how far the target runs ahead of the joint;
  2. EXTERNAL TORQUE -- tau_measured - tau_model (gravity + Coriolis +
     M * qdd: what the motion itself needs), low-pass filtered and divided by
     each motor's rated torque. Pushing on fruit, the basket, the trunk or
     the body shows up as a sustained residual. Thresholds are sustained
     for a minimum time, and a context factor raises them where contact is
     intended (the grasp, the twist / bend);
  3. STUCK -- a joint far behind its target and not moving, sustained (the
     arm trapped in the basket).
Levels: OK -> WARN -> SLOW -> STOP. STOP latches until reset(); the caller
reacts (the bench: freeze, wait, retreat along the way it came, abort the
apple). WARN / SLOW drop back after a calm period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

OK, WARN, SLOW, STOP = 0, 1, 2, 3
LEVEL_NAMES = {OK: "ok", WARN: "warn", SLOW: "slow", STOP: "stop"}
#: reason bits
R_RESID, R_STUCK, R_CMD = 1, 2, 4
REASON_NAMES = {R_RESID: "external torque", R_STUCK: "stuck", R_CMD: "command"}


@dataclass
class SafetyCfg:
    dt: float                                        # update period (s)
    rated: list[float]                               # rated torque per monitored joint (N*m)
    # 1. command stream limits (per joint, or one value for all)
    v_lim: float | list[float] = 2.5                 # rad/s
    a_lim: float | list[float] = 12.0                # rad/s^2
    j_lim: float | list[float] = 600.0               # rad/s^3
    lead_lim: float = 0.6                            # rad: target ahead of the joint
    t_cmd: float = 0.1                               # s of violation -> WARN, x2 -> SLOW
    # 2. external torque (fraction of rated), sustained times
    resid_warn: float = 0.12
    resid_slow: float = 0.20
    resid_stop: float = 0.30
    resid_floor: float = 1.0                         # N*m: never below this (small motors, noise)
    t_warn: float = 0.06
    t_slow: float = 0.10
    t_stop: float = 0.15
    filter_hz: float = 8.0
    # 3. stuck
    stuck_err: float = 0.15                          # rad behind the target
    stuck_speed: float = 0.05                        # rad/s
    t_stuck: float = 0.5
    calm_s: float = 0.4                              # below WARN this long -> back to OK


class SafetyMonitor:
    def __init__(self, cfg: SafetyCfg, n: int, device="cpu"):
        self.cfg = cfg
        self.n, self.dev = n, device
        J = len(cfg.rated)
        self.J = J
        vec = lambda x: torch.as_tensor(x if isinstance(x, list) else [x] * J, dtype=torch.float32, device=device)
        self.rated = vec(cfg.rated)
        self.v_lim, self.a_lim, self.j_lim = vec(cfg.v_lim), vec(cfg.a_lim), vec(cfg.j_lim)
        self.alpha = cfg.dt / (cfg.dt + 1.0 / (2 * math.pi * cfg.filter_hz))
        z = lambda *s: torch.zeros(*s, device=device)
        self.r_f = z(n, J)                           # filtered residual (N*m)
        self.q_cmd_hist = z(n, 3, J)                 # last three commands
        self.n_hist = torch.zeros(n, dtype=torch.long, device=device)
        self.t_resid = z(n, 3)                       # time above warn / slow / stop
        self.t_stuck = z(n)
        self.t_cmd = z(n)
        self.t_calm = z(n)
        self.level = torch.zeros(n, dtype=torch.long, device=device)
        self.reason = torch.zeros(n, dtype=torch.long, device=device)
        self.peak = z(n)                             # peak normalized residual since reset

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for t in (self.r_f, self.q_cmd_hist, self.t_resid, self.t_stuck, self.t_cmd, self.t_calm, self.peak):
            t[ids] = 0.0
        self.n_hist[ids] = 0
        self.level[ids] = OK
        self.reason[ids] = 0

    def update(self, q: torch.Tensor, dq: torch.Tensor, q_cmd: torch.Tensor, tau: torch.Tensor,
               tau_model: torch.Tensor, contact_scale: torch.Tensor | float = 1.0,
               active: torch.Tensor | None = None) -> torch.Tensor:
        """One period. q, dq, q_cmd, tau, tau_model: [n, J]. contact_scale
        [n] or float: thresholds x this where contact is intended. active [n]
        bool: envs to supervise (others hold their state). Returns level [n]."""
        c = self.cfg
        dt = c.dt
        act = torch.ones(self.n, dtype=torch.bool, device=self.dev) if active is None else active
        cs = torch.as_tensor(contact_scale, dtype=torch.float32, device=self.dev).expand(self.n)
        prev_level = self.level.clone()

        # 1. command stream
        h = self.q_cmd_hist
        h_new = torch.stack([h[:, 1], h[:, 2], q_cmd], dim=1)
        self.n_hist = torch.where(act, (self.n_hist + 1).clamp(max=3), self.n_hist)
        v = (h_new[:, 2] - h_new[:, 1]) / dt
        a = (h_new[:, 2] - 2 * h_new[:, 1] + h_new[:, 0]) / dt ** 2
        v_prev = (h[:, 2] - h[:, 1]) / dt
        a_prev = (h[:, 2] - 2 * h[:, 1] + h[:, 0]) / dt ** 2
        j = (a - a_prev) / dt
        full = (self.n_hist >= 3).unsqueeze(-1)
        bad = ((v.abs() > self.v_lim) | ((a.abs() > self.a_lim) & full) | ((j.abs() > self.j_lim) & full)
               | ((q_cmd - q).abs() > c.lead_lim)).any(-1)
        self.q_cmd_hist = torch.where(act.view(-1, 1, 1), h_new, h)
        self.t_cmd = torch.where(act & bad, self.t_cmd + dt, torch.zeros_like(self.t_cmd))

        # 2. external torque
        r_f_new = self.r_f + self.alpha * ((tau - tau_model) - self.r_f)
        self.r_f = torch.where(act.unsqueeze(-1), r_f_new, self.r_f)
        norm = (self.r_f.abs() / self.rated.clamp(min=c.resid_floor / max(c.resid_warn, 1e-6))).amax(-1)
        self.peak = torch.where(act, torch.maximum(self.peak, norm), self.peak)
        thr = torch.stack([torch.full_like(cs, c.resid_warn), torch.full_like(cs, c.resid_slow),
                           torch.full_like(cs, c.resid_stop)], -1) * cs.unsqueeze(-1)
        above = (norm.unsqueeze(-1) > thr) & act.unsqueeze(-1)
        self.t_resid = torch.where(above, self.t_resid + dt, torch.zeros_like(self.t_resid))

        # 3. stuck
        stuck = act & (((q_cmd - q).abs() > c.stuck_err) & (dq.abs() < c.stuck_speed)).any(-1)
        self.t_stuck = torch.where(stuck, self.t_stuck + dt, torch.zeros_like(self.t_stuck))

        # levels
        lvl = torch.zeros_like(self.level)
        rsn = torch.zeros_like(self.reason)
        r_w, r_s, r_x = self.t_resid[:, 0] >= c.t_warn, self.t_resid[:, 1] >= c.t_slow, self.t_resid[:, 2] >= c.t_stop
        lvl = torch.where(r_w, torch.full_like(lvl, WARN), lvl)
        lvl = torch.where(r_s, torch.full_like(lvl, SLOW), lvl)
        lvl = torch.where(r_x, torch.full_like(lvl, STOP), lvl)
        rsn = rsn | torch.where(r_w, R_RESID, 0)
        c_w, c_s = self.t_cmd >= c.t_cmd, self.t_cmd >= 2 * c.t_cmd
        lvl = torch.maximum(lvl, torch.where(c_s, SLOW, torch.where(c_w, WARN, 0)))
        rsn = rsn | torch.where(c_w, R_CMD, 0)
        st = self.t_stuck >= c.t_stuck
        lvl = torch.where(st, torch.full_like(lvl, STOP), lvl)
        rsn = rsn | torch.where(st, R_STUCK, 0)

        # STOP latches; WARN / SLOW decay after a calm period
        calm = lvl == OK
        self.t_calm = torch.where(calm & act, self.t_calm + dt, torch.zeros_like(self.t_calm))
        held = torch.where(self.t_calm >= c.calm_s, torch.zeros_like(prev_level), prev_level)
        new_level = torch.where(prev_level == STOP, prev_level, torch.maximum(lvl, held))
        self.level = torch.where(act, new_level, self.level)
        self.reason = torch.where(act & (lvl > OK), self.reason | rsn, self.reason)
        return self.level


def reasons(bits: int) -> str:
    return ", ".join(n for b, n in REASON_NAMES.items() if bits & b) or "-"
