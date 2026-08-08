# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Vectorized mirror of proximity_stop.QuinticStopController, for batch evaluation.

scripts/proximity_stop.py drives ONE arm from a background thread, which is the
right design for the interactive DDS loop but cannot score 64 parallel
environments. This module reproduces the same finite-state machine and the same
quintic profile over a batch, so the evaluation measures the controller the
robot actually runs rather than a bare policy.

Faithfulness notes -- read before changing either file:

  * The FSM, the arming hysteresis and the watchdog are transcribed from
    proximity_stop._loop(). In particular the dwell timer is armed when
    ``dist <= stop_dist`` and disarmed only when ``dist > stop_dist +
    resume_margin``: crossing back over the bare stop_dist line does NOT
    disarm it. That asymmetry is load-bearing -- residual motion near the
    target otherwise ticks a hair above stop_dist every few steps and the
    dwell time never accumulates.
  * HOLDING has no auto-exit. Only an episode reset leaves it.
  * `dist` is the RAW palm-link-origin to fruit-centre distance, matching
    run_env.py's convention -- not the grasp-reference-point metric the reward
    scores against.

One deliberate difference: the real controller polls on its own thread at
100 Hz while commands are read at 50 Hz, so a transition can land between
reads. Here the FSM advances exactly once per control step, which is the
deterministic idealization of that race.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

IDLE, STOPPING, HOLDING = 0, 1, 2


@dataclass
class BatchStopConfig:
    """Mirrors proximity_stop.QuinticStopConfig."""
    stop_dist: float = 0.09
    dwell_time: float = 0.6
    stop_duration: float = 0.4
    resume_margin: float = 0.12
    watchdog_time: float | None = 2.5


class BatchQuinticStop:
    """Batched proximity-triggered smooth stop.

    Usage per control step, in this order::

        stop.reset_envs(env_ids)                  # on episode boundaries
        stop.update(t, dist, q, dq)               # FSM advance
        override, q_cmd = stop.command(t, q_ctrl) # override rows where True
    """

    def __init__(self, num_envs: int, num_joints: int,
                 cfg: BatchStopConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or BatchStopConfig()
        self.n, self.j, self.device = num_envs, num_joints, device
        z = lambda: torch.zeros(num_envs, num_joints, device=device)
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.near_since = torch.full((num_envs,), float("nan"), device=device)
        self.episode_start_t = torch.zeros(num_envs, device=device)
        self.stop_start_t = torch.zeros(num_envs, device=device)
        self.q0, self.v0, self.q1, self.hold_target = z(), z(), z(), z()
        self.trigger = torch.zeros(num_envs, dtype=torch.long, device=device)  # 0 none 1 prox 2 wdog

    def reset_envs(self, mask: torch.Tensor, t: float = 0.0) -> None:  # noqa: D401
        """Return the masked envs to IDLE and restart their watchdog clock."""
        if not bool(mask.any()):
            return
        self.state[mask] = IDLE
        self.near_since[mask] = float("nan")
        self.episode_start_t[mask] = t
        self.trigger[mask] = 0

    def update(self, t: torch.Tensor, dist: torch.Tensor,
               q: torch.Tensor, dq: torch.Tensor) -> None:
        """Advance the FSM. `t` is PER-ENV episode time (each env has its own
        attempt clock), not a global wall time."""
        c = self.cfg
        release = c.stop_dist + c.resume_margin
        idle = self.state == IDLE

        # Arm / disarm the dwell timer (asymmetric by design, see module docstring).
        unarmed = torch.isnan(self.near_since)
        arm = idle & unarmed & (dist <= c.stop_dist)
        self.near_since = torch.where(arm, t, self.near_since)
        disarm = idle & ~unarmed & (dist > release)
        self.near_since = torch.where(disarm, torch.full_like(self.near_since, float("nan")),
                                      self.near_since)

        dwell = idle & ~torch.isnan(self.near_since) & ((t - self.near_since) >= c.dwell_time)
        wdog = idle & torch.zeros_like(dwell) if not c.watchdog_time or c.watchdog_time <= 0 \
            else idle & ((t - self.episode_start_t) >= float(c.watchdog_time))
        fire = dwell | wdog

        if bool(fire.any()):
            self.q0[fire] = q[fire]
            self.v0[fire] = dq[fire]
            self.q1[fire] = q[fire] + dq[fire] * (c.stop_duration / 2.0)
            self.stop_start_t[fire] = t[fire]
            self.state[fire] = STOPPING
            self.trigger[fire] = torch.where(dwell[fire],
                                             torch.ones_like(self.trigger[fire]),
                                             torch.full_like(self.trigger[fire], 2))

        done = (self.state == STOPPING) & ((t - self.stop_start_t) >= c.stop_duration)
        if bool(done.any()):
            self.hold_target[done] = self._quintic(
                torch.full((int(done.sum()),), c.stop_duration, device=self.device), done)
            self.state[done] = HOLDING

    def command(self, t: torch.Tensor, q_ctrl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(override_mask, q_cmd) -- q_cmd equals q_ctrl where override is False."""
        q_cmd = q_ctrl.clone()
        stopping = self.state == STOPPING
        if bool(stopping.any()):
            el = (t[stopping] - self.stop_start_t[stopping]).clamp(0.0, self.cfg.stop_duration)
            q_cmd[stopping] = self._quintic(el, stopping)
        holding = self.state == HOLDING
        if bool(holding.any()):
            q_cmd[holding] = self.hold_target[holding]
        return (stopping | holding), q_cmd

    def _quintic(self, elapsed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Quintic sample for the masked envs. Mirrors _quintic_sample().

        Boundary conditions q(0)=q0, q'(0)=v0, q''(0)=0, q(T)=q1, q'(T)=q''(T)=0.
        With q1 = q0 + v0*T/2 this reduces to a5 = 0, but the general form is
        kept so the two files stay line-by-line comparable.
        """
        T = self.cfg.stop_duration
        if T <= 0.0:
            return self.q1[mask].clone()
        q0, v0, q1 = self.q0[mask], self.v0[mask], self.q1[mask]
        h = q1 - q0
        a3 = (20.0 * h - 12.0 * v0 * T) / (2.0 * T ** 3)
        a4 = (30.0 * (-h) + 16.0 * v0 * T) / (2.0 * T ** 4)
        a5 = (12.0 * h - 6.0 * v0 * T) / (2.0 * T ** 5)
        tau = elapsed.unsqueeze(-1)
        return q0 + v0 * tau + a3 * tau ** 3 + a4 * tau ** 4 + a5 * tau ** 5
