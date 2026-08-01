# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Proximity-triggered smooth stop.

When the palm gets close to the apple and stays close for a short dwell
time, hand off control from the RL policy to a quintic (minimum-jerk)
deceleration that brings the arm's current joint velocity smoothly to zero
-- instead of letting the policy keep nudging the arm around near an object
it could damage, or instantly freezing the joints (a velocity/acceleration
discontinuity -- exactly the "bad jerk" this is meant to avoid).

Threading model: PhysX/Isaac Sim tensors may only safely be touched from the
main simulation thread. QuinticStopController's background thread NEVER
calls into the sim -- it only runs a state machine + a free-running clock
against state samples the main loop pushes into it via update_state(), and
produces a command the main loop reads back via get_command() and sends
through env.step() as usual. Same pattern as this file's existing
_listen_terminal_reset thread (a flag behind a lock, nothing more).

Quintic derivation: standard point-to-point quintic time-scaling (e.g.
Craig, "Introduction to Robotics") with boundary conditions
  q(0)=q0, q'(0)=v0, q''(0)=0      (measured position/velocity, START
                                     acceleration simplified to 0 -- see
                                     docstring on _quintic_coeffs)
  q(T)=q1, q'(T)=0,  q''(T)=0      (fully at rest by the end)
q1 (the stopping point) is not chosen arbitrarily: it's set to
q0 + v0*T/2, the displacement of a symmetric smooth deceleration from v0 to
0 over duration T (independent of the exact profile shape, to leading
order) -- i.e. "move the least amount consistent with a jerk-limited stop
in time T", rather than picking a fixed target position ahead of time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch


@dataclass
class QuinticStopConfig:
    stop_dist: float = 0.06        # metres: palm-to-apple distance that arms the trigger
    dwell_time: float = 1.0        # seconds: must stay within stop_dist this long before it fires
    stop_duration: float = 0.8     # seconds: duration of the quintic deceleration
    resume_margin: float = 0.03    # metres: hysteresis -- must clear stop_dist+margin to hand control back
    poll_interval: float = 0.01    # seconds: background thread tick rate (cheap; just a clock+FSM)


class QuinticStopController:
    """See module docstring. All public methods are thread-safe.

    Usage (once per control step, from the MAIN thread, in this order):
        controller.update_state(t=sim_time, dist=palm_apple_dist, q=joint_pos, dq=joint_vel)
        override, q_cmd = controller.get_command(t=sim_time, q_policy=policy_target)
        # if override: send q_cmd instead of q_policy this step
    """

    IDLE = "IDLE"
    STOPPING = "STOPPING"
    HOLDING = "HOLDING"

    def __init__(self, num_joints: int, cfg: QuinticStopConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or QuinticStopConfig()
        self.num_joints = num_joints
        self.device = device

        self._lock = threading.Lock()
        self._alive = True

        self._state = self.IDLE
        self._near_since: float | None = None

        # Latest readings pushed by the main thread.
        self._t = 0.0
        self._dist = None
        self._q = torch.zeros(num_joints, device=device)
        self._dq = torch.zeros(num_joints, device=device)

        # Captured once, at the moment STOPPING begins.
        self._q0 = torch.zeros(num_joints, device=device)
        self._v0 = torch.zeros(num_joints, device=device)
        self._q1 = torch.zeros(num_joints, device=device)   # derived stop target, q0 + v0*T/2
        self._stop_start_t = 0.0
        self._hold_target = torch.zeros(num_joints, device=device)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._alive = False

    def reset(self) -> None:
        """Call after an env reset -- goes back to IDLE so a stale q0/target
        captured before the reset can't leak into a command post-reset."""
        with self._lock:
            self._state = self.IDLE
            self._near_since = None
            self._dist = None

    # ── Called every control step from the main thread ──────────────────────

    def update_state(self, t: float, dist: float, q: torch.Tensor, dq: torch.Tensor) -> None:
        with self._lock:
            self._t = t
            self._dist = dist
            self._q = q.detach().clone()
            self._dq = dq.detach().clone()

    def get_command(self, t: float, q_policy: torch.Tensor):
        """Returns (override: bool, q_cmd: Tensor[num_joints]).

        override=False means "use q_policy unchanged this step" -- the
        caller is responsible for actually driving the policy in that case,
        this class only ever hands back a command when it wants control.
        """
        with self._lock:
            state = self._state
            if state == self.STOPPING:
                elapsed = min(max(t - self._stop_start_t, 0.0), self.cfg.stop_duration)
                q_cmd = self._quintic_sample(elapsed)
                return True, q_cmd
            if state == self.HOLDING:
                return True, self._hold_target.clone()
        return False, q_policy

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    # ── Background thread: FSM + free-running clock only, no sim access ─────

    def _loop(self):
        while self._alive:
            with self._lock:
                t, dist, state = self._t, self._dist, self._state
                q, dq = self._q, self._dq

            if dist is not None:
                if state == self.IDLE:
                    if dist <= self.cfg.stop_dist:
                        if self._near_since is None:
                            self._near_since = t
                        elif t - self._near_since >= self.cfg.dwell_time:
                            with self._lock:
                                self._q0 = q.clone()
                                self._v0 = dq.clone()
                                self._q1 = self._q0 + self._v0 * self.cfg.stop_duration / 2.0
                                self._stop_start_t = t
                                self._state = self.STOPPING
                    else:
                        self._near_since = None

                elif state == self.STOPPING:
                    if t - self._stop_start_t >= self.cfg.stop_duration:
                        with self._lock:
                            self._hold_target = self._quintic_sample(self.cfg.stop_duration)
                            self._state = self.HOLDING

                elif state == self.HOLDING:
                    if dist > self.cfg.stop_dist + self.cfg.resume_margin:
                        with self._lock:
                            self._state = self.IDLE
                            self._near_since = None

            time.sleep(self.cfg.poll_interval)

    # ── Quintic evaluation ────────────────────────────────────────────────

    def _quintic_sample(self, elapsed: float) -> torch.Tensor:
        """q(elapsed) for the trajectory captured at STOPPING onset.

        Simplifies q''(0)=0 (measuring actual joint acceleration at the
        trigger instant is noisy and unnecessary here -- q'(0)=v0 is matched
        exactly, which is what actually determines whether the handoff is
        visually/mechanically smooth; the q''(0)=0 assumption only affects
        the interior shape of the blend, not its endpoints).
        Caller must hold self._lock or only call with instance fields that
        aren't concurrently mutated (only invoked from within _loop or
        get_command, both of which hold the lock around the call site).
        """
        T = self.cfg.stop_duration
        if T <= 0.0:
            return self._q1.clone()
        tau = elapsed
        q0, v0, q1 = self._q0, self._v0, self._q1

        # Closed-form quintic coefficients for q(0)=q0, q'(0)=v0, q''(0)=0,
        # q(T)=q1, q'(T)=0, q''(T)=0.
        h = q1 - q0
        a3 = (20.0 * h - 12.0 * v0 * T) / (2.0 * T ** 3)
        a4 = (30.0 * (-h) + 16.0 * v0 * T) / (2.0 * T ** 4)
        a5 = (12.0 * h - 6.0 * v0 * T) / (2.0 * T ** 5)

        return q0 + v0 * tau + a3 * tau ** 3 + a4 * tau ** 4 + a5 * tau ** 5
