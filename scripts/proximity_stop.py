# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Proximity-triggered smooth stop.

When the palm gets close to the apple and stays close for a short dwell
time, hand off control from the RL policy to a quintic (minimum-jerk)
deceleration that brings the arm's current joint velocity smoothly to zero
-- instead of letting the policy keep nudging the arm around near an object
it could damage, or instantly freezing the joints (a velocity/acceleration
discontinuity -- exactly the "bad jerk" this is meant to avoid).

A second, independent trigger (QuinticStopConfig.watchdog_time) fires the
SAME smooth stop after a fixed time since the current attempt started,
regardless of whether the palm ever got near the apple at all -- a safety
fallback for a policy that diverges or oscillates and never satisfies the
proximity condition on its own.

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
    stop_dist: float = 0.05        # metres: arms the trigger (see metric note below)
    dwell_time: float = 0.6        # seconds: must stay within stop_dist this long before it fires
    stop_duration: float = 0.4     # seconds: duration of the quintic deceleration
    resume_margin: float = 0.04    # metres: hysteresis on the dwell-timer arming check (see IDLE
                                    # branch in _loop) -- NOT a "resume policy control" threshold;
                                    # HOLDING only ever exits via reset(), see its comment below
    poll_interval: float = 0.01    # seconds: background thread tick rate (cheap; just a clock+FSM)

    # DISTANCE METRIC: `dist` passed to update_state() is the palm REFERENCE
    # POINT (palm link origin + PALM_LOCAL_OFFSET, out at the finger pads) to
    # the GRASP TARGET (GRASP_STANDOFF in front of the apple surface) -- exactly the
    # quantity reach_rewards._palm_target_dist computes, so stop_dist is
    # directly comparable to RewardsCfg's own pos_tol (0.04 m). It used to be
    # fed wrist-origin-to-apple-CENTRE, which reads ~0.10 m at the very pose
    # the policy is trained to reach; stop_dist and resume_margin above were
    # retuned for the corrected metric and old values do not carry over.
    watchdog_time: float | None = 2.5
    """Seconds since the current attempt started (episode reset, or
    controller construction) before the stop triggers UNCONDITIONALLY, even
    if the palm never got within stop_dist. Independent safety fallback for
    a policy that diverges/oscillates and never satisfies the proximity
    condition on its own -- the arm still gets a smooth quintic stop instead
    of running indefinitely. Set to None (or <= 0) to disable."""

    # ── Contact trigger ────────────────────────────────────────────────
    # The reach task's goal became "arrive AND touch gently", not "arrive",
    # so actual palm contact is now the most direct evidence that the arm is
    # where it should be -- much more direct than a distance threshold, and
    # available strictly earlier than any dwell timer can confirm. Contact
    # therefore triggers the stop IMMEDIATELY, with no dwell.
    contact_force_min: float = 0.5
    """Newtons of apple-filtered palm contact force that count as 'touching'.
    Lower edge of the good band, used by BOTH the trigger and the recall
    latch. Keep in sync with RewardsCfg.gentle_touch_palm / _fingers and
    reward_reach_hold's `good_min` / `pressure_min`.

    Turn the behaviours off with the `contact_trigger` / `contact_recall`
    flags, never by zeroing this: it is a band edge, not an enable switch,
    and zeroing it would make every no-contact reading look like a valid
    contact to the recall latch."""

    contact_trigger: bool = True
    """Whether a reading >= contact_force_min fires the stop immediately.
    Independent of contact_recall: switching this off still lets a good
    contact pose be latched and returned to if a timer fires later."""

    contact_force_max: float = 5.0
    """Upper edge of the 'good' contact band, matching gentle_touch_*'s
    good_max. Only a reading inside [contact_force_min, contact_force_max]
    is latched as a pose worth returning to (see contact_recall below) -- a
    slam above this band is a pose to leave, not one to go back to."""

    contact_recall: bool = True
    """If a good contact was seen earlier in this attempt but the stop ends
    up being fired by the proximity dwell or the watchdog instead, decelerate
    back to the LATCHED CONTACT POSE rather than to wherever the arm happens
    to be at trigger time.

    This is the fix for the observed "policy makes a clean 3 N contact, then
    the stop takes over and parks the hand a few centimetres away, touching
    nothing". Two independent causes, both covered here: (a) the watchdog
    fires on elapsed time alone and does not care what the arm is doing, and
    (b) the stop target was ALWAYS q0 + v0*T/2, i.e. it blindly coasts along
    whatever joint velocity existed at the trigger instant -- for a policy
    making small corrections around the contact pose that velocity is as
    likely to point away from the apple as into it, and HOLDING never exits,
    so the arm stays parked at that coasted-away pose for the rest of the
    episode."""

    coast_fraction: float = 0.5
    """Stop target = q0 + v0 * stop_duration * coast_fraction. 0.5 is the
    displacement of a symmetric smooth deceleration from v0 to 0 (the
    original, and still the default for a plain proximity/watchdog stop with
    nothing better to aim at). 0.0 stops in place."""

    contact_coast_fraction: float = 0.0
    """coast_fraction used when the stop was fired BY contact. Zero: the arm
    is already touching, so coasting a further 0.2 s along v0 can only press
    harder into the apple. The quintic still matches q'(0) = v0 exactly, so
    the handoff stays smooth -- it just returns to q0 instead of running on
    past it."""


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
        self._episode_start_t: float = 0.0
        self.last_trigger_reason: str | None = None   # "proximity" or "watchdog", for logging

        # Latest readings pushed by the main thread.
        self._t = 0.0
        self._dist = None
        self._contact_force: float | None = None
        self._q = torch.zeros(num_joints, device=device)
        self._dq = torch.zeros(num_joints, device=device)

        # Most recent joint pose observed while palm contact force was
        # inside [contact_force_min, contact_force_max]. See contact_recall.
        self._q_at_contact: torch.Tensor | None = None

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

    def reset(self, t: float = 0.0) -> None:
        """Call after an env reset -- goes back to IDLE so a stale q0/target
        captured before the reset can't leak into a command post-reset, and
        restarts the watchdog clock. `t` should be the sim-time the reset
        actually happened at (e.g. 0.0 right after `step` is reset to 0),
        so watchdog_time is measured from the new attempt's actual start."""
        with self._lock:
            self._state = self.IDLE
            self._near_since = None
            self._dist = None
            self._contact_force = None
            # Belongs to the attempt that just ended -- returning to it in
            # the next episode would aim at a pose for a differently-placed
            # apple, exactly the kind of stale-target leak this reset exists
            # to prevent.
            self._q_at_contact = None
            self._episode_start_t = t
            self.last_trigger_reason = None

    # ── Called every control step from the main thread ──────────────────────

    def update_state(
        self,
        t: float,
        dist: float,
        q: torch.Tensor,
        dq: torch.Tensor,
        contact_force: float | None = None,
    ) -> None:
        """`contact_force`: apple-filtered palm contact force in newtons, or
        None if the caller has no contact sensing (contact trigger and
        contact recall then both stay inactive and this behaves exactly as
        it did before -- proximity dwell + watchdog only)."""
        with self._lock:
            self._t = t
            self._dist = dist
            self._contact_force = contact_force
            self._q = q.detach().clone()
            self._dq = dq.detach().clone()

            if (
                self.cfg.contact_recall
                and contact_force is not None
                and self.cfg.contact_force_min <= contact_force <= self.cfg.contact_force_max
            ):
                self._q_at_contact = self._q.clone()

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
                contact_force = self._contact_force
                q_at_contact = self._q_at_contact

            if dist is not None:
                if state == self.IDLE:
                    # dist <= stop_dist ARMS the dwell timer; once armed, only
                    # a excursion past stop_dist+resume_margin resets it --
                    # NOT crossing back over the bare stop_dist line. Small
                    # residual motion near the target (the whole reason this
                    # controller exists) was ticking a hair above stop_dist
                    # every few steps, resetting _near_since to None each
                    # time and preventing dwell_time from EVER accumulating
                    # continuously. Same tolerance already used to leave
                    # HOLDING below, now applied consistently on the way in.
                    release_dist = self.cfg.stop_dist + self.cfg.resume_margin
                    if self._near_since is None:
                        if dist <= self.cfg.stop_dist:
                            self._near_since = t
                    elif dist > release_dist:
                        self._near_since = None

                    dwell_fired = (
                        self._near_since is not None
                        and t - self._near_since >= self.cfg.dwell_time
                    )
                    # Independent of proximity entirely: fires purely on
                    # elapsed time since this attempt started, so a policy
                    # that never gets within stop_dist still gets stopped
                    # smoothly instead of running forever.
                    watchdog_fired = (
                        self.cfg.watchdog_time is not None
                        and self.cfg.watchdog_time > 0.0
                        and t - self._episode_start_t >= self.cfg.watchdog_time
                    )

                    # Contact needs no dwell: unlike "the palm is nearby",
                    # which is noisy and can be satisfied while the policy is
                    # still flailing, a real apple-filtered contact reading
                    # is unambiguous evidence the arm has arrived. Waiting
                    # dwell_time to confirm it only gives the policy 0.6 s to
                    # bounce back off the apple first.
                    contact_fired = (
                        self.cfg.contact_trigger
                        and contact_force is not None
                        and contact_force >= self.cfg.contact_force_min
                    )

                    if contact_fired or dwell_fired or watchdog_fired:
                        T = self.cfg.stop_duration
                        with self._lock:
                            self._q0 = q.clone()
                            self._v0 = dq.clone()
                            if contact_fired:
                                reason = "contact"
                                # Already touching -- do not coast further in.
                                self._q1 = self._q0 + self._v0 * T * self.cfg.contact_coast_fraction
                            elif q_at_contact is not None:
                                # Fired on a timer, but this attempt DID make
                                # good contact earlier: decelerate back onto
                                # that pose instead of parking wherever the
                                # timer happened to catch the arm.
                                reason = "proximity" if dwell_fired else "watchdog"
                                reason += "+recall"
                                self._q1 = q_at_contact.clone()
                            else:
                                reason = "proximity" if dwell_fired else "watchdog"
                                self._q1 = self._q0 + self._v0 * T * self.cfg.coast_fraction
                            self._stop_start_t = t
                            self._state = self.STOPPING
                            self.last_trigger_reason = reason

                elif state == self.STOPPING:
                    if t - self._stop_start_t >= self.cfg.stop_duration:
                        with self._lock:
                            self._hold_target = self._quintic_sample(self.cfg.stop_duration)
                            self._state = self.HOLDING

                # HOLDING has no auto-exit back to IDLE on its own -- this is
                # a safety stop, not a pause. Silently handing control back
                # to the policy the instant distance ticks past some
                # threshold defeats the point (and was actively buggy for a
                # watchdog-triggered stop: dist was never close to begin
                # with, so the old dist>release_dist check fired the very
                # next tick, bouncing STOPPING<->HOLDING forever). Only
                # reset() (called on an actual env reset, manual or
                # --auto_reset_on_success) leaves HOLDING now.

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
