# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Rate-limited joint position action.

Isaac Lab's stock JointPositionActionCfg has no notion of a maximum velocity
or acceleration on the COMMANDED TARGET itself — the only bounds in the
current ActionsCfg are a static clip on the target's absolute offset
(+-3 rad) and the actuator's velocity_limit (2.175 rad/s), which only
constrains the PD solve's internal saturation, not how fast the reference
target itself is allowed to jump between control steps. A policy is free to
command a target 3 rad away from the previous one every 20 ms; the actuator
then saturates at its velocity limit trying to catch up, which in practice
means bang-bang acceleration every control step -- exactly the "moving too
fast / big accelerations" behaviour reported when driving the real hand.

g1_arm_policies/g1_arm_env.py already solved this for the SAC pipeline with
a hard two-stage clip on the commanded joint velocity (dq_max, then
max_delta_dq bounding the CHANGE in that velocity, i.e. acceleration) before
integrating forward to a position command -- see its _step_dynamics/apply
logic. RateLimitedJointPositionAction below ports the same mechanism into
Isaac Lab's ActionTerm interface, so smoothness becomes a STRUCTURAL
constraint the policy cannot violate, not just something the reward
functions in reach_rewards.py (penalty_action_jerk, action_rate_l2) nudge it
away from.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RateLimitedJointPositionAction(JointPositionAction):
    """JointPositionAction with a hard velocity + acceleration rate limiter.

    Every control step, the usual affine-transformed target (offset +
    scale*raw_action, then statically clipped -- identical to the parent
    class) is treated as a DESIRED target, not the commanded one. The
    actually-commanded target is obtained by:

      1. v_desired = (target_desired - prev_target) / dt
      2. v_cmd = clip(v_desired, -max_velocity, +max_velocity)
      3. v_cmd = clip(v_cmd, prev_v - max_dv, prev_v + max_dv)   # max_dv = max_acceleration * dt
      4. target_cmd = prev_target + v_cmd * dt

    Step 2 hard-caps velocity; step 3 hard-caps the CHANGE in velocity, i.e.
    acceleration. This makes it structurally impossible for the commanded
    trajectory to exceed max_acceleration, regardless of what the policy
    outputs -- apply_actions() is inherited unchanged from JointPositionAction
    and simply sends this rate-limited target as the PD position target every
    physics substep.

    prev_target/prev_v are re-anchored to the ACTUAL current joint state on
    the first control step of every episode (env.episode_length_buf <= 1),
    the same convention used throughout reach_rewards.py -- otherwise the
    first command after a reset would be rate-limited relative to a stale
    target left over from the previous episode.
    """

    cfg: "RateLimitedJointPositionActionCfg"

    def __init__(self, cfg: "RateLimitedJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._prev_target = torch.zeros_like(self._processed_actions)
        self._prev_vel = torch.zeros_like(self._processed_actions)
        # Per-env speed factor in (0, 1] (deployment / safety slow-down):
        # velocity cap x s, acceleration cap x s^2 -- the same trajectory
        # shape played 1/s times slower. 1 = as trained.
        self.speed_scale = torch.ones(self.num_envs, 1, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        target_desired = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            target_desired = torch.clamp(
                target_desired, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        needs_reset = (self._env.episode_length_buf <= 1).unsqueeze(-1)
        self._prev_target = torch.where(needs_reset, current_pos, self._prev_target)
        self._prev_vel = torch.where(needs_reset, torch.zeros_like(self._prev_vel), self._prev_vel)

        dt = self._env.step_dt
        vmax = self.cfg.max_velocity * self.speed_scale
        amax = self.cfg.max_acceleration * self.speed_scale ** 2
        v_desired = (target_desired - self._prev_target) / dt
        v_cmd = torch.maximum(torch.minimum(v_desired, vmax), -vmax)
        if self.cfg.brake_to_target:
            # fastest speed that can still stop at the target under the
            # acceleration cap (discrete: v^2/2a + v*dt/2 <= |error|)
            err = (target_desired - self._prev_target).abs()
            v_stop = -0.5 * amax * dt + torch.sqrt((0.5 * amax * dt) ** 2 + 2.0 * amax * err)
            v_cmd = torch.maximum(torch.minimum(v_cmd, v_stop), -v_stop)
        max_dv = amax * dt
        v_cmd = torch.clamp(v_cmd, self._prev_vel - max_dv, self._prev_vel + max_dv)

        target_cmd = self._prev_target + v_cmd * dt

        self._prev_target = target_cmd
        self._prev_vel = v_cmd
        self._processed_actions = self._delayed(target_cmd, current_pos, needs_reset.squeeze(-1))

    def _delayed(self, target_cmd: torch.Tensor, current_pos: torch.Tensor, starting: torch.Tensor) -> torch.Tensor:
        """The target the joint PD gets: target_cmd itself, or (cfg.delay_steps_range / hiccup_prob) the one
        from d policy steps ago (d per episode) and, at random steps, the last delivered one again -- the
        robot's command path (network, the daemon's zero-order hold, late threads). The limiter above keeps
        its own undelayed state, as the laptop / onboard limiter does."""
        lo, hi = self.cfg.delay_steps_range
        if hi <= 0 and self.cfg.hiccup_prob <= 0.0:
            return target_cmd
        if not hasattr(self, "_delay_hist"):
            self._delay_hist = target_cmd.unsqueeze(0).repeat(hi + 1, 1, 1)
            self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._delivered = target_cmd.clone()
        if starting.any():
            self._delay_hist[:, starting] = current_pos[starting]
            self._delivered[starting] = current_pos[starting]
            self._delay_steps[starting] = torch.randint(lo, hi + 1, (int(starting.sum()),), device=self.device)
        self._delay_hist = torch.roll(self._delay_hist, 1, dims=0)
        self._delay_hist[0] = target_cmd
        late = self._delay_hist[self._delay_steps, torch.arange(self.num_envs, device=self.device)]
        if self.cfg.hiccup_prob > 0.0:
            keep = (torch.rand(self.num_envs, device=self.device) < self.cfg.hiccup_prob) & ~starting
            late = torch.where(keep.unsqueeze(-1), self._delivered, late)
        self._delivered = late
        return late

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # Defensive only -- process_actions already re-anchors both buffers
        # via episode_length_buf, this just avoids stale reads in between.
        if env_ids is None:
            self._prev_vel[:] = 0.0
        else:
            self._prev_vel[env_ids] = 0.0


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Config for RateLimitedJointPositionAction.

    max_velocity/max_acceleration are hard ceilings on the COMMANDED
    trajectory (rad/s, rad/s^2) -- independent of scale/clip, which only
    bound the raw affine-transformed target before rate-limiting.

    Defaults are deliberately more conservative than g1_arm_policies/
    config_right.yaml's validated real-hardware values (max_velocity=2.5,
    max_delta_dq=0.5 at their dt=0.01 -> 50 rad/s^2): max_velocity=2.0 sits
    a bit under both that and the right_arm actuator's own velocity_limit
    (2.175) in apple_grasp_env_cfg.py; max_acceleration=15.0 is ~3x under
    their derived 50 rad/s^2 ceiling. Loosen toward those numbers once
    you've watched a policy trained under this limiter and confirmed it
    still isn't fast enough for your taste; DON'T loosen past 50 rad/s^2
    without a specific reason -- that's the one number here actually
    grounded in a real-hardware-validated system rather than a starting
    guess.
    """

    class_type: type[ActionTerm] = RateLimitedJointPositionAction

    max_velocity: float = 2.0
    max_acceleration: float = 15.0
    # Brake toward the desired target so the command never overshoots it.
    # Off by default: without it, a target further than the stopping
    # distance (v^2 / 2a) overshoots and rings, and the trained arm policies
    # were all trained that way.
    brake_to_target: bool = False
    #: (min, max) policy steps the joint PD gets the target late, drawn per episode (0, 0 = off)
    delay_steps_range: tuple[int, int] = (0, 0)
    #: per step: the PD keeps the last delivered target instead (a late thread / lost packet)
    hiccup_prob: float = 0.0
