"""Squat toggle command and the scripted arm-motion disturbance.

SquatCommand
    target  = 0 stand / 1 squat — the operator toggle (one button on the real robot).
    phase s = ramps toward target at 1/transition_time per second. The policy sees
              (target, s); the deploy FSM reproduces the same ramp. The leg/waist
              reference q_ref(s) and pelvis-height reference h_ref(s) follow s.
    During training the toggle flips on a timer: stand -> squat -> hold -> stand ...,
    with occasional early flips (abort mid-transition).

ArmMotionCommand
    The squat policy never controls the arms. This term plays the role of "whatever
    else drives the arms": arms hold the stand pose while going down/up, and once the
    squat has settled they move between random poses (reaching around/down) with an
    optional object load in each hand, to train balance under arm-induced disturbances.
    When the toggle flips back to stand, the arms return to the stand pose while the
    stand-up starts. Loads stay on through the stand-up (object picked up).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from g1_squat import robot_cfg as rc

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _all_ids(env_ids, num_envs, device):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _uniform(lo_hi: tuple[float, float], n: int, device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(*lo_hi)


def sole_centres_w(robot: Articulation, feet_ids: list[int]) -> torch.Tensor:
    """World position of the centre of each sole, (N, 2, 3)."""
    from isaaclab.utils.math import quat_apply

    pos = robot.data.body_link_pos_w[:, feet_ids]
    quat = robot.data.body_link_quat_w[:, feet_ids]
    local = torch.tensor(rc.SOLE_CENTRE_LOCAL, device=pos.device).expand_as(pos)
    return pos + quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape(pos.shape)


class SquatCommand(CommandTerm):
    cfg: SquatCommandCfg

    def __init__(self, cfg: SquatCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        n, dev = self.num_envs, self.device
        self.target = torch.zeros(n, device=dev)
        self.phase = torch.zeros(n, device=dev)
        self.hold_time = torch.zeros(n, device=dev)  # time since the phase reached the target
        self._command = torch.zeros(n, 2, device=dev)

        self.joint_ids, self.joint_names = self.robot.find_joints(rc.LEG_WAIST_JOINTS, preserve_order=True)
        self.q_stand = torch.tensor([rc.STAND_FULL[j] for j in rc.LEG_WAIST_JOINTS], device=dev)
        self.q_squat = torch.tensor([rc.SQUAT_JOINT_POS[j] for j in rc.LEG_WAIST_JOINTS], device=dev)
        table = rc.reference_table()
        self._s_grid = torch.tensor(table["s"], device=dev, dtype=torch.float)
        self._h_grid = torch.tensor(table["pelvis_height"], device=dev, dtype=torch.float)
        self.feet_ids, _ = self.robot.find_bodies(rc.FEET_BODIES, preserve_order=True)

        self.metrics["height_error"] = torch.zeros(n, device=dev)
        self.metrics["squat_cycles"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        return (
            f"SquatCommand: transition {self.cfg.transition_time:.1f} s, squat pelvis "
            f"{self._h_grid[-1].item():.3f} m, stand pelvis {self._h_grid[0].item():.3f} m"
        )

    # ---- quantities shared with the action, rewards and observations -----------------
    @property
    def command(self) -> torch.Tensor:
        self._command[:, 0] = self.target
        self._command[:, 1] = self.phase
        return self._command

    @property
    def ref_weight(self) -> torch.Tensor:
        return rc.smoothstep(self.phase)

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        """(N, 15) leg+waist reference, LEG_WAIST_JOINTS order."""
        return self.q_stand + (self.q_squat - self.q_stand) * self.ref_weight.unsqueeze(-1)

    @property
    def ref_height(self) -> torch.Tensor:
        x = self.phase.clamp(0.0, 1.0) * (len(self._s_grid) - 1)
        i0 = x.floor().long().clamp(max=len(self._s_grid) - 2)
        frac = x - i0
        return self._h_grid[i0] * (1 - frac) + self._h_grid[i0 + 1] * frac

    @property
    def squat_settled(self) -> torch.Tensor:
        return (self.target > 0.5) & (self.phase >= 1.0)

    def pelvis_height(self) -> torch.Tensor:
        soles = sole_centres_w(self.robot, self.feet_ids)
        return self.robot.data.root_link_pos_w[:, 2] - soles[..., 2].mean(dim=1)

    # ---- CommandTerm interface ----------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = _all_ids(env_ids, self.num_envs, self.device)
        extras = super().reset(ids)
        self.target[ids] = 0.0
        self.phase[ids] = 0.0
        self.hold_time[ids] = 0.0
        self.time_left[ids] = _uniform(self.cfg.initial_stand_range, len(ids), self.device)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        ids = _all_ids(env_ids, self.num_envs, self.device)
        n = len(ids)
        self.target[ids] = 1.0 - self.target[ids]
        to_squat = self.target[ids] > 0.5
        hold = torch.where(
            to_squat,
            _uniform(self.cfg.squat_hold_range, n, self.device),
            _uniform(self.cfg.stand_hold_range, n, self.device),
        )
        duration = self.cfg.transition_time + hold
        abort = torch.rand(n, device=self.device) < self.cfg.abort_prob
        duration = torch.where(abort, _uniform(self.cfg.abort_range, n, self.device), duration)
        self.time_left[ids] = duration
        self.metrics["squat_cycles"][ids] += to_squat.float()

    def _update_command(self):
        dt = self._env.step_dt
        direction = self.target * 2.0 - 1.0
        new_phase = (self.phase + direction * dt / self.cfg.transition_time).clamp(0.0, 1.0)
        settled = new_phase == self.target
        self.hold_time = torch.where(settled, self.hold_time + dt, torch.zeros_like(self.hold_time))
        self.phase = new_phase

    def _update_metrics(self):
        err = (self.pelvis_height() - self.ref_height).abs()
        # running mean over the episode (reset() reports the mean of this buffer)
        steps = self._env.episode_length_buf.float().clamp(min=1.0)
        self.metrics["height_error"] += (err - self.metrics["height_error"]) / steps


@configclass
class SquatCommandCfg(CommandTermCfg):
    class_type: type = SquatCommand
    asset_name: str = "robot"
    transition_time: float = 2.0
    """Seconds for the phase to go 0 -> 1 (squat down) or 1 -> 0 (stand up)."""
    initial_stand_range: tuple[float, float] = (0.5, 3.0)
    stand_hold_range: tuple[float, float] = (1.0, 3.0)
    squat_hold_range: tuple[float, float] = (3.0, 8.0)
    abort_prob: float = 0.1
    """Probability that a toggle flips back early (mid-transition)."""
    abort_range: tuple[float, float] = (0.5, 1.5)
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # timers are set by the term itself


def _min_jerk(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x * x)


class ArmMotionCommand(CommandTerm):
    cfg: ArmMotionCommandCfg

    def __init__(self, cfg: ArmMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        n, dev = self.num_envs, self.device
        self.arm_ids, _ = self.robot.find_joints(rc.ARM_JOINTS, preserve_order=True)
        # hand joints are optional (the hand-less G1 is used for loader checks)
        self.hand_ids = [self.robot.joint_names.index(j) for j in rc.HAND_JOINTS if j in self.robot.joint_names]
        self.palm_ids, _ = self.robot.find_bodies(cfg.palm_bodies, preserve_order=True)
        self.q_default = torch.tensor([rc.ARM_REST_POS.get(j, 0.0) for j in rc.ARM_JOINTS], device=dev)
        self.hand_default = self.robot.data.default_joint_pos[0, self.hand_ids].clone()
        limits = self.robot.data.soft_joint_pos_limits[0, self.arm_ids]
        lo = torch.tensor([cfg.joint_box.get(j, (-9.0, 9.0))[0] for j in rc.ARM_JOINTS], device=dev)
        hi = torch.tensor([cfg.joint_box.get(j, (-9.0, 9.0))[1] for j in rc.ARM_JOINTS], device=dev)
        self.box_lo = torch.maximum(lo, limits[:, 0])
        self.box_hi = torch.minimum(hi, limits[:, 1])

        self.start = self.q_default.repeat(n, 1)
        self.goal = self.q_default.repeat(n, 1)
        self.current = self.q_default.repeat(n, 1)
        self.seg_t = torch.ones(n, device=dev)
        self.seg_T = torch.ones(n, device=dev)
        self.dwell_left = torch.zeros(n, device=dev)
        self.settle_delay = torch.zeros(n, device=dev)
        self.moving = torch.zeros(n, dtype=torch.bool, device=dev)  # in the random-motion mode
        self.amp = torch.zeros(n, device=dev)
        self.loads = torch.zeros(n, len(self.palm_ids), 3, device=dev)
        self._torques = torch.zeros_like(self.loads)
        self.metrics["arm_motion_amp"] = torch.zeros(n, device=dev)

    def __str__(self) -> str:
        return f"ArmMotionCommand: {len(self.arm_ids)} arm joints scripted, loads on {self.cfg.palm_bodies}"

    @property
    def command(self) -> torch.Tensor:
        """Current arm joint targets (N, 14), ARM_JOINTS order."""
        return self.current

    @property
    def load_forces(self) -> torch.Tensor:
        return self.loads.reshape(self.num_envs, -1)

    def curriculum_scale(self) -> float:
        steps = max(self.cfg.curriculum_steps, 1)
        return min(1.0, self._env.common_step_counter / steps)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = _all_ids(env_ids, self.num_envs, self.device)
        extras = super().reset(ids)
        self.start[ids] = self.q_default
        self.goal[ids] = self.q_default
        self.current[ids] = self.q_default
        self.seg_t[ids] = 1.0
        self.seg_T[ids] = 1.0
        self.dwell_left[ids] = 0.0
        self.moving[ids] = False
        self.settle_delay[ids] = _uniform(self.cfg.settle_delay_range, len(ids), self.device)
        self.loads[ids] = 0.0
        self._write_targets(ids)
        self._write_loads()
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        pass  # driven by the squat state, not by a timer

    def _start_segment(self, ids: torch.Tensor, goal: torch.Tensor, duration: torch.Tensor):
        self.start[ids] = self.current[ids]
        self.goal[ids] = goal
        self.seg_t[ids] = 0.0
        self.seg_T[ids] = duration

    def _random_goal(self, ids: torch.Tensor) -> torch.Tensor:
        n = len(ids)
        u = torch.rand(n, len(self.arm_ids), device=self.device)
        sample = self.box_lo + u * (self.box_hi - self.box_lo)
        goal = self.q_default + self.amp[ids].unsqueeze(-1) * (sample - self.q_default)
        # sometimes only one arm reaches, the other stays at the stand pose
        one_arm = torch.rand(n, device=self.device) < self.cfg.one_arm_prob
        left = torch.rand(n, device=self.device) < 0.5
        keep_left = (one_arm & ~left).unsqueeze(-1)
        keep_right = (one_arm & left).unsqueeze(-1)
        goal[:, :7] = torch.where(keep_left, self.q_default[:7], goal[:, :7])
        goal[:, 7:] = torch.where(keep_right, self.q_default[7:], goal[:, 7:])
        return goal

    def _update_command(self):
        dt = self._env.step_dt
        sq: SquatCommand = self._env.command_manager.get_term(self.cfg.squat_command_name)
        dev = self.device
        self.seg_t += dt
        seg_done = self.seg_t >= self.seg_T
        self.dwell_left = torch.where(seg_done, self.dwell_left - dt, self.dwell_left)

        # enter motion mode: squat settled for a random delay
        want = sq.squat_settled & (sq.hold_time >= self.settle_delay)
        start = want & ~self.moving
        if start.any():
            ids = start.nonzero().flatten()
            n = len(ids)
            self.moving[ids] = True
            self.amp[ids] = _uniform(self.cfg.amplitude_range, n, dev) * self.curriculum_scale()
            self.metrics["arm_motion_amp"][ids] = self.amp[ids]
            self.dwell_left[ids] = 0.0
            # object load per hand (downward) + small lateral push, kept until the next squat
            has_load = torch.rand(n, len(self.palm_ids), device=dev) < self.cfg.load_prob
            mag = torch.empty(n, len(self.palm_ids), device=dev).uniform_(*self.cfg.load_force_range)
            mag = mag * has_load * self.curriculum_scale()
            lateral = torch.empty(n, len(self.palm_ids), 2, device=dev).uniform_(
                -self.cfg.lateral_force, self.cfg.lateral_force
            ) * self.curriculum_scale()
            self.loads[ids, :, :2] = lateral
            self.loads[ids, :, 2] = -mag

        # in motion mode: new random goal after each segment + dwell
        next_goal = self.moving & want & seg_done & (self.dwell_left <= 0.0)
        if next_goal.any():
            ids = next_goal.nonzero().flatten()
            n = len(ids)
            self._start_segment(ids, self._random_goal(ids), _uniform(self.cfg.move_time_range, n, dev))
            self.dwell_left[ids] = _uniform(self.cfg.dwell_range, n, dev)

        # toggle flipped back to stand: return to the stand pose
        stop = self.moving & ~sq.squat_settled
        if stop.any():
            ids = stop.nonzero().flatten()
            self.moving[ids] = False
            goal = self.q_default.repeat(len(ids), 1)
            self._start_segment(ids, goal, torch.full((len(ids),), self.cfg.return_time, device=dev))
            self.settle_delay[ids] = _uniform(self.cfg.settle_delay_range, len(ids), dev)

        # squat started again: drop the old load
        new_squat = (sq.target > 0.5) & (sq.phase < 1.0) & ~self.moving
        self.loads[new_squat] = 0.0

        alpha = _min_jerk(self.seg_t / self.seg_T.clamp(min=1e-3)).unsqueeze(-1)
        self.current = self.start + (self.goal - self.start) * alpha
        self._write_targets(None)
        self._write_loads()

    def _write_targets(self, ids):
        if ids is None:
            self.robot.set_joint_position_target(self.current, joint_ids=self.arm_ids)
            self.robot.set_joint_position_target(
                self.hand_default.expand(self.num_envs, -1), joint_ids=self.hand_ids
            )
        else:
            self.robot.set_joint_position_target(self.current[ids], joint_ids=self.arm_ids, env_ids=ids)
            self.robot.set_joint_position_target(
                self.hand_default.expand(len(ids), -1), joint_ids=self.hand_ids, env_ids=ids
            )

    def _write_loads(self):
        if self.cfg.load_force_range[1] <= 0.0:
            return
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self.loads, torques=self._torques, body_ids=self.palm_ids, is_global=True
        )

    def _update_metrics(self):
        pass


@configclass
class ArmMotionCommandCfg(CommandTermCfg):
    class_type: type = ArmMotionCommand
    asset_name: str = "robot"
    squat_command_name: str = "squat"
    # the URDF importer merges the fixed palm link into wrist_yaw_link
    palm_bodies: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    joint_box: dict[str, tuple[float, float]] = MISSING
    """Absolute joint ranges the random arm goals are drawn from (clipped to soft limits)."""
    settle_delay_range: tuple[float, float] = (0.2, 1.0)
    move_time_range: tuple[float, float] = (0.6, 1.8)
    dwell_range: tuple[float, float] = (0.0, 1.2)
    return_time: float = 0.8
    amplitude_range: tuple[float, float] = (0.3, 1.0)
    one_arm_prob: float = 0.3
    load_prob: float = 0.6
    load_force_range: tuple[float, float] = (0.0, 6.0)
    """Downward force per hand, N (6 N ~ a 0.6 kg object)."""
    lateral_force: float = 2.0
    curriculum_steps: int = 24 * 1500
    """Env steps over which arm amplitude and loads ramp from 0 to full."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
