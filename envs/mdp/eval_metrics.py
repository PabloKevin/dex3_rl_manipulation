# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Per-episode evaluation recorder, wired in as an always-False termination term.

Why a termination term rather than an outer-loop recorder in the eval script:

    ManagerBasedRLEnv.step() does, in order,
        physics -> episode_length_buf += 1
        -> termination_manager.compute()   <- terminal state still readable HERE
        -> reward_manager.compute()
        -> _reset_idx(reset_env_ids)       <- robot teleported, apple re-sampled
        -> observation_manager.compute() -> return

    By the time env.step() RETURNS, every terminated env has already been reset:
    joint_pos, apple.data.root_pos_w and episode_length_buf all describe the
    NEXT episode. A recorder in the rollout loop therefore attributes the new
    episode's apple position and joint state to the episode that just finished
    -- silently, and only for the subset of envs that happened to terminate that
    step. With 64 envs terminating at different times (reach_success is a real
    termination, so successful episodes end early) that corrupts most records.

    TerminationManager.compute() evaluates every term unconditionally, in
    declaration order, at exactly the right instant. So the recorder lives there
    and returns all-False, contributing nothing to `dones`. Note the same trick
    does NOT work with a reward term: RewardManager.compute skips terms whose
    weight is 0.0.

The probe must be the LAST termination term so that time_out / apple_dropped /
reach_success have already written their columns this step; evaluate_policy.py
appends it after the rest and asserts active_terms[-1] == "eval_probe".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .reach_rewards import _palm_normal_angle, _palm_target_dist

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EvalProbe(ManagerTermBase):
    """Accumulates per-episode reach metrics on the GPU, flushing completed episodes to CPU.

    All public state is drained with :meth:`pop_episodes`. Returns an all-False
    done mask every step, so it is invisible to the MDP.
    """

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        # num_envs / device come from ManagerTermBase as read-only properties.
        self.dt = float(env.step_dt)
        self.T = int(env.max_episode_length)

        p = cfg.params
        self.pos_tol = float(p.get("pos_tol", 0.04))
        self.ori_tol_rad = float(p.get("ori_tol_rad", 0.30))
        self.offset_z = float(p.get("offset_z", 0.08))
        self.record_traces = bool(p.get("record_traces", True))
        self.n_joints = len(p["arm_cfg"].joint_names)

        self._global_step = 0
        self._ep_counter = 0
        self._records: list[dict] = []
        self._trace_records: list[dict] = []

        z = lambda: torch.zeros(self.num_envs, device=self.device)
        zl = lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._steps = zl()
        self._d_last, self._th_last = z(), z()
        self._d_min = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._th_min = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._first_step = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._n_in_tol_after, self._n_steps_after, self._n_exits = zl(), zl(), zl()
        self._prev_in_tol = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Geometry / state latched every step; at flush these hold the finished
        # episode's LAST step, which is the pose the policy actually reached.
        self._apple_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._shoulder_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._q_last = torch.zeros(self.num_envs, self.n_joints, device=self.device)

        # Termination flags latched this step, consumed by the NEXT step's flush.
        b = lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_success, self._last_timeout, self._last_dropped = b(), b(), b()

        if self.record_traces:
            tj = lambda k: torch.zeros(self.num_envs, self.T, k, device=self.device)
            self._tr_q, self._tr_dq = tj(self.n_joints), tj(self.n_joints)
            self._tr_a, self._tr_u = tj(self.n_joints), tj(self.n_joints)
            self._tr_d = torch.zeros(self.num_envs, self.T, device=self.device)
            self._tr_th = torch.zeros(self.num_envs, self.T, device=self.device)
            self._tr_mask = torch.zeros(self.num_envs, self.T, device=self.device, dtype=torch.bool)

    # ── manager interface ────────────────────────────────────────────────

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        arm_cfg: SceneEntityCfg,
        shoulder_cfg: SceneEntityCfg,
        action_term_name: str = "right_arm",
        pos_tol: float = 0.04,
        ori_tol_rad: float = 0.30,
        offset_z: float = 0.08,
        palm_normal_axis: Sequence[float] = (0.0, 0.0, 1.0),
        palm_local_offset: Sequence[float] = (0.0, 0.0, 0.0),
        success_term_name: str = "reach_success",
        record_traces: bool = True,
    ) -> torch.Tensor:
        # 1. Close out episodes that ended on the PREVIOUS step. episode_length_buf
        #    was zeroed by _reset_idx and is back to 1 now, so this is the first
        #    step of a fresh episode. `_steps > 0` skips the initial full reset
        #    issued by RslRlVecEnvWrapper before any episode has run.
        fresh = (env.episode_length_buf == 1) & (self._steps > 0)
        if bool(fresh.any()):
            self._flush(torch.nonzero(fresh, as_tuple=False).squeeze(-1))

        d = _palm_target_dist(env, robot_cfg, object_cfg, offset_z, palm_local_offset)
        theta = _palm_normal_angle(env, robot_cfg, palm_normal_axis)
        t = (env.episode_length_buf - 1).clamp(min=0)

        in_tol = (d < pos_tol) & (theta < ori_tol_rad)
        entered = self._first_step >= 0

        self._d_last, self._th_last = d, theta
        self._d_min = torch.minimum(self._d_min, d)
        self._th_min = torch.minimum(self._th_min, theta)
        self._first_step = torch.where(in_tol & ~entered, t, self._first_step)
        entered = self._first_step >= 0
        self._n_steps_after += entered.long()
        self._n_in_tol_after += (in_tol & entered).long()
        self._n_exits += ((~in_tol) & self._prev_in_tol & entered).long()
        self._prev_in_tol = in_tol
        self._steps = env.episode_length_buf.clone()

        robot = env.scene[robot_cfg.name]
        self._apple_w = env.scene[object_cfg.name].data.root_pos_w.clone()
        # Use the SAME shoulder reference the sampler uses, not the live body
        # pose. right_shoulder_pitch_link's origin translates by several cm as
        # the shoulder rotates, whereas randomize_tree_and_apple caches
        # env._shoulder_local_xyz once at the first reset and samples the fruit
        # against that fixed point. Reading the live pose here would put the
        # recovered (azimuth, radius) in a different frame than the one the
        # fruit was drawn in, and the coverage plot would no longer be in the
        # sampler's own coordinates.
        if hasattr(env, "_shoulder_local_xyz"):
            self._shoulder_w = env.scene.env_origins + env._shoulder_local_xyz
        else:
            self._shoulder_w = robot.data.body_pos_w[:, shoulder_cfg.body_ids[0], :].clone()
        q = robot.data.joint_pos[:, arm_cfg.joint_ids]
        dq = robot.data.joint_vel[:, arm_cfg.joint_ids]
        self._q_last = q.clone()

        if self.record_traces:
            term = env.action_manager.get_term(action_term_name)
            ti = t.clamp(max=self.T - 1)
            rows = torch.arange(self.num_envs, device=self.device)
            self._tr_q[rows, ti] = q
            self._tr_dq[rows, ti] = dq
            self._tr_a[rows, ti] = term.raw_actions
            self._tr_u[rows, ti] = term.processed_actions
            self._tr_d[rows, ti] = d
            self._tr_th[rows, ti] = theta
            self._tr_mask[rows, ti] = True

        # Latch terminations for the next step's flush. Requires this term to be
        # declared LAST so the other terms' columns are already written.
        tm = env.termination_manager
        self._last_success = tm.get_term(success_term_name).clone()
        self._last_timeout = tm.time_outs.clone()
        try:
            self._last_dropped = tm.get_term("apple_dropped").clone()
        except (KeyError, ValueError):
            self._last_dropped = torch.zeros_like(self._last_success)

        self._global_step += 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        # Deliberately a no-op on the accumulators: episode boundaries are
        # detected from episode_length_buf in __call__, which fires strictly
        # BEFORE _reset_idx and therefore still sees the terminal state. Zeroing
        # here would race that and drop the final step of every episode.
        return

    # ── draining ─────────────────────────────────────────────────────────

    def n_done(self) -> int:
        return len(self._records)

    def _flush(self, env_ids: torch.Tensor) -> None:
        """Move the finished episodes in `env_ids` to CPU and clear their slots."""
        e = env_ids
        steps = self._steps[e]
        first = self._first_step[e]
        reached = first >= 0

        t_first = torch.where(reached, first.float() * self.dt,
                              torch.full_like(first, -1, dtype=torch.float32))
        success = self._last_success[e]
        t_succ = torch.where(success, steps.float() * self.dt,
                             torch.full_like(steps, -1, dtype=torch.float32))
        frac = torch.where(
            self._n_steps_after[e] > 0,
            self._n_in_tol_after[e].float() / self._n_steps_after[e].clamp(min=1).float(),
            torch.full_like(steps, float("nan"), dtype=torch.float32),
        )

        delta = self._apple_w[e] - self._shoulder_w[e]
        r_xy = torch.norm(delta[:, :2], dim=-1)
        az = torch.rad2deg(torch.atan2(delta[:, 1], delta[:, 0]))
        dz = delta[:, 2] - self.offset_z
        r3d = torch.sqrt(r_xy ** 2 + dz ** 2)

        cpu = lambda x: x.detach().cpu().numpy()
        n = int(e.numel())
        rec = {
            "ep_env_id": cpu(e).astype(np.int16),
            "ep_global_step": np.full(n, self._global_step, dtype=np.int32),
            "ep_length": cpu(steps).astype(np.int16),
            "ep_success": cpu(success).astype(bool),
            "ep_timeout": cpu(self._last_timeout[e]).astype(bool),
            "ep_dropped": cpu(self._last_dropped[e]).astype(bool),
            "d_final": cpu(self._d_last[e]).astype(np.float32),
            "d_min": cpu(self._d_min[e]).astype(np.float32),
            "theta_final": cpu(self._th_last[e]).astype(np.float32),
            "theta_min": cpu(self._th_min[e]).astype(np.float32),
            "first_reach_step": cpu(first).astype(np.int16),
            "t_first_reach": cpu(t_first).astype(np.float32),
            "t_success": cpu(t_succ).astype(np.float32),
            "frac_in_tol_after": cpu(frac).astype(np.float32),
            "n_tol_exits": np.where(cpu(reached), cpu(self._n_exits[e]), -1).astype(np.int16),
            "apple_pos_w": cpu(self._apple_w[e]).astype(np.float32),
            "shoulder_pos_w": cpu(self._shoulder_w[e]).astype(np.float32),
            "r_xy": cpu(r_xy).astype(np.float32),
            "r3d_target": cpu(r3d).astype(np.float32),
            "azimuth_deg": cpu(az).astype(np.float32),
            "dz_target": cpu(dz).astype(np.float32),
            "q_final": cpu(self._q_last[e]).astype(np.float32),
        }
        # -1 sentinels become NaN so the numpy side never averages them in.
        for k in ("t_first_reach", "t_success"):
            rec[k] = np.where(rec[k] < 0, np.nan, rec[k]).astype(np.float32)
        self._records.append(rec)

        if self.record_traces:
            self._trace_records.append({
                "traj_q": cpu(self._tr_q[e]).astype(np.float32),
                "traj_dq": cpu(self._tr_dq[e]).astype(np.float32),
                "traj_a_raw": cpu(self._tr_a[e]).astype(np.float32),
                "traj_u": cpu(self._tr_u[e]).astype(np.float32),
                "traj_d": cpu(self._tr_d[e]).astype(np.float32),
                "traj_theta": cpu(self._tr_th[e]).astype(np.float32),
                "traj_mask": cpu(self._tr_mask[e]).astype(bool),
            })
            for buf in (self._tr_q, self._tr_dq, self._tr_a, self._tr_u,
                        self._tr_d, self._tr_th):
                buf[e] = 0.0
            self._tr_mask[e] = False

        self._steps[e] = 0
        self._d_min[e] = float("inf")
        self._th_min[e] = float("inf")
        self._first_step[e] = -1
        self._n_in_tol_after[e] = 0
        self._n_steps_after[e] = 0
        self._n_exits[e] = 0
        self._prev_in_tol[e] = False

    def pop_episodes(self, warmup_steps: int = 0) -> tuple[dict, dict]:
        """Concatenate all flushed episodes into (episodes, traces) numpy dicts.

        Episodes completing within the first `warmup_steps` rollout steps are
        dropped: the first episode after the initial reset starts from a
        not-yet-randomized scene in some envs and with cold reward/action
        history buffers, so it is not representative.
        """
        if not self._records:
            return {}, {}
        cat = lambda recs, k: np.concatenate([r[k] for r in recs], axis=0)
        eps = {k: cat(self._records, k) for k in self._records[0]}
        keep = eps["ep_global_step"] > warmup_steps
        eps = {k: v[keep] for k, v in eps.items()}
        eps["ep_index"] = np.arange(keep.sum(), dtype=np.int32)

        traces = {}
        if self.record_traces and self._trace_records:
            traces = {k: cat(self._trace_records, k)[keep] for k in self._trace_records[0]}
        return eps, traces


def make_eval_probe_term(
    offset_z: float,
    palm_local_offset: Sequence[float],
    palm_normal_axis: Sequence[float],
    arm_joint_names: Sequence[str],
    pos_tol: float = 0.04,
    ori_tol_rad: float = 0.30,
    palm_body: str = "right_hand_palm_link",
    shoulder_body: str = "right_shoulder_pitch_link",
    record_traces: bool = True,
) -> TerminationTermCfg:
    """Build the probe's TerminationTermCfg so scripts never hand-assemble it."""
    return TerminationTermCfg(
        func=EvalProbe,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=[palm_body]),
            "object_cfg": SceneEntityCfg("apple"),
            "arm_cfg": SceneEntityCfg("robot", joint_names=list(arm_joint_names),
                                      preserve_order=True),
            "shoulder_cfg": SceneEntityCfg("robot", body_names=[shoulder_body]),
            "action_term_name": "right_arm",
            "pos_tol": pos_tol,
            "ori_tol_rad": ori_tol_rad,
            "offset_z": offset_z,
            "palm_normal_axis": tuple(palm_normal_axis),
            "palm_local_offset": tuple(palm_local_offset),
            "success_term_name": "reach_success",
            "record_traces": record_traces,
        },
    )
