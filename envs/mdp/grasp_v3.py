# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""AppleGraspArmPose-v3: the grasp episode shaped like its use in the harvest (scripts/play_harvest_robot.py
--learned_grasp ... --learned_hold freeze_after_grasp).

The harvest uses the learned grasp ONLY to close the hand: once two finger groups have read > 0.5 N for 0.3 s and
kept it 0.5 s more (or 4 s after the cup with no such grip), the fingers are FROZEN at their last command and the
script pulls the fruit off its stem and carries it to the basket. AppleGraspArmPose-v2 trained something else:
the policy kept regulating the fingers under a synthetic load while the arm stood still, the stem let go on a
schedule unrelated to the grip, and its success term (a lift) was unreachable with the arm locked -- ~70% of its
episodes lost the fruit and nothing in its reward said "this close survives the harvest".

Here:
  - freeze_on_bench_grip (event): the bench's rule, the same numbers. At the freeze the hand's raw action is
    stored and replayed from then on (FreezableHandAction) -- the policy's later outputs are ignored, as on the
    bench -- and the stem lets go (env._stem_released), which starts the harvest load
    (tree_grasp.release_stem_and_twist set to the bench's straight pull: no twist, the stem holding back with
    6-14 N, then a carry force);
  - harvest_held (termination, success): the fruit still in the hand load_s after the freeze. Paid one-off,
    as large as the loss penalty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .grasp_autoplay import ReachHandoffJointPositionAction, ReachHandoffJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _st(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_v3"):
        N, d = env.num_envs, env.device
        env._v3 = dict(
            frozen=torch.zeros(N, dtype=torch.bool, device=d),
            raw=torch.zeros(N, 7, device=d),
            grip_t=torch.zeros(N, device=d),          # s the grip (>= 2 groups > thr) has lasted
            t_freeze=torch.full((N,), float("nan"), device=d),   # episode time of the freeze
            by_grip=torch.zeros(N, dtype=torch.bool, device=d),
            stats={},
        )
    return env._v3


def reset_v3(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset"."""
    S = _st(env)
    S["frozen"][env_ids] = False
    S["grip_t"][env_ids] = 0.0
    S["t_freeze"][env_ids] = float("nan")
    S["by_grip"][env_ids] = False


def freeze_on_bench_grip(env: "ManagerBasedRLEnv", env_ids, sensor_cfg: SceneEntityCfg, level_n: float = 0.5,
                         groups: int = 2, grip_s: float = 0.3, hold_s: float = 0.5, deadline_s: float = 4.1,
                         hand_action: str = "right_hand", pull_break: tuple[float, float] = (6.0, 10.0),
                         bend_break: tuple[float, float] = (0.15, 0.25)) -> None:
    """mode="interval" (every step). In the grasp phase: grip = >= `groups` finger groups (thumb / index /
    middle mean pad force, as the policy reads it) above level_n; freeze once it has lasted grip_s + hold_s, or
    deadline_s after the hand-off (the bench: 0.1 s cup + 4 s). The freeze stores the hand's raw action and
    releases the stem."""
    from .observations import tactile_force_level

    if not hasattr(env, "_grasp_phase_active"):
        return
    S = _st(env)
    if not getattr(sensor_cfg, "_v3_resolved", False):
        sensor_cfg.resolve(env.scene)
        sensor_cfg._v3_resolved = True
    lvl = tactile_force_level(env, sensor_cfg, all_apples=True)
    grip = (lvl > level_n).sum(-1) >= groups
    active = env._grasp_phase_active & ~S["frozen"]
    S["grip_t"] = torch.where(active & grip, S["grip_t"] + env.step_dt, torch.zeros_like(S["grip_t"]))
    since = (env.episode_length_buf - env._grasp_handoff_step).float() * env.step_dt
    by_grip = active & (S["grip_t"] >= grip_s + hold_s - 1e-6)
    new = by_grip | (active & (since >= deadline_s))
    if bool(new.any()):
        term = env.action_manager.get_term(hand_action)
        S["raw"][new] = term.raw_actions[new]
        S["frozen"] |= new
        S["by_grip"] |= by_grip
        S["t_freeze"][new] = env.episode_length_buf[new].float() * env.step_dt
        env._stem_released |= new
        # the bench's weaken_target (--stem_break_pull 6 10, --target_bend 0.15 0.25): the stem now breaks under
        # the pull of the arm going back (PathBackArmAction), as on the bench
        T = getattr(env, "_tree", None)
        if T is not None:
            ids = new.nonzero(as_tuple=True)[0]
            T.target_stem.f_break[ids] = torch.empty(len(ids), device=env.device).uniform_(*pull_break)
            T.target_stem.m_bend_break[ids] = torch.empty(len(ids), device=env.device).uniform_(*bend_break)


def harvest_held(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg, center_local,
                 carry_s: float = 1.5, tol: float = 0.06) -> torch.Tensor:
    """Termination [N] bool (success): after the freeze the stem has broken (the arm going back pulled the fruit
    off) and the fruit has stayed within tol of the grasp pocket for carry_s since. Early on purpose: the
    decisive actions come BEFORE the freeze, and an outcome 5 s later reached them at 0.99^250 = 8% (run
    2026-11-14_06-11-20: flat for 110 iterations)."""
    from .tree_grasp import _pocket_offset

    S = getattr(env, "_v3", None)
    if S is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    T = getattr(env, "_tree", None)
    if T is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    now = env.episode_length_buf.float() * env.step_dt
    if "t_broke" not in S:
        S["t_broke"] = torch.full_like(now, float("nan"))
    fresh = env.episode_length_buf <= 1
    S["t_broke"] = torch.where(fresh, torch.full_like(now, float("nan")), S["t_broke"])
    newly = S["frozen"] & T.target_stem.broken & torch.isnan(S["t_broke"])
    S["t_broke"] = torch.where(newly, now, S["t_broke"])
    done = torch.nan_to_num(now - S["t_broke"], nan=-1.0) >= carry_s
    return done & (_pocket_offset(env, center_local).norm(dim=-1) < tol)


def v3_metrics(env: "ManagerBasedRLEnv", env_ids, ema: float = 0.05) -> dict:
    """Curriculum-term logger (changes nothing), on the episodes ending: frozen at all / by the grip rule,
    held (success), lost, damaged, timed out."""
    S = getattr(env, "_v3", None)
    if S is None or env.common_step_counter < 1:
        return {}
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return dict(S["stats"])
    tm = env.termination_manager

    def term(k):
        return tm.get_term(k)[ids].float().mean().item() if k in tm.active_terms else 0.0

    lost = tm.get_term("apple_fell_from_grip")[ids] if "apple_fell_from_grip" in tm.active_terms else \
        torch.zeros(len(ids), dtype=torch.bool, device=env.device)
    t_fz = env.episode_length_buf[ids].float() * env.step_dt - S["t_freeze"][ids]
    T = getattr(env, "_tree", None)
    broke = T.target_stem.broken[ids] if T is not None else torch.zeros_like(lost)
    vals = dict(frozen=S["frozen"][ids].float().mean().item(), frozen_by_grip=S["by_grip"][ids].float().mean().item(),
                held=term("harvest_held"), lost=term("apple_fell_from_grip"), damaged=term("fruit_damaged"),
                timeout=env.reset_time_outs[ids].float().mean().item(),
                # where the losses happen: before the freeze (the hand broke the stem), in the pull (stem still
                # on), after it came off (the carry); and of those frozen by the grip
                lost_before_freeze=(lost & ~S["frozen"][ids]).float().mean().item(),
                lost_stem_on=(lost & S["frozen"][ids] & ~broke).float().mean().item(),
                lost_carry=(lost & S["frozen"][ids] & broke).float().mean().item(),
                lost_after_grip=(lost & S["by_grip"][ids]).float().mean().item()
                / max(S["by_grip"][ids].float().mean().item(), 1e-6),
                lost_t_after_freeze=float(torch.nan_to_num(t_fz[lost], nan=0.0).mean()) if bool(lost.any()) else 0.0)
    st = S["stats"]
    for k, v in vals.items():
        st[k] = v if k not in st else (1.0 - ema) * st[k] + ema * v
    return dict(st)


class FreezableHandAction(ReachHandoffJointPositionAction):
    """The hand term; once freeze_on_bench_grip froze an env, its stored raw action replaces the policy's."""

    def process_actions(self, actions: torch.Tensor):
        S = getattr(self._env, "_v3", None)
        if S is not None and bool(S["frozen"].any()):
            actions = torch.where(S["frozen"].unsqueeze(-1), S["raw"], actions)
        super().process_actions(actions)


@configclass
class FreezableHandActionCfg(ReachHandoffJointPositionActionCfg):
    class_type: type = FreezableHandAction


class PathBackArmAction(ReachHandoffJointPositionAction):
    """The arm term. Records its own target every reach-phase step; once the hand is frozen, the grasp-phase
    anchor (the trainee's arm scale is 0: the anchor IS the target) walks that path backwards at
    cfg.back_rate recorded steps per step -- the bench's detach (--detach_path monotone: back along the
    approach) and the move home, with the real stem, the fruit's inertia and gravity turning with the hand."""

    cfg: "PathBackArmActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, j = self._processed_actions.shape
        self._L = int(cfg.path_steps)
        self._path = torch.zeros(n, self._L, j, device=self.device)
        self._plen = torch.zeros(n, dtype=torch.long, device=self.device)
        self._back = torch.zeros(n, device=self.device)
        self._was_frozen = torch.zeros(n, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        S = getattr(self._env, "_v3", None)
        if S is not None:
            fz = S["frozen"]
            start = fz & ~self._was_frozen
            self._back = torch.where(start, (self._plen - 1).clamp(min=0).float(), self._back)
            self._was_frozen = fz.clone()
            if bool(fz.any()):
                self._back = torch.where(fz, (self._back - self.cfg.back_rate).clamp(min=0.0), self._back)
                k = self._back.long()
                tgt = self._path[torch.arange(len(k), device=self.device), k]
                # low-pass: the recorded reach targets buzz (bang-bang rate limiter); the bench's detach is smooth
                a = min(1.0, self._env.step_dt / self.cfg.smooth_s)
                tgt = self._anchor + a * (tgt - self._anchor)
                self._anchor = torch.where((fz & (self._plen > 0)).unsqueeze(-1), tgt, self._anchor)
        super().process_actions(actions)
        reaching = ~grasp_phase(self._env)
        if bool(reaching.any()):
            ids = reaching.nonzero(as_tuple=True)[0]
            k = self._plen[ids].clamp(max=self._L - 1)
            self._path[ids, k] = self._prev_target[ids]
            self._plen[ids] = (self._plen[ids] + 1).clamp(max=self._L)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._plen[env_ids] = 0
        self._back[env_ids] = 0.0
        self._was_frozen[env_ids] = False


def grasp_phase(env) -> torch.Tensor:
    from .grasp_autoplay import grasp_phase_active

    return grasp_phase_active(env)


@configclass
class PathBackArmActionCfg(ReachHandoffJointPositionActionCfg):
    class_type: type = PathBackArmAction
    #: recorded reach steps (12 s at 50 Hz)
    path_steps: int = 600
    #: recorded steps walked back per control step (1.0: as fast as the reach came in)
    back_rate: float = 0.5
    #: low-pass time constant of the walked-back target, s
    smooth_s: float = 0.25
