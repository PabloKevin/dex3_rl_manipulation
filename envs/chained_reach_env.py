# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Two-stage reach, trained CONNECTED: AppleReachFinalChained-v1.

Every episode starts at the home pose with the hand open, exactly like
AppleReachPregrasp. Per env, two stages:

  stage 1  the FROZEN part-1 policy (a pregrasp checkpoint, loaded inside the
           env) drives the arm + waist. The learner's action is ignored, its
           reward is zeroed. Stage 1 ends when part 1's own success criterion
           fires (palm at the pregrasp point, held), or after `pregrasp_s`.
  hand-off the hand targets go to pregrasp.FINAL_HAND_PRESHAPE (the cup) and
           the palm tactile input is switched on -- what the robot does
  stage 2  the LEARNER drives, with AppleReachFinal's rewards and success
           (gentle palm touch, held). Episode ends on success, knock-off or
           the time limit.

So part 2 sees the states part 1 actually leaves it in (position error,
residual velocity, arm posture), not a hand-made start library.

Training data note: the learner's stage-1 steps are executed with another
policy's action and carry zero reward, so they are no-ops for PPO's expected
gradient (they only add variance). Part 1 settling early shortens them.

Only the stage-2 learner is trained; part 1 stays untouched.
"""

from __future__ import annotations

import copy
import functools
import glob
import os

import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv

from .mdp import pregrasp as pg
from .mdp import terminations_and_events as te

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREGRASP_LOG_ROOT = os.path.join(_REPO, "logs", "rsl_rl", "apple_reach_pregrasp_v2")


def newest_checkpoint(root: str) -> str:
    """Highest model_<iter>.pt of the newest run under root (warmstart_* skipped)."""
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and "warmstart" not in d)
    for run in reversed(runs):
        cks = glob.glob(os.path.join(run, "model_*.pt"))
        if cks:
            return max(cks, key=lambda p: int(os.path.basename(p)[6:-3]))
    raise FileNotFoundError(f"no model_*.pt under {root}: train the pregrasp (part 1) first")


class FrozenActor(nn.Module):
    """The actor MLP + empirical observation normaliser of an rsl_rl checkpoint
    (rsl-rl 3.x: keys actor.<i>.* and actor_obs_normalizer.*), mean action."""

    def __init__(self, ckpt_path: str, device):
        super().__init__()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        layers = []
        for n, i in enumerate(idx):
            w = sd[f"actor.{i}.weight"]
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if n < len(idx) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.in_dim = layers[0].in_features
        self.mlp.load_state_dict({
            f"{2 * n}.{t}": sd[f"actor.{i}.{t}"] for n, i in enumerate(idx) for t in ("weight", "bias")
        })
        self.register_buffer("mean", sd["actor_obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["actor_obs_normalizer._std"].clone())
        self.eps = 1e-2  # rsl_rl EmpiricalNormalization default
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + self.eps))


# ── observation / termination terms that know the stage ─────────────────────

def stage_touch_obs(env, sensor_cfg) -> torch.Tensor:
    """Palm tactile force: zero in stage 1 (as part 1 was trained), real in
    stage 2. Signature matches observations.palm_contact_force."""
    from .mdp.observations import palm_contact_force
    f = palm_contact_force(env, sensor_cfg)
    stage2 = getattr(env, "_stage2", None)
    if stage2 is None:
        return f
    return f * stage2.unsqueeze(-1).to(f.dtype)


@functools.wraps(te.reach_success)  # the term manager checks this signature
def staged_reach_success(env, *args, **params):
    """Stage 1: observe part 1's success (never terminates). Stage 2: the
    real touch success, which terminates. Separate hold counters."""
    stage2 = env._stage2
    out = {}
    for st, p in ((1, env._p1_params), (2, params)):
        env._reach_hold_counter = env._hold_cnt[st]
        out[st] = te.reach_success(env, **p)
        env._hold_cnt[st] = env._reach_hold_counter
    env._p1_success = out[1] & ~stage2
    env._reach_success = out[2] & stage2   # read by the success-bonus reward
    return out[2] & stage2


class ChainedReachEnv(ManagerBasedRLEnv):
    """See the module docstring. Extra cfg fields (declared on the env cfg):
    pregrasp_ckpt ("" = newest under pregrasp_log_root, default
    logs/rsl_rl/apple_reach_pregrasp_v2),
    pregrasp_s (stage-1 cap, seconds)."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        n, dev = cfg.scene.num_envs, cfg.sim.device
        # state tensors exist before the managers' first observation / reset
        self._stage2 = torch.zeros(n, dtype=torch.bool, device=dev)
        self._p1_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._p1_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hold_cnt = {1: torch.zeros(n, device=dev), 2: torch.zeros(n, device=dev)}
        self._p1_params = None
        self._frozen = None
        self._pregrasp_ckpt = cfg.pregrasp_ckpt or None
        self._pregrasp_s = float(cfg.pregrasp_s)
        self._p1_root = getattr(cfg, "pregrasp_log_root", "") or PREGRASP_LOG_ROOT
        self._p1_params_cfg = cfg.pregrasp_success_params
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    # lazily, once the scene exists
    def _ensure_ready(self):
        if self._frozen is not None:
            return
        path = self._pregrasp_ckpt or newest_checkpoint(self._p1_root)
        print(f"[chained] part-1 policy: {path}")
        self._frozen = FrozenActor(path, self.device)
        # part 1 may have been trained WITHOUT the tactile term (v3: 39 inputs):
        # feed it the learner's observation minus that term's columns
        om = self.observation_manager
        cols, i = [], 0
        for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
            w = int(torch.tensor(dim).prod())
            if name != "palm_touch_force" or self._frozen.in_dim == sum(
                    int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["policy"]):
                cols += range(i, i + w)
            i += w
        if len(cols) != self._frozen.in_dim:
            raise RuntimeError(f"part-1 policy takes {self._frozen.in_dim} inputs, the env gives {len(cols)}")
        self._p1_cols = torch.tensor(cols, device=self.device)
        self._p1_params = {k: (copy.copy(v) if hasattr(v, "resolve") else v)
                           for k, v in self._p1_params_cfg.items()}
        for v in self._p1_params.values():
            if hasattr(v, "resolve"):
                v.resolve(self.scene)
        robot = self.scene["robot"]
        self._hand_ids = torch.tensor([robot.joint_names.index(j) for j in pg.FINAL_HAND_PRESHAPE], device=self.device)
        self._hand_vals = torch.tensor(list(pg.FINAL_HAND_PRESHAPE.values()), device=self.device)
        self._p1_cap = max(1, round(self._pregrasp_s / self.step_dt))
        self._palm_bid = robot.data.body_names.index("right_hand_palm_link")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._stage2[env_ids] = False
        self._p1_steps[env_ids] = 0
        self._p1_success[env_ids] = False

    def step(self, action):
        self._ensure_ready()
        in1 = ~self._stage2.clone()
        if bool(in1.any()):
            a1 = self._frozen(self.obs_buf["policy"][:, self._p1_cols])
            action = torch.where(in1.unsqueeze(-1), a1.to(action.dtype), action)

        obs, rew, terminated, truncated, extras = super().step(action)

        rew = rew * (~in1).to(rew.dtype)           # stage-1 steps carry no reward
        was_reset = terminated | truncated         # these envs are already back at stage 1
        self._p1_steps += in1.long()
        self._p1_steps[was_reset] = 0

        # the env ids that hand off now
        handoff = in1 & ~was_reset & (self._p1_success | (self._p1_steps >= self._p1_cap))
        ids = handoff.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            robot = self.scene["robot"]
            tgt = robot.data.joint_pos_target[ids].clone()
            tgt[:, self._hand_ids] = self._hand_vals
            robot.set_joint_position_target(tgt, env_ids=ids)
            self._stage2[ids] = True
            self._hold_cnt[2][ids] = 0.0
            apple = self.scene["apple"]
            d = torch.norm(apple.data.root_pos_w[ids] - robot.data.body_pos_w[ids, self._palm_bid], dim=-1)
            log = extras.setdefault("log", {})
            log["Chained/handoff_part1_success"] = float(self._p1_success[ids].float().mean())
            log["Chained/handoff_palm_apple_dist_m"] = float(d.mean())
            log["Chained/handoff_after_s"] = float(self._p1_steps[ids].float().mean() * self.step_dt)
        return obs, rew, terminated, truncated, extras
