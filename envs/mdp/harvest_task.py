# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""MDP terms of the RL HARVEST task (AppleHarvest-v1).

Episodes start from a READY GRASP taken from a start bank (recorded by
scripts/collect_harvest_starts.py with a trained grasp policy), not by
replaying reach + grasp every episode: those two phases would take more
than half of every episode while the harvest policy has nothing to do.

  reset   pick a bank state whose apple radius matches this env's apple,
          write robot joints + apple pose, put the tree's branch above the
          apple, engage the breakable stem (envs.mdp.harvest.BreakableStem)
  act     7-dim DELTA joint targets for the right arm (shoulder, elbow,
          wrist); the fingers hold the grasp's joint targets, no action
  reward  follow a lift-and-roll reference (the scripted "side-" swing +
          twist about the stem pivot) while the stem is intact; a bonus when
          the stem breaks with the fruit in the hand; then per-step pay for
          keeping it there. Penalties: pulling on the stem instead of
          bending/twisting it, force on the fruit near its bruise threshold,
          the fruit slipping in the hand.
  end     fruit left the hand, fruit bruised, or time out.

Every observation is real-robot computable (encoders, taxels, a perceived
fruit position, time since the harvest started); the stem state, pivot and
reference are privileged and only used in rewards.
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from . import harvest as hv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Start bank
# ─────────────────────────────────────────────

def load_start_bank(path: str, device) -> dict:
    """The .pt written by scripts/collect_harvest_starts.py, tensors moved
    to `device`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Harvest start bank not found: {path}\n"
            "Record one first:  python scripts/collect_harvest_starts.py --headless "
            "--checkpoint <grasp run>/model_XXX.pt"
        )
    raw = torch.load(path, map_location="cpu")
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}


def _bank(env: "ManagerBasedRLEnv", path: str) -> dict:
    if not hasattr(env, "_harvest_bank"):
        bank = load_start_bank(path, env.device)
        robot: Articulation = env.scene["robot"]
        if list(bank["joint_names"]) != list(robot.data.joint_names):
            raise RuntimeError("start bank joint order differs from this robot's -- re-record it")
        env._harvest_bank = bank
        n = env.num_envs
        z3 = lambda: torch.zeros(n, 3, device=env.device)
        env._harvest_arm_target = torch.zeros(n, len(bank["arm_joint_names"]), device=env.device)
        env._harvest_hand_target = torch.zeros(n, len(bank["hand_joint_names"]), device=env.device)
        env._harvest_palm0_pos, env._harvest_apple0_pos = z3(), z3()
        env._harvest_palm0_quat = torch.zeros(n, 4, device=env.device)
        env._harvest_swing_axis = z3()
        env._harvest_apple0_in_palm = z3()
        env._harvest_broken_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
        env._grasp_phase_active = torch.ones(n, dtype=torch.bool, device=env.device)  # for reused terminations
        env._harvest_stem = hv.BreakableStem(env)
    return env._harvest_bank


def place_tree_above(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, apple_local: torch.Tensor,
                     stem_len: torch.Tensor, branch_radius: float) -> None:
    """Move this env's (visual-only) tree so the branch TIP sits right above
    the fruit, at the stem pivot -- same construction as
    domain_randomization.randomize_tree_and_apple, driven by a given fruit
    position instead of a sampled one."""
    try:
        import omni.usd
    except ImportError:
        return
    import random
    from .domain_randomization import _set_rotate_z, _set_translate
    stage = omni.usd.get_context().get_stage()
    radius = getattr(env, "_apple_radius", None)
    for k, idx in enumerate(env_ids.tolist()):
        ax, ay, az = apple_local[k].tolist()
        r = float(radius[idx]) if isinstance(radius, torch.Tensor) else 0.04
        branch_z = az + r + float(stem_len[k]) + branch_radius
        branch_len = float(env._branch_length[idx]) if hasattr(env, "_branch_length") else 0.5
        trunk_h = float(env._tree_height[idx]) if hasattr(env, "_tree_height") else branch_z
        rot = math.radians(random.uniform(60.0, 120.0))
        tx, ty = ax - branch_len * math.cos(rot), ay - branch_len * math.sin(rot)
        ns = f"/World/envs/env_{idx}"
        p = stage.GetPrimAtPath(f"{ns}/TreeBase")
        if p.IsValid():
            _set_translate(p, tx, ty, 0.03)
        p = stage.GetPrimAtPath(f"{ns}/TreeTrunk")
        if p.IsValid():
            _set_translate(p, tx, ty, branch_z - trunk_h / 2.0)
        p = stage.GetPrimAtPath(f"{ns}/TreeBranch")
        if p.IsValid():
            _set_translate(p, tx + branch_len / 2.0 * math.cos(rot), ty + branch_len / 2.0 * math.sin(rot), branch_z)
            _set_rotate_z(p, rot)


def reset_harvest_from_bank(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    bank_path: str,
    swing_mode: str = "side",
    swing_sign: float = -1.0,
    radius_tol: float = 0.0005,
    branch_radius: float = 0.04,
) -> None:
    """mode="reset" event. Starts each episode from a recorded ready grasp
    (see the module docstring). The fingers and arm are given the grasp's
    COMMANDED joint targets (not the measured angles, which sag under
    gravity), so the grip is exactly the one the grasp policy held."""
    bank = _bank(env, bank_path)
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=env.device)
    if len(ids) == 0:
        return
    robot: Articulation = env.scene["robot"]
    apple: RigidObject = env.scene["apple"]

    # -- pick a state recorded with a fruit of (nearly) THIS env's radius --
    # Apple radii are randomized +-20% per env at startup, and a grasp only
    # fits the fruit it was made on: states within radius_tol, or the
    # nearest one if none is that close. (Drawing from the 64
    # nearest of a 40-state bank paired fruits with grasps up to 16 mm off
    # and dropped every one of them with the arm standing still.)
    r_env = getattr(env, "_apple_radius", None)
    n_bank = bank["joint_pos"].shape[0]
    if isinstance(r_env, torch.Tensor) and "apple_radius" in bank:
        d = (bank["apple_radius"].unsqueeze(0) - r_env[ids].unsqueeze(1)).abs()        # [n, B]
        nearest = d.min(dim=1, keepdim=True).values
        ok = (d <= torch.maximum(nearest, torch.full_like(nearest, radius_tol))).float()
        k = torch.multinomial(ok, 1).squeeze(-1)
        env._harvest_radius_err = getattr(env, "_harvest_radius_err", torch.zeros(env.num_envs, device=env.device))
        env._harvest_radius_err[ids] = d[torch.arange(len(ids), device=env.device), k]
    else:
        k = torch.randint(0, n_bank, (len(ids),), device=env.device)

    # Joint positions AND the commanded targets of every joint, as recorded
    # (arm and hand targets are then driven by HarvestArmAction). Neither
    # stand-in works: targeting the measured (gravity-sagged) angles let the
    # body sag further, and the defaults are not what the grasp env was
    # commanding for the left arm and legs (0.2-0.7 rad off) -- either way
    # the body moved and carried the hand 2-11 cm off the fruit with the
    # arm idle.
    q = bank["joint_pos"][k]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.set_joint_position_target(bank["joint_target"][k], env_ids=ids)

    origins = env.scene.env_origins[ids]
    apple_pos = bank["apple_pos"][k] + origins
    apple_quat = bank["apple_quat"][k]
    state = apple.data.default_root_state[ids].clone()
    state[:, :3], state[:, 3:7], state[:, 7:] = apple_pos, apple_quat, 0.0
    apple.write_root_state_to_sim(state, env_ids=ids)

    env._harvest_arm_target[ids] = bank["arm_target"][k]
    env._harvest_hand_target[ids] = bank["hand_target"][k]
    env._harvest_palm0_pos[ids] = bank["palm_pos"][k] + origins
    env._harvest_palm0_quat[ids] = bank["palm_quat"][k]
    env._harvest_apple0_pos[ids] = apple_pos
    env._harvest_apple0_in_palm[ids] = quat_apply_inverse(env._harvest_palm0_quat[ids], apple_pos - env._harvest_palm0_pos[ids])
    env._harvest_broken_seen[ids] = False

    stem: hv.BreakableStem = env._harvest_stem
    stem.engage(ids, pos=apple_pos, quat=apple_quat,
                preload=bank["stem_force"][k] if "stem_force" in bank else None)
    env._harvest_swing_axis[ids] = hv.swing_axis_for(swing_mode, apple_pos, robot.data.root_pos_w[ids], swing_sign)
    place_tree_above(env, ids, bank["apple_pos"][k], stem.pivot[ids, 2] - apple_pos[:, 2] - _radius(env, ids), branch_radius)


def _radius(env, ids):
    r = getattr(env, "_apple_radius", None)
    return r[ids] if isinstance(r, torch.Tensor) else torch.full((len(ids),), 0.04, device=env.device)


# ─────────────────────────────────────────────
#  Action: delta joint targets for the arm, fingers held
# ─────────────────────────────────────────────

class HarvestArmAction(ActionTerm):
    """target_arm += clip(action) * delta_scale each control step (rad),
    clamped to the soft joint limits; the hand joints are held at the
    episode's recorded grasp targets. Deltas rather than absolute offsets:
    the harvest needs up to ~1.5 rad of arm motion, and an absolute-offset
    action that large turns PPO's exploration noise into violent jerks of a
    hand holding a fruit; integrated deltas turn it into a slow random walk.

    Also applies the breakable stem's wrench every physics substep."""

    cfg: "HarvestArmActionCfg"

    def __init__(self, cfg: "HarvestArmActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._arm_ids, self._arm_names = self._asset.find_joints(cfg.arm_joint_names, preserve_order=True)
        self._hand_ids, self._hand_names = self._asset.find_joints(cfg.hand_joint_names, preserve_order=True)
        n = self.num_envs
        self._raw = torch.zeros(n, len(self._arm_ids), device=self.device)
        self._target = torch.zeros(n, len(self._arm_ids), device=self.device)
        lim = self._asset.data.soft_joint_pos_limits[:, self._arm_ids]
        self._lo, self._hi = lim[..., 0], lim[..., 1]

    @property
    def action_dim(self) -> int:
        return len(self._arm_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        delta = actions.clamp(-self.cfg.clip, self.cfg.clip) * self.cfg.delta_scale
        self._target[:] = torch.maximum(torch.minimum(self._target + delta, self._hi), self._lo)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._arm_ids)
        if hasattr(self._env, "_harvest_hand_target"):
            self._asset.set_joint_position_target(self._env._harvest_hand_target, joint_ids=self._hand_ids)
        stem = getattr(self._env, "_harvest_stem", None)
        if stem is not None:
            stem.apply()

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        if hasattr(self._env, "_harvest_arm_target"):
            self._target[env_ids] = self._env._harvest_arm_target[env_ids]


@configclass
class HarvestArmActionCfg(ActionTermCfg):
    class_type: type = HarvestArmAction
    arm_joint_names: list[str] = MISSING
    hand_joint_names: list[str] = MISSING
    delta_scale: float = 0.025     # rad per control step at |action| = 1  (1.25 rad/s)
    clip: float = 3.0


# ─────────────────────────────────────────────
#  Observations (real-robot computable)
# ─────────────────────────────────────────────

def _palm(env):
    robot: Articulation = env.scene["robot"]
    pid = robot.data.body_names.index("right_hand_palm_link")
    return robot.data.body_pos_w[:, pid], robot.data.body_quat_w[:, pid]


def palm_motion_since_start(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N, 6] palm displacement since the harvest started, in the START
    palm frame: position (m) and rotation (axis-angle, rad). Forward
    kinematics only."""
    if not hasattr(env, "_harvest_palm0_quat"):     # managers probe terms before the first reset
        return torch.zeros(env.num_envs, 6, device=env.device)
    pos, quat = _palm(env)
    dp = quat_apply_inverse(env._harvest_palm0_quat, pos - env._harvest_palm0_pos)
    dq = quat_mul(quat_conjugate(env._harvest_palm0_quat), quat)
    dq = torch.where(dq[:, :1] < 0, -dq, dq)
    return torch.cat([dp, axis_angle_from_quat(dq)], dim=-1)


def harvest_time(env: "ManagerBasedRLEnv", ref_duration: float = 2.0) -> torch.Tensor:
    """[N, 1] seconds since the harvest started / ref_duration, capped at 2."""
    return (env.episode_length_buf.float() * env.step_dt / ref_duration).clamp(max=2.0).unsqueeze(-1)


# ─────────────────────────────────────────────
#  Rewards
# ─────────────────────────────────────────────

def _in_hand(env, margin: float, center_local) -> torch.Tensor:
    pos, quat = _palm(env)
    c = torch.tensor(center_local, device=env.device, dtype=quat.dtype).expand(env.num_envs, 3)
    gp = pos + quat_apply(quat, c)
    return torch.norm(env.scene["apple"].data.root_pos_w - gp, dim=-1) < margin


def reference_palm_pose(env: "ManagerBasedRLEnv", duration: float, swing_deg: float, twist_deg: float):
    """The scripted lift-and-roll (envs.mdp.harvest.harvest_palm_pose) at
    this step: swing about the episode's swing axis and twist about the stem
    axis, both through the stem pivot, quintic in time."""
    stem: hv.BreakableStem = env._harvest_stem
    t = env.episode_length_buf.float() * env.step_dt
    frac = hv.smoothstep(t / duration)
    return hv.harvest_palm_pose(
        env._harvest_palm0_pos, env._harvest_palm0_quat, stem.pivot, stem.axis, env._harvest_swing_axis,
        math.radians(swing_deg) * frac, math.radians(twist_deg) * frac,
    )


def reward_track_reference(
    env: "ManagerBasedRLEnv", duration: float = 2.0, swing_deg: float = 90.0, twist_deg: float = 90.0,
    std_pos: float = 0.03, std_rot: float = 0.35,
) -> torch.Tensor:
    """Gaussian in palm position and orientation error from the reference
    lift-and-roll, while the stem is intact (after it breaks, keeping the
    fruit is what pays -- reward_detached_hold). A guide, not the goal:
    the stem-break and hold terms outweigh it, so a better motion that
    breaks the stem sooner or more gently wins."""
    pos_d, quat_d = reference_palm_pose(env, duration, swing_deg, twist_deg)
    pos, quat = _palm(env)
    e_p = torch.norm(pos_d - pos, dim=-1)
    dq = quat_mul(quat_d, quat_conjugate(quat))
    dq = torch.where(dq[:, :1] < 0, -dq, dq)
    e_r = axis_angle_from_quat(dq).norm(dim=-1)
    intact = (~env._harvest_stem.broken).float()
    return torch.exp(-(e_p / std_pos) ** 2 / 2) * torch.exp(-(e_r / std_rot) ** 2 / 2) * intact


def reward_stem_break(env: "ManagerBasedRLEnv", margin: float, center_local) -> torch.Tensor:
    """1 on the step the stem breaks with the fruit still in the hand."""
    stem: hv.BreakableStem = env._harvest_stem
    newly = stem.broken & ~env._harvest_broken_seen
    env._harvest_broken_seen |= stem.broken
    return (newly & _in_hand(env, margin, center_local)).float()


def reward_detached_hold(env: "ManagerBasedRLEnv", margin: float, center_local) -> torch.Tensor:
    """1 per step with the stem broken and the fruit in the hand."""
    return (env._harvest_stem.broken & _in_hand(env, margin, center_local)).float()


def penalty_stem_pull(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(stem pull / its break force)^2 while intact: detach by bending and
    twisting, not by yanking -- pulling tears the stem out of the fruit
    and needs the most hand force."""
    stem: hv.BreakableStem = env._harvest_stem
    return ((stem.pull / stem.f_break.clamp(min=1e-6)) ** 2) * (~stem.broken).float()


def penalty_fruit_force(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, start_frac: float = 0.6) -> torch.Tensor:
    """Squared ramp of the largest force on the fruit (any hand link, any
    face) from start_frac of this episode's bruise threshold up to it."""
    from .grasp_rewards import _max_link_apple_force
    thr = getattr(env, "_fruit_damage_force", None)
    if thr is None:
        return torch.zeros(env.num_envs, device=env.device)
    f = _max_link_apple_force(env, sensor_cfg)
    return (((f - start_frac * thr) / ((1 - start_frac) * thr)).clamp(min=0.0)) ** 2


def penalty_fruit_slip(env: "ManagerBasedRLEnv", scale: float = 0.03) -> torch.Tensor:
    """How far the fruit has moved IN THE HAND since the harvest started
    (palm frame), / scale, capped at 1. Slip is how a grasp fails before it
    lets go."""
    pos, quat = _palm(env)
    rel = quat_apply_inverse(quat, env.scene["apple"].data.root_pos_w - pos)
    return (torch.norm(rel - env._harvest_apple0_in_palm, dim=-1) / scale).clamp(max=1.0)
