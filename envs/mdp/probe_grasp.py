# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Grasp that finishes the approach by feel (AppleGraspSingle-v2,
envs/apple_grasp_single_v2_env_cfg.py).

The single reach stops 2 cm short of a perceived apple point that is itself
off by up to 2.5 cm, so the grasp has to find the fruit: move until a pad
feels it, read WHERE it feels it (tip, base, palm), bring the fruit into the
pocket with the arm, squeeze from opposite sides so it does not just push
the hanging fruit away -- closing, opening and closing again if needed --
and never drop it.

  - tactile realism (opt-in, only where reset_tactile_realism runs):
    per-taxel dead band (the real palm answered only 2 of its 9 elements at
    2.45 N, logs/sim2real/tactile_force_*.json -- it needs a hard press),
    per-taxel gain, dead taxels, per-step noise. observations.
    _sensed_tactile_forces applies it, so every sensed reader (inputs AND
    the grip detector) sees the same readings. Noise is drawn once per
    control step and shared by all readers of that step.
  - inputs: tactile_taxels (per location, the 7 places of sim2real/
    tactile_io.LOCATIONS) and tactile_memory (decaying peak: after the hand
    re-opens it still says where the fruit was felt).
  - rewards, privileged (true fruit, true stem), all zero in the reach phase:
    pocket progress (potential difference), contact depth (palm > base >
    tip), fruit displacement / stem stress / early stem break (do not push
    or detach it), centred(...) (a grasp-quality term scaled by how centred
    the fruit is in the pocket), after_release(...) (a term only once the
    stem has let go), trunk clearance, and terminations.

Rewards that are EVENTS (one step) or potential DIFFERENCES return value /
step_dt: the reward manager multiplies every term by step_dt, so their
weight reads directly as the reward per event / per unit of potential.

Wrapped terms name their inner function by a STRING (term="reward_...") and
take its params as a dict (inner={...}): Isaac Lab checks each term's
params against its function signature (no **kwargs), and the Hydra round
trip re-resolves a term's func from its "module:name" string, which would
drop a functools wrapper (see grasp_rewards' module docstring).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

PALM = "right_hand_palm_link"
#: taxel columns (scripts/tactile_layout.TAXELS order) per tactile_io.LOCATIONS
#: place: thumb prox/dist, middle prox/dist, index prox/dist, palm (3 taxels)
LOCATION_TAXELS = ([0], [1], [2], [3], [4], [5], [6, 7, 8])
PROX_TAXELS = (0, 2, 4)
DIST_TAXELS = (1, 3, 5)
PALM_TAXELS = (6, 7, 8)
#: a location "feels" the fruit above this sensed force (N)
FEEL_THRESHOLD = 0.05


# ─────────────────────────────────────────────
#  Tactile realism
# ─────────────────────────────────────────────

def reset_tactile_realism(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    finger_deadband: tuple[float, float] = (0.2, 0.6),
    palm_deadband: tuple[float, float] = (1.5, 3.5),
    gain_log_range: tuple[float, float] = (0.6, 1.6),
    dead_taxel_prob: float = 0.03,
    noise_rel: float = 0.10,
    noise_abs: float = 0.15,
) -> None:
    """mode="reset": this episode's sensor model, per env and taxel (9,
    tactile_layout.TAXELS order): dead band (N) uniform in finger_deadband,
    palm_deadband for the 3 palm taxels; gain log-uniform in gain_log_range
    (what is left of the per-pad calibration error after
    sim2real/tactile_force_calibrate); a taxel dead for the episode with
    dead_taxel_prob. Also clears the contact memory."""
    N, dev = env.num_envs, env.device
    R = getattr(env, "_tactile_realism", None)
    if R is None:
        R = env._tactile_realism = {
            "deadband": torch.zeros(N, 9, device=dev),
            "gain": torch.ones(N, 9, device=dev),
            "alive": torch.ones(N, 9, device=dev),
            "noise_rel": noise_rel,
            "noise_abs": noise_abs,
            "step": -1,
        }
    n = len(env_ids)
    db = torch.empty(n, 9, device=dev).uniform_(*finger_deadband)
    db[:, list(PALM_TAXELS)] = torch.empty(n, len(PALM_TAXELS), device=dev).uniform_(*palm_deadband)
    R["deadband"][env_ids] = db
    lo, hi = math.log(gain_log_range[0]), math.log(gain_log_range[1])
    R["gain"][env_ids] = torch.exp(torch.empty(n, 9, device=dev).uniform_(lo, hi))
    R["alive"][env_ids] = (torch.rand(n, 9, device=dev) >= dead_taxel_prob).float()
    mem = getattr(env, "_tactile_memory", None)
    if mem is not None:
        mem[env_ids] = 0.0


def apply_tactile_realism(env: "ManagerBasedRLEnv", sensed: torch.Tensor) -> torch.Tensor:
    """[N, 9] true pad force -> what the real pad reports: minus the dead
    band (zero below it, as the robot's calibrated reading with its dead
    zone), times the gain and (1 + relative noise), plus absolute noise on
    readings above the dead band only (the robot's dead zone zeroes the
    resting noise), dead taxels 0, >= 0. The noise draw is held for the
    whole control step, so the slip input, the grip detector and the
    rewards of one step all see the same reading."""
    R = env._tactile_realism
    step = int(env.common_step_counter)
    if R["step"] != step or R.get("n_rel") is None or R["n_rel"].shape != sensed.shape:
        R["n_rel"] = torch.randn_like(sensed)
        R["n_abs"] = torch.randn_like(sensed)
        R["step"] = step
    above = (sensed - R["deadband"]).clamp(min=0.0)
    on = (above > 0).float()
    out = above * R["gain"] * (1.0 + R["noise_rel"] * R["n_rel"]) + on * R["noise_abs"] * R["n_abs"]
    return (out * R["alive"]).clamp(min=0.0)


def _locations(sensed: torch.Tensor) -> torch.Tensor:
    """[N, 9] taxels -> [N, 7] places (palm = mean of its 3 taxels)."""
    return torch.stack([sensed[:, ids].mean(dim=-1) for ids in LOCATION_TAXELS], dim=-1)


def tactile_taxels(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N, 7] sensed force per place (thumb prox/dist, middle prox/dist,
    index prox/dist, palm), log1p(F / 1 N): any apple, pad face only, with
    the realism model. The robot computes the same from tactile_io forces."""
    from .observations import _sensed_tactile_forces

    return torch.log1p(_locations(_sensed_tactile_forces(env, sensor_cfg, all_apples=True)))


def tactile_memory(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, tau_s: float = 1.0) -> torch.Tensor:
    """[N, 7] decaying peak of tactile_taxels (time constant tau_s): where
    the fruit was felt recently, still there after the hand opens again.
    Updated once per control step; cleared at reset (reset_tactile_realism)."""
    now = tactile_taxels(env, sensor_cfg)
    mem = getattr(env, "_tactile_memory", None)
    if mem is None or mem.shape != now.shape:
        mem = env._tactile_memory = torch.zeros_like(now)
        env._tactile_memory_step = -1
    step = int(env.common_step_counter)
    if env._tactile_memory_step != step:
        decay = math.exp(-env.step_dt / tau_s)
        fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
        mem.copy_(torch.where(fresh, now, torch.maximum(now, mem * decay)))
        env._tactile_memory_step = step
    return mem.clone()


# ─────────────────────────────────────────────
#  Privileged state
# ─────────────────────────────────────────────

def _grasp_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    from .grasp_rewards import _grasp_mask as gm

    return gm(env)


def _released(env: "ManagerBasedRLEnv") -> torch.Tensor:
    from .grasp_autoplay import stem_released

    return stem_released(env)


def pocket_err(env: "ManagerBasedRLEnv", center_local) -> torch.Tensor:
    """[N, 3] true fruit centre in the palm frame minus the grasp pocket."""
    robot = env.scene["robot"]
    if not hasattr(env, "_probe_palm_id"):
        env._probe_palm_id = robot.data.body_names.index(PALM)
    pid = env._probe_palm_id
    q = robot.data.body_quat_w[:, pid]
    d = env.scene["apple"].data.root_pos_w - robot.data.body_pos_w[:, pid]
    return quat_apply(quat_conjugate(q), d) - torch.tensor(center_local, device=env.device, dtype=d.dtype)


def apple_displacement(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] how far the target fruit is from where it hung at the episode
    start (grasp_autoplay.reset_reach_grasp_phase records it), m."""
    ref = getattr(env, "_grasp_handoff_apple_pos", None)
    if ref is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (env.scene["apple"].data.root_pos_w - ref).norm(dim=-1)


def _target_stem(env: "ManagerBasedRLEnv"):
    T = getattr(env, "_tree", None)
    return None if T is None else getattr(T, "target_stem", None)


def stem_stress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] target stem load as a fraction of its break limit (the worst of
    pull beyond the fruit's own weight, bend, twist); 0 when not attached."""
    st = _target_stem(env)
    if st is None:
        return torch.zeros(env.num_envs, device=env.device)
    on = (st.active & ~st.broken).float()
    pull = (st.pull - st.rest_force.norm(dim=-1)).clamp(min=0.0) / st.f_break.clamp(min=1e-6)
    bend = st.bend / st.m_bend_break.clamp(min=1e-6)
    twist = st.twist / st.m_twist_break.clamp(min=1e-6)
    return torch.stack([pull, bend, twist], dim=-1).amax(dim=-1) * on


def _sensed_target(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """[N, 9] sensed taxel force from the TARGET fruit (realism applied)."""
    from .observations import _sensed_tactile_forces

    return _sensed_tactile_forces(env, sensor_cfg)


# ─────────────────────────────────────────────
#  Rewards
# ─────────────────────────────────────────────

def reset_probe_state(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """mode="reset": forget the pocket potential and the one-off latches."""
    N, dev = env.num_envs, env.device
    if not hasattr(env, "_probe_phi"):
        env._probe_phi = torch.zeros(N, device=dev)
        env._probe_phi_valid = torch.zeros(N, dtype=torch.bool, device=dev)
        env._probe_stem_broken_prev = torch.zeros(N, dtype=torch.bool, device=dev)
        env._probe_trunk_hit = torch.zeros(N, dtype=torch.bool, device=dev)
    env._probe_phi_valid[env_ids] = False
    env._probe_stem_broken_prev[env_ids] = False
    env._probe_trunk_hit[env_ids] = False


def reward_pocket_progress(env: "ManagerBasedRLEnv", center_local, scale_far: float = 0.10,
                           scale_near: float = 0.02) -> torch.Tensor:
    """[N] potential difference (Phi_now - Phi_prev) / step_dt with Phi =
    (exp(-e / scale_far) + exp(-e / scale_near)) / 2, e = |pocket error|
    (the far part still pulls at the 6-21 cm the single reach leaves, the
    near part sharpens the last 2 cm): bringing the fruit into the pocket
    pays (weight = reward for going from far to centred), drifting out
    costs as much, holding still is 0. Pushing the hanging fruit away with
    the palm pays nothing: the fruit recedes with the hand. Starts at the
    hand-off (first grasp step: 0)."""
    if not hasattr(env, "_probe_phi"):
        reset_probe_state(env, torch.arange(env.num_envs, device=env.device))
    e = pocket_err(env, center_local).norm(dim=-1)
    phi = 0.5 * (torch.exp(-e / scale_far) + torch.exp(-e / scale_near))
    active = _grasp_mask(env) > 0
    valid = env._probe_phi_valid & active
    r = torch.where(valid, phi - env._probe_phi, torch.zeros_like(phi))
    env._probe_phi.copy_(torch.where(active, phi, env._probe_phi))
    env._probe_phi_valid |= active
    return r / env.step_dt


def reward_contact_depth(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, tip: float = 0.3,
                         base: float = 0.7, disp_std: float = 0.015) -> torch.Tensor:
    """[N] the deepest place the hand FEELS the target fruit (sensed, after
    the dead band): fingertips only `tip`, a finger base `base`, the palm or
    the thumb against index/middle (opposing pads) 1. Times exp(-(disp /
    disp_std)^2) while the stem holds it: feeling it by shoving it off its
    hanging spot does not count. The "keep reaching until you feel it, and
    feel it deep in the hand" signal."""
    s = _sensed_target(env, sensor_cfg) > FEEL_THRESHOLD
    d = s[:, list(DIST_TAXELS)].any(dim=-1)
    p = s[:, list(PROX_TAXELS)].any(dim=-1)
    palm = s[:, list(PALM_TAXELS)].any(dim=-1)
    thumb = s[:, [0, 1]].any(dim=-1)
    fingers = s[:, [2, 3, 4, 5]].any(dim=-1)
    level = torch.zeros(env.num_envs, device=env.device)
    level = torch.where(d, torch.full_like(level, tip), level)
    level = torch.where(p, torch.full_like(level, base), level)
    level = torch.where(palm | (thumb & fingers), torch.ones_like(level), level)
    still = torch.exp(-(apple_displacement(env) / disp_std) ** 2)
    still = torch.where(_released(env), torch.ones_like(still), still)
    return level * still * _grasp_mask(env)


def penalty_apple_displacement(env: "ManagerBasedRLEnv", deadband: float = 0.01, ref: float = 0.02) -> torch.Tensor:
    """[N] ((disp - deadband)+ / ref)^2 while the stem holds the fruit:
    pushing it off its hanging spot. NEGATIVE weight."""
    x = (apple_displacement(env) - deadband).clamp(min=0.0) / ref
    return x ** 2 * (~_released(env)).float() * _grasp_mask(env)


def penalty_stem_stress(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] stem_stress^2 while the stem holds the fruit: pulling, bending or
    twisting it toward its break (detaching it) before the harvest. NEGATIVE
    weight."""
    return stem_stress(env) ** 2 * (~_released(env)).float() * _grasp_mask(env)


def penalty_stem_broken_early(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] one-off (/ step_dt) the step the hand breaks the target stem
    before the scheduled release. NEGATIVE weight = penalty per event."""
    st = _target_stem(env)
    if st is None:
        return torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_probe_stem_broken_prev"):
        reset_probe_state(env, torch.arange(env.num_envs, device=env.device))
    newly = st.broken & ~env._probe_stem_broken_prev
    env._probe_stem_broken_prev.copy_(st.broken)
    return (newly & ~_released(env)).float() * _grasp_mask(env) / env.step_dt


def penalty_terminated(env: "ManagerBasedRLEnv", term_keys: list[str]) -> torch.Tensor:
    """[N] one-off (/ step_dt) when any of these termination terms ended the
    episode this step (time-outs never count). NEGATIVE weight = penalty per
    event."""
    tm = env.termination_manager
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for k in term_keys:
        if k in tm.active_terms:
            hit |= tm.get_term(k)
    return (hit & ~env.reset_time_outs).float() / env.step_dt


def apple_left_hand(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
                    center_local, leave_margin: float = 0.06) -> torch.Tensor:
    """Termination [N] bool: terminations_and_events.apple_fell_from_grip
    (fruit more than leave_margin from the grasp pocket), but only once the
    fruit is OFF its stem -- the scheduled release, or the hand broke it.
    While the stem holds it, a fruit far from the pocket is one the grasp
    still has to reach: the single reach hands off with it 6-21 cm away
    (measured, policy_1600), and the plain test ended those episodes on the
    first grasp step."""
    from .terminations_and_events import apple_fell_from_grip

    fell = apple_fell_from_grip(env, object_cfg, robot_cfg=robot_cfg, center_local=center_local,
                                leave_margin=leave_margin)
    st = _target_stem(env)
    off = _released(env) if st is None else (_released(env) | st.broken)
    return fell & off


def _trunk_depth(env: "ManagerBasedRLEnv", margin: float) -> torch.Tensor:
    """[N, A] per right arm / hand sphere: intrusion (m, >= 0) into the
    trunk capsules grown by `margin` -- the LAST TWO capsules of
    env._tree_segments (apple_tree.place_apple_tree: the trunk, its upper
    segment on a spindle, else a dummy far away)."""
    from .self_collision import _build, _world_spheres

    if not hasattr(env, "_self_collision_spheres"):
        env._self_collision_spheres = _build(env)
    arm = env._self_collision_spheres["arm"]
    if not hasattr(env, "_tree_segments"):
        return torch.zeros(env.num_envs, len(arm["r"]), device=env.device)
    p = _world_spheres(env, arm)                                    # [N, A, 3]
    seg = env._tree_segments[:, -2:]                                # [N, 2, 2, 3]
    a = seg[:, :, 0].unsqueeze(1)
    ab = seg[:, :, 1].unsqueeze(1) - a
    ap = p.unsqueeze(2) - a                                         # [N, A, 2, 3]
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    dist = (ap - t.unsqueeze(-1) * ab).norm(dim=-1)                 # [N, A, 2]
    need = arm["r"].view(1, -1, 1) + env._tree_radii[-2:].view(1, 1, -1) + margin
    return (need - dist).clamp(min=0.0).amax(dim=-1)


def penalty_trunk_contact(env: "ManagerBasedRLEnv", margin: float = 0.02, ref: float = 0.02) -> torch.Tensor:
    """[N] sum over arm / hand spheres of (depth / ref)^2 into the trunk
    grown by `margin`: 0 with `margin` of clearance. The tree prims have no
    colliders, so this is all that keeps the hand out of the trunk.
    NEGATIVE weight."""
    return ((_trunk_depth(env, margin) / ref) ** 2).sum(dim=-1) * _grasp_mask(env)


def penalty_trunk_hit(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[N] one-off (/ step_dt) the first step of the episode the arm or hand
    actually enters the trunk. NEGATIVE weight = penalty per event."""
    if not hasattr(env, "_probe_trunk_hit"):
        reset_probe_state(env, torch.arange(env.num_envs, device=env.device))
    inside = (_trunk_depth(env, 0.0) > 0).any(dim=-1) & (_grasp_mask(env) > 0)
    newly = inside & ~env._probe_trunk_hit
    env._probe_trunk_hit |= inside
    return newly.float() / env.step_dt


# ─────────────────────────────────────────────
#  Wrappers around grasp_rewards terms
# ─────────────────────────────────────────────

def _call(env: "ManagerBasedRLEnv", term: str, inner: dict) -> torch.Tensor:
    """grasp_rewards.<term>(env, **inner), SceneEntityCfgs in `inner`
    resolved once (the manager only resolves top-level params)."""
    from . import grasp_rewards as gr

    done = getattr(env, "_probe_resolved", None)
    if done is None:
        done = env._probe_resolved = set()
    for v in inner.values():
        if isinstance(v, SceneEntityCfg) and id(v) not in done:
            v.resolve(env.scene)
            done.add(id(v))
    return getattr(gr, term)(env, **inner)


def centred(env: "ManagerBasedRLEnv", term: str, inner: dict, center_local, std: float = 0.02,
            floor: float = 0.25) -> torch.Tensor:
    """grasp_rewards.<term> times max(floor, exp(-|pocket error|^2 / 2 std^2)):
    a grasp closed with the fruit on the fingertips earns `floor` of it, one
    with the fruit in the pocket all of it -- what makes opening and
    repositioning worth it."""
    e = pocket_err(env, center_local).norm(dim=-1)
    gate = torch.exp(-0.5 * (e / std) ** 2).clamp(min=floor)
    return _call(env, term, inner) * gate


def after_release(env: "ManagerBasedRLEnv", term: str, inner: dict) -> torch.Tensor:
    """grasp_rewards.<term>, only once the stem has let go (the final grasp):
    probing -- closing, opening, closing again -- before that is free."""
    return _call(env, term, inner) * _released(env).float()
