# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach -> grasp hand-off, as an IN-EPISODE phase.

Every grasp episode has two phases, tracked per env in env._grasp_phase_active:

  1. REACH phase (episode start): the frozen, already-trained reach policy
     drives the arm, the hand is held at its default (open) pose -- exactly
     the conditions the reach policy was trained under -- and the apple is
     pinned in place. The grasp trainee's actions are IGNORED, and every
     grasp reward/termination is gated to zero (see grasp_phase_active).
  2. GRASP phase: the trainee drives arm+hand. Its arm actions are offsets
     from the arm target the reach policy left behind (not from the default
     pose), so a zero action holds the hand-off pose. The apple stays pinned
     until the trainee first achieves force closure, then gravity is real.

The switch happens in update_reach_grasp_phase (an every-step interval
event, so the policy observation computed right after it already carries
the new phase flag): after a fixed handoff_time_s ("fixed"), or once the
palm has held the reach-success neighbourhood for hold_time ("converged",
with handoff_time_s as the watchdog).

Why not run the reach policy inside a mode="reset" event (the previous
design): PhysX steps every env together, so each reset batch's autoplay
advanced physics time for ALL the other, mid-episode envs while their
policies were frozen and their apples were no longer re-pinned -- a
silent multi-second time jump in the middle of their trajectories. And
since resets are desynchronised across envs, that autoplay ran almost
every step: the last run with it collected at ~160 FPS vs ~3400 FPS for
reach training. In-episode, the reach phase costs one extra (tiny) policy
forward pass per step and nothing else. The price is that the reach-phase
steps are PPO samples whose actions have no effect; the grasp_phase
observation lets the critic tell them apart, and their policy-gradient
contribution is zero in expectation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import os

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .actions import RateLimitedJointPositionAction, RateLimitedJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_reach_policy_cache: dict[str, torch.jit.ScriptModule] = {}

# Same hardcoded-with-env-override convention as scripts/run_env.py's
# BASE_DIR / G1_DEX3_USD_PATH.
_BASE_DIR = os.environ.get("DEX3_RL_BASE_DIR",
                           os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# The reach policy the grasp task hands off from, PINNED to a frozen copy
# rather than "latest apple_grasp_reach run" -- so training the reach task
# further never silently changes (or, mid-run with no export yet, breaks)
# the grasp task. Copied from logs/rsl_rl/apple_grasp_reach/
# 2026-09-18_04-01-01/exported/policy.pt (commit 5f6276a, 29-dim obs). To
# hand off from a newer reach policy, overwrite this file on purpose (or
# point the arm action's reach_policy_path somewhere else).
PINNED_REACH_POLICY = os.path.join(_BASE_DIR, "envs", "agents", "best_weights", "grasp_handoff_reach_apple.pt")


def _find_last_reach_policy(experiment_name: str = "apple_grasp_reach") -> str:
    """Path to the latest run's JIT-exported policy.pt under
    logs/rsl_rl/<experiment_name> -- same convention (and same "only the
    exported file works with torch.jit.load, not the raw model_*.pt
    checkpoints" caveat) as scripts/run_env.py's _find_last_policy.

    No longer used for the hand-off itself (see PINNED_REACH_POLICY); kept
    for scripts/run_grasp_env.py, which uses it generically to find the
    latest GRASP export.
    """
    base_dir = os.path.join(_BASE_DIR, "logs", "rsl_rl", experiment_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No training runs found at {base_dir}.")
    latest_folder = max(os.listdir(base_dir))
    exported_path = os.path.join(base_dir, latest_folder, "exported", "policy.pt")
    if not os.path.isfile(exported_path):
        raise FileNotFoundError(
            f"No exported policy at {exported_path}. Run play.py against this "
            f"checkpoint first (it exports policy.pt as a side effect)."
        )
    return exported_path


def _load_reach_policy(policy_path: str, device) -> torch.jit.ScriptModule:
    """Lazily torch.jit.load the frozen reach policy once per process. The
    exported JIT module already bakes in the observation normalizer, so
    calling it directly reproduces what the reach task itself would output
    for the same observation.
    """
    key = os.path.abspath(policy_path)
    if key not in _reach_policy_cache:
        if not os.path.isfile(key):
            raise FileNotFoundError(
                f"Reach hand-off policy not found at {key}. Export one with "
                f"scripts/play.py --task AppleReach-Play-v1 and copy its "
                f"exported/policy.pt there (see grasp_autoplay.PINNED_REACH_POLICY)."
            )
        _reach_policy_cache[key] = torch.jit.load(key, map_location=device).eval()
    return _reach_policy_cache[key]


def _reach_obs(
    env: "ManagerBasedRLEnv",
    arm_cfg: SceneEntityCfg,
    robot_body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_local_offset: tuple[float, float, float],
    prev_raw_action: torch.Tensor,
    layout: str = "v1",
    waist: dict | None = None,
) -> torch.Tensor:
    """layout "waist": the AppleReachSpindle (waist) policy's 40-dim vector --
    the "tree" 37 with the palm orientation taken in the torso frame WITHOUT
    the waist yaw (waist_reach.palm_quat_torso_unyawed), the clearance ring
    at waist["ring_range"], then waist yaw position (vs default), velocity
    and the policy's previous raw waist action (waist["joint_cfg"],
    waist["prev_raw"] [N, 1]).

    layout "tree": the AppleReachTree-v1 policy's 37-dim vector instead --
    the same 29 with palm_touch_force from ANY contact, then the 8-sector
    neighbour clearance ring (envs/apple_reach_tree_env_cfg.py).

    Reassemble the reach policy's own 29-dim observation vector (all
    envs) by calling the SAME functions envs/apple_grasp_env_cfg.py's
    ObservationsCfg.PolicyObs wires up -- NOT the ObservationManager, which
    belongs to the grasp task here.

    Order MUST match apple_grasp_env_cfg.ObservationsCfg.PolicyObs exactly:
    right_arm_pos(7), right_arm_vel(7), apple_pos_palm(3),
    palm_orientation_torso(4), last_action(7), palm_touch_force(1) = 29.
    Update this the moment that ObservationsCfg's term list/order changes
    again -- there is no automatic check that these two stay in sync
    (scripts/reach_apple_hybrid.py's OBS_POLICY_DIM guard catches the
    DDS-deployment side of the same drift risk, but not this one).

    `arm_cfg` must be resolved WITHOUT preserve_order, like the reach task's
    own right_arm_pos/right_arm_vel terms, so the joint order matches what
    the reach policy was trained on.
    """
    from .observations import (
        joint_pos_selected,
        joint_vel_selected,
        object_pos_relative_to_body,
        body_quat_relative_to_body,
        palm_contact_force,
    )

    if not hasattr(env, "_autoplay_torso_cfg"):
        torso_cfg = SceneEntityCfg("robot", body_names=["torso_link"])
        torso_cfg.resolve(env.scene)
        env._autoplay_torso_cfg = torso_cfg
    if not hasattr(env, "_autoplay_sensor_cfg"):
        sensor_cfg = SceneEntityCfg("contact_forces")
        sensor_cfg.resolve(env.scene)
        env._autoplay_sensor_cfg = sensor_cfg

    if layout == "visible":
        # AppleReachVisible-v1 (envs/apple_reach_visible_env_cfg.py): waist held,
        # 37 = arm pos (7), vel (7), the PERCEIVED apple - palm in the torso's yaw
        # frame (3), palm orientation in the torso frame (4), previous raw action
        # (7), the perceived ring with the unseen neighbours masked (8), hand phase
        from .single_reach import hand_phase_obs, neighbour_ring_measured
        from .visible_reach import apple_pos_palm_measured_torso

        return torch.cat([
            joint_pos_selected(env, arm_cfg),
            joint_vel_selected(env, arm_cfg),
            apple_pos_palm_measured_torso(env, object_cfg, robot_body_cfg, local_offset=palm_local_offset),
            body_quat_relative_to_body(env, robot_body_cfg, env._autoplay_torso_cfg),
            prev_raw_action,
            neighbour_ring_measured(env, max_range=waist["ring_range"]),
            hand_phase_obs(env),
        ], dim=-1)

    if layout == "single":
        # AppleReachSingle-v1 (envs/apple_reach_single_env_cfg.py): the
        # "waist" vector without the touch slot, the PERCEIVED apple
        # (single_reach.apple_pos_measured), hand schedule phase last = 40
        from .single_reach import apple_pos_palm_measured, hand_phase_obs, neighbour_ring_measured
        from .waist_reach import palm_quat_torso_unyawed

        if not hasattr(env, "_autoplay_pelvis_cfg"):
            env._autoplay_pelvis_cfg = SceneEntityCfg("robot", body_names=["pelvis"])
            env._autoplay_pelvis_cfg.resolve(env.scene)
        wcfg = waist["joint_cfg"]
        return torch.cat([
            joint_pos_selected(env, arm_cfg),
            joint_vel_selected(env, arm_cfg),
            apple_pos_palm_measured(env, object_cfg, robot_body_cfg, local_offset=palm_local_offset),
            palm_quat_torso_unyawed(env, robot_body_cfg, env._autoplay_torso_cfg, env._autoplay_pelvis_cfg, wcfg),
            prev_raw_action,
            neighbour_ring_measured(env, max_range=waist["ring_range"]),
            joint_pos_selected(env, wcfg),
            joint_vel_selected(env, wcfg),
            waist["prev_raw"],
            hand_phase_obs(env),
        ], dim=-1)

    if layout == "waist":
        from .tree_reach import neighbour_clearance_ring, palm_contact_force_any
        from .waist_reach import palm_quat_torso_unyawed

        if not hasattr(env, "_autoplay_pelvis_cfg"):
            env._autoplay_pelvis_cfg = SceneEntityCfg("robot", body_names=["pelvis"])
            env._autoplay_pelvis_cfg.resolve(env.scene)
        wcfg = waist["joint_cfg"]
        return torch.cat([
            joint_pos_selected(env, arm_cfg),
            joint_vel_selected(env, arm_cfg),
            object_pos_relative_to_body(env, object_cfg, robot_body_cfg, local_offset=palm_local_offset),
            palm_quat_torso_unyawed(env, robot_body_cfg, env._autoplay_torso_cfg, env._autoplay_pelvis_cfg, wcfg),
            prev_raw_action,
            palm_contact_force_any(env, env._autoplay_sensor_cfg),
            neighbour_clearance_ring(env, max_range=waist["ring_range"]),
            joint_pos_selected(env, wcfg),
            joint_vel_selected(env, wcfg),
            waist["prev_raw"],
        ], dim=-1)

    parts = [
        joint_pos_selected(env, arm_cfg),
        joint_vel_selected(env, arm_cfg),
        object_pos_relative_to_body(env, object_cfg, robot_body_cfg, local_offset=palm_local_offset),
        body_quat_relative_to_body(env, robot_body_cfg, env._autoplay_torso_cfg),
        prev_raw_action,
    ]
    if layout == "tree":
        from .tree_reach import neighbour_clearance_ring, palm_contact_force_any
        parts += [palm_contact_force_any(env, env._autoplay_sensor_cfg), neighbour_clearance_ring(env)]
    else:
        parts.append(palm_contact_force(env, env._autoplay_sensor_cfg))
    return torch.cat(parts, dim=-1)


# ─────────────────────────────────────────────
#  Phase state
# ─────────────────────────────────────────────

def grasp_phase_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[num_envs] bool: True once an env has been handed off to the grasp
    trainee this episode. Every grasp reward/termination multiplies by (or
    ANDs with) this, so the trainee is never credited or blamed for what the
    frozen reach policy did. All-True before the first reset has created the
    state (defensive only).
    """
    if not hasattr(env, "_grasp_phase_active"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return env._grasp_phase_active


def stem_released(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[num_envs] bool: has the compliant stem let go of the apple this
    episode -- on the trainee's first latched grip OR once this episode's
    randomized release time after hand-off has passed, whichever is first
    (see update_first_closure). Falls back to grip_established when the
    timed release isn't configured."""
    if not hasattr(env, "_stem_released"):
        return grip_established(env)
    return env._stem_released


def shaping_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[num_envs] bool: may the grasp-shaping rewards pay this step? True
    once the stem has let go (the hand holds the fruit itself), and before
    that only for env._shaping_grace_s after the hand-off. All-True when no
    grace is configured, or before the first reset created the state."""
    grace = getattr(env, "_shaping_grace_s", None)
    if grace is None or not hasattr(env, "_grasp_handoff_step"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if isinstance(grace, (tuple, list)):
        grace = _grace_schedule(env, *grace)
    since_handoff = (env.episode_length_buf - env._grasp_handoff_step).float() * env.step_dt
    return stem_released(env) | (since_handoff < grace)


def _grace_schedule(
    env: "ManagerBasedRLEnv",
    start_s: float,
    end_s: float,
    start_step: int,
    end_step: int,
) -> float:
    """Training curriculum for the shaping grace: start_s until
    env.common_step_counter reaches start_step, then linearly to end_s at
    end_step, end_s after. common_step_counter counts env.step() calls
    (one per PPO rollout step), so with num_steps_per_env = 64,
    iteration i is step 64 * i.

    Why a curriculum: a fixed 2 s grace from the first iteration cut the
    signal for closing the fingers to 2 of the 5 s grasp window, and a
    fresh policy needs most of that just to close -- 400 iterations
    learned nothing. Earlier runs, with no grace, learned to close within
    ~100 iterations; the grace is only needed once there is a grasp to
    keep undetected.
    """
    t = env.common_step_counter
    if t <= start_step:
        return start_s
    if t >= end_step:
        return end_s
    return start_s + (end_s - start_s) * (t - start_step) / (end_step - start_step)


def grip_established(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """[num_envs] bool: has the trainee held a closure long enough this
    episode for the stem to let go (env._grasp_ever_closed)? All-False
    before the first reset has created the state."""
    if not hasattr(env, "_grasp_ever_closed"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._grasp_ever_closed


def grasp_phase_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Policy observation: 1.0 in the grasp phase, 0.0 in the reach phase.
    Lets the critic tell the (action-independent) reach-phase steps apart.

    Shape: [num_envs, 1]
    """
    return grasp_phase_active(env).float().unsqueeze(-1)


def reset_reach_grasp_phase(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    stem_release_time_range: tuple[float, float] | None = None,
    shaping_grace_s: float | tuple[float, float, int, int] | None = None,
    stem_release_schedule: tuple[float, float, int, int, float] | None = None,
    release_on_latch: bool = True,
) -> None:
    """mode="reset" event -- MUST come after the apple placement event
    (randomize_tree), since it records the apple's freshly-placed pose as
    this episode's pin pose.

    Puts the env back into the reach phase and records the per-episode
    bookkeeping every other grasp term reads:
      _grasp_phase_active       -> False (reach phase)
      _grasp_ever_closed        -> False (apple pinned until first closure)
      _grasp_handoff_apple_pos/quat -> the pin pose; also the reference
                                   apple_fell_from_grip / grasp_success's
                                   lift check measure against
      _grasp_handoff_joint_pos  -> placeholder until the real hand-off
                                   (penalty_joint_deviation_from_handoff is
                                   gated off until then anyway)
      _stem_release_time        -> seconds after hand-off at which the stem
                                   lets go even without a latched grip,
                                   sampled from stem_release_time_range
                                   (inf when None: release on latch only)
      _stem_released            -> False
    Also stores shaping_grace_s as env._shaping_grace_s (see
    shaping_active): seconds, or (start_s, end_s, start_step, end_step)
    for a training curriculum (_grace_schedule); None disables it.

    stem_release_schedule = (start_s, end_s, start_step, end_step, jitter_s)
    samples each episode's release time as _grace_schedule(...) +
    U(0, jitter_s) instead -- a curriculum on WHEN the stem lets go.
    release_on_latch=False makes the stem (and so the arm unlock, which
    follows it) ignore the grip latch entirely: see update_first_closure.
    """
    env._shaping_grace_s = shaping_grace_s
    env._stem_release_on_latch = release_on_latch
    apple: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=device, dtype=torch.long)

    if not hasattr(env, "_grasp_phase_active"):
        env._grasp_phase_active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._grasp_ever_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._reach_phase_hold_counter = torch.zeros(env.num_envs, device=device)
        env._grasp_handoff_apple_pos = torch.zeros(env.num_envs, 3, device=device)
        env._grasp_handoff_apple_quat = torch.zeros(env.num_envs, 4, device=device)
        env._grasp_handoff_joint_pos = robot.data.joint_pos.clone()
        env._grasp_handoff_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._stem_release_time = torch.full((env.num_envs,), float("inf"), device=device)
        env._stem_released = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    env._grasp_phase_active[ids] = False
    env._stem_released[ids] = False
    if stem_release_schedule is not None:
        s0, s1, t0, t1, jitter = stem_release_schedule
        base = _grace_schedule(env, s0, s1, t0, t1)
        env._stem_release_time[ids] = base + torch.empty(len(ids), device=device).uniform_(0.0, jitter)
    elif stem_release_time_range is not None:
        lo, hi = stem_release_time_range
        env._stem_release_time[ids] = torch.empty(len(ids), device=device).uniform_(lo, hi)
    env._grasp_ever_closed[ids] = False
    env._reach_phase_hold_counter[ids] = 0.0
    # write_root_pose_to_sim updates the data buffers immediately, so this
    # reads the pose randomize_tree_and_apple just wrote.
    env._grasp_handoff_apple_pos[ids] = apple.data.root_pos_w[ids].clone()
    env._grasp_handoff_apple_quat[ids] = apple.data.root_quat_w[ids].clone()
    env._grasp_handoff_joint_pos[ids] = robot.data.joint_pos[ids].clone()


def update_reach_grasp_phase(
    env: "ManagerBasedRLEnv",
    env_ids,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    handoff_mode: str = "fixed",
    handoff_time_s: float = 2.5,
    standoff: float = 0.015,
    palm_normal_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pos_tol: float = 0.04,
    tilt_tol_rad: float | None = None,
    stem_tol_rad: float | None = None,
    hold_time: float = 0.5,
) -> None:
    """Every-step interval event: switch envs from the reach phase to the
    grasp phase.

    handoff_mode:
      "fixed"     -- after exactly handoff_time_s of reach-policy control,
                     whatever state it reached. Doesn't depend on the reach
                     policy settling into a stable hold, which it was never
                     trained to do past reach_success.
      "converged" -- as soon as the palm has stayed inside the reach-success
                     neighbourhood (pos_tol/tilt_tol_rad, same test as the
                     reach task's reach_success) for hold_time, with
                     handoff_time_s as a watchdog.

    Runs after the step's physics/terminations/rewards/resets and before
    the observation, so the observation the trainee acts on next already
    says "grasp phase". A just-reset env has episode_length_buf == 0 and so
    never switches here.
    """
    if not hasattr(env, "_grasp_phase_active"):
        return
    reaching = ~env._grasp_phase_active
    if not reaching.any():
        return

    elapsed = env.episode_length_buf.float() * env.step_dt
    switch = reaching & (elapsed >= handoff_time_s - 1e-6)

    if handoff_mode == "converged":
        from .reach_rewards import reach_neighbourhood

        inside = reach_neighbourhood(
            env, robot_cfg, object_cfg, pos_tol, tilt_tol_rad, standoff,
            palm_normal_axis, palm_local_offset, stem_tol_rad,
        )
        env._reach_phase_hold_counter.copy_(torch.where(
            inside & reaching, env._reach_phase_hold_counter + 1, torch.zeros_like(env._reach_phase_hold_counter)
        ))
        switch = switch | (reaching & (env._reach_phase_hold_counter >= max(1, round(hold_time / env.step_dt))))
    elif handoff_mode != "fixed":
        raise ValueError(f"handoff_mode must be 'fixed' or 'converged', got {handoff_mode!r}")

    if not switch.any():
        return
    ids = switch.nonzero(as_tuple=True)[0]
    robot: Articulation = env.scene[robot_cfg.name]
    env._grasp_handoff_joint_pos[ids] = robot.data.joint_pos[ids].clone()
    env._grasp_handoff_step[ids] = env.episode_length_buf[ids]
    env._grasp_phase_active[ids] = True


def update_first_closure(
    env: "ManagerBasedRLEnv",
    env_ids,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.5,
    min_steps: int = 5,
) -> None:
    """Every-step interval event: latch env._grasp_ever_closed the first time
    the TRAINEE achieves force closure this episode. Until then the apple
    stays pinned ("still on the branch", see ReachHandoffJointPositionAction.
    apply_actions), so the policy can establish a grip with no gravity/time
    pressure; from then on gravity is real and apple_fell_from_grip /
    grasp_success test whether the grip actually holds.

    Closure must hold for min_steps consecutive control steps before the
    apple is released, so a one-step contact blip while the fingers are
    still moving can't drop it into a grip that doesn't exist yet.

    Also updates env._stem_released: the latch above, OR this episode's
    randomized release time after hand-off having passed. Releasing ONLY
    on the latch let the policy keep the stem holding the fruit forever by
    never letting its grip be detected -- and a trained policy did exactly
    that: it held index_1 at its open limit (commanded 0.9 rad past it) and
    kept the apple off the palm, the two cheapest ways to keep the
    detector's contact groups below three, because every latch it did allow
    released the stem into a grip that then failed (apple_fell_from_grip
    in 10-19% of training episodes) and cost it the rest of the episode.
    With a deadline, not being detected no longer keeps the apple up; being
    detected still unlocks the arm for the lift (unlock_with_grip).
    """
    if not hasattr(env, "_grasp_ever_closed"):
        return
    from .grasp_rewards import _force_closure_ok

    closure_now = _force_closure_ok(env, sensor_cfg, contact_threshold=contact_threshold)
    if not hasattr(env, "_closure_streak"):
        env._closure_streak = torch.zeros(env.num_envs, device=env.device)
    env._closure_streak.copy_(torch.where(
        closure_now & env._grasp_phase_active, env._closure_streak + 1, torch.zeros_like(env._closure_streak)
    ))
    env._grasp_ever_closed |= env._closure_streak >= min_steps

    if hasattr(env, "_stem_released"):
        since_handoff = (env.episode_length_buf - env._grasp_handoff_step).float() * env.step_dt
        deadline = env._grasp_phase_active & (since_handoff >= env._stem_release_time)
        # With release_on_latch=False the latch changes NOTHING the policy
        # could fear -- stem, arm unlock and the observed flag all follow
        # the deadline alone -- so being detected is pure bonus
        # (grasp_hold). Every variant that released on the latch taught a
        # policy to dodge detection by withholding one contact group: first
        # index_1, then (with the deadline removed) the thumb, commanded
        # ~0.6 rad past its open limit so the palm + index + middle, which
        # all push from one side, could never reach closure.
        latch = env._grasp_ever_closed if getattr(env, "_stem_release_on_latch", True) else False
        env._stem_released |= latch | deadline


# ─────────────────────────────────────────────
#  Action term
# ─────────────────────────────────────────────

class ReachHandoffJointPositionAction(RateLimitedJointPositionAction):
    """RateLimitedJointPositionAction that is aware of the reach/grasp phase.

    Reach phase: the trainee's actions for this term are ignored. If
    cfg.drive_with_reach_policy, the frozen reach policy's action is used
    instead (arm); otherwise the joints are held at their default pose
    (hand -- the reach policy was trained with the hand never commanded,
    i.e. sitting at its default open pose).

    Grasp phase: the trainee drives. With cfg.anchor_to_handoff, its action
    is an offset from the rate-limited target this term was commanding at
    the moment of hand-off, instead of from the default pose -- otherwise a
    zero action (where a fresh policy starts) would pull the arm straight
    back toward its home pose, away from the apple.

    Both phases go through the same velocity/acceleration limiter (same
    math as the parent class), and the limiter state carries over the
    switch, so the hand-off itself is smooth.

    With cfg.pin_object set, this term also re-pins that object at
    env._grasp_handoff_apple_pos every PHYSICS substep, for envs in the
    reach phase or not yet in force closure. It lives here only because
    action terms are the one per-substep hook ManagerBasedRLEnv offers;
    once per control step would let gravity and palm contact move the apple
    between pins, unlike the kinematic apple the reach policy trained on.
    """

    cfg: "ReachHandoffJointPositionActionCfg"

    def __init__(self, cfg: "ReachHandoffJointPositionActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        n, j = self._processed_actions.shape
        self._anchor = torch.zeros(n, j, device=self.device)
        self._anchored = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._reach_prev_raw = torch.zeros(n, j, device=self.device)
        self._grasp_offset = self._offset.clone() if isinstance(self._offset, torch.Tensor) \
            else torch.full((n, j), float(self._offset), device=self.device)
        for name, value in (cfg.grasp_offset or {}).items():
            self._grasp_offset[:, self._joint_names.index(name)] = value
        # reach-phase target when no reach policy drives this term (the hand):
        # the default pose, or cfg.reach_offset (e.g. the reach's pre-shape)
        self._reach_hold = self._offset.clone() if isinstance(self._offset, torch.Tensor) \
            else torch.full((n, j), float(self._offset), device=self.device)
        for name, value in (cfg.reach_offset or {}).items():
            self._reach_hold[:, self._joint_names.index(name)] = value
        # reach-phase hand schedule (open -> closed by env._hand_phase)
        self._sched = None
        if cfg.reach_hold_schedule is not None:
            open_t, closed_t = cfg.reach_hold_schedule
            o, c = self._reach_hold[0].clone(), self._reach_hold[0].clone()
            for name, value in open_t.items():
                o[self._joint_names.index(name)] = value
            for name, value in closed_t.items():
                c[self._joint_names.index(name)] = value
            self._sched = (o, c)

        # Waist yaw driven by the reach policy's extra (8th) output during
        # the reach, then held where the reach left it (braking to a stop);
        # not part of this term's actions (the trainee never drives it).
        self._waist_id = None
        if cfg.reach_waist_joint is not None:
            robot = env.scene[cfg.asset_name]
            self._waist_id = robot.joint_names.index(cfg.reach_waist_joint)
            self._waist_cfg = SceneEntityCfg(cfg.asset_name, joint_names=[cfg.reach_waist_joint], preserve_order=True)
            self._waist_cfg.resolve(env.scene)
            self._waist_prev_raw = torch.zeros(n, 1, device=self.device)
            self._waist_target = torch.zeros(n, device=self.device)
            self._waist_vel = torch.zeros(n, device=self.device)
            # external waist command (rad) for the harvest bench after the
            # grasp (the arm-goal policy's waist); NaN = hold / reach policy
            self.waist_override = torch.full((n,), float("nan"), device=self.device)

        if cfg.drive_with_reach_policy:
            # Resolved WITHOUT preserve_order, matching the reach task's own
            # observation terms and this term's own joint ids.
            self._reach_arm_cfg = SceneEntityCfg("robot", joint_names=cfg.joint_names)
            self._reach_arm_cfg.resolve(env.scene)
            self._palm_cfg = SceneEntityCfg("robot", body_names=[cfg.palm_body_name])
            self._palm_cfg.resolve(env.scene)
            self._apple_cfg = SceneEntityCfg(cfg.object_name)
            self._apple_cfg.resolve(env.scene)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        phase = grasp_phase_active(self._env).unsqueeze(-1)
        # Proximal arm unlocks when the STEM lets go (grip latched, or the
        # release deadline), not on the latch alone. Unlocking only on the
        # latch made being detected the one thing that exposed the grasp to
        # arm exploration noise -- which shook latched apples loose in
        # 10-19% of training episodes -- so the policy learned to keep its
        # grip undetected (index_1 held open, apple kept off the palm) and
        # the arm frozen. Now both paths reach the same free arm.
        grip = stem_released(self._env)

        # -- grasp-phase (trainee) target --
        offset = self._grasp_offset
        if self.cfg.anchor_to_handoff:
            newly = grasp_phase_active(self._env) & ~self._anchored
            if newly.any():
                self._anchor[newly] = self._prev_target[newly]
                self._anchored |= newly
            offset = torch.where(self._anchored.unsqueeze(-1), self._anchor, self._offset)
        scale = self._scale
        if self.cfg.unlock_with_grip:
            # PROXIMAL joints (all but the last cfg.pre_grip_free_dims of
            # this term's own joint list, e.g. shoulder+elbow) stay FROZEN
            # at the hand-off pose until the fingers have actually
            # established a grip this episode, then unlock (for the lift).
            # The arm's 87 Nm actuators overpower the stem by more than 5x,
            # so before a grip exists proximal motion just shoves the apple
            # out of the hand: measured, PPO-scale exploration noise on the
            # (then fully unlocked) arm dragged the apple 7 cm off its
            # anchor within 2.4 s at noise std 0.8, and 82 cm at std 1.5,
            # ending the episode with no contact reward -- which removed the
            # learning signal, which grew the noise further.
            #
            # DISTAL joints (the last pre_grip_free_dims, e.g. the 3 wrist
            # joints) stay free even pre-grip. Freezing them too was a
            # blunt fix that removed the ONE thing that can correct a badly
            # aligned hand-off before any finger ever touches the apple --
            # once fingers close, they can pull the apple around a little
            # via the compliant stem, but they cannot re-point the palm.
            # The wrist sits close to the palm, so a given joint angle
            # produces much less hand TRANSLATION (and so much less pull on
            # the stem) than the same angle at the shoulder -- verified
            # empirically before enabling this (see the cfg field's own
            # docstring for the measured numbers).
            n_dims = self.action_dim
            free = self.cfg.pre_grip_free_dims
            gate = torch.ones(self._env.num_envs, n_dims, device=self.device)
            if free < n_dims:
                gate[:, : n_dims - free] = grip.unsqueeze(-1).float()
            scale = scale * gate
        trainee_target = self._raw_actions * scale + offset

        # -- reach-phase target --
        waist_raw = None
        if self.cfg.drive_with_reach_policy and not bool(phase.all()):
            policy = _load_reach_policy(self.cfg.reach_policy_path, self.device)
            waist = {"ring_range": self.cfg.reach_ring_range} if self._waist_id is None else \
                {"joint_cfg": self._waist_cfg, "prev_raw": self._waist_prev_raw, "ring_range": self.cfg.reach_ring_range}
            obs = _reach_obs(
                self._env, self._reach_arm_cfg, self._palm_cfg, self._apple_cfg,
                self.cfg.palm_local_offset, self._reach_prev_raw, layout=self.cfg.reach_obs_layout, waist=waist,
            )
            with torch.inference_mode():
                reach_raw = policy(obs)
            # In-place: rollouts run under inference_mode, and a reassigned
            # inference tensor could not be reset in place outside it later.
            self._reach_prev_raw.copy_(reach_raw[:, : self._reach_prev_raw.shape[1]])
            if self._waist_id is not None:
                waist_raw = reach_raw[:, self._reach_prev_raw.shape[1]:]
                self._waist_prev_raw.copy_(torch.where(phase, self._waist_prev_raw, waist_raw))
            reach_target = self._reach_prev_raw * self.cfg.reach_scale + self._offset
        elif self._sched is not None:
            ph = getattr(self._env, "_hand_phase", None)
            ph = torch.zeros(self._env.num_envs, device=self.device) if ph is None else ph
            o, c = self._sched
            reach_target = o + ph.unsqueeze(-1) * (c - o)
        else:
            reach_target = self._reach_hold

        target_desired = torch.where(phase, trainee_target, reach_target)
        if self.cfg.clip is not None:
            target_desired = torch.clamp(
                target_desired, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # -- rate limiter: identical to RateLimitedJointPositionAction --
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        needs_reset = (self._env.episode_length_buf <= 1).unsqueeze(-1)
        self._prev_target = torch.where(needs_reset, current_pos, self._prev_target)
        self._prev_vel = torch.where(needs_reset, torch.zeros_like(self._prev_vel), self._prev_vel)

        dt = self._env.step_dt
        # caps: the trainee's in the grasp phase; the reach phase may have the
        # frozen reach's own (reach_max_velocity / _acceleration / _brake_to_target)
        c = self.cfg
        vcap = torch.where(phase, c.max_velocity,
                           c.max_velocity if c.reach_max_velocity is None else c.reach_max_velocity)
        acap = torch.where(phase, c.max_acceleration,
                           c.max_acceleration if c.reach_max_acceleration is None else c.reach_max_acceleration)
        vmax = vcap * self.speed_scale                              # speed factor (see the parent class)
        amax = acap * self.speed_scale ** 2
        v_desired = (target_desired - self._prev_target) / dt
        v_cmd = torch.maximum(torch.minimum(v_desired, vmax), -vmax)
        brake_reach = c.brake_to_target if c.reach_brake_to_target is None else c.reach_brake_to_target
        if c.brake_to_target or brake_reach:
            # as RateLimitedJointPositionAction: the fastest speed that still stops at the target
            err = (target_desired - self._prev_target).abs()
            v_stop = -0.5 * amax * dt + torch.sqrt((0.5 * amax * dt) ** 2 + 2.0 * amax * err)
            braked = torch.maximum(torch.minimum(v_cmd, v_stop), -v_stop)
            if c.brake_to_target and brake_reach:
                v_cmd = braked
            else:
                v_cmd = torch.where(~phase if brake_reach else phase, braked, v_cmd)
        max_dv = amax * dt
        v_cmd = torch.clamp(v_cmd, self._prev_vel - max_dv, self._prev_vel + max_dv)

        target_cmd = self._prev_target + v_cmd * dt

        self._prev_target = target_cmd
        self._prev_vel = v_cmd
        self._processed_actions = target_cmd

        if self._waist_id is not None:
            self._process_waist(phase.squeeze(-1), needs_reset.squeeze(-1), dt)

    def _process_waist(self, phase: torch.Tensor, needs_reset: torch.Tensor, dt: float) -> None:
        """Waist target: the reach policy's raw waist output * reach_scale
        (default 0), clipped to +-reach_waist_limit, in the reach phase; in
        the grasp phase the current target (stop). Rate-limited with braking
        toward the target, as the reach task's waist term (brake_to_target)."""
        cur = self._asset.data.joint_pos[:, self._waist_id]
        self._waist_target = torch.where(needs_reset, cur, self._waist_target)
        self._waist_vel = torch.where(needs_reset, torch.zeros_like(self._waist_vel), self._waist_vel)
        lim = self.cfg.reach_waist_limit
        desired = (self._waist_prev_raw.squeeze(-1) * self.cfg.reach_scale).clamp(-lim, lim)
        desired = torch.where(phase, self._waist_target, desired)
        ov = self.waist_override
        desired = torch.where(torch.isfinite(ov), ov.clamp(-lim, lim), desired)
        sp = self.speed_scale.squeeze(-1)
        a, vmax = self.cfg.reach_waist_max_acceleration * sp ** 2, self.cfg.reach_waist_max_velocity * sp
        err = desired - self._waist_target
        v = torch.maximum(torch.minimum(err / dt, vmax), -vmax)
        v_stop = -0.5 * a * dt + torch.sqrt((0.5 * a * dt) ** 2 + 2.0 * a * err.abs())
        v = torch.maximum(torch.minimum(v, v_stop), -v_stop)
        v = torch.clamp(v, self._waist_vel - a * dt, self._waist_vel + a * dt)
        self._waist_target = self._waist_target + v * dt
        self._waist_vel = v

    def apply_actions(self):
        super().apply_actions()
        if self._waist_id is not None:
            self._asset.set_joint_position_target(self._waist_target.unsqueeze(-1), joint_ids=[self._waist_id])
        if self.cfg.pin_object is None or not hasattr(self._env, "_grasp_ever_closed"):
            return
        obj: RigidObject = self._env.scene[self.cfg.pin_object]
        phase = self._env._grasp_phase_active
        anchor = self._env._grasp_handoff_apple_pos

        # REACH phase: rigid pin (teleport). The reach policy was trained
        # against a kinematic, immovable apple, so this reproduces exactly
        # the contact it learned on.
        hard = ~phase
        if hard.any():
            ids = hard.nonzero(as_tuple=True)[0]
            pose = torch.cat([anchor[ids], self._env._grasp_handoff_apple_quat[ids]], dim=-1)
            obj.write_root_pose_to_sim(pose, env_ids=ids)
            obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

        # GRASP phase, before the first closure: a COMPLIANT STEM instead of
        # the rigid pin -- a spring-damper pulling the apple back to the
        # anchor, force-clamped. The rigid pin made the apple infinitely
        # stiff AND infinitely heavy: closing fingers could not nudge or
        # centre it the way they would a real fruit, so contacts landed on
        # whatever single spot the sphere happened to occupy (measured: 3 of
        # the 4 contact groups almost never engaged at once, with force
        # spikes into the hundreds and once ~3800 N). Simply releasing it
        # instead is no good either -- the hand-off leaves the palm tilted
        # ~26 deg, so an unheld sphere rolls straight off before the fingers
        # can close (measured: contacts drop to ~0.5 groups, nothing held).
        # The stem is what the apple actually hangs from, and the scene
        # already carries APPLE_STEM_BREAK_FORCE/TORQUE for the harvest
        # phase, so this is also the physically honest version.
        soft = phase & ~stem_released(self._env)
        d = anchor - obj.data.root_pos_w
        f = self.cfg.stem_stiffness * d - self.cfg.stem_damping * obj.data.root_lin_vel_w
        f_mag = f.norm(dim=-1, keepdim=True)
        f = f * (self.cfg.stem_max_force / f_mag.clamp(min=self.cfg.stem_max_force))
        # Zero for every env past first closure, so the wrench buffer never
        # keeps pulling on an apple the policy has already taken.
        f = torch.where(soft.unsqueeze(-1), f, torch.zeros_like(f))
        obj.permanent_wrench_composer.set_forces_and_torques(
            forces=f.unsqueeze(1), torques=torch.zeros_like(f).unsqueeze(1), is_global=True
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._anchored[env_ids] = False
        # Zero on the first tick, mirroring mdp.last_action's zero-at-
        # episode-start convention in the reach task.
        self._reach_prev_raw[env_ids] = 0.0
        if self._waist_id is not None:
            self._waist_prev_raw[env_ids] = 0.0
            self._waist_vel[env_ids] = 0.0


@configclass
class ReachHandoffJointPositionActionCfg(RateLimitedJointPositionActionCfg):
    """Config for ReachHandoffJointPositionAction.

    When drive_with_reach_policy is set, scale/clip/max_velocity/
    max_acceleration apply to the reach policy's output too, so they MUST
    match the reach task's own ActionsCfg.right_arm (apple_grasp_env_cfg.py)
    -- except `scale`, which is the TRAINEE's scale; the reach policy's is
    reach_scale.
    """

    class_type: type = ReachHandoffJointPositionAction

    drive_with_reach_policy: bool = False
    anchor_to_handoff: bool = False
    reach_policy_path: str = PINNED_REACH_POLICY
    reach_obs_layout: str = "v1"
    """"v1": the 29-dim AppleReach-v1 observation; "tree": the 37-dim
    AppleReachTree-v1 one (any-contact palm force + clearance ring) -- must
    match the policy at reach_policy_path."""
    reach_scale: float = 0.5
    reach_max_velocity: float | None = None
    reach_max_acceleration: float | None = None
    reach_brake_to_target: bool | None = None
    """The rate limiter's caps / braking during the REACH phase only (None: the
    trainee's max_velocity / max_acceleration / brake_to_target, as before):
    set them to the frozen reach's own training values."""
    reach_waist_joint: str | None = None
    """Waist yaw joint driven by the reach policy's extra output (layout
    "waist"), held after the hand-off; None: the waist is not touched."""
    reach_waist_limit: float = 0.785
    reach_waist_max_velocity: float = 1.0
    reach_waist_max_acceleration: float = 4.0
    reach_ring_range: float = 0.10
    """Clearance-ring range of the reach policy's input (layout "waist")."""
    reach_offset: dict[str, float] | None = None
    """Joint name -> target held during the REACH phase when no reach policy
    drives this term (the hand: the reach's pre-shape); default pose otherwise."""
    reach_hold_schedule: tuple[dict, dict] | None = None
    """(open, closed) joint name -> target dicts: instead of reach_offset, the
    reach-phase target goes open -> closed with env._hand_phase in [0, 1]
    (single_reach.step_hand_schedule, write_targets=False)."""
    # With unlock_with_grip, the LAST pre_grip_free_dims joints (in
    # cfg.joint_names order) keep their normal scale even before the first
    # grip; the rest are held at exactly the hand-off pose until then. For
    # right_arm's 7 joints (shoulder x3, elbow, wrist x3 -- see
    # RIGHT_ARM_JOINTS), 3 means the wrist. Measured (16 envs, wrist-only
    # N(0, std) exploration noise, shoulder+elbow held exactly at hand-off,
    # displacement of the apple from its stem anchor): std 0.8 stayed at
    # ~0.65 cm at 2.4 s (0/16 episodes lost the apple) -- barely above the
    # ~0.5 cm baseline sag with NO action at all. std 1.5 reached ~10.9 cm
    # by 2.4 s (3/16 lost) -- real risk, but far short of the WHOLE-ARM
    # number at the same std in process_actions's own comment (82 cm,
    # 9/16). PPO's init_noise_std here is 0.8 (see
    # AppleGraspGraspPPORunnerCfg), so the safe case is the common one; if
    # Policy/mean_noise_std grows past ~1.2 during training the way it has
    # before, watch apple_fell_from_grip for a repeat of the same collapse
    # and consider a smaller, wrist-specific scale rather than reverting to
    # a full freeze.
    pre_grip_free_dims: int = 0
    palm_body_name: str = "right_hand_palm_link"
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    object_name: str = "apple"
    pin_object: str | None = None
    unlock_with_grip: bool = False
    grasp_offset: dict[str, float] | None = None
    """Joint name -> target (rad) that a ZERO trainee action means in the
    grasp phase, replacing the default-pose offset for those joints (the
    reach phase keeps the default pose). With the default offset every
    Dex3 flexion joint sits at its OPEN limit at zero action, so exploration
    noise around zero spends half its time in a dead zone past the limit;
    for the thumb, the ~0.9 rad from there to the fruit was never crossed
    in four training runs even when contact was worth ~2x the reward.
    A pre-grasp offset centres exploration where the decision is."""
    # Compliant-stem spring-damper (grasp phase, before first closure).
    # Softened from 200 N/m / 15 N: that resisted the fingers repositioning
    # the apple nearly as stiffly as the old rigid pin, and 15 N is far
    # more than a real stem holds (APPLE_STEM_BREAK_FORCE is ~2.5 N). At
    # 100 N/m the 0.15 kg apple sags ~1.5 cm unsupported (it is resting on
    # the palm here, so in practice much less), omega*dt ~ 0.13 per physics
    # substep, damping ~0.5 critical.
    stem_stiffness: float = 100.0
    stem_damping: float = 4.0
    stem_max_force: float = 6.0
