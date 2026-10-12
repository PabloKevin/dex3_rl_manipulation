# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Termination conditions and reset/randomisation events."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import reset_root_state_uniform  # built-in Isaac Lab

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ─────────────────────────────────────────────
#  Terminations
# ─────────────────────────────────────────────

def apple_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height: float = -0.1,
) -> torch.Tensor:
    """True if apple falls below min_height (dropped / out of reach).

    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    return apple.data.root_pos_w[:, 2] < min_height


def apple_lifted_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_height: float,
    hold_time: float = 1.0,
) -> torch.Tensor:
    """True when apple has been held above target_height for hold_time seconds.

    Uses env.episode_length_buf to measure elapsed time.
    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    above = apple.data.root_pos_w[:, 2] >= target_height  # [N] bool

    # How many steps is hold_time?
    hold_steps = int(hold_time / (env.cfg.sim.dt * env.cfg.decimation))

    # We need a persistent counter — store on env if not present
    if not hasattr(env, "_hold_counter"):
        env._hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    env._hold_counter[above] += 1
    env._hold_counter[~above] = 0

    return env._hold_counter >= hold_steps


def reach_success(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    pos_tol: float = 0.04,
    tilt_tol_rad: float | None = None,
    stem_tol_rad: float | None = None,
    standoff: float = 0.015,
    palm_normal_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    hold_time: float = 0.5,
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sensor_cfg: SceneEntityCfg | None = None,
    pressure_min: float = 0.5,
    pressure_max: float | None = 5.0,
) -> torch.Tensor:
    """Terminate once the palm has held within [pos_tol, tilt_tol_rad] of the
    grasp point for `hold_time` continuous seconds — the same neighbourhood
    check as reach_rewards.reward_reach_hold's per-step bonus, counted here
    toward a one-time terminal success instead. Mirrors g1_arm_env's
    hold-to-succeed / R_success design (config_right.yaml: hold.time=0.5,
    R_success=700).

    sensor_cfg (optional, default None = old position/orientation-only
    behaviour, fully backward compatible): when given, ALSO requires palm
    contact force in (pressure_min, pressure_max] -- "reach" now means
    "arrive AND gently touch", not just arrive. MUST be passed with the
    same pressure_min/pressure_max as the paired reward_reach_hold call in
    RewardsCfg (and this function's own params must match reward_reach_
    success's expectations, same as before), or the two disagree about
    what counts as success.

    IsaacLab computes terminations BEFORE rewards each step (see
    ManagerBasedRLEnv.step: termination_manager.compute() then
    reward_manager.compute()), so this caches the result on env._reach_success
    for reach_rewards.reward_reach_success to read the same step it fires —
    that reward function must NOT recompute the check itself, or a policy
    could see the bonus without this termination also firing (or vice versa).

    Shape: [num_envs]  bool
    """
    from .reach_rewards import reach_neighbourhood

    in_neighbourhood = reach_neighbourhood(
        env, robot_cfg, object_cfg, pos_tol, tilt_tol_rad, standoff,
        palm_normal_axis, palm_local_offset, stem_tol_rad,
    )

    if sensor_cfg is not None:
        from .observations import palm_contact_force
        force = palm_contact_force(env, sensor_cfg).squeeze(-1)
        touching = force > pressure_min
        if pressure_max is not None:
            touching = touching & (force <= pressure_max)
        in_neighbourhood = in_neighbourhood & touching

    if not hasattr(env, "_reach_hold_counter") or env._reach_hold_counter.shape != in_neighbourhood.shape:
        env._reach_hold_counter = torch.zeros(in_neighbourhood.shape, device=env.device)

    # Episode boundary: don't carry a stale count into a fresh episode.
    env._reach_hold_counter = torch.where(
        env.episode_length_buf <= 1, torch.zeros_like(env._reach_hold_counter), env._reach_hold_counter
    )
    env._reach_hold_counter = torch.where(
        in_neighbourhood, env._reach_hold_counter + 1, torch.zeros_like(env._reach_hold_counter)
    )

    hold_steps_required = max(1, round(hold_time / env.step_dt))
    success = env._reach_hold_counter >= hold_steps_required

    env._reach_success = success   # read by reach_rewards.reward_reach_success

    # env._suppress_reach_success_termination is never set during training --
    # this getattr is a no-op there, success terminates/auto-resets exactly
    # as before. run_env.py sets it so it can own reset timing itself
    # (manual 'r', or its own delayed auto-reset) instead of
    # ManagerBasedRLEnv.step() instantly resetting the moment this fires --
    # env._reach_success above is still computed/cached either way, so
    # anything watching it (the reward bonus, run_env.py itself) still sees
    # success the instant it happens even when the termination is suppressed.
    if getattr(env, "_suppress_reach_success_termination", False):
        return torch.zeros_like(success)
    return success


def apple_fell_from_grip(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    drop_margin: float = 0.08,
    robot_cfg: SceneEntityCfg | None = None,
    center_local: tuple[float, float, float] | None = None,
    leave_margin: float = 0.06,
) -> torch.Tensor:
    """True once the apple has fallen `drop_margin` metres below where the
    reach->grasp hand-off left it (env._grasp_handoff_apple_pos) -- the real
    "failed to hold it" signal for the grasp phase.

    Why this exists alongside apple_dropped: apple_dropped's own
    min_height=-0.1 (an ABSOLUTE world-frame height) was calibrated for the
    reach phase, where the apple was kinematic and never touched anything.
    Now that the apple is a real dynamic body, a failed grasp just lets it
    fall and REST ON THE GROUND at roughly its own radius (~0.04 m) --
    comfortably above -0.1 m -- so apple_dropped alone never fires on a
    failed grasp; the episode idles out to time_out with no further
    learning signal and no real penalty for having failed. (Confirmed
    empirically during early training: 100% time_out, 0% apple_dropped,
    episode length pinned at the max every single episode.)

    This measures against the apple's OWN per-episode hand-off height
    instead (mirrors how grasp_success's lift check is anchored to the same
    tensor, for the same reason -- an absolute reference doesn't account
    for per-episode randomized apple height), so it catches a failed or
    slipping grasp promptly regardless of where the apple ends up (on the
    ground, rolled aside, wherever) rather than only once it crosses some
    fixed absolute coordinate. apple_dropped is kept alongside this as a
    coarse backstop (e.g. the apple somehow ending up far below the scene
    entirely), not removed.

    Shape: [num_envs] bool
    """
    from .grasp_autoplay import grasp_phase_active

    apple: RigidObject = env.scene[object_cfg.name]
    if robot_cfg is not None and center_local is not None:
        # RELATIVE TO THE HAND: the fruit centre more than leave_margin
        # from the palm's grasp point (a fixed point in the palm frame).
        # The absolute-height version above counted a fruit held firmly in
        # a hand that moved down as "fell from grip" -- which is exactly
        # what happened once the stem let go and the arm unlocked: the
        # fruit stayed within a millimetre of its place in the hand while
        # the arm carried it 8 cm down.
        from isaaclab.utils.math import quat_apply
        robot = env.scene[robot_cfg.name]
        body = robot_cfg.body_ids[0]
        quat = robot.data.body_quat_w[:, body]
        c = torch.tensor(center_local, device=env.device, dtype=quat.dtype).expand(env.num_envs, 3)
        grasp_point = robot.data.body_pos_w[:, body] + quat_apply(quat, c)
        fell = torch.norm(apple.data.root_pos_w - grasp_point, dim=-1) > leave_margin
        return fell & grasp_phase_active(env)
    if not hasattr(env, "_grasp_handoff_apple_pos"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fell = apple.data.root_pos_w[:, 2] < (env._grasp_handoff_apple_pos[:, 2] - drop_margin)
    return fell & grasp_phase_active(env)


def grasp_success(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    hold_time: float = 0.5,
    lift_margin: float = 0.03,
    contact_threshold: float = 0.5,
    terminate: bool = True,
) -> torch.Tensor:
    """Terminate once the hand has enclosed the apple (see
    grasp_rewards._force_closure_ok for the proxy used) and that has held
    for hold_time continuous seconds AND the apple has been lifted
    lift_margin above its post-hand-off height. Mirrors reach_success's
    hold-counter idiom exactly (same episode-boundary reset via
    episode_length_buf, same "cache on env, read by the paired reward"
    contract as reward_reach_success / reach_success).

    The lift check matters even though force closure alone might look like
    "grasped": with a 14-DoF action space the arm has enough authority to
    make a degenerate "palm-as-shelf" solution look stable without the
    fingers actually gripping. Anchored to env._grasp_handoff_apple_pos
    (the apple's pinned pose, recorded by grasp_autoplay.
    reset_reach_grasp_phase) rather than a fixed absolute height, for the same reason
    rewards.reward_lift anchors to env._stem_anchor_pos — the apple's height
    at that moment varies per episode.

    IsaacLab computes terminations BEFORE rewards each step (see
    ManagerBasedRLEnv.step), so this caches env._force_closure_ok (read by
    grasp_rewards.reward_grasp_hold) and env._grasp_success (read by
    grasp_rewards.reward_grasp_success) fresh every step before either
    reward reads them.

    Only counts in the grasp phase (grasp_autoplay.grasp_phase_active) --
    nothing the frozen reach policy does can succeed or fail the episode.

    Shape: [num_envs] bool
    """
    from .grasp_rewards import _force_closure_ok
    from .grasp_autoplay import grasp_phase_active

    apple: RigidObject = env.scene[object_cfg.name]
    closure_ok = _force_closure_ok(env, sensor_cfg, contact_threshold=contact_threshold)
    env._force_closure_ok = closure_ok

    if hasattr(env, "_grasp_handoff_apple_pos"):
        lifted = apple.data.root_pos_w[:, 2] >= (env._grasp_handoff_apple_pos[:, 2] + lift_margin)
    else:
        # Autoplay hasn't run yet (e.g. this task's env got stepped once
        # before any reset) — fail closed rather than crash.
        lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    in_neighbourhood = closure_ok & lifted & grasp_phase_active(env)

    if not hasattr(env, "_grasp_hold_counter") or env._grasp_hold_counter.shape[0] != env.num_envs:
        env._grasp_hold_counter = torch.zeros(env.num_envs, device=env.device)

    env._grasp_hold_counter = torch.where(
        env.episode_length_buf <= 1, torch.zeros_like(env._grasp_hold_counter), env._grasp_hold_counter
    )
    env._grasp_hold_counter = torch.where(
        in_neighbourhood, env._grasp_hold_counter + 1, torch.zeros_like(env._grasp_hold_counter)
    )

    hold_steps_required = max(1, round(hold_time / env.step_dt))
    success = env._grasp_hold_counter >= hold_steps_required
    env._grasp_success = success   # read by grasp_rewards.reward_grasp_success
    if not terminate:
        # Checked every step, never ends the episode: env._grasp_success
        # then stays True for as long as the lifted grasp keeps holding, so
        # reward_grasp_success pays PER STEP for holding it. Ending the
        # episode on success cost the policy all the reward it would have
        # collected by continuing (~0.22/step of shaping terms, worth ~20
        # over the grasp window at gamma = 0.99) against a one-off bonus
        # of 95 * dt = 1.9 -- i.e. it taught the policy NOT to succeed.
        # Holding the fruit to the end of the episode is also closer to
        # what harvesting needs than "held it for 0.5 s once".
        return torch.zeros_like(success)
    return success


def fruit_damaged(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Hard, immediate failure: the apple force on any single hand link, on
    any face of it (grasp_rewards._max_link_apple_force -- palm, all thumb
    segments incl. the pad-less thumb_0, index, middle), exceeds the
    per-episode randomized env._fruit_damage_force (Li et al., 2026 +
    broader pome-fruit literature range — see
    domain_randomization.randomize_fruit_damage_force). Privileged on
    purpose: bruising does not care whether the contact landed on a taxel.

    Unlike grasp_success, this is NOT hold-counted — bio-yield is a
    near-instantaneous threshold effect (cellular rupture), not something to
    average or require sustained. A single crossing fails the episode, on
    par with apple_dropped as a first-class failure mode.

    Shape: [num_envs] bool
    """
    from .grasp_rewards import _max_link_apple_force
    from .grasp_autoplay import grasp_phase_active

    if not hasattr(env, "_fruit_damage_force"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    max_force = _max_link_apple_force(env, sensor_cfg)
    # Grasp phase only: a hard reach-policy arrival isn't the trainee's doing.
    return (max_force > env._fruit_damage_force) & grasp_phase_active(env)


def apple_placed_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float],
    tolerance: float = 0.06,
) -> torch.Tensor:
    """True when apple is within tolerance metres of the target (basket).

    Shape: [num_envs]  bool
    """
    apple: RigidObject = env.scene[object_cfg.name]
    goal = torch.tensor(target_pos, device=env.device).unsqueeze(0)
    dist = torch.norm(apple.data.root_pos_w - goal, dim=-1)
    return dist < tolerance


# ─────────────────────────────────────────────
#  Events (reset / randomisation)
# ─────────────────────────────────────────────

def reset_apple_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    base_pos: tuple[float, float, float],
    pos_range_x: tuple[float, float] = (-0.05, 0.05),
    pos_range_y: tuple[float, float] = (-0.05, 0.05),
    pos_range_z: tuple[float, float] = (-0.02, 0.02),
) -> None:
    """Reset apple to base_pos + uniform noise within the given ranges.

    Called at the start of each episode for the specified env_ids.
    """
    apple: RigidObject = env.scene[object_cfg.name]
    n = len(env_ids)

    # Sample position noise
    dx = torch.empty(n, device=env.device).uniform_(*pos_range_x)
    dy = torch.empty(n, device=env.device).uniform_(*pos_range_y)
    dz = torch.empty(n, device=env.device).uniform_(*pos_range_z)

    # Build new root state [n, 13]: pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    root_state = apple.data.default_root_state[env_ids].clone()  # [n, 13]

    # Override position (add env origins so position is in world frame)
    env_origins = env.scene.env_origins[env_ids]   # [n, 3]
    base = torch.tensor(base_pos, device=env.device).unsqueeze(0)  # [1, 3]
    noise = torch.stack([dx, dy, dz], dim=-1)       # [n, 3]

    root_state[:, :3] = env_origins + base + noise

    # Zero velocity
    root_state[:, 7:] = 0.0

    # Write back
    apple.write_root_state_to_sim(root_state, env_ids=env_ids)


def hold_default_joint_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset event: set every joint's PD position target to its default.

    Isaac Lab initialises joint_pos_target to ZERO, and only joints driven by
    an action term ever get a new target. Without this, every other joint is
    pulled from its configured default to 0 as soon as the episode starts --
    in the reach task that left the thumb sticking straight out of the palm
    (thumb_1 0.6 -> 0), the waist upright (0.1 -> 0) and the left arm and
    legs drooping. Action terms still overwrite their own joints every step.
    Run it after the joint-state reset.
    """
    robot = env.scene[asset_cfg.name]
    robot.set_joint_position_target(robot.data.default_joint_pos[env_ids], env_ids=env_ids)


def hold_joint_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset event: PD position targets of the named joints (not driven by
    any action term) to fixed values -- e.g. the reach's hand pre-shape.
    Run it after hold_default_joint_targets. The joints move there under
    their own PD; the joint state is not teleported."""
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_held_target_ids"):
        env._held_target_ids = torch.tensor([robot.joint_names.index(j) for j in joint_targets], device=env.device)
        env._held_target_vals = torch.tensor(list(joint_targets.values()), device=env.device)
    tgt = robot.data.joint_pos_target[env_ids].clone()
    tgt[:, env._held_target_ids] = env._held_target_vals
    robot.set_joint_position_target(tgt, env_ids=env_ids)
