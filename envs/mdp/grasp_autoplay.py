# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Reach-policy autoplay — the reach->grasp hand-off, implemented as a single
mode="reset" EventTerm.

Why an event term and not a subclassed environment: Isaac Lab already calls
mode="reset" event terms with EXACTLY the env_ids that just terminated,
during ManagerBasedRLEnv._reset_idx, BEFORE action_manager.reset()/
observation_manager.reset() fire for those envs and before episode_length_buf
is zeroed for them (confirmed by reading manager_based_rl_env.py directly:
_reset_idx's order is curriculum -> scene.reset -> event_manager.apply(mode=
"reset") -> observation/action/reward/... manager .reset() -> episode_length_
buf[env_ids] = 0). This means the grasp trainee (whichever PPO policy rsl_rl
is training) is NEVER asked to act during autoplay -- rsl_rl only ever sees
the environment AFTER this function returns, so there is no way for it to
record "trainee proposed action A, but action B was actually applied",
which would corrupt PPO's on-policy assumptions. No library-level surgery,
no reentrant env.step() call, no custom rollout loop.

Physics still advances for the WHOLE batched scene while this runs (PhysX
steps every env together, always) -- autoplay substeps for the resetting
envs also advance physics time for every other, currently-training env.
That's harmless to their trajectories (their commanded joint-position
targets are simply left untouched, so their PD controllers just keep
tracking whatever was already commanded), but it IS extra wall-clock cost.
Budget for it: keep watchdog_s tight and episode_length_s longer than the
reach phase's own value (see the grasp env cfg), and measure actual
iterations/sec before assuming this is fine at your num_envs.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import math
import os

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_reach_policy_cache: dict[str, torch.jit.ScriptModule] = {}

# Same hardcoded-with-env-override convention as scripts/run_env.py's
# BASE_DIR / G1_DEX3_USD_PATH.
_BASE_DIR = os.environ.get("DEX3_RL_BASE_DIR", "/home/pablo_kevin/dex3_rl_manipulation")


def _find_last_reach_policy(experiment_name: str = "apple_grasp_reach") -> str:
    """Path to the latest reach-training run's JIT-exported policy.pt --
    same convention (and same "only the exported file works with
    torch.jit.load, not the raw model_*.pt checkpoints" caveat) as
    scripts/run_env.py's _find_last_policy.

    Resolved LAZILY (only ever called from _load_reach_policy, which only
    runs when reach_autoplay_handoff actually fires at env reset) rather
    than as a module-level or class-level default -- importing this module,
    and therefore envs/__init__.py registering every task including the
    reach task itself, must not require a reach policy to already exist on
    disk.
    """
    base_dir = os.path.join(_BASE_DIR, "logs", "rsl_rl", experiment_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"No reach-training runs found at {base_dir}. Grasp-phase training "
            f"depends on an already-trained, exported reach policy for the "
            f"reach-autoplay hand-off (see grasp_autoplay.reach_autoplay_handoff) "
            f"-- train one first: scripts/train.py --task AppleGrasp-v0, then "
            f"scripts/play.py --task AppleGrasp-Play-v0 to export policy.pt."
        )
    latest_folder = max(os.listdir(base_dir))
    exported_path = os.path.join(base_dir, latest_folder, "exported", "policy.pt")
    if not os.path.isfile(exported_path):
        raise FileNotFoundError(
            f"No exported policy at {exported_path}. Run play.py against this "
            f"checkpoint first (it exports policy.pt as a side effect)."
        )
    return exported_path


def _load_reach_policy(policy_path: str, device) -> torch.jit.ScriptModule:
    """Lazily torch.jit.load the frozen, already-trained reach policy once
    per process. The exported JIT module already bakes in the observation
    normalizer (same reasoning scripts/reach_apple_hybrid.py's own docstring
    gives for replaying it outside Isaac Lab), so calling it directly here
    reproduces exactly what the reach task's own --policy in-process mode
    would output for the same observation.

    `policy_path`: an explicit path, or "" to auto-resolve the latest
    apple_grasp_reach export (see _find_last_reach_policy) the first time
    this is called.
    """
    resolved = policy_path or _find_last_reach_policy()
    key = os.path.abspath(resolved)
    if key not in _reach_policy_cache:
        _reach_policy_cache[key] = torch.jit.load(key, map_location=device).eval()
    return _reach_policy_cache[key]


def _reach_obs(
    env: "ManagerBasedRLEnv",
    ids: torch.Tensor,
    arm_cfg: SceneEntityCfg,
    robot_body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_local_offset: tuple[float, float, float],
    prev_raw_action: torch.Tensor,
) -> torch.Tensor:
    """Reassemble the reach policy's own 28-dim observation vector, for just
    `ids`, by calling the SAME functions envs/apple_grasp_env_cfg.py's
    ObservationsCfg.PolicyObs wires up (joint_pos_selected/joint_vel_selected/
    object_pos_relative_to_body/body_quat_relative_to_body) -- NOT the
    ObservationManager, since we don't want the grasp task's own reward/
    termination managers computed during this inner loop. Reusing the real
    functions guarantees behavioral parity with how the reach policy was
    actually trained, and survives future edits to those functions.

    Order MUST match apple_grasp_env_cfg.ObservationsCfg.PolicyObs exactly:
    right_arm_pos(7), right_arm_vel(7), apple_pos_palm(3),
    palm_orientation_torso(4), last_action(7) = 28.
    """
    from .observations import (
        joint_pos_selected,
        joint_vel_selected,
        object_pos_relative_to_body,
        body_quat_relative_to_body,
    )

    if not hasattr(env, "_autoplay_torso_cfg"):
        torso_cfg = SceneEntityCfg("robot", body_names=["torso_link"])
        torso_cfg.resolve(env.scene)
        env._autoplay_torso_cfg = torso_cfg

    arm_pos = joint_pos_selected(env, arm_cfg)[ids]
    arm_vel = joint_vel_selected(env, arm_cfg)[ids]
    apple_pos_palm = object_pos_relative_to_body(
        env, object_cfg, robot_body_cfg, local_offset=palm_local_offset
    )[ids]
    palm_orientation_torso = body_quat_relative_to_body(
        env, robot_body_cfg, env._autoplay_torso_cfg
    )[ids]

    return torch.cat(
        [arm_pos, arm_vel, apple_pos_palm, palm_orientation_torso, prev_raw_action], dim=-1
    )


def hold_apple_until_first_closure(
    env: "ManagerBasedRLEnv",
    env_ids,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mu: float = 0.5,
    contact_threshold: float = 0.05,
) -> None:
    """Keep the apple pinned at its hand-off pose ("still attached to the
    branch") for every env that hasn't yet achieved force closure even once
    this episode -- continues reach_autoplay_handoff's own pin mechanism
    into the grasp trainee's real, counted env.step() calls, so the policy
    gets to establish a grip with NO gravity/time pressure, exactly like a
    real apple that doesn't need to be "caught" while still on the branch.
    Releases (stops re-pinning) permanently for an env the moment force
    closure is first detected -- from that point on, gravity is real and
    apple_fell_from_grip/grasp_success become a genuine test of whether the
    established grip actually holds.

    Registered as a mode="interval" EventTerm with interval_range_s=(0.0,
    0.0), which makes Isaac Lab's EventManager call this every single
    control step for every env (see managers/event_manager.py's interval
    handling: an env's timer starts at 0 and is immediately due again after
    the first `dt` subtraction) -- env_ids received here is effectively "all
    envs, every step"; this function itself decides per-env whether to
    actually touch anything, based on env._grasp_ever_closed.

    Ignored for envs where reach_autoplay_handoff has never run yet
    (defensive only -- shouldn't happen once training is underway, since
    every episode starts with a hand-off).
    """
    if not hasattr(env, "_grasp_handoff_apple_pos") or not hasattr(env, "_grasp_ever_closed"):
        return

    apple: RigidObject = env.scene[object_cfg.name]

    from .grasp_rewards import _force_closure_ok
    closure_now = _force_closure_ok(
        env, sensor_cfg, robot_cfg, object_cfg, mu=mu, contact_threshold=contact_threshold
    )
    env._grasp_ever_closed = env._grasp_ever_closed | closure_now

    still_holding = ~env._grasp_ever_closed
    if not still_holding.any():
        return
    ids = still_holding.nonzero(as_tuple=True)[0]

    pose = torch.cat([env._grasp_handoff_apple_pos[ids], env._grasp_handoff_apple_quat[ids]], dim=-1)
    apple.write_root_pose_to_sim(pose, env_ids=ids)
    apple.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def reach_autoplay_handoff(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    arm_cfg: SceneEntityCfg,
    robot_body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    offset_z: float,
    palm_normal_axis: tuple[float, float, float],
    palm_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pos_tol: float = 0.04,
    ori_tol_rad: float = 0.30,
    hold_time: float = 0.5,
    watchdog_s: float = 2.0,
    scale: float = 0.5,
    clip_val: float = 3.0,
    max_velocity: float = 2.0,
    max_acceleration: float = 15.0,
    policy_path: str = "",
) -> None:
    """Run the frozen reach policy, live, for the just-reset env_ids, until
    each converges to the grasp point (or its own per-env watchdog fires),
    then hand control to the grasp trainee. Place this LAST in the grasp
    task's EventsCfg, after apple placement, so it sees the apple at its
    final randomized position for this episode.

    policy_path: explicit path to an exported reach policy.pt, or "" (the
    default) to auto-resolve the latest AppleGrasp-v0 training run's export
    the first time this fires (see _find_last_reach_policy) -- resolved
    lazily at reset time, never at config-import time.

    scale/clip_val/max_velocity/max_acceleration MUST match the reach task's
    OWN ActionsCfg.right_arm (apple_grasp_env_cfg.py) -- this reproduces
    RateLimitedJointPositionAction.process_actions's exact affine + clip +
    velocity/acceleration-clamp math locally (see envs/mdp/actions.py), so
    the replayed trajectory matches what the reach policy actually learned
    against.
    """
    from .reach_rewards import _palm_target_dist, _palm_normal_angle

    robot: Articulation = env.scene["robot"]
    apple: RigidObject = env.scene[object_cfg.name]
    device = env.device

    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
        env_ids, device=device, dtype=torch.long
    )
    n = ids.numel()
    if n == 0:
        return

    arm_ids = arm_cfg.joint_ids
    n_arm = len(arm_ids)

    policy = _load_reach_policy(policy_path, device)

    default_pos = robot.data.default_joint_pos[ids][:, arm_ids].clone()
    prev_target = robot.data.joint_pos[ids][:, arm_ids].clone()
    prev_vel = torch.zeros(n, n_arm, device=device)
    # Zero on the very first tick, mirroring mdp.last_action's zero-at-
    # episode-start convention (the real reach env's first-ever observation
    # after a reset genuinely sees zero last-action, since reset() computes
    # obs_buf before any process_action() call).
    prev_raw_action = torch.zeros(n, n_arm, device=device)
    hold_counter = torch.zeros(n, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)

    # Pin the apple for the WHOLE batch's autoplay window, released only
    # once every env in THIS batch has converged/watchdogged -- releasing
    # per-env immediately on convergence would expose an early-converged
    # env's apple (open hand, no real grip yet) to gravity/drift while
    # physics keeps stepping for its still-autoplaying batch-mates.
    pinned_pose = torch.cat([apple.data.root_pos_w[ids], apple.data.root_quat_w[ids]], dim=-1)
    zero_vel = torch.zeros(n, 6, device=device)

    dt = env.step_dt
    max_ticks = max(1, math.ceil(watchdog_s / dt))
    hold_steps_required = max(1, round(hold_time / dt))

    for _tick in range(max_ticks):
        obs = _reach_obs(env, ids, arm_cfg, robot_body_cfg, object_cfg, palm_local_offset, prev_raw_action)
        with torch.inference_mode():
            raw_action = policy(obs)
        prev_raw_action = raw_action.clone()

        # -- RateLimitedJointPositionAction.process_actions, reproduced --
        # offset+scale, then an ABSOLUTE clamp to [-clip_val, clip_val] (NOT
        # relative to default_pos -- confirmed against Isaac Lab's own
        # JointAction.process_actions: clip bounds are applied directly to
        # the already-offset target).
        target_desired = raw_action * scale + default_pos
        target_desired = torch.clamp(target_desired, min=-clip_val, max=clip_val)

        v_desired = (target_desired - prev_target) / dt
        v_cmd = torch.clamp(v_desired, -max_velocity, max_velocity)
        max_dv = max_acceleration * dt
        v_cmd = torch.clamp(v_cmd, prev_vel - max_dv, prev_vel + max_dv)
        target_cmd = prev_target + v_cmd * dt

        # Rows that already converged/watchdogged just keep re-asserting
        # their last command (a no-op hold) rather than continuing to move.
        target_cmd = torch.where(active.unsqueeze(-1), target_cmd, prev_target)
        v_cmd = torch.where(active.unsqueeze(-1), v_cmd, torch.zeros_like(v_cmd))

        prev_target = target_cmd
        prev_vel = v_cmd

        robot.set_joint_position_target(target_cmd, joint_ids=arm_ids, env_ids=ids)

        for _sub in range(env.cfg.decimation):
            # Re-pin every substep, not once per tick -- gravity acts on the
            # (always-dynamic) apple every substep between control ticks too.
            apple.write_root_pose_to_sim(pinned_pose, env_ids=ids)
            apple.write_root_velocity_to_sim(zero_vel, env_ids=ids)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

        # -- convergence check, same neighbourhood test reach_success uses --
        d_all = _palm_target_dist(env, robot_body_cfg, object_cfg, offset_z, palm_local_offset)
        theta_all = _palm_normal_angle(env, robot_body_cfg, palm_normal_axis)
        d = d_all[ids]
        theta = theta_all[ids]
        in_neighbourhood = (d < pos_tol) & (theta < ori_tol_rad)

        hold_counter = torch.where(in_neighbourhood, hold_counter + 1, torch.zeros_like(hold_counter))
        newly_converged = hold_counter >= hold_steps_required
        active = active & ~newly_converged

        if not active.any():
            break
    # else: per-env watchdog -- any row still `active` here simply hands off
    # anyway once max_ticks is exhausted, accepting an imperfect hand-off
    # into grasp training rather than looping forever. Bounds worst-case
    # per-reset cost and mirrors scripts/reach_apple_hybrid.py's own
    # switch_dist/rl_max_time watchdog fallback.

    # -- hand-off bookkeeping, read by terminations_and_events.grasp_success
    #    and grasp_rewards' hold/success rewards --
    if not hasattr(env, "_grasp_handoff_apple_pos"):
        env._grasp_handoff_apple_pos = torch.zeros(env.num_envs, 3, device=device)
    if not hasattr(env, "_grasp_handoff_apple_quat"):
        env._grasp_handoff_apple_quat = torch.zeros(env.num_envs, 4, device=device)
    if not hasattr(env, "_grasp_handoff_arm_pos"):
        env._grasp_handoff_arm_pos = torch.zeros(env.num_envs, n_arm, device=device)
    if not hasattr(env, "_grasp_ever_closed"):
        env._grasp_ever_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._grasp_handoff_apple_pos[ids] = apple.data.root_pos_w[ids].clone()
    env._grasp_handoff_apple_quat[ids] = apple.data.root_quat_w[ids].clone()
    env._grasp_handoff_arm_pos[ids] = robot.data.joint_pos[ids][:, arm_ids].clone()
    # Fresh episode -> fresh "hasn't gripped it yet" state, read by
    # hold_apple_until_first_closure below (must reset here, at the one
    # point every new episode passes through, or the flag would incorrectly
    # stay True forever after the first episode that ever closed).
    env._grasp_ever_closed[ids] = False

    # No pin-release here: the apple stays supported ("still attached to the
    # branch") for as long as it takes the grasp trainee to first achieve
    # force closure on its own -- see hold_apple_until_first_closure below,
    # a separate mode="interval" event that continues the SAME pin/hold
    # mechanism into the trainee's own real, counted env.step() calls (this
    # function's own pin can't do that -- it only runs once per reset,
    # before the trainee is ever exposed to the environment). Deliberate:
    # a real apple doesn't need to be "caught" while still on the branch --
    # only once the policy has actually established a grip does gravity
    # become the thing being tested.
