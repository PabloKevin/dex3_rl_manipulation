# Copyright 2026 - G1 Dex3 Apple Grasp Project
"""One batch of policy episodes recorded WITHOUT the success termination.

evaluate_policy.py ends each episode's recording the instant reach_success
fires -- a real termination, by design (see the paper: success is defined as
a *held* pose, and the environment ends the episode there). That is correct
for scoring, but it means the recorded trajectory of any successful episode
literally stops at that tick: there is no further arm state to plot, because
the simulation stopped advancing that env. Rendered next to CasADi/hybrid rows
that run to the full window, a policy trace stopping mid-slope reads as an
artifact even though it is not one.

This script disables ONLY the reach_success termination (time_out stays at the
5 s horizon; apple_dropped stays, it never fires against a kinematic fruit) so
a full batch of envs runs in lockstep with no early resets, and records the
same per-step quantities evaluate_policy.py does -- read directly off the
action term and robot/apple state, since EvalProbe cannot be reused here (it
reads its success flag from the reach_success termination, which no longer
exists as a registered term). Success/hold scoring is recomputed after the
fact from the recorded d(t)/theta(t) traces, using the identical
pos_tol/ori_tol_rad/hold_time reach_success uses -- see _score_episodes below.

Output uses eval_utils.save_results' schema exactly, so the result loads
through eval_utils.load_results and feeds benchmark_casadi.py /
benchmark_hybrid.py / plot_results.py unmodified.

Usage
-----
    PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python
    $PY scripts/evaluation/eval_continuous_traj.py --headless --num_envs 64 \
        --n_steps 150 --out results/eval_continuous.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Record full-length (non-terminating) policy trajectories for the fig_compare figure.")
parser.add_argument("--task", type=str, default="AppleReach-v1", help="Task name (must be the TRAINING task).")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--n_steps", type=int, default=150, help="Control steps to record (150 = 3 s at 50 Hz).")
parser.add_argument("--proximity_stop", type=str, default="on", choices=["on", "off"],
                    help="Run the quintic proximity-stop supervisor on top of the policy, as deployed.")
parser.add_argument("--stop_dist", type=float, default=0.09)
parser.add_argument("--stop_dwell", type=float, default=0.6)
parser.add_argument("--stop_duration", type=float, default=0.4)
parser.add_argument("--stop_watchdog", type=float, default=2.5)
parser.add_argument("--out", type=str, default=None, help="Output .npz path.")
parser.add_argument("--tag", type=str, default="", help="Free-form label stored in the npz metadata.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric / use USD I/O.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if "-Play" in args_cli.task:
    raise SystemExit(
        f"[FATAL] --task {args_cli.task} is a Play variant: it sets episode_length_s=1e6 and "
        "nulls the reset events. Use the training task (AppleReach-v1)."
    )

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import json
import os
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import envs  # noqa: E402,F401

import eval_utils  # noqa: E402
from envs.apple_grasp_env_cfg import (  # noqa: E402
    GRASP_STANDOFF,
    PALM_LOCAL_OFFSET,
    PALM_NORMAL_AXIS,
    RIGHT_ARM_JOINTS,
)
from envs.mdp.reach_rewards import _palm_normal_angle, _palm_target_dist  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

POS_TOL = 0.04
ORI_TOL_RAD = 0.30
HOLD_TIME_S = 0.5


def _latest_checkpoint(experiment_name: str) -> str:
    root = os.path.join(_REPO, "logs", "rsl_rl", experiment_name)
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not runs:
        raise SystemExit(f"[FATAL] no runs under {root}")
    for run in reversed(runs):
        ckpts = glob.glob(os.path.join(run, "model_*.pt"))
        if ckpts:
            return max(ckpts, key=lambda p: int(os.path.basename(p)[6:-3]))
    raise SystemExit(f"[FATAL] no model_*.pt under any run in {root}")


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", _REPO, "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _score_episodes(traj_d: np.ndarray, traj_theta: np.ndarray, dt: float) -> dict:
    """Recompute reach_success's exact hold criterion from recorded d(t)/theta(t).

    in_tol must hold for a continuous run of >= hold_steps samples; t_success
    is the time of the FIRST sample at which that run has just completed
    (mirrors terminations_and_events.reach_success's counter-threshold logic).
    """
    n, T = traj_d.shape
    hold_steps = int(round(HOLD_TIME_S / dt))
    in_tol = (traj_d < POS_TOL) & (traj_theta < ORI_TOL_RAD)

    ep_success = np.zeros(n, bool)
    t_success = np.full(n, -1.0, np.float32)
    first_step = np.full(n, -1, np.int32)
    n_in_tol_after = np.zeros(n, np.int64)
    n_steps_after = np.zeros(n, np.int64)
    n_exits = np.zeros(n, np.int64)

    for i in range(n):
        run = 0
        entered = False
        prev_in = False
        for t in range(T):
            if in_tol[i, t]:
                run += 1
                if not entered:
                    entered = True
                    first_step[i] = t
                if entered:
                    n_steps_after[i] += 1
                    n_in_tol_after[i] += 1
                if run == hold_steps and not ep_success[i]:
                    ep_success[i] = True
                    t_success[i] = t * dt
            else:
                if entered:
                    n_steps_after[i] += 1
                    if prev_in:
                        n_exits[i] += 1
                run = 0
            prev_in = in_tol[i, t]

    frac_in_tol_after = np.where(n_steps_after > 0,
                                 n_in_tol_after / np.maximum(n_steps_after, 1),
                                 np.nan).astype(np.float32)
    t_first_reach = np.where(first_step >= 0, first_step * dt, np.nan).astype(np.float32)
    return {
        "ep_success": ep_success,
        "t_success": t_success,
        "first_reach_step": first_step.astype(np.int16),
        "t_first_reach": t_first_reach,
        "frac_in_tol_after": frac_in_tol_after,
        "n_tol_exits": np.where(first_step >= 0, n_exits, -1).astype(np.int16),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint \
        else _latest_checkpoint(agent_cfg.experiment_name)
    run_dir = os.path.dirname(resume_path)
    ckpt_iter = int(os.path.basename(resume_path)[6:-3])

    act = env_cfg.actions.right_arm
    v_max, a_max = float(act.max_velocity), float(act.max_acceleration)

    # ── the one substantive change vs evaluate_policy.py: no early success
    #    termination, so envs run in lockstep for the whole recording window ──
    env_cfg.terminations.reach_success = None
    print("[INFO] reach_success termination disabled -- episodes run to the "
          "full horizon (only time_out/apple_dropped remain)")

    print(f"[INFO] checkpoint : {resume_path}  (iteration {ckpt_iter})")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=u.device)
    policy_nn = runner.alg.policy

    robot = u.scene["robot"]
    apple = u.scene["apple"]
    robot_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
    robot_cfg.resolve(u.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)
    arm_cfg.resolve(u.scene)
    arm_ids = arm_cfg.joint_ids
    palm_id = robot_cfg.body_ids[0]

    stop = None
    if args_cli.proximity_stop == "on":
        from proximity_stop_batch import BatchQuinticStop, BatchStopConfig
        stop = BatchQuinticStop(
            u.num_envs, len(RIGHT_ARM_JOINTS), device=u.device,
            cfg=BatchStopConfig(stop_dist=args_cli.stop_dist, dwell_time=args_cli.stop_dwell,
                                stop_duration=args_cli.stop_duration, watchdog_time=args_cli.stop_watchdog))
        a_scale = float(env_cfg.actions.right_arm.scale)
        q_def = robot.data.default_joint_pos[:, arm_ids]
        print(f"[INFO] proximity stop ON  (stop_dist={args_cli.stop_dist:.3f} m, "
              f"dwell={args_cli.stop_dwell:.2f} s, T={args_cli.stop_duration:.2f} s, "
              f"watchdog={args_cli.stop_watchdog:.2f} s)")
    else:
        print("[INFO] proximity stop OFF (bare controller)")

    n, T = args_cli.num_envs, args_cli.n_steps
    J = len(RIGHT_ARM_JOINTS)
    dev = u.device
    tr_q = torch.zeros(n, T, J, device=dev)
    tr_dq = torch.zeros(n, T, J, device=dev)
    tr_a = torch.zeros(n, T, J, device=dev)
    tr_u = torch.zeros(n, T, J, device=dev)
    tr_d = torch.zeros(n, T, device=dev)
    tr_th = torch.zeros(n, T, device=dev)

    obs = env.get_observations()
    # First reset is issued by RslRlVecEnvWrapper before any step; wait one
    # step so env._shoulder_local_xyz (written by randomize_tree_and_apple at
    # mode="reset") exists before we read it below.
    t0 = time.time()
    with torch.inference_mode():
        for t in range(T):
            actions = policy(obs)
            if stop is not None:
                stop.reset_envs(u.episode_length_buf <= 1, t=0.0)
                dist = torch.norm(apple.data.root_pos_w - robot.data.body_pos_w[:, palm_id, :], dim=-1)
                q = robot.data.joint_pos[:, arm_ids]
                dq = robot.data.joint_vel[:, arm_ids]
                t_env = u.episode_length_buf.float() * u.step_dt
                stop.update(t_env, dist, q, dq)
                ov, q_cmd = stop.command(t_env, q_def + a_scale * actions)
                if bool(ov.any()):
                    actions = torch.where(ov.unsqueeze(-1), (q_cmd - q_def) / a_scale, actions)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            if bool(dones.any()):
                print(f"[warn] step {t}: {int(dones.sum())} env(s) reset unexpectedly "
                      "(time_out/apple_dropped) -- their trailing samples repeat the new episode")

            term = u.action_manager.get_term("right_arm")
            tr_q[:, t] = robot.data.joint_pos[:, arm_ids]
            tr_dq[:, t] = robot.data.joint_vel[:, arm_ids]
            tr_a[:, t] = term.raw_actions
            tr_u[:, t] = term.processed_actions
            tr_d[:, t] = _palm_target_dist(u, robot_cfg, SceneEntityCfg("apple"), GRASP_STANDOFF, PALM_LOCAL_OFFSET)
            tr_th[:, t] = _palm_normal_angle(u, robot_cfg, PALM_NORMAL_AXIS)
            if (t + 1) % 50 == 0:
                print(f"  step {t + 1:4d}/{T}  ({(t + 1) * n / (time.time() - t0):.0f} env-steps/s)")

    shoulder_w = u.scene.env_origins + u._shoulder_local_xyz
    apple_w = apple.data.root_pos_w.clone()

    d_np, th_np = tr_d.cpu().numpy(), tr_th.cpu().numpy()
    scored = _score_episodes(d_np, th_np, float(u.step_dt))

    delta = (apple_w - shoulder_w).cpu().numpy()
    r_xy = np.linalg.norm(delta[:, :2], axis=-1)
    az = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    dz = delta[:, 2]   # side-on target: same height as the apple centre
    r3d = np.sqrt(r_xy ** 2 + dz ** 2)

    episodes = {
        "ep_index": np.arange(n, dtype=np.int32),
        "ep_env_id": np.arange(n, dtype=np.int16),
        "ep_global_step": np.full(n, T, dtype=np.int32),
        "ep_length": np.full(n, T, dtype=np.int16),
        "ep_timeout": np.zeros(n, bool),
        "ep_dropped": np.zeros(n, bool),
        "d_final": d_np[:, -1].astype(np.float32),
        "d_min": d_np.min(axis=1).astype(np.float32),
        "theta_final": th_np[:, -1].astype(np.float32),
        "theta_min": th_np.min(axis=1).astype(np.float32),
        "apple_pos_w": apple_w.cpu().numpy().astype(np.float32),
        "shoulder_pos_w": shoulder_w.cpu().numpy().astype(np.float32),
        "r_xy": r_xy.astype(np.float32),
        "r3d_target": r3d.astype(np.float32),
        "azimuth_deg": az.astype(np.float32),
        "dz_target": dz.astype(np.float32),
        "q_final": tr_q[:, -1].cpu().numpy().astype(np.float32),
        **scored,
    }
    traces = {
        "traj_mask": np.ones((n, T), bool),
        "traj_q": tr_q.cpu().numpy().astype(np.float32),
        "traj_dq": tr_dq.cpu().numpy().astype(np.float32),
        "traj_a_raw": tr_a.cpu().numpy().astype(np.float32),
        "traj_u": tr_u.cpu().numpy().astype(np.float32),
        "traj_d": d_np.astype(np.float32),
        "traj_theta": th_np.astype(np.float32),
    }
    scalars = {
        "dt": np.float32(u.step_dt),
        "pos_tol": np.float32(POS_TOL),
        "ori_tol_rad": np.float32(ORI_TOL_RAD),
        "hold_steps": np.int32(round(HOLD_TIME_S / u.step_dt)),
        "standoff": np.float32(GRASP_STANDOFF),
        "joint_names": np.array(RIGHT_ARM_JOINTS, dtype="<U32"),
    }
    meta = {
        "checkpoint": resume_path,
        "run_dir": run_dir,
        "checkpoint_iteration": ckpt_iter,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "n_episodes": int(n),
        "n_steps_recorded": T,
        "reach_success_termination": "disabled",
        "rate_limit": "on",
        "proximity_stop": args_cli.proximity_stop,
        "stop_dist": args_cli.stop_dist,
        "stop_dwell": args_cli.stop_dwell,
        "stop_duration": args_cli.stop_duration,
        "stop_watchdog": args_cli.stop_watchdog,
        "max_velocity": v_max,
        "max_acceleration": a_max,
        "seed": agent_cfg.seed,
        "palm_normal_axis": list(PALM_NORMAL_AXIS),
        "palm_local_offset": list(PALM_LOCAL_OFFSET),
        "grasp_standoff": float(GRASP_STANDOFF),
        "reach_min": float(env_cfg.events.randomize_tree.params["reach_min"]),
        "reach_max": float(env_cfg.events.randomize_tree.params["reach_max"]),
        "azimuth_center_deg": float(env_cfg.events.randomize_tree.params["reach_azimuth_center_deg"]),
        "azimuth_span_deg": float(env_cfg.events.randomize_tree.params["reach_azimuth_span_deg"]),
        "git_rev": _git_rev(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args_cli.tag,
        "note": ("reach_success termination disabled for this run so the recorded trajectory "
                 "continues past the pose that would normally end the episode -- for the "
                 "fig_compare trajectory figure only, NOT for success-rate scoring: this file's "
                 "episode count/randomization is far too small to replace results/eval_on.npz."),
    }

    out = args_cli.out or os.path.join(_REPO, "results", "eval_continuous.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    eval_utils.save_results(out, {**episodes, **scalars}, traces, meta)
    print(f"[INFO] wrote {out}")
    print(f"[INFO] {int(scored['ep_success'].sum())}/{n} episodes reached a held success "
          f"within the recorded {T * float(u.step_dt):.1f} s window")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
