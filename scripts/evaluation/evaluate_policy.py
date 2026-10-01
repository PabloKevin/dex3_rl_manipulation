# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Batch evaluation of a trained reach-and-orient policy.

Rolls the deterministic policy over many randomized episodes and writes one
compressed .npz of per-episode metrics and per-timestep traces, plus a printed
summary table. Structure follows scripts/play.py (AppLauncher before any
isaaclab import, hydra_task_config resolving --task against the gym registry).

Two things this script does NOT do the obvious way, both on purpose:

  * It records nothing in the rollout loop. All bookkeeping lives in
    envs.mdp.eval_metrics.EvalProbe, an always-False termination term, because
    terminated envs are already reset by the time env.step() returns -- see that
    module's docstring.

  * The rate-limiter ablation raises max_velocity/max_acceleration to 1e9
    rather than swapping in a different action term. At those values
    RateLimitedJointPositionAction is a mathematically exact no-op
    (clip(v, +-1e9) = v, so target_cmd = target_desired), identical to the
    stock JointPositionAction including the episode-start re-anchor -- so both
    arms of the ablation go through the same code path.

Usage
-----
    PY=/home/pablo_kevin/miniconda3/envs/isaaclab_curobo/bin/python

    # defaults to the newest run's newest checkpoint
    $PY scripts/evaluation/evaluate_policy.py --headless --rate_limit on \
        --out results/eval_on.npz

    $PY scripts/evaluation/evaluate_policy.py --headless --rate_limit off \
        --out results/eval_off.npz --seed 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a trained reach policy over many episodes.")
parser.add_argument("--task", type=str, default="AppleReach-v1", help="Task name (must be the TRAINING task).")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--n_episodes", type=int, default=512, help="Completed episodes to collect.")
parser.add_argument("--max_steps", type=int, default=None, help="Safety cap on rollout steps.")
parser.add_argument("--warmup_steps", type=int, default=250,
                    help="Discard episodes completing within this many rollout steps.")
parser.add_argument("--rate_limit", type=str, default="on", choices=["on", "off"],
                    help="'off' disables the action term's velocity/acceleration limits (ablation).")
parser.add_argument("--proximity_stop", type=str, default="on", choices=["on", "off"],
                    help="Run the quintic proximity-stop supervisor on top of the policy, as the "
                         "deployed system does. 'off' measures the bare policy.")
parser.add_argument("--stop_dist", type=float, default=0.09)
parser.add_argument("--stop_dwell", type=float, default=0.6)
parser.add_argument("--stop_duration", type=float, default=0.4)
parser.add_argument("--stop_watchdog", type=float, default=2.5)
parser.add_argument("--out", type=str, default=None, help="Output .npz path.")
parser.add_argument("--no_traces", action="store_true", help="Skip per-timestep trace arrays.")
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
        "nulls the reset events, so it produces exactly ONE episode and no domain randomization. "
        "Evaluate against the training task (AppleReach-v1)."
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
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

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
from envs.mdp.eval_metrics import make_eval_probe_term  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

_ABLATION_LIMIT = 1.0e9


def _latest_checkpoint(experiment_name: str) -> str:
    """Newest model_*.pt in the newest run directory.

    Resolved from __file__, not the CWD: play.py builds its log root with a
    relative os.path.join("logs", ...), which is why stray scripts/logs/ and
    scripts/outputs/ directories exist in this repo.
    """
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = (retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint
                   else _latest_checkpoint(agent_cfg.experiment_name))
    run_dir = os.path.dirname(resume_path)
    ckpt_iter = int(os.path.basename(resume_path)[6:-3])

    # ── rate-limiter ablation: mutate the cfg BEFORE the env is built ──────
    act = env_cfg.actions.right_arm
    if args_cli.rate_limit == "off":
        act.max_velocity = _ABLATION_LIMIT
        act.max_acceleration = _ABLATION_LIMIT
    v_max, a_max = float(act.max_velocity), float(act.max_acceleration)

    # ── attach the recorder as the LAST termination term ───────────────────
    # Assigned here, inside main(), rather than at module scope: the Hydra
    # decorator snapshots env_cfg.to_dict() from the gym registry, and a
    # ManagerTermBase subclass does not round-trip through OmegaConf.
    env_cfg.terminations.eval_probe = make_eval_probe_term(
        standoff=GRASP_STANDOFF,
        palm_local_offset=PALM_LOCAL_OFFSET,
        palm_normal_axis=PALM_NORMAL_AXIS,
        arm_joint_names=RIGHT_ARM_JOINTS,
        record_traces=not args_cli.no_traces,
    )

    print(f"[INFO] checkpoint : {resume_path}  (iteration {ckpt_iter})")
    print(f"[INFO] rate limit : {args_cli.rate_limit}  (v_max={v_max:g}, a_max={a_max:g})")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    active = env.unwrapped.termination_manager.active_terms
    assert active[-1] == "eval_probe", (
        f"eval_probe must be the last termination term so the others' flags are "
        f"already written when it reads them; got order {active}"
    )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)   # deterministic mean
    policy_nn = runner.alg.policy
    probe = env.unwrapped.termination_manager.get_term_cfg("eval_probe").func

    max_steps = args_cli.max_steps or (
        args_cli.warmup_steps + (args_cli.n_episodes // max(args_cli.num_envs, 1) + 2) * 300
    )

    # ── proximity-stop supervisor (as deployed) ────────────────────────────
    u = env.unwrapped
    stop = None
    if args_cli.proximity_stop == "on":
        from proximity_stop_batch import BatchQuinticStop, BatchStopConfig
        stop = BatchQuinticStop(
            u.num_envs, len(RIGHT_ARM_JOINTS), device=u.device,
            cfg=BatchStopConfig(stop_dist=args_cli.stop_dist,
                                dwell_time=args_cli.stop_dwell,
                                stop_duration=args_cli.stop_duration,
                                watchdog_time=args_cli.stop_watchdog))
        robot = u.scene["robot"]
        arm_ids = u.termination_manager.get_term_cfg("eval_probe").params["arm_cfg"].joint_ids
        palm_id = u.termination_manager.get_term_cfg("eval_probe").params["robot_cfg"].body_ids[0]
        apple = u.scene["apple"]
        q_def = robot.data.default_joint_pos[:, arm_ids]
        a_scale = float(env_cfg.actions.right_arm.scale)
        print(f"[INFO] proximity stop ON  (stop_dist={args_cli.stop_dist:.3f} m, "
              f"dwell={args_cli.stop_dwell:.2f} s, T={args_cli.stop_duration:.2f} s, "
              f"watchdog={args_cli.stop_watchdog:.2f} s)")
    else:
        print("[INFO] proximity stop OFF (bare controller)")

    obs = env.get_observations()
    step = 0
    t0 = time.time()
    with torch.inference_mode():
        while probe.n_done() < args_cli.n_episodes + args_cli.num_envs and step < max_steps:
            actions = policy(obs)
            if stop is not None:
                # Episode time, and FSM reset on episode boundaries. episode_length_buf
                # is 0 on the first step after a reset.
                stop.reset_envs(u.episode_length_buf <= 1, t=0.0)
                # Raw palm-link-origin to fruit distance -- run_env.py's convention.
                dist = torch.norm(apple.data.root_pos_w
                                  - robot.data.body_pos_w[:, palm_id, :], dim=-1)
                q = robot.data.joint_pos[:, arm_ids]
                dq = robot.data.joint_vel[:, arm_ids]
                # Per-env episode time: the supervisor's clock is per attempt.
                t_env = u.episode_length_buf.float() * u.step_dt
                stop.update(t_env, dist, q, dq)
                ov, q_cmd = stop.command(t_env, q_def + a_scale * actions)
                if bool(ov.any()):
                    actions = torch.where(ov.unsqueeze(-1),
                                          (q_cmd - q_def) / a_scale, actions)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            step += 1
            if step % 250 == 0:
                print(f"  step {step:5d} | episodes {probe.n_done():5d} "
                      f"| {step * args_cli.num_envs / (time.time() - t0):.0f} env-steps/s")

    episodes, traces = probe.pop_episodes(warmup_steps=args_cli.warmup_steps)
    if not episodes:
        raise SystemExit("[FATAL] no episodes completed — increase --max_steps.")

    n = episodes["ep_success"].shape[0]
    print(f"[INFO] {n} episodes after warm-up ({probe.n_done()} flushed, "
          f"{args_cli.num_envs} left in flight and discarded)")

    scalars = {
        "dt": np.float32(env.unwrapped.step_dt),
        "pos_tol": np.float32(probe.pos_tol),
        "ori_tol_rad": np.float32(probe.ori_tol_rad),
        "hold_steps": np.int32(round(0.5 / env.unwrapped.step_dt)),
        "standoff": np.float32(GRASP_STANDOFF),
        "joint_names": np.array(RIGHT_ARM_JOINTS, dtype="<U32"),
    }
    meta = {
        "checkpoint": resume_path,
        "run_dir": run_dir,
        "checkpoint_iteration": ckpt_iter,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "n_episodes_requested": args_cli.n_episodes,
        "n_episodes": int(n),
        "rate_limit": args_cli.rate_limit,
        "proximity_stop": args_cli.proximity_stop,
        "stop_dist": args_cli.stop_dist,
        "stop_dwell": args_cli.stop_dwell,
        "stop_duration": args_cli.stop_duration,
        "stop_watchdog": args_cli.stop_watchdog,
        "max_velocity": v_max,
        "max_acceleration": a_max,
        "seed": agent_cfg.seed,
        "scene_seed": os.environ.get("SCENE_SEED", "12345"),
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
        "note": "PhysX GPU stepping is not bitwise deterministic; runs reproduce in distribution.",
    }

    out = args_cli.out or os.path.join(
        _REPO, "results", f"eval_{os.path.basename(run_dir)}_{ckpt_iter}_{args_cli.rate_limit}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    eval_utils.save_results(out, {**episodes, **scalars}, traces, meta)
    print(f"[INFO] wrote {out}")

    res = eval_utils.load_results(out)
    _self_check(res)
    summary = eval_utils.summarize(res)
    text = eval_utils.format_summary_table(summary)
    print(text)
    txt_path = os.path.splitext(out)[0] + "_summary.txt"
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    print(f"[INFO] wrote {txt_path}")

    env.close()


def _self_check(res) -> None:
    """Assertions that would fail if per-episode bookkeeping crossed a boundary.

    All five follow from the termination logic, so a violation is a bug in the
    recorder rather than an interesting property of the policy.
    """
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("[CHECK] episode bookkeeping")
    succ = res.ep_success.astype(bool)
    hold_steps = int(res.hold_steps)
    check((res.ep_length <= 250).all(), "ep_length <= max_episode_length")
    if succ.any():
        check((res.d_final[succ] < float(res.pos_tol) + 1e-4).all(),
              "success => final position within tolerance")
        check((res.theta_final[succ] < float(res.ori_tol_rad) + 1e-4).all(),
              "success => final orientation within tolerance")
        check((res.ep_length[succ] >= hold_steps).all(),
              f"success => episode lasted at least the {hold_steps}-step hold")
    else:
        print("  SKIP  no successful episodes to check")
    uniq = np.unique(np.round(res.apple_pos_w, 4), axis=0).shape[0]
    check(uniq > 0.5 * res.n_episodes, f"apple positions vary across episodes ({uniq}/{res.n_episodes} unique)")
    m = res.meta
    check(((res.azimuth_deg >= m["azimuth_center_deg"] - m["azimuth_span_deg"] / 2 - 1.0)
           & (res.azimuth_deg <= m["azimuth_center_deg"] + m["azimuth_span_deg"] / 2 + 1.0)).all(),
          "sampled azimuth inside the configured span")

    if getattr(res, "has_traces", False):
        chk = eval_utils.check_rate_limits(res.traj_u, res.traj_mask, res.dt,
                                           m["max_velocity"], m["max_acceleration"])
        print("[CHECK] rate limiter")
        print(f"  commanded |dq|  max = {chk['cmd_vel_max']:.4f} rad/s   (limit {m['max_velocity']:g})")
        print(f"  commanded |ddq| max = {chk['cmd_acc_max']:.4f} rad/s^2 (limit {m['max_acceleration']:g})")
        if m["rate_limit"] == "on":
            check(chk["vel_within"], "commanded velocity within the configured bound")
            check(chk["acc_within"], "commanded acceleration within the configured bound")
        else:
            check(chk["cmd_vel_max"] > 2.0,
                  "ablation took effect (commanded velocity exceeds the nominal 2.0 rad/s)")
    if not ok:
        print("[WARN] one or more checks FAILED — treat the numbers as suspect.")


if __name__ == "__main__":
    main()
    simulation_app.close()
