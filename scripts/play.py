# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py — this repo
# isn't an installed Isaac Lab extension, so it needs its own copy that registers
# this repo's gym tasks (see the "import envs" block below) before Hydra resolves
# --task against the gym registry.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--no_camera_view", action="store_true", default=False,
                    help="Tasks with a head camera model (env cfg head_camera, e.g. AppleReachVisible-v1): do not "
                         "draw its view frustum / seen apples, nor (with --video) paste its image into the video.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# register this repo's gym tasks (AppleReach-v1, AppleReach-Play-v1, ...) — must
# happen before the @hydra_task_config decorator below resolves --task against the
# gym registry. Mirrors the same sys.path pattern scripts/run_env.py already uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs  # noqa: E402

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # head camera model (AppleReachVisible-v1): its own image as a sensor, for the video
    head_camera = None if args_cli.no_camera_view else getattr(env_cfg, "head_camera", None)
    head_sensor = None
    if head_camera is not None and (args_cli.video or args_cli.enable_cameras):
        from envs.mdp.visible_reach import head_camera_sensor_cfg

        head_sensor = "head_camera_image"
        setattr(env_cfg.scene, head_sensor, head_camera_sensor_cfg(head_camera))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # what the head camera sees: frustum + seen (green) / unseen (magenta) apples in the scene,
    # perceived target (blue), and its image top-right in the video
    if head_camera is not None:
        from envs.mdp.visible_reach import camera_view_wrapper

        env = camera_view_wrapper(env, head_camera, sensor_name=head_sensor, scale=1)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # ─────────────────────────────────────────────────────────────────────
    #  DIAGNOSTICS — palm-axis probe, per-episode pose readout, reset check
    #
    #  (A) prints the local X/Y/Z axes of right_hand_palm_link in world
    #      coordinates for the CURRENT pose, so you can read off which local
    #      axis is the palm normal and set PALM_NORMAL_AXIS in
    #      apple_grasp_env_cfg.py accordingly (whichever axis has world-Z
    #      component nearest +1 in a palm-up pose; negate if nearest -1).
    #  (B) every 2s, prints the 7 arm joint angles + palm-to-target distance,
    #      so you can read the collapsed/bad pose off a live checkpoint and
    #      fill in RewardsCfg.posture_box's lower/upper bounds.
    #  (C) on every episode reset, prints q and |dq| right after the reset —
    #      expect q ~= default +-0.20 rad, |dq| < 0.05. Anything else means
    #      the problem is in default_joint_pos, not in the reset event.
    #
    #  NOTE: AppleGraspEnvCfg_PLAY disables both reset event terms and sets
    #  episode_length_s = 1e6, so `dones` will essentially never fire here
    #  and (C) will not print. To exercise the reset path, run this script
    #  against AppleReach-v1 (the non-Play task) instead, or temporarily
    #  re-enable the reset events in AppleGraspEnvCfg_PLAY.__post_init__.
    # ─────────────────────────────────────────────────────────────────────
    from isaaclab.utils.math import matrix_from_quat as _mfq

    _ARM_NAMES = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    _robot = env.unwrapped.scene["robot"]
    _apple = env.unwrapped.scene["apple"]
    _jids = [_robot.data.joint_names.index(n) for n in _ARM_NAMES]
    _palm_bid = _robot.data.body_names.index("right_hand_palm_link")

    def _fmt(v):
        return [round(float(x), 3) for x in v]

    def _probe_palm_axes():
        q = _robot.data.body_quat_w[0:1, _palm_bid, :]
        R = _mfq(q)[0]  # [3,3], columns are local X/Y/Z axes in world frame
        print("\n─── PALM AXIS PROBE (local axis -> world vector) ───")
        for name, i in (("local X", 0), ("local Y", 1), ("local Z", 2)):
            ax = R[:, i]
            print(f"  {name} -> world {_fmt(ax)}    (world-Z component = {float(ax[2]): .3f})")
        print("  Pick the axis whose world-Z component is nearest +1 in a")
        print("  palm-up pose; negate it if it is nearest -1.")
        print("────────────────────────────────────────────────────\n")

    _probe_palm_axes()
    print("default arm q :", _fmt(_robot.data.default_joint_pos[0, _jids]))
    print("apple world   :", _fmt(_apple.data.root_pos_w[0]))
    print("palm  world   :", _fmt(_robot.data.body_pos_w[0, _palm_bid]))
    print("SANITY: apple z - 1.085 (shoulder) should lie in REACH_TARGET_DZ_RANGE")
    print("        from params/env.yaml of the run you are loading.\n")

    _step_i = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            _step_i += 1

            # (B) periodic pose / distance readout
            if _step_i % 100 == 0:
                _q = _robot.data.joint_pos[0, _jids]
                _dq = _robot.data.joint_vel[0, _jids]
                _palm = _robot.data.body_pos_w[0, _palm_bid]
                _tgt = _apple.data.root_pos_w[0].clone()
                _tgt[2] -= 0.08
                _d = float(torch.norm(_tgt - _palm))

                _Rp = _mfq(_robot.data.body_quat_w[0:1, _palm_bid, :])[0]
                # change the column index below (0/1/2) to match your
                # PALM_NORMAL_AXIS once the probe above tells you which one it is
                _up = float(_Rp[:, 2][2])

                print(f"[{_step_i:5d}] d={_d*100:6.1f} cm  palm_up_cos={_up:+.2f}  "
                      f"|dq|={float(torch.norm(_dq)):5.2f}")
                print(f"         q  = {_fmt(_q)}")
                print(f"         dq = {_fmt(_dq)}")

            # (C) reset verification
            if dones.any():
                _q = _robot.data.joint_pos[0, _jids]
                _dq = _robot.data.joint_vel[0, _jids]
                print(f"  RESET -> q={_fmt(_q)}  |dq|={float(torch.norm(_dq)):.4f}")

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()