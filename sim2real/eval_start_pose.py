#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Does the reach policy reach as well from a given arm START pose? AppleReachSpindle-v3
with the pinned policy, the arm reset EXACTLY to default + --offset (no jitter, waist
at 0), --episodes per env; prints the share of episodes that end in the task's own
reach_success termination, how fast, and how many knock the target off.

    ~/miniconda3/envs/env_isaaclab/bin/python sim2real/eval_start_pose.py --headless \
        --offset -0.15 -0.15 -0.15 0 0 0 0
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--offset", type=float, nargs=7, default=[0.0] * 7, help="rad, added to the default arm pose")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--policy", default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.envs.mdp as mdp  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401,E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_reach_spindle_env_cfg import AppleReachSpindleV3EnvCfg  # noqa: E402

cfg = AppleReachSpindleV3EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.events.reset_arm_jitter = None
cfg.events.reset_waist_jitter = None
for j, (name, off) in enumerate(zip(RIGHT_ARM_JOINTS, args.offset)):
    setattr(cfg.events, f"start_pose_{j}", EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[name]),
                "position_range": (off, off), "velocity_range": (0.0, 0.0)}))
env = gym.make("AppleReachSpindle-v3", cfg=cfg).unwrapped
pol = torch.jit.load(args.policy or os.path.join(REPO, "envs/agents/best_weights/grasp_handoff_reach_spindle.pt"),
                     map_location=env.device).eval()
tm = env.termination_manager
names = list(tm.active_terms)
N = env.num_envs
done_n = np.zeros(N, int)
outcome = {n: 0 for n in names}
t_succ = []
obs = env.reset()[0]["policy"]
steps = 0
while done_n.min() < args.episodes:
    with torch.no_grad():
        a = pol(obs)
    ep_len = env.episode_length_buf.clone()
    obs, _, term, trunc, _ = env.step(a)
    obs = obs["policy"]
    steps += 1
    dn = (term | trunc).cpu().numpy()
    for i in np.flatnonzero(dn):
        if done_n[i] >= args.episodes:
            continue
        done_n[i] += 1
        for n in names:
            if bool(tm.get_term(n)[i]):
                outcome[n] += 1
                if n == "reach_success":
                    t_succ.append(float(ep_len[i] + 1) * env.step_dt)
                break
tot = N * args.episodes
print(f"[start pose] offset {args.offset}: {tot} episodes -> " +
      ", ".join(f"{n} {100 * v / tot:.0f}%" for n, v in outcome.items() if v) +
      (f"; time to success median {np.median(t_succ):.2f} s, p90 {np.percentile(t_succ, 90):.2f} s" if t_succ else ""),
      flush=True)
env.close()
