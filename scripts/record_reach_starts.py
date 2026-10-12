#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Record a START BANK for the harvest-path goal policy (AppleArmGoal-v1): the
arm poses where the trained tree reach really leaves the hand, instead of
random library poses.

Runs AppleReachTree-v1 (multi-apple tree, random target) with the frozen
tree reach policy and reads the right arm's joints at the grasp hand-off
time (--at_s, the grasp task's REACH_HANDOFF_TIME_S). A pose is kept only
when the reach really got there (grasp pocket within --max_miss of the
apple) and the arm + hand AND an apple held at the pocket clear the body and
the harvest basket by the self-collision margin -- the same tests as
scripts/build_harvest_pose_library.py, so every start is collision-free.

Output: envs/assets/reach_end_poses.npz (q [M, 7] in RIGHT_ARM_JOINTS order,
joint_names, palm_local, palm_normal). envs/mdp/arm_goal.reset_arm_goal
starts 70% of its pick-like episodes from it when the file exists.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/record_reach_starts.py --headless --count 3000

--spindle: the v3 waist reach (AppleReachSpindle-v3, pinned best_weights/
grasp_handoff_reach_spindle.pt) on the spindle tree; also records the waist
yaw ("waist"); default output envs/assets/reach_end_poses_waist.npz (the
AppleArmGoalWaist-v1 start bank).
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--count", type=int, default=3000)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--at_s", type=float, default=2.5, help="seconds after the episode start (the grasp hand-off time)")
p.add_argument("--max_miss", type=float, default=0.05, help="grasp pocket at most this far from the apple centre (m)")
p.add_argument("--policy", type=str, default=None, help="exported reach policy (default best_weights/grasp_handoff_reach_tree.pt)")
p.add_argument("--apple_radius", type=float, default=0.048)
p.add_argument("--out", type=str, default=None)
p.add_argument("--spindle", action="store_true", help="v3 waist reach on the spindle tree; records the waist yaw too")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401
from envs.apple_reach_tree_env_cfg import AppleReachTreeEnvCfg
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS, PALM_NORMAL_AXIS
from envs.apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL
from envs.mdp import basket as bk
from envs.mdp import self_collision as scm
from isaaclab.utils.math import quat_apply

if args.spindle:
    from envs.apple_reach_spindle_env_cfg import AppleReachSpindleV3EnvCfg
    cfg = AppleReachSpindleV3EnvCfg()
    cfg.events.reset_arm_jitter = None             # the robot starts every reach from home
    cfg.events.reset_waist_jitter = None
else:
    cfg = AppleReachTreeEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.at_s + 0.1            # one pose per episode
for name in list(vars(cfg.terminations)):
    if not name.startswith("_") and name != "time_out":
        setattr(cfg.terminations, name, None)
env = gym.make("AppleReachSpindle-v3" if args.spindle else "AppleReachTree-v1", cfg=cfg).unwrapped
default_pol = "grasp_handoff_reach_spindle.pt" if args.spindle else "grasp_handoff_reach_tree.pt"
pol = torch.jit.load(args.policy or os.path.join(REPO, "envs/agents/best_weights", default_pol),
                     map_location=env.device).eval()
waist_id = robot_waist = None
robot, apple = env.scene["robot"], env.scene["apple"]
arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
waist_id = robot.joint_names.index("waist_yaw_joint")
palm = robot.data.body_names.index("right_hand_palm_link")
at_step = int(round(args.at_s / env.step_dt))
c_local = torch.tensor(GRASP_CENTER_LOCAL, device=env.device)
n_axis = torch.tensor(PALM_NORMAL_AXIS, device=env.device)
margin = scm.DEFAULT_MARGIN

obs, _ = env.reset()
kept, seen = [], 0
while len(kept) < args.count:
    with torch.inference_mode():
        obs, *_ = env.step(pol(obs["policy"]))
        now = (env.episode_length_buf == at_step).nonzero(as_tuple=True)[0]
        if len(now) == 0:
            continue
        seen += len(now)
        q = robot.data.joint_pos[now][:, arm_ids]
        pq, pp = robot.data.body_quat_w[now, palm], robot.data.body_pos_w[now, palm]
        pocket = pp + quat_apply(pq, c_local.expand(len(now), 3))
        normal = quat_apply(pq, n_axis.expand(len(now), 3))
        arrived = (pocket - apple.data.root_pos_w[now]).norm(dim=-1) < args.max_miss
        sets = scm._build(env) if not hasattr(env, "_self_collision_spheres") else env._self_collision_spheres
        env._self_collision_spheres = sets
        arm_ok = scm.self_collision_depth(env, margin)[now].amax(-1) <= 0.0
        body = scm._world_spheres(env, sets["body"])[now]
        apple_ok = ((body - pocket.unsqueeze(1)).norm(dim=-1) - sets["body"]["r"]).amin(-1) >= args.apple_radius + margin
        o = env.scene.env_origins[now]
        arm_c = scm._world_spheres(env, sets["arm"])[now] - o.unsqueeze(1)
        basket_ok = (bk.basket_intrusion(arm_c, sets["arm"]["r"], margin).amax(-1) <= 0.0) & \
            (bk.basket_intrusion((pocket - o).unsqueeze(1), torch.tensor([args.apple_radius], device=env.device),
                                 margin)[:, 0] <= 0.0)
        good = arrived & arm_ok & apple_ok & basket_ok
        for j in good.nonzero(as_tuple=True)[0].tolist():
            kept.append((q[j].cpu().numpy(), (pp[j] - o[j]).cpu().numpy(), normal[j].cpu().numpy(),
                         float(robot.data.joint_pos[now[j], waist_id])))
        print(f"[record] {len(kept)}/{args.count} kept of {seen} reach ends "
              f"(arrived {int(arrived.sum())}/{len(now)}, clear {int((arm_ok & apple_ok & basket_ok).sum())}/{len(now)})",
              flush=True)
kept = kept[:args.count]
out = args.out or os.path.join(REPO, "envs", "assets", "reach_end_poses_waist.npz" if args.spindle else "reach_end_poses.npz")
extra = {"waist": np.array([k[3] for k in kept], dtype=np.float32)} if args.spindle else {}
np.savez(out, q=np.stack([k[0] for k in kept]), palm_local=np.stack([k[1] for k in kept]),
         palm_normal=np.stack([k[2] for k in kept]), joint_names=np.array(RIGHT_ARM_JOINTS), **extra)
print(f"[record] {len(kept)} poses ({100 * len(kept) / max(seen, 1):.0f}% of reach ends) -> {out}", flush=True)
os._exit(0)
