#!/usr/bin/env python3
"""Build a library of SAFE right-arm configurations for the harvest test
bench's random moves (scripts/harvest_test_bench.py).

Random arm joint vectors (inside the soft limits, shrunk by --limit_frac) are
applied kinematically (teleported, one physics step to refresh poses) and
kept only when:
  - the arm + hand clear the rest of the body by the training margin
    (envs/mdp/self_collision.py), AND an apple held at the grasp pocket
    (radius --apple_radius) clears the body by the same margin -- the held
    fruit collides with the robot in PhysX, so it must not swing into it;
  - the palm reference point is in front of the robot: forward of the
    shoulder by >= --min_forward, within --max_reach of it, and between
    --z_lo and --z_hi above/below it;
  - right_shoulder_roll stays inside the posture box (< 0.35 rad);
  - arm, hand and held apple clear the harvest basket (envs/mdp/basket.py);
  - the arm really is at the sampled pose after the checking step (the
    task apple is parked far away so nothing can shove it).
Palm orientation is left free on purpose: the bench should turn the fruit
every which way.

Output (default envs/assets/harvest_pose_library.npz): q [M, 7] in
RIGHT_ARM_JOINTS order, palm_local [M, 3] (palm reference point, env-local),
palm_normal [M, 3] (world == env frame; fixed base).

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/build_harvest_pose_library.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=40)
parser.add_argument("--limit_frac", type=float, default=0.85)
parser.add_argument("--apple_radius", type=float, default=0.048)
parser.add_argument("--min_forward", type=float, default=0.12)
parser.add_argument("--max_reach", type=float, default=0.45)
parser.add_argument("--z_lo", type=float, default=-0.35)
parser.add_argument("--z_hi", type=float, default=0.30)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, PALM_LOCAL_OFFSET, RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL  # noqa: E402
from envs.mdp import self_collision as scm  # noqa: E402
from envs.mdp import basket as bk  # noqa: E402
from envs.mdp import reach_rewards as rr  # noqa: E402


def main():
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    # Park the task's (kinematic) apple far away: random arm configurations
    # that pass through it get shoved out by PhysX within the one checking
    # step, and the check would then score a different pose than sampled.
    apple_obj = env.scene["apple"]
    far = apple_obj.data.default_root_state.clone()
    far[:, :3] = env.scene.env_origins + torch.tensor([50.0, 0.0, 1.0], device=env.device)
    far[:, 7:] = 0.0
    apple_obj.write_root_state_to_sim(far)
    robot = env.scene["robot"]
    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    palm_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"]); palm_cfg.resolve(env.scene)
    pid = palm_cfg.body_ids[0]
    lo = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * args_cli.limit_frac
    roll_i = RIGHT_ARM_JOINTS.index("right_shoulder_roll_joint")
    pocket = torch.tensor(GRASP_CENTER_LOCAL, device=env.device).expand(env.num_envs, 3)
    off = torch.tensor(PALM_LOCAL_OFFSET, device=env.device).expand(env.num_envs, 3)
    margin = scm.DEFAULT_MARGIN

    keep_q, keep_p, keep_n = [], [], []
    for b in range(args_cli.batches):
        q = mid + half * (2 * torch.rand_like(mid) - 1)
        jp = robot.data.joint_pos.clone(); jp[:, arm_ids] = q
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone(); tgt[:, arm_ids] = q
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()          # else PhysX keeps the old PD targets and pulls the arm away
        env.sim.step(render=False); env.scene.update(env.physics_dt)

        arm_ok = scm.self_collision_depth(env, margin).amax(-1) <= 0.0
        # held apple vs. body spheres
        sets = env._self_collision_spheres
        body = scm._world_spheres(env, sets["body"])                                  # [N, B, 3]
        pq, pp = robot.data.body_quat_w[:, pid], robot.data.body_pos_w[:, pid]
        apple_c = pp + quat_apply(pq, pocket)
        d = (body - apple_c.unsqueeze(1)).norm(dim=-1) - sets["body"]["r"].view(1, -1)
        apple_ok = d.amin(-1) >= args_cli.apple_radius + margin
        sh = env.scene.env_origins + env._shoulder_local_xyz
        ref = pp + quat_apply(pq, off)
        rel = ref - sh
        reach_ok = (rel[:, 0] >= args_cli.min_forward) & (rel.norm(dim=-1) <= args_cli.max_reach) \
            & (rel[:, 2] >= args_cli.z_lo) & (rel[:, 2] <= args_cli.z_hi)
        settled = (robot.data.joint_pos[:, arm_ids] - q).abs().amax(-1) < 0.01   # pose really is q
        # stay clear of the harvest basket too (envs/mdp/basket.py)
        arm_c = scm._world_spheres(env, sets["arm"]) - env.scene.env_origins.unsqueeze(1)
        basket_ok = (bk.basket_intrusion(arm_c, sets["arm"]["r"], margin).amax(-1) <= 0.0) & \
            (bk.basket_intrusion((apple_c - env.scene.env_origins).unsqueeze(1),
                                 torch.tensor([args_cli.apple_radius], device=env.device), margin)[:, 0] <= 0.0)
        ok = arm_ok & apple_ok & reach_ok & basket_ok & settled & (q[:, roll_i] < 0.35)
        keep_q.append(q[ok].cpu()); keep_p.append((ref - env.scene.env_origins)[ok].cpu())
        keep_n.append(rr._palm_normal_world(env, palm_cfg, (0.0, 1.0, 0.0))[ok].cpu())
        print(f"batch {b + 1}/{args_cli.batches}: kept {int(ok.sum())}/{env.num_envs} "
              f"(arm ok {arm_ok.float().mean():.2f}, apple ok {apple_ok.float().mean():.2f}, in front {reach_ok.float().mean():.2f})",
              flush=True)

    q = torch.cat(keep_q).numpy(); p = torch.cat(keep_p).numpy(); n = torch.cat(keep_n).numpy()
    tilt = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    print(f"\nlibrary: {len(q)} configurations; palm normal tilt from horizontal p10/50/90: "
          f"{np.percentile(tilt, [10, 50, 90]).round(0)} deg (+ = facing up)")
    out = args_cli.out or os.path.join(REPO, "envs", "assets", "harvest_pose_library.npz")
    np.savez(out, q=q, palm_local=p, palm_normal=n, joint_names=np.array(RIGHT_ARM_JOINTS))
    print(f"saved {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
