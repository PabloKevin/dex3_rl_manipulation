#!/usr/bin/env python3
"""Build the start / goal pose library of AppleArmPose-v1
(envs/apple_arm_pose_env_cfg.py, envs/mdp/arm_pose.py).

Random right-arm joint vectors (uniform inside the soft limits shrunk by
--limit_frac, and inside the training's posture box) are teleported in the
task's own scene -- waist pitch 0, the reach's cupped hand
(REACH_HAND_PRESHAPE) -- and checked with the full-body sphere model
(envs/mdp/self_collision.py) at every waist yaw of --yaw_grid: the arm hangs
on the torso, so its pose in the torso frame is the same at every yaw, but
the legs move relative to it. Stored per pose: the smallest sphere gap to
the body at each grid yaw (m, negative = overlap), so the env decides which
clearance a start or a goal needs. Poses overlapping at waist yaw 0, or
that do not settle at the sampled joint values, are dropped.

Output (default envs/assets/arm_pose_library.npz):
  q [M, 7]               RIGHT_ARM_JOINTS order
  clearance [M, Y]       min arm/hand-to-body sphere gap at each yaw_grid value
  yaw_grid [Y]           rad
  pocket_torso [M, 3]    grasp pocket (GRASP_CENTER_LOCAL), torso_link frame
  normal_torso [M, 3]    palm normal (palm +y), torso_link frame
  finger_torso [M, 3]    finger direction (palm +x), torso_link frame
  shoulder_torso [3]     right_shoulder_pitch_link origin, torso_link frame
  joint_names [7]

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/build_arm_pose_library.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--batches", type=int, default=60)
parser.add_argument("--limit_frac", type=float, default=0.95)
parser.add_argument("--yaw_grid", type=float, nargs="+", default=[-45, -35, -25, -15, -5, 5, 15],
                    help="waist yaws (deg) the clearance is measured at; brackets between neighbours are sampled")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_arm_pose_env_cfg import AppleArmPoseEnvCfg, POCKET, POSTURE_LOWER, POSTURE_UPPER  # noqa: E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_reach_spindle_env_cfg import REACH_HAND_PRESHAPE  # noqa: E402
from envs.mdp import self_collision as scm  # noqa: E402


def min_gap(env) -> torch.Tensor:
    """[N] smallest arm/hand-to-body sphere gap (m; < 0 = overlap), masked pairs skipped."""
    scm.self_collision_depth(env)                      # builds the sphere sets once
    sets = env._self_collision_spheres
    a = scm._world_spheres(env, sets["arm"])
    b = scm._world_spheres(env, sets["body"])
    gap = torch.cdist(a, b) - sets["arm"]["r"].view(1, -1, 1) - sets["body"]["r"].view(1, 1, -1)
    gap = torch.where(sets["mask"].unsqueeze(0), gap, torch.full_like(gap, 1.0))
    return gap.amin(dim=(-1, -2))


def main():
    cfg = AppleArmPoseEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.events.reset_arm_pose = None           # needs the library this script makes
    env = ManagerBasedRLEnv(cfg)
    env.reset()
    robot = env.scene["robot"]
    dev = env.device
    N = env.num_envs
    arm = torch.tensor([robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS], device=dev)
    hand = torch.tensor([robot.joint_names.index(n) for n in REACH_HAND_PRESHAPE], device=dev)
    hand_v = torch.tensor(list(REACH_HAND_PRESHAPE.values()), device=dev)
    waist = robot.joint_names.index("waist_yaw_joint")
    palm = robot.data.body_names.index("right_hand_palm_link")
    torso = robot.data.body_names.index("torso_link")
    shoulder = robot.data.body_names.index("right_shoulder_pitch_link")
    lo = torch.maximum(robot.data.soft_joint_pos_limits[0, arm, 0], torch.tensor(POSTURE_LOWER, device=dev))
    hi = torch.minimum(robot.data.soft_joint_pos_limits[0, arm, 1], torch.tensor(POSTURE_UPPER, device=dev))
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * args_cli.limit_frac
    yaws = [math.radians(y) for y in args_cli.yaw_grid]
    i0 = min(range(len(yaws)), key=lambda k: abs(yaws[k]))     # the grid yaw nearest 0
    pocket_l = torch.tensor(POCKET, device=dev).expand(N, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3)

    def teleport(q, yaw):
        jp = robot.data.joint_pos.clone()
        jp[:, arm] = q
        jp[:, hand] = hand_v
        jp[:, waist] = yaw
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, arm] = q
        tgt[:, hand] = hand_v
        tgt[:, waist] = yaw
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()          # else PhysX keeps the old PD targets and pulls the arm away
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    out = {k: [] for k in ("q", "clearance", "pocket", "normal", "finger")}
    shoulder_t = None
    for b in range(args_cli.batches):
        q = mid + half * (2 * torch.rand(N, 7, device=dev) - 1)
        clear = torch.zeros(N, len(yaws), device=dev)
        settled = torch.ones(N, dtype=torch.bool, device=dev)
        for k, y in enumerate(yaws):
            teleport(q, y)
            clear[:, k] = min_gap(env)
            settled &= (robot.data.joint_pos[:, arm] - q).abs().amax(-1) < 0.01
            if k == i0:
                tp, tq = robot.data.body_pos_w[:, torso], robot.data.body_quat_w[:, torso]
                pq, pp = robot.data.body_quat_w[:, palm], robot.data.body_pos_w[:, palm]
                pocket = quat_apply_inverse(tq, pp + quat_apply(pq, pocket_l) - tp)
                normal = quat_apply_inverse(tq, quat_apply(pq, ey))
                finger = quat_apply_inverse(tq, quat_apply(pq, ex))
                if shoulder_t is None:
                    shoulder_t = quat_apply_inverse(tq[:1], robot.data.body_pos_w[:1, shoulder] - tp[:1])[0]
        ok = settled & (clear[:, i0] > 0.0)
        for key, v in (("q", q), ("clearance", clear), ("pocket", pocket), ("normal", normal), ("finger", finger)):
            out[key].append(v[ok].cpu())
        c = clear[ok]
        print(f"batch {b + 1}/{args_cli.batches}: kept {int(ok.sum())}/{N} (settled {settled.float().mean():.2f}); "
              f"clear >= 1.5 cm at all yaws {(c.amin(-1) >= 0.015).float().mean():.2f}, "
              f">= 3 cm {(c.amin(-1) >= 0.03).float().mean():.2f}", flush=True)

    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    rel = res["pocket"] - shoulder_t.cpu().numpy()
    n = res["normal"]
    tilt = np.degrees(np.arcsin(np.clip(n[:, 2], -1, 1)))
    head = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    print(f"\nlibrary: {len(res['q'])} poses; pocket - shoulder (torso frame) x p10/50/90 "
          f"{np.percentile(rel[:, 0], [10, 50, 90]).round(2)}, z {np.percentile(rel[:, 2], [10, 50, 90]).round(2)}; "
          f"palm tilt p10/50/90 {np.percentile(tilt, [10, 50, 90]).round(0)} deg, heading "
          f"{np.percentile(head, [10, 50, 90]).round(0)} deg", flush=True)
    path = args_cli.out or os.path.join(REPO, "envs", "assets", "arm_pose_library.npz")
    np.savez(path, q=res["q"], clearance=res["clearance"], yaw_grid=np.array(yaws, dtype=np.float32),
             pocket_torso=res["pocket"], normal_torso=res["normal"], finger_torso=res["finger"],
             shoulder_torso=shoulder_t.cpu().numpy(), joint_names=np.array(RIGHT_ARM_JOINTS))
    print(f"saved {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
