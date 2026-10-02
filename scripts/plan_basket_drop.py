#!/usr/bin/env python3
"""Plan the fixed HOME -> basket drop path for the harvest test bench.

Kinematic search (arm teleported, one physics step to refresh poses):
  1. random arm configurations (inside the soft limits) whose arm, hand AND
     held apple clear the body (envs/mdp/self_collision.py) and the basket
     (envs/mdp/basket.py), shoulder_roll inside the posture box;
  2. DROP poses among them: apple (grasp pocket) over the basket axis,
     --drop_above m above the rim (apple bottom), palm facing DOWN;
     PRE poses: same, --pre_above m higher (palm roughly down);
  3. chains HOME -> PRE -> DROP (or HOME -> LIFT -> PRE -> DROP when no direct
     chain is clean), every joint-space interpolation point checked against
     body + basket; the shortest clean chain wins.
Output: envs/assets/basket_drop_path.npz -- waypoints [K, 7] in
RIGHT_ARM_JOINTS order (first = home), the apple position at each.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/plan_basket_drop.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=60)
parser.add_argument("--apple_radius", type=float, default=0.048, help="largest randomized apple")
parser.add_argument("--drop_above", type=float, nargs=2, default=[0.04, 0.09], help="apple bottom above the rim (m)")
parser.add_argument("--pre_above", type=float, nargs=2, default=[0.10, 0.18], help="PRE apple higher than DROP by (m)")
parser.add_argument("--xy_tol", type=float, default=0.03)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL  # noqa: E402
from envs.mdp import basket as bk  # noqa: E402
from envs.mdp import self_collision as scm  # noqa: E402


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
    N, dev = env.num_envs, env.device
    robot = env.scene["robot"]
    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    pid = robot.data.body_names.index("right_hand_palm_link")
    lo = robot.data.soft_joint_pos_limits[0, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[0, arm_ids, 1]
    home = robot.data.default_joint_pos[0, arm_ids].clone()
    roll_i = RIGHT_ARM_JOINTS.index("right_shoulder_roll_joint")
    pocket = torch.tensor(GRASP_CENTER_LOCAL, device=dev).expand(N, 3)
    margin = scm.DEFAULT_MARGIN
    ra = args_cli.apple_radius

    def check(q):
        """q [N, 7] -> ok [N], apple local [N, 3], palm normal z [N]."""
        jp = robot.data.joint_pos.clone(); jp[:, arm_ids] = q
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone(); tgt[:, arm_ids] = q
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()          # else PhysX keeps the old PD targets and pulls the arm away
        env.sim.step(render=False); env.scene.update(env.physics_dt)
        body_ok = scm.self_collision_depth(env, margin).amax(-1) <= 0.0
        sets = env._self_collision_spheres
        arm_c = scm._world_spheres(env, sets["arm"]) - env.scene.env_origins.unsqueeze(1)
        pq, pp = robot.data.body_quat_w[:, pid], robot.data.body_pos_w[:, pid]
        apple = pp + quat_apply(pq, pocket) - env.scene.env_origins
        bodyc = scm._world_spheres(env, sets["body"]) - env.scene.env_origins.unsqueeze(1)
        apple_body = ((bodyc - apple.unsqueeze(1)).norm(dim=-1) - sets["body"]["r"].view(1, -1)).amin(-1) >= ra + margin
        basket_ok = (bk.basket_intrusion(arm_c, sets["arm"]["r"], margin).amax(-1) <= 0.0) \
            & (bk.basket_intrusion(apple.unsqueeze(1), torch.tensor([ra], device=dev), margin)[:, 0] <= 0.0)
        nz = quat_apply(pq, torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(N, 3))[:, 2]
        settled = (robot.data.joint_pos[:, arm_ids] - q).abs().amax(-1) < 0.01   # pose really is q
        return body_ok & apple_body & basket_ok & settled & (q[:, roll_i] < 0.35), apple, nz

    # 1. random safe configurations
    Q, A, NZ = [], [], []
    for b in range(args_cli.batches):
        q = lo + (hi - lo) * (0.025 + 0.95 * torch.rand(N, 7, device=dev))
        ok, apple, nz = check(q)
        Q.append(q[ok]); A.append(apple[ok]); NZ.append(nz[ok])
    Q, A, NZ = torch.cat(Q), torch.cat(A), torch.cat(NZ)
    print(f"safe configurations: {len(Q)}", flush=True)

    rad = torch.hypot(A[:, 0] - bk.CENTER_XY[0], A[:, 1] - bk.CENTER_XY[1])
    drop_z = (bk.RIM_Z + ra + args_cli.drop_above[0], bk.RIM_Z + ra + args_cli.drop_above[1])
    D = ((rad < args_cli.xy_tol) & (A[:, 2] > drop_z[0]) & (A[:, 2] < drop_z[1]) & (NZ < -0.8)).nonzero(as_tuple=True)[0]
    P_mask = (rad < args_cli.xy_tol + 0.01) & (NZ < -0.6)
    print(f"DROP candidates {len(D)}, PRE pool {int(P_mask.sum())}", flush=True)
    if len(D) == 0:
        sys.exit("no palm-down DROP pose found -- more --batches, or move/lower the basket")

    # 2. chains HOME -> PRE -> DROP: for each drop, the 3 nearest PRE poses at the right height
    chains = []
    for d in D.tolist():
        dz = A[:, 2] - A[d, 2]
        cand = (P_mask & (dz > args_cli.pre_above[0]) & (dz < args_cli.pre_above[1])).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            continue
        near = cand[(Q[cand] - Q[d]).norm(dim=-1).argsort()[:3]]
        chains += [[home, Q[p], Q[d]] for p in near.tolist()]
    print(f"HOME->PRE->DROP chains to validate: {len(chains)}", flush=True)

    def validate(chain_list, pts_per_rad=25):
        """Every interpolation point of every chain against body + basket.
        Returns a bool per chain."""
        good = torch.ones(len(chain_list), dtype=torch.bool, device=dev)
        for s in range(0, len(chain_list), N):
            batch = chain_list[s:s + N]
            n = len(batch)
            segs = len(batch[0]) - 1
            for g in range(segs):
                a0 = torch.stack([c[g] for c in batch]); a1 = torch.stack([c[g + 1] for c in batch])
                K = max(10, int(pts_per_rad * float((a1 - a0).norm(dim=-1).max())))
                for k in range(K + 1):
                    q = a0 + (a1 - a0) * (k / K)
                    if n < N:
                        q = torch.cat([q, home.expand(N - n, 7)])
                    ok, _, _ = check(q)
                    good[s:s + n] &= ok[:n]
        return good

    best = None
    if chains:
        good = validate(chains)
        print(f"clean direct chains: {int(good.sum())}/{len(chains)}", flush=True)
        if bool(good.any()):
            lens = torch.tensor([float(sum((c[i + 1] - c[i]).norm() for i in range(len(c) - 1))) for c in chains], device=dev)
            lens[~good] = float("inf")
            best = chains[int(lens.argmin())]
    if best is None:
        # 3. with a LIFT waypoint: apple well above the rim, beside the basket on the robot's side
        L = ((A[:, 2] > bk.RIM_Z + 0.20) & (rad > bk.STAND_RADIUS + 0.08) & (A[:, 1] > bk.CENTER_XY[1])).nonzero(as_tuple=True)[0]
        print(f"no clean direct chain; trying LIFT waypoints ({len(L)} candidates)", flush=True)
        chains_l = []
        for c in chains[:300]:
            for l_ in L[(Q[L] - c[1]).norm(dim=-1).argsort()[:3]].tolist():
                chains_l.append([home, Q[l_], c[1], c[2]])
        if chains_l:
            good = validate(chains_l)
            print(f"clean lift chains: {int(good.sum())}/{len(chains_l)}", flush=True)
            if bool(good.any()):
                lens = torch.tensor([float(sum((c[i + 1] - c[i]).norm() for i in range(len(c) - 1))) for c in chains_l], device=dev)
                lens[~good] = float("inf")
                best = chains_l[int(lens.argmin())]
    if best is None:
        sys.exit("no clean path found -- try more --batches")

    W = torch.stack(best)
    apples = []
    for w in W:
        _, apple, nz = check(w.expand(N, 7))
        apples.append(apple[0].cpu().numpy())
    print("waypoints (RIGHT_ARM_JOINTS order) and apple position:")
    for w, a in zip(W.cpu().numpy(), apples):
        print("  ", np.round(w, 3), " apple", np.round(a, 3))
    out = args_cli.out or os.path.join(REPO, "envs", "assets", "basket_drop_path.npz")
    np.savez(out, waypoints=W.cpu().numpy(), apple=np.array(apples), joint_names=np.array(RIGHT_ARM_JOINTS),
             rim_z=bk.RIM_Z, center_xy=np.array(bk.CENTER_XY))
    print(f"saved {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
