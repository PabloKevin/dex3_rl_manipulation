#!/usr/bin/env python3
"""Feasibility probe for the side-on reach pose (palm facing the apple
horizontally from the robot's side, fingers up).

Spawns the reach env with a WIDER apple-placement envelope than training
uses, drives the right arm with damped-least-squares IK (through the env's
own rate-limited action term, so joint limits / PD tracking are real) toward
the full target pose -- palm reference point at the grasp target, palm
normal along the approach direction, fingers along world +Z -- and reports
where that pose is actually achievable, binned by target height above the
shoulder, distance and azimuth.

Local IK restarted from several arm seeds per apple (the default pose plus
random ones), keeping the best; a "miss" can still be a local minimum
rather than true infeasibility, so read it as a lower bound on the
reachable set. Used to set REACH_MIN/REACH_MAX/REACH_TARGET_DZ_RANGE/
REACH_AZIMUTH_* in envs/apple_grasp_env_cfg.py.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/probe_side_reach.py --headless --num_envs 256
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--rounds", type=int, default=4, help="Resets (fresh apple samples) to run.")
parser.add_argument("--seeds", type=int, default=4, help="Arm seeds per apple sample (first = default pose).")
parser.add_argument("--steps", type=int, default=250, help="IK iterations per seed.")
parser.add_argument("--fingers_weight", type=float, default=1.0,
                    help="IK weight of fingers-up relative to palm-facing; < 1 lets the fingers tilt where needed.")
parser.add_argument("--tag", type=str, default="", help="Suffix for the saved .npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import (  # noqa: E402
    AppleGraspEnvCfg, GRASP_STANDOFF, PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS, RIGHT_ARM_JOINTS,
)
from envs.mdp import reach_rewards as rr  # noqa: E402

# Sweep envelope (wider than training on purpose).
SWEEP = dict(reach_min=0.18, reach_max=0.50, reach_azimuth_center_deg=-40.0,
             reach_azimuth_span_deg=110.0, target_dz_range=(-0.35, 0.30))
OK_POS, OK_FACE, OK_FINGERS = 0.02, 0.20, 0.785   # fingers: FINGERS_UP_TOL_RAD


def main():
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    cfg.events.randomize_tree.params.update(SWEEP)
    env = ManagerBasedRLEnv(cfg)

    robot = env.scene["robot"]
    palm_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
    palm_cfg.resolve(env.scene)
    apple_cfg = SceneEntityCfg("apple")
    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    term = env.action_manager.get_term("right_arm")
    term_ids = [term._joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    palm_id = palm_cfg.body_ids[0]
    jac_id = palm_id - 1 if robot.is_fixed_base else palm_id
    jac_cols = arm_ids if robot.is_fixed_base else [i + 6 for i in arm_ids]
    off_local = torch.tensor(PALM_LOCAL_OFFSET, device=env.device)
    n_local = torch.tensor(PALM_NORMAL_AXIS, device=env.device)
    f_local = torch.tensor(rr.PALM_FINGER_AXIS, device=env.device)
    lo = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
    default = robot.data.default_joint_pos[:, arm_ids]
    shoulder_roll = RIGHT_ARM_JOINTS.index("right_shoulder_roll_joint")

    state = {"q": default.clone()}

    def ik_step():
        quat = robot.data.body_quat_w[:, palm_id]
        off_w = matrix_from_quat(quat) @ off_local
        ref = robot.data.body_pos_w[:, palm_id] + off_w

        d = rr._approach_dir(env, apple_cfg)
        z = torch.zeros_like(d)
        z[:, 2] = 1.0
        R = matrix_from_quat(quat)
        tgt = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF)
        pos_err = tgt - ref
        # Two separate axis-alignment errors (rotation vectors taking the
        # current axis onto the desired one): palm normal -> approach dir,
        # fingers -> world up, weighted independently.
        n_err = torch.cross(R @ n_local, d, dim=-1)
        f_err = torch.cross(R @ f_local, z, dim=-1)
        # Clamp the task-space error: DLS is badly conditioned on large errors.
        pos_err = pos_err * (0.05 / pos_err.norm(dim=-1, keepdim=True).clamp(min=0.05))

        J = robot.root_physx_view.get_jacobians()[:, jac_id, :, :][:, :, jac_cols].clone()
        # Linear part at the reference point, not the link origin.
        ox, oy, oz = off_w[:, 0], off_w[:, 1], off_w[:, 2]
        zero = torch.zeros_like(ox)
        skew = torch.stack([torch.stack([zero, -oz, oy], -1),
                            torch.stack([oz, zero, -ox], -1),
                            torch.stack([-oy, ox, zero], -1)], -2)
        J[:, :3] -= skew @ J[:, 3:]
        wf = args_cli.fingers_weight
        J = torch.cat([J[:, :3], 0.3 * J[:, 3:], 0.3 * wf * J[:, 3:]], dim=1)
        e = torch.cat([pos_err, 0.3 * n_err, 0.3 * wf * f_err], dim=-1).unsqueeze(-1)
        JJt = J @ J.transpose(1, 2) + (0.03 ** 2) * torch.eye(9, device=env.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e)).squeeze(-1).clamp(-0.1, 0.1)
        # Kinematic IK: teleport the arm to the new q and take one physics
        # step only to refresh body poses / Jacobians. (Driving it through
        # the rate-limited PD instead winds up and oscillates: the arm lags
        # the command the error is integrated on.)
        state["q"] = torch.clamp(state["q"] + dq, lo, hi)
        set_arm(state["q"])
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    def set_arm(q0):
        jp = robot.data.joint_pos.clone()
        jp[:, arm_ids] = q0
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp)
        term._prev_target[:, term_ids] = q0
        term._prev_vel[:] = 0.0

    def measure():
        d_pos = rr._palm_target_dist(env, palm_cfg, apple_cfg, GRASP_STANDOFF, PALM_LOCAL_OFFSET)
        face = rr._palm_normal_angle(env, palm_cfg, PALM_NORMAL_AXIS, apple_cfg)
        fing = rr._fingers_up_angle(env, palm_cfg)
        q = robot.data.joint_pos[:, arm_ids]
        torso = rr.penalty_torso_self_collision(env, SceneEntityCfg("robot"), margin=0.02, ref=0.05)
        reached = (d_pos < OK_POS) & (face < OK_FACE) & (fing < OK_FINGERS)
        clean = reached & (torso < 1e-3) & (q[:, shoulder_roll] < 0.35)
        score = d_pos / OK_POS + face / OK_FACE + fing / OK_FINGERS - 10.0 * clean.float() - 5.0 * reached.float()
        return score, [d_pos, face, fing, torso, q[:, shoulder_roll],
                       torch.minimum(q - lo, hi - q).min(-1).values], q.clone()

    rows, best_q = [], []
    for rnd in range(args_cli.rounds):
        env.reset()
        best_score = torch.full((env.num_envs,), float("inf"), device=env.device)
        best_cols, bq = None, None
        for seed in range(args_cli.seeds):
            q0 = default.clone() if seed == 0 else lo + (hi - lo) * (0.15 + 0.7 * torch.rand_like(default))
            state["q"] = q0
            set_arm(q0)
            for _ in range(args_cli.steps):
                ik_step()
            score, cols, q = measure()
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            if best_cols is None:
                best_cols, bq = cols, q
            else:
                best_cols = [torch.where(better, c, b) for c, b in zip(cols, best_cols)]
                bq = torch.where(better.unsqueeze(-1), q, bq)
        sh = env.scene.env_origins + env._shoulder_local_xyz
        tgt = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF) - sh
        r3 = tgt.norm(dim=-1)
        az = torch.rad2deg(torch.atan2(tgt[:, 1], tgt[:, 0]))
        rows.append(torch.stack([r3, az, tgt[:, 2]] + best_cols, -1).cpu().numpy())
        best_q.append(bq.cpu().numpy())
        print(f"round {rnd + 1}/{args_cli.rounds} done")

    a = np.concatenate(rows)
    r3, az, dz, dpos, face, fing, torso, sroll, lim = a.T
    ok = (dpos < OK_POS) & (face < OK_FACE) & (fing < OK_FINGERS)
    ok_clean = ok & (torso < 1e-3) & (sroll < 0.35)
    print(f"\nN={len(a)}  pose reached: {ok.mean():.2f}   reached w/o torso hit & shoulder_roll<0.35: {ok_clean.mean():.2f}")
    print(f"median residuals: pos {np.median(dpos)*100:.1f} cm  face {np.degrees(np.median(face)):.0f} deg  fingers {np.degrees(np.median(fing)):.0f} deg")

    def table(name, x, edges):
        print(f"\n{name:>14s} |   n  reached  clean  pos_cm  face_deg  fing_deg")
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (x >= lo_) & (x < hi_)
            if m.sum() == 0:
                continue
            print(f"{lo_:6.2f}..{hi_:6.2f} | {m.sum():4d}   {ok[m].mean():.2f}    {ok_clean[m].mean():.2f}"
                  f"   {np.median(dpos[m])*100:5.1f}   {np.degrees(np.median(face[m])):6.0f}    {np.degrees(np.median(fing[m])):6.0f}")

    table("dz (m)", dz, np.arange(-0.35, 0.31, 0.05))
    table("r3d (m)", r3, np.arange(0.15, 0.56, 0.05))
    table("azimuth (deg)", az, np.arange(-100, 21, 10))

    # 2D: dz x r3d clean-reach fraction
    dz_e, r_e = np.arange(-0.35, 0.31, 0.05), np.arange(0.15, 0.56, 0.05)
    print("\nclean fraction, rows dz, cols r3d " + " ".join(f"{x:5.2f}" for x in r_e[:-1]))
    for i in range(len(dz_e) - 1):
        line = []
        for j in range(len(r_e) - 1):
            m = (dz >= dz_e[i]) & (dz < dz_e[i + 1]) & (r3 >= r_e[j]) & (r3 < r_e[j + 1])
            line.append(f"{ok_clean[m].mean():5.2f}" if m.sum() >= 3 else "    .")
        print(f"   {dz_e[i]:+.2f}..{dz_e[i+1]:+.2f}              " + " ".join(line))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"probe_side_reach{args_cli.tag}.npz")
    np.savez(out, data=a, arm_q=np.concatenate(best_q), columns=np.array(["r3d", "az", "dz", "pos", "face", "fingers", "torso", "shoulder_roll", "limit_margin"]))
    print(f"\nsaved {os.path.abspath(out)}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
