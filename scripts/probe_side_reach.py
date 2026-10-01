#!/usr/bin/env python3
"""Map the reach workspace: where can the right arm put the palm against the
apple (palm normal along the stem, palm reference point at the grasp
target) WITHOUT any self-collision?

Spawns the reach env with a wide apple-placement sweep, solves kinematic IK
(damped least squares, arm teleported each iteration -- driving it through
the PD instead lags and winds up) from several arm seeds per sample, keeps
the best solution, and calls a sample CLEAN when:
  - palm reference point within OK_POS of the grasp target, palm normal
    within OK_FACE of the stem direction;
  - no right-arm/hand sphere within 1.5 cm of the rest of the body
    (envs/mdp/self_collision.py, the same model as the training penalty);
  - right_shoulder_roll inside the posture box (< 0.35 rad).
Finger direction is free (as in the task).

Grasp-target positions relative to right_shoulder_pitch_link are binned on a
Cartesian grid (CELL m). A cell is FEASIBLE when it has >= MIN_SAMPLES samples
and a clean fraction >= MIN_CLEAN_FRAC. IK can stall in local minima, so
this is conservative: a cell can be marked infeasible although some posture
reaches it, never the reverse beyond the sampling noise. The map is written to
envs/assets/reach_workspace.npz (REACH_WORKSPACE_MAP in
envs/apple_grasp_env_cfg.py), and apple placement only samples feasible
cells. It is also the robot's safe reach workspace for planning where to
stand (locomotion).

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/probe_side_reach.py --headless --num_envs 512 --rounds 8
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=8, help="Resets (fresh apple samples) to run.")
parser.add_argument("--seeds", type=int, default=8, help="Arm seeds per apple sample (first = default pose).")
parser.add_argument("--steps", type=int, default=150, help="IK iterations per seed.")
parser.add_argument("--out", type=str, default=None, help="Map path (default envs/assets/reach_workspace.npz).")
parser.add_argument("--focus", action="store_true",
                    help="Sample only around the training placement box (FOCUS) for more samples per cell there.")
parser.add_argument("--merge", type=str, default=None,
                    help="Existing map .npz whose samples are merged with this run's before binning.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

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
from envs.mdp.self_collision import self_collision_depth  # noqa: E402

# Wide sweep of grasp-target positions (spherical around the shoulder).
SWEEP = dict(reach_min=0.12, reach_max=0.42, reach_azimuth_center_deg=-35.0,
             reach_azimuth_span_deg=130.0, target_dz_range=(-0.40, 0.35),
             reach_min_forward=0.0, reach_min_horizontal=0.0, workspace_map=None)
# Training placement box (REACH_* in envs/apple_grasp_env_cfg.py) plus a margin.
FOCUS = dict(reach_min=0.17, reach_max=0.37, reach_azimuth_center_deg=-45.0,
             reach_azimuth_span_deg=80.0, target_dz_range=(-0.33, 0.31),
             reach_min_forward=0.05, reach_min_horizontal=0.09, workspace_map=None)
OK_POS, OK_FACE = 0.02, 0.30
CELL = 0.05
GRID_LO = np.array([-0.15, -0.45, -0.40])   # shoulder-relative x (fwd), y (left), z (up)
GRID_HI = np.array([0.45, 0.20, 0.35])
MIN_SAMPLES, MIN_CLEAN_FRAC = 3, 0.8


def main():
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    cfg.events.randomize_tree.params.update(FOCUS if args_cli.focus else SWEEP)
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
    lo = robot.data.soft_joint_pos_limits[:, arm_ids, 0]
    hi = robot.data.soft_joint_pos_limits[:, arm_ids, 1]
    default = robot.data.default_joint_pos[:, arm_ids]
    shoulder_roll = RIGHT_ARM_JOINTS.index("right_shoulder_roll_joint")
    state = {"q": default.clone()}

    def set_arm(q0):
        jp = robot.data.joint_pos.clone()
        jp[:, arm_ids] = q0
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, arm_ids] = q0
        robot.set_joint_position_target(tgt)
        term._prev_target[:, term_ids] = q0
        term._prev_vel[:] = 0.0

    def ik_step():
        quat = robot.data.body_quat_w[:, palm_id]
        R = matrix_from_quat(quat)
        off_w = R @ off_local
        ref = robot.data.body_pos_w[:, palm_id] + off_w
        d = rr._approach_dir(env, apple_cfg)
        pos_err = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF) - ref
        pos_err = pos_err * (0.05 / pos_err.norm(dim=-1, keepdim=True).clamp(min=0.05))
        n_err = torch.cross(R @ n_local, d, dim=-1)   # rotation taking the palm normal onto d

        J = robot.root_physx_view.get_jacobians()[:, jac_id, :, :][:, :, jac_cols].clone()
        ox, oy, oz = off_w[:, 0], off_w[:, 1], off_w[:, 2]
        zero = torch.zeros_like(ox)
        skew = torch.stack([torch.stack([zero, -oz, oy], -1),
                            torch.stack([oz, zero, -ox], -1),
                            torch.stack([-oy, ox, zero], -1)], -2)
        J[:, :3] -= skew @ J[:, 3:]                    # linear part at the reference point
        J = torch.cat([J[:, :3], 0.3 * J[:, 3:]], dim=1)
        e = torch.cat([pos_err, 0.3 * n_err], dim=-1).unsqueeze(-1)
        JJt = J @ J.transpose(1, 2) + (0.03 ** 2) * torch.eye(6, device=env.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e)).squeeze(-1).clamp(-0.1, 0.1)
        state["q"] = torch.clamp(state["q"] + dq, lo, hi)
        set_arm(state["q"])
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    def measure():
        d_pos = (rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF)
                 - rr._palm_ref_pos(env, palm_cfg, PALM_LOCAL_OFFSET)).norm(dim=-1)
        face = rr._palm_normal_angle(env, palm_cfg, PALM_NORMAL_AXIS, apple_cfg)
        coll = self_collision_depth(env).amax(dim=-1)
        q = robot.data.joint_pos[:, arm_ids]
        reached = (d_pos < OK_POS) & (face < OK_FACE)
        clean = reached & (coll <= 0.0) & (q[:, shoulder_roll] < 0.35)
        score = d_pos / OK_POS + face / OK_FACE + coll / 0.01 - 10.0 * clean.float() - 5.0 * reached.float()
        return score, [d_pos, face, coll, q[:, shoulder_roll], clean.float()]

    # Sanity: the default pose must be collision-free under the model.
    env.reset()
    set_arm(default.clone())
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    print(f"default pose self-collision depth (max over envs): {self_collision_depth(env).max():.4f} m")

    rows = []
    for rnd in range(args_cli.rounds):
        env.reset()
        best_score = torch.full((env.num_envs,), float("inf"), device=env.device)
        best = None
        for seed in range(args_cli.seeds):
            q0 = default.clone() if seed == 0 else lo + (hi - lo) * (0.15 + 0.7 * torch.rand_like(default))
            state["q"] = q0
            set_arm(q0)
            for _ in range(args_cli.steps):
                ik_step()
            score, cols = measure()
            better = score < best_score
            best_score = torch.where(better, score, best_score)
            best = cols if best is None else [torch.where(better, c, b) for c, b in zip(cols, best)]
        sh = env.scene.env_origins + env._shoulder_local_xyz
        tgt = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF) - sh
        rows.append(torch.cat([tgt] + [c.unsqueeze(-1) for c in best], -1).cpu().numpy())
        print(f"round {rnd + 1}/{args_cli.rounds}: clean so far "
              f"{np.concatenate(rows)[:, -1].mean():.2f}", flush=True)

    a = np.concatenate(rows)
    if args_cli.merge:
        a = np.concatenate([np.load(args_cli.merge)["samples"], a])
    xyz, clean = a[:, :3], a[:, -1] > 0.5
    print(f"\nN={len(a)}  clean {clean.mean():.2f}   median pos {np.median(a[:, 3])*100:.1f} cm  "
          f"face {np.degrees(np.median(a[:, 4])):.0f} deg   collisions {np.mean(a[:, 5] > 0):.2f}")

    edges = [np.arange(GRID_LO[i], GRID_HI[i] + 1e-9, CELL) for i in range(3)]
    shape = tuple(len(e) - 1 for e in edges)
    idx = np.stack([np.floor((xyz[:, i] - GRID_LO[i]) / CELL).astype(int) for i in range(3)], -1)
    inside = np.all((idx >= 0) & (idx < np.array(shape)), axis=-1)
    count = np.zeros(shape, int)
    nclean = np.zeros(shape, int)
    np.add.at(count, tuple(idx[inside].T), 1)
    np.add.at(nclean, tuple(idx[inside].T), clean[inside].astype(int))
    frac = np.where(count > 0, nclean / np.maximum(count, 1), 0.0)
    feasible = (count >= MIN_SAMPLES) & (frac >= MIN_CLEAN_FRAC)
    sampled = count >= MIN_SAMPLES
    print(f"cells: {sampled.sum()} sampled (>= {MIN_SAMPLES}), {feasible.sum()} feasible "
          f"({CELL*100:.0f} cm cells, volume {feasible.sum() * CELL**3 * 1e3:.1f} L)")

    # Slices: feasible cells per height band (rows y left->right, cols x near->far).
    for k in range(shape[2]):
        if not sampled[:, :, k].any():
            continue
        z0 = edges[2][k]
        print(f"\nz {z0:+.2f}..{z0 + CELL:+.2f} m vs shoulder   '#' feasible, '.' sampled but not, ' ' not sampled")
        print("          x " + "".join(f"{e:+.2f}".rjust(6) for e in edges[0][:-1]))
        for j in reversed(range(shape[1])):
            line = "".join(("#" if feasible[i, j, k] else ("." if sampled[i, j, k] else " ")).rjust(6)
                           for i in range(shape[0]))
            print(f"  y {edges[1][j]:+.2f}   {line}")

    out = args_cli.out or os.path.join(REPO, "envs", "assets", "reach_workspace.npz")
    np.savez(out, x_edges=edges[0], y_edges=edges[1], z_edges=edges[2], feasible=feasible,
             count=count, clean_frac=frac, samples=a,
             sample_columns=np.array(["x", "y", "z", "pos_err", "face_err", "collision_depth", "shoulder_roll", "clean"]),
             min_samples=MIN_SAMPLES, min_clean_frac=MIN_CLEAN_FRAC, cell=CELL)
    print(f"\nsaved {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
