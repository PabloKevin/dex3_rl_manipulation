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

--waist: the waist yaw joins the IK (within +-WAIST_YAW_LIMIT, the
AppleReachTreeWaist-v1 action range) and the sweep/grid widen toward the
front-centre and further out. Cells stay relative to the shoulder AT WAIST
YAW 0 (a fixed point, as placement uses it). Writes
envs/assets/reach_workspace_waist.npz (REACH_WAIST_WORKSPACE_MAP).

--single_v2: AppleReachSingle-v2's end pose (implies --waist): the apple
centre in the hand's pocket (palm touching it, single_reach.enclosure_state),
the palm heading along the horizontal pelvis -> apple line (no stem), the
fingers DOWN (up for fruit above shoulder + 15 cm) and leaning ~30 deg
forward along the approach so all fingertips sit at one depth, the hand in
the closed cup. CLEAN: pocket error < OK_POS, heading < 45 deg, tilt < 40
deg, finger roll < 30 deg, no self-collision, shoulder roll in the box. Cells
hold the APPLE CENTRE relative to the shoulder (tree_sampler.Envelope
apple_ref=True). Wider sweep and grid. Writes
envs/assets/reach_workspace_single_v2.npz.

--single_v2 --fixed_waist LO HI (deg): the same end pose with the waist OUT
of the IK, held at a fixed yaw drawn per env and round in [LO, HI]
(AppleReachVisible-v1: the scan script turns the torso, the reach runs with
it held). The sampled apple is turned with the torso about the waist axis,
and cells keep the UNTURNED apple centre relative to the shoulder at yaw 0
(the frame the tree is sampled in); what still changes with the yaw is the
legs vs the arm (self-collision). --sweep_visible samples only the front
band the head camera can see (the camera test itself is applied at
placement). Writes envs/assets/reach_workspace_single_v2_fixed_waist.npz.
"""
import argparse
import math
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
parser.add_argument("--union", type=str, default=None,
                    help="Map .npz whose feasible cells are added (e.g. the arm-only map into the --waist one: "
                         "waist yaw 0 is always available, and the 8-DOF IK stalls more often).")
parser.add_argument("--waist", action="store_true",
                    help="Waist yaw in the IK (+-WAIST_YAW_LIMIT), wider sweep; writes reach_workspace_waist.npz.")
parser.add_argument("--single_v2", action="store_true",
                    help="AppleReachSingle-v2 end pose (apple in the pocket, tilted palm, cup); implies --waist.")
parser.add_argument("--fixed_waist", type=float, nargs=2, default=None, metavar=("LO_DEG", "HI_DEG"),
                    help="With --single_v2: waist held at a random fixed yaw in [LO, HI] deg, not in the IK.")
parser.add_argument("--waist_pitch", type=float, default=None,
                    help="Hold waist_pitch_joint here (rad) instead of the env default 0.1 (the real robot: 0).")
parser.add_argument("--sweep_visible", action="store_true",
                    help="Sweep only the front band the head D435i can see (shoulder-relative, yaw-0 frame).")
parser.add_argument("--union_shift", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="Shift (m) added to the --union map's cell centres (grasp-target map -> apple-centre map).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.fixed_waist is not None and not args_cli.single_v2:
    parser.error("--fixed_waist needs --single_v2")
if args_cli.single_v2 and args_cli.fixed_waist is None:
    args_cli.waist = True
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
from envs.apple_reach_tree_waist_env_cfg import WAIST_YAW_JOINT, WAIST_YAW_LIMIT  # noqa: E402
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
# --waist: the torso turns too, so further out and across the front.
SWEEP_WAIST = dict(reach_min=0.12, reach_max=0.55, reach_azimuth_center_deg=-25.0,
                   reach_azimuth_span_deg=170.0, target_dz_range=(-0.40, 0.35),
                   reach_min_forward=0.0, reach_min_horizontal=0.0, workspace_map=None)
# --single_v2: wider still (apple centres up to ~0.7 m out, all round the front)
SWEEP_SINGLE_V2 = dict(reach_min=0.10, reach_max=0.70, reach_azimuth_center_deg=-25.0,
                       reach_azimuth_span_deg=220.0, target_dz_range=(-0.50, 0.50),
                       reach_min_forward=-0.10, reach_min_horizontal=0.0, workspace_map=None)
# --sweep_visible: apple centres the head camera can see at yaw 0 lie at
# shoulder azimuth -30..+75 deg, up to ~shoulder height (scratchpad
# fov_coverage.py, 2026-10-16), plus a margin
SWEEP_VISIBLE = dict(reach_min=0.12, reach_max=0.70, reach_azimuth_center_deg=20.0,
                     reach_azimuth_span_deg=140.0, target_dz_range=(-0.50, 0.15),
                     reach_min_forward=-0.05, reach_min_horizontal=0.0, workspace_map=None)
OK_POS, OK_FACE = 0.02, 0.30
# --single_v2 end pose (envs/apple_reach_single_v2_env_cfg.py)
V2_TOUCH_GAP = 0.016
V2_LEAN = math.radians(30.0)
V2_FINGER_SWITCH_DZ = 0.15
V2_OK_HEADING, V2_OK_TILT, V2_OK_ROLL = math.radians(45.0), math.radians(40.0), math.radians(30.0)
CELL = 0.05
GRID_LO = np.array([-0.15, -0.45, -0.40])   # shoulder-relative x (fwd), y (left), z (up)
GRID_HI = np.array([0.45, 0.20, 0.35])
if args_cli.waist:
    GRID_LO = np.array([-0.15, -0.55, -0.40])
    GRID_HI = np.array([0.60, 0.40, 0.35])
if args_cli.single_v2:
    GRID_LO = np.array([-0.20, -0.75, -0.55])
    GRID_HI = np.array([0.80, 0.55, 0.55])
MIN_SAMPLES, MIN_CLEAN_FRAC = 3, 0.8


def main():
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.waist_pitch is not None:
        jp = dict(cfg.scene.robot.init_state.joint_pos)
        jp["waist_pitch_joint"] = args_cli.waist_pitch
        cfg.scene.robot = cfg.scene.robot.replace(init_state=cfg.scene.robot.init_state.replace(joint_pos=jp))
    cfg.episode_length_s = 1.0e6
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    cfg.events.randomize_tree.params.update(
        SWEEP_VISIBLE if args_cli.sweep_visible else
        SWEEP_SINGLE_V2 if args_cli.single_v2 else SWEEP_WAIST if args_cli.waist else FOCUS if args_cli.focus else SWEEP)
    env = ManagerBasedRLEnv(cfg)
    if args_cli.single_v2:
        # the approach is the horizontal pelvis -> apple line (single_reach.set_approach_origin)
        env._approach_from_body = env.scene["robot"].data.body_names.index("pelvis")

    robot = env.scene["robot"]
    palm_cfg = SceneEntityCfg("robot", body_names=["right_hand_palm_link"])
    palm_cfg.resolve(env.scene)
    apple_cfg = SceneEntityCfg("apple")
    term = env.action_manager.get_term("right_arm")
    term_ids = [term._joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    # IK joints: the arm, plus the waist yaw with --waist (last column; the
    # base env holds it by PD target, so teleporting it works the same way)
    arm_ids = [robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    ik_ids = arm_ids + ([robot.joint_names.index(WAIST_YAW_JOINT)] if args_cli.waist else [])
    palm_id = palm_cfg.body_ids[0]
    jac_id = palm_id - 1 if robot.is_fixed_base else palm_id
    jac_cols = ik_ids if robot.is_fixed_base else [i + 6 for i in ik_ids]
    off_local = torch.tensor(PALM_LOCAL_OFFSET, device=env.device)
    n_local = torch.tensor(PALM_NORMAL_AXIS, device=env.device)
    lo = robot.data.soft_joint_pos_limits[:, ik_ids, 0].clone()
    hi = robot.data.soft_joint_pos_limits[:, ik_ids, 1].clone()
    if args_cli.waist:
        lo[:, -1], hi[:, -1] = -WAIST_YAW_LIMIT, WAIST_YAW_LIMIT
    default = robot.data.default_joint_pos[:, ik_ids]
    shoulder_roll = RIGHT_ARM_JOINTS.index("right_shoulder_roll_joint")
    state = {"q": default.clone()}
    fixed = args_cli.fixed_waist is not None
    waist_id = robot.joint_names.index(WAIST_YAW_JOINT)
    yaw = torch.zeros(env.num_envs, device=env.device)
    hand_ids, hand_q = [], None
    if args_cli.single_v2:
        from envs.apple_reach_single_env_cfg import HAND_CLOSED
        hand_ids = [robot.joint_names.index(n) for n in HAND_CLOSED]
        hand_q = torch.tensor(list(HAND_CLOSED.values()), device=env.device)
        shoulder_id = robot.data.body_names.index("right_shoulder_pitch_link")
    up = torch.tensor([0.0, 0.0, 1.0], device=env.device)
    f_local = torch.tensor(rr.PALM_FINGER_AXIS, device=env.device)

    def set_arm(q0):
        jp = robot.data.joint_pos.clone()
        jp[:, ik_ids] = q0
        if hand_ids:
            jp[:, hand_ids] = hand_q
        if fixed:
            jp[:, waist_id] = yaw
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, ik_ids] = q0
        if hand_ids:
            tgt[:, hand_ids] = hand_q
        if fixed:
            tgt[:, waist_id] = yaw
        robot.set_joint_position_target(tgt)
        term._prev_target[:, term_ids] = q0[:, :len(arm_ids)]
        term._prev_vel[:] = 0.0

    def v2_goal():
        """(palm-point target [N, 3], desired normal n* [N, 3], desired finger axis f* [N, 3])."""
        d = rr._approach_dir(env, apple_cfg)
        apple = env.scene["apple"].data.root_pos_w
        s = torch.where(apple[:, 2] - robot.data.body_pos_w[:, shoulder_id, 2] > V2_FINGER_SWITCH_DZ, 1.0, -1.0)
        s = s.unsqueeze(-1)
        n_star = math.cos(V2_LEAN) * d - s * math.sin(V2_LEAN) * up
        f_star = math.sin(V2_LEAN) * d + s * math.cos(V2_LEAN) * up
        radius = env._apple_radius if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) \
            else torch.full((env.num_envs,), rr.DEFAULT_APPLE_RADIUS, device=env.device)
        return apple - (radius + V2_TOUCH_GAP).unsqueeze(-1) * n_star, n_star, f_star, s.squeeze(-1)

    def ik_step():
        quat = robot.data.body_quat_w[:, palm_id]
        R = matrix_from_quat(quat)
        off_w = R @ off_local
        ref = robot.data.body_pos_w[:, palm_id] + off_w
        if args_cli.single_v2:
            goal, n_star, f_star, _ = v2_goal()
            pos_err = goal - ref
            n_err = 0.5 * (torch.cross(R @ n_local, n_star, dim=-1) + torch.cross(R @ f_local, f_star, dim=-1))
        else:
            d = rr._approach_dir(env, apple_cfg)
            pos_err = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF) - ref
            n_err = torch.cross(R @ n_local, d, dim=-1)   # rotation taking the palm normal onto d
        pos_err = pos_err * (0.05 / pos_err.norm(dim=-1, keepdim=True).clamp(min=0.05))

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
        coll = self_collision_depth(env).amax(dim=-1)
        q = robot.data.joint_pos[:, ik_ids]
        if args_cli.single_v2:
            # pocket error: the apple vs radius + touch gap out along the ACTUAL palm normal
            R = matrix_from_quat(robot.data.body_quat_w[:, palm_id])
            n_w, f_w = R @ n_local, R @ f_local
            radius = env._apple_radius if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) \
                else torch.full((env.num_envs,), rr.DEFAULT_APPLE_RADIUS, device=env.device)
            seat = rr._palm_ref_pos(env, palm_cfg, PALM_LOCAL_OFFSET) + (radius + V2_TOUCH_GAP).unsqueeze(-1) * n_w
            d_pos = (env.scene["apple"].data.root_pos_w - seat).norm(dim=-1)
            d = rr._approach_dir(env, apple_cfg)
            nh = n_w.clone()
            nh[:, 2] = 0.0
            nh = nh / nh.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            heading = torch.acos((nh * d).sum(-1).clamp(-1.0, 1.0))
            tilt = torch.asin(n_w[:, 2].abs().clamp(max=1.0))
            _, _, _, s = v2_goal()
            lat = torch.cross(up.expand_as(d), d, dim=-1)
            roll = torch.atan2((f_w * lat).sum(-1).abs(), s * f_w[:, 2])
            face = torch.maximum(heading / V2_OK_HEADING, torch.maximum(tilt / V2_OK_TILT, roll / V2_OK_ROLL)) * OK_FACE
        else:
            d_pos = (rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF)
                     - rr._palm_ref_pos(env, palm_cfg, PALM_LOCAL_OFFSET)).norm(dim=-1)
            face = rr._palm_normal_angle(env, palm_cfg, PALM_NORMAL_AXIS, apple_cfg)
        reached = (d_pos < OK_POS) & (face < OK_FACE)
        clean = reached & (coll <= 0.0) & (q[:, shoulder_roll] < 0.35)
        score = d_pos / OK_POS + face / OK_FACE + coll / 0.01 - 10.0 * clean.float() - 5.0 * reached.float()
        cols = [d_pos, face, coll, q[:, shoulder_roll]]
        if fixed:
            cols.append(yaw.clone())
        if args_cli.waist:
            # among clean solutions, prefer the least torso turn
            score = score + 0.5 * q[:, -1].abs() / WAIST_YAW_LIMIT
            cols.append(q[:, -1])
        return score, cols + [clean.float()]

    # Sanity: the default pose must be collision-free under the model.
    env.reset()
    set_arm(default.clone())
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    print(f"default pose self-collision depth (max over envs): {self_collision_depth(env).max():.4f} m")

    rows = []
    pel = robot.data.body_names.index("pelvis")
    pivot = (robot.data.body_pos_w[:, pel, :2] - env.scene.env_origins[:, :2]).clone()
    for rnd in range(args_cli.rounds):
        env.reset()
        apple0 = env.scene["apple"].data.root_pos_w.clone()      # as sampled: the torso at yaw 0
        if fixed:
            lo_y, hi_y = (math.radians(v) for v in args_cli.fixed_waist)
            yaw[:] = lo_y + (hi_y - lo_y) * torch.rand_like(yaw)
            c, s_ = torch.cos(yaw), torch.sin(yaw)
            rel = apple0[:, :2] - env.scene.env_origins[:, :2] - pivot
            turned = apple0.clone()
            turned[:, 0] = env.scene.env_origins[:, 0] + pivot[:, 0] + c * rel[:, 0] - s_ * rel[:, 1]
            turned[:, 1] = env.scene.env_origins[:, 1] + pivot[:, 1] + s_ * rel[:, 0] + c * rel[:, 1]
            apple = env.scene["apple"]
            pose = apple.data.root_pose_w.clone()
            pose[:, :3] = turned
            apple.write_root_pose_to_sim(pose)
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
        if args_cli.single_v2:   # cells hold the APPLE CENTRE (Envelope apple_ref=True), torso at yaw 0
            tgt = apple0 - sh
        else:
            tgt = rr._grasp_target_pos(env, apple_cfg, GRASP_STANDOFF) - sh
        rows.append(torch.cat([tgt] + [c.unsqueeze(-1) for c in best], -1).cpu().numpy())
        print(f"round {rnd + 1}/{args_cli.rounds}: clean so far "
              f"{np.concatenate(rows)[:, -1].mean():.2f}", flush=True)

    a = np.concatenate(rows)
    columns = ["x", "y", "z", "pos_err", "face_err", "collision_depth", "shoulder_roll"] \
        + (["waist_yaw"] if (args_cli.waist or fixed) else []) + ["clean"]
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
    if args_cli.union:
        u = np.load(args_cli.union)
        added = 0
        for i, j, k in zip(*np.nonzero(u["feasible"])):
            c = np.array([u["x_edges"][i], u["y_edges"][j], u["z_edges"][k]]) + 0.5 * CELL \
                + np.asarray(args_cli.union_shift)
            n = np.floor((c - GRID_LO) / CELL).astype(int)
            if np.all((n >= 0) & (n < np.array(shape))):
                added += int(not feasible[tuple(n)])
                feasible[tuple(n)] = sampled[tuple(n)] = True
        print(f"union with {args_cli.union}: +{added} cells")
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

    if args_cli.waist:
        w = a[clean, 7]
        print(f"\nwaist yaw over clean samples: median |yaw| {np.degrees(np.median(np.abs(w))):.0f} deg, "
              f"at the limit {np.mean(np.abs(w) > WAIST_YAW_LIMIT - 0.02):.2f}")
    default_map = ("reach_workspace_single_v2_fixed_waist.npz" if fixed else
                   "reach_workspace_single_v2.npz" if args_cli.single_v2 else
                   "reach_workspace_waist.npz" if args_cli.waist else "reach_workspace.npz")
    out = args_cli.out or os.path.join(REPO, "envs", "assets", default_map)
    np.savez(out, x_edges=edges[0], y_edges=edges[1], z_edges=edges[2], feasible=feasible,
             count=count, clean_frac=frac, samples=a,
             sample_columns=np.array(columns),
             min_samples=MIN_SAMPLES, min_clean_frac=MIN_CLEAN_FRAC, cell=CELL)
    print(f"\nsaved {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
