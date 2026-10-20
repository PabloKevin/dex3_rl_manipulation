#!/usr/bin/env python3
"""Can the arm-pose policy (AppleArmPose-v1) carry the apple from home to the BASKET with the waist turned?

The basket is where the original picker's tree setup has it (apple_picker_config.json box.tree): the drop
goal is the hand pose of its robot-verified drop configuration drop_q_right, measured here at waist 0 and
then held FIXED TO THE GROUND (pelvis frame) while the waist is turned, as a real basket would be. Each env:
the waist at one yaw of --yaws, the arm at the robot's HOME_Q with the reach's cup, the policy given the
drop pose as one stop goal (the fingers' direction counted), speed factor --speed (the robot's 0.3).

Per yaw: reached (the trained 2 cm / 15 deg / still test) and when, final pocket / normal error, the body:
sphere-model overlap (the training's collision termination) and near misses (within 3 cm), and the basket:
any hand point below the rim within the basket's outer radius (hitting its wall or stand).

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/eval_basket_deposit.py --headless \\
        --policy logs/rsl_rl/apple_arm_pose/2026-10-19_13-31-55/exported/policy.pt
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="AppleArmPose-Play-v1")
parser.add_argument("--policy", required=True)
parser.add_argument("--yaws", type=float, nargs="+", default=[-45, -30, -15, 0, 15], help="waist yaw, deg")
parser.add_argument("--reps", type=int, default=4, help="envs per yaw")
parser.add_argument("--speed", type=float, default=0.3)
parser.add_argument("--timeout_s", type=float, default=None, help="default 4 / speed (the toy's waypoint timeout)")
parser.add_argument("--settle_steps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_arm_pose_env_cfg import POCKET  # noqa: E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.mdp import arm_pose as ap  # noqa: E402
from envs.mdp import self_collision as scm  # noqa: E402

PICKER_CFG = os.path.expanduser("~/g1_fruits_picker/apple_picker/apple_picker_config.json")
#: the robot's home (sim2real/reach_io.HOME_Q): the default pose with the shoulder 0.15 rad forward / out / rotated
HOME_Q = [-0.15, -0.15, -0.15, 0.5, 0.0, 0.0, 0.0]
#: the picker's tree basket (its config comment): centre torso (0.14, -0.36, -0.46) at waist 0, rim z -0.38;
#: radius as the sim's slatted basket (envs/mdp/basket.py: inner 0.14 + slats)
BASKET_CENTER_TORSO = (0.14, -0.36, -0.46)
BASKET_RIM_Z_TORSO = -0.38
BASKET_RADIUS = 0.16


def main():
    cfg_j = json.load(open(PICKER_CFG))
    drop_q = cfg_j["box"]["tree"]["drop_q_right"]
    yaws = [math.radians(y) for y in args.yaws]
    N = len(yaws) * args.reps
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
    env_cfg.episode_length_s = 60.0
    env_cfg.events.advance_goals = None                 # the goal is ours
    env_cfg.terminations.collision = None               # an overlap is recorded, not an episode end
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    dev = env.device
    robot = env.scene["robot"]
    arm = torch.tensor([robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS], device=dev)
    wj = robot.joint_names.index("waist_yaw_joint")
    term = env.action_manager.get_term("right_arm")
    term.speed_scale[:] = args.speed
    pol = torch.jit.load(args.policy, map_location=dev).eval()
    P = ap.pose_state(env)
    pel = robot.data.body_names.index("pelvis")
    torso = robot.data.body_names.index("torso_link")
    env.reset()

    def place(q_arm, yaw):
        jp = robot.data.joint_pos.clone()
        jp[:, arm] = torch.as_tensor(q_arm, device=dev, dtype=jp.dtype)
        jp[:, wj] = yaw
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, arm] = jp[:, arm]
        tgt[:, wj] = yaw
        robot.set_joint_position_target(tgt)

    def hold_action():
        return (robot.data.joint_pos[:, arm] - term._offset) / term._scale

    def settle(n):
        for _ in range(n):
            P.pending[:] = True
            env.step(hold_action())

    def to_pel(p_w):
        return quat_apply_inverse(robot.data.body_quat_w[:, pel], p_w - robot.data.body_pos_w[:, pel])

    def from_pel(p):
        return robot.data.body_pos_w[:, pel] + quat_apply(robot.data.body_quat_w[:, pel], p)

    # 1. the drop pose: the picker's drop_q_right at waist 0 (pelvis frame), and the basket there
    place(drop_q, torch.zeros(N, device=dev))
    settle(args.settle_steps)
    pk, nm, fg = ap.hand_frame(env, POCKET)
    drop_p, drop_n, drop_f = to_pel(pk)[0].clone(), quat_apply_inverse(robot.data.body_quat_w[:, pel], nm)[0].clone(), \
        quat_apply_inverse(robot.data.body_quat_w[:, pel], fg)[0].clone()
    tq = robot.data.body_quat_w[:, torso]
    tp = robot.data.body_pos_w[:, torso]
    c_w = tp + quat_apply(tq, torch.tensor(BASKET_CENTER_TORSO, device=dev).expand(N, 3))
    basket_c = to_pel(c_w)[0].clone()
    rim_w = tp + quat_apply(tq, torch.tensor([0.0, 0.0, BASKET_RIM_Z_TORSO], device=dev).expand(N, 3))
    rim_z = float(to_pel(rim_w)[0, 2])
    print(f"[basket] drop pocket (pelvis) {np.round(drop_p.cpu().numpy(), 3).tolist()}, palm normal "
          f"{np.round(drop_n.cpu().numpy(), 2).tolist()}; basket centre {np.round(basket_c.cpu().numpy(), 3).tolist()}, "
          f"rim z {rim_z:+.3f}")

    # 2. every env: its waist yaw, the arm at home, then the drop goal
    yaw_env = torch.tensor([yaws[i // args.reps] for i in range(N)], device=dev)
    place(HOME_Q, yaw_env)
    settle(args.settle_steps)
    P.target[:] = from_pel(drop_p.expand(N, 3))
    P.normal[:] = quat_apply(robot.data.body_quat_w[:, pel], drop_n.expand(N, 3))
    P.finger[:] = quat_apply(robot.data.body_quat_w[:, pel], drop_f.expand(N, 3))
    P.finger_on[:] = True
    P.via[:] = False
    P.pending[:] = False
    P.hold[:] = 0.0
    P.arrived[:] = False
    P.success[:] = False
    timeout = args.timeout_s or 4.0 / args.speed
    steps = int(round(timeout / env.step_dt))
    t_reach = torch.full((N,), float("nan"), device=dev)
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    near = torch.zeros(N, dtype=torch.bool, device=dev)
    basket_hit = torch.zeros(N, dtype=torch.bool, device=dev)
    last = torch.zeros(N, 7, device=dev)
    b_xy = basket_c[:2]
    for k in range(steps):
        obs = torch.cat([robot.data.joint_pos[:, arm] - robot.data.default_joint_pos[:, arm],
                         robot.data.joint_vel[:, arm], ap.obs_goal_delta(env, POCKET), ap.obs_goal_position(env, POCKET),
                         ap.obs_goal_normal(env, POCKET), ap.obs_goal_finger(env, POCKET), ap.obs_goal_flags(env),
                         last], dim=-1)
        with torch.inference_mode():
            act = pol(obs)
        env.step(act)
        last = act.clone()
        ap.track_goal(env, POCKET)
        newly = P.arrived & torch.isnan(t_reach)
        t_reach = torch.where(newly, torch.full_like(t_reach, (k + 1) * env.step_dt), t_reach)
        hit |= scm.self_collision_depth(env, margin=0.0).amax(-1) > 0.0
        near |= scm.self_collision_depth(env, margin=0.03).amax(-1) > 0.0
        # the hand vs the basket: palm origin, pocket, fingertips below the rim within its outer radius
        pts = [robot.data.body_pos_w[:, robot.data.body_names.index(b)] for b in
               ("right_hand_palm_link", "right_wrist_yaw_link", "right_hand_index_1_link",
                "right_hand_middle_1_link", "right_hand_thumb_2_link")] + [ap.hand_frame(env, POCKET)[0]]
        for pw in pts:
            pp = to_pel(pw)
            inside = ((pp[:, :2] - b_xy).norm(dim=-1) < BASKET_RADIUS) & (pp[:, 2] < rim_z)
            basket_hit |= inside
    d, ang, fang, _ = ap.errors(env, POCKET)
    print(f"\n[deposit] speed {args.speed}, {timeout:.1f} s per try, {args.reps} envs per waist yaw "
          f"(goal: the picker's drop pose, fixed to the ground)")
    print(" waist   reached   time s     pocket cm   normal deg   body overlap   body <3cm   basket wall")
    rows = []
    for yi, y in enumerate(args.yaws):
        sl = slice(yi * args.reps, (yi + 1) * args.reps)
        r = ~torch.isnan(t_reach[sl])
        tr = t_reach[sl][r]
        row = dict(yaw=y, reached=int(r.sum()), n=args.reps,
                   t=float(tr.mean()) if len(tr) else float("nan"), d=float(d[sl].mean()) * 100,
                   ang=math.degrees(float(ang[sl].mean())), hit=int(hit[sl].sum()), near=int(near[sl].sum()),
                   basket=int(basket_hit[sl].sum()))
        rows.append(row)
        print(f" {y:+5.0f}   {row['reached']}/{row['n']}      {row['t']:6.2f}    {row['d']:6.1f}      {row['ang']:6.1f}"
              f"        {row['hit']}/{row['n']}          {row['near']}/{row['n']}        {row['basket']}/{row['n']}")
    out = os.path.join(REPO, "logs", "sim2real", "basket_deposit_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"policy": args.policy, "speed": args.speed, "rows": rows}, open(out, "w"), indent=1)
    print(f"-> {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
