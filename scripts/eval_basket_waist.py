#!/usr/bin/env python3
"""The side basket reached by the arm-pose policy (AppleArmPose-v1) WITH THE WAIST TURNED RIGHT.

User 2026-10-19: the basket is where the right arm points stretched out to the right (shoulder height) with
the waist at 0 -- fixed to the ground; the waist may turn right so the basket comes in front of the arm. Per
deposit waist yaw (--yaws), start at READY (reach_io's HOME_Q), the toy's plan as for an apple
(arm_pose_io.make_plan: STAGING at the drop's height --stage_radius from the torso axis, an advance goal, then
the DROP point: the basket point, pulled in toward the torso axis until it is --max_reach from the shoulder;
the palm normal along the horizontal torso -> basket line), then back: staging, READY. Recorded: reached /
time / error per leg, how far the drop point is from the basket point (horizontally), body overlap / near
(sphere model), the hand's highest point and its largest distance from the torso axis on the way.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/eval_basket_waist.py --headless \\
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
parser.add_argument("--yaws", type=float, nargs="+", default=[-45, -30, -15, 0])
parser.add_argument("--ready_q", default="-0.15,-0.15,-0.15,0.5,0,0,0")
parser.add_argument("--basket_shoulder", type=float, nargs=3, default=[0.0, -0.522, -0.049],
                    help="the basket point minus the right shoulder, torso frame at waist 0 [m]")
parser.add_argument("--max_reach", type=float, default=0.50, help="drop point at most this far from the shoulder")
parser.add_argument("--stage_radius", type=float, default=0.24)
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.3)
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


def main():
    N = len(args.yaws) * args.reps
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
    env_cfg.episode_length_s = 200.0
    env_cfg.events.advance_goals = None
    env_cfg.terminations.collision = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    dev = env.device
    robot = env.scene["robot"]
    arm = torch.tensor([robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS], device=dev)
    wj = robot.joint_names.index("waist_yaw_joint")
    term = env.action_manager.get_term("right_arm")
    term.speed_scale[:] = args.speed
    pol = torch.jit.load(args.policy, map_location=dev).eval()
    P = ap.pose_state(env)
    torso = robot.data.body_names.index("torso_link")
    shoulder = robot.data.body_names.index("right_shoulder_pitch_link")
    env.reset()
    ready = torch.tensor([float(v) for v in args.ready_q.split(",")], device=dev).expand(N, 7)

    def place(yaw):
        jp = robot.data.joint_pos.clone()
        jp[:, arm] = ready
        jp[:, wj] = yaw
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, arm] = ready
        tgt[:, wj] = yaw
        robot.set_joint_position_target(tgt)

    def hold_action():
        return (robot.data.joint_pos[:, arm] - term._offset) / term._scale

    def settle(n=25):
        for _ in range(n):
            P.pending[:] = True
            env.step(hold_action())

    # the basket point, fixed to the ground: from the shoulder / torso with the waist at 0
    place(torch.zeros(N, device=dev))
    settle()
    tq, tp = robot.data.body_quat_w[:, torso].clone(), robot.data.body_pos_w[:, torso].clone()
    B = robot.data.body_pos_w[:, shoulder] + quat_apply(tq, torch.tensor(args.basket_shoulder, device=dev).expand(N, 3))
    # every env at its deposit yaw, the arm at READY
    yaw = torch.tensor([math.radians(args.yaws[i // args.reps]) for i in range(N)], device=dev)
    place(yaw)
    settle()
    tq, tp = robot.data.body_quat_w[:, torso].clone(), robot.data.body_pos_w[:, torso].clone()
    sh = robot.data.body_pos_w[:, shoulder].clone()
    ready_g = tuple(x.clone() for x in ap.hand_frame(env, POCKET))
    # drop point: B pulled toward the torso axis (horizontally) until within max_reach of the shoulder
    u = B - tp
    u[:, 2] = 0.0
    dB = u.norm(dim=-1, keepdim=True)
    u = u / dB
    drop = B.clone()
    for i in range(N):
        lo, hi = 0.0, float(dB[i])
        if float((B[i] - sh[i]).norm()) > args.max_reach:
            for _ in range(40):
                m = 0.5 * (lo + hi)
                p = B[i] - m * u[i]
                lo, hi = (lo, m) if float((p - sh[i]).norm()) <= args.max_reach else (m, hi)
            drop[i] = B[i] - hi * u[i]
    stage = tp + args.stage_radius * u
    stage[:, 2] = drop[:, 2]
    finger_now = ready_g[2]
    chain = [("staging", (stage, u, finger_now), False, True), ("basket", (drop, u, finger_now), False, False),
             ("staging back", (stage, u, finger_now), False, True), ("ready", ready_g, True, False)]

    def set_goal(ids, G, f_on):
        P.target[ids], P.normal[ids], P.finger[ids] = G[0][ids], G[1][ids], G[2][ids]
        P.finger_on[ids] = f_on
        P.via[ids] = False
        P.pending[ids] = False
        P.hold[ids] = 0.0
        P.arrived[ids] = False
        P.success[ids] = False

    set_goal(torch.arange(N, device=dev), chain[0][1], chain[0][2])
    L = len(chain)
    leg = torch.zeros(N, dtype=torch.long, device=dev)
    t_leg = torch.zeros(N, device=dev)
    timeout = 4.0 / args.speed
    adv_s = 2.4 / args.speed
    t_r = torch.full((N, L), float("nan"), device=dev)
    d_e = torch.full((N, L), float("nan"), device=dev)
    a_e = torch.full((N, L), float("nan"), device=dev)
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    near = torch.zeros(N, dtype=torch.bool, device=dev)
    top = torch.full((N,), -9.0, device=dev)
    rad = torch.zeros(N, device=dev)
    last = torch.zeros(N, 7, device=dev)
    for _ in range(int(L * timeout / env.step_dt) + 10):
        obs = torch.cat([robot.data.joint_pos[:, arm] - robot.data.default_joint_pos[:, arm],
                         robot.data.joint_vel[:, arm], ap.obs_goal_delta(env, POCKET), ap.obs_goal_position(env, POCKET),
                         ap.obs_goal_normal(env, POCKET), ap.obs_goal_finger(env, POCKET), ap.obs_goal_flags(env),
                         last], dim=-1)
        with torch.inference_mode():
            act = pol(obs)
        act = torch.where((leg >= L)[:, None], hold_action(), act)
        env.step(act)
        last = act.clone()
        ap.track_goal(env, POCKET)
        t_leg += env.step_dt
        live = leg < L
        hit |= live & (scm.self_collision_depth(env, margin=0.0).amax(-1) > 0.0)
        near |= live & (scm.self_collision_depth(env, margin=0.03).amax(-1) > 0.0)
        pk = ap.hand_frame(env, POCKET)[0]
        rel_s = quat_apply_inverse(robot.data.body_quat_w[:, torso], pk - robot.data.body_pos_w[:, shoulder])
        rel_t = quat_apply_inverse(robot.data.body_quat_w[:, torso], pk - robot.data.body_pos_w[:, torso])
        top = torch.where(live, torch.maximum(top, rel_s[:, 2]), top)
        rad = torch.where(live, torch.maximum(rad, rel_t[:, :2].norm(dim=-1)), rad)
        d, ang, _, _ = ap.errors(env, POCKET)
        adv = torch.tensor([chain[int(k)][3] if k < L else False for k in leg], device=dev)
        passed = adv & (((d < 0.05) & (ang < math.radians(30))) | (t_leg >= adv_s))
        done = live & (P.arrived | passed | (t_leg >= timeout))
        for k in range(L):
            m = done & (leg == k)
            t_r[:, k] = torch.where(m & (P.arrived | passed), t_leg, t_r[:, k])
            d_e[:, k] = torch.where(m, d * 100, d_e[:, k])
            a_e[:, k] = torch.where(m, torch.rad2deg(ang), a_e[:, k])
        leg = torch.where(done, leg + 1, leg)
        t_leg = torch.where(done, torch.zeros_like(t_leg), t_leg)
        for k in range(1, L):
            m = done & (leg == k)
            if m.any():
                set_goal(m.nonzero().squeeze(-1), chain[k][1], chain[k][2])
        if bool((leg >= L).all()):
            break
    short = (B - drop)[:, :2].norm(dim=-1) * 100
    rows = []
    lines = [f"[basket waist] speed {args.speed}, {timeout:.1f} s per leg (staging: advance at 5 cm / 30 deg or "
             f"{adv_s:.1f} s), drop within {args.max_reach} m of the shoulder"]
    for yi, y in enumerate(args.yaws):
        sel = slice(yi * args.reps, (yi + 1) * args.reps)
        legs = []
        for k, (name, _, _, _) in enumerate(chain):
            r = ~torch.isnan(t_r[sel, k])
            legs.append(dict(leg=name, ok=int(r.sum()), t=float(t_r[sel, k][r].mean()) if r.any() else float("nan"),
                             cm=float(d_e[sel, k].mean()), deg=float(a_e[sel, k].mean())))
        row = dict(yaw=y, legs=legs, n=args.reps, drop_short_cm=float(short[sel].mean()), overlap=int(hit[sel].sum()),
                   near=int(near[sel].sum()), top_vs_shoulder_cm=float(top[sel].max()) * 100,
                   max_radius_m=float(rad[sel].max()))
        rows.append(row)
        lines.append(f" waist {y:+4.0f}: drop {row['drop_short_cm']:.1f} cm short of the basket point | " +
                     "  ".join(f"{l['leg']} {l['ok']}/{args.reps} {l['t']:.1f}s {l['cm']:.1f}cm {l['deg']:.0f}deg"
                               for l in legs) +
                     f" | overlap {row['overlap']}/{args.reps} <3cm {row['near']}/{args.reps} | hand highest "
                     f"{row['top_vs_shoulder_cm']:+.0f} cm vs shoulder, farthest {row['max_radius_m']:.2f} m from the axis")
    out = os.path.join(REPO, "logs", "sim2real", "basket_waist_eval.json")
    json.dump({"policy": args.policy, "speed": args.speed, "rows": rows, "lines": lines}, open(out, "w"), indent=1)
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
