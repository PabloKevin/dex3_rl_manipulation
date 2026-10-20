#!/usr/bin/env python3
"""Can the arm-pose policy (AppleArmPose-v1) start from a TUCKED home (the arm down by the hip, out of the
way of the apples while the waist turns) and come back to it?

Per candidate home (--homes, joint vectors; "old" = reach_io.HOME_Q, the arm in front of the belly), per
waist yaw (--yaws) and per staging point (--stages: the toy's staging -- 0.24 m from the torso axis at an
azimuth / a height in the torso frame, the palm normal along the horizontal torso -> point line), one env:
  leg 1  home -> staging (a stop goal, the fingers free)
  leg 2  staging -> home (the home's own pocket / normal / fingers, as toy_arm_pose's home goal)
each with the toy's budget (4 / speed s). Recorded: reached and when, final pocket / normal error, body
overlap (the training's sphere model) deeper than the home pose's own reading and near misses (3 cm), the arm's largest horizontal radius from the
torso axis at home (what sweeps the apples when the waist turns).

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/eval_harvest_home.py --headless \\
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
parser.add_argument("--homes", nargs="+", default=["old", "0.2,-0.4,0.8,1.4,0,0,0", "0.3,-0.6,0.8,1.0,0,0,0",
                                                    "0.1,-0.35,0.8,1.5,0,0,0"])
parser.add_argument("--yaws", type=float, nargs="+", default=[-45, -15, 15], help="waist yaw, deg")
parser.add_argument("--stages", nargs="+", default=["-35,-0.10", "-35,0.10", "-10,-0.05", "-10,0.15", "10,0.0"],
                    help="'azimuth deg,height m' in the torso frame (0.24 m from its axis)")
parser.add_argument("--stage_radius", type=float, default=0.24)
parser.add_argument("--speed", type=float, default=0.3)
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

OLD_HOME = [-0.15, -0.15, -0.15, 0.5, 0.0, 0.0, 0.0]


def main():
    homes = [OLD_HOME if h == "old" else [float(v) for v in h.split(",")] for h in args.homes]
    stages = [tuple(float(v) for v in s.split(",")) for s in args.stages]
    combos = [(hi, yi, si) for hi in range(len(homes)) for yi in range(len(args.yaws)) for si in range(len(stages))]
    N = len(combos)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=N)
    env_cfg.episode_length_s = 120.0
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
    env.reset()

    q_home = torch.tensor([homes[c[0]] for c in combos], device=dev)
    yaw = torch.tensor([math.radians(args.yaws[c[1]]) for c in combos], device=dev)

    def place(q_arm, yaw_):
        jp = robot.data.joint_pos.clone()
        jp[:, arm] = q_arm
        jp[:, wj] = yaw_
        robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        tgt = robot.data.joint_pos_target.clone()
        tgt[:, arm] = jp[:, arm]
        tgt[:, wj] = yaw_
        robot.set_joint_position_target(tgt)

    def hold_action():
        return (robot.data.joint_pos[:, arm] - term._offset) / term._scale

    place(q_home, yaw)
    for _ in range(args.settle_steps):
        P.pending[:] = True
        env.step(hold_action())
    # the home pose's own reading of the sphere model (the start of leg 1, the end of leg 2)
    depth_home = scm.self_collision_depth(env, margin=0.0).amax(-1).clone()
    # the home goal: the pocket / normal / fingers at home (world)
    home_p, home_n, home_f = (x.clone() for x in ap.hand_frame(env, POCKET))
    # the arm's sweep radius at home: the largest horizontal distance of a hand / forearm point from the torso axis
    tq, tp = robot.data.body_quat_w[:, torso], robot.data.body_pos_w[:, torso]
    pts = [robot.data.body_pos_w[:, robot.data.body_names.index(b)] for b in
           ("right_elbow_link", "right_wrist_yaw_link", "right_hand_palm_link", "right_hand_index_1_link",
            "right_hand_middle_1_link", "right_hand_thumb_2_link")] + [home_p]
    rad = torch.stack([quat_apply_inverse(tq, p - tp)[:, :2].norm(dim=-1) for p in pts], -1).amax(-1)
    # the staging goals (torso frame at this yaw -> world)
    st_p, st_n = [], []
    for c in combos:
        az, h = stages[c[2]]
        u = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
        st_p.append(np.array([0.0, 0.0, h]) + args.stage_radius * u)
        st_n.append(u)
    st_p = tp + quat_apply(tq, torch.tensor(np.array(st_p), device=dev, dtype=torch.float32))
    st_n = quat_apply(tq, torch.tensor(np.array(st_n), device=dev, dtype=torch.float32))

    def set_goal(ids, p, n, f, f_on):
        P.target[ids] = p[ids]
        P.normal[ids] = n[ids]
        P.finger[ids] = f[ids]
        P.finger_on[ids] = f_on
        P.via[ids] = False
        P.pending[ids] = False
        P.hold[ids] = 0.0
        P.arrived[ids] = False
        P.success[ids] = False

    every = torch.arange(N, device=dev)
    set_goal(every, st_p, st_n, home_f, False)
    leg = torch.zeros(N, dtype=torch.long, device=dev)
    t_leg = torch.zeros(N, device=dev)
    timeout = 4.0 / args.speed
    res = {k: torch.full((N,), float("nan"), device=dev) for k in ("t1", "t2", "d1", "d2", "a1", "a2")}
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    depth_max = torch.zeros(N, device=dev)
    near = torch.zeros(N, dtype=torch.bool, device=dev)
    last = torch.zeros(N, 7, device=dev)
    steps = int(round(2 * timeout / env.step_dt)) + 10
    for _ in range(steps):
        obs = torch.cat([robot.data.joint_pos[:, arm] - robot.data.default_joint_pos[:, arm],
                         robot.data.joint_vel[:, arm], ap.obs_goal_delta(env, POCKET), ap.obs_goal_position(env, POCKET),
                         ap.obs_goal_normal(env, POCKET), ap.obs_goal_finger(env, POCKET), ap.obs_goal_flags(env),
                         last], dim=-1)
        with torch.inference_mode():
            act = pol(obs)
        act = torch.where((leg >= 2)[:, None], hold_action(), act)
        env.step(act)
        last = act.clone()
        ap.track_goal(env, POCKET)
        t_leg += env.step_dt
        live = leg < 2
        dep = scm.self_collision_depth(env, margin=0.0).amax(-1)
        depth_max = torch.where(live, torch.maximum(depth_max, dep), depth_max)
        # an overlap deeper than the home pose's own (the sphere model is conservative by the hip)
        hit |= live & (dep > depth_home + 0.002)
        near |= live & (scm.self_collision_depth(env, margin=0.03).amax(-1) > 0.0)
        d, ang, _, _ = ap.errors(env, POCKET)
        done = live & (P.arrived | (t_leg >= timeout))
        for L, (kt, kd, ka) in enumerate((("t1", "d1", "a1"), ("t2", "d2", "a2"))):
            m = done & (leg == L)
            if m.any():
                res[kt] = torch.where(m & P.arrived, t_leg, res[kt])
                res[kd] = torch.where(m, d * 100, res[kd])
                res[ka] = torch.where(m, torch.rad2deg(ang), res[ka])
        to_home = done & (leg == 0)
        if to_home.any():
            ids = to_home.nonzero().squeeze(-1)
            set_goal(ids, home_p, home_n, home_f, True)
        leg = torch.where(done, leg + 1, leg)
        t_leg = torch.where(done, torch.zeros_like(t_leg), t_leg)
        if bool((leg >= 2).all()):
            break
    out_rows = []
    print(f"\n[harvest home] speed {args.speed}, {timeout:.1f} s per leg; per home: "
          f"{len(args.yaws)} yaws x {len(stages)} staging points")
    print(" home                          sweep r   ->staging  t s   cm    |  ->home  t s   cm   deg  | overlap  <3cm")
    for hi, h in enumerate(homes):
        sel = torch.tensor([c[0] == hi for c in combos], device=dev)
        r1 = ~torch.isnan(res["t1"][sel])
        r2 = ~torch.isnan(res["t2"][sel])
        n = int(sel.sum())
        row = dict(home=h, sweep_r=float(rad[sel].mean()), reach1=int(r1.sum()), reach2=int(r2.sum()), n=n,
                   t1=float(res["t1"][sel][r1].mean()) if r1.any() else float("nan"),
                   t2=float(res["t2"][sel][r2].mean()) if r2.any() else float("nan"),
                   d1=float(res["d1"][sel].nanmean()), d2=float(res["d2"][sel].nanmean()),
                   a2=float(res["a2"][sel].nanmean()), hit=int(hit[sel].sum()), near=int(near[sel].sum()),
                   depth_home_cm=float(depth_home[sel].max()) * 100, depth_max_cm=float(depth_max[sel].max()) * 100)
        out_rows.append(row)
        name = "old " + str(h) if h == OLD_HOME else str(h)
        print(f" {name[:30]:30s} {row['sweep_r']:.2f} m   {row['reach1']:2d}/{n}  {row['t1']:4.1f}  {row['d1']:4.1f}  |  "
              f"{row['reach2']:2d}/{n}  {row['t2']:4.1f}  {row['d2']:4.1f}  {row['a2']:4.0f}  |  {row['hit']:2d}     {row['near']:2d}   (sphere model at home {row['depth_home_cm']:.1f} cm, worst while moving {row['depth_max_cm']:.1f} cm)")
        for yi, y in enumerate(args.yaws):
            for si, s_ in enumerate(stages):
                i = combos.index((hi, yi, si))
                print(f"     yaw {y:+4.0f} stage {s_}: ->staging {'ok' if not math.isnan(res['t1'][i]) else 'NO'} "
                      f"{res['d1'][i]:.1f} cm, ->home {'ok' if not math.isnan(res['t2'][i]) else 'NO'} "
                      f"{res['d2'][i]:.1f} cm {res['a2'][i]:.0f} deg{f'  OVERLAP {depth_max[i] * 100:.1f} cm' if hit[i] else ''}")
    out = os.path.join(REPO, "logs", "sim2real", "harvest_home_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"policy": args.policy, "speed": args.speed, "rows": out_rows}, open(out, "w"), indent=1)
    print(f"-> {out}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
