#!/usr/bin/env python3
"""Reach with the point-to-point arm policy (AppleArmPose-v1) driven by
waypoints, in the visible-reach world (AppleReachVisible-Play-v1: the tree,
the fixed random waist yaw, the head camera's view of the apples).

Waypoints, all with the palm normal along the horizontal line from the body
(torso / waist axis) to the apple:
  1. home      -- where every episode starts (the default arm pose);
  2. staging   -- at the apple's height, --stage_radius m from the body
                  axis horizontally (the same for every apple), on that line:
                  outside the cluster, so the hand rises / lowers there and
                  not between the fruit;
  3. apple     -- the grasp pocket (an enclosed apple's centre) on the apple
                  centre, moved --apple_offset m along the approach (negative
                  = stop short of the centre: calibrate it if the hand pushes);
  4. (--return_home) back to home after --hold_s.
The arm-pose policy gets them one at a time, as in training: a goal is
reached at 2 cm / 15 deg, held 0.2 s (via: 3 cm / 25 deg, no stop); after
--waypoint_timeout s the next one comes anyway. The hand is held in the
reach's cup (the arm-pose policy trained with it), the arm targets clamped to
the joint limits (as in training). The visible reach's own success
termination is off, so each episode runs the whole plan.

Printed per episode: when each waypoint was reached, the pocket error, and
how far the target and the other apples were pushed (max displacement from
where they hung when the plan started).

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/play_arm_pose_reach.py --headless --enable_cameras \\
        --video --episodes 4
"""
import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="AppleReachVisible-Play-v1", help="the world (a visible-reach task)")
parser.add_argument("--policy", type=str, default=None,
                    help="exported arm-pose policy (TorchScript); default: the newest "
                         "logs/rsl_rl/apple_arm_pose*/*/exported/policy.pt (v1 or v2)")
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--stage_radius", type=float, default=0.30,
                    help="staging point's horizontal distance from the body (waist) axis, m")
parser.add_argument("--min_stage_back", type=float, default=0.08,
                    help="staging point at least this far (horizontally) short of the apple goal, m")
parser.add_argument("--apple_offset", type=float, default=0.0,
                    help="apple goal along the approach direction from the apple centre, m (negative = short)")
parser.add_argument("--true_apple", action="store_true",
                    help="aim at the true apple centre (default: the perceived one, as on the robot)")
parser.add_argument("--stage_via", action="store_true",
                    help="fly through the staging point (via goal) instead of stopping there")
parser.add_argument("--fingers", choices=["free", "home"], default="free",
                    help="finger direction at the staging point / apple: free, or kept as at home")
parser.add_argument("--hold_s", type=float, default=1.5, help="time kept at the apple once reached, s")
parser.add_argument("--return_home", action="store_true")
parser.add_argument("--waypoint_timeout", type=float, default=4.0)
parser.add_argument("--settle_steps", type=int, default=10, help="steps held still after a reset")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=2000, help="max frames recorded (50 per second)")
parser.add_argument("--no_camera_view", action="store_true", help="no head-camera inset / frustum")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_arm_pose_env_cfg import POCKET  # noqa: E402
from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS  # noqa: E402
from envs.apple_reach_spindle_env_cfg import ARM_TARGET_CLIP, REACH_HAND_PRESHAPE  # noqa: E402
from envs.mdp import arm_pose as ap  # noqa: E402
from envs.mdp import terminations_and_events as te  # noqa: E402

# the arm-pose policy's goal test (AppleArmPose-v1 TerminationsCfg.goal_tracker)
TRACK = dict(pos_tol=0.02, ang_tol_deg=15.0, finger_tol_deg=25.0, speed_tol=0.08, hold_time=0.2,
             via_pos_tol=0.03, via_ang_tol_deg=25.0)


def find_policy() -> str:
    if args_cli.policy:
        return args_cli.policy
    runs = sorted(glob.glob(os.path.join(REPO, "logs", "rsl_rl", "apple_arm_pose*", "*", "exported", "policy.pt")),
                  key=os.path.getmtime)
    if not runs:
        raise SystemExit("no exported arm-pose policy found; pass --policy")
    return runs[-1]


class WaypointMarkers:
    """Every waypoint of the plan as a small white sphere (env 0); the
    current goal is drawn by arm_pose.GoalViz (orange / cyan + arrows)."""

    def __init__(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.m = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/ArmPoseWaypoints",
            markers={"wp": sim_utils.SphereCfg(radius=0.01, visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.95, 0.95)))}))

    def show(self, pts: torch.Tensor):
        self.m.visualize(translations=pts)


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    # the whole plan in one episode; the hand in the cup; the arm-pose policy's target clamp
    env_cfg.episode_length_s = 30.0
    if getattr(env_cfg.terminations, "reach_success", None) is not None:
        env_cfg.terminations.reach_success = None
    if getattr(env_cfg.events, "hand_schedule", None) is not None:
        env_cfg.events.hand_schedule = None
    # one hold_joint_targets event only (it caches its joint ids on the env)
    if getattr(env_cfg.events, "hand_preshape", None) is not None:
        env_cfg.events.hand_preshape.params = {**env_cfg.events.hand_preshape.params,
                                               "joint_targets": dict(REACH_HAND_PRESHAPE)}
    else:
        env_cfg.events.hand_preshape = EventTerm(func=te.hold_joint_targets, mode="reset",
                                                 params={"joint_targets": dict(REACH_HAND_PRESHAPE)})
    env_cfg.actions.right_arm.clip = dict(ARM_TARGET_CLIP)

    head_camera = None if args_cli.no_camera_view else getattr(env_cfg, "head_camera", None)
    head_sensor = None
    if head_camera is not None and args_cli.video:
        from envs.mdp.visible_reach import head_camera_sensor_cfg

        head_sensor = "head_camera_image"
        setattr(env_cfg.scene, head_sensor, head_camera_sensor_cfg(head_camera))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if head_camera is not None:
        from envs.mdp.visible_reach import camera_view_wrapper

        env = camera_view_wrapper(env, head_camera, sensor_name=head_sensor, scale=1)
    policy_path = find_policy()
    if args_cli.video:
        folder = os.path.join(os.path.dirname(os.path.dirname(policy_path)), "videos", "reach_waypoints")
        env = gym.wrappers.RecordVideo(env, video_folder=folder, step_trigger=lambda s: s == 0,
                                       video_length=args_cli.video_length, disable_logger=True,
                                       name_prefix=f"reach_r{int(args_cli.stage_radius * 100)}"
                                                   f"_off{int(round(args_cli.apple_offset * 1000))}mm")
        print(f"[INFO] video -> {folder}")
    uenv = env.unwrapped
    assert uenv.action_manager.total_action_dim == 7, "expected the 7-dim right-arm action only"
    print(f"[INFO] arm-pose policy: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=uenv.device).eval()

    robot = uenv.scene["robot"]
    dev = uenv.device
    arm = torch.tensor([robot.joint_names.index(n) for n in RIGHT_ARM_JOINTS], device=dev)
    term = uenv.action_manager.get_term("right_arm")
    torso = robot.data.body_names.index("torso_link")
    P = ap.pose_state(uenv)
    viz = ap.GoalViz(uenv)
    wpm = WaypointMarkers()

    def hold_action():
        return (robot.data.joint_pos[:, arm] - term._offset) / term._scale

    def observation(last):
        return torch.cat([
            robot.data.joint_pos[:, arm] - robot.data.default_joint_pos[:, arm],
            robot.data.joint_vel[:, arm],
            ap.obs_goal_delta(uenv, POCKET), ap.obs_goal_position(uenv, POCKET),
            ap.obs_goal_normal(uenv, POCKET), ap.obs_goal_finger(uenv, POCKET),
            ap.obs_goal_flags(uenv), last], dim=-1)

    def apples():
        T = getattr(uenv, "_tree", None)
        if T is None:
            return uenv.scene["apple"].data.root_pos_w[:1]
        return torch.stack([o.data.root_pos_w[0] for o in T.objs])

    def make_plan():
        """[(name, pocket, normal, finger, finger_on, via)] in the world, env 0."""
        pocket, normal, finger = ap.hand_frame(uenv, POCKET)
        home = ("home", pocket[0].clone(), normal[0].clone(), finger[0].clone(), True, False)
        a = uenv._apple_meas[0] if (not args_cli.true_apple and hasattr(uenv, "_apple_meas")) \
            else uenv.scene["apple"].data.root_pos_w[0]
        axis = robot.data.body_pos_w[0, torso]
        u = a - axis
        u[2] = 0.0
        d_apple = float(u.norm())
        u = u / u.norm().clamp(min=1e-6)
        goal = a + args_cli.apple_offset * u
        d_goal = float(((goal - axis) * u).sum())
        r = args_cli.stage_radius
        if r > d_goal - args_cli.min_stage_back:
            r = d_goal - args_cli.min_stage_back
            print(f"  [plan] apple only {d_goal:.2f} m out: staging point pulled in to {r:.2f} m")
        stage = axis + r * u
        stage[2] = goal[2]
        fin = home[3] if args_cli.fingers == "home" else finger[0]
        fon = args_cli.fingers == "home"
        plan = [("staging", stage, u.clone(), fin, fon, bool(args_cli.stage_via)),
                ("apple", goal, u.clone(), fin, fon, False)]
        if args_cli.return_home:
            plan.append(home)
        yaw = math.degrees(float(uenv._fixed_waist_yaw[0])) if hasattr(uenv, "_fixed_waist_yaw") else 0.0
        print(f"  [plan] waist yaw {yaw:+.0f} deg, apple {d_apple:.2f} m from the body axis, "
              f"{float(a[2] - axis[2]):+.2f} m vs torso; staging {r:.2f} m out")
        return plan

    def set_goal(wp):
        name, p, n, f, fon, via = wp
        P.target[0], P.normal[0], P.finger[0] = p, n, f
        P.finger_on[0], P.via[0] = fon, via
        P.start[0] = ap.hand_frame(uenv, POCKET)[0][0]
        P.pending[0] = False
        P.hold[0] = 0.0
        P.success[0] = False
        P.arrived[0] = False

    env.reset()
    last = torch.zeros(1, 7, device=dev)
    ep, step, k_settle = 0, 0, 0
    plan, wi, t_wp, t_hold, rest, report = None, 0, 0.0, None, None, []
    while app.is_running() and ep < args_cli.episodes and (not args_cli.video or step < args_cli.video_length):
        with torch.inference_mode():
            if plan is None:
                act = hold_action()
                P.pending[0] = True
            else:
                act = policy(observation(last))
            _, _, terminated, truncated, _ = env.step(act)
            last = act.clone()
            step += 1
            done = bool(terminated[0]) or bool(truncated[0])
            if plan is not None:
                viz.update()
            if done:
                tm = uenv.termination_manager
                reason = [k for k in tm.active_terms if bool(tm.get_term(k)[0])]
                print(f"[ep {ep}] ended by {reason} at waypoint {wi} ({plan[wi][0] if plan and wi < len(plan) else '-'})")
                ep += 1
                plan, k_settle, last = None, 0, torch.zeros(1, 7, device=dev)
                continue
            if plan is None:
                k_settle += 1
                if k_settle >= args_cli.settle_steps:
                    print(f"[ep {ep}]")
                    plan, wi, t_wp, t_hold = make_plan(), 0, 0.0, None
                    rest = apples().clone()
                    tgt_i = int((rest - uenv.scene["apple"].data.root_pos_w[0]).norm(dim=-1).argmin())
                    set_goal(plan[0])
                    wpm.show(torch.stack([w[1] for w in plan]))
                continue
            # goal test as in training
            ap.track_goal(uenv, POCKET, **TRACK)
            t_wp += uenv.step_dt
            name = plan[wi][0]
            d, ang, _, _ = ap.errors(uenv, POCKET)
            moved = (apples() - rest).norm(dim=-1)
            if name == "apple" and t_hold is not None:
                t_hold += uenv.step_dt
                if t_hold >= args_cli.hold_s:
                    others = torch.cat([moved[:tgt_i], moved[tgt_i + 1:]])
                    print(f"  apple held {args_cli.hold_s:.1f} s: pocket error {float(d[0]) * 100:.1f} cm, "
                          f"normal {math.degrees(float(ang[0])):.0f} deg; target pushed {float(moved[tgt_i]) * 100:.1f} cm, "
                          f"other apples max {float(others.max()) * 100 if len(others) else 0.0:.1f} cm")
                    wi += 1
                    t_hold = None
                    if wi < len(plan):
                        set_goal(plan[wi]); t_wp = 0.0
                    else:
                        ep += 1
                        env.reset()
                        plan, k_settle, last = None, 0, torch.zeros(1, 7, device=dev)
                continue
            if bool(P.success[0]) or t_wp >= args_cli.waypoint_timeout:
                ok = bool(P.success[0])
                print(f"  {name:8s} {'reached' if ok else 'TIMEOUT'} after {t_wp:.2f} s: "
                      f"error {float(d[0]) * 100:.1f} cm / {math.degrees(float(ang[0])):.0f} deg; "
                      f"apples moved max {float(moved.max()) * 100:.1f} cm")
                if name == "apple":
                    t_hold = 0.0
                    continue
                wi += 1
                if wi < len(plan):
                    set_goal(plan[wi]); t_wp = 0.0
            if wi >= len(plan):
                # plan done: next episode
                ep += 1
                env.reset()
                plan, k_settle, last = None, 0, torch.zeros(1, 7, device=dev)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
