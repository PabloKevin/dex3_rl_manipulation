#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Harvest TEST BENCH: does the trained grasp keep the apple in the hand --
without bruising it -- through a harvest and arbitrary arm/wrist motion?

Nothing is trained here. Per env, a state machine:
  GRASP    frozen reach policy, then the frozen grasp policy (AppleGrasp-v1);
           the arm's joint path is recorded from the start of the episode
  (READY)  envs.mdp.harvest.GraspReadyDetector: closure + thumb pressure +
           settled fingers for 0.5 s (all real-hand signals). The finger
           joint TARGETS freeze there for the rest of the episode (the hand
           PD keeps squeezing), and a breakable stem (envs.mdp.harvest.
           BreakableStem) replaces the grasp phase's soft pin, along the real
           stem direction (env._stem_dir)
  TWIST    the hand + fruit rotate about the PALM NORMAL through the grasp
           pocket -- what the robot knows from its own kinematics; the
           approach makes it roughly the stem axis -- (IK-tracked, null-space
           pull to the grasp-time arm pose), up to --twist_deg, stopping as
           soon as the stem breaks. The simulated stem keeps its TRUE axis,
           so this tests whether the imperfect twist still breaks it; torsion
           needs far less force than a straight pull (~0.1 N*m vs 15-25 N)
  HOME     the recorded REACH path (not the grasp phase) played backwards,
           smoothed, re-timed to --home_speed rad/s (harvest.retime_joint_path),
           with the twist undone gradually along the way
  MOVES    --moves random arm poses from the safe-pose library
           (scripts/build_harvest_pose_library.py: arm, hand AND held apple
           clear of the body, palm in front, any palm orientation), palm kept
           --tree_clearance from the trunk/branch, each reached by a smooth
           joint-space move (quintic, peak --move_speed rad/s) and followed by
           --circle_s s of joint-space circles: wrist roll/pitch/yaw and
           shoulder/elbow sinusoids (--wrist_amp / --arm_amp rad)
  BASKET   home, then the fixed planned path (scripts/plan_basket_drop.py:
           every point clear of the body and the basket) up over the basket
           and down to the palm-down drop pose (envs/mdp/basket.py: slatted
           basket on a stand, robot's right, behind the shoulder line)
  RELEASE  open the hand, wait --release_s, check the apple is in the basket
  (--no_basket: HOLD --hold_after s still instead)

Report per env: grasp-ready time, stem broke (cause, twist angle at break),
apple kept in the hand throughout (fruit within --leave_margin of the grasp
pocket), the phase it fell in, max slip in the palm frame, peak contact force
on the fruit vs. that env's bruising threshold, whether it ended in the
basket (FINISHED = held through every motion AND in the basket), arm/hand
intrusion into the basket, self-collision overlap per phase (arm/hand and the
held apple vs. the body, same sphere model as training), tree intrusion after
going home.

Rounds: when every env is done the table is printed and all envs reset;
'r' + Enter ends the round early. --rounds 0 runs until the window closes.

Usage:
  python scripts/harvest_test_bench.py --num_envs 4                       (viewer)
  python scripts/harvest_test_bench.py --headless --num_envs 4 --rounds 1 --video logs/videos/harvest.mp4
                                          (real-speed video, no screen capture needed; --cam_eye/--cam_lookat to re-aim)
  python scripts/harvest_test_bench.py --headless --num_envs 32 --rounds 2
  python scripts/harvest_test_bench.py --headless --policy <run>/exported/policy.pt
"""
import argparse
import glob
import math
import os
import sys
import threading

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="Harvest test bench for the grasp policy",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--policy", type=str, default="last",
               help="JIT-exported grasp policy.pt, or 'last' for the newest logs/rsl_rl/apple_grasp_grasp/*/exported/policy.pt")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--rounds", type=int, default=0, help="rounds before exiting (0 = until the window is closed)")
p.add_argument("--grasp_timeout", type=float, default=5.0,
               help="seconds after the hand-off; a grasp not READY by then is harvested anyway, labelled 'timeout-ready'")
p.add_argument("--twist_deg", type=float, default=90.0, help="maximum twist; it stops as soon as the stem breaks")
p.add_argument("--twist_s", type=float, default=1.5, help="seconds for the full --twist_deg")
p.add_argument("--home_speed", type=float, default=1.2, help="peak joint-space speed (rad/s) going home")
p.add_argument("--moves", type=int, default=4)
p.add_argument("--move_speed", type=float, default=1.0, help="peak joint-space speed (rad/s) between poses")
p.add_argument("--max_jump", type=float, default=1.2, help="next pose at most this far in joint space (rad)")
p.add_argument("--circle_hz", type=float, default=1.0)
p.add_argument("--circle_s", type=float, default=2.0)
p.add_argument("--wrist_amp", type=float, default=0.3, help="wrist roll/pitch/yaw circle amplitude (rad)")
p.add_argument("--arm_amp", type=float, default=0.08, help="shoulder pitch / elbow circle amplitude (rad)")
p.add_argument("--library", type=str, default=None, help="pose library (default envs/assets/harvest_pose_library.npz)")
p.add_argument("--tree_clearance", type=float, default=0.15, help="waypoints at least this far from trunk/branch (m)")
p.add_argument("--hold_after", type=float, default=0.5, help="(--no_basket only) seconds to hold still at the end")
p.add_argument("--no_basket", action="store_true", help="no basket: end with a still HOLD instead of the drop")
p.add_argument("--basket_path", type=str, default=None, help="default envs/assets/basket_drop_path.npz")
p.add_argument("--drop_speed", type=float, default=0.4, help="peak joint speed (rad/s) for the last descent to the drop pose")
p.add_argument("--release_s", type=float, default=1.5, help="seconds after opening the hand before checking the basket")
p.add_argument("--leave_margin", type=float, default=0.06, help="fruit this far from the grasp pocket = dropped")
p.add_argument("--video", type=str, default=None,
               help="write an mp4 here, timed in SIMULATED time (plays at true speed however slow the sim runs); "
                    "Default camera: from the robots' RIGHT side (the arm they use), framing every env. Works headless.")
p.add_argument("--video_fps", type=int, default=25)
p.add_argument("--video_res", type=int, nargs=2, default=[1280, 720])
p.add_argument("--cam_eye", type=float, nargs=3, default=None,
               help="video camera position, relative to the centre of the env grid (1 env: env 0's origin)")
p.add_argument("--cam_lookat", type=float, nargs=3, default=None, help="video camera target, same frame as --cam_eye")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True          # needed for any render product
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401  (registers the gym tasks)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (axis_angle_from_quat, quat_apply, quat_apply_inverse, quat_conjugate,
                                 quat_from_angle_axis, quat_from_matrix, quat_mul)
from envs.apple_grasp_grasp_env_cfg import AppleGraspGraspEnvCfg, GRASP_CENTER_LOCAL
from envs.mdp import harvest as hv
from envs.mdp.grasp_rewards import _max_link_apple_force
from envs.mdp.self_collision import self_collision_depth, tree_contact_depth

# ── env: the grasp task with the stem never released and nothing ending episodes ──
cfg = AppleGraspGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
if not args.no_basket:
    from envs.mdp import basket as bk
    bk.add_to_scene(cfg.scene)
cfg.episode_length_s = 600.0
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.events.reset_reach_grasp_phase.params.update(
    stem_release_schedule=None, stem_release_time_range=None, release_on_latch=False, shaping_grace_s=None)
if args.video:
    cfg.viewer.resolution = tuple(args.video_res)
env = gym.make("AppleGrasp-v1", cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
N, dev, dt = env.num_envs, env.device, env.step_dt
video = None
if args.video:
    # From the robots' RIGHT side, slightly raised: they all use the right
    # arm, so from their left the torso hides the hand and apple, and from the
    # front the trunks (in front of each robot) block the view. Framed on the
    # centre of the env grid; pulled back further for more envs.
    centre = env.scene.env_origins.mean(0).cpu().numpy()
    if N == 1:
        eye, look = (0.35, -1.7, 1.55), (0.3, -0.2, 1.05)
    else:
        spread = float(env.scene.env_origins[:, 1].max() - env.scene.env_origins[:, 1].min())
        eye, look = (0.35, -2.9 - 0.9 * spread, 1.9), (0.35, 0.0, 1.0)
    eye = args.cam_eye or eye
    look = args.cam_lookat or look
    env.viewport_camera_controller.update_view_location(
        eye=tuple(float(c + e) for c, e in zip(centre, eye)), lookat=tuple(float(c + l) for c, l in zip(centre, look)))
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    video = imageio.get_writer(args.video, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=8)
    video_every = max(1, round(1.0 / (args.video_fps * dt)))      # control steps per frame

if args.policy == "last":
    cands = sorted(glob.glob(os.path.join(REPO, "logs/rsl_rl/apple_grasp_grasp/*/exported/policy.pt")), key=os.path.getmtime)
    if not cands:
        sys.exit("no exported grasp policy found -- run scripts/play.py --task AppleGrasp-Play-v1 once, or pass --policy")
    args.policy = cands[-1]
_jit = torch.jit.load(args.policy, map_location=dev).eval()
policy = lambda: _jit(env.observation_manager.compute()["policy"])

robot = env.scene["robot"]
apple = env.scene["apple"]
arm = env.action_manager.get_term("right_arm")
hand = env.action_manager.get_term("right_hand")
arm_dim = arm.action_dim
palm_id = robot.data.body_names.index("right_hand_palm_link")
arm_ids = list(arm._joint_ids)
sc = SceneEntityCfg("contact_forces"); sc.resolve(env.scene)

stem = hv.BreakableStem(env)
ready_det = hv.GraspReadyDetector(env, sc, hand._joint_ids)
ik = hv.DifferentialIK(robot, "right_hand_palm_link", arm_ids)

_orig_apply = hand.apply_actions
def _apply_with_stem():          # stem wrench every PHYSICS substep
    _orig_apply()
    stem.apply()
hand.apply_actions = _apply_with_stem

lib = np.load(args.library or os.path.join(REPO, "envs", "assets", "harvest_pose_library.npz"))
lib_q = torch.tensor(lib["q"], device=dev, dtype=torch.float32)                 # [M, 7], RIGHT_ARM_JOINTS order
lib_p = torch.tensor(lib["palm_local"], device=dev, dtype=torch.float32)         # [M, 3], env-local
from envs.apple_grasp_grasp_env_cfg import RIGHT_ARM_JOINTS
_term_order = [robot.joint_names[j] for j in arm_ids]
lib_q = lib_q[:, [RIGHT_ARM_JOINTS.index(n) for n in _term_order]]            # -> action-term joint order
_lo = robot.data.soft_joint_pos_limits[0, arm_ids, 0]; _hi = robot.data.soft_joint_pos_limits[0, arm_ids, 1]
_wrist = [_term_order.index(n) for n in ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
_sp, _el = _term_order.index("right_shoulder_pitch_joint"), _term_order.index("right_elbow_joint")
c_local = torch.tensor(GRASP_CENTER_LOCAL, device=dev).expand(N, 3)
up = torch.tensor([0.0, 0.0, 1.0], device=dev)

GRASP, TWIST, HOME, MOVE, CIRCLE, HOLD, DONE, BASKET, RELEASE = range(9)
PHASE = {GRASP: "grasp", TWIST: "twist", HOME: "home", MOVE: "move", CIRCLE: "circle", HOLD: "hold", DONE: "done",
         BASKET: "basket", RELEASE: "release"}

if not args.no_basket:
    _bp = np.load(args.basket_path or os.path.join(REPO, "envs", "assets", "basket_drop_path.npz"))
    from envs.apple_grasp_grasp_env_cfg import RIGHT_ARM_JOINTS as _RAJ
    _order = [_RAJ.index(robot.joint_names[j]) for j in arm_ids]
    basket_wps = torch.tensor(_bp["waypoints"], device=dev, dtype=torch.float32)[:, _order]   # home, ..., drop
    hand_open = robot.data.default_joint_pos[:, hand._joint_ids]


def fruit_to_hand():
    gp = robot.data.body_pos_w[:, palm_id] + quat_apply(robot.data.body_quat_w[:, palm_id], c_local)
    return torch.norm(apple.data.root_pos_w - gp, dim=-1)


def fruit_in_palm():
    return quat_apply_inverse(robot.data.body_quat_w[:, palm_id], apple.data.root_pos_w - robot.data.body_pos_w[:, palm_id])


def apple_body_clearance():
    """Held apple vs. the body spheres: min surface gap (m; < margin = too close)."""
    from envs.mdp import self_collision as scm
    if not hasattr(env, "_self_collision_spheres"):
        scm.self_collision_depth(env)
    sets = env._self_collision_spheres
    body = scm._world_spheres(env, sets["body"])
    rad = env._apple_radius if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) else torch.full((N,), 0.04, device=dev)
    d = (body - apple.data.root_pos_w.unsqueeze(1)).norm(dim=-1) - sets["body"]["r"].view(1, -1)
    return d.amin(-1) - rad


def pick_pose(i, q_from):
    """A library pose for env i: within --max_jump of q_from, palm clear of
    this env's trunk and branch."""
    seg = (env._tree_segments[i, 1:] - env.scene.env_origins[i]).cpu().numpy()      # branch, trunk
    jump = (lib_q - q_from).norm(dim=-1)
    cand = (jump <= args.max_jump).nonzero(as_tuple=True)[0]
    if len(cand) == 0:
        cand = jump.argsort()[:50]
    for k in cand[torch.randperm(len(cand), device=dev)][:200].tolist():
        pos = lib_p[k].cpu().numpy()
        clear = True
        for a_, b_ in seg:
            ab = b_ - a_
            t = np.clip(np.dot(pos - a_, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            clear &= np.linalg.norm(pos - (a_ + t * ab)) >= args.tree_clearance
        if clear:
            return lib_q[k]
    return lib_q[int(cand[0])]


def new_round():
    global S
    S = dict(
        state=torch.full((N,), GRASP, dtype=torch.long, device=dev), t=torch.zeros(N, device=dev),
        path=[[] for _ in range(N)], handoff_idx=[None] * N, home=[None] * N, home_k=[0] * N, twist_dq=[None] * N,
        q_ready=[None] * N, tw_axis=torch.zeros(N, 3, device=dev), tw_pivot=torch.zeros(N, 3, device=dev),
        hand_hold=torch.zeros(N, hand.action_dim, device=dev),
        palm0_p=torch.zeros(N, 3, device=dev), palm0_q=torch.zeros(N, 4, device=dev),
        q_from=torch.zeros(N, len(arm_ids), device=dev), q_to=torch.zeros(N, len(arm_ids), device=dev),
        seg_T=torch.ones(N, device=dev), move_k=[0] * N,
        t_ready=torch.full((N,), float("nan"), device=dev), fruit0=torch.zeros(N, 3, device=dev),
        drop_phase=["" for _ in range(N)], twist_at_break=torch.full((N,), float("nan"), device=dev),
        max_slip=torch.zeros(N, device=dev), max_f=torch.zeros(N, device=dev),
        max_self={k: torch.zeros(N, device=dev) for k in ("twist", "home", "moves", "basket")},
        chain=[None] * N, seg=[0] * N, in_basket=[None] * N, max_basket=torch.zeros(N, device=dev),
        min_apple_gap=torch.full((N,), float("inf"), device=dev), max_tree=torch.zeros(N, device=dev),
        max_ik=torch.zeros(N, device=dev),
        outcome=["" for _ in range(N)], not_ready=["" for _ in range(N)],
    )
    stem.active[:] = False; stem.broken[:] = False; stem.break_cause[:] = 0
    for t in (stem.peak_pull, stem.peak_bend, stem.peak_twist):
        t[:] = 0.0
    ready_det.reset(slice(None))


def start_move(i):
    q0 = arm._processed_actions[i].clone()
    q1 = pick_pose(i, q0)
    S["q_from"][i], S["q_to"][i] = q0, q1
    S["seg_T"][i] = max(1.0, 1.875 * float((q1 - q0).norm()) / args.move_speed)
    S["state"][i] = MOVE; S["t"][i] = 0.0


def start_basket(i):
    """Current pose -> home -> the planned waypoints to the drop pose."""
    S["chain"][i] = [arm._processed_actions[i].clone()] + list(basket_wps)
    S["seg"][i] = 0
    next_segment(i)
    S["state"][i] = BASKET


def next_segment(i):
    k = S["seg"][i]
    q0, q1 = S["chain"][i][k], S["chain"][i][k + 1]
    last = k + 1 == len(S["chain"][i]) - 1
    speed = args.drop_speed if last else args.move_speed
    S["q_from"][i], S["q_to"][i] = q0, q1
    S["seg_T"][i] = max(0.8, 1.875 * float((q1 - q0).norm()) / speed)
    S["t"][i] = 0.0


def basket_depth_arm():
    """Arm/hand sphere intrusion into the stand + basket (m, actual overlap)."""
    from envs.mdp import self_collision as scm
    sets = env._self_collision_spheres
    c = scm._world_spheres(env, sets["arm"]) - env.scene.env_origins.unsqueeze(1)
    return bk.basket_intrusion(c, sets["arm"]["r"], margin=0.0).amax(-1)


def start_circle(i):
    S["state"][i] = CIRCLE; S["t"][i] = 0.0


fmt = lambda x, s="{:.2f}": "-" if x != x else s.format(x)


def report(rnd, early=False):
    from collections import Counter
    dmg = getattr(env, "_fruit_damage_force", torch.full((N,), float("nan"), device=dev))
    print(f"\n===== ROUND {rnd}" + (" (ended early)" if early else "") + f" =====   policy: {args.policy}")
    print(" env ready_s stem   cause twist@brk  held  fell_in  max_slip  peak_F (bruise thr)  self-overlap mm tw/home/moves/bskt  apple gap  tree mm  basket(in, arm mm)  -> outcome")
    for i in range(N):
        st = int(S["state"][i])
        if not S["outcome"][i]:
            if st == GRASP:
                S["outcome"][i] = "NOT READY" if not early else "NOT READY (cut)"
            elif S["drop_phase"][i]:
                S["outcome"][i] = f"DROPPED in {S['drop_phase'][i]}"
            elif st != DONE:
                S["outcome"][i] = f"INTERRUPTED in {PHASE[st]}"
            elif args.no_basket:
                S["outcome"][i] = "HELD" if stem.broken[i] else "HELD, STEM INTACT?"
            else:
                S["outcome"][i] = "FINISHED" if S["in_basket"][i] else "HELD, MISSED BASKET"
            if S["outcome"][i] in ("HELD", "FINISHED") and S["max_f"][i] > dmg[i]:
                S["outcome"][i] += " but BRUISED"
            if S["not_ready"][i] and st != GRASP:
                S["outcome"][i] += " (timeout-ready)"
        print(f" {i:3d} {fmt(S['t_ready'][i].item()):>6s}  {'broke' if stem.broken[i] else 'intact':6s} "
              f"{hv.BREAK_CAUSES[int(stem.break_cause[i])]:5s} {fmt(S['twist_at_break'][i].item(), '{:.0f}deg'):>7s}   "
              f"{'NO' if S['drop_phase'][i] else 'yes':4s} {S['drop_phase'][i] or '-':7s} {S['max_slip'][i] * 100:5.1f} cm  "
              f"{S['max_f'][i]:6.1f} N ({dmg[i]:4.1f})     "
              + "/".join(f"{S['max_self'][k][i] * 1000:3.0f}" for k in ("twist", "home", "moves", "basket"))
              + f"        {fmt(S['min_apple_gap'][i].item() * 100, '{:+.1f}cm'):>7s}  {S['max_tree'][i] * 1000:3.0f}      "
              + f"{'-' if S['in_basket'][i] is None else ('yes' if S['in_basket'][i] else 'NO'):>3s} {S['max_basket'][i] * 1000:4.0f}"
              + f"          -> {S['outcome'][i]}")
    c = Counter(S["outcome"])
    print(f"\nsummary: {dict(c)}")
    for label, grp in (("detector-READY", [i for i in range(N) if not S["not_ready"][i] and not S["outcome"][i].startswith("NOT READY")]),
                       ("timeout-ready", [i for i in range(N) if S["not_ready"][i]])):
        if grp:
            held = sum(1 for i in grp if S["outcome"][i].startswith(("HELD", "FINISHED")))
            fin = sum(1 for i in grp if S["outcome"][i].startswith("FINISHED"))
            print(f"  {label:15s}: {held}/{len(grp)} kept the apple through every motion"
                  + ("" if args.no_basket else f", {fin}/{len(grp)} FINISHED in the basket") + ", median max slip "
                  f"{np.median([float(S['max_slip'][i]) for i in grp]) * 100:.1f} cm, median peak force "
                  f"{np.median([float(S['max_f'][i]) for i in grp]):.1f} N")
    for i in range(N):
        if S["not_ready"][i]:
            print(f"   env {i} not READY at timeout: {S['not_ready'][i]}")
    tally.update(S["outcome"])
    print(f"  running tally: {dict(tally)}")
    sys.stdout.flush()


_reset_requested = False
_reset_lock = threading.Lock()


def _listen():
    global _reset_requested
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.strip().lower() == "r":
            with _reset_lock:
                _reset_requested = True


threading.Thread(target=_listen, daemon=True).start()

from collections import Counter
tally = Counter()
env.reset()
new_round()
print(f"[bench] {N} envs, policy {args.policy}", flush=True)
print(f"[bench] twist {args.twist_deg} deg / {args.twist_s} s, home at {args.home_speed} rad/s, {args.moves} moves "
      f"(library of {len(lib_q)} poses, <= {args.move_speed} rad/s) + {args.circle_s} s circles @ {args.circle_hz} Hz "
      f"(wrist {args.wrist_amp} rad, arm {args.arm_amp} rad)", flush=True)
rnd, step = 1, 0
twist_rad = math.radians(args.twist_deg)
import time
_wall0, _step0 = time.time(), 0
while app.is_running():
    with torch.inference_mode():
        a = policy().clone()
        st = S["state"]
        since_handoff = (env.episode_length_buf - env._grasp_handoff_step).float() * dt

        # record the arm path while grasping (for the way home)
        q_now = robot.data.joint_pos[:, arm_ids]
        for i in (st == GRASP).nonzero(as_tuple=True)[0].tolist():
            S["path"][i].append(q_now[i].clone())
            if S["handoff_idx"][i] is None and bool(env._grasp_phase_active[i]):
                S["handoff_idx"][i] = len(S["path"][i])          # reach part ends here

        ready = ready_det.update() & (st == GRASP)
        timeout = (st == GRASP) & env._grasp_phase_active & (since_handoff > args.grasp_timeout) & ~ready
        if bool(timeout.any()):
            L = ready_det.last
            for i in timeout.nonzero(as_tuple=True)[0].tolist():
                S["not_ready"][i] = (f"closure {int(L['closure'][i])}, thumb {float(L['thumb_N'][i]):.1f} N, "
                                     f"finger span {float(L['max_dq'][i]):.2f} rad")
            ready = ready | timeout
        # GRASP -> READY -> TWIST
        for i in ready.nonzero(as_tuple=True)[0].tolist():
            S["t_ready"][i] = since_handoff[i]
            S["hand_hold"][i] = hand._processed_actions[i]
            S["palm0_p"][i] = robot.data.body_pos_w[i, palm_id]
            S["palm0_q"][i] = robot.data.body_quat_w[i, palm_id]
            S["q_ready"][i] = arm._processed_actions[i].clone()
            S["fruit0"][i] = fruit_in_palm()[i]
            # Twist axis/pivot the ROBOT knows: its palm normal, through the
            # grasp pocket (forward kinematics only). The simulated stem keeps
            # its true axis, so this also tests whether that imperfect twist
            # still breaks it.
            pq_i = robot.data.body_quat_w[i:i + 1, palm_id]
            S["tw_axis"][i] = quat_apply(pq_i, torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
            S["tw_pivot"][i] = robot.data.body_pos_w[i, palm_id] + quat_apply(pq_i, c_local[:1])[0]
            env._stem_released[i] = True                      # soft grasp pin off, arm unlocked
            st[i] = TWIST; S["t"][i] = 0.0
        if bool(ready.any()):
            ids = ready.nonzero(as_tuple=True)[0]
            stem.engage(ids, axis_w=env._stem_dir[ids])
        # targets for every harvesting env
        q_cmd = arm._processed_actions.clone()
        pose_p, pose_q = None, None
        tw = st == TWIST
        if bool(tw.any()):
            frac = hv.smoothstep(S["t"] / args.twist_s)
            zero = torch.zeros(N, device=dev)
            pose_p, pose_q = hv.harvest_palm_pose(S["palm0_p"], S["palm0_q"], S["tw_pivot"], S["tw_axis"],
                                                  torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(N, 3), zero, twist_rad * frac)
        mv, ci = (st == MOVE) | (st == BASKET), st == CIRCLE
        if bool(mv.any()):
            f = hv.smoothstep(S["t"] / S["seg_T"]).unsqueeze(-1)
            q_cmd = torch.where(mv.unsqueeze(-1), S["q_from"] + (S["q_to"] - S["q_from"]) * f, q_cmd)
        if bool(ci.any()):
            w = (2 * math.pi * args.circle_hz * S["t"]).unsqueeze(-1)
            env_amp = (hv.smoothstep(S["t"] / 0.3) * hv.smoothstep((args.circle_s - S["t"]) / 0.3)).unsqueeze(-1)
            dq = torch.zeros(N, len(arm_ids), device=dev)
            for j, ph in zip(_wrist, (0.0, math.pi / 2, math.pi)):
                dq[:, j] = args.wrist_amp * (torch.sin(w[:, 0] + ph))
            dq[:, _sp] = args.arm_amp * torch.sin(w[:, 0])
            dq[:, _el] = args.arm_amp * (torch.cos(w[:, 0]) - 1.0)
            q_c = torch.clamp(S["q_to"] + env_amp * dq, _lo, _hi)
            q_cmd = torch.where(ci.unsqueeze(-1), q_c, q_cmd)
        if bool(tw.any()):
            q_ready = torch.stack([S["q_ready"][i] if S["q_ready"][i] is not None else arm._processed_actions[i]
                                   for i in range(N)])
            q_ik = ik.step(pose_p, pose_q, base=arm._processed_actions, max_dq=0.05,
                           null_target=q_ready, null_gain=0.2)
            q_cmd = torch.where(tw.unsqueeze(-1), q_ik, q_cmd)
        for i in (st == HOME).nonzero(as_tuple=True)[0].tolist():
            k = S["home_k"][i]
            fade = 1.0 - float(hv.smoothstep(torch.tensor(k / max(len(S["home"][i]) - 1, 1))))
            q_cmd[i] = S["home"][i][min(k, len(S["home"][i]) - 1)] + fade * S["twist_dq"][i]
            S["home_k"][i] = k + 1
        rl = st == RELEASE
        if bool(rl.any()):
            q_cmd = torch.where(rl.unsqueeze(-1), S["q_to"], q_cmd)
        h = st != GRASP
        h &= st != DONE
        a[h, :arm_dim] = ((q_cmd - arm._anchor) / arm._scale)[h]
        a[h, arm_dim:] = ((S["hand_hold"] - hand._grasp_offset) / hand._scale)[h]

        env.step(a)
        step += 1
        if video is not None and step % video_every == 0:
            frame = env.render()
            if frame is not None:
                video.append_data(frame)

        # bookkeeping
        S["t"] = S["t"] + dt * h.float()
        live = h & (st != RELEASE) & torch.tensor([not d for d in S["drop_phase"]], device=dev)
        d_hand = fruit_to_hand()
        slip = (fruit_in_palm() - S["fruit0"]).norm(dim=-1)
        S["max_slip"] = torch.where(live, torch.maximum(S["max_slip"], slip), S["max_slip"])
        S["max_f"] = torch.where(live, torch.maximum(S["max_f"], _max_link_apple_force(env, sc)), S["max_f"])
        sc_d = self_collision_depth(env, margin=0.0).amax(-1)        # actual sphere overlap
        after_home = (st == MOVE) | (st == CIRCLE) | (st == HOLD)
        if not args.no_basket:
            S["max_basket"] = torch.where(h, torch.maximum(S["max_basket"], basket_depth_arm()), S["max_basket"])
        for k, m in (("twist", st == TWIST), ("home", st == HOME), ("moves", after_home),
                     ("basket", (st == BASKET) | (st == RELEASE))):
            S["max_self"][k] = torch.where(m, torch.maximum(S["max_self"][k], sc_d), S["max_self"][k])
        S["min_apple_gap"] = torch.where(live, torch.minimum(S["min_apple_gap"], apple_body_clearance()), S["min_apple_gap"])
        S["max_tree"] = torch.where(after_home, torch.maximum(S["max_tree"], tree_contact_depth(env, 0.0).amax(-1)), S["max_tree"])
        newly = stem.broken & torch.isnan(S["twist_at_break"]) & h
        S["twist_at_break"] = torch.where(newly, stem.twist_deg.abs(), S["twist_at_break"])
        for i in (live & (d_hand > args.leave_margin)).nonzero(as_tuple=True)[0].tolist():
            S["drop_phase"][i] = PHASE[int(st[i])]
            st[i] = DONE

        # transitions
        for i in (h & (st != DONE)).nonzero(as_tuple=True)[0].tolist():
            s_i, t_i = int(st[i]), float(S["t"][i])
            if s_i == TWIST and (t_i >= args.twist_s or (bool(stem.broken[i]) and t_i >= 0.1)):
                # Home = the REACH part only, backwards (the grasp phase's small
                # wrist/finger oscillations would otherwise be retraced too),
                # lightly smoothed, starting exactly at the READY pose.
                k_ho = S["handoff_idx"][i] or len(S["path"][i])
                reach = torch.stack(S["path"][i][:k_ho]) if k_ho > 0 else S["q_ready"][i].unsqueeze(0)
                reach = hv.smooth_joint_path(reach.flip(0))
                path = torch.cat([S["q_ready"][i].unsqueeze(0), reach])
                S["home"][i] = hv.retime_joint_path(path, dt, args.home_speed)
                S["twist_dq"][i] = arm._processed_actions[i] - S["q_ready"][i]
                S["home_k"][i] = 0
                st[i] = HOME; S["t"][i] = 0.0
            elif s_i == HOME and S["home_k"][i] >= len(S["home"][i]) + int(0.3 / dt):
                if args.moves > 0:
                    start_move(i)
                elif args.no_basket:
                    st[i] = HOLD; S["t"][i] = 0.0
                else:
                    start_basket(i)
            elif s_i == MOVE and t_i >= float(S["seg_T"][i]) + 0.2:
                start_circle(i)
            elif s_i == CIRCLE and t_i >= args.circle_s:
                S["move_k"][i] += 1
                if S["move_k"][i] < args.moves:
                    start_move(i)
                elif args.no_basket:
                    st[i] = HOLD; S["t"][i] = 0.0
                else:
                    start_basket(i)
            elif s_i == BASKET and t_i >= float(S["seg_T"][i]) + 0.15:
                S["seg"][i] += 1
                if S["seg"][i] < len(S["chain"][i]) - 1:
                    next_segment(i)
                else:
                    S["hand_hold"][i] = hand_open[i]           # open the fingers: drop
                    st[i] = RELEASE; S["t"][i] = 0.0
            elif s_i == RELEASE and t_i >= args.release_s:
                rad = env._apple_radius[i:i + 1] if isinstance(getattr(env, "_apple_radius", None), torch.Tensor) \
                    else torch.full((1,), 0.04, device=dev)
                S["in_basket"][i] = bool(bk.apple_in_basket(
                    (apple.data.root_pos_w[i:i + 1] - env.scene.env_origins[i:i + 1]), rad)[0])
                st[i] = DONE
            elif s_i == HOLD and t_i >= args.hold_after:
                st[i] = DONE

        if step % 100 == 0:
            cnt = Counter(PHASE[int(x)] for x in st)
            now = time.time()
            rtf = (step - _step0) * dt / max(now - _wall0, 1e-6)
            _wall0, _step0 = now, step
            print(f"  t={step * dt:5.1f}s  real-time factor {rtf:4.2f}x  {dict(cnt)}  stems broken {int(stem.broken.sum())}  "
                  f"dropped {sum(1 for d in S['drop_phase'] if d)}", flush=True)
        with _reset_lock:
            manual, _reset_requested = _reset_requested, False
        if manual or bool((st == DONE).all()):
            report(rnd, early=manual)
            if args.rounds and rnd >= args.rounds:
                break
            rnd += 1
            env.reset()
            new_round()
            step = 0
            _wall0, _step0 = time.time(), 0
            print(f"\n>>> ROUND {rnd} <<<\n", flush=True)

if video is not None:
    video.close()
    print(f"[bench] video written: {args.video} ({args.video_fps} fps, simulated time)", flush=True)
sys.stdout.flush()
os._exit(0)
