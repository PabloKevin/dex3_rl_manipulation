#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Tree harvest bench: harvest EVERY apple of a multi-apple tree, one at a
time, with the frozen reach and grasp policies -- the neighbours of each
apple are real (dynamic, on breakable stems), so the hand and fingers can
push them, and knock them off (envs/apple_grasp_tree_env_cfg.py,
envs/mdp/apple_tree.py: APPLES_PER_TREE apples in clusters of 1-4).

Nothing is trained here. Per apple, per env:
  GRASP    frozen reach policy, then the frozen grasp policy; the arm's
           joint path is recorded
  (READY)  envs.mdp.harvest.GraspReadyDetector, or --grasp_timeout s after
           the hand-off ("timeout-ready"); finger targets freeze, the stem
           (a BreakableStem along the apple's real stem) replaces the pin
  TWIST    hand + fruit rotate about the palm normal through the grasp
           pocket (IK), up to --twist_deg, until the stem breaks -- in the
           direction that keeps the hand and wrist clear of the neighbours:
           both sweeps are checked with the self-collision hand/wrist
           spheres against the fruit still hanging (a neighbour on one side
           -> twist the other way, whatever the hand's orientation)
  HOME     the recorded REACH path played backwards, smoothed, re-timed to
           --home_speed rad/s, the twist undone along the way
  BASKET   the planned path from home over the basket (scripts/
           plan_basket_drop.py), down to the palm-down drop pose
  RELEASE  open the hand, wait --release_s, check the fruit is in the basket
  RETURN   the same path back home
  -> the next apple of the tree (an env reset that keeps the tree: the
     robot is at home, where every reach starts)
Every fruit, the target included, hangs on its breakable stem from the
start (no pin): a reach or grasp can knock the target off, which also sends
the arm home. A dropped apple sends the arm home at once (the rest of the reach path, or
the basket path back) and on to the next apple. When a tree is done its
table is printed; when every env's tree is done, a new round (new trees).

Per apple: grasp-ready time, stem (cause, twist angle at break), kept in
hand / phase it fell in, max slip in the palm, peak contact force on the
CARRIED fruit (twist / home / basket) vs. that apple's bruising threshold,
the peak during reach + grasp (on the pinned fruit), in the basket, worst self-collision overlap,
how far the hand pushed the neighbours still on the tree, and how many it
knocked off. FINISHED = held through everything and in the basket.

Reach: --reach_policy last (default) = the newest AppleReachTree-v1 checkpoint,
exported on the fly (scripts/export_policy_jit.py); v1 = the single-apple
reach the grasp task pins. Grasp: the trained policy, or --grasp scripted
(plan B for sim2real): each finger closes until it feels --close_force, then
holds a target --squeeze past that point.

Usage:
  python scripts/tree_harvest_bench.py --num_envs 2                       (viewer)
  python scripts/tree_harvest_bench.py --headless --num_envs 1 --video logs/videos/tree_harvest.mp4
  python scripts/tree_harvest_bench.py --headless --num_envs 8 --rounds 2
"""
import argparse
import glob
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description="Harvest every apple of a tree, one at a time",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--policy", type=str, default="last",
               help="grasp policy: 'last' = newest AppleGraspTree-v1 checkpoint (exported on the fly), else the newest "
                    "logs/rsl_rl/apple_grasp_grasp/*/exported/policy.pt; or a model_*.pt / exported .pt")
p.add_argument("--fingers", type=str, default="auto", choices=["auto", "policy", "freeze"],
               help="after READY: 'policy' = the grasp policy keeps driving the fingers through twist, home and "
                    "basket; 'freeze' = hold its last finger targets (the old bench); 'auto' = policy for a hand-only "
                    "(41-input) policy, freeze otherwise (its arm inputs would be out of distribution)")
p.add_argument("--reach_policy", type=str, default="last",
               help="'last' = newest AppleReachTree-v1 checkpoint (exported on the fly), 'v1' = the pinned "
                    "single-apple reach the grasp task was trained with, or a model_*.pt / exported policy .pt")
p.add_argument("--grasp", type=str, default="policy", choices=["policy", "scripted"],
               help="'scripted' = plan B: close each finger until it feels --close_force, then squeeze --squeeze")
p.add_argument("--close_speed", type=float, default=0.8, help="(scripted) finger closing speed, rad/s")
p.add_argument("--close_force", type=float, default=1.0, help="(scripted) contact force that stops a finger, N")
p.add_argument("--squeeze", type=float, default=0.8,
               help="(scripted) target past the contact point, rad: the Dex3 PD (kp 1 N*m/rad) only presses by "
                    "error. Measured, 4 trees each: 0.35 -> 2/28 in the basket (slips out), 0.8 -> 10/28, "
                    "1.2 -> 9/28 with 5 bruised")
p.add_argument("--settle_s", type=float, default=0.3, help="(scripted) hold after the last finger stops before READY")
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--rounds", type=int, default=1, help="trees per env before exiting (0 = until the window is closed)")
p.add_argument("--apples", type=int, default=None, help="apples per tree (default APPLES_PER_TREE of the env cfg)")
p.add_argument("--order", type=str, default=None, choices=["top_down", "bottom_up", "random", "clearance"],
               help="harvest order (default HARVEST_ORDER of the env cfg)")
p.add_argument("--grasp_timeout", type=float, default=5.0,
               help="seconds after the hand-off; a grasp not READY by then is harvested anyway ('timeout-ready')")
p.add_argument("--twist_deg", type=float, default=90.0)
p.add_argument("--twist_s", type=float, default=1.5)
p.add_argument("--twist_arm_weight", type=float, default=0.1,
               help="share of the twist's IK motion for shoulder/elbow (wrist = 1): wrist-first twisting")
p.add_argument("--twist_body_margin", type=float, default=0.04,
               help="stop the twist when an arm/hand sphere comes this close to the body (the training danger zone)")
p.add_argument("--twist_fruit_margin", type=float, default=0.0,
               help="stop the twist when a hand/wrist sphere comes this close to a neighbour (where it hung when "
                    "the twist began -- what the robot's vision gives it)")
p.add_argument("--home_speed", type=float, default=1.2, help="peak joint-space speed (rad/s) going home")
p.add_argument("--move_speed", type=float, default=1.9,
               help="peak joint speed (rad/s) on the basket path (the arm's rate limiter caps at 2.0)")
p.add_argument("--drop_speed", type=float, default=1.0, help="peak joint speed (rad/s) for the last descent to the drop pose")
p.add_argument("--grip_level", type=float, default=0.45,
               help="grip-force command for the grasp policy (its target_grip_level input), as a fraction of that "
                    "apple's bruising threshold; training samples it at random, the bench sets it")
p.add_argument("--grip_step", type=float, default=0.15, help="added to --grip_level on every retry of the same apple")
p.add_argument("--grip_max", type=float, default=0.75, help="cap on the grip-force command (fraction of the threshold)")
p.add_argument("--path", type=str, default="policy", choices=["policy", "scripted"],
               help="after the twist: 'policy' = pull back, then the goal-reaching arm policy (AppleArmGoal-v1) to the "
                    "safe zone, the basket and home (falls back to 'scripted' if none is trained); 'scripted' = the "
                    "reach path reversed to home, the planned basket path, the same path back")
p.add_argument("--goal_policy", type=str, default="last",
               help="'last' = newest AppleArmGoal-v1 checkpoint (exported on the fly), or a model_*.pt / exported .pt")
p.add_argument("--pull", type=float, default=0.05, help="(policy path) straight pull back after the twist, m")
p.add_argument("--pull_s", type=float, default=0.6, help="(policy path) seconds for the pull")
p.add_argument("--safe_depth", type=float, default=0.18,
               help="(policy path) safe zone: up to this far straight back from the pick, clamped clear of the body")
p.add_argument("--goal_timeout", type=float, default=5.0, help="(policy path) seconds per goal before moving on anyway")
p.add_argument("--max_attempts", type=int, default=3,
               help="attempts per apple: a grasp that fails with the fruit still on its stem is retried")
p.add_argument("--release_s", type=float, default=1.5)
p.add_argument("--leave_margin", type=float, default=0.06, help="fruit this far from the grasp pocket = dropped")
p.add_argument("--video", type=str, default=None, help="mp4 in SIMULATED time (works headless)")
p.add_argument("--video_fps", type=int, default=25)
p.add_argument("--video_res", type=int, nargs=2, default=[1280, 720])
p.add_argument("--cam_eye", type=float, nargs=3, default=None, help="relative to the centre of the env grid")
p.add_argument("--cam_lookat", type=float, nargs=3, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import envs  # noqa: F401
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from envs import apple_grasp_tree_env_cfg as tree_cfg_mod
from envs.apple_grasp_grasp_env_cfg import GRASP_CENTER_LOCAL, RIGHT_ARM_JOINTS
from envs.mdp import apple_tree as at
from envs.mdp import basket as bk
from envs.mdp import harvest as hv
from envs.mdp import self_collision as scm
from envs.mdp.grasp_rewards import _max_link_apple_force
from envs.mdp.self_collision import self_collision_depth

if args.apples is not None:
    tree_cfg_mod.APPLES_PER_TREE = args.apples           # read by the scene / event cfgs below
cfg = tree_cfg_mod.AppleGraspTreeEnvCfg()


def _resolve_reach_policy(spec):
    """Path of an exported reach policy for --reach_policy, and its obs layout."""
    import importlib.util
    from envs.mdp.grasp_autoplay import PINNED_REACH_POLICY
    _sp = importlib.util.spec_from_file_location("export_policy_jit", os.path.join(REPO, "scripts", "export_policy_jit.py"))
    ex = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(ex)
    if spec == "v1":
        path = PINNED_REACH_POLICY
    else:
        if spec == "last":
            runs = [d for d in glob.glob(os.path.join(REPO, "logs/rsl_rl/apple_reach_tree/20*/")) if "validate" not in d]
            ckpts = [c for d in runs for c in glob.glob(os.path.join(d, "model_*.pt"))]
            if not ckpts:
                sys.exit("no AppleReachTree-v1 checkpoint found -- pass --reach_policy v1 or a path")
            spec = max(ckpts, key=lambda c: (os.path.getmtime(os.path.dirname(c)), int(c.rsplit("_", 1)[1][:-3])))
        path = ex.export(spec) if os.path.basename(spec).startswith("model_") else spec
    return path, ("tree" if ex.input_dim(path) == 37 else "v1")


reach_path, reach_layout = _resolve_reach_policy(args.reach_policy)


def _resolve_grasp_policy(spec):
    """Path of an exported grasp policy for --policy, and its input size."""
    import importlib.util
    _sp = importlib.util.spec_from_file_location("export_policy_jit", os.path.join(REPO, "scripts", "export_policy_jit.py"))
    ex = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(ex)
    if spec == "last":
        runs = [d for d in glob.glob(os.path.join(REPO, "logs/rsl_rl/apple_grasp_tree/20*/"))
                if not any(w in d for w in ("validate", "smoke"))]
        ckpts = [c for d in runs for c in glob.glob(os.path.join(d, "model_*.pt"))]
        if ckpts:
            spec = max(ckpts, key=lambda c: (os.path.getmtime(os.path.dirname(c)), int(c.rsplit("_", 1)[1][:-3])))
        else:
            cands = sorted(glob.glob(os.path.join(REPO, "logs/rsl_rl/apple_grasp_grasp/*/exported/policy.pt")),
                           key=os.path.getmtime)
            if not cands:
                sys.exit("no grasp policy found -- pass --policy")
            spec = cands[-1]
    path = ex.export(spec) if os.path.basename(spec).startswith("model_") else spec
    return path, ex.input_dim(path)


args.policy, grasp_dim = _resolve_grasp_policy(args.policy)
HAND_ONLY = grasp_dim in (41, 44)                   # hand-only grasp (44: + gravity in the palm frame)
if HAND_ONLY:
    cfg.observations = tree_cfg_mod.HandOnlyObservationsCfg()
    if grasp_dim == 41:
        cfg.observations.policy.gravity_palm = None
FINGERS_POLICY = args.grasp == "policy" and (args.fingers == "policy" or (args.fingers == "auto" and HAND_ONLY))
cfg.actions.right_arm.reach_policy_path = reach_path
cfg.actions.right_arm.reach_obs_layout = reach_layout
cfg.scene.num_envs = args.num_envs
bk.add_to_scene(cfg.scene)
cfg.episode_length_s = 1.0e5
for name in list(vars(cfg.terminations)):
    if not name.startswith("_"):
        setattr(cfg.terminations, name, None)
cfg.terminations.next_apple = TerminationTermCfg(func=at.next_apple_requested)
cfg.events.reset_reach_grasp_phase.params.update(
    stem_release_schedule=None, stem_release_time_range=None, release_on_latch=False, shaping_grace_s=None)
if args.order:
    cfg.events.randomize_tree.params["order"] = args.order
if args.video:
    cfg.viewer.resolution = tuple(args.video_res)
env = gym.make("AppleGrasp-v1", cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
N, dev, dt = env.num_envs, env.device, env.step_dt
M = tree_cfg_mod.APPLES_PER_TREE

video = None
if args.video:
    centre = env.scene.env_origins.mean(0).cpu().numpy()
    # Behind the robot's right shoulder: from straight to the side the fruit
    # line up and their spacing along the wall is lost; far enough back to
    # keep the basket and its stand in frame, left of the trunk.
    if N == 1:
        eye, look = (-1.05, -1.85, 1.9), (0.15, -0.3, 0.85)
    else:
        spread = float(env.scene.env_origins[:, 1].max() - env.scene.env_origins[:, 1].min())
        eye, look = (-1.3, -3.1 - 0.9 * spread, 2.1), (0.2, 0.0, 0.9)
    eye, look = args.cam_eye or eye, args.cam_lookat or look
    env.viewport_camera_controller.update_view_location(
        eye=tuple(float(c + e) for c, e in zip(centre, eye)), lookat=tuple(float(c + l) for c, l in zip(centre, look)))
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    video = imageio.get_writer(args.video, fps=args.video_fps, codec="libx264", quality=8, macro_block_size=8)
    video_every = max(1, round(1.0 / (args.video_fps * dt)))

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
ready_det = hv.GraspReadyDetector(env, sc, hand._joint_ids)

# ── plan B: scripted grasp (close each finger until it feels the fruit) ──
_hn = list(hand._joint_names)                                      # action order = articulation order
_lim = robot.data.soft_joint_pos_limits[0, hand._joint_ids]        # [7, 2]
_open = robot.data.default_joint_pos[0, hand._joint_ids]
_far = torch.where((_lim[:, 1] - _open).abs() > (_lim[:, 0] - _open).abs(), _lim[:, 1], _lim[:, 0])
_dir = torch.sign(_far - _open)                                    # closing direction per joint
_goal = _open + 0.95 * (_far - _open)                              # fully closed (if nothing is felt)
SC_GROUPS = {"thumb": ["right_hand_thumb_1_joint", "right_hand_thumb_2_joint"],
             "index": ["right_hand_index_0_joint", "right_hand_index_1_joint"],
             "middle": ["right_hand_middle_0_joint", "right_hand_middle_1_joint"]}
SC_LINKS = {"thumb": ["right_hand_thumb_1_link", "right_hand_thumb_2_link"],
            "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
            "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"]}
_sensor = env.scene["contact_forces"]
_sc_joint_mask = torch.zeros(3, len(_hn), device=dev)
_sc_link_ids = []
for g, name in enumerate(SC_GROUPS):
    for j in SC_GROUPS[name]:
        _sc_joint_mask[g, _hn.index(j)] = 1.0
    _sc_link_ids.append([_sensor.body_names.index(l) for l in SC_LINKS[name]])
sc_state = dict(target=torch.zeros(N, len(_hn), device=dev), stopped=torch.zeros(N, 3, dtype=torch.bool, device=dev),
                t_all=torch.full((N,), float("nan"), device=dev), started=torch.zeros(N, dtype=torch.bool, device=dev))
if args.grasp == "scripted":
    print("[tree bench] scripted grasp: closing directions " + ", ".join(
        f"{n.replace('right_hand_', '').replace('_joint', '')} {'+' if d > 0 else '-'}{abs(float(g_ - o)):.2f}"
        for n, d, g_, o in zip(_hn, _dir.tolist(), _goal.tolist(), _open.tolist())), flush=True)


def scripted_grasp(a, st):
    """Hand actions for envs in GRASP after the hand-off: per finger group,
    close at --close_speed until its links feel --close_force (any contact,
    as real pads), then hold the target --squeeze past that point. Arm holds
    the hand-off pose. Returns [N] bool: every finger stopped, settled."""
    act = (st == GRASP) & env._grasp_phase_active
    new = act & ~sc_state["started"]
    if bool(new.any()):                                            # start from the grasp-phase pre-shape
        sc_state["target"][new] = hand._grasp_offset[new] if hand._grasp_offset.dim() == 2 else hand._grasp_offset
        sc_state["stopped"][new] = False
        sc_state["t_all"][new] = float("nan")
        sc_state["started"] |= new
    if not bool(act.any()):
        return torch.zeros(N, dtype=torch.bool, device=dev)
    f = _sensor.data.net_forces_w.norm(dim=-1)                     # [N, links]
    feel = torch.stack([f[:, ids].amax(-1) for ids in _sc_link_ids], dim=1)   # [N, 3]
    newly = act.unsqueeze(-1) & ~sc_state["stopped"] & (feel >= args.close_force)
    tgt = sc_state["target"]
    if bool(newly.any()):                                          # squeeze past the contact point
        tgt += (newly.float() @ _sc_joint_mask) * _dir * args.squeeze
    sc_state["stopped"] |= newly
    moving = (~sc_state["stopped"]).float() @ _sc_joint_mask       # [N, 7] joints still closing
    tgt += act.unsqueeze(-1).float() * moving * _dir * args.close_speed * dt
    # a finger that reached full closure without feeling anything stops there
    lo, hi = torch.minimum(_open, _goal + _dir * args.squeeze), torch.maximum(_open, _goal + _dir * args.squeeze)
    sc_state["target"] = torch.maximum(torch.minimum(tgt, hi), lo)
    at_goal = (((sc_state["target"] - _goal) * _dir) >= 0)
    full = torch.stack([(at_goal | (_sc_joint_mask[g] == 0)).all(-1) for g in range(3)], dim=1)
    sc_state["stopped"] |= act.unsqueeze(-1) & full
    a[act, :arm_dim] = 0.0                                         # = the hand-off pose (anchored)
    a[act, arm_dim:] = ((sc_state["target"] - hand._grasp_offset) / hand._scale)[act]
    all_stop = act & sc_state["stopped"].all(-1)
    first = all_stop & torch.isnan(sc_state["t_all"])
    since = (env.episode_length_buf - env._grasp_handoff_step).float() * dt
    sc_state["t_all"] = torch.where(first, since, sc_state["t_all"])
    return all_stop & (since - sc_state["t_all"] >= args.settle_s)
ik = hv.DifferentialIK(robot, "right_hand_palm_link", arm_ids)
c_local = torch.tensor(GRASP_CENTER_LOCAL, device=dev).expand(N, 3)
_bp = np.load(os.path.join(REPO, "envs", "assets", "basket_drop_path.npz"))
_order = [RIGHT_ARM_JOINTS.index(robot.joint_names[j]) for j in arm_ids]
basket_wps = torch.tensor(_bp["waypoints"], device=dev, dtype=torch.float32)[:, _order]   # home, ..., drop
hand_open = robot.data.default_joint_pos[:, hand._joint_ids]

GRASP, TWIST, HOME, BASKET, RELEASE, RETURN, NEXT, TREE_DONE, PULL, GOAL = range(10)
G_SAFE, G_DROP, G_HOME = 0, 1, 2
PHASE = {GRASP: "grasp", TWIST: "twist", HOME: "home", BASKET: "basket", RELEASE: "release", PULL: "pull", GOAL: "goal",
         RETURN: "return", NEXT: "next", TREE_DONE: "done"}
fmt = lambda x, s="{:.2f}": "-" if x != x else s.format(x)
_cm = lambda x: "free" if x != x else f"{x * 100:4.1f}"


def T():
    return env._tree


def fruit_to_hand():
    gp = robot.data.body_pos_w[:, palm_id] + quat_apply(robot.data.body_quat_w[:, palm_id], c_local)
    return torch.norm(apple.data.root_pos_w - gp, dim=-1)


def fruit_in_palm():
    return quat_apply_inverse(robot.data.body_quat_w[:, palm_id], apple.data.root_pos_w - robot.data.body_pos_w[:, palm_id])


def neighbours_on_tree():
    """[N, M] bool spots whose fruit is still hanging (not the target), and
    [N, M] its displacement from where it grew (m)."""
    tr = T()
    pos = torch.stack([o.data.root_pos_w for o in tr.objs], dim=1) - env.scene.env_origins.unsqueeze(1)   # by entity
    occ = tr.occupant
    p_spot = torch.gather(pos, 1, occ.unsqueeze(-1).expand(-1, -1, 3))
    broken = torch.stack([torch.zeros(N, dtype=torch.bool, device=dev)] + [s.broken for s in tr.stems[1:]], dim=1)
    hanging = (tr.status == at.WAITING) & (occ > 0) & ~torch.gather(broken, 1, occ)
    return hanging, (p_spot - tr.spot_pos).norm(dim=-1)


# ── goal-reaching arm policy for the harvest path (AppleArmGoal-v1) ──
PATH_POLICY = False
if args.path == "policy":
    import importlib.util
    from envs.mdp import arm_goal as ag
    from envs.mdp.observations import joint_pos_selected, joint_vel_selected
    _sp = importlib.util.spec_from_file_location("export_policy_jit", os.path.join(REPO, "scripts", "export_policy_jit.py"))
    _ex = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_ex)
    spec = args.goal_policy
    if spec == "last":
        runs = [d for d in glob.glob(os.path.join(REPO, "logs/rsl_rl/apple_arm_goal/20*/"))
                if not any(w in d for w in ("validate", "smoke"))]
        ckpts = [c for d in runs for c in glob.glob(os.path.join(d, "model_*.pt"))]
        spec = max(ckpts, key=lambda c: (os.path.getmtime(os.path.dirname(c)), int(c.rsplit("_", 1)[1][:-3]))) \
            if ckpts else None
    if spec is None:
        print("[tree bench] no AppleArmGoal-v1 policy trained yet -> scripted harvest path", flush=True)
    else:
        goal_path = _ex.export(spec) if os.path.basename(spec).startswith("model_") else spec
        _goal_jit = torch.jit.load(goal_path, map_location=dev).eval()
        _goal_dim = _ex.input_dim(goal_path)                   # 31: palm normal only; 34: + finger direction
        PATH_POLICY = True
        G = ag.goal_state(env)
        G.basket[:] = torch.tensor(ag.BASKET_NOMINAL, device=dev)       # the bench's basket (basket.py)
        G.pending[:] = False
        G.mass[:] = 0.0                                                  # the real fruit is in the hand here
        _goal_arm_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS, preserve_order=True)
        _goal_arm_cfg.resolve(env.scene)
        _goal_prev = torch.zeros(N, arm_dim, device=dev)
        _arm_default = robot.data.default_joint_pos[:, arm_ids]
        _POCKET = tuple(GRASP_CENTER_LOCAL)

        def goal_obs():
            return torch.cat([joint_pos_selected(env, _goal_arm_cfg), joint_vel_selected(env, _goal_arm_cfg),
                              ag.obs_goal_pocket(env, _POCKET), ag.obs_goal_normal(env, _POCKET),
                              ag.obs_basket(env, _POCKET), ag.obs_carrying(env), _goal_prev]
                             + ([ag.obs_goal_finger(env)] if _goal_dim == 34 else []), dim=-1)

# Wrist-first twist: shoulder and elbow get a small share of the IK motion.
# Rotating the palm about its normal is often not a pure wrist motion, and
# equal weights swung the shoulder/elbow -- once through the torso. Where
# the wrist runs out of range the twist simply stops short.
_twist_w = torch.tensor([args.twist_arm_weight if robot.joint_names[j] in
                         ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                          "right_elbow_joint") else 1.0 for j in arm_ids], device=dev)
_sph = scm._build(env)["arm"]
_sph_keep = torch.tensor([robot.data.body_names[int(b)] not in ("right_shoulder_yaw_link", "right_elbow_link")
                          for b in _sph["ids"]], device=dev)     # hand + wrist: what the twist swings round


def twist_sign(i, n_samples=10, margin=0.03):
    """+1 or -1: the twist direction that swings the hand and wrist the LEAST
    into the fruit still hanging, and the two costs (m).

    The hand/wrist spheres are rotated rigidly about the twist axis (palm
    normal, through the grasp pocket) over the whole sweep, each way; the
    cost is the mean, over the sweep, of the summed intrusion of every
    sphere into a `margin` around every neighbour. Scoring the CLOSEST
    approach instead (first version) made both directions tie whenever a
    sphere already touched a neighbour at the start -- usual in a cluster,
    the spheres are a little inflated -- and the tie went to the default
    direction, often the one that swung the fingers into the neighbour."""
    c = scm._world_spheres(env, _sph)[i][_sph_keep]                  # [S, 3]
    rad = _sph["r"][_sph_keep]
    hanging, _ = neighbours_on_tree()
    tr = T()
    ent = tr.occupant[i][hanging[i]]
    if len(ent) == 0:
        return 1.0, float("nan"), float("nan")
    q = torch.stack([tr.objs[int(e)].data.root_pos_w[i] for e in ent])       # [K, 3]
    r_a = float(env._apple_radius[i])
    k, piv = S["tw_axis"][i], S["tw_pivot"][i]
    v = c - piv
    out = {}
    for sgn in (1.0, -1.0):
        cost = 0.0
        for th in torch.linspace(0.0, twist_rad, n_samples, device=dev)[1:]:
            a = sgn * th
            vr = v * torch.cos(a) + torch.cross(k.expand_as(v), v, dim=-1) * torch.sin(a) \
                + k * (v @ k).unsqueeze(-1) * (1 - torch.cos(a))
            d = ((piv + vr).unsqueeze(1) - q.unsqueeze(0)).norm(dim=-1) - rad.unsqueeze(1) - r_a
            cost += float((margin - d).clamp(min=0.0).sum())
        out[sgn] = cost / (n_samples - 1)
    sign = 1.0 if out[1.0] <= out[-1.0] + 1e-4 else -1.0
    return sign, out[1.0], out[-1.0]


def new_apple(ids):
    """Per-apple state for the envs in ids (their reset has just happened)."""
    for i in ids:
        S["state"][i] = GRASP
        S["t"][i] = 0.0
        S["path"][i], S["handoff_idx"][i] = [], None
        S["home"][i], S["home_k"][i], S["twist_dq"][i], S["q_ready"][i] = None, 0, None, None
        S["chain"][i], S["seg"][i] = None, 0
        S["drop_phase"][i], S["not_ready"][i], S["in_basket"][i] = "", "", None
        S["tw_sign"][i], S["tw_clear"][i] = 1.0, (float("nan"), float("nan"))
        S["pending_home"][i] = False
        S["twist_stop"][i], S["twist_stop_why"][i], S["twist_fruit"][i], S["twist_base"][i] = False, "", None, None
        S["goal_timeouts"][i] = 0
        S["hand_hold_open"][i] = False
        S["row_done"][i] = False
        S["home_tries"][i] = 0
        for k in ("t_ready", "twist_at_break"):
            S[k][i] = float("nan")
        for k in ("max_slip", "max_f", "max_f_reach", "max_self", "max_push"):
            S[k][i] = 0.0
        if S["retry"][i]:                          # another attempt at the same apple
            S["retry"][i] = False
            S["attempt"][i] += 1
        else:
            S["attempt"][i] = 1
            hanging, _ = neighbours_on_tree()
            S["hanging0"][i] = hanging[i].clone()   # knock-offs count over every attempt
            S["spot"][i] = T().current_spot(i)
    ready_det.reset(torch.tensor(ids, device=dev))
    sc_state["started"][ids] = False
    # The grip-force command (the grasp policy's target_grip_level input):
    # the reset just drew it at random, as in training; here it is set, and a
    # retry grips harder. Same fraction of this apple's bruising threshold.
    if hasattr(env, "_target_grip_level"):
        for i in ids:
            frac = min(args.grip_level + args.grip_step * (S["attempt"][i] - 1), args.grip_max)
            env._target_grip_level[i] = frac * env._fruit_damage_force[i]


def new_trees():
    global S
    S = dict(
        state=torch.full((N,), GRASP, dtype=torch.long, device=dev), t=torch.zeros(N, device=dev),
        path=[[] for _ in range(N)], handoff_idx=[None] * N, home=[None] * N, home_k=[0] * N, twist_dq=[None] * N,
        q_ready=[None] * N, tw_axis=torch.zeros(N, 3, device=dev), tw_pivot=torch.zeros(N, 3, device=dev),
        hand_hold=torch.zeros(N, hand.action_dim, device=dev),
        palm0_p=torch.zeros(N, 3, device=dev), palm0_q=torch.zeros(N, 4, device=dev),
        q_from=torch.zeros(N, len(arm_ids), device=dev), q_to=torch.zeros(N, len(arm_ids), device=dev),
        seg_T=torch.ones(N, device=dev), chain=[None] * N, seg=[0] * N,
        t_ready=torch.full((N,), float("nan"), device=dev), twist_at_break=torch.full((N,), float("nan"), device=dev),
        fruit0=torch.zeros(N, 3, device=dev), max_slip=torch.zeros(N, device=dev), max_f=torch.zeros(N, device=dev),
        max_f_reach=torch.zeros(N, device=dev),
        max_self=torch.zeros(N, device=dev), max_push=torch.zeros(N, device=dev),
        drop_phase=[""] * N, not_ready=[""] * N, in_basket=[None] * N, hanging0=[None] * N, spot=[0] * N,
        tw_sign=torch.ones(N, device=dev), tw_clear=[(float("nan"), float("nan"))] * N, pending_home=[False] * N, row_done=[False] * N, home_tries=[0] * N, retry=[False] * N, attempt=[1] * N,
        twist_stop=[False] * N, twist_stop_why=[""] * N, twist_fruit=[None] * N, twist_base=[None] * N,
        goal_k=torch.zeros(N, dtype=torch.long, device=dev), goal_timeouts=[0] * N, hand_hold_open=[False] * N,
        pull_p0=torch.zeros(N, 3, device=dev), pull_q0=torch.zeros(N, 4, device=dev), pull_dir=torch.zeros(N, 3, device=dev),
        rows=[[] for _ in range(N)], knocked={i: {} for i in range(N)}, reported=[False] * N,
    )
    if hasattr(env, "_tree"):                      # (created by the first reset, which grows new trees anyway)
        T().new_tree[:] = True
    env.reset()
    new_apple(list(range(N)))


def start_chain(i, points):
    """Joint-space path through `points`, one quintic segment per leg (zero
    velocity and acceleration at every waypoint: the basket path turns ~140
    deg over the basket, too sharp to blend through without cutting the
    planned clearance). Legs shorter than 0.02 rad are dropped (e.g. home ->
    home after the way back), they only added a pause."""
    pts = [points[0]]
    for q in points[1:]:
        if float((q - pts[-1]).norm()) > 0.02:
            pts.append(q)
    if len(pts) < 2:
        pts.append(points[-1])
    S["chain"][i], S["seg"][i] = pts, 0
    next_segment(i)


def next_segment(i):
    k = S["seg"][i]
    q0, q1 = S["chain"][i][k], S["chain"][i][k + 1]
    slow = S["state"][i] == BASKET and k + 1 == len(S["chain"][i]) - 1
    speed = args.drop_speed if slow else args.move_speed
    S["q_from"][i], S["q_to"][i] = q0, q1
    S["seg_T"][i] = max(0.4, 1.875 * float((q1 - q0).norm()) / speed)
    S["t"][i] = 0.0


def go_basket(i):
    S["state"][i] = BASKET
    start_chain(i, [arm._processed_actions[i].clone()] + list(basket_wps))


def go_back_from_basket(i):
    """Home along the basket path, from wherever on it the arm is."""
    k = S["seg"][i]
    pts = [arm._processed_actions[i].clone()] + [S["chain"][i][j] for j in range(k, 0, -1)]
    if len(pts) < 2 or (pts[-1] - basket_wps[0]).norm() > 1e-4:
        pts.append(basket_wps[0])
    S["state"][i] = RETURN
    start_chain(i, pts)


def start_home(i):
    k_ho = S["handoff_idx"][i] or len(S["path"][i])
    reach = torch.stack(S["path"][i][:k_ho]) if k_ho > 0 else S["q_ready"][i].unsqueeze(0)
    reach = hv.smooth_joint_path(reach.flip(0))
    start = arm._processed_actions[i].clone()
    q_ref = S["q_ready"][i] if S["q_ready"][i] is not None else start
    S["home"][i] = hv.retime_joint_path(torch.cat([q_ref.unsqueeze(0), reach]), dt, args.home_speed)
    S["twist_dq"][i] = start - q_ref
    S["home_k"][i] = 0
    S["state"][i] = HOME
    S["t"][i] = 0.0


_last_danger = ""


def twist_danger(i, worse=0.005):
    """True when the twist should stop because it is making things WORSE:
    an arm/hand sphere going deeper into the body's danger zone (FK + the
    self-collision sphere model, as the arm goal policy is trained), or a
    hand/wrist sphere pressing further into a neighbour (where it hung when
    the twist began), by more than `worse` m beyond the twist's start -- in a
    cluster the (slightly inflated) spheres often touch at the start already."""
    global _last_danger
    body = float(self_collision_depth(env, margin=args.twist_body_margin)[i].amax())
    if S["twist_fruit"][i] is None:
        hanging, _ = neighbours_on_tree()
        ent = T().occupant[i][hanging[i]]
        S["twist_fruit"][i] = torch.stack([T().objs[int(e)].data.root_pos_w[i] for e in ent]) if len(ent) else \
            torch.zeros(0, 3, device=dev)
        S["twist_base"][i] = (body, None)
    q = S["twist_fruit"][i]
    fruit = 0.0
    if len(q):
        c = scm._world_spheres(env, _sph)[i][_sph_keep]
        gap = (c.unsqueeze(1) - q.unsqueeze(0)).norm(dim=-1) - _sph["r"][_sph_keep].unsqueeze(1) - float(env._apple_radius[i])
        fruit = float((args.twist_fruit_margin - gap).clamp(min=0.0).max())
    b0, f0 = S["twist_base"][i]
    if f0 is None:
        S["twist_base"][i] = (b0, fruit)
        f0 = fruit
    if body > b0 + worse:
        _last_danger = "body"
        return True
    if fruit > f0 + worse:
        _last_danger = "neighbour"
        return True
    return False


def start_pull(i):
    """Straight back along the way the hand came in, orientation kept."""
    pq = robot.data.body_quat_w[i:i + 1, palm_id]
    n = quat_apply(pq, torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
    back = -n.clone(); back[2] = 0.0
    back = back / back.norm() if float(back.norm()) > 0.3 else torch.tensor([-1.0, 0.0, 0.0], device=dev)
    S["pull_p0"][i] = robot.data.body_pos_w[i, palm_id]
    S["pull_q0"][i] = pq[0]
    S["pull_dir"][i] = back
    S["state"][i] = PULL
    S["t"][i] = 0.0


def start_goal(i, k):
    """Hand the arm to the goal policy, goal k (safe zone / drop / home)."""
    G = ag.goal_state(env)
    ii = torch.tensor([i], device=dev)
    pocket, normal = ag.pocket_and_normal(env, _POCKET)
    holding = not S["drop_phase"][i] and S["hand_hold_open"][i] is False
    if k == G_SAFE:
        G.target[i] = ag.safe_zone_point(env, ii, torch.tensor([args.safe_depth], device=dev), _POCKET)[0]
        G.normal[i] = normal[i]
        G.finger[i] = ag.palm_finger(env)[i]
    elif k == G_DROP:
        b = G.basket[i]
        G.target[i] = env.scene.env_origins[i] + torch.stack([b[0] + ag.DROP_OFFSET_XY[0], b[1] + ag.DROP_OFFSET_XY[1],
                                                              b[2] + 0.13])
        dn = torch.tensor(ag.DROP_NORMAL, device=dev)
        G.normal[i] = dn / dn.norm()
        G.finger[i] = torch.tensor(ag.DROP_FINGER, device=dev)
    else:
        G.target[i] = env.scene.env_origins[i] + torch.tensor(ag.HOME_POCKET, device=dev)
        G.normal[i] = torch.tensor(ag.HOME_NORMAL, device=dev)
        G.finger[i] = torch.tensor(ag.HOME_FINGER, device=dev)
    G.carry[i] = holding and k != G_HOME
    G.start[i] = pocket[i]
    S["goal_k"][i] = k
    S["state"][i] = GOAL
    S["t"][i] = 0.0


def goal_done(i):
    G = ag.goal_state(env)
    pocket, normal = ag.pocket_and_normal(env, _POCKET)
    d = float((G.target[i] - pocket[i]).norm())
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float((G.normal[i] * normal[i]).sum())))))
    v = float(robot.data.body_lin_vel_w[i, palm_id].norm())
    return d < 0.03 and ang < 25.0 and v < 0.2


def finish_apple(i):
    """Record this apple's row (once), make sure the arm is home -- every
    reach must start there -- then ask for the next apple."""
    if not S["row_done"][i]:
        stem = T().target_stem
        on_tree = bool(stem.active[i]) and not bool(stem.broken[i])
        if on_tree and S["attempt"][i] < args.max_attempts:
            S["retry"][i] = True                       # still hanging: reach + grasp + harvest it again
        else:
            record_row(i)
        S["row_done"][i] = True
    q = arm._processed_actions[i]
    S["home_tries"][i] += 1
    if float((q - basket_wps[0]).norm()) > 0.05 and S["home_tries"][i] <= 2:
        S["state"][i] = RETURN
        start_chain(i, [q.clone(), basket_wps[0]])
        return
    if S["retry"][i]:
        T().retry_request[i] = True
        T().next_request[i] = True
        S["state"][i] = NEXT
    elif T().remaining(i) > 0:
        T().next_request[i] = True
        S["state"][i] = NEXT
    else:
        S["state"][i] = TREE_DONE
        report_tree(i)


def record_row(i):
    tr = T()
    dmg = env._fruit_damage_force[i] if hasattr(env, "_fruit_damage_force") else float("nan")
    stem = tr.target_stem
    if S["drop_phase"][i] == "reach/grasp":
        out = "KNOCKED OFF in reach/grasp"
    elif not bool(stem.broken[i]):
        out = "LEFT ON TREE" + (f" (slipped out in {S['drop_phase'][i]})" if S["drop_phase"][i] else "")
    elif S["drop_phase"][i]:
        out = f"DROPPED in {S['drop_phase'][i]}"
    elif S["in_basket"][i]:
        out = "FINISHED"
    else:
        out = "MISSED BASKET"
    if out == "FINISHED" and float(S["max_f"][i]) > float(dmg):
        out += " but BRUISED"
    if S["not_ready"][i]:
        out += " (timeout-ready)"
    hanging, _ = neighbours_on_tree()
    knocked = (S["hanging0"][i] & ~hanging[i] & (tr.status[i] == at.WAITING)).nonzero(as_tuple=True)[0].tolist()
    n_pick = len(S["rows"][i]) + 1
    for s in knocked:
        S["knocked"][i][s] = n_pick
    s = S["spot"][i]
    a = tr.spot_pos[i]
    gaps = (a - a[s]).norm(dim=-1) - 2 * float(env._apple_radius[i])
    gaps[s] = float("inf")
    S["rows"][i].append(dict(
        n=n_pick, spot=s, cluster=int((tr.cluster[i] == tr.cluster[i, s]).sum()), z=float(a[s, 2]),
        gap=float(gaps.min()), ready=float(S["t_ready"][i]), broke=bool(stem.broken[i]),
        cause=hv.BREAK_CAUSES[int(stem.break_cause[i])], twist=float(S["twist_at_break"][i]),
        slip=float(S["max_slip"][i]), f=float(S["max_f"][i]), f_reach=float(S["max_f_reach"][i]), thr=float(dmg), self=float(S["max_self"][i]),
        push=float(S["max_push"][i]), knocked=len(knocked), out=out, not_ready=S["not_ready"][i],
        tw=("+" if S["tw_sign"][i] > 0 else "-") if not math.isnan(S["tw_clear"][i][0]) else " ",
        tw_clear=S["tw_clear"][i], attempts=S["attempt"][i], goal_to=S["goal_timeouts"][i],
        tw_stop=S["twist_stop_why"][i],
        grip=float(env._target_grip_level[i]) if hasattr(env, "_target_grip_level") else float("nan")))


def report_tree(i):
    if S["reported"][i]:
        return
    S["reported"][i] = True
    tr = T()
    r = float(env._apple_radius[i])
    print(f"\n===== TREE env {i}  ({M} apples, radius {r * 1000:.0f} mm, clusters "
          f"{sorted(torch.bincount(tr.cluster[i]).tolist(), reverse=True)})   reach {os.path.basename(reach_path)}, grasp {'scripted' if args.grasp == 'scripted' else os.path.basename(os.path.dirname(os.path.dirname(args.policy)))}")
    print("  #  spot clu  z(m) gap mm ready_s stem   cause twist@brk dir (cost +/- cm)  slip   peak_F (thr)  reach+grasp_F  "
          "self mm  push cm knocked -> outcome")
    for row in S["rows"][i]:
        print(f" {row['n']:2d}   {row['spot']:2d}   {row['cluster']}  {row['z']:.2f}  {row['gap'] * 1000:+5.0f}  "
              f"{fmt(row['ready']):>6s}  {'broke' if row['broke'] else 'intact':6s} {row['cause']:5s} "
              f"{fmt(row['twist'], '{:.0f}deg'):>7s}  {row['tw']}  ({_cm(row['tw_clear'][0])}/{_cm(row['tw_clear'][1])})  {row['slip'] * 100:4.1f}cm {row['f']:5.1f} N ({row['thr']:4.1f})  {row['f_reach']:6.1f} N    "
              f"  {row['self'] * 1000:5.0f}   {row['push'] * 100:5.1f}   {row['knocked']:3d}     -> {row['out']}"
              + (f"  [attempt {row['attempts']}, grip command {row['grip']:.0f} N]" if row['attempts'] > 1 else
                 f"  [grip command {row['grip']:.0f} N]")
              + (f"  [goal timeouts {row['goal_to']}]" if row.get('goal_to') else "")
              + (f"  [twist stopped: {row['tw_stop']}]" if row.get('tw_stop') else ""))
    for s, by in sorted(S["knocked"][i].items()):
        print(f"      spot {s:2d}: KNOCKED OFF while picking #{by}")
    rows = S["rows"][i]
    fin = sum(r_["out"].startswith("FINISHED") for r_ in rows)
    print(f"  tree env {i}: {fin}/{M} in the basket, {sum(r_['out'].startswith('DROPPED') for r_ in rows)} dropped, "
          f"{sum(r_['out'].startswith('LEFT') for r_ in rows)} left on the tree, {len(S['knocked'][i])} knocked off by the hand, "
          f"{sum('BRUISED' in r_['out'] for r_ in rows)} bruised, {sum(bool(r_['not_ready']) for r_ in rows)} timeout-ready")
    sys.stdout.flush()


print(f"[tree bench] {N} envs x {M} apples, reach {reach_path} ({reach_layout} obs), grasp "
      f"{'SCRIPTED (plan B)' if args.grasp == 'scripted' else args.policy} ({grasp_dim} inputs), fingers after READY: "
      f"{'policy' if FINGERS_POLICY else 'frozen'}", flush=True)
new_trees()
rnd, step = 1, 0
twist_rad = math.radians(args.twist_deg)
tot = {"apples": 0, "finished": 0, "dropped": 0, "knocked": 0, "left": 0}
_wall0, _step0 = time.time(), 0
while app.is_running():
    with torch.inference_mode():
        a = policy().clone()
        st = S["state"]
        since_handoff = (env.episode_length_buf - env._grasp_handoff_step).float() * dt

        q_now = robot.data.joint_pos[:, arm_ids]
        for i in (st == GRASP).nonzero(as_tuple=True)[0].tolist():
            S["path"][i].append(q_now[i].clone())
            if S["handoff_idx"][i] is None and bool(env._grasp_phase_active[i]):
                S["handoff_idx"][i] = len(S["path"][i])

        ready = ready_det.update() & (st == GRASP)
        if args.grasp == "scripted":
            ready = scripted_grasp(a, st) & (st == GRASP)
        timeout = (st == GRASP) & env._grasp_phase_active & (since_handoff > args.grasp_timeout) & ~ready
        if bool(timeout.any()):
            L = ready_det.last
            for i in timeout.nonzero(as_tuple=True)[0].tolist():
                S["not_ready"][i] = (f"closure {int(L['closure'][i])}, thumb {float(L['thumb_N'][i]):.1f} N, "
                                     f"finger span {float(L['max_dq'][i]):.2f} rad")
            ready = ready | timeout
        for i in ready.nonzero(as_tuple=True)[0].tolist():
            S["t_ready"][i] = since_handoff[i]
            S["hand_hold"][i] = hand._processed_actions[i]
            S["palm0_p"][i] = robot.data.body_pos_w[i, palm_id]
            S["palm0_q"][i] = robot.data.body_quat_w[i, palm_id]
            S["q_ready"][i] = arm._processed_actions[i].clone()
            S["fruit0"][i] = fruit_in_palm()[i]
            pq_i = robot.data.body_quat_w[i:i + 1, palm_id]
            S["tw_axis"][i] = quat_apply(pq_i, torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0]
            S["tw_pivot"][i] = robot.data.body_pos_w[i, palm_id] + quat_apply(pq_i, c_local[:1])[0]
            sgn, c_pos, c_neg = twist_sign(i)
            S["tw_sign"][i], S["tw_clear"][i] = sgn, (c_pos, c_neg)
            env._stem_released[i] = True                  # arm unlock (the fruit is on its stem all along)
            st[i] = TWIST; S["t"][i] = 0.0

        # The reach or the grasp knocked the target off its stem. While the
        # reach phase lasts, the arm term follows the frozen reach policy and
        # ignores these commands -- it chased the falling fruit into the body
        # -- so end the reach phase now (the hand-off), and go home on the
        # next step, once the arm term has anchored to it.
        for i in [i for i in range(N) if S["pending_home"][i]]:
            S["pending_home"][i] = False
            S["q_ready"][i] = arm._processed_actions[i].clone()
            S["hand_hold"][i] = hand._processed_actions[i]
            start_home(i)
        tgt_off = (st == GRASP) & T().target_stem.broken & (fruit_to_hand() > args.leave_margin) \
            & ((apple.data.root_pos_w - env._stem_anchor_pos).norm(dim=-1) > 0.08) \
            & torch.tensor([not p_ for p_ in S["pending_home"]], device=dev)
        for i in tgt_off.nonzero(as_tuple=True)[0].tolist():
            S["drop_phase"][i] = "reach/grasp"
            if not bool(env._grasp_phase_active[i]):
                env._grasp_handoff_joint_pos[i] = robot.data.joint_pos[i].clone()
                env._grasp_handoff_step[i] = env.episode_length_buf[i]
                env._grasp_phase_active[i] = True
            env._stem_released[i] = True              # unlocks the proximal arm (locked until a grip otherwise)
            S["pending_home"][i] = True

        # ── arm targets ──
        q_cmd = arm._processed_actions.clone()
        tw = st == TWIST
        if bool(tw.any()):
            frac = hv.smoothstep(S["t"] / args.twist_s)
            zero = torch.zeros(N, device=dev)
            pose_p, pose_q = hv.harvest_palm_pose(S["palm0_p"], S["palm0_q"], S["tw_pivot"], S["tw_axis"],
                                                  torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(N, 3), zero,
                                                  twist_rad * frac * S["tw_sign"])
            q_ready = torch.stack([S["q_ready"][i] if S["q_ready"][i] is not None else arm._processed_actions[i]
                                   for i in range(N)])
            q_ik = ik.step(pose_p, pose_q, base=arm._processed_actions, max_dq=0.05, null_target=q_ready, null_gain=0.2,
                           weights=_twist_w)
            q_cmd = torch.where(tw.unsqueeze(-1), q_ik, q_cmd)
        pl = st == PULL
        if bool(pl.any()):
            frac = hv.smoothstep(S["t"] / args.pull_s).unsqueeze(-1)
            pose_p = S["pull_p0"] + S["pull_dir"] * args.pull * frac
            q_ik = ik.step(pose_p, S["pull_q0"], base=arm._processed_actions, max_dq=0.05,
                           null_target=arm._processed_actions, null_gain=0.0)
            q_cmd = torch.where(pl.unsqueeze(-1), q_ik, q_cmd)
        gl = st == GOAL
        if PATH_POLICY and bool(gl.any()):
            raw = _goal_jit(goal_obs())
            q_goal = torch.clamp(_arm_default + 0.5 * raw, -3.0, 3.0)        # the reach action's scale and clip
            q_cmd = torch.where(gl.unsqueeze(-1), q_goal, q_cmd)
            _goal_prev[gl] = raw[gl]
        mv = (st == BASKET) | (st == RETURN)
        if bool(mv.any()):
            f = hv.smoothstep(S["t"] / S["seg_T"]).unsqueeze(-1)
            q_cmd = torch.where(mv.unsqueeze(-1), S["q_from"] + (S["q_to"] - S["q_from"]) * f, q_cmd)
        for i in (st == HOME).nonzero(as_tuple=True)[0].tolist():
            k = S["home_k"][i]
            fade = 1.0 - float(hv.smoothstep(torch.tensor(k / max(len(S["home"][i]) - 1, 1))))
            q_cmd[i] = S["home"][i][min(k, len(S["home"][i]) - 1)] + fade * S["twist_dq"][i]
            S["home_k"][i] = k + 1
        rl = st == RELEASE
        q_cmd = torch.where(rl.unsqueeze(-1), S["q_to"], q_cmd)
        h = st != GRASP
        a[h, :arm_dim] = ((q_cmd - arm._anchor) / arm._scale)[h]
        if FINGERS_POLICY:                                    # the policy keeps the fingers while the fruit is carried
            carried = (st == TWIST) | (st == PULL) | (st == HOME) | (st == BASKET) | ((st == GOAL) & (S["goal_k"] != G_HOME))
            h_frozen = h & ~carried
        else:
            h_frozen = h
        a[h_frozen, arm_dim:] = ((S["hand_hold"] - hand._grasp_offset) / hand._scale)[h_frozen]

        pending = T().next_request.clone()
        env.step(a)
        step += 1
        if video is not None and step % video_every == 0:
            frame = env.render()
            if frame is not None:
                video.append_data(frame)
        reset_ids = pending.nonzero(as_tuple=True)[0].tolist()
        if reset_ids:
            new_apple(reset_ids)
            st = S["state"]

        # ── bookkeeping ──
        S["t"] = S["t"] + dt * (st != GRASP).float()
        carrying = (st == TWIST) | (st == PULL) | (st == HOME) | (st == BASKET) | ((st == GOAL) & (S["goal_k"] != G_HOME))
        live = carrying & torch.tensor([not d for d in S["drop_phase"]], device=dev)
        slip = (fruit_in_palm() - S["fruit0"]).norm(dim=-1)
        S["max_slip"] = torch.where(live, torch.maximum(S["max_slip"], slip), S["max_slip"])
        busy = (st != NEXT) & (st != TREE_DONE)
        # carried fruit only, like harvest_test_bench (the reach can hit the pinned
        # target, and the open hand can hit fruit lying in the basket)
        f_now = _max_link_apple_force(env, sc)
        S["max_f"] = torch.where(live, torch.maximum(S["max_f"], f_now), S["max_f"])
        S["max_f_reach"] = torch.where(st == GRASP, torch.maximum(S["max_f_reach"], f_now), S["max_f_reach"])
        S["max_self"] = torch.where(busy, torch.maximum(S["max_self"], self_collision_depth(env, margin=0.0).amax(-1)),
                                    S["max_self"])
        hanging, disp = neighbours_on_tree()
        push = torch.where(hanging, disp, torch.zeros_like(disp)).amax(-1)
        S["max_push"] = torch.where(busy, torch.maximum(S["max_push"], push), S["max_push"])
        stem = T().target_stem
        newly = stem.broken & torch.isnan(S["twist_at_break"]) & (st != GRASP)
        S["twist_at_break"] = torch.where(newly, stem.twist_deg.abs(), S["twist_at_break"])
        for i in (live & (fruit_to_hand() > args.leave_margin)).nonzero(as_tuple=True)[0].tolist():
            S["drop_phase"][i] = PHASE[int(st[i])] if st[i] != GOAL else ("to safe zone" if S["goal_k"][i] == G_SAFE else "to basket")
            if PATH_POLICY:
                if st[i] in (TWIST, PULL):
                    start_goal(i, G_SAFE)         # empty hand: still out of the wall first, then home
                elif st[i] == GOAL and S["goal_k"][i] == G_DROP:
                    start_goal(i, G_HOME)
                elif st[i] == GOAL:
                    ag.goal_state(env).carry[i] = False
            elif st[i] == TWIST:
                start_home(i)                     # straight home from wherever the twist got to
            elif st[i] == BASKET:
                go_back_from_basket(i)

        # ── transitions ──
        for i in ((st != GRASP) & busy).nonzero(as_tuple=True)[0].tolist():
            s_i, t_i = int(st[i]), float(S["t"][i])
            if s_i == TWIST and not S["twist_stop"][i] and twist_danger(i):
                S["twist_stop"][i] = True                 # into the body's danger zone / onto a neighbour
                S["twist_stop_why"][i] = _last_danger
                start_pull(i) if PATH_POLICY else start_home(i)
            elif s_i == TWIST and (t_i >= args.twist_s or (bool(stem.broken[i]) and t_i >= 0.1)):
                start_pull(i) if PATH_POLICY else start_home(i)
            elif s_i == PULL and t_i >= args.pull_s + 0.1:
                start_goal(i, G_SAFE)
            elif s_i == GOAL and (goal_done(i) or t_i >= args.goal_timeout):
                if t_i >= args.goal_timeout:
                    S["goal_timeouts"][i] += 1
                k = int(S["goal_k"][i])
                if k == G_SAFE:
                    start_goal(i, G_DROP if not S["drop_phase"][i] and bool(stem.broken[i]) else G_HOME)
                elif k == G_DROP:
                    S["q_to"][i] = arm._processed_actions[i].clone()
                    S["hand_hold"][i] = hand_open[i]
                    S["hand_hold_open"][i] = True
                    st[i] = RELEASE; S["t"][i] = 0.0
                else:
                    finish_apple(i)
            elif s_i == HOME and S["home_k"][i] >= len(S["home"][i]) + int(0.3 / dt):
                if S["drop_phase"][i] or not bool(stem.broken[i]):
                    finish_apple(i)
                else:
                    go_basket(i)
            elif s_i in (BASKET, RETURN) and t_i >= float(S["seg_T"][i]) + 0.05:
                S["seg"][i] += 1
                if S["seg"][i] < len(S["chain"][i]) - 1:
                    next_segment(i)
                elif s_i == BASKET:
                    S["hand_hold"][i] = hand_open[i]
                    st[i] = RELEASE; S["t"][i] = 0.0
                else:
                    finish_apple(i)
            elif s_i == RELEASE and t_i >= args.release_s:
                rad = env._apple_radius[i:i + 1]
                S["in_basket"][i] = bool(bk.apple_in_basket(
                    apple.data.root_pos_w[i:i + 1] - env.scene.env_origins[i:i + 1], rad)[0])
                if PATH_POLICY:
                    start_goal(i, G_HOME)
                else:
                    S["state"][i] = RETURN
                    start_chain(i, [S["q_to"][i].clone()] + list(basket_wps.flip(0)[1:]))

        if step % 250 == 0:
            from collections import Counter
            cnt = Counter(PHASE[int(x)] for x in st)
            now = time.time()
            rtf = (step - _step0) * dt / max(now - _wall0, 1e-6)
            _wall0, _step0 = now, step
            done = sum(len(r_) for r_ in S["rows"])
            print(f"  t={step * dt:6.1f}s  real-time factor {rtf:4.2f}x  {dict(cnt)}  apples handled {done}/{N * M}", flush=True)

        if bool((S["state"] == TREE_DONE).all()):
            for i in range(N):
                rows = S["rows"][i]
                tot["apples"] += M
                tot["finished"] += sum(r_["out"].startswith("FINISHED") for r_ in rows)
                tot["dropped"] += sum(r_["out"].startswith("DROPPED") for r_ in rows)
                tot["left"] += sum(r_["out"].startswith("LEFT") for r_ in rows)
                tot["knocked"] += len(S["knocked"][i])
            print(f"\n===== ROUND {rnd} done: {N} trees.  Running totals over {tot['apples']} apples: "
                  f"{tot['finished']} in the basket, {tot['dropped']} dropped, {tot['left']} left on the tree, "
                  f"{tot['knocked']} knocked off by the hand", flush=True)
            if args.rounds and rnd >= args.rounds:
                break
            rnd += 1
            new_trees()
            step = 0
            _wall0, _step0 = time.time(), 0
            print(f"\n>>> ROUND {rnd} <<<\n", flush=True)

if video is not None:
    video.close()
    print(f"[tree bench] video written: {args.video} ({args.video_fps} fps, simulated time)", flush=True)
sys.stdout.flush()
os._exit(0)
