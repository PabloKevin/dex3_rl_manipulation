#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Check sim2real/arm_pose_io.py (the AppleArmPose policy on the robot) against Isaac Lab's
AppleArmPose-v1, in two steps (two conda envs), as validate_single_io.py:

1. COLLECT (env_isaaclab): a few envs driven by the policy (the env chains its own goals), every step's
   joints, body poses, the goal (target / normal / finger / flags / pending), observation, raw and
   processed actions, the goal tracker's success, plus the cfg constants:

    ~/miniconda3/envs/env_isaaclab/bin/python sim2real/validate_arm_pose_io.py --collect --headless \\
        --policy logs/rsl_rl/apple_arm_pose/<run>/exported/policy.pt

2. CHECK (g1_dex3_sim2real: pinocchio + the g1_fruits_picker URDF, as on the robot):

    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/validate_arm_pose_io.py --check

   constants, the 35 inputs' terms in order, every slice rebuilt from the joints and the goal, the rate
   limiter one step at a time, the goal tracker's reached events, and the policy on the rebuilt inputs.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_NPZ = os.path.join(REPO, "logs", "sim2real", "arm_pose_io_reference.npz")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--collect", action="store_true")
p.add_argument("--check", action="store_true")
p.add_argument("--npz", default=DEFAULT_NPZ)
p.add_argument("--policy", default=None)
p.add_argument("--task", default="AppleArmPose-v1")
p.add_argument("--num_envs", type=int, default=6)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--slow", type=float, default=0.5)
p.add_argument("--slower", type=float, default=0.3)
p.add_argument("--urdf", default=None)
args, _rest = p.parse_known_args()
if args.collect == args.check:
    p.error("pick one of --collect / --check")

EXPECTED_TERMS = ["arm_pos", "arm_vel", "goal_delta", "goal_position", "goal_normal", "goal_finger", "goal_flags",
                  "last_action"]
EXPECTED_DIMS = [7, 7, 3, 3, 3, 3, 2, 7]


def collect():
    from isaaclab.app import AppLauncher
    ap_ = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap_)
    app = AppLauncher(ap_.parse_args(_rest)).app  # noqa: F841

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import envs  # noqa: F401
    from envs import apple_arm_pose_env_cfg as A
    from envs.apple_grasp_env_cfg import RIGHT_ARM_JOINTS
    from envs.mdp import arm_pose as ap

    if args.policy is None:
        sys.exit("--policy: the AppleArmPose policy to check")
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    dev = env.device
    pol = torch.jit.load(args.policy, map_location=dev).eval()
    robot = env.scene["robot"]
    arm_t = env.action_manager.get_term("right_arm")
    N = env.num_envs
    speed = torch.ones(N, 1, device=dev)
    speed[N // 3:] = args.slow
    speed[2 * N // 3:] = args.slower
    arm_t.speed_scale[:] = speed
    bodies = ["pelvis", "torso_link", "right_hand_palm_link", "right_shoulder_pitch_link"]
    bid = [robot.body_names.index(b) for b in bodies]
    obs = env.reset()[0]["policy"]
    rec = {k: [] for k in ("joint_pos", "joint_vel", "body_pos", "body_quat", "target", "normal", "finger",
                           "finger_on", "via", "pending", "obs", "raw", "arm_target", "ep_len", "success")}
    for _ in range(args.steps):
        org = env.scene.env_origins
        P = ap.pose_state(env)
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        rec["body_pos"].append((robot.data.body_pos_w[:, bid] - org[:, None]).cpu().numpy())
        rec["body_quat"].append(robot.data.body_quat_w[:, bid].cpu().numpy())
        rec["target"].append((P.target - org).cpu().numpy())
        rec["normal"].append(P.normal.cpu().numpy())
        rec["finger"].append(P.finger.cpu().numpy())
        rec["finger_on"].append(P.finger_on.cpu().numpy())
        rec["via"].append(P.via.cpu().numpy())
        rec["pending"].append(P.pending.cpu().numpy())
        rec["obs"].append(obs.cpu().numpy())
        rec["ep_len"].append(env.episode_length_buf.cpu().numpy().copy())
        with torch.no_grad():
            a = pol(obs)
        rec["raw"].append(a.cpu().numpy())
        obs = env.step(a)[0]["policy"]
        rec["arm_target"].append(arm_t.processed_actions.cpu().numpy())
        rec["success"].append(ap.pose_state(env).n_reached.cpu().numpy().copy())
    arm_cfg = cfg.actions.right_arm
    consts = {
        "dt": cfg.sim.dt * cfg.decimation,
        "arm_joint_names": [robot.joint_names[i] for i in arm_t._joint_ids],
        "arm_default": robot.data.default_joint_pos[0, arm_t._joint_ids].cpu().tolist(),
        "waist_pitch_default": float(robot.data.default_joint_pos[0, robot.joint_names.index("waist_pitch_joint")]),
        "arm_scale": arm_cfg.scale, "arm_vmax": arm_cfg.max_velocity, "arm_amax": arm_cfg.max_acceleration,
        "arm_brake": bool(arm_cfg.brake_to_target), "arm_clip": [list(arm_cfg.clip[n]) for n in RIGHT_ARM_JOINTS],
        "pocket": list(A.POCKET), "track": dict(cfg.terminations.goal_tracker.params, pocket_local=None),
        "hand": dict(cfg.events.hand_preshape.params.get("targets", {}) if hasattr(cfg.events, "hand_preshape") else {}),
        "obs_terms": list(env.observation_manager.active_terms["policy"]),
        "obs_dims": [int(d[0]) if isinstance(d, tuple) else int(d)
                     for d in env.observation_manager.group_obs_term_dim["policy"]],
        "action_terms": list(env.action_manager.active_terms),
    }
    os.makedirs(os.path.dirname(args.npz), exist_ok=True)
    np.savez_compressed(args.npz, **{k: np.stack(v) for k, v in rec.items()}, speed=speed[:, 0].cpu().numpy(),
                        joint_names=np.array(robot.joint_names), bodies=np.array(bodies), consts=json.dumps(consts),
                        policy=args.policy)
    print(f"[collect] {args.steps} steps x {N} envs -> {args.npz}")
    env.close()


def check():
    import numpy as np
    sys.path.insert(0, HERE)
    import reach_io as R
    import arm_pose_io as AP

    AP.use_constants()
    d = dict(np.load(args.npz, allow_pickle=False))
    C = json.loads(str(d["consts"]))
    names = list(d["joint_names"])
    T, N, _ = d["joint_pos"].shape
    fails = []

    def report(what, err, tol, unit=""):
        ok = err <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {what:<44s} max err {err:.3g}{unit} (tol {tol:g}{unit})")
        if not ok:
            fails.append(what)

    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and all(k in b and same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple, np.ndarray)):
            return isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b) and all(map(same, a, b))
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return abs(float(a) - float(b)) <= 1e-6

    print("constants (arm_pose_io / reach_io vs the cfg):")
    tr = {k: v for k, v in C["track"].items() if k != "pocket_local"}
    for what, mine, theirs in [
        ("dt", R.DT, C["dt"]), ("arm joint order", R.RIGHT_ARM_JOINTS, C["arm_joint_names"]),
        ("arm default", list(R.ARM_DEFAULT), C["arm_default"]), ("waist pitch default", R.WAIST_DEFAULT[2],
                                                                  C["waist_pitch_default"]),
        ("arm scale/vmax/amax/brake", [R.ARM_SCALE, R.ARM_MAX_VELOCITY, R.ARM_MAX_ACCELERATION, R.ARM_BRAKE],
         [C["arm_scale"], C["arm_vmax"], C["arm_amax"], C["arm_brake"]]),
        ("arm target clip", [[a, b] for a, b in zip(R.ARM_TARGET_LO, R.ARM_TARGET_HI)], C["arm_clip"]),
        ("pocket", list(AP.POCKET), C["pocket"]), ("goal tracker", AP.TRACK, tr),
        ("action terms", ["right_arm"], C["action_terms"]),
        ("obs terms", EXPECTED_TERMS, C["obs_terms"]), ("obs dims", EXPECTED_DIMS, C["obs_dims"]),
    ]:
        ok = same(mine, theirs)
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + ("" if ok else f": mine {mine} vs cfg {theirs}"))
        if not ok:
            fails.append(what)

    def qmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    kin = R.G1Kinematics(args.urdf or R.URDF_DEFAULT)
    arm_i = [names.index(n) for n in R.RIGHT_ARM_JOINTS]
    waist_i = [names.index(n) for n in R.WAIST_JOINTS]
    pel = list(d["bodies"]).index("pelvis")
    sl = {"arm q": slice(0, 7), "arm dq": slice(7, 14), "goal - pocket (torso)": slice(14, 17),
          "goal - shoulder (torso)": slice(17, 20), "goal normal (palm)": slice(20, 23),
          "goal finger (palm)": slice(23, 26), "flags": slice(26, 28), "prev raw": slice(28, 35)}
    tol = {"arm q": 1e-5, "arm dq": 1e-4, "goal - pocket (torso)": 1.5e-3, "goal - shoulder (torso)": 1.5e-3,
           "goal normal (palm)": 2e-3, "goal finger (palm)": 2e-3, "flags": 0.0, "prev raw": 1e-5}
    err = {k: 0.0 for k in sl}
    mine_all = np.zeros_like(d["obs"])
    n_succ = n_match = 0
    for e in range(N):
        trk = AP.Tracker(dt=C["dt"])
        last_target = None
        for t in range(T):
            jp, jv = d["joint_pos"][t, e], d["joint_vel"][t, e]
            k = int(d["ep_len"][t, e])
            Rp = qmat(d["body_quat"][t, e, pel])
            to_pel = lambda p: Rp.T @ (p - d["body_pos"][t, e, pel])
            fr = kin.frames(jp[waist_i], jp[arm_i])
            if bool(d["pending"][t, e]):
                pocket, normal, finger = AP.hand_frame(fr)
                g = AP.Goal("pending", pocket, normal, finger, bool(d["finger_on"][t, e]), bool(d["via"][t, e]))
            else:
                g = AP.Goal("g", to_pel(d["target"][t, e]), Rp.T @ d["normal"][t, e], Rp.T @ d["finger"][t, e],
                            bool(d["finger_on"][t, e]), bool(d["via"][t, e]))
            prev = d["raw"][t - 1, e] if (t > 0 and k > 0) else np.zeros(7)
            o = AP.build_obs(jp[arm_i], jv[arm_i], fr, g, prev)
            mine_all[t, e] = o
            for key, s in sl.items():
                err[key] = max(err[key], float(np.abs(o[s] - d["obs"][t, e][s]).max()))
            # the tracker, on the step's post-physics state (the next record) vs the sim's success
            if t + 1 < T and int(d["ep_len"][t + 1, e]) > 0 and not bool(d["pending"][t, e]):
                if last_target is None or np.linalg.norm(g.target - last_target) > 1e-6:
                    trk.reset()
                    last_target = g.target.copy()
                jp1 = d["joint_pos"][t + 1, e]
                fr1 = kin.frames(jp1[waist_i], jp1[arm_i])
                mine_s = trk.step(fr1, g)
                # the env's per-episode "goals reached" counter went up on this step
                sim_s = int(d["ep_len"][t + 1, e]) > int(d["ep_len"][t, e]) and \
                    float(d["success"][t, e]) > (float(d["success"][t - 1, e]) if t > 0 and k > 0 else 0.0)
                n_succ += int(sim_s)
                n_match += int(sim_s and mine_s)
    print("observation, rebuilt vs Isaac Lab's:")
    for key in sl:
        report(key, err[key], tol[key])
    print(f"goal tracker: {n_match} of {n_succ} sim 'reached' events found on the same step (finite-difference speed)")
    if n_succ and n_match < 0.8 * n_succ:
        fails.append("goal tracker")

    def one_step(dtype):
        ea = 0.0
        for e in range(N):
            L = R.make_limiters("visible", dtype=dtype)["arm"]
            sp = float(d["speed"][e])
            for t in range(T):
                jp = d["joint_pos"][t, e]
                k = int(d["ep_len"][t, e])
                tgt = d["arm_target"]
                L.k = min(k, R.RateLimiter.ANCHOR_STEPS)
                if k >= R.RateLimiter.ANCHOR_STEPS:
                    prev_prev = d["joint_pos"][t - 1, e, arm_i] if d["ep_len"][t - 1, e] <= 1 else tgt[t - 2, e]
                    L.prev = tgt[t - 1, e].astype(dtype)
                    L.vel = ((tgt[t - 1, e] - prev_prev) / R.DT).astype(dtype)
                out = L.step(d["raw"][t, e, :7], jp[arm_i], speed=sp)
                ea = max(ea, float(np.abs(out - tgt[t, e]).max()))
        return ea
    print(f"rate limiter (speed factors {sorted(set(np.round(d['speed'], 3)))}):")
    for dt_ in (np.float64, np.float32):
        report(f"arm targets, one step ({np.dtype(dt_).name})", one_step(dt_), 1e-4, " rad")

    import torch
    net = torch.jit.load(str(d["policy"]), map_location="cpu").eval()
    e_pol = 0.0
    for t in range(0, T, 3):
        with torch.no_grad():
            out = net(torch.as_tensor(mine_all[t].astype(np.float32))).numpy()
        e_pol = max(e_pol, float(np.abs(out - d["raw"][t]).max()))
    print("policy on the rebuilt observation vs the sim's actions:")
    report("raw actions", e_pol, 2e-2)
    print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if args.collect:
    collect()
else:
    check()
