#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Check reach_io.py's "single" layout (what toy_reach_single.py runs on the
robot) against Isaac Lab's AppleReachSingle-v1, in two steps -- the two
halves need different conda envs (as validate_reach_io.py, which checks the
"waist" / "tree" layouts and is not changed by this one):

1. COLLECT, in Isaac Lab (env_isaaclab): a few envs of AppleReachSingle-v1 at
   speed factors 1 / --slow / --slower, every step's joints, PERCEIVED and true
   apple, body poses, hand phase and hand targets, observation, raw and
   processed actions, the sim's goal distance, plus the cfg constants:

    ~/miniconda3/envs/env_isaaclab/bin/python sim2real/validate_single_io.py --collect --headless \
        --policy logs/rsl_rl/apple_reach_single_v1/<run>/exported/policy_<it>.pt

   Without --policy the pinned spindle reach policy (40 in / 8 out, a different
   layout) only DRIVES the arm so the palm moves near the apple and the hand
   schedule latches; the policy check is then skipped.

2. CHECK, without Isaac Lab (g1_dex3_sim2real -- pinocchio + the
   g1_fruits_picker URDF, exactly as on the robot):

    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/validate_single_io.py --check

   constants  : reach_io's SINGLE_* copies and the action constants vs the cfg
   layout     : the 40 inputs' terms and sizes, in order
   observation: every slice rebuilt from the joints and the perceived apple
                (clearance ring taken from the sim)
   hand       : SingleHandSchedule replayed from the rebuilt palm point and the
                perceived apple vs the sim's phase; its hand targets vs the sim's
   goal       : single_seat_error vs reach_rewards._palm_target_dist
   limiter    : RateLimiter one step at a time vs the action terms
   policy     : (trained single policy only) the policy on the rebuilt inputs
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_NPZ = os.path.join(REPO, "logs", "sim2real", "single_io_reference.npz")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--collect", action="store_true", help="step 1 (Isaac Lab)")
p.add_argument("--check", action="store_true", help="step 2 (no Isaac Lab)")
p.add_argument("--npz", default=DEFAULT_NPZ)
p.add_argument("--policy", default=None, help="exported AppleReachSingle-v1 policy (default: a driver only)")
p.add_argument("--num_envs", type=int, default=6)
p.add_argument("--steps", type=int, default=400, help="control steps to record (50 Hz)")
p.add_argument("--slow", type=float, default=0.5)
p.add_argument("--slower", type=float, default=0.3)
p.add_argument("--urdf", default=None)
args, _rest = p.parse_known_args()
if args.collect == args.check:
    p.error("pick one of --collect / --check")

EXPECTED_TERMS = ["right_arm_pos", "right_arm_vel", "apple_pos_palm", "palm_orientation_torso", "last_action",
                  "neighbour_ring", "waist_yaw_pos", "waist_yaw_vel", "waist_last_action", "hand_phase"]
EXPECTED_DIMS = [7, 7, 3, 4, 7, 8, 1, 1, 1, 1]


# ─────────────────────────────────────────────
#  1. collect (Isaac Lab)
# ─────────────────────────────────────────────
def collect():
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    a2 = ap.parse_args(_rest)
    app = AppLauncher(a2).app  # noqa: F841

    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab.managers import SceneEntityCfg

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import envs  # noqa: F401
    from envs import apple_grasp_env_cfg as G
    from envs import apple_reach_single_env_cfg as S
    from envs.mdp import reach_rewards as rr
    from envs.mdp import single_reach as sr

    cfg = S.AppleReachSingleEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("AppleReachSingle-v1", cfg=cfg).unwrapped
    dev = env.device
    trained = args.policy is not None
    pol_path = args.policy or os.path.join(REPO, "envs", "agents", "best_weights", "grasp_handoff_reach_spindle.pt")
    pol = torch.jit.load(pol_path, map_location=dev).eval()
    robot, apple = env.scene["robot"], env.scene["apple"]
    arm_t = env.action_manager.get_term("right_arm")
    waist_t = env.action_manager.get_term("waist_yaw")
    N = env.num_envs
    speed = torch.ones(N, 1, device=dev)
    speed[N // 3:] = args.slow
    speed[2 * N // 3:] = args.slower
    for t in (arm_t, waist_t):
        if not hasattr(t, "speed_scale"):
            sys.exit("the action terms have no speed_scale (envs/mdp/actions.py without the speed factor)")
        t.speed_scale[:] = speed
    bodies = ["pelvis", "torso_link", "right_hand_palm_link", "right_shoulder_pitch_link", "d435_link"]
    bid = [robot.body_names.index(b) for b in bodies]
    hp = cfg.events.hand_schedule.params
    hand_names = list(hp["open_targets"])
    hand_ids = [robot.joint_names.index(n) for n in hand_names]
    goal_cfg = env.reward_manager.get_term_cfg("hold").params["robot_cfg"]       # resolved body ids
    apple_cfg = SceneEntityCfg("apple")

    obs = env.reset()[0]["policy"]
    rec = {k: [] for k in ("joint_pos", "joint_vel", "apple", "apple_meas", "obs", "raw", "arm_target",
                           "waist_target", "ep_len", "body_pos", "body_quat", "hand_phase", "hand_target",
                           "goal_dist", "radius")}
    for _ in range(args.steps):
        org = env.scene.env_origins
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        rec["apple"].append((apple.data.root_pos_w - org).cpu().numpy())
        rec["apple_meas"].append((sr.apple_pos_measured(env, apple_cfg) - org).cpu().numpy())
        rec["body_pos"].append((robot.data.body_pos_w[:, bid] - org[:, None]).cpu().numpy())
        rec["body_quat"].append(robot.data.body_quat_w[:, bid].cpu().numpy())
        rec["hand_phase"].append(getattr(env, "_hand_phase", torch.zeros(N, device=dev)).cpu().numpy().copy())
        rec["hand_target"].append(robot.data.joint_pos_target[:, hand_ids].cpu().numpy())
        rec["goal_dist"].append(rr._palm_target_dist(env, goal_cfg, apple_cfg, S.SINGLE_STANDOFF,
                                                     G.PALM_LOCAL_OFFSET, G.PALM_NORMAL_AXIS).cpu().numpy())
        r = getattr(env, "_apple_radius", None)
        rec["radius"].append(r.cpu().numpy() if isinstance(r, torch.Tensor) else np.full(N, rr.DEFAULT_APPLE_RADIUS))
        rec["obs"].append(obs.cpu().numpy())
        rec["ep_len"].append(env.episode_length_buf.cpu().numpy().copy())
        with torch.no_grad():
            a = pol(obs)
        rec["raw"].append(a.cpu().numpy())
        obs = env.step(a)[0]["policy"]
        rec["arm_target"].append(arm_t.processed_actions.cpu().numpy())
        rec["waist_target"].append(waist_t.processed_actions.cpu().numpy())

    arm_cfg, w_cfg = cfg.actions.right_arm, cfg.actions.waist_yaw
    consts = {
        "dt": cfg.sim.dt * cfg.decimation,
        "arm_joint_names": [robot.joint_names[i] for i in arm_t._joint_ids],
        "arm_default": robot.data.default_joint_pos[0, arm_t._joint_ids].cpu().tolist(),
        "waist_default": [float(robot.data.default_joint_pos[0, robot.joint_names.index(n)])
                          for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")],
        "arm_scale": arm_cfg.scale, "arm_vmax": arm_cfg.max_velocity, "arm_amax": arm_cfg.max_acceleration,
        "arm_brake": bool(arm_cfg.brake_to_target),
        "arm_clip": [list(arm_cfg.clip[n]) for n in G.RIGHT_ARM_JOINTS],
        "waist_scale": w_cfg.scale, "waist_vmax": w_cfg.max_velocity, "waist_amax": w_cfg.max_acceleration,
        "waist_brake": bool(w_cfg.brake_to_target), "waist_clip": list(w_cfg.clip[".*"]),
        "palm_local_offset": list(G.PALM_LOCAL_OFFSET), "palm_normal_axis": list(G.PALM_NORMAL_AXIS),
        "single_standoff": S.SINGLE_STANDOFF, "single_pos_tol": S.SINGLE_POS_TOL,
        "success_pos_tol": float(cfg.terminations.reach_success.params.get("pos_tol", float("nan"))),
        "finger_switch_dz": S.FINGER_SWITCH_DZ,
        "hand_open": dict(S.HAND_OPEN), "hand_closed": dict(S.HAND_CLOSED),
        "hand_level_tol": hp["level_tol"], "hand_ramp_s": hp["ramp_s"],
        "hand_names": hand_names, "approach_origin": cfg.events.approach_origin.params["body"],
        "obs_terms": list(env.observation_manager.active_terms["policy"]),
        "obs_dims": [int(d[0]) if isinstance(d, tuple) else int(d)
                     for d in env.observation_manager.group_obs_term_dim["policy"]],
        "action_terms": list(env.action_manager.active_terms),
    }
    os.makedirs(os.path.dirname(args.npz), exist_ok=True)
    np.savez_compressed(args.npz, **{k: np.stack(v) for k, v in rec.items()},
                        speed=speed[:, 0].cpu().numpy(), joint_names=np.array(robot.joint_names),
                        bodies=np.array(bodies), consts=json.dumps(consts), policy=pol_path,
                        trained=np.array(trained))
    print(f"[collect] {args.steps} steps x {N} envs -> {args.npz} (policy {'single' if trained else 'driver only'})")
    env.close()


# ─────────────────────────────────────────────
#  2. check (no Isaac Lab)
# ─────────────────────────────────────────────
def check():
    import numpy as np
    sys.path.insert(0, HERE)
    import reach_io as R

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
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple, np.ndarray)):
            return isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b) and all(map(same, a, b))
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return abs(float(a) - float(b)) <= 1e-6

    print("constants (reach_io vs the AppleReachSingle-v1 cfg):")
    cmp = [
        ("dt", R.DT, C["dt"]), ("arm joint order", R.RIGHT_ARM_JOINTS, C["arm_joint_names"]),
        ("arm default", list(R.ARM_DEFAULT), C["arm_default"]),
        ("waist default (yaw, roll, pitch)", list(R.WAIST_DEFAULT), C["waist_default"]),
        ("arm scale/vmax/amax/brake", [R.ARM_SCALE, R.ARM_MAX_VELOCITY, R.ARM_MAX_ACCELERATION, False],
         [C["arm_scale"], C["arm_vmax"], C["arm_amax"], C["arm_brake"]]),
        ("arm target clip (v3)", [[a, b] for a, b in zip(R.ARM_TARGET_LO, R.ARM_TARGET_HI)], C["arm_clip"]),
        ("waist scale/vmax/amax/brake", [R.WAIST_SCALE, R.WAIST_MAX_VELOCITY, R.WAIST_MAX_ACCELERATION, True],
         [C["waist_scale"], C["waist_vmax"], C["waist_amax"], C["waist_brake"]]),
        ("waist clip", [-R.WAIST_YAW_LIMIT, R.WAIST_YAW_LIMIT], C["waist_clip"]),
        ("palm offset / normal", [list(R.PALM_LOCAL_OFFSET), list(R.PALM_NORMAL_AXIS)],
         [C["palm_local_offset"], C["palm_normal_axis"]]),
        ("standoff / goal tol / success tol", [R.SINGLE_STANDOFF, R.SINGLE_POS_TOL, R.SINGLE_POS_TOL],
         [C["single_standoff"], C["single_pos_tol"], C["success_pos_tol"]]),
        ("finger switch height", R.SINGLE_FINGER_SWITCH_DZ, C["finger_switch_dz"]),
        ("hand open", R.SINGLE_HAND_OPEN, C["hand_open"]), ("hand closed", R.SINGLE_HAND_CLOSED, C["hand_closed"]),
        ("hand level tol / ramp", [R.SINGLE_HAND_LEVEL_TOL, R.SINGLE_HAND_RAMP_S],
         [C["hand_level_tol"], C["hand_ramp_s"]]),
        ("approach origin", "pelvis", C["approach_origin"]),
        ("action terms", ["right_arm", "waist_yaw"], C["action_terms"]),
        ("obs terms", EXPECTED_TERMS, C["obs_terms"]), ("obs dims", EXPECTED_DIMS, C["obs_dims"]),
        ("layout size", list(R.LAYOUT_DIMS["single"]), [sum(C["obs_dims"]), 8]),
    ]
    for what, mine, theirs in cmp:
        ok = same(mine, theirs)
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + ("" if ok else f": reach_io {mine} vs cfg {theirs}"))
        if not ok:
            fails.append(what)

    def qmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    kin = R.G1Kinematics(args.urdf or R.URDF_DEFAULT)
    ji = {n: names.index(n) for n in R.RIGHT_ARM_JOINTS + R.WAIST_JOINTS}
    arm_i = [ji[n] for n in R.RIGHT_ARM_JOINTS]
    waist_i = [ji[n] for n in R.WAIST_JOINTS]
    hand_names = [n for n in names if n.startswith("right_hand")]
    bodies = list(d["bodies"])
    pel = bodies.index("pelvis")

    sl = {"arm q": slice(0, 7), "arm dq": slice(7, 14), "apple - palm (perceived)": slice(14, 17),
          "palm quat": slice(17, 21), "prev raw arm": slice(21, 28), "ring": slice(28, 36),
          "waist q": slice(36, 37), "waist dq": slice(37, 38), "prev raw waist": slice(38, 39),
          "hand phase": slice(39, 40)}
    tol = {"arm q": 1e-5, "arm dq": 1e-4, "apple - palm (perceived)": 1.5e-3, "palm quat": 5e-3,
           "prev raw arm": 1e-5, "ring": 1e-5, "waist q": 1e-5, "waist dq": 1e-4, "prev raw waist": 1e-5,
           "hand phase": 1e-5}
    err = {k: 0.0 for k in sl}
    flips = 0
    e_phase = e_htgt = e_goal = 0.0
    latches = 0
    mine_all = np.zeros_like(d["obs"])
    hand_order = [hand_names.index(f"right_hand_{n}_joint") for n in R.DEX3_ORDER] if len(hand_names) else []
    sim_hand = list(C["hand_names"])
    sim_order = [sim_hand.index(f"right_hand_{n}_joint") for n in R.DEX3_ORDER]
    for e in range(N):
        sched = R.SingleHandSchedule(ramp_s=C["hand_ramp_s"], level_tol=C["hand_level_tol"], dt=C["dt"])
        was = False
        for t in range(T):
            jp, jv = d["joint_pos"][t, e], d["joint_vel"][t, e]
            o_sim = d["obs"][t, e]
            k = int(d["ep_len"][t, e])
            prev = d["raw"][t - 1, e] if (t > 0 and k > 0) else np.zeros(8)
            Rp = qmat(d["body_quat"][t, e, pel])
            to_pel = lambda p: Rp.T @ (p - d["body_pos"][t, e, pel])
            meas, true = to_pel(d["apple_meas"][t, e]), to_pel(d["apple"][t, e])
            fr = kin.frames(jp[waist_i], jp[arm_i], {n: jp[names.index(n)] for n in hand_names})
            # the schedule: reset with the episode, then this step's update (an interval event,
            # after the reset events and before the observation)
            if t == 0 or k == 0:
                sched.reset()
                was = False
            if t > 0:
                sched.step(fr.palm_ref, meas)
            if sched.latched and not was:
                latches += 1
            was = sched.latched
            e_phase = max(e_phase, abs(sched.phase - float(d["hand_phase"][t, e])))
            if t > 0:
                e_htgt = max(e_htgt, float(np.abs(sched.targets() - d["hand_target"][t, e][sim_order]).max()))
            st = R.ReachState(arm_q=jp[arm_i], arm_dq=jv[arm_i], waist_q=jp[waist_i], waist_dq=jv[waist_i],
                              apple=meas, ring=o_sim[28:36], hand_phase=float(d["hand_phase"][t, e]))
            o, _ = R.build_obs("single", kin, st, prev, fr)
            if np.dot(o[17:21], o_sim[17:21]) < 0:
                flips += 1
                o[17:21] *= -1
            mine_all[t, e] = o
            for key, s in sl.items():
                err[key] = max(err[key], float(np.abs(o[s] - o_sim[s]).max()))
            e_goal = max(e_goal, abs(R.single_seat_error(fr, true, float(d["radius"][t, e])) - float(d["goal_dist"][t, e])))
    print("observation, rebuilt vs Isaac Lab's (ring injected from the sim; hand phase from the sim):")
    for key in sl:
        report(key, err[key], tol[key])
    print(f"  palm quat sign: {flips} of {T * N} sim quaternions had w < 0 after the w >= 0 rebuild")
    print(f"hand schedule replayed from the rebuilt palm point + the perceived apple ({latches} latches seen):")
    report("hand phase", e_phase, 0.08)
    report("hand targets (Dex3 order)", e_htgt, 0.08, " rad")
    if latches == 0:
        print("  WARN no latch in the recording: the schedule's closing was not exercised (longer --steps)")
        fails.append("hand schedule not exercised")
    print("goal (seat) distance:")
    report("single_seat_error vs _palm_target_dist", e_goal * 1000, 2.0, " mm")

    def one_step(dtype):
        ea = ew = 0.0
        for e in range(N):
            L = R.make_limiters("single", dtype=dtype)
            sp = float(d["speed"][e])
            for t in range(T):
                jp = d["joint_pos"][t, e]
                k = int(d["ep_len"][t, e])
                for name, ids, tgt, raw in (("arm", arm_i, d["arm_target"], d["raw"][t, e, :7]),
                                            ("waist", waist_i[:1], d["waist_target"], d["raw"][t, e, 7:8])):
                    lim = L[name]
                    lim.k = min(k, R.RateLimiter.ANCHOR_STEPS)
                    if k >= R.RateLimiter.ANCHOR_STEPS:
                        prev_prev = (d["joint_pos"][t - 1, e, ids] if d["ep_len"][t - 1, e] <= 1 else tgt[t - 2, e])
                        lim.prev = tgt[t - 1, e].astype(dtype)
                        lim.vel = ((tgt[t - 1, e] - prev_prev) / R.DT).astype(dtype)
                    out = lim.step(raw, jp[ids], speed=sp)
                    er = float(np.abs(out - tgt[t, e]).max())
                    if name == "arm":
                        ea = max(ea, er)
                    else:
                        ew = max(ew, er)
        return ea, ew
    print(f"rate limiter (speed factors {sorted(set(np.round(d['speed'], 3)))}):")
    for dt_ in (np.float64, np.float32):
        ea, ew = one_step(dt_)
        report(f"arm targets, one step ({np.dtype(dt_).name})", ea, 1e-4, " rad")
        report(f"waist target, one step ({np.dtype(dt_).name})", ew, 1e-4, " rad")

    if bool(d["trained"]):
        pol = R.ReachPolicy(str(d["policy"]), layout="single")
        e_pol = 0.0
        for t in range(0, T, 3):
            for e in range(N):
                e_pol = max(e_pol, float(np.abs(pol(mine_all[t, e]) - d["raw"][t, e]).max()))
        print("policy on the rebuilt observation vs the sim's actions:")
        report("raw actions", e_pol, 2e-2)
    else:
        print("policy: skipped (recorded with a driver policy; --collect --policy <single policy> to check it)")

    print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if args.collect:
    collect()
else:
    check()
