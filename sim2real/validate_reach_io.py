#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Check sim2real/reach_io.py (what the robot runs) against Isaac Lab (what the
reach policy was trained in), in two steps -- the two halves need different
conda envs:

1. COLLECT, in Isaac Lab (env_isaaclab): run AppleReachSpindle-v3 with the
   pinned reach policy, a few envs at speed factor 1 and a few slower, and
   save every step's joints, apple, body poses, observation vector, raw and
   processed actions, plus the configuration constants:

    ~/miniconda3/envs/env_isaaclab/bin/python sim2real/validate_reach_io.py --collect --headless

2. CHECK, without Isaac Lab (g1_dex3_sim2real -- pinocchio + the
   g1_fruits_picker URDF, exactly as on the robot):

    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/validate_reach_io.py --check

   constants  : reach_io's copies vs the env cfg
   FK         : palm / torso / shoulder pose relative to the pelvis, URDF vs USD
   observation: every slice of the 40-dim vector rebuilt from the joints and
                the apple (touch force and clearance ring taken from the sim)
   limiter    : RateLimiter, one step at a time from Isaac's own previous
                target, vs the action terms' processed targets, at speed
                factors 1 / 0.5 / 0.3 (the deployment speed factor)
   policy     : the policy on the rebuilt observation vs on Isaac's
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_NPZ = os.path.join(REPO, "logs", "sim2real", "reach_io_reference.npz")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--collect", action="store_true", help="step 1 (Isaac Lab)")
p.add_argument("--check", action="store_true", help="step 2 (no Isaac Lab)")
p.add_argument("--npz", default=DEFAULT_NPZ)
p.add_argument("--policy", default=None, help="exported reach policy (default: the pinned spindle reach)")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=400, help="control steps to record (50 Hz)")
p.add_argument("--slow", type=float, default=0.5, help="speed factor of the second third of the envs")
p.add_argument("--slower", type=float, default=0.3, help="speed factor of the last third of the envs")
p.add_argument("--urdf", default=None)
args, _rest = p.parse_known_args()
if args.collect == args.check:
    p.error("pick one of --collect / --check")


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

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import envs  # noqa: F401
    from envs import apple_grasp_env_cfg as G
    from envs import apple_reach_spindle_env_cfg as S
    from envs import apple_reach_tree_waist_env_cfg as W
    from envs.apple_grasp_tree_env_cfg import TREE_APPLE_RADIUS_RANGE

    cfg = S.AppleReachSpindleV3EnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("AppleReachSpindle-v3", cfg=cfg).unwrapped
    dev = env.device
    pol_path = args.policy or os.path.join(REPO, "envs", "agents", "best_weights", "grasp_handoff_reach_spindle.pt")
    pol = torch.jit.load(pol_path, map_location=dev).eval()
    robot, apple = env.scene["robot"], env.scene["apple"]
    arm_t = env.action_manager.get_term("right_arm")
    waist_t = env.action_manager.get_term("waist_yaw")
    N = env.num_envs
    # speed factors 1, --slow, --slower in thirds (the check replays each)
    speed = torch.ones(N, 1, device=dev)
    speed[N // 3:] = args.slow
    speed[2 * N // 3:] = args.slower
    for t in (arm_t, waist_t):
        if not hasattr(t, "speed_scale"):
            sys.exit("the action terms have no speed_scale (envs/mdp/actions.py without the speed factor)")
        t.speed_scale[:] = speed
    bodies = ["pelvis", "torso_link", "right_hand_palm_link", "right_shoulder_pitch_link", "d435_link"]
    bid = [robot.body_names.index(b) for b in bodies]

    obs = env.reset()[0]["policy"]
    rec = {k: [] for k in ("joint_pos", "joint_vel", "apple", "obs", "raw", "arm_target", "waist_target",
                           "ep_len", "body_pos", "body_quat")}
    for _ in range(args.steps):
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        rec["apple"].append((apple.data.root_pos_w - env.scene.env_origins).cpu().numpy())
        rec["body_pos"].append((robot.data.body_pos_w[:, bid] - env.scene.env_origins[:, None]).cpu().numpy())
        rec["body_quat"].append(robot.data.body_quat_w[:, bid].cpu().numpy())
        rec["obs"].append(obs.cpu().numpy())
        rec["ep_len"].append(env.episode_length_buf.cpu().numpy().copy())
        with torch.no_grad():
            a = pol(obs)
        rec["raw"].append(a.cpu().numpy())
        obs = env.step(a)[0]["policy"]
        rec["arm_target"].append(arm_t.processed_actions.cpu().numpy())
        rec["waist_target"].append(waist_t.processed_actions.cpu().numpy())

    arm_cfg, w_cfg = cfg.actions.right_arm, cfg.actions.waist_yaw
    acts = cfg.scene.robot.actuators
    jp = cfg.scene.robot.init_state.joint_pos
    consts = {
        "dt": cfg.sim.dt * cfg.decimation,
        "arm_joint_names": [robot.joint_names[i] for i in arm_t._joint_ids],
        "arm_default": robot.data.default_joint_pos[0, arm_t._joint_ids].cpu().tolist(),
        "waist_default": [float(robot.data.default_joint_pos[0, robot.joint_names.index(n)])
                          for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")],
        "arm_scale": arm_cfg.scale, "arm_vmax": arm_cfg.max_velocity, "arm_amax": arm_cfg.max_acceleration,
        "arm_brake": bool(arm_cfg.brake_to_target),
        "arm_clip": [list(arm_cfg.clip[n]) for n in G.RIGHT_ARM_JOINTS],
        "arm_kp": float(acts["right_arm"].stiffness), "arm_kd": float(acts["right_arm"].damping),
        "waist_scale": w_cfg.scale, "waist_vmax": w_cfg.max_velocity, "waist_amax": w_cfg.max_acceleration,
        "waist_brake": bool(w_cfg.brake_to_target), "waist_clip": list(w_cfg.clip[".*"]),
        "palm_local_offset": list(G.PALM_LOCAL_OFFSET), "palm_normal_axis": list(G.PALM_NORMAL_AXIS),
        "grasp_standoff": G.GRASP_STANDOFF, "palm_touch_height": G._PALM_TOUCH_HEIGHT,
        "apple_stiffness": G.APPLE_CONTACT_STIFFNESS,
        "touch_min": G.TOUCH_FORCE_MIN, "touch_max": G.TOUCH_FORCE_MAX,
        "apple_radius_range": list(TREE_APPLE_RADIUS_RANGE),
        "hand_preshape": dict(S.REACH_HAND_PRESHAPE),
        "envelope_waist": [W.REACH_WAIST_MIN, W.REACH_WAIST_MAX, W.REACH_WAIST_AZIMUTH_CENTER_DEG,
                           W.REACH_WAIST_AZIMUTH_SPAN_DEG, list(W.REACH_WAIST_DZ_RANGE),
                           W.REACH_WAIST_MIN_FORWARD, W.REACH_WAIST_MIN_HORIZONTAL],
        "envelope_arm": [G.REACH_MIN, G.REACH_MAX, G.REACH_AZIMUTH_CENTER_DEG, G.REACH_AZIMUTH_SPAN_DEG,
                         list(G.REACH_TARGET_DZ_RANGE), G.REACH_MIN_FORWARD, G.REACH_MIN_HORIZONTAL],
        "stem_heading_max_deg": G.BRANCH_YAW_SPREAD_DEG,
        "obs_terms": list(env.observation_manager.active_terms["policy"]),
        "obs_dims": [int(d[0]) if isinstance(d, tuple) else int(d)
                     for d in env.observation_manager.group_obs_term_dim["policy"]],
        "action_terms": list(env.action_manager.active_terms),
        "init_joint_pos": {k: float(v) for k, v in jp.items()},
    }
    os.makedirs(os.path.dirname(args.npz), exist_ok=True)
    np.savez_compressed(args.npz, **{k: np.stack(v) for k, v in rec.items()},
                        speed=speed[:, 0].cpu().numpy(), joint_names=np.array(robot.joint_names),
                        bodies=np.array(bodies), consts=json.dumps(consts), policy=pol_path)
    print(f"[collect] {args.steps} steps x {N} envs -> {args.npz}")
    env.close()


# ─────────────────────────────────────────────
#  2. check (no Isaac Lab)
# ─────────────────────────────────────────────
def check():
    import numpy as np
    sys.path.insert(0, HERE)
    import reach_io as R

    d = dict(np.load(args.npz, allow_pickle=False))          # in memory: npz members decompress on every access
    C = json.loads(str(d["consts"]))
    names = list(d["joint_names"])
    T, N, _ = d["joint_pos"].shape
    fails = []

    def report(what, err, tol, unit=""):
        ok = err <= tol
        print(f"  {'ok  ' if ok else 'FAIL'} {what:<44s} max err {err:.3g}{unit} (tol {tol:g}{unit})")
        if not ok:
            fails.append(what)

    # constants
    print("constants (reach_io vs the env cfg):")
    cmp = [
        ("dt", R.DT, C["dt"]), ("arm joint order", R.RIGHT_ARM_JOINTS, C["arm_joint_names"]),
        ("arm default", list(R.ARM_DEFAULT), C["arm_default"]),
        ("waist default (yaw, roll, pitch)", list(R.WAIST_DEFAULT), C["waist_default"]),
        ("arm scale/vmax/amax/brake", [R.ARM_SCALE, R.ARM_MAX_VELOCITY, R.ARM_MAX_ACCELERATION, False],
         [C["arm_scale"], C["arm_vmax"], C["arm_amax"], C["arm_brake"]]),
        ("arm target clip (v3)", [[a, b] for a, b in zip(R.ARM_TARGET_LO, R.ARM_TARGET_HI)], C["arm_clip"]),
        ("arm kp/kd", [R.ARM_KP, R.ARM_KD], [C["arm_kp"], C["arm_kd"]]),
        ("waist scale/vmax/amax/brake", [R.WAIST_SCALE, R.WAIST_MAX_VELOCITY, R.WAIST_MAX_ACCELERATION, True],
         [C["waist_scale"], C["waist_vmax"], C["waist_amax"], C["waist_brake"]]),
        ("waist clip", [-R.WAIST_YAW_LIMIT, R.WAIST_YAW_LIMIT], C["waist_clip"]),
        ("palm offset / normal", [list(R.PALM_LOCAL_OFFSET), list(R.PALM_NORMAL_AXIS)],
         [C["palm_local_offset"], C["palm_normal_axis"]]),
        ("standoff / touch height / stiffness", [R.GRASP_STANDOFF, R.PALM_TOUCH_HEIGHT, R.APPLE_CONTACT_STIFFNESS],
         [C["grasp_standoff"], C["palm_touch_height"], C["apple_stiffness"]]),
        ("touch band", [R.TOUCH_FORCE_MIN, R.TOUCH_FORCE_MAX], [C["touch_min"], C["touch_max"]]),
        ("apple radius range", list(R.APPLE_RADIUS_RANGE), C["apple_radius_range"]),
        ("hand preshape", R.REACH_HAND_PRESHAPE, C["hand_preshape"]),
        ("waist envelope", [R.ENVELOPE_WAIST[k] for k in ("reach_min", "reach_max", "az_center_deg", "az_span_deg")]
         + [list(R.ENVELOPE_WAIST["dz_range"]), R.ENVELOPE_WAIST["min_forward"], R.ENVELOPE_WAIST["min_horizontal"]],
         C["envelope_waist"]),
        ("arm envelope", [R.ENVELOPE_ARM[k] for k in ("reach_min", "reach_max", "az_center_deg", "az_span_deg")]
         + [list(R.ENVELOPE_ARM["dz_range"]), R.ENVELOPE_ARM["min_forward"], R.ENVELOPE_ARM["min_horizontal"]],
         C["envelope_arm"]),
        ("stem heading spread", R.STEM_HEADING_MAX_DEG, C["stem_heading_max_deg"]),
        ("action terms", ["right_arm", "waist_yaw"], C["action_terms"]),
    ]
    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple, np.ndarray)):
            return isinstance(b, (list, tuple, np.ndarray)) and len(a) == len(b) and all(map(same, a, b))
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return abs(float(a) - float(b)) <= 1e-6

    for what, mine, theirs in cmp:
        ok = same(mine, theirs)
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + ("" if ok else f": reach_io {mine} vs cfg {theirs}"))
        if not ok:
            fails.append(what)
    print(f"  obs terms: {list(zip(C['obs_terms'], C['obs_dims']))}")

    kin = R.G1Kinematics(args.urdf or R.URDF_DEFAULT)
    ji = {n: names.index(n) for n in R.RIGHT_ARM_JOINTS + R.WAIST_JOINTS}
    arm_i = [ji[n] for n in R.RIGHT_ARM_JOINTS]
    waist_i = [ji[n] for n in R.WAIST_JOINTS]
    hand_names = [n for n in names if n.startswith("right_hand")]

    # FK
    def qmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    e_pos = {b: 0.0 for b in ("palm", "torso", "shoulder", "camera")}
    off = {b: [] for b in e_pos}
    e_rot = {b: 0.0 for b in ("palm", "torso")}
    bodies = list(d["bodies"])
    bi = {k: bodies.index(v) for k, v in (("pelvis", "pelvis"), ("torso", "torso_link"),
                                           ("palm", "right_hand_palm_link"), ("shoulder", "right_shoulder_pitch_link"),
                                           ("camera", "d435_link"))}
    for t in range(0, T, 5):
        for e in range(N):
            jp = d["joint_pos"][t, e]
            fr = kin.frames(jp[waist_i], jp[arm_i], {n: jp[names.index(n)] for n in hand_names})
            Rp = qmat(d["body_quat"][t, e, bi["pelvis"]])
            p0 = d["body_pos"][t, e, bi["pelvis"]]
            rel = lambda b: Rp.T @ (d["body_pos"][t, e, bi[b]] - p0)
            relR = lambda b: Rp.T @ qmat(d["body_quat"][t, e, bi[b]])
            for b, mine in (("palm", fr.palm_pos), ("torso", fr.torso_pos), ("shoulder", fr.shoulder_pos),
                            ("camera", fr.camera_pos)):
                e_pos[b] = max(e_pos[b], float(np.linalg.norm(mine - rel(b))))
                off[b].append(mine - rel(b))
            for b, mine in (("palm", fr.palm_R), ("torso", fr.torso_R)):
                c = np.clip((np.trace(mine.T @ relR(b)) - 1) / 2, -1, 1)
                e_rot[b] = max(e_rot[b], float(np.degrees(np.arccos(c))))
    pelvis_tilt = max(float(np.degrees(np.arccos(np.clip(qmat(q)[2, 2], -1, 1))))
                      for q in d["body_quat"][:, :, bi["pelvis"]].reshape(-1, 4))
    print(f"forward kinematics, pinocchio + {os.path.basename(args.urdf or R.URDF_DEFAULT)} vs the USD "
          f"(pelvis frame; sim pelvis tilt max {pelvis_tilt:.2f} deg):")
    for b, v in e_pos.items():
        mean = np.mean(off[b], axis=0) * 1000
        if b == "torso":
            # torso_link's ORIGIN differs between the URDF and the USD; only its
            # orientation is a policy input, so the offset is reported, not judged
            print(f"  info torso_link origin offset (not a policy input): {np.round(mean, 1)} mm")
            continue
        report(f"{b} position (mean offset {np.round(mean, 2)} mm)", v * 1000, 2.0, " mm")
    for b, v in e_rot.items():
        report(f"{b} orientation", v, 0.2, " deg")

    # observation
    layout = "waist"
    sl = {"arm q": slice(0, 7), "arm dq": slice(7, 14), "apple - palm": slice(14, 17), "palm quat": slice(17, 21),
          "prev raw arm": slice(21, 28), "touch": slice(28, 29), "ring": slice(29, 37), "waist q": slice(37, 38),
          "waist dq": slice(38, 39), "prev raw waist": slice(39, 40)}
    err = {k: 0.0 for k in sl}
    flips = 0
    mine_all = np.zeros_like(d["obs"])
    for t in range(T):
        for e in range(N):
            jp, jv = d["joint_pos"][t, e], d["joint_vel"][t, e]
            o_sim = d["obs"][t, e]
            prev = d["raw"][t - 1, e] if (t > 0 and d["ep_len"][t, e] > 0) else np.zeros(8)
            # the apple in the pelvis frame (the sim's world axes, origin at the pelvis)
            Rp = qmat(d["body_quat"][t, e, bi["pelvis"]])
            apple_pel = Rp.T @ (d["apple"][t, e] - d["body_pos"][t, e, bi["pelvis"]])
            st = R.ReachState(arm_q=jp[arm_i], arm_dq=jv[arm_i], waist_q=jp[waist_i], waist_dq=jv[waist_i],
                              apple=apple_pel, touch_force=float(o_sim[28]), ring=o_sim[29:37])
            fr = kin.frames(jp[waist_i], jp[arm_i], {n: jp[names.index(n)] for n in hand_names})
            o, _ = R.build_obs(layout, kin, st, prev, fr)
            if np.dot(o[17:21], o_sim[17:21]) < 0:
                flips += 1
                o[17:21] *= -1
            mine_all[t, e] = o
            for k, s in sl.items():
                err[k] = max(err[k], float(np.abs(o[s] - o_sim[s]).max()))
    print("observation, rebuilt vs Isaac Lab's (touch + ring injected from the sim):")
    # apple - palm: 1.5 mm = the constant ~1 mm pelvis-origin offset between the URDF and the USD
    tol = {"arm q": 1e-5, "arm dq": 1e-4, "apple - palm": 1.5e-3, "palm quat": 5e-3, "prev raw arm": 1e-5,
           "touch": 1e-5, "ring": 1e-5, "waist q": 1e-5, "waist dq": 1e-4, "prev raw waist": 1e-5}
    for k in sl:
        report(k, err[k], tol[k])
    print(f"  palm quat sign: {flips} of {T * N} sim quaternions had w < 0 after the w >= 0 rebuild "
          f"({'the policy saw both signs' if flips else 'the sim keeps w >= 0, as reach_io'})")

    # limiter, ONE STEP at a time from Isaac's own previous target and velocity
    # (teacher forced): the logic, exactly. Free-running replays drift apart through
    # rounding (GPU fused multiply-add vs CPU) that the bang-bang limiter keeps.
    def one_step(dtype):
        ea = ew = 0.0
        for e in range(N):
            L = R.make_limiters(layout, dtype=dtype)
            sp = float(d["speed"][e])
            for t in range(T):
                jp = d["joint_pos"][t, e]
                k = int(d["ep_len"][t, e])
                for name, ids, tgt, raw in (("arm", arm_i, d["arm_target"], d["raw"][t, e, :7]),
                                            ("waist", waist_i[:1], d["waist_target"], d["raw"][t, e, 7:8])):
                    lim = L[name]
                    lim.k = min(k, R.RateLimiter.ANCHOR_STEPS)
                    if k >= R.RateLimiter.ANCHOR_STEPS:
                        # the command before tgt[t-1]: the measured joints if t-1 re-anchored
                        prev_prev = (d["joint_pos"][t - 1, e, ids] if d["ep_len"][t - 1, e] <= 1
                                     else tgt[t - 2, e])
                        lim.prev = tgt[t - 1, e].astype(dtype)
                        lim.vel = ((tgt[t - 1, e] - prev_prev) / R.DT).astype(dtype)
                    out = lim.step(raw, jp[ids], speed=sp)
                    err = float(np.abs(out - tgt[t, e]).max())
                    if name == "arm":
                        ea = max(ea, err)
                    else:
                        ew = max(ew, err)
        return ea, ew

    def free_run(dtype):
        ea = ew = 0.0
        for e in range(N):
            L = R.make_limiters(layout, dtype=dtype)
            sp = float(d["speed"][e])
            for t in range(T):
                if d["ep_len"][t, e] == 0:
                    L["arm"].reset()
                    L["waist"].reset()
                jp = d["joint_pos"][t, e]
                ta = L["arm"].step(d["raw"][t, e, :7], jp[arm_i], speed=sp)
                tw = L["waist"].step(d["raw"][t, e, 7:8], jp[waist_i[:1]], speed=sp)
                ea = max(ea, float(np.abs(ta - d["arm_target"][t, e]).max()))
                ew = max(ew, float(np.abs(tw - d["waist_target"][t, e]).max()))
        return ea, ew
    print(f"rate limiter on the policy's raw actions (speed factors {sorted(set(np.round(d['speed'], 3)))}):")
    for dt_ in (np.float64, np.float32):
        ea, ew = one_step(dt_)
        report(f"arm targets, one step ({np.dtype(dt_).name})", ea, 1e-4, " rad")
        report(f"waist target, one step ({np.dtype(dt_).name})", ew, 1e-4, " rad")
    ea, ew = free_run(np.float64)
    print(f"  info free-running float64 replay (what the robot runs) vs Isaac: arm {ea:.2g} rad, waist {ew:.2g} rad "
          "-- rounding differences the bang-bang limiter keeps (1e-6 noise on the raw actions moves it as much); "
          "closed loop, the policy corrects them")

    # policy
    pol = R.ReachPolicy(str(d["policy"]))
    e_pol = 0.0
    for t in range(0, T, 3):
        for e in range(N):
            e_pol = max(e_pol, float(np.abs(pol(mine_all[t, e]) - d["raw"][t, e]).max()))
    print("policy on the rebuilt observation vs the sim's actions:")
    report("raw actions", e_pol, 2e-2)

    print("\nALL CHECKS PASSED" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if args.collect:
    collect()
else:
    check()
