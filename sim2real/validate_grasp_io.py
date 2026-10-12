#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Check the grasp half of sim2real/reach_io.py (what the robot runs) against Isaac
Lab (what the grasp policy was trained in). Same two steps as validate_reach_io.py:

1. COLLECT, in Isaac Lab (env_isaaclab): AppleGraspSpindle-v2 with its own reach
   hand-off and the grasp policy; every step's joints, palm / pelvis / fruit poses,
   the 44-dim observation, the 14 raw actions, the processed targets, the phase
   flags, the grip command and the nine sensed taxel forces:

    ~/miniconda3/envs/env_isaaclab/bin/python sim2real/validate_grasp_io.py --collect --headless

2. CHECK, without Isaac Lab (g1_dex3_sim2real):

    ~/miniconda3/envs/g1_dex3_sim2real/bin/python sim2real/validate_grasp_io.py --check
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_NPZ = os.path.join(REPO, "logs", "sim2real", "grasp_io_reference.npz")

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--collect", action="store_true")
p.add_argument("--check", action="store_true")
p.add_argument("--npz", default=DEFAULT_NPZ)
p.add_argument("--policy", default=None, help="exported grasp policy (default: the pinned v2-1499)")
p.add_argument("--task", default="AppleGraspSpindle-v2")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=450)
p.add_argument("--urdf", default=None)
args, _rest = p.parse_known_args()
if args.collect == args.check:
    p.error("pick one of --collect / --check")


def collect():
    from isaaclab.app import AppLauncher
    ap = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(ap)
    app = AppLauncher(ap.parse_args(_rest)).app  # noqa: F841

    import numpy as np
    import torch
    import gymnasium as gym
    from isaaclab.managers import SceneEntityCfg

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import envs  # noqa: F401
    from envs import apple_grasp_grasp_env_cfg as GG
    from envs.apple_grasp_spindle_env_cfg import AppleGraspSpindleV2EnvCfg
    from envs.mdp import observations as O

    cfg = AppleGraspSpindleV2EnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=cfg).unwrapped
    dev = env.device
    pol_path = args.policy or os.path.join(REPO, "envs", "agents", "best_weights", "grasp_spindle_v2_1499.pt")
    pol = torch.jit.load(pol_path, map_location=dev).eval()
    robot, apple = env.scene["robot"], env.scene["apple"]
    arm_t = env.action_manager.get_term("right_arm")
    hand_t = env.action_manager.get_term("right_hand")
    names = robot.joint_names
    hand_i = [names.index(n) for n in GG.RIGHT_HAND_JOINTS]
    palm_b, pelvis_b = robot.body_names.index("right_hand_palm_link"), robot.body_names.index("pelvis")
    sc = SceneEntityCfg("contact_forces")

    def sensed():
        return O._sensed_tactile_forces(env, sc, all_apples=True).cpu().numpy()

    rec = {k: [] for k in ("joint_pos", "joint_vel", "apple", "palm_pos", "palm_quat", "pelvis_pos", "pelvis_quat",
                           "obs", "raw", "hand_target", "arm_target", "phase", "released", "grip", "sensed",
                           "ep_len")}

    def snap(obs):
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy())
        rec["apple"].append((apple.data.root_pos_w - env.scene.env_origins).cpu().numpy())
        rec["palm_pos"].append((robot.data.body_pos_w[:, palm_b] - env.scene.env_origins).cpu().numpy())
        rec["palm_quat"].append(robot.data.body_quat_w[:, palm_b].cpu().numpy())
        rec["pelvis_pos"].append((robot.data.body_pos_w[:, pelvis_b] - env.scene.env_origins).cpu().numpy())
        rec["pelvis_quat"].append(robot.data.body_quat_w[:, pelvis_b].cpu().numpy())
        rec["obs"].append(obs.cpu().numpy())
        rec["phase"].append(env._grasp_phase_active.cpu().numpy().astype(float))
        rec["released"].append(env._stem_released.cpu().numpy().astype(float))
        rec["grip"].append(env._target_grip_level.cpu().numpy())
        rec["sensed"].append(sensed())
        rec["ep_len"].append(env.episode_length_buf.cpu().numpy().copy())

    obs = env.reset()[0]["policy"]
    for _ in range(args.steps):
        snap(obs)
        with torch.no_grad():
            a = pol(obs)
        rec["raw"].append(a.cpu().numpy())
        obs = env.step(a)[0]["policy"]
        rec["hand_target"].append(hand_t.processed_actions.cpu().numpy())
        rec["arm_target"].append(arm_t.processed_actions.cpu().numpy())

    hc, ac = cfg.actions.right_hand, cfg.actions.right_arm
    consts = {
        "dt": cfg.sim.dt * cfg.decimation,
        "hand_art_names": list(hand_t._joint_names), "arm_art_names": list(arm_t._joint_names),
        "obs_hand_names": list(GG.RIGHT_HAND_JOINTS),
        "hand_default": {n: float(robot.data.default_joint_pos[0, names.index(n)]) for n in GG.RIGHT_HAND_JOINTS},
        "hand_limits": {n: robot.data.soft_joint_pos_limits[0, names.index(n)].cpu().tolist() for n in GG.RIGHT_HAND_JOINTS},
        "grasp_offset": dict(hc.grasp_offset), "hand_scale": hc.scale, "hand_vmax": hc.max_velocity,
        "hand_amax": hc.max_acceleration, "hand_clip": list(hc.clip[".*"]),
        "arm_scale": ac.scale, "arm_vmax": ac.max_velocity, "arm_amax": ac.max_acceleration,
        "pre_grip_free_dims": ac.pre_grip_free_dims, "unlock_with_grip": bool(ac.unlock_with_grip),
        "grasp_center_local": list(GG.GRASP_CENTER_LOCAL),
        "obs_terms": list(env.observation_manager.active_terms["policy"]),
        "obs_dims": [int(d[0]) if isinstance(d, tuple) else int(d)
                     for d in env.observation_manager.group_obs_term_dim["policy"]],
        "action_terms": list(env.action_manager.active_terms),
        "damage_force_range": [20.0, 40.0],
    }
    os.makedirs(os.path.dirname(args.npz), exist_ok=True)
    np.savez_compressed(args.npz, **{k: np.stack(v) for k, v in rec.items()}, joint_names=np.array(names),
                        consts=json.dumps(consts), policy=pol_path)
    print(f"[collect] {args.steps} steps x {env.num_envs} envs -> {args.npz}")
    env.close()


def check():
    import numpy as np
    sys.path.insert(0, HERE)
    import reach_io as R

    d = dict(np.load(args.npz, allow_pickle=False))
    C = json.loads(str(d["consts"]))
    names = list(d["joint_names"])
    T, N, _ = d["joint_pos"].shape
    fails = []

    def ok_(what, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f": {detail}" if detail else ""))
        if not ok:
            fails.append(what)

    def err_(what, err, tol):
        ok_(what, err <= tol, f"max err {err:.3g} (tol {tol:g})")

    print("constants (reach_io vs the env cfg):")
    ok_("hand action order (articulation)", ["right_hand_%s_joint" % n for n in R.HAND_ART_ORDER] == C["hand_art_names"],
        f"{C['hand_art_names']}")
    ok_("hand observation order", ["right_hand_%s_joint" % n for n in R.DEX3_ORDER] == C["obs_hand_names"])
    ok_("hand default", all(abs(R.HAND_DEFAULT[n] - C["hand_default"]["right_hand_%s_joint" % n]) < 1e-6
                            for n in R.DEX3_ORDER))
    ok_("grasp offset", all(abs(R.GRASP_HAND_OFFSET[n] - C["grasp_offset"].get("right_hand_%s_joint" % n,
                                                                                C["hand_default"]["right_hand_%s_joint" % n])) < 1e-6
                            for n in R.HAND_ART_ORDER))
    ok_("hand scale/vmax/amax", [R.GRASP_HAND_SCALE, R.GRASP_HAND_VMAX, R.GRASP_HAND_AMAX]
        == [C["hand_scale"], C["hand_vmax"], C["hand_amax"]])
    ok_("arm scale/vmax/amax", [R.GRASP_ARM_SCALE, R.GRASP_ARM_VMAX, R.GRASP_ARM_AMAX]
        == [C["arm_scale"], C["arm_vmax"], C["arm_amax"]])
    ok_("hand joint limits (URDF vs sim soft limits)", all(
        np.allclose(R.HAND_LIMITS[n], C["hand_limits"]["right_hand_%s_joint" % n], atol=1e-3) for n in R.DEX3_ORDER),
        str({n: np.round(C["hand_limits"]["right_hand_%s_joint" % n], 3).tolist() for n in R.DEX3_ORDER}))
    ok_("grasp pocket", np.allclose(R.GRASP_CENTER_LOCAL, C["grasp_center_local"]))
    ok_("pre-grip free dims / unlock", (C["pre_grip_free_dims"], C["unlock_with_grip"]) == (3, True))
    print(f"  obs terms: {list(zip(C['obs_terms'], C['obs_dims']))}   action terms {C['action_terms']}")

    def qmat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    kin = R.G1Kinematics(args.urdf or R.URDF_DEFAULT)
    ji = {n: names.index(n) for n in R.RIGHT_ARM_JOINTS + R.WAIST_JOINTS}
    arm_i = [ji[n] for n in R.RIGHT_ARM_JOINTS]
    waist_i = [ji[n] for n in R.WAIST_JOINTS]
    hand_i = [names.index("right_hand_%s_joint" % n) for n in R.DEX3_ORDER]
    obs = d["obs"]
    raw = d["raw"]
    e = {k: 0.0 for k in ("hand_q", "hand_dq", "pocket_usd", "pocket_fk", "gravity_usd", "gravity_fk", "last", "phase",
                          "grip", "slip", "balance", "level")}
    tf = [R.TactileFeatures() for _ in range(N)]
    for t in range(T):
        for k in range(N):
            jp, jv = d["joint_pos"][t, k], d["joint_vel"][t, k]
            if d["ep_len"][t, k] <= 1:                     # new episode: the sim's slip starts from zero
                tf[k] = R.TactileFeatures()
            Rp = qmat(d["pelvis_quat"][t, k])
            Rpalm = qmat(d["palm_quat"][t, k])
            # joints (obs order)
            e["hand_q"] = max(e["hand_q"], float(np.max(np.abs(
                np.asarray(jp[hand_i]) - np.array([R.HAND_DEFAULT[n] for n in R.DEX3_ORDER]) - obs[t, k, 0:7]))))
            e["hand_dq"] = max(e["hand_dq"], float(np.max(np.abs(jv[hand_i] - obs[t, k, 7:14]))))
            # pocket: sim bodies, then pinocchio FK of the same joints
            loc = Rpalm.T @ (d["apple"][t, k] - d["palm_pos"][t, k]) - R.GRASP_CENTER_LOCAL
            e["pocket_usd"] = max(e["pocket_usd"], float(np.max(np.abs(loc - obs[t, k, 31:34]))))
            fr = kin.frames(jp[waist_i], jp[arm_i], {"right_hand_%s_joint" % n: jp[names.index("right_hand_%s_joint" % n)]
                                                    for n in R.DEX3_ORDER})
            apple_pel = Rp.T @ (d["apple"][t, k] - d["pelvis_pos"][t, k])
            e["pocket_fk"] = max(e["pocket_fk"], float(np.max(np.abs(
                R.apple_in_palm_frame(fr, apple_pel) - R.GRASP_CENTER_LOCAL - obs[t, k, 31:34]))))
            g_usd = Rpalm.T @ np.array([0.0, 0.0, -1.0])
            e["gravity_usd"] = max(e["gravity_usd"], float(np.max(np.abs(g_usd - obs[t, k, 41:44]))))
            e["gravity_fk"] = max(e["gravity_fk"], float(np.max(np.abs(R.gravity_in_palm(fr) - obs[t, k, 41:44]))))
            prev = raw[t - 1, k, 7:14] if t > 0 and d["ep_len"][t, k] > 0 else np.zeros(7)
            e["last"] = max(e["last"], float(np.max(np.abs(prev - obs[t, k, 34:41]))))
            e["phase"] = max(e["phase"], abs(float(d["phase"][t, k]) - float(obs[t, k, 30])))
            e["grip"] = max(e["grip"], abs(float(d["grip"][t, k]) - float(obs[t, k, 29])))
            slip, bal, lvl = tf[k].step(d["sensed"][t, k])
            e["slip"] = max(e["slip"], float(np.max(np.abs(slip - obs[t, k, 14:23]))))
            e["balance"] = max(e["balance"], float(np.max(np.abs(bal - obs[t, k, 23:26]))))
            e["level"] = max(e["level"], float(np.max(np.abs(lvl - obs[t, k, 26:29]))))
    print("observation slices (rebuilt vs the sim's 44-dim vector):")
    err_("hand q - default", e["hand_q"], 1e-5)
    err_("hand dq", e["hand_dq"], 1e-4)
    err_("fruit in palm frame, sim bodies", e["pocket_usd"], 1e-4)
    err_("fruit in palm frame, pinocchio FK", e["pocket_fk"], 3e-3)
    err_("gravity in palm, sim bodies", e["gravity_usd"], 1e-4)
    err_("gravity in palm, pinocchio FK", e["gravity_fk"], 0.02)
    err_("last hand action", e["last"], 1e-5)
    err_("phase flag", e["phase"], 1e-6)
    err_("grip command", e["grip"], 1e-5)
    err_("tactile slip (from the sensed taxels)", e["slip"], 1e-2)
    err_("tactile balance", e["balance"], 1e-4)
    err_("tactile level", e["level"], 1e-4)

    # the hand limiter, one step at a time from the sim's own previous target
    print("hand limiter (one step from the sim's previous target):")
    worst = 0.0
    for k in range(N):
        for t in range(2, T):
            if not d["phase"][t, k] or not d["phase"][t - 1, k] or d["ep_len"][t, k] <= 2:
                continue
            L = R.make_grasp_limiters()["hand"]
            L.k = L.ANCHOR_STEPS
            prev_t = np.asarray(d["hand_target"][t - 1, k], float)
            L.prev = prev_t.copy()
            pv = np.asarray(d["hand_target"][t - 1, k], float) - np.asarray(d["hand_target"][t - 2, k], float)
            L.vel = pv / R.DT
            out = L.step(raw[t, k, 7:14], None, 1.0)
            worst = max(worst, float(np.max(np.abs(out - d["hand_target"][t, k]))))
    err_("hand targets", worst, 2e-3)
    print("\n== grasp interface check:", "ALL OK" if not fails else f"{len(fails)} FAILED: {fails}")
    sys.exit(1 if fails else 0)


collect() if args.collect else check()
