#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
START LIBRARY for the final-approach task (AppleReachFinal-v1, part 2 of the
two-stage reach -- envs/mdp/pregrasp.py): hand poses in front of an apple,
where part 1 (AppleReachPregrasp-v1) leaves the hand.

From the v3 reach's RECORDED end poses (envs/assets/reach_end_poses_waist.npz,
scripts/record_reach_starts.py --spindle: palm at the apple, arm + waist), each
pose is backed off along the palm's approach by --back cm and its palm
orientation turned by up to --rot_deg, solved with damped least squares on
the 7 arm joints + the waist yaw (pinocchio, the g1_fruits_picker URDF --
the same model sim2real/reach_io.py uses; checked here against the recorded
palm poses first). Stored per pose: arm q, waist yaw, palm reference point
and normal (env-local, = pelvis frame + ROBOT_ROOT_HEIGHT in z), and the
keep-out sphere centres of the hand (in pregrasp.FINAL_HAND_PRESHAPE) and
forearm, so the tree can be built without fruit inside the hand.

No Isaac Sim needed (conda env g1_dex3_sim2real, or any python with numpy +
pinocchio):
  python3 scripts/build_pregrasp_starts.py                 # -> envs/assets/pregrasp_starts.npz
Rebuild after changing pregrasp.FINAL_FINGER_CURL (the task checks it).
"""
import argparse
import importlib.util
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim2real"))
import reach_io as R  # noqa: E402

# envs/mdp/pregrasp.py alone (the envs package imports Isaac Lab)
_spec = importlib.util.spec_from_file_location("pregrasp", os.path.join(REPO, "envs", "mdp", "pregrasp.py"))
PG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PG)

ROBOT_ROOT_HEIGHT = 0.793          # apple_grasp_env_cfg.ROBOT_ROOT_HEIGHT: env-local = pelvis + this in z
WAIST_LIMIT = math.radians(45.0)   # apple_reach_tree_waist_env_cfg.WAIST_YAW_LIMIT

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--src", default=os.path.join(REPO, "envs", "assets", "reach_end_poses_waist.npz"))
p.add_argument("--out", default=PG.START_LIBRARY)
p.add_argument("--per_pose", type=int, default=3, help="back-off variants per recorded pose")
p.add_argument("--back", type=float, nargs=2, default=(0.03, 0.08),
               help="m: palm moved back along its approach from the recorded (touching) pose")
p.add_argument("--rot_deg", type=float, default=10.0, help="palm orientation perturbation, uniform up to this")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--urdf", default=R.URDF_DEFAULT)
args = p.parse_args()
rng = np.random.default_rng(args.seed)

kin = R.G1Kinematics(args.urdf)
pin, model, data = kin.pin, kin.model, kin.data
palm_fid = kin.fid["palm"]
# "forearm_mid": halfway between the elbow and the wrist roll link origins
_links = [link for link, _, _ in PG.KEEP_OUT_SPHERES]
keep_fids = [model.getFrameId(link) if link != "forearm_mid" else None for link in _links]
for link, f in zip(_links, keep_fids):
    if f is not None and f >= model.nframes:
        raise KeyError(f"{link} not in {args.urdf}")
mid_ends = (model.getFrameId("right_elbow_link"), model.getFrameId("right_wrist_roll_link"))
keep_off = [None if o is None else np.asarray(o, float) for _, o, _ in PG.KEEP_OUT_SPHERES]
keep_rad = np.array([r for _, _, r in PG.KEEP_OUT_SPHERES], float)
offset = np.asarray(R.PALM_LOCAL_OFFSET, float)
n_axis = np.asarray(R.PALM_NORMAL_AXIS, float)
root = np.array([0.0, 0.0, ROBOT_ROOT_HEIGHT])
hand = PG.FINAL_HAND_PRESHAPE
lo = np.r_[R.ARM_TARGET_LO, -WAIST_LIMIT] * 0.98
hi = np.r_[R.ARM_TARGET_HI, WAIST_LIMIT] * 0.98
v_ids = kin.arm_v + [kin.waist_v[0]]


def config(x):
    """x = 7 arm + waist yaw -> full pinocchio q (waist roll 0, pitch the sim's 0.1, hand cupped)."""
    return kin.set_q([x[7], R.WAIST_DEFAULT[1], R.WAIST_DEFAULT[2]], x[:7], hand)


def palm(x):
    q = config(x)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    X = data.oMf[palm_fid]
    Rm = np.array(X.rotation)
    return q, np.array(X.translation) + Rm @ offset, Rm


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def ik(x0, p_goal, R_goal, iters=80, lam=0.03):
    """damped least squares on palm reference point + palm orientation."""
    x = x0.copy()
    for _ in range(iters):
        q, p_now, R_now = palm(x)
        e_p = p_goal - p_now
        e_r = np.asarray(pin.log3(R_goal @ R_now.T))
        if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < math.radians(0.5):
            return x, True
        J = pin.computeFrameJacobian(model, data, q, palm_fid, pin.LOCAL_WORLD_ALIGNED)[:, v_ids]
        Jp = J[:3] - skew(R_now @ offset) @ J[3:]
        Jf = np.vstack([Jp, J[3:]])
        e = np.r_[e_p, 0.3 * e_r]
        W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])
        dx = Jf.T @ np.linalg.solve(W @ Jf @ Jf.T + lam ** 2 * np.eye(6), e)
        x = np.clip(x + np.clip(dx, -0.2, 0.2), lo, hi)
    _, p_now, R_now = palm(x)
    ok = np.linalg.norm(p_goal - p_now) < 3e-3 and np.linalg.norm(pin.log3(R_goal @ R_now.T)) < math.radians(2.0)
    return x, ok


def random_rotation(max_deg):
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    return np.asarray(pin.exp3(a * math.radians(rng.uniform(0.0, max_deg))))


src = np.load(args.src)
names = [str(n) for n in src["joint_names"]]
if names != PG.ARM_JOINTS:
    raise SystemExit(f"{args.src}: joint order {names}")
Q, W = src["q"].astype(float), src["waist"].astype(float)

# the model vs the sim: the recorded palm link origin (env-local) from the recorded joints
err = []
for i in range(0, len(Q), max(1, len(Q) // 200)):
    fr = kin.frames([W[i], R.WAIST_DEFAULT[1], R.WAIST_DEFAULT[2]], Q[i])
    err.append(np.linalg.norm(fr.palm_pos + root - src["palm_local"][i]))
print(f"model check: palm origin vs recorded, max {max(err) * 1000:.1f} mm, mean {np.mean(err) * 1000:.1f} mm "
      f"({len(err)} poses)")
if max(err) > 0.005:
    raise SystemExit("the URDF does not match the sim's recorded poses (> 5 mm): wrong --urdf?")

out = {k: [] for k in ("q", "waist", "palm_ref", "palm_normal", "keep_pts", "back")}
fail = 0
for i in range(len(Q)):
    x0 = np.clip(np.r_[Q[i], W[i]], lo, hi)
    _, p0, R0 = palm(x0)
    n0 = R0 @ n_axis
    a = np.array([n0[0], n0[1], 0.0])
    a /= max(np.linalg.norm(a), 1e-6)
    for _ in range(args.per_pose):
        back = rng.uniform(*args.back)
        x, ok = ik(x0, p0 - a * back, random_rotation(args.rot_deg) @ R0)
        if not ok:
            fail += 1
            continue
        q, p_ref, Rm = palm(x)
        pts = np.array([0.5 * (np.array(data.oMf[mid_ends[0]].translation) + np.array(data.oMf[mid_ends[1]].translation))
                        if f is None else np.array(data.oMf[f].translation) + np.array(data.oMf[f].rotation) @ o
                        for f, o in zip(keep_fids, keep_off)])
        out["q"].append(x[:7]); out["waist"].append(x[7])
        out["palm_ref"].append(p_ref + root); out["palm_normal"].append(Rm @ n_axis)
        out["keep_pts"].append(pts + root); out["back"].append(back)

M = len(out["q"])
print(f"{M} starts from {len(Q)} recorded poses x {args.per_pose} ({fail} IK failures)")
np.savez(args.out,
         q=np.array(out["q"], np.float32), waist=np.array(out["waist"], np.float32),
         palm_ref=np.array(out["palm_ref"], np.float32), palm_normal=np.array(out["palm_normal"], np.float32),
         keep_pts=np.array(out["keep_pts"], np.float32), keep_rad=keep_rad.astype(np.float32),
         back=np.array(out["back"], np.float32), joint_names=np.array(PG.ARM_JOINTS),
         finger_curl=np.float32(PG.FINAL_FINGER_CURL))
print(f"wrote {args.out}")
