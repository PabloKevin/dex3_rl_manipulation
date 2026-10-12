#!/usr/bin/env python3
"""Find the reach's hand PRE-SHAPE: every finger a little closed around the
apple, but open enough for the biggest apple, off-centre by the reach's
error, to arrive on the palm without a finger touching it first.

The hand is posed kinematically (one batch env per candidate pose) and its
collision-sphere model (envs/mdp/self_collision.HAND_SPHERES, slightly
conservative) is checked against an apple of APPLE_R_MAX resting on the palm
at first touch (surface 16 mm out from the palm reference point,
apple_grasp_env_cfg._PALM_TOUCH_HEIGHT), centred and shifted OFFSET in each
direction across the palm. A pose passes when every finger / thumb sphere
keeps CLEARANCE from the apple in every case; among those, the one whose
finger and thumb tips reach furthest out of the palm plane (toward the
apple: enclosing it) wins.

Candidates: index+middle closed by a fraction f_fing of their range, thumb
flexion (thumb_1, thumb_2) by f_thumb, thumb_0 (opposition) at THUMB0 values.
Closing direction per joint = toward the joint limit farther from the open
default, as the bench's scripted grasp uses.

    ~/miniconda3/envs/env_isaaclab/bin/python scripts/probe_reach_hand_preshape.py --headless
"""
import argparse
import itertools
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--clearance", type=float, default=0.005, help="min finger-apple gap (m)")
parser.add_argument("--offset", type=float, default=0.02, help="apple off-centre across the palm (m)")
parser.add_argument("--apple_r", type=float, default=0.048, help="largest apple radius (m)")
parser.add_argument("--shifts", type=str, default="x-,x+,z-,z+",
                    help="off-centre directions checked: x- toward the wrist (the thumb sits there, at its "
                         "open limit: no pose helps), x+ toward the fingertips, z-/z+ across the palm")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_conjugate  # noqa: E402

import envs  # noqa: E402,F401
from envs.apple_grasp_env_cfg import AppleGraspEnvCfg, PALM_LOCAL_OFFSET, PALM_NORMAL_AXIS, _PALM_TOUCH_HEIGHT  # noqa: E402
from envs.mdp.self_collision import HAND_SPHERES  # noqa: E402

F_FING = np.round(np.arange(0.0, 0.41, 0.025), 3)
F_THUMB = np.round(np.arange(0.0, 0.31, 0.025), 3)
THUMB0 = tuple(float(v) for v in os.environ.get("THUMB0", "0.0").split(","))
FINGERS = ["right_hand_index_0_joint", "right_hand_index_1_joint",
           "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
THUMB = ["right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]
TIPS = ["right_hand_index_1_link", "right_hand_middle_1_link", "right_hand_thumb_2_link"]


def main():
    cands = list(itertools.product(F_FING, F_THUMB, THUMB0))
    cfg = AppleGraspEnvCfg()
    cfg.scene.num_envs = len(cands)
    cfg.episode_length_s = 1.0e6
    cfg.terminations.reach_success = None
    cfg.events.reset_arm_jitter = None
    env = ManagerBasedRLEnv(cfg)
    robot = env.scene["robot"]
    names = robot.joint_names
    dev = env.device
    env.reset()

    lim = robot.data.soft_joint_pos_limits[0]
    default = robot.data.default_joint_pos[0]

    def far(j):
        i = names.index(j)
        lo, hi = float(lim[i, 0]), float(lim[i, 1])
        return hi if abs(hi - float(default[i])) > abs(lo - float(default[i])) else lo

    q = robot.data.default_joint_pos.clone()
    for e, (ff, ft, t0) in enumerate(cands):
        for j in FINGERS:
            i = names.index(j)
            q[e, i] = default[i] + ff * (far(j) - default[i])
        for j in THUMB:
            i = names.index(j)
            q[e, i] = default[i] + ft * (far(j) - default[i])
        q[e, names.index("right_hand_thumb_0_joint")] = t0
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)

    bn = robot.data.body_names
    palm = bn.index("right_hand_palm_link")
    qp = robot.data.body_quat_w[:, palm]
    pp = robot.data.body_pos_w[:, palm]

    def to_palm(p_w):                                     # world -> palm frame
        return quat_apply(quat_conjugate(qp), p_w - pp)

    # finger / thumb spheres in the palm frame
    sph, rad = [], []
    for link, c, r in HAND_SPHERES:
        if link == "right_hand_palm_link":
            continue
        b = bn.index(link)
        c_w = robot.data.body_pos_w[:, b] + quat_apply(robot.data.body_quat_w[:, b],
                                                       torch.tensor(c, device=dev).expand(len(cands), 3))
        sph.append(to_palm(c_w)); rad.append(r)
    sph = torch.stack(sph, 1)                             # [C, S, 3]
    rad = torch.tensor(rad, device=dev)

    n = torch.tensor(PALM_NORMAL_AXIS, device=dev)
    ref = torch.tensor(PALM_LOCAL_OFFSET, device=dev)
    R = args_cli.apple_r
    base = ref + n * (_PALM_TOUCH_HEIGHT + R)              # apple centre at first palm touch
    shifts = [torch.zeros(3, device=dev)]
    for ax, sgn, tag in ((0, -1.0, "x-"), (0, 1.0, "x+"), (2, -1.0, "z-"), (2, 1.0, "z+")):
        if tag in args_cli.shifts.split(","):             # palm-plane axes: x (toward the fingers), z (across)
            v = torch.zeros(3, device=dev); v[ax] = sgn * args_cli.offset
            shifts.append(v)
    clear = torch.full((len(cands),), 1.0, device=dev)
    groups = {"index": [], "middle": [], "thumb": []}
    for k, (link, _, _) in enumerate([h for h in HAND_SPHERES if h[0] != "right_hand_palm_link"]):
        groups["index" if "index" in link else "middle" if "middle" in link else "thumb"].append(k)
    detail = {}
    for si, sft in enumerate(shifts):
        c = base + sft
        d = (sph - c).norm(dim=-1) - rad - R              # [C, S]
        clear = torch.minimum(clear, d.amin(-1))
        for g, ks in groups.items():
            detail[(si, g)] = d[:, ks].amin(-1).cpu().numpy()
    # enclosure: how far the fingertips stand out of the palm plane, toward the apple
    tip_h = torch.stack([(to_palm(robot.data.body_pos_w[:, bn.index(t)]) * n).sum(-1) for t in TIPS], 1)   # [C, 3]
    # (tip link ORIGINS; the pads reach ~2-3 cm further)

    clear, tip_h = clear.cpu().numpy(), tip_h.cpu().numpy()
    print(f"\napple r {R*1000:.0f} mm, off-centre {args_cli.offset*100:.0f} cm, required clearance {args_cli.clearance*1000:.0f} mm")
    print("closing direction / full range (rad): " + ", ".join(
        f"{j.replace('right_hand_', '').replace('_joint', '')} {far(j) - float(default[names.index(j)]):+.2f}"
        for j in FINGERS + THUMB))
    print("\nmin clearance (mm) -- rows: index+middle closed fraction, cols: thumb closed fraction")
    print("        " + "".join(f"{ft:6.2f}" for ft in F_THUMB))
    for ff in F_FING:
        row = [clear[cands.index((ff, ft, 0.0))] * 1000 for ft in F_THUMB]
        print(f"  {ff:4.2f}  " + "".join(f"{v:6.1f}" for v in row))
    if os.environ.get("DETAIL"):
        labels = ["centre"] + args_cli.shifts.split(",")
        for probe in [(0.0, 0.0), (0.2, 0.0), (0.3, 0.2), (0.4, 0.3)]:
            if (probe[0], probe[1], THUMB0[0]) not in cands:
                continue
            for t0 in THUMB0:
                e = cands.index((probe[0], probe[1], t0))
                print(f"  fingers {probe[0]:.2f} thumb {probe[1]:.2f} thumb_0 {t0:+.2f}: " + " | ".join(
                    f"{labels[si]}: " + " ".join(f"{g[0]}{detail[(si, g)][e]*1000:+.0f}" for g in groups)
                    for si in range(len(shifts))) + f"   tips {np.round(tip_h[e]*1000, 0)}")
        for j in ["right_hand_thumb_0_joint"] + THUMB:
            i = names.index(j)
            print(f"  {j}: limits {float(lim[i,0]):+.2f}..{float(lim[i,1]):+.2f}, default {float(default[i]):+.2f}")
    ok = clear >= args_cli.clearance
    score = np.where(ok, tip_h.mean(-1), -np.inf)
    best = int(np.argmax(score))
    ff, ft, t0 = cands[best]
    print(f"\nchosen: fingers {ff:.2f}, thumb {ft:.2f} of their range (thumb_0 {t0}); clearance {clear[best]*1000:.1f} mm; "
          f"tip heights above the palm plane (index, middle, thumb) {np.round(tip_h[best]*1000, 0)} mm "
          f"(open hand: {np.round(tip_h[cands.index((0.0, 0.0, 0.0))]*1000, 0)} mm)")
    print("REACH_HAND_PRESHAPE = {")
    for j in ["right_hand_thumb_0_joint"] + THUMB + FINGERS:
        print(f'    "{j}": {float(q[best, names.index(j)]):.3f},')
    print("}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
