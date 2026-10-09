"""Design the nominal deep-squat pose offline (numpy FK, no Isaac Sim).

Searches symmetric flat-foot squats (hip pitch, knee, ankle pitch, waist pitch;
hip roll/yaw and ankle roll at 0, arms at the stand pose) for which

  * both soles are flat on the ground (pelvis pitch follows from the leg angles),
  * the whole-body CoM projects onto the centre of the support strip
    (sole spheres x = -0.05 .. 0.12 m in the ankle_roll frame),
  * joints keep a margin to their URDF limits,
  * a straight arm from the shoulder reaches a point on the ground in front of the toes.

Among the feasible ones it prefers the SHALLOWEST squat (least knee/ankle torque,
largest stability margin) that still reaches. Prints the chosen pose as a dict to
paste into squat/robot_cfg.py (SQUAT_JOINT_POS) and its pelvis height.

Run: ~/miniconda3/envs/env_isaaclab/bin/python squat/tools/design_squat_pose.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_kinematics import UrdfKinematics, rot_y, _tf  # noqa: E402

URDF = os.path.expanduser("~/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf")

# Unitree velocity-policy stand pose (deploy default_joint_pos) — the hand-over pose.
STAND = {
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97, "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15, "right_wrist_roll_joint": -0.15,
}

SOLE_Z = -0.035  # sole spheres: z = -0.03, r = 0.005 in the ankle_roll frame
SOLE_X = (-0.05, 0.12)
FOOT_CENTRE_X = 0.5 * (SOLE_X[0] + SOLE_X[1])


def place(kin: UrdfKinematics, q: dict) -> dict:
    """Pose the floating base so the left sole is flat on z = 0 (legs are symmetric)."""
    tfs = kin.fk(q)
    foot = tfs["left_ankle_roll_link"]
    pitch = np.arctan2(foot[0, 2], foot[2, 2])  # foot z-axis tilt about y
    root = _tf(rot_y(-pitch), np.zeros(3))
    tfs = kin.fk(q, root)
    sole_z = tfs["left_ankle_roll_link"][:3, :3] @ np.array([0, 0, SOLE_Z]) + tfs["left_ankle_roll_link"][:3, 3]
    root[2, 3] = -sole_z[2]
    return kin.fk(q, root)


def com_offset(kin, tfs) -> float:
    """CoM x minus support-strip centre x (world frame, + = in front)."""
    foot = tfs["left_ankle_roll_link"]
    centre = foot[:3, :3] @ np.array([FOOT_CENTRE_X, 0, SOLE_Z]) + foot[:3, 3]
    return kin.com(tfs)[0] - centre[0]


def leg_pose(hip, knee, ankle, waist) -> dict:
    q = dict(STAND)
    for side in ("left", "right"):
        q[f"{side}_hip_pitch_joint"] = hip
        q[f"{side}_knee_joint"] = knee
        q[f"{side}_ankle_pitch_joint"] = ankle
    q["waist_pitch_joint"] = waist
    return q


def arm_length(kin, elbow: float = 1.4) -> float:
    """Shoulder-pitch origin to the grasp point (palm +8 cm along the fingers), arm extended.

    G1 elbow convention: 0 = forearm horizontal (90 deg bend), ~+1.45 = straight arm.
    """
    tfs = kin.fk({"left_elbow_joint": elbow})
    sh = tfs["left_shoulder_pitch_link"][:3, 3]
    palm = tfs["left_hand_palm_link"]
    grasp = palm[:3, :3] @ np.array([0.08, 0, -0.02]) + palm[:3, 3]  # palm +x = fingers
    return float(np.linalg.norm(grasp - sh))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reach_ahead", type=float, default=0.10, help="ground target, m in front of the toes")
    parser.add_argument("--object_height", type=float, default=0.06, help="grasp height above the ground, m")
    parser.add_argument("--reach_frac", type=float, default=0.92, help="usable fraction of the straight-arm length")
    args = parser.parse_args()

    kin = UrdfKinematics(URDF)
    lim = kin.limits
    print(f"total mass {kin.total_mass:.2f} kg")

    tfs = place(kin, STAND)
    print(f"stand: pelvis z {tfs['pelvis'][2, 3]:.3f} m, CoM offset {com_offset(kin, tfs) * 100:+.1f} cm")
    L = arm_length(kin)
    print(f"extended arm length (shoulder -> grasp point) {L:.3f} m")

    margin = 0.08
    results = []
    for knee in np.arange(1.0, min(2.6, lim["left_knee_joint"][1] - margin), 0.1):
        for ankle in np.arange(lim["left_ankle_pitch_joint"][0] + margin, -0.2, 0.05):
            for waist in np.arange(0.0, lim["waist_pitch_joint"][1] - 0.1, 0.1):
                # bisection on hip pitch: CoM offset grows as the hip flexes (more negative)
                lo, hi = lim["left_hip_pitch_joint"][0] + margin, 0.3
                f_lo = com_offset(kin, place(kin, leg_pose(lo, knee, ankle, waist)))
                f_hi = com_offset(kin, place(kin, leg_pose(hi, knee, ankle, waist)))
                if np.sign(f_lo) == np.sign(f_hi):
                    continue
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    f_mid = com_offset(kin, place(kin, leg_pose(mid, knee, ankle, waist)))
                    if np.sign(f_mid) == np.sign(f_lo):
                        lo, f_lo = mid, f_mid
                    else:
                        hi = mid
                hip = 0.5 * (lo + hi)
                q = leg_pose(hip, knee, ankle, waist)
                tfs = place(kin, q)
                pelvis = tfs["pelvis"]
                torso = tfs["torso_link"]
                torso_lean = np.degrees(np.arctan2(torso[0, 2], torso[2, 2]))
                pelvis_pitch = np.degrees(np.arctan2(pelvis[0, 2], pelvis[2, 2]))
                foot = tfs["left_ankle_roll_link"]
                toe_x = (foot[:3, :3] @ np.array([SOLE_X[1], 0, SOLE_Z]) + foot[:3, 3])[0]
                sh = tfs["left_shoulder_pitch_link"][:3, 3]
                target = np.array([toe_x + args.reach_ahead, sh[1], args.object_height])
                reach = np.linalg.norm(target - sh)
                knee_tau = abs(kin.gravity_torque(tfs, "left_knee_link", np.array([0, 1, 0]))) / 2
                ankle_tau = abs(kin.gravity_torque(tfs, "left_ankle_pitch_link", np.array([0, 1, 0]))) / 2
                knee_x = tfs["left_knee_link"][0, 3]
                results.append(dict(knee_ahead_of_toe=knee_x - toe_x,
                    hip=hip, knee=knee, ankle=ankle, waist=waist, pelvis_z=pelvis[2, 3],
                    pelvis_pitch=pelvis_pitch, torso_lean=torso_lean, shoulder_z=sh[2],
                    reach=reach, reach_ok=reach <= args.reach_frac * L, knee_tau=knee_tau, ankle_tau=ankle_tau,
                    knee_z=tfs["left_knee_link"][2, 3],
                ))

    ok = [r for r in results if r["reach_ok"] and r["torso_lean"] < 50]
    print(f"{len(results)} balanced poses, {len(ok)} reach the ground target")
    ok.sort(key=lambda r: -r["pelvis_z"])
    hdr = "  hip   knee  ankle waist | pelvis_z pel_pitch torso_lean shoulder_z reach | knee_tau ankle_tau"
    print(hdr)
    for r in ok[:15]:
        print(f"{r['hip']:+.2f} {r['knee']:+.2f} {r['ankle']:+.2f} {r['waist']:+.2f} | "
              f"{r['pelvis_z']:.3f}  {r['pelvis_pitch']:+6.1f}  {r['torso_lean']:+6.1f}    {r['shoulder_z']:.3f}   "
              f"{r['reach']:.3f} | {r['knee_tau']:5.1f}  {r['ankle_tau']:5.1f}")
    if not ok:
        return
    best = ok[0]
    print("\nchosen:", {k: round(float(v), 3) for k, v in best.items() if not isinstance(v, (bool, np.bool_))})


if __name__ == "__main__":
    main()
