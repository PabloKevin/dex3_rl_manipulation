"""G1-29dof + Dex3 hands, floating base, for the squat task.

Built from unitree_rl_lab's UNITREE_G1_29DOF_CFG (same actuator gains as the deployed
velocity policy, so the hand-over between the two policies sees identical PD loops),
with the Dex3 URDF swapped in and a soft drive for the finger joints.

Nominal poses
-------------
STAND_JOINT_POS  = Unitree velocity-policy default pose (deploy default_joint_pos) — the
                   hand-over pose in both directions.
SQUAT_JOINT_POS  = deep flat-foot squat designed with tools/design_squat_pose.py:
                   pelvis 0.353 m, torso lean ~53 deg, whole-body CoM over the centre of
                   the feet, grasp point (palm +8 cm) reaches 10 cm above the ground at the
                   toes with ~6 cm spare. Arms at the stand pose.
The policy acts as a residual around the reference q_ref(s) = lerp(stand, squat, smoothstep(s)).
"""

from __future__ import annotations

import functools
import os

import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG, UnitreeUrdfFileCfg

G1_DEX3_URDF = os.environ.get(
    "G1_DEX3_URDF", os.path.expanduser("~/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf")
)

# Unitree SDK motor order (29 body joints). Policy inputs/outputs use this order so the
# deploy side needs no re-indexing.
SDK_JOINTS = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)
LEG_WAIST_JOINTS = SDK_JOINTS[:15]  # 6 left leg, 6 right leg, waist yaw/roll/pitch
ARM_JOINTS = SDK_JOINTS[15:]  # 7 left arm, 7 right arm
HAND_JOINTS = [
    f"{side}_hand_{finger}_joint"
    for side in ("left", "right")
    for finger in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
]
FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# sole contact spheres in the ankle_roll frame: x -0.05 .. 0.12, z -0.03 (r 0.005)
SOLE_CENTRE_LOCAL = (0.035, 0.0, -0.035)
SOLE_HALF_LENGTH = 0.085

STAND_JOINT_POS = {
    "left_hip_pitch_joint": -0.1, "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3, "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2, "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.3, "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97, "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15, "right_wrist_roll_joint": -0.15,
}

# Arm pose held while squatting down / standing up. Unitree's default (shoulder roll
# +-0.25) puts the Dex3 thumbs on the hip-pitch links from stand to mid-squat (checked
# with a kinematic sweep in sim); +-0.40 clears hips, thighs and knees over the whole
# transition. Observations keep Unitree's default pose as offset (hand-over consistency).
ARM_REST_POS = {
    **{k: v for k, v in STAND_JOINT_POS.items() if "shoulder" in k or "elbow" in k or "wrist" in k},
    "left_shoulder_roll_joint": 0.40, "right_shoulder_roll_joint": -0.40,
}

SQUAT_LEG_WAIST = {"hip_pitch": -2.324, "knee": 2.30, "ankle_pitch": -0.75, "waist_pitch": 0.15}


def _pose(values: dict[str, float]) -> dict[str, float]:
    """Full 29-joint dict (0 where not given) from STAND + overrides."""
    q = {name: 0.0 for name in SDK_JOINTS}
    q.update({k: v for k, v in STAND_JOINT_POS.items()})
    q.update(values)
    return q


SQUAT_JOINT_POS = _pose({
    "left_hip_pitch_joint": SQUAT_LEG_WAIST["hip_pitch"], "right_hip_pitch_joint": SQUAT_LEG_WAIST["hip_pitch"],
    "left_knee_joint": SQUAT_LEG_WAIST["knee"], "right_knee_joint": SQUAT_LEG_WAIST["knee"],
    "left_ankle_pitch_joint": SQUAT_LEG_WAIST["ankle_pitch"], "right_ankle_pitch_joint": SQUAT_LEG_WAIST["ankle_pitch"],
    "waist_pitch_joint": SQUAT_LEG_WAIST["waist_pitch"],
})
STAND_FULL = _pose({})


def smoothstep(s):
    return s * s * (3.0 - 2.0 * s)


@functools.cache
def reference_table(num: int = 41) -> dict[str, np.ndarray]:
    """Kinematic reference along the squat phase s in [0, 1] (feet flat on z = 0).

    Returns s, pelvis height above the soles and torso pitch (rad, + = leaning forward)
    of q_ref(s), computed once with the offline URDF kinematics.
    """
    from g1_squat.tools.design_squat_pose import place
    from g1_squat.tools.urdf_kinematics import UrdfKinematics

    kin = UrdfKinematics(G1_DEX3_URDF)
    s_grid = np.linspace(0.0, 1.0, num)
    heights, torso_pitch = [], []
    for s in s_grid:
        w = smoothstep(s)
        q = {k: STAND_FULL[k] + (SQUAT_JOINT_POS[k] - STAND_FULL[k]) * w for k in SDK_JOINTS}
        tfs = place(kin, q)
        heights.append(tfs["pelvis"][2, 3])
        torso = tfs["torso_link"]
        torso_pitch.append(np.arctan2(torso[0, 2], torso[2, 2]))
    return {"s": s_grid, "pelvis_height": np.array(heights), "torso_pitch": np.array(torso_pitch)}


# PD gains of the DEPLOYED velocity policy (unitree_rl_lab deploy/.../velocity/v0/params/deploy.yaml),
# i.e. what the real robot runs while walking. They differ from unitree_rl_lab's current sim cfg
# (waist roll/pitch kp 40 -> 200, arm kd 1 -> 10); with the sim cfg gains the v0 policy falls when
# walking in sim. Using the deployed gains keeps the PD loops identical across the hand-over.
_ACT = UNITREE_G1_29DOF_CFG.actuators
DEPLOY_ACTUATORS = {
    "N7520-14.3": _ACT["N7520-14.3"].replace(
        stiffness={".*_hip_.*": 100.0, "waist_yaw_joint": 200.0}, damping={".*_hip_.*": 2.0, "waist_yaw_joint": 5.0}
    ),
    "N7520-22.5": _ACT["N7520-22.5"].replace(
        stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0}, damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0}
    ),
    "N5020-16": _ACT["N5020-16"].replace(
        stiffness={".*_shoulder_.*": 40.0, ".*_elbow_.*": 40.0, ".*_wrist_roll.*": 40.0, ".*_ankle_.*": 40.0,
                   "waist_roll_joint": 200.0, "waist_pitch_joint": 200.0},
        damping={".*_shoulder_.*": 10.0, ".*_elbow_.*": 10.0, ".*_wrist_roll.*": 10.0, ".*_ankle_.*": 2.0,
                 "waist_roll_joint": 5.0, "waist_pitch_joint": 5.0},
    ),
    "W4010-25": _ACT["W4010-25"].replace(stiffness=40.0, damping=10.0),
}

G1_DEX3_SQUAT_CFG = UNITREE_G1_29DOF_CFG.replace(
    spawn=UnitreeUrdfFileCfg(asset_path=G1_DEX3_URDF),
    init_state=UNITREE_G1_29DOF_CFG.init_state.replace(pos=(0.0, 0.0, 0.80)),
    # Unitree uses 0.9; the squat hip pitch (-2.32) sits 0.2 rad inside the hard limit
    # (-2.53) but outside a 0.9 soft range, so widen it for the limit penalty.
    soft_joint_pos_limit_factor=0.95,
    actuators={
        **DEPLOY_ACTUATORS,
        "dex3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_.*_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=10.0,
            stiffness=2.0,
            damping=0.1,
            armature=0.001,
        ),
    },
)
