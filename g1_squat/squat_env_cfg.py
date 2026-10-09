"""G1 squat task: lower the CoM to a deep flat-foot squat, hold it while the arms move,
and stand back up to the velocity policy's stand pose — never falling.

Policy: 15 joints (6+6 legs, 3 waist) as residuals around the stand<->squat reference;
arms and Dex3 fingers are driven by the scripted ArmMotionCommand (stand pose while
moving down/up, random reaching + hand loads while the squat is held).
Observation layout mirrors unitree_rl_lab's velocity policy (5-step history, same
scales/noise), with the squat command (target, phase) instead of the velocity command.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from g1_squat import mdp
from g1_squat import robot_cfg as rc

SDK_JOINTS_CFG = SceneEntityCfg("robot", joint_names=rc.SDK_JOINTS, preserve_order=True)
LEG_WAIST_CFG = SceneEntityCfg("robot", joint_names=rc.LEG_WAIST_JOINTS, preserve_order=True)
FEET_CFG = SceneEntityCfg("robot", body_names=rc.FEET_BODIES, preserve_order=True)

# Random arm goals while squatting (absolute joint ranges, rad). Shoulder pitch < 0 = forward,
# elbow ~1.45 = straight arm. Covers reaching forward/down/sideways around the knees.
ARM_BOX = {
    "left_shoulder_pitch_joint": (-1.6, 0.6), "right_shoulder_pitch_joint": (-1.6, 0.6),
    "left_shoulder_roll_joint": (-0.2, 0.9), "right_shoulder_roll_joint": (-0.9, 0.2),
    "left_shoulder_yaw_joint": (-0.8, 0.8), "right_shoulder_yaw_joint": (-0.8, 0.8),
    "left_elbow_joint": (-0.2, 1.5), "right_elbow_joint": (-0.2, 1.5),
    "left_wrist_roll_joint": (-1.0, 1.0), "right_wrist_roll_joint": (-1.0, 1.0),
    "left_wrist_pitch_joint": (-0.8, 0.8), "right_wrist_pitch_joint": (-0.8, 0.8),
    "left_wrist_yaw_joint": (-0.8, 0.8), "right_wrist_yaw_joint": (-0.8, 0.8),
}


@configclass
class SquatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = rc.G1_DEX3_SQUAT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class CommandsCfg:
    squat = mdp.SquatCommandCfg()
    arm_motion = mdp.ArmMotionCommandCfg(joint_box=ARM_BOX)


@configclass
class ActionsCfg:
    legs_waist = mdp.SquatResidualJointActionCfg(
        asset_name="robot", joint_names=rc.LEG_WAIST_JOINTS, preserve_order=True, scale=0.25, command_name="squat"
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        squat_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "squat"})
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SDK_JOINTS_CFG}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SDK_JOINTS_CFG}, scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        torso_gravity = ObsTerm(
            func=mdp.torso_projected_gravity, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}
        )
        squat_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "squat"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SDK_JOINTS_CFG})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SDK_JOINTS_CFG}, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        height_error = ObsTerm(func=mdp.pelvis_height_error, params={"command_name": "squat"}, scale=10.0)
        com_offset = ObsTerm(func=mdp.com_offset_b, params={"asset_cfg": FEET_CFG}, scale=10.0)
        feet_forces = ObsTerm(
            func=mdp.feet_contact_forces,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=rc.FEET_BODIES, preserve_order=True)},
        )
        arm_targets = ObsTerm(func=mdp.arm_targets, params={"command_name": "arm_motion"})
        hand_loads = ObsTerm(func=mdp.hand_loads, params={"command_name": "arm_motion"})

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.03, 0.03)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": LEG_WAIST_CFG,
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    # reset: start in the velocity policy's stand pose (the hand-over state)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (-0.3, 0.3)},
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}},
    )


MIRROR_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
]


@configclass
class RewardsCfg:
    # -- task
    pelvis_height = RewTerm(func=mdp.pelvis_height_tracking, weight=2.0, params={"std": 0.05})
    posture = RewTerm(func=mdp.posture_tracking, weight=1.0, params={"std": 0.25})
    com_over_support = RewTerm(func=mdp.com_over_support, weight=1.5, params={"std": 0.04, "asset_cfg": FEET_CFG})
    stand_settled = RewTerm(func=mdp.stand_settled, weight=0.5, params={"std": 0.1})
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    fall = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- base / balance
    base_lin_vel_xy = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-1.0)
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    torso_roll = RewTerm(
        func=mdp.body_roll_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}
    )
    pelvis_roll = RewTerm(
        func=mdp.body_roll_l2, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}
    )

    # -- feet: planted and flat
    feet_flat = RewTerm(func=mdp.feet_flat_l2, weight=-2.0, params={"asset_cfg": FEET_CFG})
    feet_contact_lost = RewTerm(
        func=mdp.feet_contact_lost,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=rc.FEET_BODIES, preserve_order=True)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # -- joints
    off_plane_joints = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_ankle_roll_joint",
                                      "waist_yaw_joint", "waist_roll_joint"]
            )
        },
    )
    mirror = RewTerm(func=mdp.mirror_symmetry_l2, weight=-0.5, params={"pairs": MIRROR_PAIRS})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": LEG_WAIST_CFG})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": LEG_WAIST_CFG})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": LEG_WAIST_CFG})
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5, params={"asset_cfg": LEG_WAIST_CFG})
    effort_near_limit = RewTerm(
        func=mdp.effort_near_limit, weight=-2.0, params={"ratio": 0.8, "asset_cfg": LEG_WAIST_CFG}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["pelvis", "torso_link", ".*_hip_.*_link", ".*_knee_link", "waist_.*_link"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pelvis_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.20})
    body_on_ground = DoneTerm(
        func=mdp.bodies_below_height,
        params={
            "minimum_height": 0.07,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee_link", "torso_link", "pelvis"]),
        },
    )
    torso_tilt = DoneTerm(
        func=mdp.body_tilt_exceeds,
        params={"max_pitch": 1.35, "max_roll": 0.6, "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    pelvis_tilt = DoneTerm(
        func=mdp.body_tilt_exceeds,
        params={"max_pitch": 1.2, "max_roll": 0.6, "asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    )


@configclass
class G1SquatEnvCfg(ManagerBasedRLEnvCfg):
    scene: SquatSceneCfg = SquatSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4  # 50 Hz policy, same as the velocity policy
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (2.5, 2.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class G1SquatEnvCfg_PLAY(G1SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.arm_motion.curriculum_steps = 0  # full arm motion from the start
