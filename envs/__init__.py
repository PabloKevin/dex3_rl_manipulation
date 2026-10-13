# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Register Isaac Lab tasks with gymnasium.

v1 = commercial fruiting-wall tree, side-on reach (palm normal along the
stem, finger direction free). The v0 IDs (AppleGrasp-v0 = reach, AppleGraspGrasp-v0,
AppleGraspHarvest-v0: hanging apple, palm-up reach) were retired with that
change; their last version is commit f023a00. Log folders still follow the
runner's experiment_name (apple_grasp_reach / _grasp / _harvest).
"""

import gymnasium as gym
from .apple_grasp_env_cfg import (
    AppleGraspEnvCfg,
    AppleGraspEnvCfg_PLAY,
    AppleGraspCameraEnvCfg,
)
from .apple_grasp_grasp_env_cfg import (
    AppleGraspGraspEnvCfg,
    AppleGraspGraspEnvCfg_PLAY,
)
from .apple_grasp_harvest_env_cfg import (
    AppleGraspHarvestEnvCfg,
    AppleGraspHarvestEnvCfg_PLAY,
)
from .apple_reach_tree_env_cfg import (
    AppleReachTreeEnvCfg,
    AppleReachTreeEnvCfg_PLAY,
)
from .apple_reach_tree_waist_env_cfg import (
    AppleReachTreeWaistEnvCfg,
    AppleReachTreeWaistEnvCfg_PLAY,
    AppleReachTreeWaistFaceEnvCfg,
    AppleReachTreeWaistFaceEnvCfg_PLAY,
)
from .apple_reach_spindle_env_cfg import (
    AppleReachSpindleEnvCfg,
    AppleReachSpindleEnvCfg_PLAY,
    AppleReachSpindleV2EnvCfg,
    AppleReachSpindleV2EnvCfg_PLAY,
    AppleReachSpindleV3EnvCfg,
    AppleReachSpindleV3EnvCfg_PLAY,
)
from .apple_grasp_tree_train_env_cfg import (
    AppleGraspTreeTrainEnvCfg,
    AppleGraspTreeTrainEnvCfg_PLAY,
)
from .apple_grasp_spindle_env_cfg import (
    AppleGraspSpindleEnvCfg,
    AppleGraspSpindleEnvCfg_PLAY,
    AppleGraspSpindleV2EnvCfg,
    AppleGraspSpindleV2EnvCfg_PLAY,
    AppleGraspSpindleV3EnvCfg,
    AppleGraspSpindleV3EnvCfg_PLAY,
)
from .apple_arm_goal_env_cfg import (
    AppleArmGoalEnvCfg,
    AppleArmGoalEnvCfg_PLAY,
)
from .apple_arm_goal_waist_env_cfg import (
    AppleArmGoalWaistEnvCfg,
    AppleArmGoalWaistEnvCfg_PLAY,
)
from .apple_reach_twostage_env_cfg import (
    AppleReachPregraspEnvCfg,
    AppleReachPregraspEnvCfg_PLAY,
    AppleReachFinalEnvCfg,
    AppleReachFinalEnvCfg_PLAY,
)
from .agents.rsl_rl_ppo_cfg import (
    AppleGraspPPORunnerCfg,
    AppleGraspGraspPPORunnerCfg,
    AppleGraspHarvestPPORunnerCfg,
    AppleReachTreePPORunnerCfg,
    AppleReachTreeWaistPPORunnerCfg,
    AppleReachTreeWaistFacePPORunnerCfg,
    AppleReachSpindlePPORunnerCfg,
    AppleReachSpindleV2PPORunnerCfg,
    AppleReachSpindleV3PPORunnerCfg,
    AppleGraspTreePPORunnerCfg,
    AppleGraspSpindlePPORunnerCfg,
    AppleGraspSpindleV2PPORunnerCfg,
    AppleGraspSpindleV3PPORunnerCfg,
    AppleArmGoalPPORunnerCfg,
    AppleArmGoalWaistPPORunnerCfg,
    AppleReachPregraspPPORunnerCfg,
    AppleReachFinalPPORunnerCfg,
)

gym.register(
    id="AppleReach-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReach-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReach-Camera-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": AppleGraspCameraEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="AppleGrasp-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspGraspEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGrasp-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleHarvest-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspHarvestEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspHarvestPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleHarvest-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspHarvestEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspHarvestPPORunnerCfg,
    },
    disable_env_checker=True,
)

# Reach on the multi-apple tree (neighbours on stems, clearance-ring input).
gym.register(
    id="AppleReachTree-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachTreePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachTree-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachTreePPORunnerCfg,
    },
    disable_env_checker=True,
)

# Tree reach that also turns the torso (waist yaw): further, more central fruit.
gym.register(
    id="AppleReachTreeWaist-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeWaistEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachTreeWaistPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachTreeWaist-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeWaistEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachTreeWaistPPORunnerCfg,
    },
    disable_env_checker=True,
)

# Same, preferring the torso (head camera) facing the apple where the reach still works.
gym.register(
    id="AppleReachTreeWaistFace-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeWaistFaceEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachTreeWaistFacePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachTreeWaistFace-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachTreeWaistFaceEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachTreeWaistFacePPORunnerCfg,
    },
    disable_env_checker=True,
)

# Facing waist reach on the tall-spindle tree (thin leaning trunk, short branches).
gym.register(
    id="AppleReachSpindle-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachSpindlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachSpindle-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachSpindlePPORunnerCfg,
    },
    disable_env_checker=True,
)

# Spindle reach v2: keeps off the neighbours (clearance, fingers sideways in clusters, height first).
gym.register(
    id="AppleReachSpindle-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleV2EnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachSpindleV2PPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachSpindle-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleV2EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachSpindleV2PPORunnerCfg,
    },
    disable_env_checker=True,
)

# Spindle reach v3: v2 + motor load (energy) penalty.
gym.register(
    id="AppleReachSpindle-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleV3EnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachSpindleV3PPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachSpindle-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachSpindleV3EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachSpindleV3PPORunnerCfg,
    },
    disable_env_checker=True,
)

# Grasp on the multi-apple tree (neighbours, fruit on its stem, hold-under-load).
gym.register(
    id="AppleGraspTree-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspTreeTrainEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspTreePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspTree-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspTreeTrainEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspTreePPORunnerCfg,
    },
    disable_env_checker=True,
)

# Grasp after the waist reach (v3), on the spindle tree.
gym.register(
    id="AppleGraspSpindle-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspSpindlePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspSpindle-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspSpindlePPORunnerCfg,
    },
    disable_env_checker=True,
)

# Grasp v2: stem-like twist load (breaks at the stem's torque), harvest grip command.
gym.register(
    id="AppleGraspSpindle-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleV2EnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspSpindleV2PPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspSpindle-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleV2EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspSpindleV2PPORunnerCfg,
    },
    disable_env_checker=True,
)

# Goal-reaching arm policy for the harvest path (safe zone -> basket -> home).
gym.register(
    id="AppleArmGoal-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleArmGoalEnvCfg,
        "rsl_rl_cfg_entry_point": AppleArmGoalPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleArmGoal-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleArmGoalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleArmGoalPPORunnerCfg,
    },
    disable_env_checker=True,
)

# Harvest-path goal policy with the waist yaw (after the waist reach).
gym.register(
    id="AppleArmGoalWaist-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleArmGoalWaistEnvCfg,
        "rsl_rl_cfg_entry_point": AppleArmGoalWaistPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleArmGoalWaist-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleArmGoalWaistEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleArmGoalWaistPPORunnerCfg,
    },
    disable_env_checker=True,
)

# Grasp v3: + the bench's pull after an unbroken twist.
gym.register(
    id="AppleGraspSpindle-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleV3EnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspSpindleV3PPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspSpindle-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspSpindleV3EnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspSpindleV3PPORunnerCfg,
    },
    disable_env_checker=True,
)

# Two-stage reach (apple_reach_twostage_env_cfg.py): part 1 to the pregrasp point, fingers open;
# part 2 from there to the palm touch, cupped hand. Magnet-like stems on every fruit.

gym.register(
    id="AppleReachPregrasp-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachPregraspEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachPregraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachPregrasp-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachPregraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachPregraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachFinal-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachFinalEnvCfg,
        "rsl_rl_cfg_entry_point": AppleReachFinalPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleReachFinal-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleReachFinalEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleReachFinalPPORunnerCfg,
    },
    disable_env_checker=True,
)
