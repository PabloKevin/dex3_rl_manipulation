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
from .agents.rsl_rl_ppo_cfg import AppleGraspPPORunnerCfg, AppleGraspGraspPPORunnerCfg, AppleGraspHarvestPPORunnerCfg

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
