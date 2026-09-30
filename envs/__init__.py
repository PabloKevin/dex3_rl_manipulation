# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""Register Isaac Lab tasks with gymnasium."""

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
    id="AppleGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGrasp-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": AppleGraspCameraEnvCfg},
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspGraspEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspGrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspGraspPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspHarvest-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspHarvestEnvCfg,
        "rsl_rl_cfg_entry_point": AppleGraspHarvestPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="AppleGraspHarvest-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": AppleGraspHarvestEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AppleGraspHarvestPPORunnerCfg,
    },
    disable_env_checker=True,
)
