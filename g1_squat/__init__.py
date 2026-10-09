"""G1 squat task — separate from the apple-harvest tasks in envs/ (nothing here imports envs/).

Registers:
  G1-Squat-v0       training (4096 envs, pushes, arm-motion curriculum)
  G1-Squat-Play-v0  evaluation (16 envs, no pushes, full arm motion)
"""

import gymnasium as gym

gym.register(
    id="G1-Squat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:G1SquatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1SquatPPORunnerCfg",
    },
)

gym.register(
    id="G1-Squat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.squat_env_cfg:G1SquatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1SquatPPORunnerCfg",
    },
)
