import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="X2-Flat-Train",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.X2.envs:BipedalManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)

gym.register(
    id="X2-Flat-Play",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.X2.envs:BipedalManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)

gym.register(
    id="X2-Rough-Train",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.X2.envs:BipedalManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.X2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2RoughPPORunnerCfg",
    },
)

gym.register(
    id="X2-Rough-Play",
    entry_point="isaaclab_tasks.manager_based.locomotion.velocity.config.X2.envs:BipedalManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.X2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2RoughPPORunnerCfg",
    },
)
