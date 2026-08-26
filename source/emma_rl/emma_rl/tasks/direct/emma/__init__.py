"""Gymnasium registrations for the maintained Emma navigation tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="Emma-RL-ObsNav-Direct-v0",
    entry_point=f"{__name__}.emma_obs_nav.emma_obs_nav_env:EmmaObsNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.emma_obs_nav.emma_obs_nav_env_cfg:EmmaObsNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_emma_obs_nav_ppo_cfg.yaml",
    },
)

gym.register(
    id="Emma-RL-CostmapNav-Direct-v0",
    entry_point=f"{__name__}.emma_costmap_nav.emma_costmap_nav_env:EmmaCostmapNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.emma_costmap_nav.emma_costmap_nav_env_cfg:EmmaCostmapNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_emma_costmap_nav_ppo_cfg.yaml",
    },
)
