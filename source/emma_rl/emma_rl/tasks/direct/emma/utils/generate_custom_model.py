"""Build the custom shared actor/critic used by the maintained Emma PPO tasks."""

from __future__ import annotations

import copy
from typing import Any

from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model

from ..emma_costmap_nav.emma_costmap_nav_network import EmmaCostmapShared
from ..emma_obs_nav.emma_obs_nav_network import EmmaNavShared


def generate_custom_model(
    runner,
    env: Wrapper | MultiAgentEnvWrapper,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Model]]:
    """Generate one of the two supported Emma shared PPO models."""

    multi_agent = isinstance(env, MultiAgentEnvWrapper)
    device = env.device
    possible_agents = env.possible_agents if multi_agent else ["agent"]
    observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
    state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
    action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

    del cfg["models"]["separate"]
    model_classes = {
        "EmmaNavShared": EmmaNavShared,
        "EmmaCostmapShared": EmmaCostmapShared,
    }
    model_name = cfg["models"].pop("name")
    try:
        shared_model_cls = model_classes[model_name]
    except KeyError as exc:
        raise ValueError(f"Unknown maintained custom model: {model_name}") from exc

    models: dict[str, dict[str, Model]] = {}
    for agent_id in possible_agents:
        model_cfg = copy.deepcopy(cfg).get("models")
        if not model_cfg:
            raise ValueError("The 'models' field is not defined in the specified configuration")

        parameters = {}
        for role, role_cfg in model_cfg.items():
            if not role_cfg.get("class"):
                raise ValueError(f"The 'models.{role}.class' field is not defined")
            del role_cfg["class"]
            parameters[role] = runner._process_cfg(role_cfg)

        shared_model = shared_model_cls(
            observation_spaces[agent_id],
            state_spaces[agent_id],
            action_spaces[agent_id],
            parameters,
            device,
        )
        models[agent_id] = {"policy": shared_model, "value": shared_model}
        for role, model in models[agent_id].items():
            model.init_state_dict(role=role)

    return models
