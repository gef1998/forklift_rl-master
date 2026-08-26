#!/usr/bin/env python3
"""Pure Torch tests for the signed Beta action distribution."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import types

import torch
from torch.distributions import Beta, Independent


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "emma_rl"
    / "emma_rl"
    / "tasks"
    / "direct"
    / "emma"
    / "emma_obs_nav"
    / "beta_policy.py"
)
SPEC = importlib.util.spec_from_file_location("beta_policy", MODULE_PATH)
beta_policy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(beta_policy)


def test_initial_concentration_bias_produces_beta_two_two():
    bias = beta_policy.concentration_bias(2.0, 1.0)
    raw = torch.full((4, 2), bias)
    alpha, beta = beta_policy.beta_concentrations(raw, raw, 1.0)
    torch.testing.assert_close(alpha, torch.full_like(alpha, 2.0))
    torch.testing.assert_close(beta, torch.full_like(beta, 2.0))


def test_signed_action_mapping_round_trip():
    unit = torch.tensor([[0.1, 0.5, 0.9]])
    signed = beta_policy.unit_to_signed(unit)
    restored = beta_policy.signed_to_unit(signed, eps=1.0e-6)
    torch.testing.assert_close(signed, torch.tensor([[-0.8, 0.0, 0.8]]))
    torch.testing.assert_close(restored, unit)


def test_beta_samples_are_strictly_bounded_after_mapping():
    torch.manual_seed(7)
    distribution = Independent(
        Beta(torch.full((4096, 2), 2.0), torch.full((4096, 2), 2.0)), 1)
    actions = beta_policy.unit_to_signed(distribution.sample())
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)


def test_mapped_log_prob_contains_affine_jacobian():
    alpha = torch.tensor([[2.0, 3.0]])
    beta = torch.tensor([[4.0, 5.0]])
    distribution = Independent(Beta(alpha, beta), 1)
    actions = torch.tensor([[-0.5, 0.25]])
    unit = (actions + 1.0) * 0.5
    expected = distribution.log_prob(unit) - 2.0 * math.log(2.0)
    actual = beta_policy.mapped_beta_log_prob(distribution, actions, eps=1.0e-6)
    torch.testing.assert_close(actual, expected)


def test_beta_mean_maps_to_neutral_signed_action():
    alpha = torch.full((3, 2), 2.0)
    beta = torch.full((3, 2), 2.0)
    mean_action = beta_policy.mapped_beta_mean(alpha, beta)
    torch.testing.assert_close(mean_action, torch.zeros_like(mean_action))


def test_beta_mean_preserves_reverse_and_in_place_rotation():
    alpha = torch.tensor([
        [1.5, 4.0],  # reverse while turning left
        [2.0, 4.0],  # zero linear velocity and positive angular velocity
    ])
    beta = torch.tensor([
        [4.0, 1.5],
        [2.0, 1.5],
    ])
    mean_action = beta_policy.mapped_beta_mean(alpha, beta)

    assert mean_action[0, 0] < 0.0
    torch.testing.assert_close(mean_action[1, 0], torch.tensor(0.0))
    assert mean_action[1, 1] > 0.0


def test_skrl_model_act_uses_two_value_contract():
    """Exercise the skrl 1.4.x rollout and PPO-update return contract."""

    from gymnasium.spaces import Box

    package_name = "_emma_obs_nav_model_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.beta_policy"] = beta_policy

    network_path = MODULE_PATH.parent / "emma_obs_nav_network.py"
    network_spec = importlib.util.spec_from_file_location(
        f"{package_name}.emma_obs_nav_network", network_path
    )
    network_module = importlib.util.module_from_spec(network_spec)
    assert network_spec.loader is not None
    network_spec.loader.exec_module(network_module)

    model = network_module.EmmaNavShared(
        observation_space=Box(low=-float("inf"), high=float("inf"), shape=(130,)),
        state_space=None,
        action_space=Box(low=-1.0, high=1.0, shape=(2,)),
        parameters={
            "policy": {
                "min_concentration": 1.0,
                "initial_concentration": 2.0,
                "action_eps": 1.0e-6,
            },
            "value": {"clip_actions": False},
        },
        device="cpu",
    )
    observations = torch.zeros(8, 130)

    rollout_result = model.act({"observations": observations}, role="policy")
    assert isinstance(rollout_result, tuple) and len(rollout_result) == 2
    actions, outputs = rollout_result
    assert actions.shape == (8, 2)
    assert outputs["log_prob"].shape == (8, 1)
    assert outputs["mean_actions"].shape == (8, 2)

    _, update_outputs = model.act(
        {"observations": observations, "taken_actions": actions}, role="policy"
    )
    assert update_outputs["log_prob"].shape == (8, 1)


def test_costmap_beta_policy_is_bounded_and_cpu_lightweight():
    """Exercise the Costmap model without importing Isaac Lab."""

    from gymnasium.spaces import Box

    emma_dir = MODULE_PATH.parent.parent
    root_name = "_emma_costmap_model_test"
    root_package = types.ModuleType(root_name)
    root_package.__path__ = [str(emma_dir)]
    sys.modules[root_name] = root_package

    obs_package = types.ModuleType(f"{root_name}.emma_obs_nav")
    obs_package.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[f"{root_name}.emma_obs_nav"] = obs_package
    sys.modules[f"{root_name}.emma_obs_nav.beta_policy"] = beta_policy

    costmap_dir = emma_dir / "emma_costmap_nav"
    costmap_package = types.ModuleType(f"{root_name}.emma_costmap_nav")
    costmap_package.__path__ = [str(costmap_dir)]
    sys.modules[f"{root_name}.emma_costmap_nav"] = costmap_package
    network_path = costmap_dir / "costmap_nav_network.py"
    network_spec = importlib.util.spec_from_file_location(
        f"{root_name}.emma_costmap_nav.costmap_nav_network", network_path
    )
    network_module = importlib.util.module_from_spec(network_spec)
    assert network_spec.loader is not None
    network_spec.loader.exec_module(network_module)

    assert network_module.lite_encoder_macs(48) < 1_000_000
    model = network_module.EmmaCostmapShared(
        observation_space=Box(low=-float("inf"), high=float("inf"), shape=(4620,)),
        state_space=None,
        action_space=Box(low=-1.0, high=1.0, shape=(2,)),
        parameters={
            "policy": {
                "min_concentration": 1.0,
                "initial_concentration": 2.0,
                "action_eps": 1.0e-6,
                "costmap_channels": 2,
                "costmap_resolution": 48,
                "robot_state_dim": 12,
            },
            "value": {"clip_actions": False},
        },
        device="cpu",
    )
    observations = torch.zeros(4, 4620)
    actions, outputs = model.act({"observations": observations}, role="policy")
    assert actions.shape == (4, 2)
    assert torch.all(actions > -1.0) and torch.all(actions < 1.0)
    assert outputs["log_prob"].shape == (4, 1)
    values, _ = model.act({"observations": observations}, role="value")
    assert values.shape == (4, 1)


if __name__ == "__main__":
    tests = [
        test_initial_concentration_bias_produces_beta_two_two,
        test_signed_action_mapping_round_trip,
        test_beta_samples_are_strictly_bounded_after_mapping,
        test_mapped_log_prob_contains_affine_jacobian,
        test_beta_mean_maps_to_neutral_signed_action,
        test_beta_mean_preserves_reverse_and_in_place_rotation,
        test_skrl_model_act_uses_two_value_contract,
        test_costmap_beta_policy_is_bounded_and_cpu_lightweight,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
