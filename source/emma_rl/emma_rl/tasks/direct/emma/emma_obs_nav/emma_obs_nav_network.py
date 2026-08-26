"""Shared Beta actor-critic network for EmmaObsNavEnv."""

import torch
import torch.nn as nn
from torch.distributions import Beta, Independent
from skrl.models.torch import DeterministicMixin, Model

from .beta_policy import (
    beta_concentrations,
    concentration_bias,
    mapped_beta_entropy,
    mapped_beta_log_prob,
    mapped_beta_mean,
    unit_to_signed,
)


class EmmaNavShared(DeterministicMixin, Model):
    """Shared actor-critic for EmmaObsNavEnv.

    Architecture:
      Actor:  obs -> 256(ELU) -> 128(ELU) -> 64(ELU) -> (alpha, beta)
      Critic: obs -> 256(ELU) -> 128(ELU) -> 64(ELU) -> 1

    The actor samples ``u ~ Beta(alpha, beta)`` in ``(0, 1)`` and exposes the
    signed differential-drive action ``a = 2u - 1`` to PPO and the environment.
    """

    def __init__(self, observation_space, state_space, action_space, parameters, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )

        policy_params = parameters["policy"]
        value_params = parameters["value"]
        DeterministicMixin.__init__(
            self, clip_actions=value_params["clip_actions"], role="value",
        )
        self._min_concentration = float(policy_params["min_concentration"])
        self._initial_concentration = float(policy_params["initial_concentration"])
        self._action_eps = float(policy_params["action_eps"])
        if self._initial_concentration <= self._min_concentration:
            raise ValueError("initial_concentration must exceed min_concentration")
        if not 0.0 < self._action_eps < 0.5:
            raise ValueError("action_eps must be in (0, 0.5)")
        self._beta_distribution: Independent | None = None

        # Shared backbone
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # Policy head
        self.policy_layer = nn.Linear(64, self.num_actions * 2)
        nn.init.orthogonal_(self.policy_layer.weight, gain=0.01)
        nn.init.constant_(
            self.policy_layer.bias,
            concentration_bias(
                self._initial_concentration, self._min_concentration),
        )

        # Value head
        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            alpha, outputs = self.compute(inputs, role=role)
            beta = outputs["beta"]
            self._beta_distribution = Independent(
                Beta(alpha, beta, validate_args=False), 1)

            unit_actions = self._beta_distribution.sample()
            actions = unit_to_signed(unit_actions)
            actions_for_log_prob = inputs.get("taken_actions", actions)
            log_prob = mapped_beta_log_prob(
                self._beta_distribution,
                actions_for_log_prob,
                self._action_eps,
            ).unsqueeze(-1)
            # skrl 1.4.x model API returns exactly ``(actions, outputs)``.
            # PPO reads the behavior-policy likelihood from this dictionary
            # both during rollout collection and minibatch updates.
            outputs["log_prob"] = log_prob
            outputs["mean_actions"] = mapped_beta_mean(alpha, beta)
            return actions, outputs
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role=role)

    def get_entropy(self, role: str = "policy") -> torch.Tensor:
        if role != "policy" or self._beta_distribution is None:
            raise RuntimeError("Beta policy entropy requested before policy.act")
        return mapped_beta_entropy(
            self._beta_distribution, self.num_actions).unsqueeze(-1)

    def distribution(self, role: str = "policy") -> Independent:
        if role != "policy" or self._beta_distribution is None:
            raise RuntimeError("Beta policy distribution requested before policy.act")
        return self._beta_distribution

    def get_log_std(self, role: str = "policy") -> torch.Tensor:
        """Return the signed-action marginal log-std for skrl diagnostics."""

        if role != "policy" or self._beta_distribution is None:
            raise RuntimeError("Beta policy std requested before policy.act")
        variance = self._beta_distribution.base_dist.variance
        return torch.log((2.0 * torch.sqrt(variance)).clamp_min(1.0e-8))

    def get_std(self, role: str = "policy") -> torch.Tensor:
        return self.get_log_std(role).exp()

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["observations"])
            raw_alpha, raw_beta = self.policy_layer(self._shared_output).chunk(2, dim=-1)
            alpha, beta = beta_concentrations(
                raw_alpha, raw_beta, self._min_concentration)
            return alpha, {"beta": beta, "alpha": alpha}
        elif role == "value":
            if self._shared_output is None:
                shared_output = self.net(inputs["observations"])
            else:
                shared_output = self._shared_output
                self._shared_output = None
            return self.value_layer(shared_output), {}
