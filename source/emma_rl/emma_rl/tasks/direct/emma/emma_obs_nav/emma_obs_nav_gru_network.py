"""Shared Beta actor/critic with causal GRU memory for Emma ObsNav.

The skrl agent owns per-environment hidden states. This module never keeps an
implicit inference cache, so policy, value and bootstrap calls cannot advance
one another's memory or reuse features belonging to a different observation.
"""
from __future__ import annotations

import torch
from torch import nn

from .beta_policy import beta_concentrations, concentration_bias
from .emma_obs_nav_network import EmmaNavShared


class EmmaNavGRU(EmmaNavShared):
    def __init__(self, observation_space, state_space, action_space, parameters, device):
        super().__init__(observation_space, state_space, action_space, parameters, device)
        recurrent = parameters["recurrent"]
        self.hidden_size = int(recurrent["hidden_size"])
        self.num_layers = int(recurrent["num_layers"])
        self.sequence_length = int(recurrent["sequence_length"])
        self.num_envs = int(recurrent["num_envs"])
        if min(self.hidden_size, self.num_layers, self.num_envs) < 1 or self.sequence_length < 2:
            raise ValueError("GRU sizes must be positive and sequence_length must be >= 2")
        # Keep the same observation encoder and signed Beta action model as the MLP.
        # The encoder input width comes from observation_space, so it tracks
        # single_obs_dim in the env config.
        self.gru = nn.GRU(64, self.hidden_size, self.num_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        self.policy_layer = nn.Linear(self.hidden_size, self.num_actions * 2)
        nn.init.orthogonal_(self.policy_layer.weight, gain=0.01)
        nn.init.constant_(self.policy_layer.bias, concentration_bias(
            self._initial_concentration, self._min_concentration))
        self.value_layer = nn.Linear(self.hidden_size, 1)

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}

    def _features(self, inputs):
        encoded = self.net(inputs["observations"])
        states = inputs.get("rnn", [])
        if "terminated" not in inputs:
            # Rollout, playback, value bootstrap, or lazy-model initialization.
            hidden = states[0] if states else encoded.new_zeros(
                self.num_layers, encoded.shape[0], self.hidden_size)
            output, hidden = self.gru(encoded.unsqueeze(1), hidden.contiguous())
            return output[:, 0], hidden

        # skrl supplies environment-major, contiguous sequences and the stored
        # input hidden state for EVERY transition. Only each sequence's first
        # stored state starts truncated BPTT; the rest must be recomputed.
        length = self.sequence_length
        if encoded.shape[0] % length or not states:
            raise ValueError("GRU minibatches must contain complete sequences and stored hidden states")
        sequence = encoded.reshape(-1, length, encoded.shape[-1])
        hidden = states[0].reshape(self.num_layers, -1, length, self.hidden_size)[:, :, 0].contiguous()
        done = inputs["terminated"].bool() | inputs.get(
            "truncated", torch.zeros_like(inputs["terminated"], dtype=torch.bool)).bool()
        done = done.reshape(-1, length)
        # Process whole uninterrupted chunks with cuDNN. A done at t resets
        # memory BEFORE t+1, independently for each sequence/environment.
        cuts = (done[:, :-1].any(dim=0).nonzero(as_tuple=False).flatten() + 1).tolist()
        outputs = []
        start = 0
        for end in cuts + [length]:
            output, hidden = self.gru(sequence[:, start:end], hidden)
            outputs.append(output)
            hidden = hidden * (~done[:, end-1]).to(hidden.dtype)[None, :, None]
            start = end
        return torch.cat(outputs, dim=1).reshape(-1, self.hidden_size), hidden

    def compute(self, inputs, role):
        features, hidden = self._features(inputs)
        if role == "policy":
            raw_alpha, raw_beta = self.policy_layer(features).chunk(2, dim=-1)
            alpha, beta = beta_concentrations(raw_alpha, raw_beta, self._min_concentration)
            return alpha, {"alpha": alpha, "beta": beta, "rnn": [hidden]}
        if role == "value":
            return self.value_layer(features), {"rnn": [hidden]}
        raise ValueError(f"Unsupported GRU model role: {role}")
