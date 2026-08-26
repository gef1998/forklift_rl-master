"""Pure Torch helpers for a Beta policy mapped from ``(0, 1)`` to ``(-1, 1)``."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def beta_concentrations(
    raw_alpha: torch.Tensor,
    raw_beta: torch.Tensor,
    min_concentration: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert unconstrained network outputs to stable Beta concentrations."""

    if min_concentration <= 0:
        raise ValueError("min_concentration must be positive")
    return (
        F.softplus(raw_alpha) + min_concentration,
        F.softplus(raw_beta) + min_concentration,
    )


def concentration_bias(initial_concentration: float, min_concentration: float) -> float:
    """Return the raw bias producing ``initial_concentration`` after softplus."""

    residual = initial_concentration - min_concentration
    if residual <= 0:
        raise ValueError("initial_concentration must exceed min_concentration")
    return math.log(math.expm1(residual))


def unit_to_signed(unit_actions: torch.Tensor) -> torch.Tensor:
    """Map Beta samples from ``(0, 1)`` to signed actions in ``(-1, 1)``."""

    return unit_actions * 2.0 - 1.0


def signed_to_unit(signed_actions: torch.Tensor, eps: float) -> torch.Tensor:
    """Invert the affine action map with numerical protection for log-prob."""

    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    return ((signed_actions + 1.0) * 0.5).clamp(eps, 1.0 - eps)


def mapped_beta_log_prob(
    distribution: torch.distributions.Independent,
    signed_actions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Log-probability after the affine map ``a = 2u - 1``."""

    unit_actions = signed_to_unit(signed_actions, eps)
    action_dim = signed_actions.shape[-1]
    return distribution.log_prob(unit_actions) - action_dim * math.log(2.0)


def mapped_beta_entropy(
    distribution: torch.distributions.Independent,
    action_dim: int,
) -> torch.Tensor:
    """Entropy after mapping a factorized Beta distribution to ``(-1, 1)``."""

    return distribution.entropy() + action_dim * math.log(2.0)


def mapped_beta_mean(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Deterministic signed action obtained from the Beta mean."""

    return unit_to_signed(alpha / (alpha + beta))
