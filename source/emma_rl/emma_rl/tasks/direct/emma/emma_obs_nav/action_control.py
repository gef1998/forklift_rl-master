"""Pure Torch action mappings for the Emma differential-drive base."""

from __future__ import annotations

import math
import torch


SUPPORTED_ACTION_CONTROL_MODES = ("body_twist", "wheel_velocity")


def limit_wheel_velocity_acceleration(
    desired_wheels: torch.Tensor,
    previous_wheels: torch.Tensor,
    dt: float,
    wheel_radius: float,
    wheel_base: float,
    max_linear_acceleration: float,
    max_angular_acceleration: float,
    wheel_radius_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound nominal body-twist command changes, including braking/reversal.

    Interpolate both wheels by one fraction. If both endpoints satisfy the
    wheel/body speed limits, the result stays in that convex feasible set.
    Independent wheel or twist-component clipping does not have this property.
    This is a command bound, not a bound on collision impulses or measured
    acceleration. Apply the safety shield afterwards so braking is not delayed.
    """
    if desired_wheels.shape != previous_wheels.shape:
        raise ValueError("desired and previous wheels must have the same shape")
    if any(not math.isfinite(x) or x <= 0 for x in (
        dt, max_linear_acceleration, max_angular_acceleration
    )):
        raise ValueError("dt and acceleration limits must be finite and positive")
    delta_wheels = desired_wheels - previous_wheels
    delta_twist = wheel_velocities_to_body_twist(
        delta_wheels, wheel_radius, wheel_base, wheel_radius_scale
    )
    linear_scale = (max_linear_acceleration * dt / delta_twist[..., 0].abs().clamp_min(1e-8))
    angular_scale = (max_angular_acceleration * dt / delta_twist[..., 1].abs().clamp_min(1e-8))
    scale = torch.minimum(linear_scale, angular_scale).clamp(max=1.0)
    return previous_wheels + delta_wheels * scale.unsqueeze(-1), scale


def differential_drive_kinematic_limits(
    wheel_max_velocity: float,
    wheel_radius: float,
    wheel_base: float,
) -> tuple[float, float]:
    """返回轮速上限对应的直行和原地旋转运动学极限。

    ``wheel_base`` 在本项目中表示左右驱动轮接地点的横向中心距
    （track width），不是车辆前后方向的轴距。
    """

    if wheel_max_velocity <= 0 or wheel_radius <= 0 or wheel_base <= 0:
        raise ValueError("wheel limit, radius, and track width must be positive")
    max_linear_velocity = wheel_max_velocity * wheel_radius
    max_angular_velocity = (
        2.0 * wheel_max_velocity * wheel_radius / wheel_base
    )
    return max_linear_velocity, max_angular_velocity


def normalize_action_control_mode(mode: str) -> str:
    """Normalize and validate the configured action interpretation."""

    normalized = mode.strip().lower()
    if normalized not in SUPPORTED_ACTION_CONTROL_MODES:
        choices = ", ".join(SUPPORTED_ACTION_CONTROL_MODES)
        raise ValueError(
            f"Unknown action_control_mode={mode!r}; expected one of: {choices}"
        )
    return normalized


def actions_to_raw_wheel_velocities(
    actions: torch.Tensor,
    control_mode: str,
    wheel_max_velocity: float,
    wheel_radius: float,
    wheel_base: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    wheel_radius_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map signed policy actions to wheel velocities before feasibility projection.

    ``body_twist`` interprets the two action components as normalized linear and
    angular velocity. ``wheel_velocity`` interprets them directly as normalized
    left and right wheel angular velocity.
    """

    if actions.shape[-1] != 2:
        raise ValueError("Emma actions must have exactly two components")
    if wheel_max_velocity <= 0 or wheel_radius <= 0 or wheel_base <= 0:
        raise ValueError("wheel limits, radius, and base must be positive")
    if max_linear_velocity <= 0 or max_angular_velocity <= 0:
        raise ValueError("body velocity limits must be positive")

    mode = normalize_action_control_mode(control_mode)
    signed_actions = actions.clamp(-1.0, 1.0)
    if mode == "wheel_velocity":
        return signed_actions * wheel_max_velocity

    effective_radius = actions.new_tensor(wheel_radius)
    if wheel_radius_scale is not None:
        effective_radius = effective_radius * wheel_radius_scale.to(
            device=actions.device, dtype=actions.dtype
        )

    linear_velocity = signed_actions[..., 0] * max_linear_velocity
    angular_velocity = signed_actions[..., 1] * max_angular_velocity
    half_base = wheel_base / 2.0
    left = (linear_velocity - angular_velocity * half_base) / effective_radius
    right = (linear_velocity + angular_velocity * half_base) / effective_radius
    return torch.stack((left, right), dim=-1)


def project_wheel_velocities(
    wheel_velocities: torch.Tensor,
    wheel_max_velocity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale both wheels together so limits are met without changing curvature."""

    if wheel_velocities.shape[-1] != 2:
        raise ValueError("wheel_velocities must end with (left, right)")
    if wheel_max_velocity <= 0:
        raise ValueError("wheel_max_velocity must be positive")

    peak = wheel_velocities.abs().amax(dim=-1)
    scale = (wheel_max_velocity / peak.clamp_min(1.0e-8)).clamp(max=1.0)
    return wheel_velocities * scale.unsqueeze(-1), scale


def project_wheel_velocities_to_motion_limits(
    wheel_velocities: torch.Tensor,
    wheel_max_velocity: float,
    wheel_radius: float,
    wheel_base: float,
    max_linear_velocity: float,
    max_angular_velocity: float,
    wheel_radius_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project wheels to independent body and actuator limits.

    One scale is applied to both wheels, preserving the requested curvature.
    This is required for ``wheel_velocity`` actions: a valid individual wheel
    target can still request body linear or angular velocity above the robot's
    configured deployment limits.
    """

    if max_linear_velocity <= 0 or max_angular_velocity <= 0:
        raise ValueError("body velocity limits must be positive")

    twist = wheel_velocities_to_body_twist(
        wheel_velocities,
        wheel_radius,
        wheel_base,
        wheel_radius_scale,
    )
    linear_peak = twist[..., 0].abs()
    angular_peak = twist[..., 1].abs()
    linear_scale = (
        max_linear_velocity / linear_peak.clamp_min(1.0e-8)
    ).clamp(max=1.0)
    angular_scale = (
        max_angular_velocity / angular_peak.clamp_min(1.0e-8)
    ).clamp(max=1.0)
    body_scale = torch.minimum(linear_scale, angular_scale)

    body_limited = wheel_velocities * body_scale.unsqueeze(-1)
    wheel_limited, wheel_scale = project_wheel_velocities(
        body_limited, wheel_max_velocity
    )
    total_scale = body_scale * wheel_scale
    return wheel_limited, total_scale


def wheel_velocities_to_body_twist(
    wheel_velocities: torch.Tensor,
    wheel_radius: float,
    wheel_base: float,
    wheel_radius_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert left/right wheel angular velocity to body ``(v, omega)``."""

    if wheel_velocities.shape[-1] != 2:
        raise ValueError("wheel_velocities must end with (left, right)")
    if wheel_radius <= 0 or wheel_base <= 0:
        raise ValueError("wheel_radius and wheel_base must be positive")

    effective_radius = wheel_velocities.new_tensor(wheel_radius)
    if wheel_radius_scale is not None:
        effective_radius = effective_radius * wheel_radius_scale.to(
            device=wheel_velocities.device, dtype=wheel_velocities.dtype
        )

    left = wheel_velocities[..., 0]
    right = wheel_velocities[..., 1]
    linear_velocity = 0.5 * effective_radius * (left + right)
    angular_velocity = effective_radius * (right - left) / wheel_base
    return torch.stack((linear_velocity, angular_velocity), dim=-1)
