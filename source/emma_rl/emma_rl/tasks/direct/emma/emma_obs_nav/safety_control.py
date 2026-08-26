"""Pure Torch runtime safety projection for the Emma differential drive."""

from __future__ import annotations

import math

import torch


def _as_batch_tensor(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.ndim == 0:
        return tensor.expand(reference.shape[0])
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(f"{name} must be scalar or have shape {tuple(reference.shape)}")


def _wheel_velocities_to_twist(
    wheel_velocities: torch.Tensor,
    wheel_radius: float,
    wheel_base: float,
    wheel_radius_scale: torch.Tensor | None,
) -> torch.Tensor:
    effective_radius = wheel_velocities.new_full(
        (wheel_velocities.shape[0],), wheel_radius
    )
    if wheel_radius_scale is not None:
        effective_radius = effective_radius * _as_batch_tensor(
            wheel_radius_scale, effective_radius, "wheel_radius_scale"
        )
    left = wheel_velocities[:, 0]
    right = wheel_velocities[:, 1]
    linear = 0.5 * effective_radius * (left + right)
    angular = effective_radius * (right - left) / wheel_base
    return torch.stack((linear, angular), dim=-1)


def apply_safety_control(
    wheel_targets: torch.Tensor,
    shield_clearance: torch.Tensor,
    sector_angles: torch.Tensor,
    shield_aoi: torch.Tensor,
    *,
    wheel_radius: float,
    wheel_base: float,
    lateral_accel_limit: float,
    braking_deceleration: float,
    reaction_time_s: float | torch.Tensor,
    static_margin: float,
    possible_obstacle_speed: float,
    forward_half_angle_deg: float,
    rear_half_angle_deg: float,
    ttc_slow_s: float,
    ttc_stop_s: float,
    max_lidar_aoi_s: float,
    wheel_radius_scale: torch.Tensor | None = None,
    has_valid_scan: torch.Tensor | None = None,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project wheel targets into a conservative dynamic-obstacle safe set.

    The input wheel targets are assumed to have already passed through action
    delay/noise and the wheel-speed feasibility projection. This function only
    scales both wheels together, so it cannot introduce a new wheel-limit
    violation and preserves the requested path curvature whenever it slows.

    ``shield_clearance`` is footprint clearance (not sensor range) from the
    last valid scan, ego-motion-warped into the current body frame.

    ``enabled=False`` is an unsafe experimental ablation: S1 curve limiting,
    S2 braking and S3 emergency stopping are all bypassed. Non-finite wheel
    commands are still replaced by zero so NaNs are never sent to an actuator.
    """

    if wheel_targets.ndim != 2 or wheel_targets.shape[1] != 2:
        raise ValueError("wheel_targets must have shape (N, 2)")
    if shield_clearance.ndim != 2 or shield_clearance.shape[0] != wheel_targets.shape[0]:
        raise ValueError("shield_clearance must have shape (N, sectors)")
    if shield_aoi.shape != (wheel_targets.shape[0],):
        raise ValueError("shield_aoi must have shape (N,)")
    if sector_angles.ndim == 1:
        if sector_angles.shape[0] != shield_clearance.shape[1]:
            raise ValueError("sector_angles length must match clearance sectors")
        angles = sector_angles.unsqueeze(0).expand_as(shield_clearance)
    elif sector_angles.shape == shield_clearance.shape:
        angles = sector_angles
    else:
        raise ValueError("sector_angles must have shape (sectors,) or (N, sectors)")

    positive_values = {
        "wheel_radius": wheel_radius,
        "wheel_base": wheel_base,
        "lateral_accel_limit": lateral_accel_limit,
        "braking_deceleration": braking_deceleration,
        "ttc_slow_s": ttc_slow_s,
        "ttc_stop_s": ttc_stop_s,
        "max_lidar_aoi_s": max_lidar_aoi_s,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if (
        not math.isfinite(static_margin)
        or not math.isfinite(possible_obstacle_speed)
        or static_margin < 0
        or possible_obstacle_speed < 0
    ):
        raise ValueError("safety margins and obstacle speed must be non-negative")
    if (
        not math.isfinite(forward_half_angle_deg)
        or not math.isfinite(rear_half_angle_deg)
        or not 0 < forward_half_angle_deg <= 90
        or not 0 < rear_half_angle_deg <= 90
    ):
        raise ValueError("safety cone half angles must be in (0, 90]")
    if ttc_stop_s >= ttc_slow_s:
        raise ValueError("ttc_stop_s must be smaller than ttc_slow_s")

    finite_inputs = torch.isfinite(wheel_targets).all(dim=-1)
    finite_inputs &= torch.isfinite(shield_clearance).all(dim=-1)
    finite_inputs &= torch.isfinite(angles).all(dim=-1)
    finite_inputs &= torch.isfinite(shield_aoi)
    candidate_wheels = torch.where(
        torch.isfinite(wheel_targets), wheel_targets, torch.zeros_like(wheel_targets)
    )
    clearance = torch.where(
        torch.isfinite(shield_clearance),
        shield_clearance,
        torch.full_like(shield_clearance, -float("inf")),
    )
    angles = torch.where(torch.isfinite(angles), angles, torch.zeros_like(angles))

    batch_reference = shield_aoi.to(dtype=wheel_targets.dtype)
    reaction_time = _as_batch_tensor(
        reaction_time_s, batch_reference, "reaction_time_s"
    )
    finite_inputs &= torch.isfinite(reaction_time)
    reaction_time = torch.nan_to_num(
        reaction_time, nan=max_lidar_aoi_s, posinf=max_lidar_aoi_s, neginf=0.0
    ).clamp_min(0.0)
    aoi = torch.nan_to_num(
        shield_aoi.to(dtype=wheel_targets.dtype),
        nan=max_lidar_aoi_s + 1.0,
        posinf=max_lidar_aoi_s + 1.0,
        neginf=0.0,
    ).clamp_min(0.0)
    safe_radius_scale = None
    if wheel_radius_scale is not None:
        radius_scale = _as_batch_tensor(
            wheel_radius_scale, batch_reference, "wheel_radius_scale"
        )
        finite_inputs &= torch.isfinite(radius_scale) & (radius_scale > 0.0)
        safe_radius_scale = torch.where(
            torch.isfinite(radius_scale) & (radius_scale > 0.0),
            radius_scale,
            torch.ones_like(radius_scale),
        )

    # S1: scale both wheels so a_lat = |v * omega| stays within the limit.
    raw_twist = _wheel_velocities_to_twist(
        candidate_wheels, wheel_radius, wheel_base, safe_radius_scale
    )
    lateral_accel = (raw_twist[:, 0] * raw_twist[:, 1]).abs()
    if enabled:
        curve_scale = torch.sqrt(
            candidate_wheels.new_tensor(lateral_accel_limit)
            / lateral_accel.clamp_min(1.0e-8)
        ).clamp(max=1.0)
    else:
        curve_scale = torch.ones_like(lateral_accel)
    curve_safe_wheels = candidate_wheels * curve_scale.unsqueeze(-1)
    curve_twist = _wheel_velocities_to_twist(
        curve_safe_wheels, wheel_radius, wheel_base, safe_radius_scale
    )
    speed = curve_twist[:, 0].abs()

    # Select clearance in the direction of travel. In-place rotation has zero
    # translational braking demand; the stale/invalid fail-safe still applies.
    cos_angles = torch.cos(angles)
    front_mask = cos_angles >= math.cos(math.radians(forward_half_angle_deg))
    rear_mask = cos_angles <= -math.cos(math.radians(rear_half_angle_deg))
    inf = torch.full_like(clearance, float("inf"))
    front_clearance = torch.where(front_mask, clearance, inf).amin(dim=-1)
    rear_clearance = torch.where(rear_mask, clearance, inf).amin(dim=-1)
    travel_clearance = torch.where(
        curve_twist[:, 0] < 0.0, rear_clearance, front_clearance
    )

    obstacle_speed = candidate_wheels.new_tensor(possible_obstacle_speed)
    d_now_lower = travel_clearance - static_margin - obstacle_speed * aoi

    # S2: dynamic-obstacle braking envelope. The obstacle is allowed to move
    # toward the robot during both future reaction time and robot braking.
    decel = candidate_wheels.new_tensor(braking_deceleration)
    b_term = reaction_time + obstacle_speed / decel
    d_budget = d_now_lower - obstacle_speed * reaction_time
    safe_speed_braking = decel * (
        torch.sqrt(b_term.square() + 2.0 * d_budget.clamp_min(0.0) / decel)
        - b_term
    )
    safe_speed_braking = safe_speed_braking.clamp_min(0.0)

    # Also keep the current conservative TTC above the slow threshold.
    safe_speed_ttc = (
        d_now_lower.clamp_min(0.0) / ttc_slow_s - obstacle_speed
    ).clamp_min(0.0)
    safe_speed = torch.minimum(safe_speed_braking, safe_speed_ttc)
    if enabled:
        brake_scale = torch.where(
            speed > 1.0e-8,
            (safe_speed / speed.clamp_min(1.0e-8)).clamp(max=1.0),
            torch.ones_like(speed),
        )
    else:
        brake_scale = torch.ones_like(speed)
    brake_safe_wheels = curve_safe_wheels * brake_scale.unsqueeze(-1)

    # S3: fail closed on emergency TTC, exhausted clearance, stale data, or
    # absence of any delivered scan. An emergency stop bypasses smooth scaling.
    closing_speed_bound = speed + obstacle_speed
    min_ttc = torch.where(
        closing_speed_bound > 1.0e-8,
        d_now_lower / closing_speed_bound,
        torch.full_like(closing_speed_bound, float("inf")),
    )
    emergency = torch.zeros_like(speed, dtype=torch.bool)
    if enabled:
        emergency |= (min_ttc < ttc_stop_s) | (d_now_lower <= 0.0)
        emergency |= aoi > max_lidar_aoi_s
        emergency |= ~finite_inputs
    if enabled and has_valid_scan is not None:
        if has_valid_scan.shape != (wheel_targets.shape[0],):
            raise ValueError("has_valid_scan must have shape (N,)")
        emergency |= ~has_valid_scan.to(dtype=torch.bool)

    safe_wheels = torch.where(
        emergency.unsqueeze(-1), torch.zeros_like(brake_safe_wheels), brake_safe_wheels
    )
    total_scale = curve_scale * brake_scale
    total_scale = torch.where(emergency, torch.zeros_like(total_scale), total_scale)
    intervention_amount = (1.0 - total_scale).clamp(0.0, 1.0)

    diagnostics = {
        "raw_twist": raw_twist,
        "safe_twist": _wheel_velocities_to_twist(
            safe_wheels, wheel_radius, wheel_base, safe_radius_scale
        ),
        "lateral_accel": lateral_accel,
        "curve_scale": curve_scale,
        "brake_scale": brake_scale,
        "total_scale": total_scale,
        "safe_speed": safe_speed,
        "min_ttc": min_ttc,
        "travel_clearance": travel_clearance,
        "d_now_lower": d_now_lower,
        "intervention_amount": intervention_amount,
        "shield_intervened": intervention_amount > 1.0e-6,
        "emergency_stopped": emergency,
    }
    return safe_wheels, diagnostics
