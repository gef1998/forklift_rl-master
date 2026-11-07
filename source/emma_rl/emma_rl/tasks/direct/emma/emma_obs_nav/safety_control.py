"""Forward-clearance speed governor for the Emma differential drive."""

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


def _effective_radius(
    reference: torch.Tensor,
    wheel_radius: float,
    wheel_radius_scale: torch.Tensor | None,
) -> torch.Tensor:
    radius = reference.new_full((reference.shape[0],), wheel_radius)
    if wheel_radius_scale is not None:
        radius = radius * _as_batch_tensor(
            wheel_radius_scale, radius, "wheel_radius_scale"
        )
    return radius


def _wheel_velocities_to_twist(
    wheel_velocities: torch.Tensor,
    wheel_radius: float,
    track_width: float,
    wheel_radius_scale: torch.Tensor | None,
) -> torch.Tensor:
    radius = _effective_radius(wheel_velocities, wheel_radius, wheel_radius_scale)
    left = wheel_velocities[:, 0]
    right = wheel_velocities[:, 1]
    linear = 0.5 * radius * (left + right)
    angular = radius * (right - left) / track_width
    return torch.stack((linear, angular), dim=-1)


def _twist_to_feasible_wheels(
    twist: torch.Tensor,
    wheel_radius: float,
    track_width: float,
    wheel_max_velocity: float,
    wheel_radius_scale: torch.Tensor | None,
) -> torch.Tensor:
    radius = _effective_radius(twist, wheel_radius, wheel_radius_scale)
    half_track = track_width / 2.0
    wheels = torch.stack(
        (
            (twist[:, 0] - twist[:, 1] * half_track) / radius,
            (twist[:, 0] + twist[:, 1] * half_track) / radius,
        ),
        dim=-1,
    )
    peak = wheels.abs().amax(dim=-1)
    projection = (wheel_max_velocity / peak.clamp_min(1.0e-8)).clamp(max=1.0)
    return wheels * projection.unsqueeze(-1)


def apply_safety_control(
    wheel_targets: torch.Tensor,
    shield_clearance: torch.Tensor,
    sector_angles: torch.Tensor,
    shield_aoi: torch.Tensor,
    *,
    wheel_radius: float,
    track_width: float,
    wheel_max_velocity: float,
    forward_half_angle_deg: float,
    stop_clearance: float,
    full_speed_clearance: float,
    min_forward_speed_scale: float,
    max_lidar_aoi_s: float,
    wheel_radius_scale: torch.Tensor | None = None,
    has_valid_scan: torch.Tensor | None = None,
    enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reduce only positive body velocity as forward clearance decreases.

    ``shield_clearance`` is distance outside the rectangular robot footprint.
    Positive velocity uses ``min_forward_speed_scale`` at and below
    ``stop_clearance``, is unchanged at ``full_speed_clearance``, and follows a
    shifted smoothstep between them. Angular and reverse commands remain
    available so the policy can turn away or recover. An unusable scan still
    blocks positive velocity instead of applying the configured floor.
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

    positive = {
        "wheel_radius": wheel_radius,
        "track_width": track_width,
        "wheel_max_velocity": wheel_max_velocity,
        "forward_half_angle_deg": forward_half_angle_deg,
        "full_speed_clearance": full_speed_clearance,
        "max_lidar_aoi_s": max_lidar_aoi_s,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(stop_clearance) or stop_clearance < 0:
        raise ValueError("stop_clearance must be finite and non-negative")
    if full_speed_clearance <= stop_clearance:
        raise ValueError("full_speed_clearance must exceed stop_clearance")
    if (
        not math.isfinite(min_forward_speed_scale)
        or min_forward_speed_scale < 0
        or min_forward_speed_scale >= 1
    ):
        raise ValueError("min_forward_speed_scale must be in [0, 1)")
    if forward_half_angle_deg > 90:
        raise ValueError("forward_half_angle_deg must not exceed 90")

    finite_wheels = torch.isfinite(wheel_targets).all(dim=-1)
    candidate_wheels = torch.where(
        torch.isfinite(wheel_targets), wheel_targets, torch.zeros_like(wheel_targets)
    )
    raw_twist = _wheel_velocities_to_twist(
        candidate_wheels, wheel_radius, track_width, wheel_radius_scale
    )

    finite_sectors = torch.isfinite(shield_clearance) & torch.isfinite(angles)
    safe_angles = torch.where(torch.isfinite(angles), angles, torch.zeros_like(angles))
    front_mask = torch.cos(safe_angles) >= math.cos(
        math.radians(forward_half_angle_deg)
    )
    usable_front = front_mask & finite_sectors
    inf = torch.full_like(shield_clearance, float("inf"))
    front_clearance = torch.where(
        usable_front, shield_clearance, inf
    ).amin(dim=-1)
    front_covered = usable_front.any(dim=-1)

    aoi = torch.nan_to_num(
        shield_aoi.to(dtype=wheel_targets.dtype),
        nan=max_lidar_aoi_s + 1.0,
        posinf=max_lidar_aoi_s + 1.0,
        neginf=0.0,
    ).clamp_min(0.0)
    scan_usable = finite_wheels & front_covered & (aoi <= max_lidar_aoi_s)
    if has_valid_scan is not None:
        if has_valid_scan.shape != (wheel_targets.shape[0],):
            raise ValueError("has_valid_scan must have shape (N,)")
        scan_usable &= has_valid_scan.to(dtype=torch.bool)

    normalized = (
        (front_clearance - stop_clearance)
        / (full_speed_clearance - stop_clearance)
    ).clamp(0.0, 1.0)
    smoothstep = normalized.square() * (3.0 - 2.0 * normalized)
    clearance_scale = min_forward_speed_scale + (
        1.0 - min_forward_speed_scale
    ) * smoothstep
    forward_motion = raw_twist[:, 0] > 1.0e-8
    if enabled:
        forward_scale = torch.where(
            forward_motion & scan_usable,
            clearance_scale,
            torch.where(
                forward_motion,
                torch.zeros_like(clearance_scale),
                torch.ones_like(clearance_scale),
            ),
        )
    else:
        forward_scale = torch.ones_like(clearance_scale)

    safe_twist_target = torch.stack(
        (raw_twist[:, 0] * forward_scale, raw_twist[:, 1]), dim=-1
    )
    safe_wheels = _twist_to_feasible_wheels(
        safe_twist_target,
        wheel_radius,
        track_width,
        wheel_max_velocity,
        wheel_radius_scale,
    )
    safe_wheels = torch.where(
        finite_wheels.unsqueeze(-1), safe_wheels, torch.zeros_like(safe_wheels)
    )
    safe_twist = _wheel_velocities_to_twist(
        safe_wheels, wheel_radius, track_width, wheel_radius_scale
    )
    intervention = enabled & forward_motion & (forward_scale < 1.0 - 1.0e-6)

    diagnostics = {
        "raw_twist": raw_twist,
        "safe_twist": safe_twist,
        "forward_speed_scale": forward_scale,
        "forward_clearance": front_clearance,
        "safe_speed": safe_twist[:, 0].abs(),
        "shield_intervened": intervention,
        "emergency_stopped": ~finite_wheels,
    }
    return safe_wheels, diagnostics
