"""Pure Torch utilities for asynchronous 2D LiDAR scheduling and SE(2) warping."""

from __future__ import annotations

import math

import torch


def bpearl_planar_ray_angles(
    num_rays: int,
    *,
    horizontal_fov_deg: float,
    horizontal_resolution_deg: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return Bpearl's actual planar directions in raw range order.

    Isaac Lab constructs internal yaw ``h`` from ``-FOV/2`` and negates the
    final Cartesian direction, so physical planar yaw is ``h + 180°``.
    """

    if num_rays < 1:
        raise ValueError("num_rays must be positive")
    if (
        not math.isfinite(horizontal_fov_deg)
        or horizontal_fov_deg <= 0.0
        or not math.isfinite(horizontal_resolution_deg)
        or horizontal_resolution_deg <= 0.0
    ):
        raise ValueError("Bpearl FOV and resolution must be finite and positive")
    internal_yaw_deg = (
        torch.arange(num_rays, device=device) * horizontal_resolution_deg
        - horizontal_fov_deg / 2.0
    )
    physical_yaw = torch.deg2rad(internal_yaw_deg + 180.0)
    return torch.atan2(torch.sin(physical_yaw), torch.cos(physical_yaw))


def sample_sensor_periods(
    num_envs: int,
    frequencies_hz: tuple[float, ...] | list[float],
    device: str | torch.device,
) -> torch.Tensor:
    """Sample one sensor period independently for each environment."""

    frequencies = torch.as_tensor(frequencies_hz, dtype=torch.float32, device=device)
    if frequencies.numel() == 0 or torch.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain positive values")
    indices = torch.randint(0, frequencies.numel(), (num_envs,), device=device)
    return frequencies[indices].reciprocal()


def sensor_due_mask(current_time: torch.Tensor, next_timestamp: torch.Tensor) -> torch.Tensor:
    """Return the per-environment sensor generation mask."""

    return current_time + 1.0e-7 >= next_timestamp


def advance_sensor_schedule(
    current_time: torch.Tensor,
    next_timestamp: torch.Tensor,
    sensor_period: torch.Tensor,
    due_mask: torch.Tensor,
    control_dt: float,
    jitter_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance due sensors to a strictly future generation timestamp.

    ``jitter_steps`` is passed explicitly so scheduling remains deterministic in
    unit tests. The environment samples it independently for each due sensor.
    """

    updated = next_timestamp.clone()
    if not torch.any(due_mask):
        return updated

    jitter = torch.zeros_like(sensor_period)
    if jitter_steps is not None:
        jitter = jitter_steps.to(sensor_period.dtype) * control_dt
    candidate = next_timestamp + sensor_period + jitter
    minimum_future = current_time + control_dt
    updated[due_mask] = torch.maximum(candidate[due_mask], minimum_future[due_mask])
    return updated


def compute_aoi(current_time: torch.Tensor, cached_timestamp: torch.Tensor) -> torch.Tensor:
    """Compute non-negative Age of Information in seconds."""

    return (current_time - cached_timestamp).clamp_min(0.0)


def sinusoidal_aoi_encoding(aoi: torch.Tensor, encoding_dim: int, scale_s: float) -> torch.Tensor:
    """Encode actual AoI seconds using a continuous sinusoidal encoding."""

    if encoding_dim <= 0:
        return aoi.new_zeros((aoi.shape[0], 0))
    if scale_s <= 0:
        raise ValueError("scale_s must be positive")

    position = (aoi / scale_s).unsqueeze(-1)
    even_indices = torch.arange(0, encoding_dim, 2, device=aoi.device, dtype=aoi.dtype)
    div_term = torch.exp(even_indices * (-math.log(10000.0) / encoding_dim))
    encoded = aoi.new_zeros((aoi.shape[0], encoding_dim))
    encoded[:, 0::2] = torch.sin(position * div_term)
    if encoding_dim > 1:
        encoded[:, 1::2] = torch.cos(position * div_term[: encoded[:, 1::2].shape[1]])
    return encoded


def smooth_clearance_gate(
    clearance: torch.Tensor,
    stop_clearance: float,
    full_clearance: float,
) -> torch.Tensor:
    """Smoothstep gate: zero at ``stop_clearance`` and one at ``full_clearance``."""

    if full_clearance <= stop_clearance:
        raise ValueError("full_clearance must exceed stop_clearance")
    value = ((clearance - stop_clearance) / (full_clearance - stop_clearance)).clamp(0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


def fill_clearance_outside_forward_fov(
    clearance: torch.Tensor,
    sector_angles: torch.Tensor,
    forward_half_fov_deg: float,
    fill_clearance: float,
) -> torch.Tensor:
    """Fill policy-only blind sectors while preserving the observation shape.

    Angles are expressed in the robot body frame, where zero points forward.
    A 110-degree half-FOV therefore keeps the real forward 220 degrees and
    fills the remaining rear sectors. This helper does not alter the clean
    clearance used by collision detection or runtime safety.
    """

    if not math.isfinite(forward_half_fov_deg) or not 0.0 < forward_half_fov_deg <= 180.0:
        raise ValueError("forward_half_fov_deg must be in (0, 180]")
    if not math.isfinite(fill_clearance) or fill_clearance < 0.0:
        raise ValueError("fill_clearance must be finite and non-negative")
    if clearance.shape[-1] != sector_angles.shape[-1]:
        raise ValueError("clearance and sector_angles must have the same final dimension")

    half_fov = math.radians(forward_half_fov_deg)
    wrapped_angles = torch.atan2(torch.sin(sector_angles), torch.cos(sector_angles))
    visible = wrapped_angles.abs() <= half_fov + 1.0e-7
    fill = torch.as_tensor(
        fill_clearance, device=clearance.device, dtype=clearance.dtype
    )
    return torch.where(visible, clearance, fill)


def ranges_to_points(ranges: torch.Tensor, ray_angles: torch.Tensor) -> torch.Tensor:
    """Restore planar LiDAR ranges to points in the scan frame."""

    angles = ray_angles.unsqueeze(0) if ray_angles.ndim == 1 else ray_angles
    return torch.stack((ranges * torch.cos(angles), ranges * torch.sin(angles)), dim=-1)


def min_pool_clearance_rays(
    ranges: torch.Tensor,
    ray_angles: torch.Tensor,
    footprint_projection: torch.Tensor,
    num_sectors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the minimum-clearance ray, rather than minimum range, per sector."""

    if ranges.shape != ray_angles.shape or ranges.shape != footprint_projection.shape:
        raise ValueError("ranges, ray_angles, and footprint_projection must have identical shapes")
    if ranges.shape[1] % num_sectors != 0:
        raise ValueError("the ray count must be divisible by num_sectors")
    rays_per_sector = ranges.shape[1] // num_sectors
    grouped_ranges = ranges.reshape(ranges.shape[0], num_sectors, rays_per_sector)
    grouped_angles = ray_angles.reshape(ranges.shape[0], num_sectors, rays_per_sector)
    grouped_projection = footprint_projection.reshape(
        ranges.shape[0], num_sectors, rays_per_sector)
    indices = (grouped_ranges - grouped_projection).argmin(dim=-1, keepdim=True)
    selected_ranges = torch.gather(grouped_ranges, -1, indices).squeeze(-1)
    selected_angles = torch.gather(grouped_angles, -1, indices).squeeze(-1)
    selected_clearance = torch.gather(
        grouped_ranges - grouped_projection, -1, indices).squeeze(-1)
    return selected_ranges, selected_angles, selected_clearance


def min_pool_clearance_rays_grouped(
    ranges: torch.Tensor,
    ray_angles: torch.Tensor,
    footprint_projection: torch.Tensor,
    sector_groups: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-pool consecutive rays with piecewise-constant sector widths.

    Each ``(sector_count, rays_per_sector)`` group describes a contiguous
    angular region. This keeps the fast reshape/gather implementation used by
    uniform sectors while allowing the policy to use finer front sectors.
    """

    if ranges.shape != ray_angles.shape or ranges.shape != footprint_projection.shape:
        raise ValueError(
            "ranges, ray_angles, and footprint_projection must have identical shapes"
        )
    if not sector_groups:
        raise ValueError("sector_groups must not be empty")
    expected_rays = sum(
        int(sector_count) * int(rays_per_sector)
        for sector_count, rays_per_sector in sector_groups
    )
    if expected_rays != ranges.shape[1]:
        raise ValueError(
            f"sector_groups describe {expected_rays} rays, got {ranges.shape[1]}"
        )

    range_parts: list[torch.Tensor] = []
    angle_parts: list[torch.Tensor] = []
    clearance_parts: list[torch.Tensor] = []
    ray_offset = 0
    for sector_count, rays_per_sector in sector_groups:
        if sector_count < 1 or rays_per_sector < 1:
            raise ValueError("sector counts and rays per sector must be positive")
        ray_count = sector_count * rays_per_sector
        group_slice = slice(ray_offset, ray_offset + ray_count)
        grouped_ranges = ranges[:, group_slice].reshape(
            ranges.shape[0], sector_count, rays_per_sector
        )
        grouped_angles = ray_angles[:, group_slice].reshape(
            ranges.shape[0], sector_count, rays_per_sector
        )
        grouped_projection = footprint_projection[:, group_slice].reshape(
            ranges.shape[0], sector_count, rays_per_sector
        )
        grouped_clearance = grouped_ranges - grouped_projection
        indices = grouped_clearance.argmin(dim=-1, keepdim=True)
        range_parts.append(torch.gather(grouped_ranges, -1, indices).squeeze(-1))
        angle_parts.append(torch.gather(grouped_angles, -1, indices).squeeze(-1))
        clearance_parts.append(
            torch.gather(grouped_clearance, -1, indices).squeeze(-1)
        )
        ray_offset += ray_count

    return (
        torch.cat(range_parts, dim=-1),
        torch.cat(angle_parts, dim=-1),
        torch.cat(clearance_parts, dim=-1),
    )


def hybrid_180_sector_edges(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return edges for 1-degree front and 3-degree elsewhere (180 sectors)."""

    degree_edges = torch.cat(
        (
            torch.arange(0.0, 46.0, 1.0, device=device, dtype=dtype),
            torch.arange(48.0, 316.0, 3.0, device=device, dtype=dtype),
            torch.arange(316.0, 361.0, 1.0, device=device, dtype=dtype),
        )
    )
    if degree_edges.numel() != 181:
        raise RuntimeError("hybrid LiDAR layout must contain 180 sectors")
    return torch.deg2rad(degree_edges)


def sector_centers_from_edges(sector_edges: torch.Tensor) -> torch.Tensor:
    """Return angular centers for monotonically increasing [0, 2pi] edges."""

    if sector_edges.ndim != 1 or sector_edges.numel() < 2:
        raise ValueError("sector_edges must be a one-dimensional edge vector")
    if torch.any(sector_edges[1:] <= sector_edges[:-1]):
        raise ValueError("sector_edges must be strictly increasing")
    return (sector_edges[:-1] + sector_edges[1:]) * 0.5


def transform_points_se2(
    points_source: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
) -> torch.Tensor:
    """Transform points from ``source_pose`` frame into ``target_pose`` frame.

    Poses use ``(x, y, yaw)`` in a shared odometry/world frame.
    """

    source_yaw = source_pose[:, 2]
    cs, ss = torch.cos(source_yaw), torch.sin(source_yaw)
    x_world = (
        points_source[..., 0] * cs.unsqueeze(-1)
        - points_source[..., 1] * ss.unsqueeze(-1)
        + source_pose[:, 0].unsqueeze(-1)
    )
    y_world = (
        points_source[..., 0] * ss.unsqueeze(-1)
        + points_source[..., 1] * cs.unsqueeze(-1)
        + source_pose[:, 1].unsqueeze(-1)
    )

    dx = x_world - target_pose[:, 0].unsqueeze(-1)
    dy = y_world - target_pose[:, 1].unsqueeze(-1)
    target_yaw = target_pose[:, 2]
    ct, st = torch.cos(target_yaw), torch.sin(target_yaw)
    x_target = dx * ct.unsqueeze(-1) + dy * st.unsqueeze(-1)
    y_target = -dx * st.unsqueeze(-1) + dy * ct.unsqueeze(-1)
    return torch.stack((x_target, y_target), dim=-1)


def points_to_ranges(
    points: torch.Tensor,
    num_sectors: int,
    max_range: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project planar points into equally spaced 360-degree range sectors."""

    ranges = torch.linalg.norm(points, dim=-1)
    angles = torch.remainder(torch.atan2(points[..., 1], points[..., 0]), 2.0 * math.pi)
    sector_width = 2.0 * math.pi / num_sectors
    indices = torch.floor(angles / sector_width).long().clamp_(0, num_sectors - 1)

    valid = (ranges > 1.0e-6) & (ranges < max_range)
    if valid_mask is not None:
        valid &= valid_mask
    values = torch.where(valid, ranges, torch.full_like(ranges, max_range))
    output = torch.full(
        (points.shape[0], num_sectors),
        max_range,
        dtype=points.dtype,
        device=points.device,
    )
    output.scatter_reduce_(1, indices, values, reduce="amin", include_self=True)
    return output


def rectangular_footprint_projection(
    ray_angles: torch.Tensor,
    half_length: float,
    half_width: float,
    sensor_offset_x: float,
    sensor_offset_y: float,
) -> torch.Tensor:
    """Distance from an in-footprint sensor to a rectangular footprint edge."""

    if half_length <= 0 or half_width <= 0:
        raise ValueError("half_length and half_width must be positive")
    if abs(sensor_offset_x) >= half_length or abs(sensor_offset_y) >= half_width:
        raise ValueError("the planar LiDAR origin must lie inside the rectangular footprint")

    cos_a = torch.cos(ray_angles)
    sin_a = torch.sin(ray_angles)
    inf = torch.full_like(ray_angles, float("inf"))
    eps = 1.0e-7
    front = torch.where(cos_a > eps, (half_length - sensor_offset_x) / cos_a, inf)
    rear = torch.where(cos_a < -eps, (-half_length - sensor_offset_x) / cos_a, inf)
    left = torch.where(sin_a > eps, (half_width - sensor_offset_y) / sin_a, inf)
    right = torch.where(sin_a < -eps, (-half_width - sensor_offset_y) / sin_a, inf)
    return torch.stack((front, rear, left, right), dim=-1).amin(dim=-1)


def points_to_ranges_and_clearances(
    points: torch.Tensor,
    num_sectors: int,
    max_range: float,
    half_length: float,
    half_width: float,
    sensor_offset_x: float,
    sensor_offset_y: float,
    valid_mask: torch.Tensor | None = None,
    sector_edges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project points and aggregate range and footprint clearance per sector.

    Clearance is computed for every transformed point before the sector-wise
    minimum. This avoids subtracting a sector-center footprint projection from
    a range that was measured by a different ray.
    """

    point_ranges = torch.linalg.norm(points, dim=-1)
    point_angles = torch.remainder(
        torch.atan2(points[..., 1], points[..., 0]), 2.0 * math.pi)
    if sector_edges is None:
        sector_width = 2.0 * math.pi / num_sectors
        indices = torch.floor(point_angles / sector_width).long().clamp_(
            0, num_sectors - 1
        )
        sector_centers = (
            torch.arange(num_sectors, device=points.device, dtype=points.dtype) + 0.5
        ) * sector_width
    else:
        sector_edges = sector_edges.to(device=points.device, dtype=points.dtype)
        if sector_edges.ndim != 1 or sector_edges.numel() != num_sectors + 1:
            raise ValueError("sector_edges must contain num_sectors + 1 values")
        indices = torch.bucketize(
            point_angles.contiguous(), sector_edges[1:-1].contiguous(), right=True
        ).clamp_(0, num_sectors - 1)
        sector_centers = (sector_edges[:-1] + sector_edges[1:]) * 0.5

    valid = (point_ranges > 1.0e-6) & (point_ranges < max_range)
    if valid_mask is not None:
        valid &= valid_mask

    default_clearance = max_range - rectangular_footprint_projection(
        sector_centers,
        half_length,
        half_width,
        sensor_offset_x,
        sensor_offset_y,
    )
    output_ranges = torch.full(
        (points.shape[0], num_sectors),
        max_range,
        dtype=points.dtype,
        device=points.device,
    )
    output_clearances = default_clearance.unsqueeze(0).repeat(points.shape[0], 1)

    range_values = torch.where(valid, point_ranges, torch.full_like(point_ranges, max_range))
    point_projection = rectangular_footprint_projection(
        point_angles,
        half_length,
        half_width,
        sensor_offset_x,
        sensor_offset_y,
    )
    clearance_values = torch.where(
        valid,
        point_ranges - point_projection,
        torch.full_like(point_ranges, float("inf")),
    )
    output_ranges.scatter_reduce_(1, indices, range_values, reduce="amin", include_self=True)
    output_clearances.scatter_reduce_(
        1, indices, clearance_values, reduce="amin", include_self=True)
    return output_ranges, output_clearances


def warp_cached_ranges(
    cached_ranges: torch.Tensor,
    cached_pose: torch.Tensor,
    current_pose: torch.Tensor,
    ray_angles: torch.Tensor,
    max_range: float,
) -> torch.Tensor:
    """Warp an immutable cached range scan to the current sensor frame."""

    valid = cached_ranges < max_range
    cached_points = ranges_to_points(cached_ranges, ray_angles)
    current_points = transform_points_se2(cached_points, cached_pose, current_pose)
    return points_to_ranges(current_points, cached_ranges.shape[1], max_range, valid)


def warp_cached_ranges_and_clearances(
    cached_ranges: torch.Tensor,
    cached_pose: torch.Tensor,
    current_pose: torch.Tensor,
    ray_angles: torch.Tensor,
    num_sectors: int,
    max_range: float,
    half_length: float,
    half_width: float,
    sensor_offset_x: float,
    sensor_offset_y: float,
    sector_edges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp cached scan endpoints and aggregate current-frame clearances."""

    valid = cached_ranges < max_range
    cached_points = ranges_to_points(cached_ranges, ray_angles)
    current_points = transform_points_se2(cached_points, cached_pose, current_pose)
    return points_to_ranges_and_clearances(
        current_points,
        num_sectors,
        max_range,
        half_length,
        half_width,
        sensor_offset_x,
        sensor_offset_y,
        valid,
        sector_edges,
    )
