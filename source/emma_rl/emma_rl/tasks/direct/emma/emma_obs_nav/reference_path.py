"""Vectorized reference-polyline projection for privileged training rewards.

The path is used only by the simulator reward. It is deliberately absent from
policy observations so exported policies keep the deployment input contract.
"""

from __future__ import annotations

import torch
import math


def project_points_to_polylines(
    points: torch.Tensor,
    waypoints: torch.Tensor,
    point_counts: torch.Tensor,
    *,
    return_tangent: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Return arc-length progress and cross-track error for padded polylines.

    Args:
        points: ``(N, 2)`` query positions.
        waypoints: ``(N, P, 2)`` padded polyline points.
        point_counts: number of valid points in each row; values below two mark
            an unavailable reference path.
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if waypoints.ndim != 3 or waypoints.shape[0] != points.shape[0] or waypoints.shape[-1] != 2:
        raise ValueError("waypoints must have shape (N, P, 2)")
    if point_counts.shape != (points.shape[0],):
        raise ValueError("point_counts must have shape (N,)")
    if waypoints.shape[1] < 2:
        raise ValueError("waypoints must provide at least two padded points")

    starts = waypoints[:, :-1]
    segments = waypoints[:, 1:] - starts
    length_sq = segments.square().sum(dim=-1)
    lengths = torch.sqrt(length_sq.clamp_min(torch.finfo(points.dtype).eps))
    segment_ids = torch.arange(
        segments.shape[1], device=points.device
    ).unsqueeze(0)
    segment_valid = segment_ids < (point_counts - 1).clamp_min(0).unsqueeze(1)
    segment_valid &= length_sq > torch.finfo(points.dtype).eps

    relative = points.unsqueeze(1) - starts
    fraction = (relative * segments).sum(dim=-1) / length_sq.clamp_min(
        torch.finfo(points.dtype).eps
    )
    fraction = fraction.clamp(0.0, 1.0)
    projections = starts + fraction.unsqueeze(-1) * segments
    distance_sq = (points.unsqueeze(1) - projections).square().sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~segment_valid, torch.inf)

    nearest = distance_sq.argmin(dim=1)
    prefix = torch.cumsum(lengths * segment_valid, dim=1) - lengths * segment_valid
    progress_candidates = prefix + fraction * lengths
    progress = progress_candidates.gather(1, nearest.unsqueeze(1)).squeeze(1)
    cross_track = torch.sqrt(
        distance_sq.gather(1, nearest.unsqueeze(1)).squeeze(1).clamp_min(0.0)
    )

    valid = point_counts >= 2
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    cross_track = torch.where(valid, cross_track, torch.zeros_like(cross_track))
    if return_tangent:
        tangent = _selected_unit_tangent(segments,nearest,segment_valid.any(-1))
        return progress,cross_track,tangent
    return progress, cross_track


def project_continuous_progress(points, previous_points, waypoints, point_counts,
                                previous_progress, gates, gate_counts, passed_gates,
                                yaw, forward_sign, *, half_length, half_width,
                                collision_margin, max_step=.10, return_tangent=False):
    """Project only onto locally reachable arc length; unlock gates by crossing.

    Paths are ordered start->goal. gates[..., :] = (x, y, half_gap, path_arc).
    Rotation produces no translation reward. Backtracking remains signed.
    Large jumps (reset/teleport/contact impulses) do not unlock a future gate.
    """
    starts = waypoints[:, :-1]
    segments = waypoints[:, 1:] - starts
    squared = segments.square().sum(-1)
    lengths = squared.sqrt()
    ids = torch.arange(segments.shape[1], device=points.device)[None]
    valid = (ids < (point_counts-1).clamp_min(0)[:, None]) & (squared > 1e-10)
    lengths = lengths * valid
    prefix = torch.cumsum(lengths, -1) - lengths
    displacement = torch.linalg.vector_norm(points-previous_points, dim=-1)
    normal_step = displacement <= max_step
    window = (2*displacement).clamp_max(max_step)
    window = torch.where(normal_step, window, torch.zeros_like(window))
    lo = (previous_progress-window).clamp_min(0)[:, None]
    hi = (previous_progress+window)[:, None]
    valid &= (prefix <= hi) & (prefix+lengths >= lo)
    fraction = ((points[:, None]-starts)*segments).sum(-1) / squared.clamp_min(1e-10)
    fraction = torch.maximum(fraction, ((lo-prefix)/lengths.clamp_min(1e-10)).clamp(0,1))
    fraction = torch.minimum(fraction, ((hi-prefix)/lengths.clamp_min(1e-10)).clamp(0,1))
    nearest_points = starts + fraction[..., None]*segments
    distances = (points[:, None]-nearest_points).square().sum(-1).masked_fill(~valid, torch.inf)
    nearest = distances.argmin(-1)
    progress = (prefix+fraction*lengths).gather(1, nearest[:, None]).squeeze(1)
    error = distances.gather(1, nearest[:, None]).squeeze(1).sqrt()
    any_valid = valid.any(-1)
    progress = torch.where(any_valid, progress, previous_progress)
    error = torch.where(any_valid, error, torch.zeros_like(error))

    next_id = passed_gates.clamp(max=gates.shape[1]-1)
    next_gate = gates.gather(1, next_id[:, None, None].expand(-1,1,4)).squeeze(1)
    before = (previous_points[:, 1]-next_gate[:, 1])*forward_sign
    after = (points[:, 1]-next_gate[:, 1])*forward_sign
    dy = points[:, 1]-previous_points[:, 1]
    ratio = torch.where(dy.abs() > 1e-8, (next_gate[:, 1]-previous_points[:, 1])/dy,
                        torch.zeros_like(dy)).clamp(0,1)
    crossing_x = previous_points[:, 0]+ratio*(points[:, 0]-previous_points[:, 0])
    body_extent = half_length*yaw.cos().abs()+half_width*yaw.sin().abs()+collision_margin
    crossed = ((passed_gates < gate_counts) & (before <= 0) & (after > 0)
               & ((crossing_x-next_gate[:, 0]).abs()+body_extent <= next_gate[:, 2])
               & normal_step)
    passed = passed_gates + crossed.long()
    next_id = passed.clamp(max=gates.shape[1]-1)
    limit = gates[:, :, 3].gather(1, next_id[:, None]).squeeze(1)
    progress = torch.where(passed < gate_counts, torch.minimum(progress, limit), progress)
    progress = torch.where(point_counts >= 2, progress, torch.zeros_like(progress))
    if return_tangent:
        tangent = _selected_unit_tangent(segments,nearest,any_valid & normal_step)
        return progress,error,passed,tangent
    return progress, error, passed


def _selected_unit_tangent(segments, nearest, valid):
    """Use the same segment as progress projection; never a global nearest branch."""
    tangent = segments.gather(1,nearest[:,None,None].expand(-1,1,2)).squeeze(1)
    tangent = tangent/torch.linalg.vector_norm(tangent,dim=-1,keepdim=True).clamp_min(1e-8)
    return torch.where(valid[:,None],tangent,torch.zeros_like(tangent))


def reference_velocity_direction_cost(velocity, tangent, valid, *, step_dt,
                                      cost_per_second, min_speed, full_speed,
                                      tolerance_deg):
    """Complement signed progress with a bounded lateral-velocity penalty.

    Progress already rewards forward / penalizes backward travel. The new
    term penalizes sin(angle)^2, avoiding a second backward cost or forcing
    vehicle heading to equal travel direction. Low speeds smoothly fade out.
    Both vectors must describe the same root-link point in the world frame.
    Returns (cost, cosine, speed_weight) for reward and diagnostics.
    """
    speed = torch.linalg.vector_norm(velocity,dim=-1)
    tangent_norm = torch.linalg.vector_norm(tangent,dim=-1)
    usable = (valid & torch.isfinite(velocity).all(-1) & torch.isfinite(tangent).all(-1)
              & (tangent_norm > .5) & (speed > min_speed))
    cosine = (velocity*tangent).sum(-1)/(speed*tangent_norm).clamp_min(1e-8)
    cosine = torch.where(usable,cosine.clamp(-1.,1.),torch.zeros_like(speed))
    weight = ((speed-min_speed)/(full_speed-min_speed)).clamp(0.,1.)
    weight = torch.where(usable,weight,torch.zeros_like(weight))
    tolerance_sq = math.sin(math.radians(tolerance_deg))**2
    lateral_sq = (1.-cosine.square()).clamp(0.,1.)
    mismatch = ((lateral_sq-tolerance_sq)/(1.-tolerance_sq)).clamp(0.,1.)
    cost = -cost_per_second*step_dt*weight*mismatch
    return torch.where(usable,cost,torch.zeros_like(cost)),cosine,weight
