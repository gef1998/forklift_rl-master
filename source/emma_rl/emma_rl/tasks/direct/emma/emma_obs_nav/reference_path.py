"""Vectorized reference-polyline projection for privileged training rewards.

The path is used only by the simulator reward. It is deliberately absent from
policy observations so exported policies keep the deployment input contract.
"""

from __future__ import annotations

import torch


def project_points_to_polylines(
    points: torch.Tensor,
    waypoints: torch.Tensor,
    point_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return progress, cross_track
