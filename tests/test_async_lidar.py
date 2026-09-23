#!/usr/bin/env python3
"""Unit tests for asynchronous LiDAR geometry and scheduling without Isaac Sim."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "emma_rl"
    / "emma_rl"
    / "tasks"
    / "direct"
    / "emma"
    / "emma_obs_nav"
    / "async_lidar.py"
)
SPEC = importlib.util.spec_from_file_location("async_lidar", MODULE_PATH)
async_lidar = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(async_lidar)


def _angles(num_sectors: int = 4) -> torch.Tensor:
    width = 2.0 * math.pi / num_sectors
    return (torch.arange(num_sectors, dtype=torch.float32) + 0.5) * width


def test_bpearl_raw_scan_starts_at_body_forward_and_turns_left():
    angles = async_lidar.bpearl_planar_ray_angles(
        1440,
        horizontal_fov_deg=360.0,
        horizontal_resolution_deg=0.25,
        device="cpu",
    )
    torch.testing.assert_close(angles[0], torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        angles[360], torch.tensor(math.pi / 2.0), atol=1.0e-6, rtol=0.0
    )
    assert abs(abs(angles[720].item()) - math.pi) < 1.0e-6
    torch.testing.assert_close(
        angles[1080], torch.tensor(-math.pi / 2.0), atol=1.0e-6, rtol=0.0
    )


def test_hybrid_180_layout_has_one_degree_front_and_three_degree_elsewhere():
    edges = async_lidar.hybrid_180_sector_edges(device="cpu")
    centers = torch.rad2deg(async_lidar.sector_centers_from_edges(edges))
    widths = torch.rad2deg(edges[1:] - edges[:-1])

    assert edges.numel() == 181
    assert centers.numel() == 180
    torch.testing.assert_close(widths[:45], torch.ones(45), atol=3.0e-5, rtol=0.0)
    torch.testing.assert_close(
        widths[45:135], torch.full((90,), 3.0), atol=3.0e-5, rtol=0.0
    )
    torch.testing.assert_close(widths[135:], torch.ones(45), atol=3.0e-5, rtol=0.0)
    torch.testing.assert_close(centers[0], torch.tensor(0.5), atol=5.0e-5, rtol=0.0)
    torch.testing.assert_close(centers[44], torch.tensor(44.5), atol=5.0e-5, rtol=0.0)
    torch.testing.assert_close(centers[45], torch.tensor(46.5), atol=5.0e-5, rtol=0.0)
    torch.testing.assert_close(centers[-1], torch.tensor(359.5), atol=5.0e-5, rtol=0.0)


def test_hybrid_180_grouped_pooling_produces_180_sectors():
    angles = async_lidar.bpearl_planar_ray_angles(
        1440,
        horizontal_fov_deg=360.0,
        horizontal_resolution_deg=0.25,
        device="cpu",
    ).unsqueeze(0)
    ranges = torch.full_like(angles, 10.0)
    projection = torch.zeros_like(angles)
    ranges[0, 3] = 1.0
    ranges[0, 180 + 11] = 2.0
    ranges[0, 1260 + 2] = 3.0

    pooled_ranges, pooled_angles, pooled_clearance = (
        async_lidar.min_pool_clearance_rays_grouped(
            ranges,
            angles,
            projection,
            ((45, 4), (90, 12), (45, 4)),
        )
    )
    assert pooled_ranges.shape == (1, 180)
    assert pooled_angles.shape == (1, 180)
    torch.testing.assert_close(pooled_ranges[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(pooled_ranges[0, 45], torch.tensor(2.0))
    torch.testing.assert_close(pooled_ranges[0, 135], torch.tensor(3.0))
    torch.testing.assert_close(pooled_clearance, pooled_ranges)


def test_hybrid_180_static_warp_preserves_sector_ranges():
    edges = async_lidar.hybrid_180_sector_edges(device="cpu")
    angles = async_lidar.sector_centers_from_edges(edges).unsqueeze(0)
    ranges = torch.linspace(1.0, 4.0, 180).unsqueeze(0)
    pose = torch.zeros(1, 3)

    warped_ranges, _ = async_lidar.warp_cached_ranges_and_clearances(
        ranges,
        pose,
        pose,
        angles,
        180,
        20.0,
        0.4115,
        0.266,
        0.0,
        0.0,
        edges,
    )
    torch.testing.assert_close(warped_ranges, ranges, atol=1.0e-5, rtol=0.0)


def test_static_pose_preserves_ranges():
    ranges = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    pose = torch.zeros(1, 3)
    warped = async_lidar.warp_cached_ranges(ranges, pose, pose, _angles(), 12.0)
    torch.testing.assert_close(warped, ranges)


def test_forward_translation_reduces_front_range():
    angles = _angles(120)
    ranges = torch.full((1, 120), 12.0)
    ranges[0, 0] = 5.0
    cached_pose = torch.zeros(1, 3)
    current_pose = torch.tensor([[1.0, 0.0, 0.0]])
    warped = async_lidar.warp_cached_ranges(ranges, cached_pose, current_pose, angles, 12.0)
    # The first 3-degree sector is centered at 1.5 degrees, so the expected
    # reduction is numerically almost exactly one metre.
    torch.testing.assert_close(5.0 - warped[0, 0], torch.tensor(1.0), atol=1.0e-3, rtol=0.0)


def test_ninety_degree_rotation_changes_obstacle_sector():
    ranges = torch.tensor([[5.0, 12.0, 12.0, 12.0]])
    cached_pose = torch.zeros(1, 3)
    current_pose = torch.tensor([[0.0, 0.0, math.pi / 2.0]])
    warped = async_lidar.warp_cached_ranges(ranges, cached_pose, current_pose, _angles(), 12.0)
    torch.testing.assert_close(warped[0, 3], torch.tensor(5.0))
    assert warped[0, 0] == 12.0


def test_repeated_warp_has_no_accumulated_drift():
    cached = torch.tensor([[5.0, 12.0, 12.0, 12.0]])
    cached_copy = cached.clone()
    source_pose = torch.zeros(1, 3)
    current_pose = torch.tensor([[0.5, -0.25, 0.3]])
    first = async_lidar.warp_cached_ranges(cached, source_pose, current_pose, _angles(), 12.0)
    second = async_lidar.warp_cached_ranges(cached, source_pose, current_pose, _angles(), 12.0)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(cached, cached_copy)


def test_per_env_schedule_is_independent():
    current = torch.tensor([0.04, 0.04, 0.04])
    next_timestamp = torch.tensor([0.02, 0.03, 0.08])
    periods = torch.tensor([0.02, 0.05, 0.10])
    due = async_lidar.sensor_due_mask(current, next_timestamp)
    assert due.tolist() == [True, True, False]
    advanced = async_lidar.advance_sensor_schedule(
        current, next_timestamp, periods, due, control_dt=0.02,
    )
    assert torch.all(advanced[due] > current[due])
    torch.testing.assert_close(advanced[~due], next_timestamp[~due])


def test_dropout_keeps_cached_timestamp_and_aoi_growing():
    cached_timestamp = torch.tensor([0.0, 0.08])
    aoi_before = async_lidar.compute_aoi(torch.tensor([0.10, 0.10]), cached_timestamp)
    # Env 0 drops its new scan: its cached timestamp remains unchanged.
    after_dropout = async_lidar.compute_aoi(torch.tensor([0.20, 0.20]), cached_timestamp)
    assert after_dropout[0] > aoi_before[0]
    # A scan generated at 0.15 is delivered to env 1 at t=0.20.
    delivered_timestamp = cached_timestamp.clone()
    delivered_timestamp[1] = 0.15
    after_delivery = async_lidar.compute_aoi(torch.tensor([0.20, 0.20]), delivered_timestamp)
    torch.testing.assert_close(after_delivery[1], torch.tensor(0.05))


def test_clearance_uses_actual_hit_angle_before_sector_reduction():
    angle = torch.tensor(math.radians(159.25))
    point = torch.tensor([[[math.cos(angle.item()), math.sin(angle.item())]]])
    ranges, clearances = async_lidar.points_to_ranges_and_clearances(
        point,
        num_sectors=120,
        max_range=12.0,
        half_length=0.42,
        half_width=0.267,
        sensor_offset_x=0.38,
        sensor_offset_y=0.0,
    )
    sector = 53
    expected = 1.0 - async_lidar.rectangular_footprint_projection(
        angle,
        half_length=0.42,
        half_width=0.267,
        sensor_offset_x=0.38,
        sensor_offset_y=0.0,
    )
    torch.testing.assert_close(ranges[0, sector], torch.tensor(1.0))
    torch.testing.assert_close(clearances[0, sector], expected, atol=1.0e-6, rtol=0.0)

    sector_center = torch.tensor(math.radians(160.5))
    center_approximation = 1.0 - async_lidar.rectangular_footprint_projection(
        sector_center,
        half_length=0.42,
        half_width=0.267,
        sensor_offset_x=0.38,
        sensor_offset_y=0.0,
    )
    assert torch.abs(expected - center_approximation) > 0.04


def test_sector_pool_selects_minimum_clearance_not_minimum_range():
    ranges = torch.tensor([[1.00, 1.02, 3.0, 3.0]])
    angles = torch.tensor([[0.0, 0.1, 1.0, 1.1]])
    projection = torch.tensor([[0.10, 0.20, 0.10, 0.10]])
    selected_range, selected_angle, selected_clearance = (
        async_lidar.min_pool_clearance_rays(
            ranges, angles, projection, num_sectors=2)
    )
    torch.testing.assert_close(selected_range[0, 0], torch.tensor(1.02))
    torch.testing.assert_close(selected_angle[0, 0], torch.tensor(0.1))
    torch.testing.assert_close(selected_clearance[0, 0], torch.tensor(0.82))


def test_approach_clearance_gate_is_bounded_and_monotonic():
    clearance = torch.tensor([0.0, 1.0, 1.5, 2.0, 3.0])
    gate = async_lidar.smooth_clearance_gate(clearance, 1.0, 2.0)
    torch.testing.assert_close(gate[[0, 1]], torch.zeros(2))
    torch.testing.assert_close(gate[2], torch.tensor(0.5))
    torch.testing.assert_close(gate[[3, 4]], torch.ones(2))
    assert torch.all(gate[1:] >= gate[:-1])


def test_points_to_ranges_valid_mask_drops_rear_point():
    points = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]]])
    projected = async_lidar.points_to_ranges(
        points,
        num_sectors=220,
        max_range=20.0,
        valid_mask=torch.tensor([[True, False]]),
    )
    hits = projected < 20.0
    assert hits.sum() == 1
    torch.testing.assert_close(projected[hits], torch.tensor([2.0]))


def test_policy_clearance_fills_only_rear_blind_sectors():
    angles = torch.deg2rad(torch.tensor(
        [-178.5, -112.5, -109.5, -1.5, 1.5, 109.5, 112.5, 178.5]
    ))
    clearance = torch.arange(2.0, 10.0).unsqueeze(0)
    masked = async_lidar.fill_clearance_outside_forward_fov(
        clearance,
        angles,
        forward_half_fov_deg=110.0,
        fill_clearance=1.0,
    )
    expected = torch.tensor([[1.0, 1.0, 4.0, 5.0, 6.0, 7.0, 1.0, 1.0]])
    torch.testing.assert_close(masked, expected)
    # 输入的完整 clearance 不能被原地改写。
    torch.testing.assert_close(clearance, torch.arange(2.0, 10.0).unsqueeze(0))


def test_front_first_120_sector_order_masks_indices_37_through_82():
    sector_width = 2.0 * math.pi / 120
    angles = (torch.arange(120, dtype=torch.float32) + 0.5) * sector_width
    angles = torch.atan2(torch.sin(angles), torch.cos(angles))
    clearance = torch.full((1, 120), 5.0)
    masked = async_lidar.fill_clearance_outside_forward_fov(
        clearance,
        angles,
        forward_half_fov_deg=110.0,
        fill_clearance=1.0,
    )
    filled = torch.where(masked[0] == 1.0)[0]
    assert filled.tolist() == list(range(37, 83))


def test_full_policy_fov_preserves_all_clearances():
    angles = torch.tensor([-math.pi, -1.0, 0.0, 1.0, math.pi])
    clearance = torch.tensor([[0.2, 0.5, 2.0, 4.0, 8.0]])
    masked = async_lidar.fill_clearance_outside_forward_fov(
        clearance,
        angles,
        forward_half_fov_deg=180.0,
        fill_clearance=1.0,
    )
    torch.testing.assert_close(masked, clearance)


if __name__ == "__main__":
    tests = [
        test_bpearl_raw_scan_starts_at_body_forward_and_turns_left,
        test_static_pose_preserves_ranges,
        test_forward_translation_reduces_front_range,
        test_ninety_degree_rotation_changes_obstacle_sector,
        test_repeated_warp_has_no_accumulated_drift,
        test_per_env_schedule_is_independent,
        test_dropout_keeps_cached_timestamp_and_aoi_growing,
        test_clearance_uses_actual_hit_angle_before_sector_reduction,
        test_sector_pool_selects_minimum_clearance_not_minimum_range,
        test_approach_clearance_gate_is_bounded_and_monotonic,
        test_points_to_ranges_valid_mask_drops_rear_point,
        test_policy_clearance_fills_only_rear_blind_sectors,
        test_front_first_120_sector_order_masks_indices_37_through_82,
        test_full_policy_fov_preserves_all_clearances,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
