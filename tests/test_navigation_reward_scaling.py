#!/usr/bin/env python3
"""Pure Torch checks for Emma local-navigation reward scales."""

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
    / "navigation_rewards.py"
)
SPEC = importlib.util.spec_from_file_location("navigation_rewards", MODULE_PATH)
navigation_rewards = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(navigation_rewards)


def _sector_angles(num_sectors: int = 120) -> torch.Tensor:
    step = 2.0 * math.pi / num_sectors
    return -math.pi + (torch.arange(num_sectors) + 0.5) * step


def test_vfh_tracks_goal_in_open_space():
    angles = _sector_angles()
    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        torch.zeros(1, angles.numel()),
        angles,
        torch.tensor([0.4]),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        smoothing_radius_bins=3,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=1.5,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.35,
        switch_margin=0.05,
        max_reference_delta_rad=0.1,
    )
    assert abs((reference - 0.4).item()) <= math.radians(1.6)
    assert diagnostics["free_mask"].all()
    assert not diagnostics["goal_blocked"].item()
    torch.testing.assert_close(
        diagnostics["effective_goal_cost"], torch.tensor([1.0])
    )
    torch.testing.assert_close(
        diagnostics["effective_obstacle_cost"], torch.tensor([1.5])
    )


def test_vfh_selects_one_free_side_instead_of_averaging_through_obstacle():
    angles = _sector_angles()
    density = torch.zeros(1, angles.numel())
    density[:, angles.abs() < math.radians(25.0)] = 1.0
    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.zeros(1),
        torch.tensor([0.7]),
        torch.ones(1, dtype=torch.bool),
        smoothing_radius_bins=3,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=1.5,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.35,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(8.0),
    )
    assert reference.item() > 0.0
    assert abs(reference.item()) >= math.radians(20.0)
    assert not diagnostics["free_mask"][0, angles.abs().argmin()]


def test_vfh_limits_reference_change_when_goal_switches_side():
    angles = _sector_angles()
    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        torch.zeros(1, angles.numel()),
        angles,
        torch.tensor([-1.0]),
        torch.tensor([0.8]),
        torch.ones(1, dtype=torch.bool),
        smoothing_radius_bins=3,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=1.5,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.35,
        switch_margin=0.05,
        max_reference_delta_rad=0.1,
    )
    torch.testing.assert_close(reference, torch.tensor([0.7]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        diagnostics["reference_delta"], torch.tensor([-0.1]), atol=1.0e-6, rtol=0.0
    )


def test_vfh_prefers_open_detour_when_goal_sector_is_blocked():
    angles = _sector_angles()
    density = torch.ones(1, angles.numel())
    left_open = (angles > math.radians(-100.0)) & (
        angles < math.radians(-80.0)
    )
    narrow_goal_side = (angles > math.radians(15.0)) & (
        angles < math.radians(30.0)
    )
    density[:, left_open] = 0.0
    density[:, narrow_goal_side] = 0.2

    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=1.5,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.35,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(8.0),
    )

    assert diagnostics["goal_blocked"].item()
    torch.testing.assert_close(
        diagnostics["effective_goal_cost"], torch.tensor([0.2])
    )
    torch.testing.assert_close(
        diagnostics["effective_obstacle_cost"], torch.tensor([3.5])
    )
    assert math.radians(-100.0) < reference.item() < math.radians(-80.0)


def test_vfh_passage_mask_rejects_narrow_gap_but_keeps_wide_gap():
    angles = _sector_angles()
    ranges = torch.full((2, angles.numel()), 20.0)
    for env_index, half_gap in enumerate((0.45, 0.80)):
        boundary_angle = math.atan2(half_gap, 1.0)
        boundary_range = math.hypot(1.0, half_gap)
        for sign in (-1.0, 1.0):
            index = (angles - sign * boundary_angle).abs().argmin()
            ranges[env_index, index] = boundary_range

    passage_mask = navigation_rewards.vfh_passage_mask(
        ranges,
        torch.tensor([2.5, 2.5]),
        min_passage_width=1.15,
    )
    goal_index = angles.abs().argmin()
    assert not passage_mask[0, goal_index]
    assert passage_mask[1, goal_index]


def test_vfh_passage_mask_ignores_obstacle_beyond_goal_horizon():
    angles = _sector_angles()
    ranges = torch.full((2, angles.numel()), 20.0)
    goal_index = angles.abs().argmin()
    ranges[:, goal_index] = 1.5

    passage_mask = navigation_rewards.vfh_passage_mask(
        ranges,
        torch.tensor([1.0, 2.0]),
        min_passage_width=1.15,
    )
    assert passage_mask[0, goal_index]
    assert not passage_mask[1, goal_index]


def test_bounded_smoothstep_blockage_has_finite_support():
    blockage = navigation_rewards.bounded_smoothstep_blockage(
        torch.tensor([0.4, 3.5, 3.75, 4.0, 8.0]),
        blocked_distance=3.5,
        free_distance=4.0,
    )
    torch.testing.assert_close(
        blockage,
        torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_vfh_opening_quality_prefers_wide_deep_valley():
    angles = _sector_angles()
    passage_mask = torch.zeros(1, angles.numel(), dtype=torch.bool)
    clearance = torch.full((1, angles.numel()), 0.2)
    narrow = (angles > math.radians(-52.0)) & (angles < math.radians(-38.0))
    wide = (angles > math.radians(35.0)) & (angles < math.radians(85.0))
    passage_mask[:, narrow | wide] = True
    clearance[:, narrow] = 2.0
    clearance[:, wide] = 4.0

    width_quality, depth_quality = navigation_rewards.vfh_opening_quality(
        clearance,
        passage_mask,
        route_lookahead=4.0,
        valley_radius_bins=5,
    )
    narrow_index = (angles - math.radians(-45.0)).abs().argmin()
    wide_index = (angles - math.radians(60.0)).abs().argmin()
    assert width_quality[0, wide_index] > width_quality[0, narrow_index]
    assert depth_quality[0, wide_index] > depth_quality[0, narrow_index]
    assert width_quality[0, wide_index] > 0.95
    assert depth_quality[0, wide_index] > 0.95


def test_vfh_smoothly_interpolates_blocked_mode_costs():
    angles = _sector_angles()
    density = torch.zeros(1, angles.numel())
    goal_index = angles.abs().argmin()
    density[0, goal_index] = 0.5
    _, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=2.0,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.25,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(2.4),
    )
    torch.testing.assert_close(
        diagnostics["goal_blockage"], torch.tensor([0.5]), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        diagnostics["effective_goal_cost"],
        torch.tensor([0.6]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        diagnostics["effective_obstacle_cost"],
        torch.tensor([2.75]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_vfh_blockage_state_changes_gradually_after_initial_scan():
    angles = _sector_angles()
    density = torch.ones(1, angles.numel())
    _, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=2.0,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.25,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(2.4),
        previous_blockage=torch.tensor([0.2]),
        blockage_smoothing_alpha=0.25,
    )
    torch.testing.assert_close(
        diagnostics["goal_blockage"], torch.tensor([0.4]), atol=1.0e-6, rtol=0.0
    )
    assert not diagnostics["goal_blocked"].item()


def test_vfh_prefers_wider_deeper_detour_over_goal_side_gap():
    angles = _sector_angles()
    density = torch.ones(1, angles.numel())
    passage_mask = torch.zeros_like(density, dtype=torch.bool)
    clearance = torch.full_like(density, 0.2)
    narrow_goal_side = (
        (angles > math.radians(-38.0)) & (angles < math.radians(-22.0))
    )
    wide_detour = (
        (angles > math.radians(38.0)) & (angles < math.radians(82.0))
    )
    density[:, narrow_goal_side | wide_detour] = 0.0
    passage_mask[:, narrow_goal_side | wide_detour] = True
    clearance[:, narrow_goal_side] = 2.0
    clearance[:, wide_detour] = 4.0
    width_quality, depth_quality = navigation_rewards.vfh_opening_quality(
        clearance,
        passage_mask,
        route_lookahead=4.0,
        valley_radius_bins=5,
    )

    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=2.0,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.25,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(2.4),
        passage_mask=passage_mask,
        opening_width_quality=width_quality,
        opening_depth_quality=depth_quality,
        opening_width_cost=0.8,
        opening_depth_cost=0.6,
        valley_keep_half_angle_rad=math.radians(40.0),
    )
    assert diagnostics["goal_blocked"].item()
    assert math.radians(38.0) < reference.item() < math.radians(82.0)
    assert diagnostics["selected_width_quality"].item() > 0.95


def test_vfh_tracks_valley_center_instead_of_freezing_previous_angle():
    angles = _sector_angles()
    density = torch.ones(1, angles.numel())
    opening = (angles > math.radians(10.0)) & (angles < math.radians(90.0))
    density[:, opening] = 0.0
    passage_mask = opening.unsqueeze(0)
    clearance = torch.where(
        passage_mask, torch.full_like(density, 4.0), torch.full_like(density, 0.2)
    )
    width_quality, depth_quality = navigation_rewards.vfh_opening_quality(
        clearance,
        passage_mask,
        route_lookahead=4.0,
        valley_radius_bins=5,
    )
    previous = torch.tensor([math.radians(25.0)])
    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        density,
        angles,
        torch.tensor([math.radians(55.0)]),
        previous,
        torch.ones(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=2.0,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.25,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(2.4),
        passage_mask=passage_mask,
        opening_width_quality=width_quality,
        opening_depth_quality=depth_quality,
        opening_width_cost=0.8,
        opening_depth_cost=0.6,
        valley_keep_half_angle_rad=math.radians(40.0),
    )
    assert diagnostics["kept_previous"].item()
    assert reference.item() > previous.item()
    torch.testing.assert_close(
        diagnostics["reference_delta"],
        torch.tensor([math.radians(2.4)]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_vfh_reference_excludes_goal_aligned_narrow_passage():
    angles = _sector_angles()
    ranges = torch.full((1, angles.numel()), 20.0)
    for sign in (-1.0, 1.0):
        boundary_angle = sign * math.atan2(0.45, 1.0)
        index = (angles - boundary_angle).abs().argmin()
        ranges[0, index] = math.hypot(1.0, 0.45)
    passage_mask = navigation_rewards.vfh_passage_mask(
        ranges,
        torch.tensor([2.5]),
        min_passage_width=1.15,
    )

    reference, diagnostics = navigation_rewards.vfh_reference_angle(
        torch.zeros_like(ranges),
        angles,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        smoothing_radius_bins=0,
        blocked_threshold=0.55,
        goal_cost=1.0,
        obstacle_cost=2.0,
        blocked_goal_cost=0.2,
        blocked_obstacle_cost=3.5,
        continuity_cost=0.35,
        switch_margin=0.05,
        max_reference_delta_rad=math.radians(8.0),
        passage_mask=passage_mask,
    )
    selected_index = (
        navigation_rewards.wrap_to_pi(angles - reference.item()).abs().argmin()
    )
    assert diagnostics["goal_blocked"].item()
    assert passage_mask[0, selected_index]
    assert reference.abs().item() > math.radians(40.0)


def test_step_turn_rewards_correct_direction_not_standstill():
    reference = torch.deg2rad(torch.tensor([90.0, 90.0, 90.0, 0.0]))
    angular_velocity = torch.tensor([1.0, -1.0, 0.0, 0.0])
    reward, score = navigation_rewards.step_turn_direction_reward(
        reference,
        angular_velocity,
        max_angular_velocity=1.0,
        scale_per_step=0.15,
        deadband_rad=math.radians(6.0),
    )
    torch.testing.assert_close(reward[:2], torch.tensor([0.15, -0.15]))
    torch.testing.assert_close(reward[2:], torch.zeros(2))
    torch.testing.assert_close(score[:2], torch.tensor([1.0, -1.0]))


def test_step_turn_scale_is_bounded_and_grows_with_reference_error():
    reward, score = navigation_rewards.step_turn_direction_reward(
        torch.deg2rad(torch.tensor([6.0, 45.0, 90.0, 135.0])),
        torch.ones(4),
        max_angular_velocity=1.0,
        scale_per_step=0.15,
        deadband_rad=math.radians(6.0),
    )
    torch.testing.assert_close(reward[0], torch.tensor(0.0))
    assert 0.0 < reward[1] < reward[2]
    torch.testing.assert_close(reward[2:], torch.tensor([0.15, 0.15]))
    assert torch.all(score.abs() <= 1.0)


def test_step_turn_deadband_ignores_small_corrections():
    reference = torch.deg2rad(torch.tensor([4.0, 4.0, 45.0]))
    angular_velocity = torch.tensor([0.5, -1.0, 0.5])
    reward, _ = navigation_rewards.step_turn_direction_reward(
        reference,
        angular_velocity,
        max_angular_velocity=1.0,
        scale_per_step=0.15,
        deadband_rad=math.radians(6.0),
    )
    torch.testing.assert_close(reward[:2], torch.zeros(2))
    assert reward[2] > 0.0


def test_configured_step_turn_reward_is_immediately_larger_than_idle_cost():
    reward, _ = navigation_rewards.step_turn_direction_reward(
        torch.tensor([math.pi / 2.0]),
        torch.tensor([1.0]),
        max_angular_velocity=1.0,
        scale_per_step=0.15,
        deadband_rad=math.radians(6.0),
    )
    assert abs(reward.item() - 0.15) < 1.0e-6
    assert reward.item() > 0.03


def test_step_safe_forward_uses_bounded_non_trigonometric_window():
    reward = navigation_rewards.step_safe_forward_reward(
        torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.0, -0.9]),
        torch.deg2rad(
            torch.tensor([0.0, 10.0, 27.5, 45.0, 90.0, 0.0, 0.0])
        ),
        max_forward_speed=0.9,
        scale_per_step=0.35,
        full_reward_angle_rad=math.radians(10.0),
        cutoff_angle_rad=math.radians(45.0),
    )
    torch.testing.assert_close(
        reward,
        torch.tensor([0.35, 0.35, 0.175, 0.0, 0.0, 0.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_step_idle_penalty_accepts_correct_turn_or_safe_forward():
    penalty, activity = navigation_rewards.step_idle_penalty(
        torch.tensor([0.0, 0.9, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, -1.0, 0.5]),
        torch.deg2rad(torch.tensor([0.0, 0.0, 90.0, 90.0, 90.0])),
        max_forward_speed=0.9,
        max_angular_velocity=1.0,
        scale_per_step=0.03,
        turn_deadband_rad=math.radians(6.0),
        forward_full_reward_angle_rad=math.radians(10.0),
        forward_cutoff_angle_rad=math.radians(45.0),
    )
    torch.testing.assert_close(
        activity, torch.tensor([0.0, 1.0, 1.0, 0.0, 0.5])
    )
    torch.testing.assert_close(
        penalty, torch.tensor([-0.03, 0.0, 0.0, -0.03, -0.015])
    )


def test_repulse_is_motion_direction_symmetric_and_side_weight_is_nonzero():
    angles = torch.deg2rad(torch.tensor([0.0, 180.0]))
    clearance = torch.tensor([[0.1, 20.0], [20.0, 0.1]])
    penalty, risk = navigation_rewards.closing_repulse_penalty(
        clearance,
        angles,
        torch.tensor([0.9, -0.9]),
        torch.zeros(2),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
    )
    torch.testing.assert_close(penalty[0], penalty[1])
    torch.testing.assert_close(risk[0], risk[1])

    side_penalty, side_risk = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.1]]),
        torch.deg2rad(torch.tensor([90.0])),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.3,
    )
    assert side_penalty[0] < 0.0
    danger = 1.0 / (0.1 + 0.15) - 1.0 / (1.0 + 0.15)
    torch.testing.assert_close(side_risk, torch.tensor([danger * 0.3]))


def test_repulse_keeps_a_floor_while_moving_away_from_obstacle():
    approaching, _ = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.1]]),
        torch.tensor([0.0]),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.3,
    )
    moving_away, _ = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.1]]),
        torch.tensor([0.0]),
        torch.tensor([-0.9]),
        torch.tensor([0.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.3,
    )
    assert moving_away < 0.0
    torch.testing.assert_close(
        moving_away.abs(), approaching.abs() * 0.3
    )


def test_repulse_penalizes_rotation_sweep_at_standstill():
    penalty, risk = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.1, 0.1]]),
        torch.deg2rad(torch.tensor([0.0, 90.0])),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
    )
    assert penalty[0] < 0.0
    assert risk[0] > 0.0


def test_repulse_motion_floor_penalizes_only_near_obstacle_at_standstill():
    penalty, risk = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.3], [1.2]]),
        torch.tensor([0.0]),
        torch.zeros(2),
        torch.zeros(2),
        start_clearance=1.2,
        max_speed=0.9,
        scale_per_step=0.12,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.3,
        rotation_weight=0.35,
        motion_floor=0.05,
    )
    danger = 1.0 / (0.3 + 0.15) - 1.0 / (1.2 + 0.15)
    expected_near_risk = danger * 0.05
    torch.testing.assert_close(
        risk[0], torch.tensor(expected_near_risk), atol=1.0e-7, rtol=0.0
    )
    torch.testing.assert_close(
        penalty[0],
        torch.tensor(-0.12 * expected_near_risk),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(risk[1], torch.tensor(0.0))
    torch.testing.assert_close(penalty[1], torch.tensor(0.0))


def test_goal_aware_relevance_ignores_wall_behind_goal_only_in_goal_corridor():
    angles = torch.deg2rad(torch.tensor([-30.0, 0.0, 30.0]))
    relevance = navigation_rewards.goal_aware_obstacle_relevance(
        torch.tensor([[0.2, 2.2, 0.2]]),
        angles,
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        goal_reach_dist=0.3,
        collision_clearance_threshold=0.05,
    )
    torch.testing.assert_close(relevance[0, 0], torch.tensor(1.0))
    assert relevance[0, 1] < 0.02
    torch.testing.assert_close(relevance[0, 2], torch.tensor(1.0))


def test_goal_aware_relevance_keeps_obstacle_before_goal():
    relevance = navigation_rewards.goal_aware_obstacle_relevance(
        torch.tensor([[1.0]]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        goal_reach_dist=0.3,
        collision_clearance_threshold=0.05,
    )
    assert relevance[0, 0] > 0.99


def test_goal_aware_relevance_suppresses_repulse_from_wall_behind_goal():
    relevance = navigation_rewards.goal_aware_obstacle_relevance(
        torch.tensor([[0.9]]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.5]),
        goal_reach_dist=0.3,
        collision_clearance_threshold=0.05,
    )
    unmasked, _ = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.9]]),
        torch.tensor([0.0]),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
    )
    masked, _ = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.9]]),
        torch.tensor([0.0]),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        start_clearance=1.0,
        max_speed=0.9,
        scale_per_step=0.04,
        robot_half_length=0.42,
        robot_half_width=0.267,
        obstacle_relevance=relevance,
    )
    assert masked.abs() < unmasked.abs() * 0.01


def test_action_jitter_penalizes_angular_changes_more_than_linear_changes():
    penalty, jitter = navigation_rewards.action_jitter_penalty(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
        linear_scale=0.002,
        angular_scale=0.008,
    )
    torch.testing.assert_close(jitter, torch.tensor([0.5, 0.5, 4.0]))
    torch.testing.assert_close(penalty, torch.tensor([-0.002, -0.008, -0.040]))


def test_inverse_repulse_balances_full_speed_drive_at_045m():
    repulse, risk = navigation_rewards.closing_repulse_penalty(
        torch.tensor([[0.45]]),
        torch.tensor([0.0]),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
        start_clearance=1.2,
        max_speed=0.9,
        scale_per_step=0.591,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.65,
        rotation_weight=0.15,
        motion_floor=0.15,
    )
    drive_budget = 0.50 + 0.05 - 0.003
    expected_danger = 1.0 / (0.45 + 0.15) - 1.0 / (1.2 + 0.15)
    torch.testing.assert_close(
        risk, torch.tensor([expected_danger]), atol=1.0e-6, rtol=0.0
    )
    assert abs((repulse + drive_budget).item()) < 5.0e-4


def test_three_tenths_side_clearance_is_negative_but_half_meter_is_passable():
    clearances = torch.tensor([[0.3], [0.5]])
    repulse, _ = navigation_rewards.closing_repulse_penalty(
        clearances,
        torch.tensor([math.pi / 2.0]),
        torch.tensor([0.9, 0.9]),
        torch.tensor([0.0, 0.0]),
        start_clearance=1.2,
        max_speed=0.9,
        scale_per_step=0.591,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lateral_weight_floor=0.65,
        rotation_weight=0.15,
        motion_floor=0.15,
    )
    drive_budget = 0.50 + 0.05 - 0.003
    assert (drive_budget + repulse[0]).item() < 0.0
    assert (drive_budget + repulse[1]).item() > 0.0


def test_step_progress_restores_reward_immediately_after_detour():
    away = navigation_rewards.step_progress_reward(
        torch.tensor([3.0]),
        torch.tensor([3.5]),
        full_progress_distance=0.018,
        scale_per_step=0.05,
    )
    recovered = navigation_rewards.step_progress_reward(
        torch.tensor([3.5]),
        torch.tensor([3.0]),
        full_progress_distance=0.018,
        scale_per_step=0.05,
    )
    advanced = navigation_rewards.step_progress_reward(
        torch.tensor([3.0]),
        torch.tensor([2.991]),
        full_progress_distance=0.018,
        scale_per_step=0.05,
    )
    capped = navigation_rewards.step_progress_reward(
        torch.tensor([3.0]),
        torch.tensor([2.0]),
        full_progress_distance=0.018,
        scale_per_step=0.05,
    )
    torch.testing.assert_close(away, torch.zeros(1))
    torch.testing.assert_close(recovered, torch.tensor([0.05]))
    torch.testing.assert_close(advanced, torch.tensor([0.025]), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(capped, torch.tensor([0.05]))


if __name__ == "__main__":
    tests = [
        test_vfh_tracks_goal_in_open_space,
        test_vfh_selects_one_free_side_instead_of_averaging_through_obstacle,
        test_vfh_limits_reference_change_when_goal_switches_side,
        test_vfh_prefers_open_detour_when_goal_sector_is_blocked,
        test_vfh_passage_mask_rejects_narrow_gap_but_keeps_wide_gap,
        test_vfh_passage_mask_ignores_obstacle_beyond_goal_horizon,
        test_bounded_smoothstep_blockage_has_finite_support,
        test_vfh_opening_quality_prefers_wide_deep_valley,
        test_vfh_smoothly_interpolates_blocked_mode_costs,
        test_vfh_blockage_state_changes_gradually_after_initial_scan,
        test_vfh_prefers_wider_deeper_detour_over_goal_side_gap,
        test_vfh_tracks_valley_center_instead_of_freezing_previous_angle,
        test_vfh_reference_excludes_goal_aligned_narrow_passage,
        test_step_turn_rewards_correct_direction_not_standstill,
        test_step_turn_scale_is_bounded_and_grows_with_reference_error,
        test_step_turn_deadband_ignores_small_corrections,
        test_configured_step_turn_reward_is_immediately_larger_than_idle_cost,
        test_step_safe_forward_uses_bounded_non_trigonometric_window,
        test_step_idle_penalty_accepts_correct_turn_or_safe_forward,
        test_repulse_is_motion_direction_symmetric_and_side_weight_is_nonzero,
        test_repulse_keeps_a_floor_while_moving_away_from_obstacle,
        test_repulse_penalizes_rotation_sweep_at_standstill,
        test_repulse_motion_floor_penalizes_only_near_obstacle_at_standstill,
        test_goal_aware_relevance_ignores_wall_behind_goal_only_in_goal_corridor,
        test_goal_aware_relevance_keeps_obstacle_before_goal,
        test_goal_aware_relevance_suppresses_repulse_from_wall_behind_goal,
        test_action_jitter_penalizes_angular_changes_more_than_linear_changes,
        test_inverse_repulse_balances_full_speed_drive_at_045m,
        test_three_tenths_side_clearance_is_negative_but_half_meter_is_passable,
        test_step_progress_restores_reward_immediately_after_detour,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
