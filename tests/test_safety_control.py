#!/usr/bin/env python3
"""Pure Torch tests for the Emma runtime safety projection."""

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
    / "safety_control.py"
)
SPEC = importlib.util.spec_from_file_location("safety_control", MODULE_PATH)
safety_control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(safety_control)


WHEEL_RADIUS = 0.09
WHEEL_BASE = 0.40
ANGLES = torch.deg2rad(torch.arange(0.0, 360.0, 3.0) + 1.5)


def _apply(
    wheels: torch.Tensor,
    clearance: torch.Tensor,
    aoi: torch.Tensor | None = None,
    **overrides,
):
    n = wheels.shape[0]
    params = dict(
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        lateral_accel_limit=0.8,
        braking_deceleration=1.0,
        reaction_time_s=0.16,
        static_margin=0.3,
        possible_obstacle_speed=0.0,
        forward_half_angle_deg=45.0,
        rear_half_angle_deg=45.0,
        ttc_slow_s=1.5,
        ttc_stop_s=0.4,
        max_lidar_aoi_s=0.2,
        has_valid_scan=torch.ones(n, dtype=torch.bool),
    )
    params.update(overrides)
    return safety_control.apply_safety_control(
        wheels,
        clearance,
        ANGLES,
        torch.zeros(n) if aoi is None else aoi,
        **params,
    )


def _clearance(n: int, distance: float = 20.0) -> torch.Tensor:
    return torch.full((n, ANGLES.numel()), distance)


def test_clear_straight_motion_is_unchanged():
    wheels = torch.tensor([[10.0, 10.0]])
    safe, diag = _apply(wheels, _clearance(1))
    torch.testing.assert_close(safe, wheels)
    assert not diag["shield_intervened"].item()


def test_curve_limiter_preserves_curvature():
    wheels = torch.tensor([[4.0, 20.0]])
    safe, diag = _apply(
        wheels,
        _clearance(1),
        lateral_accel_limit=0.2,
        possible_obstacle_speed=0.0,
    )
    assert 0.0 < diag["curve_scale"].item() < 1.0
    torch.testing.assert_close(
        safe, wheels * diag["total_scale"].unsqueeze(-1)
    )
    raw_twist = diag["raw_twist"][0]
    safe_twist = diag["safe_twist"][0]
    torch.testing.assert_close(
        safe_twist[1] / safe_twist[0], raw_twist[1] / raw_twist[0]
    )


def test_less_clearance_and_more_aoi_reduce_safe_speed():
    wheels = torch.tensor([[20.0, 20.0]]).repeat(3, 1)
    clearance = _clearance(3)
    clearance[1] = 4.0
    clearance[2] = 4.0
    _, diag = _apply(
        wheels,
        clearance,
        torch.tensor([0.0, 0.0, 0.1]),
        possible_obstacle_speed=1.5,
    )
    assert diag["safe_speed"][0] > diag["safe_speed"][1]
    assert diag["safe_speed"][1] > diag["safe_speed"][2]


def test_dynamic_obstacle_bound_is_more_conservative_than_static():
    wheels = torch.tensor([[20.0, 20.0]])
    clearance = _clearance(1, 5.0)
    _, static_diag = _apply(wheels, clearance, possible_obstacle_speed=0.0)
    _, dynamic_diag = _apply(wheels, clearance, possible_obstacle_speed=1.5)
    assert dynamic_diag["safe_speed"].item() < static_diag["safe_speed"].item()


def test_static_front_obstacle_produces_smooth_slowdown_before_stop():
    wheels = torch.tensor([[10.0, 10.0]]).repeat(3, 1)
    clearance = _clearance(3)
    front = torch.cos(ANGLES) >= math.cos(math.radians(45.0))
    clearance[0, front] = 5.0
    clearance[1, front] = 1.5
    clearance[2, front] = 0.9
    safe, diag = _apply(
        wheels,
        clearance,
        possible_obstacle_speed=0.0,
        static_margin=0.3,
        forward_half_angle_deg=45.0,
        rear_half_angle_deg=45.0,
        ttc_slow_s=1.5,
        ttc_stop_s=0.4,
    )
    assert not diag["emergency_stopped"].any()
    torch.testing.assert_close(safe[0], wheels[0])
    assert 0.0 < safe[2, 0] < safe[1, 0] < safe[0, 0]


def test_front_emergency_stops_forward_motion():
    wheels = torch.tensor([[10.0, 10.0]])
    clearance = _clearance(1)
    front = torch.cos(ANGLES) >= math.cos(math.radians(45.0))
    clearance[:, front] = 0.6
    safe, diag = _apply(wheels, clearance)
    torch.testing.assert_close(safe, torch.zeros_like(safe))
    assert diag["emergency_stopped"].item()


def test_reverse_motion_uses_rear_clearance():
    wheels = torch.tensor([[-10.0, -10.0]])
    clearance = _clearance(1)
    rear = torch.cos(ANGLES) <= -math.cos(math.radians(45.0))
    clearance[:, rear] = 0.6
    safe, diag = _apply(wheels, clearance)
    torch.testing.assert_close(safe, torch.zeros_like(safe))
    assert diag["emergency_stopped"].item()


def test_stale_or_missing_scan_fails_closed():
    wheels = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
    safe, diag = _apply(
        wheels,
        _clearance(2),
        aoi=torch.tensor([0.21, 0.0]),
        has_valid_scan=torch.tensor([True, False]),
    )
    torch.testing.assert_close(safe, torch.zeros_like(safe))
    assert diag["emergency_stopped"].all()


def test_non_finite_input_fails_closed_without_nan_command():
    wheels = torch.tensor([[float("nan"), 10.0]])
    safe, diag = _apply(wheels, _clearance(1))
    torch.testing.assert_close(safe, torch.zeros_like(safe))
    assert torch.isfinite(safe).all()
    assert diag["emergency_stopped"].item()


def test_disabled_bypasses_curve_brake_and_emergency_projection():
    wheels = torch.tensor([[4.0, 20.0], [10.0, 10.0]])
    clearance = _clearance(2, 0.1)
    safe, diag = _apply(
        wheels,
        clearance,
        aoi=torch.tensor([1.0, 1.0]),
        has_valid_scan=torch.tensor([False, False]),
        lateral_accel_limit=0.01,
        enabled=False,
    )
    torch.testing.assert_close(safe, wheels)
    torch.testing.assert_close(diag["curve_scale"], torch.ones(2))
    torch.testing.assert_close(diag["brake_scale"], torch.ones(2))
    assert not diag["shield_intervened"].any()
    assert not diag["emergency_stopped"].any()


def test_front_only_lidar_fails_closed_for_reverse_motion():
    front_angles = torch.deg2rad(torch.arange(-109.5, 110.0, 1.0))
    wheels = torch.tensor([[-10.0, -10.0]])
    clearance = torch.full((1, front_angles.numel()), 20.0)
    safe, diag = safety_control.apply_safety_control(
        wheels,
        clearance,
        front_angles,
        torch.zeros(1),
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        lateral_accel_limit=0.8,
        braking_deceleration=1.0,
        reaction_time_s=0.16,
        static_margin=0.3,
        possible_obstacle_speed=0.0,
        forward_half_angle_deg=45.0,
        rear_half_angle_deg=45.0,
        ttc_slow_s=1.5,
        ttc_stop_s=0.4,
        max_lidar_aoi_s=0.2,
        has_valid_scan=torch.ones(1, dtype=torch.bool),
    )
    torch.testing.assert_close(safe, torch.zeros_like(safe))
    assert diag["emergency_stopped"].item()


if __name__ == "__main__":
    tests = [
        test_clear_straight_motion_is_unchanged,
        test_curve_limiter_preserves_curvature,
        test_less_clearance_and_more_aoi_reduce_safe_speed,
        test_dynamic_obstacle_bound_is_more_conservative_than_static,
        test_static_front_obstacle_produces_smooth_slowdown_before_stop,
        test_front_emergency_stops_forward_motion,
        test_reverse_motion_uses_rear_clearance,
        test_stale_or_missing_scan_fails_closed,
        test_non_finite_input_fails_closed_without_nan_command,
        test_disabled_bypasses_curve_brake_and_emergency_projection,
        test_front_only_lidar_fails_closed_for_reverse_motion,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
