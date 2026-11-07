#!/usr/bin/env python3
"""Pure Torch tests for switchable Emma action control modes."""

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
    / "action_control.py"
)
SPEC = importlib.util.spec_from_file_location("action_control", MODULE_PATH)
action_control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(action_control)


WHEEL_MAX = 10.0
WHEEL_RADIUS = 0.09
TRACK_WIDTH = 0.40
LINEAR_MAX = 0.2
ANGULAR_MAX = 0.2


def _map(actions: torch.Tensor, mode: str) -> torch.Tensor:
    return action_control.actions_to_raw_wheel_velocities(
        actions,
        mode,
        WHEEL_MAX,
        WHEEL_RADIUS,
        TRACK_WIDTH,
        LINEAR_MAX,
        ANGULAR_MAX,
    )


def _project(wheels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return action_control.project_wheel_velocities_to_motion_limits(
        wheels,
        WHEEL_MAX,
        WHEEL_RADIUS,
        TRACK_WIDTH,
        LINEAR_MAX,
        ANGULAR_MAX,
    )


def _twist(wheels: torch.Tensor) -> torch.Tensor:
    return action_control.wheel_velocities_to_body_twist(
        wheels, WHEEL_RADIUS, TRACK_WIDTH
    )


def test_kinematic_limits_match_straight_and_spin_wheel_commands():
    linear_max, angular_max = (
        action_control.differential_drive_kinematic_limits(
            WHEEL_MAX, WHEEL_RADIUS, TRACK_WIDTH
        )
    )
    assert abs(linear_max - 0.9) < 1.0e-9
    expected_angular = 2.0 * WHEEL_MAX * WHEEL_RADIUS / TRACK_WIDTH
    assert abs(angular_max - expected_angular) < 1.0e-9

    straight = _twist(torch.tensor([[WHEEL_MAX, WHEEL_MAX]]))[0]
    spin = _twist(torch.tensor([[-WHEEL_MAX, WHEEL_MAX]]))[0]
    torch.testing.assert_close(straight, torch.tensor([linear_max, 0.0]))
    torch.testing.assert_close(spin, torch.tensor([0.0, angular_max]))


def test_wheel_velocity_mode_has_expected_motion_semantics():
    actions = torch.tensor([
        [1.0, 1.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
        [0.0, 0.0],
    ])
    raw = _map(actions, "wheel_velocity")
    torch.testing.assert_close(raw, actions * WHEEL_MAX)

    projected, _ = _project(raw)
    twist = _twist(projected)
    assert twist[0, 0] > 0.0 and twist[0, 1] == 0.0
    assert twist[1, 0] < 0.0 and twist[1, 1] == 0.0
    assert twist[2, 0] == 0.0 and twist[2, 1] > 0.0
    torch.testing.assert_close(twist[3], torch.zeros(2))


def test_both_modes_respect_independent_body_limits():
    for mode in action_control.SUPPORTED_ACTION_CONTROL_MODES:
        actions = torch.tensor([
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ])
        projected, _ = _project(_map(actions, mode))
        twist = _twist(projected)
        assert projected.abs().max() <= WHEEL_MAX + 1.0e-6
        assert twist[:, 0].abs().max() <= LINEAR_MAX + 1.0e-6
        assert twist[:, 1].abs().max() <= ANGULAR_MAX + 1.0e-6


def test_wheel_mode_straight_and_spin_are_body_limited():
    raw = _map(torch.tensor([[1.0, 1.0], [-1.0, 1.0]]), "wheel_velocity")
    raw_twist = _twist(raw)
    assert raw_twist[0, 0] > LINEAR_MAX
    assert raw_twist[1, 1] > ANGULAR_MAX

    projected, scale = _project(raw)
    projected_twist = _twist(projected)
    torch.testing.assert_close(
        projected_twist, raw_twist * scale.unsqueeze(-1)
    )
    torch.testing.assert_close(
        projected_twist[:, 0].abs().clamp(max=LINEAR_MAX),
        projected_twist[:, 0].abs(),
    )
    torch.testing.assert_close(
        projected_twist[:, 1].abs().clamp(max=ANGULAR_MAX),
        projected_twist[:, 1].abs(),
    )


def test_coupled_projection_preserves_curvature():
    raw = _map(torch.tensor([[0.2, 1.0]]), "wheel_velocity")
    projected, scale = _project(raw)
    assert 0.0 < scale.item() < 1.0
    torch.testing.assert_close(projected, raw * scale.unsqueeze(-1))

    raw_twist = _twist(raw)
    projected_twist = _twist(projected)
    torch.testing.assert_close(
        projected_twist[0, 1] / projected_twist[0, 0],
        raw_twist[0, 1] / raw_twist[0, 0],
    )


def test_feasible_body_twist_is_not_scaled():
    raw = _map(torch.tensor([[0.3, 0.2]]), "body_twist")
    projected, scale = _project(raw)
    torch.testing.assert_close(projected, raw)
    torch.testing.assert_close(scale, torch.ones_like(scale))


def test_acceleration_limits_cover_reversal_and_scaled_radii():
    radius_scale = torch.tensor([0.8, 1.0, 1.2])
    previous = torch.tensor([[8.0, 8.0], [8.0, -8.0], [-3.0, 7.0]])
    desired = -previous
    result, scale = action_control.limit_wheel_velocity_acceleration(
        desired, previous, 0.02, 0.05, 0.43, 1.0, 2.0, radius_scale
    )
    delta = action_control.wheel_velocities_to_body_twist(
        result - previous, 0.05, 0.43, radius_scale
    )
    assert torch.all(delta[:, 0].abs() <= 0.020001)
    assert torch.all(delta[:, 1].abs() <= 0.040001)
    assert torch.all((scale >= 0) & (scale <= 1))
    torch.testing.assert_close(result, previous + scale[:, None] * (desired - previous))
    # Each physics substep gets an equal fraction, avoiding 50 Hz stair steps.
    halfway = previous + 0.5 * (result - previous)
    substep_delta = action_control.wheel_velocities_to_body_twist(
        halfway - previous, 0.05, 0.43, radius_scale
    )
    assert torch.all(substep_delta[:, 0].abs() <= 0.010001)
    assert torch.all(substep_delta[:, 1].abs() <= 0.020001)


def test_acceleration_interpolation_stays_in_speed_feasible_set():
    generator = torch.Generator().manual_seed(12)
    previous, _ = _project((torch.rand(1024, 2, generator=generator) - 0.5) * 40)
    desired, _ = _project((torch.rand(1024, 2, generator=generator) - 0.5) * 40)
    result, _ = action_control.limit_wheel_velocity_acceleration(
        desired, previous, 0.02, WHEEL_RADIUS, TRACK_WIDTH, 1.0, 2.0
    )
    twist = _twist(result)
    assert torch.all(result.abs() <= WHEEL_MAX + 1e-6)
    assert torch.all(twist[:, 0].abs() <= LINEAR_MAX + 1e-6)
    assert torch.all(twist[:, 1].abs() <= ANGULAR_MAX + 1e-6)
    unchanged, scale = action_control.limit_wheel_velocity_acceleration(
        previous, previous, 0.02, WHEEL_RADIUS, TRACK_WIDTH, 1.0, 2.0
    )
    torch.testing.assert_close(unchanged, previous)
    torch.testing.assert_close(scale, torch.ones_like(scale))


# Real Emma ObsNav geometry and limits (`emma_obs_nav_env_cfg.py`), quoted with the
# command scale factors, so the bounds below are stated in the units of the
# deployed base: 0.05 m wheels, 0.43 m track width, 0.5 m/s and 1.5 rad/s body
# limits, and a 2.0 rad/s^2 angular limit scaled by 0.94 to the 1.884 rad/s^2 the
# base was calibrated to track.
NAV_WHEEL_MAX = 20.0
NAV_WHEEL_RADIUS = 0.05
NAV_TRACK_WIDTH = 0.43
NAV_LINEAR_MAX = 0.5
NAV_ANGULAR_MAX = 1.5
NAV_STEP_DT = 0.02
NAV_LINEAR_ACCEL_LIMIT = 1.0 * 0.90
NAV_ANGULAR_ACCEL_LIMIT = 2.0 * 0.94


def _nav_desired_wheels(actions: torch.Tensor) -> torch.Tensor:
    return action_control.actions_to_raw_wheel_velocities(
        actions,
        "body_twist",
        NAV_WHEEL_MAX,
        NAV_WHEEL_RADIUS,
        NAV_TRACK_WIDTH,
        NAV_LINEAR_MAX,
        NAV_ANGULAR_MAX,
    )


def _nav_yaw_of(wheels: torch.Tensor) -> torch.Tensor:
    return action_control.wheel_velocities_to_body_twist(
        wheels, NAV_WHEEL_RADIUS, NAV_TRACK_WIDTH
    )[..., 1]


def _run_rate_limit(desired: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    previous = start
    bounded_steps = []
    for step in range(desired.shape[0]):
        bounded, _ = action_control.limit_wheel_velocity_acceleration(
            desired[step : step + 1],
            previous,
            NAV_STEP_DT,
            NAV_WHEEL_RADIUS,
            NAV_TRACK_WIDTH,
            NAV_LINEAR_ACCEL_LIMIT,
            NAV_ANGULAR_ACCEL_LIMIT,
        )
        bounded_steps.append(bounded)
        previous = bounded
    return torch.cat(bounded_steps)


def test_nyquist_yaw_dither_is_bounded_by_the_calibrated_acceleration():
    """A yaw command that flips sign every step must become an achievable demand.

    ``play_diagnostics`` traces of a dithering checkpoint show the angular action
    changing sign every ~1.9 steps, a ~27 rad/s^2 yaw demand where the real base
    tracks 1.884 rad/s^2. The unlimited simulator accepts that demand, which is
    why the behaviour is invisible in sim and becomes sway on hardware.
    """
    steps = 24
    steering, dither = 0.15, 0.18
    actions = torch.tensor(
        [[0.8, steering + (dither if step % 2 == 0 else -dither)] for step in range(steps)]
    )
    desired = _nav_desired_wheels(actions)
    yaw_desired = _nav_yaw_of(desired)
    # Demand as seen from rest, so the first step counts.
    unlimited_peak = yaw_desired.diff().abs().max().item()

    yaw_limited = _nav_yaw_of(_run_rate_limit(desired, torch.zeros(1, 2)))
    limited_peak = torch.cat((torch.zeros(1), yaw_limited)).diff().abs().max().item()

    bound = NAV_ANGULAR_ACCEL_LIMIT * NAV_STEP_DT
    assert limited_peak <= bound + 1e-6, (limited_peak, bound)
    # Without the bound, the simulator accepts an order of magnitude more than
    # the base can produce.
    assert unlimited_peak > 10.0 * bound, (unlimited_peak, bound)


def test_rate_limit_is_transparent_for_a_slow_steering_command():
    """The bound must not fight legitimate manoeuvres.

    The measured dither is a control-rate artefact, not the steering signal, so
    the limit has to pass the frequencies the task actually uses. Starting from
    the commanded steady state isolates steering from the initial acceleration
    out of rest, which is correctly limited.
    """
    steps = 60
    sweep_hz = 0.2
    actions = torch.tensor(
        [
            [0.8, 0.6 * math.sin(2.0 * math.pi * sweep_hz * step * NAV_STEP_DT)]
            for step in range(steps)
        ]
    )
    desired = _nav_desired_wheels(actions)
    bounded = _run_rate_limit(desired[1:], desired[0:1])
    torch.testing.assert_close(bounded, desired[1:], atol=1e-5, rtol=0)


def test_invalid_acceleration_configuration_is_rejected():
    for dt, linear, angular in [(0, 1, 2), (.02, -1, 2), (.02, 1, float('nan'))]:
        try:
            action_control.limit_wheel_velocity_acceleration(
                torch.zeros(1, 2), torch.zeros(1, 2), dt, .05, .43, linear, angular
            )
        except ValueError:
            pass
        else:
            raise AssertionError('invalid acceleration configuration accepted')


def test_command_low_pass_has_exact_step_response_and_shared_alpha():
    desired = torch.tensor([[8.0, 2.0]])
    value = torch.zeros_like(desired)
    dt, tau = 0.02, 0.06
    expected_alpha = 1.0 - math.exp(-dt / tau)
    for step in range(1, 11):
        value, alpha = action_control.low_pass_wheel_velocity_command(
            desired, value, dt, tau
        )
        assert abs(alpha - expected_alpha) < 1e-12
        expected = desired * (1.0 - math.exp(-step * dt / tau))
        torch.testing.assert_close(value, expected, atol=2e-6, rtol=0)


def test_invalid_low_pass_configuration_is_rejected():
    for dt, tau in [(0.0, 0.06), (0.02, 0.0), (0.02, float("nan"))]:
        try:
            action_control.low_pass_wheel_velocity_command(
                torch.zeros(1, 2), torch.zeros(1, 2), dt, tau
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid low-pass configuration accepted")


def test_invalid_action_control_mode_is_rejected():
    try:
        _map(torch.zeros(1, 2), "unknown")
    except ValueError as exc:
        assert "action_control_mode" in str(exc)
    else:
        raise AssertionError("invalid action control mode was accepted")


if __name__ == "__main__":
    tests = [
        test_kinematic_limits_match_straight_and_spin_wheel_commands,
        test_wheel_velocity_mode_has_expected_motion_semantics,
        test_both_modes_respect_independent_body_limits,
        test_wheel_mode_straight_and_spin_are_body_limited,
        test_coupled_projection_preserves_curvature,
        test_feasible_body_twist_is_not_scaled,
        test_invalid_action_control_mode_is_rejected,
        test_acceleration_limits_cover_reversal_and_scaled_radii,
        test_acceleration_interpolation_stays_in_speed_feasible_set,
        test_nyquist_yaw_dither_is_bounded_by_the_calibrated_acceleration,
        test_rate_limit_is_transparent_for_a_slow_steering_command,
        test_invalid_acceleration_configuration_is_rejected,
        test_command_low_pass_has_exact_step_response_and_shared_alpha,
        test_invalid_low_pass_configuration_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
