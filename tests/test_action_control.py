#!/usr/bin/env python3
"""Pure Torch tests for switchable Emma action control modes."""

from __future__ import annotations

import importlib.util
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
WHEEL_BASE = 0.40
LINEAR_MAX = 0.2
ANGULAR_MAX = 0.2


def _map(actions: torch.Tensor, mode: str) -> torch.Tensor:
    return action_control.actions_to_raw_wheel_velocities(
        actions,
        mode,
        WHEEL_MAX,
        WHEEL_RADIUS,
        WHEEL_BASE,
        LINEAR_MAX,
        ANGULAR_MAX,
    )


def _project(wheels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return action_control.project_wheel_velocities_to_motion_limits(
        wheels,
        WHEEL_MAX,
        WHEEL_RADIUS,
        WHEEL_BASE,
        LINEAR_MAX,
        ANGULAR_MAX,
    )


def _twist(wheels: torch.Tensor) -> torch.Tensor:
    return action_control.wheel_velocities_to_body_twist(
        wheels, WHEEL_RADIUS, WHEEL_BASE
    )


def test_kinematic_limits_match_straight_and_spin_wheel_commands():
    linear_max, angular_max = (
        action_control.differential_drive_kinematic_limits(
            WHEEL_MAX, WHEEL_RADIUS, WHEEL_BASE
        )
    )
    assert abs(linear_max - 0.9) < 1.0e-9
    expected_angular = 2.0 * WHEEL_MAX * WHEEL_RADIUS / WHEEL_BASE
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
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
