#!/usr/bin/env python3
"""Pure Torch tests for the forward-clearance speed governor."""

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
    / "safety_control.py"
)
SPEC = importlib.util.spec_from_file_location("safety_control", MODULE_PATH)
safety_control = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(safety_control)


WHEEL_RADIUS = 0.05
WHEEL_BASE = 0.43
WHEEL_MAX = 20.0
ANGLES = torch.deg2rad(torch.arange(0.0, 360.0, 3.0) + 1.5)


def _wheels(vx: float, wz: float) -> torch.Tensor:
    half = WHEEL_BASE / 2.0
    return torch.tensor([[
        (vx - wz * half) / WHEEL_RADIUS,
        (vx + wz * half) / WHEEL_RADIUS,
    ]])


def _clearance(n: int, distance: float = 20.0) -> torch.Tensor:
    return torch.full((n, ANGLES.numel()), distance)


def _apply(
    wheels: torch.Tensor,
    clearance: torch.Tensor,
    aoi: torch.Tensor | None = None,
    **overrides,
):
    params = dict(
        wheel_radius=WHEEL_RADIUS,
        wheel_base=WHEEL_BASE,
        wheel_max_velocity=WHEEL_MAX,
        forward_half_angle_deg=30.0,
        stop_clearance=0.3,
        full_speed_clearance=0.9,
        min_forward_speed_scale=0.2,
        max_lidar_aoi_s=0.3,
        has_valid_scan=torch.ones(wheels.shape[0], dtype=torch.bool),
    )
    params.update(overrides)
    return safety_control.apply_safety_control(
        wheels,
        clearance,
        ANGLES,
        torch.zeros(wheels.shape[0]) if aoi is None else aoi,
        **params,
    )


def _set_front(clearance: torch.Tensor, distances: torch.Tensor) -> None:
    front = torch.cos(ANGLES) >= torch.cos(torch.deg2rad(torch.tensor(30.0)))
    clearance[:, front] = distances.unsqueeze(-1)


def test_clear_forward_motion_is_unchanged():
    wheels = _wheels(0.5, 0.3)
    safe, diag = _apply(wheels, _clearance(1))
    torch.testing.assert_close(safe, wheels)
    assert diag["forward_speed_scale"].item() == 1.0
    assert not diag["shield_intervened"].item()


def test_minimum_midpoint_and_full_speed_thresholds():
    expected_scale = torch.tensor([
        0.2, 0.2592592593, 0.325, 0.6, 0.875, 0.9407407407, 1.0
    ])
    wheels = _wheels(0.6, 0.0).repeat(expected_scale.numel(), 1)
    clearance = _clearance(expected_scale.numel())
    _set_front(clearance, torch.tensor([0.3, 0.4, 0.45, 0.6, 0.75, 0.8, 0.9]))
    _, diag = _apply(wheels, clearance)
    torch.testing.assert_close(
        diag["forward_speed_scale"], expected_scale
    )
    torch.testing.assert_close(
        diag["safe_twist"][:, 0], 0.6 * expected_scale
    )


def test_decreasing_clearance_monotonically_reduces_forward_speed():
    wheels = _wheels(0.7, 0.0).repeat(6, 1)
    clearance = _clearance(6)
    _set_front(clearance, torch.tensor([1.2, 0.9, 0.75, 0.6, 0.45, 0.3]))
    _, diag = _apply(wheels, clearance)
    speed = diag["safe_twist"][:, 0]
    assert torch.all(speed[:-1] >= speed[1:])
    assert abs(speed[-1].item() - 0.14) < 1e-6


def test_minimum_forward_scale_keeps_angular_command_for_escape_turn():
    wheels = _wheels(0.3, 0.8)
    clearance = _clearance(1)
    _set_front(clearance, torch.tensor([0.2]))
    _, diag = _apply(wheels, clearance)
    assert abs(diag["safe_twist"][0, 0].item() - 0.06) < 1e-6
    assert abs(diag["safe_twist"][0, 1].item() - 0.8) < 1e-6


def test_reverse_and_in_place_rotation_are_unchanged():
    wheels = torch.cat((_wheels(-0.3, 0.4), _wheels(0.0, -0.8)), dim=0)
    clearance = _clearance(2, 0.1)
    safe, diag = _apply(wheels, clearance)
    torch.testing.assert_close(safe, wheels)
    torch.testing.assert_close(diag["forward_speed_scale"], torch.ones(2))


def test_stale_or_missing_scan_blocks_forward_but_keeps_turning():
    wheels = _wheels(0.3, 0.6).repeat(2, 1)
    _, diag = _apply(
        wheels,
        _clearance(2),
        aoi=torch.tensor([0.31, 0.0]),
        has_valid_scan=torch.tensor([True, False]),
    )
    torch.testing.assert_close(diag["safe_twist"][:, 0], torch.zeros(2))
    torch.testing.assert_close(diag["safe_twist"][:, 1], torch.full((2,), 0.6))
    assert diag["shield_intervened"].all()


def test_disabled_bypasses_forward_governor():
    wheels = _wheels(0.5, 0.4)
    clearance = _clearance(1, 0.1)
    safe, diag = _apply(
        wheels,
        clearance,
        aoi=torch.tensor([1.0]),
        has_valid_scan=torch.tensor([False]),
        enabled=False,
    )
    torch.testing.assert_close(safe, wheels)
    assert diag["forward_speed_scale"].item() == 1.0


def test_invalid_thresholds_are_rejected():
    try:
        _apply(_wheels(0.3, 0.0), _clearance(1), full_speed_clearance=0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid clearance thresholds were accepted")
    try:
        _apply(
            _wheels(0.3, 0.0),
            _clearance(1),
            min_forward_speed_scale=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid minimum speed scale was accepted")


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"PASSED: {name}")
