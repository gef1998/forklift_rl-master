"""MPPI 专家核心的纯 Torch 回归测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source"
    / "emma_rl"
    / "emma_rl"
    / "tasks"
    / "direct"
    / "emma"
    / "emma_mppi_nav"
    / "mppi_controller.py"
)
SPEC = importlib.util.spec_from_file_location("mppi_controller", MODULE_PATH)
mppi = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mppi
SPEC.loader.exec_module(mppi)


def test_wheel_projection_preserves_curvature_and_limits() -> None:
    wheels = torch.tensor([[10.0, 10.0], [-10.0, 10.0], [8.0, 10.0]])
    projected = mppi.project_wheel_sequences(
        wheels,
        wheel_radius=0.09,
        track_width=0.40,
        wheel_speed_limit=10.0,
        max_linear_velocity=0.9,
        max_angular_velocity=2.0,
    )
    twist = mppi.wheel_to_twist(projected, 0.09, 0.40)
    assert projected.abs().amax() <= 10.0 + 1.0e-6
    assert twist[:, 0].abs().amax() <= 0.9 + 1.0e-6
    assert twist[:, 1].abs().amax() <= 2.0 + 1.0e-6
    original_twist = mppi.wheel_to_twist(wheels, 0.09, 0.40)
    original_curvature = original_twist[2, 1] / original_twist[2, 0]
    projected_curvature = twist[2, 1] / twist[2, 0]
    assert torch.allclose(original_curvature, projected_curvature, atol=1.0e-6)


def test_rectangle_clearance_detects_front_and_side_points() -> None:
    points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    x = torch.zeros(1, 1)
    y = torch.zeros(1, 1)
    yaw = torch.zeros(1, 1)
    clearance = mppi.rectangle_clearance_to_points(
        x,
        y,
        yaw,
        points,
        valid,
        robot_half_length=0.42,
        robot_half_width=0.267,
    )
    assert torch.allclose(clearance, torch.tensor([[0.58]]), atol=1.0e-5)


def test_clearance_reconstruction_honors_lidar_offset_and_no_return() -> None:
    angle = torch.tensor([[torch.pi / 4.0]])
    projection = (0.42 - 0.38) / torch.cos(angle)
    measured_range = torch.tensor([[1.0]])
    clearance = measured_range - projection
    points, valid = mppi.clearance_scan_to_points(
        clearance,
        angle,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lidar_offset_x=0.38,
        lidar_offset_y=0.0,
        lidar_max_distance=20.0,
    )
    expected = torch.stack(
        (
            0.38 + measured_range * torch.cos(angle),
            measured_range * torch.sin(angle),
        ),
        dim=-1,
    ).squeeze(1)
    assert valid.item()
    assert torch.allclose(points.squeeze(1), expected, atol=1.0e-6)

    no_return_clearance = 20.0 - projection
    _, no_return_valid = mppi.clearance_scan_to_points(
        no_return_clearance,
        angle,
        robot_half_length=0.42,
        robot_half_width=0.267,
        lidar_offset_x=0.38,
        lidar_offset_y=0.0,
        lidar_max_distance=20.0,
    )
    assert not no_return_valid.item()


def _small_controller() -> mppi.TorchMppiController:
    cfg = mppi.MppiControllerCfg(
        horizon=18,
        num_samples=64,
        iterations=1,
        prediction_dt=0.10,
        temperature=0.5,
        wheel_noise_std=2.0,
        obstacle_point_stride=1,
    )
    return mppi.TorchMppiController(1, "cpu", cfg)


def test_mppi_drives_forward_in_open_space() -> None:
    torch.manual_seed(7)
    controller = _small_controller()
    clearance = torch.full((1, 120), 20.0)
    goal_xy = torch.tensor([[4.0, 0.0]])
    current_twist = torch.zeros(1, 2)
    action, diagnostics = controller.command(clearance, goal_xy, current_twist)
    assert action[0, 0] > 0.15
    assert diagnostics["selected_cost"].isfinite().all()


def test_mppi_action_respects_joint_wheel_and_body_limits() -> None:
    torch.manual_seed(9)
    controller = _small_controller()
    clearance = torch.full((1, 120), 20.0)
    action, diagnostics = controller.command(
        clearance,
        torch.tensor([[1.0, 3.0]]),
        torch.tensor([[0.8, 1.5]]),
    )
    wheels = diagnostics["wheel_command"]
    twist = mppi.wheel_to_twist(wheels, 0.09, 0.40)
    assert action.abs().amax() <= 1.0 + 1.0e-6
    assert wheels.abs().amax() <= 10.0 + 1.0e-6
    assert twist[:, 0].abs().amax() <= 0.9 + 1.0e-6
    assert twist[:, 1].abs().amax() <= 2.0 + 1.0e-6
    assert torch.allclose(action[:, 0], twist[:, 0] / 0.9, atol=1.0e-6)
    assert torch.allclose(action[:, 1], twist[:, 1] / 2.0, atol=1.0e-6)


def test_mppi_does_not_keep_driving_straight_into_front_obstacle() -> None:
    torch.manual_seed(11)
    controller = _small_controller()
    clearance = torch.full((1, 120), 20.0)
    clearance[0, [0, 1, 2, 118, 119]] = 0.25
    goal_xy = torch.tensor([[4.0, 0.0]])
    current_twist = torch.zeros(1, 2)
    action, _ = controller.command(clearance, goal_xy, current_twist)
    unsafe_straight = (action[0, 0] > 0.20) and (action[0, 1].abs() < 0.10)
    assert not bool(unsafe_straight)


if __name__ == "__main__":
    tests = [
        test_wheel_projection_preserves_curvature_and_limits,
        test_rectangle_clearance_detects_front_and_side_points,
        test_clearance_reconstruction_honors_lidar_offset_and_no_return,
        test_mppi_drives_forward_in_open_space,
        test_mppi_action_respects_joint_wheel_and_body_limits,
        test_mppi_does_not_keep_driving_straight_into_front_obstacle,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
