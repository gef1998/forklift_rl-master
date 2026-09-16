#!/usr/bin/env python3
"""Pure Torch tests for the straight-obstacle evaluation geometry."""

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
    / "straight_obstacle_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("straight_obstacle_geometry", MODULE_PATH)
geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(geometry)


ARENA_LENGTH = 12.0
PASSAGE_WIDTH = 1.5
WALL_THICKNESS = 0.15
OBSTACLE_LENGTH = 1.0
OBSTACLE_WIDTH = 0.8


def _specs():
    return geometry.straight_obstacle_specs(
        arena_length=ARENA_LENGTH,
        passage_width=PASSAGE_WIDTH,
        wall_thickness=WALL_THICKNESS,
        wall_height=1.0,
        obstacle_length=OBSTACLE_LENGTH,
        obstacle_width=OBSTACLE_WIDTH,
        obstacle_height=1.0,
    )


def test_obstacle_has_exact_equal_side_passages():
    specs = {name: values for name, *values in _specs()}
    arena_width = geometry.straight_obstacle_arena_width(
        PASSAGE_WIDTH, OBSTACLE_WIDTH, WALL_THICKNESS
    )
    top_inner_y = specs["TopWall"][1] - specs["TopWall"][3] / 2.0
    obstacle_top_y = specs["CenterObstacle"][1] + specs["CenterObstacle"][3] / 2.0
    assert abs((top_inner_y - obstacle_top_y) - PASSAGE_WIDTH) < 1.0e-6
    assert arena_width == 2.0 * PASSAGE_WIDTH + OBSTACLE_WIDTH + 2.0 * WALL_THICKNESS


def test_spawn_and_goal_are_opposite_and_centered():
    torch.manual_seed(17)
    emma, yaw, goal, emma_end, goal_end = geometry.sample_straight_obstacle_spawn_pairs(
        1024,
        arena_length=ARENA_LENGTH,
        passage_width=PASSAGE_WIDTH,
        wall_thickness=WALL_THICKNESS,
        obstacle_length=OBSTACLE_LENGTH,
        obstacle_width=OBSTACLE_WIDTH,
        spawn_wall_margin=1.0,
        robot_half_width=0.267,
        device="cpu",
    )
    assert (emma[:, 1] == 0.0).all() and (goal[:, 1] == 0.0).all()
    assert torch.equal(goal, -emma)
    assert (goal_end == 1 - emma_end).all()
    direction = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    to_goal = torch.nn.functional.normalize(goal - emma, dim=-1)
    torch.testing.assert_close(direction, to_goal, atol=1.0e-6, rtol=0.0)


def test_passage_narrower_than_robot_is_rejected():
    try:
        geometry.validate_straight_obstacle_geometry(
            arena_length=ARENA_LENGTH,
            passage_width=0.5,
            wall_thickness=WALL_THICKNESS,
            obstacle_length=OBSTACLE_LENGTH,
            obstacle_width=OBSTACLE_WIDTH,
            spawn_wall_margin=1.0,
            robot_half_width=0.267,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a passage narrower than Emma was accepted")


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
