#!/usr/bin/env python3
"""Pure Torch tests for the 10 m mixed-obstacle local maze."""

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
    / "ring_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("ring_geometry", MODULE_PATH)
ring_geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ring_geometry)


ARENA_SIZE = 10.0
WALL_THICKNESS = 0.15
INTERIOR_WALL_THICKNESS = 0.15
CORRIDOR_WIDTH = 2.0
WALL_TIP_OFFSET = 1.0
OBSTACLE_WALL_CLEARANCE = 0.05
PASSAGE_SIDE_CLEARANCE = 0.30
MIN_PASSAGE_WIDTH = 1.15
SPAWN_EXCLUSION_RADIUS = 1.20
GATE_EXCLUSION_RADIUS = 0.75
MIN_EDGE_SPACING = 0.40
GOAL_MIN_DISTANCE = 7.0


def _wall_specs():
    return ring_geometry.maze_wall_specs(
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        interior_wall_thickness=INTERIOR_WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        wall_tip_offset=WALL_TIP_OFFSET,
    )


def _mixed_bounds(count: int = 20) -> torch.Tensor:
    cylinder = 0.30
    cube = math.sqrt(2.0) * 0.20
    return torch.tensor(
        [cylinder if index % 2 == 0 else cube for index in range(count)]
    )


def test_phase_templates_contain_both_obstacle_types_evenly():
    types = ring_geometry.alternating_obstacle_types(20)
    assert types.count("cylinder") == 10
    assert types.count("cube") == 10


def test_maze_contains_four_outer_and_four_staggered_inner_walls():
    specs = {name: (cx, cy, sx, sy) for name, cx, cy, sx, sy in _wall_specs()}
    assert len(specs) == 8

    top = specs["TopWall"]
    upper = specs["UpperLeftWall"]
    top_inner_face = top[1] - top[3] / 2.0
    upper_outer_face = upper[1] + upper[3] / 2.0
    assert abs((top_inner_face - upper_outer_face) - CORRIDOR_WIDTH) < 1.0e-6

    right = specs["RightWall"]
    upper_right = specs["UpperRightWall"]
    right_inner_face = right[0] - right[2] / 2.0
    inner_right_face = upper_right[0] + upper_right[2] / 2.0
    assert abs((right_inner_face - inner_right_face) - CORRIDOR_WIDTH) < 1.0e-6

    assert specs["UpperLeftWall"][0] < 0.0
    assert specs["UpperRightWall"][1] > 0.0
    assert specs["LowerLeftWall"][1] < 0.0
    assert specs["LowerRightWall"][0] > 0.0
    expected_inner_wall_length = ARENA_SIZE / 4.0 + WALL_TIP_OFFSET
    assert abs(specs["UpperLeftWall"][2] - expected_inner_wall_length) < 1.0e-6
    assert abs(specs["UpperRightWall"][3] - expected_inner_wall_length) < 1.0e-6


def test_spawn_and_goal_use_different_safe_corner_ends():
    torch.manual_seed(3)
    emma, yaw, goal, emma_corner, goal_corner = (
        ring_geometry.sample_maze_spawn_pairs(
            4096,
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            corridor_width=CORRIDOR_WIDTH,
            spawn_jitter=0.25,
            min_goal_distance=GOAL_MIN_DISTANCE,
            device="cpu",
        )
    )
    anchors = ring_geometry.maze_corner_anchors(
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        device="cpu",
    )
    assert (emma_corner != goal_corner).all()
    assert torch.linalg.norm(goal - emma, dim=-1).min() >= GOAL_MIN_DISTANCE
    assert (emma - anchors[emma_corner]).abs().amax() <= 0.25 + 1.0e-6
    assert (goal - anchors[goal_corner]).abs().amax() <= 0.25 + 1.0e-6
    assert (yaw >= 0.0).all() and (yaw < 2.0 * math.pi).all()
    assert yaw.std() > 1.0


def test_goal_distance_rejection_resamples_only_invalid_pairs():
    """较长阈值应筛掉相邻角落，而不是因一个无效样本终止整批环境。"""

    torch.manual_seed(17)
    emma, _, goal, emma_corner, goal_corner = (
        ring_geometry.sample_maze_spawn_pairs(
            4096,
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            corridor_width=CORRIDOR_WIDTH,
            spawn_jitter=0.25,
            min_goal_distance=9.0,
            device="cpu",
        )
    )
    assert (emma_corner != goal_corner).all()
    assert torch.linalg.norm(goal - emma, dim=-1).min() >= 9.0


def test_impossible_goal_distance_is_rejected_before_sampling():
    try:
        ring_geometry.sample_maze_spawn_pairs(
            1,
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            corridor_width=CORRIDOR_WIDTH,
            spawn_jitter=0.25,
            min_goal_distance=20.0,
            device="cpu",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible goal distance was accepted")


def test_random_obstacles_leave_walls_spawns_gates_and_passages_safe():
    torch.manual_seed(5)
    bounds = _mixed_bounds()
    positions = ring_geometry.sample_maze_obstacle_positions(
        16,
        bounds,
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        interior_wall_thickness=INTERIOR_WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        wall_tip_offset=WALL_TIP_OFFSET,
        obstacle_wall_clearance=OBSTACLE_WALL_CLEARANCE,
        min_passage_width=MIN_PASSAGE_WIDTH,
        spawn_exclusion_radius=SPAWN_EXCLUSION_RADIUS,
        gate_exclusion_radius=GATE_EXCLUSION_RADIUS,
        min_edge_spacing=MIN_EDGE_SPACING,
    )

    flat = positions.reshape(-1, 2)
    repeated_bounds = bounds.repeat(positions.shape[0])
    wall_distance = ring_geometry._point_to_wall_distance(flat, _wall_specs())
    assert (
        wall_distance
        >= repeated_bounds.unsqueeze(-1) + OBSTACLE_WALL_CLEARANCE - 1.0e-6
    ).all()

    anchors = ring_geometry.maze_corner_anchors(
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        device="cpu",
    )
    spawn_distance = torch.linalg.norm(
        positions.unsqueeze(2) - anchors.view(1, 1, 4, 2), dim=-1
    )
    assert (
        spawn_distance
        >= bounds.view(1, -1, 1) + SPAWN_EXCLUSION_RADIUS - 1.0e-6
    ).all()

    gates = ring_geometry.maze_gate_centers(
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        interior_wall_thickness=INTERIOR_WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        wall_tip_offset=WALL_TIP_OFFSET,
        device="cpu",
    )
    gate_distance = torch.linalg.norm(
        positions.unsqueeze(2) - gates.view(1, 1, 4, 2), dim=-1
    )
    assert (
        gate_distance
        >= bounds.view(1, -1, 1) + GATE_EXCLUSION_RADIUS - 1.0e-6
    ).all()

    passage_safe = ring_geometry._corridor_passage_is_valid(
        flat,
        repeated_bounds,
        inner_edge=ARENA_SIZE / 2.0 - WALL_THICKNESS,
        corridor_width=CORRIDOR_WIDTH,
        min_passage_width=MIN_PASSAGE_WIDTH,
    )
    assert passage_safe.all()
    emma_width = 2.0 * 0.267
    assert MIN_PASSAGE_WIDTH >= emma_width + 2.0 * PASSAGE_SIDE_CLEARANCE

    pair_distance = torch.cdist(positions, positions)
    required = (
        bounds.view(1, -1, 1)
        + bounds.view(1, 1, -1)
        + MIN_EDGE_SPACING
    )
    diagonal = torch.eye(bounds.numel(), dtype=torch.bool).unsqueeze(0)
    assert (pair_distance.masked_fill(diagonal, float("inf")) >= required).all()


def test_invalid_corridor_geometry_is_rejected():
    try:
        ring_geometry.validate_maze_geometry(
            arena_size=4.0,
            wall_thickness=0.15,
            interior_wall_thickness=0.15,
            corridor_width=2.0,
            wall_tip_offset=1.0,
            spawn_jitter=0.25,
            min_passage_width=MIN_PASSAGE_WIDTH,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible corridor layout was accepted")


if __name__ == "__main__":
    tests = [
        test_phase_templates_contain_both_obstacle_types_evenly,
        test_maze_contains_four_outer_and_four_staggered_inner_walls,
        test_spawn_and_goal_use_different_safe_corner_ends,
        test_goal_distance_rejection_resamples_only_invalid_pairs,
        test_impossible_goal_distance_is_rejected_before_sampling,
        test_random_obstacles_leave_walls_spawns_gates_and_passages_safe,
        test_invalid_corridor_geometry_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
