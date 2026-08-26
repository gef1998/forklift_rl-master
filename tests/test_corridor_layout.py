#!/usr/bin/env python3
"""Pure Torch tests for the S-shaped corridor curriculum world."""

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
    / "corridor_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("corridor_geometry", MODULE_PATH)
corridor_geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(corridor_geometry)


ARENA_SIZE = 10.0
WALL_THICKNESS = 0.15
INTERIOR_WALL_THICKNESS = 0.20
INNER_WALL_OFFSET = 1.50
GATE_WIDTH = 2.50
SPAWN_WALL_MARGIN = 1.0
SPAWN_JITTER_X = 0.75
SPAWN_JITTER_Y = 0.20
GOAL_MIN_DISTANCE = 7.0


def _wall_specs():
    return corridor_geometry.s_corridor_wall_specs(
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        interior_wall_thickness=INTERIOR_WALL_THICKNESS,
        inner_wall_offset=INNER_WALL_OFFSET,
        gate_width=GATE_WIDTH,
    )


def test_corridor_contains_four_outer_and_two_staggered_walls():
    specs = {name: (cx, cy, sx, sy) for name, cx, cy, sx, sy in _wall_specs()}
    assert len(specs) == 6
    assert set(specs) == {
        "TopWall",
        "BottomWall",
        "LeftWall",
        "RightWall",
        "LowerRightBarrier",
        "UpperLeftBarrier",
    }
    lower = specs["LowerRightBarrier"]
    upper = specs["UpperLeftBarrier"]
    assert lower[0] > 0.0 and lower[1] < 0.0
    assert upper[0] < 0.0 and upper[1] > 0.0
    assert lower[2] == upper[2]
    assert lower[3] == upper[3] == INTERIOR_WALL_THICKNESS


def test_alternating_barriers_leave_exact_configured_gate_width():
    specs = {name: (cx, cy, sx, sy) for name, cx, cy, sx, sy in _wall_specs()}
    inner_edge = ARENA_SIZE / 2.0 - WALL_THICKNESS
    lower = specs["LowerRightBarrier"]
    upper = specs["UpperLeftBarrier"]
    lower_left_tip = lower[0] - lower[2] / 2.0
    upper_right_tip = upper[0] + upper[2] / 2.0
    assert abs((lower_left_tip - (-inner_edge)) - GATE_WIDTH) < 1.0e-6
    assert abs((inner_edge - upper_right_tip) - GATE_WIDTH) < 1.0e-6


def test_mirrored_layout_swaps_left_and_right_gates():
    original = {
        name: (cx, cy, sx, sy) for name, cx, cy, sx, sy in _wall_specs()
    }
    mirrored = {
        name: (cx, cy, sx, sy)
        for name, cx, cy, sx, sy in corridor_geometry.s_corridor_wall_specs(
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            interior_wall_thickness=INTERIOR_WALL_THICKNESS,
            inner_wall_offset=INNER_WALL_OFFSET,
            gate_width=GATE_WIDTH,
            mirrored=True,
        )
    }
    for name in ("LowerRightBarrier", "UpperLeftBarrier"):
        assert mirrored[name][0] == -original[name][0]
        assert mirrored[name][1:] == original[name][1:]


def test_spawn_and_goal_swap_opposite_corridor_ends_safely():
    torch.manual_seed(29)
    emma, yaw, goal, emma_end, goal_end = (
        corridor_geometry.sample_s_corridor_spawn_pairs(
            4096,
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            interior_wall_thickness=INTERIOR_WALL_THICKNESS,
            inner_wall_offset=INNER_WALL_OFFSET,
            gate_width=GATE_WIDTH,
            spawn_wall_margin=SPAWN_WALL_MARGIN,
            spawn_jitter_x=SPAWN_JITTER_X,
            spawn_jitter_y=SPAWN_JITTER_Y,
            min_goal_distance=GOAL_MIN_DISTANCE,
            device="cpu",
        )
    )
    assert (goal_end == 1 - emma_end).all()
    assert (emma_end == 0).any() and (emma_end == 1).any()
    assert torch.linalg.norm(goal - emma, dim=-1).min() >= GOAL_MIN_DISTANCE
    assert emma[:, 0].abs().max() <= SPAWN_JITTER_X + 1.0e-6
    assert goal[:, 0].abs().max() <= SPAWN_JITTER_X + 1.0e-6
    assert (yaw >= 0.0).all() and (yaw < 2.0 * torch.pi).all()


def test_direct_centerline_to_goal_is_blocked_by_both_barriers():
    specs = {name: (cx, cy, sx, sy) for name, cx, cy, sx, sy in _wall_specs()}
    for name in ("LowerRightBarrier", "UpperLeftBarrier"):
        center_x, _, size_x, _ = specs[name]
        assert center_x - size_x / 2.0 < 0.0 < center_x + size_x / 2.0


def test_impossible_corridor_goal_distance_is_rejected():
    try:
        corridor_geometry.sample_s_corridor_spawn_pairs(
            1,
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            interior_wall_thickness=INTERIOR_WALL_THICKNESS,
            inner_wall_offset=INNER_WALL_OFFSET,
            gate_width=GATE_WIDTH,
            spawn_wall_margin=SPAWN_WALL_MARGIN,
            spawn_jitter_x=SPAWN_JITTER_X,
            spawn_jitter_y=SPAWN_JITTER_Y,
            min_goal_distance=9.0,
            device="cpu",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible corridor goal distance was accepted")


if __name__ == "__main__":
    tests = [
        test_corridor_contains_four_outer_and_two_staggered_walls,
        test_alternating_barriers_leave_exact_configured_gate_width,
        test_mirrored_layout_swaps_left_and_right_gates,
        test_spawn_and_goal_swap_opposite_corridor_ends_safely,
        test_direct_centerline_to_goal_is_blocked_by_both_barriers,
        test_impossible_corridor_goal_distance_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
