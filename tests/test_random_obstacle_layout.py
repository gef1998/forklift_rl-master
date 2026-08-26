#!/usr/bin/env python3
"""随机圆柱/方柱课程场景的纯 Torch 几何测试。"""

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
    / "random_obstacle_geometry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "random_obstacle_geometry", MODULE_PATH
)
random_obstacle_geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(random_obstacle_geometry)


ARENA_SIZE = 10.0
WALL_THICKNESS = 0.15
NUM_OBSTACLES = 12
MIN_EDGE_SPACING = 1.50
OBSTACLE_WALL_CLEARANCE = 0.30
EMMA_CLEARANCE_RADIUS = 0.75
GOAL_CLEARANCE_RADIUS = 0.60
GOAL_MIN_DISTANCE = 5.20


def _bound_radii() -> torch.Tensor:
    """使用配置上限验证最保守的圆柱/方柱占位。"""

    cylinder_radius = 0.30
    cube_bound_radius = 0.5 * math.hypot(0.50, 0.50)
    return torch.tensor(
        [
            cylinder_radius if index % 2 == 0 else cube_bound_radius
            for index in range(NUM_OBSTACLES)
        ],
        dtype=torch.float32,
    )


def _sample_layout(num_envs: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    radii = _bound_radii()
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    positions = random_obstacle_geometry.sample_random_obstacle_positions(
        num_envs,
        radii,
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        min_edge_spacing=MIN_EDGE_SPACING,
        obstacle_wall_clearance=OBSTACLE_WALL_CLEARANCE,
        candidate_pool_size=48,
        generator=generator,
    )
    return positions, radii


def test_world_contains_exactly_four_thin_outer_walls():
    specs = random_obstacle_geometry.outer_wall_specs(
        ARENA_SIZE, WALL_THICKNESS
    )
    assert len(specs) == 4
    names = {name for name, *_ in specs}
    assert names == {"TopWall", "BottomWall", "LeftWall", "RightWall"}
    for name, _, _, size_x, size_y in specs:
        if name in ("TopWall", "BottomWall"):
            assert size_x == ARENA_SIZE and size_y == WALL_THICKNESS
        else:
            assert size_x == WALL_THICKNESS and size_y == ARENA_SIZE


def test_obstacle_templates_alternate_and_are_balanced():
    types = random_obstacle_geometry.alternating_obstacle_types(NUM_OBSTACLES)
    assert types[::2] == ["cylinder"] * (NUM_OBSTACLES // 2)
    assert types[1::2] == ["cube"] * (NUM_OBSTACLES // 2)
    assert types.count("cylinder") == types.count("cube") == 6


def test_random_obstacle_edges_are_at_least_1p5m_apart():
    positions, radii = _sample_layout()
    # 并行环境不能共享同一份布局，否则 1024 环境仍只是一张训练地图。
    unique_layouts = torch.unique(positions.reshape(positions.shape[0], -1), dim=0)
    assert unique_layouts.shape[0] == positions.shape[0]
    center_distance = torch.cdist(positions, positions)
    edge_clearance = (
        center_distance
        - radii.view(1, -1, 1)
        - radii.view(1, 1, -1)
    )
    diagonal = torch.eye(NUM_OBSTACLES, dtype=torch.bool).unsqueeze(0)
    edge_clearance = edge_clearance.masked_fill(diagonal, torch.inf)
    assert edge_clearance.amin().item() >= MIN_EDGE_SPACING - 1.0e-5


def test_random_obstacles_keep_configured_clearance_from_outer_walls():
    positions, radii = _sample_layout()
    inner_half = ARENA_SIZE / 2.0 - WALL_THICKNESS
    wall_clearance = inner_half - positions.abs() - radii.view(1, -1, 1)
    assert wall_clearance.amin().item() >= OBSTACLE_WALL_CLEARANCE - 1.0e-5


def test_dead_end_sampling_uses_a_safe_randomized_fallback():
    radii = _bound_radii()
    generator = torch.Generator(device="cpu").manual_seed(20260805)
    # 单候选、单次尝试会刻意让大量贪心序列陷入死角，用于覆盖回退路径。
    positions = random_obstacle_geometry.sample_random_obstacle_positions(
        64,
        radii,
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        min_edge_spacing=MIN_EDGE_SPACING,
        obstacle_wall_clearance=OBSTACLE_WALL_CLEARANCE,
        candidate_pool_size=1,
        max_attempts=1,
        generator=generator,
    )
    center_distance = torch.cdist(positions, positions)
    edge_clearance = (
        center_distance
        - radii.view(1, -1, 1)
        - radii.view(1, 1, -1)
    )
    diagonal = torch.eye(NUM_OBSTACLES, dtype=torch.bool).unsqueeze(0)
    edge_clearance = edge_clearance.masked_fill(diagonal, torch.inf)
    assert edge_clearance.amin().item() >= MIN_EDGE_SPACING - 1.0e-5

    inner_half = ARENA_SIZE / 2.0 - WALL_THICKNESS
    wall_clearance = inner_half - positions.abs() - radii.view(1, -1, 1)
    assert wall_clearance.amin().item() >= OBSTACLE_WALL_CLEARANCE - 1.0e-5
    assert torch.unique(positions.reshape(64, -1), dim=0).shape[0] == 64


def test_emma_and_goal_spawn_safely_more_than_5m_apart():
    positions, radii = _sample_layout()
    generator = torch.Generator(device="cpu").manual_seed(20260804)
    emma, yaw, goal, _, _ = random_obstacle_geometry.sample_safe_spawn_pairs(
        positions,
        radii,
        arena_size=ARENA_SIZE,
        wall_thickness=WALL_THICKNESS,
        emma_clearance_radius=EMMA_CLEARANCE_RADIUS,
        goal_clearance_radius=GOAL_CLEARANCE_RADIUS,
        min_goal_distance=GOAL_MIN_DISTANCE,
        candidate_pool_size=128,
        generator=generator,
    )

    distance = torch.linalg.norm(goal - emma, dim=-1)
    assert (distance > GOAL_MIN_DISTANCE).all()
    assert (yaw >= 0.0).all() and (yaw < 2.0 * torch.pi).all()

    emma_obstacle_distance = torch.linalg.norm(
        emma.unsqueeze(1) - positions, dim=-1
    )
    goal_obstacle_distance = torch.linalg.norm(
        goal.unsqueeze(1) - positions, dim=-1
    )
    assert (
        emma_obstacle_distance
        >= radii.view(1, -1) + EMMA_CLEARANCE_RADIUS - 1.0e-5
    ).all()
    assert (
        goal_obstacle_distance
        >= radii.view(1, -1) + GOAL_CLEARANCE_RADIUS - 1.0e-5
    ).all()

    inner_half = ARENA_SIZE / 2.0 - WALL_THICKNESS
    assert (
        inner_half - emma.abs() >= EMMA_CLEARANCE_RADIUS - 1.0e-5
    ).all()
    assert (
        inner_half - goal.abs() >= GOAL_CLEARANCE_RADIUS - 1.0e-5
    ).all()


def test_impossible_obstacle_density_is_rejected():
    try:
        random_obstacle_geometry.sample_random_obstacle_positions(
            1,
            torch.tensor([0.80]),
            arena_size=2.0,
            wall_thickness=0.15,
            min_edge_spacing=MIN_EDGE_SPACING,
            obstacle_wall_clearance=0.30,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "an obstacle that cannot fit inside the arena was accepted"
        )


def test_impossible_goal_distance_is_rejected():
    try:
        random_obstacle_geometry.sample_safe_spawn_pairs(
            torch.empty(1, 0, 2),
            torch.empty(0),
            arena_size=ARENA_SIZE,
            wall_thickness=WALL_THICKNESS,
            emma_clearance_radius=EMMA_CLEARANCE_RADIUS,
            goal_clearance_radius=GOAL_CLEARANCE_RADIUS,
            min_goal_distance=20.0,
            candidate_pool_size=8,
            max_attempts=1,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("an impossible Emma-to-goal distance was accepted")


if __name__ == "__main__":
    tests = [
        test_world_contains_exactly_four_thin_outer_walls,
        test_obstacle_templates_alternate_and_are_balanced,
        test_random_obstacle_edges_are_at_least_1p5m_apart,
        test_random_obstacles_keep_configured_clearance_from_outer_walls,
        test_dead_end_sampling_uses_a_safe_randomized_fallback,
        test_emma_and_goal_spawn_safely_more_than_5m_apart,
        test_impossible_obstacle_density_is_rejected,
        test_impossible_goal_distance_is_rejected,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
