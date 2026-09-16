"""Pure geometry for the documented straight-line obstacle test."""

from __future__ import annotations

import math

import torch


def validate_straight_obstacle_geometry(
    *,
    arena_length: float,
    passage_width: float,
    wall_thickness: float,
    obstacle_length: float,
    obstacle_width: float,
    spawn_wall_margin: float,
    robot_half_width: float = 0.0,
) -> None:
    """Validate a channel whose obstacle has equal clear passages on both sides."""

    values = {
        "arena_length": arena_length,
        "passage_width": passage_width,
        "wall_thickness": wall_thickness,
        "obstacle_length": obstacle_length,
        "obstacle_width": obstacle_width,
        "spawn_wall_margin": spawn_wall_margin,
        "robot_half_width": robot_half_width,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("straight-obstacle geometry values must be finite")
    if min(
        arena_length,
        passage_width,
        wall_thickness,
        obstacle_length,
        obstacle_width,
        spawn_wall_margin,
    ) <= 0.0:
        raise ValueError("straight-obstacle dimensions must be positive")
    if robot_half_width < 0.0:
        raise ValueError("robot_half_width must be non-negative")
    if passage_width <= 2.0 * robot_half_width:
        raise ValueError("passage_width must exceed the robot full width")
    if obstacle_length >= arena_length - 2.0 * (wall_thickness + spawn_wall_margin):
        raise ValueError("obstacle leaves no safe spawn and goal regions")


def straight_obstacle_arena_width(
    passage_width: float,
    obstacle_width: float,
    wall_thickness: float,
) -> float:
    """Return total outside width for two equal clear passages."""

    return 2.0 * passage_width + obstacle_width + 2.0 * wall_thickness


def straight_obstacle_specs(
    *,
    arena_length: float,
    passage_width: float,
    wall_thickness: float,
    wall_height: float,
    obstacle_length: float,
    obstacle_width: float,
    obstacle_height: float,
) -> list[tuple[str, float, float, float, float, float]]:
    """Return ``(name, cx, cy, sx, sy, sz)`` for walls and the obstacle."""

    validate_straight_obstacle_geometry(
        arena_length=arena_length,
        passage_width=passage_width,
        wall_thickness=wall_thickness,
        obstacle_length=obstacle_length,
        obstacle_width=obstacle_width,
        spawn_wall_margin=wall_thickness,
    )
    if wall_height <= 0.0 or obstacle_height <= 0.0:
        raise ValueError("wall and obstacle heights must be positive")
    arena_width = straight_obstacle_arena_width(
        passage_width, obstacle_width, wall_thickness
    )
    wall_y = arena_width / 2.0 - wall_thickness / 2.0
    wall_x = arena_length / 2.0 - wall_thickness / 2.0
    return [
        ("TopWall", 0.0, wall_y, arena_length, wall_thickness, wall_height),
        ("BottomWall", 0.0, -wall_y, arena_length, wall_thickness, wall_height),
        ("LeftWall", -wall_x, 0.0, wall_thickness, arena_width, wall_height),
        ("RightWall", wall_x, 0.0, wall_thickness, arena_width, wall_height),
        ("CenterObstacle", 0.0, 0.0, obstacle_length, obstacle_width, obstacle_height),
    ]


def sample_straight_obstacle_spawn_pairs(
    num_envs: int,
    *,
    arena_length: float,
    passage_width: float,
    wall_thickness: float,
    obstacle_length: float,
    obstacle_width: float,
    spawn_wall_margin: float,
    robot_half_width: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place Emma and goal at opposite channel ends on the original centerline."""

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    validate_straight_obstacle_geometry(
        arena_length=arena_length,
        passage_width=passage_width,
        wall_thickness=wall_thickness,
        obstacle_length=obstacle_length,
        obstacle_width=obstacle_width,
        spawn_wall_margin=spawn_wall_margin,
        robot_half_width=robot_half_width,
    )
    end_x = arena_length / 2.0 - wall_thickness - spawn_wall_margin
    emma_end = torch.randint(
        0,
        2,
        (num_envs,),
        device=device,
        generator=generator,
    )
    direction = torch.where(emma_end == 0, 1.0, -1.0)
    emma = torch.stack((-direction * end_x, torch.zeros_like(direction)), dim=-1)
    goal = torch.stack((direction * end_x, torch.zeros_like(direction)), dim=-1)
    yaw = torch.where(
        direction > 0.0,
        torch.zeros_like(direction),
        torch.full_like(direction, math.pi),
    )
    return emma, yaw, goal, emma_end, 1 - emma_end
