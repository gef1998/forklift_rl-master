"""Pure Torch geometry for the S-shaped curriculum corridor."""

from __future__ import annotations

import math

import torch


def _outer_wall_specs(
    arena_size: float,
    wall_thickness: float,
) -> list[tuple[str, float, float, float, float]]:
    half = arena_size / 2.0
    center = half - wall_thickness / 2.0
    return [
        ("TopWall", 0.0, center, arena_size, wall_thickness),
        ("BottomWall", 0.0, -center, arena_size, wall_thickness),
        ("LeftWall", -center, 0.0, wall_thickness, arena_size),
        ("RightWall", center, 0.0, wall_thickness, arena_size),
    ]


def validate_s_corridor_geometry(
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    inner_wall_offset: float,
    gate_width: float,
    spawn_wall_margin: float,
    spawn_jitter_x: float,
    spawn_jitter_y: float,
    min_goal_distance: float,
) -> None:
    """Reject layouts that cannot form two gates and safe end spawn regions."""

    values = {
        "arena_size": arena_size,
        "wall_thickness": wall_thickness,
        "interior_wall_thickness": interior_wall_thickness,
        "inner_wall_offset": inner_wall_offset,
        "gate_width": gate_width,
        "spawn_wall_margin": spawn_wall_margin,
        "spawn_jitter_x": spawn_jitter_x,
        "spawn_jitter_y": spawn_jitter_y,
        "min_goal_distance": min_goal_distance,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("corridor geometry values must be finite")
    if arena_size <= 0.0 or wall_thickness <= 0.0 or interior_wall_thickness <= 0.0:
        raise ValueError("wall dimensions and arena_size must be positive")
    if min(
        inner_wall_offset,
        gate_width,
        spawn_wall_margin,
        min_goal_distance,
    ) <= 0.0:
        raise ValueError("corridor offsets, gate, margin, and goal distance must be positive")
    if spawn_jitter_x < 0.0 or spawn_jitter_y < 0.0:
        raise ValueError("spawn jitter must be non-negative")

    inner_edge = arena_size / 2.0 - wall_thickness
    if inner_edge <= 0.0:
        raise ValueError("outer walls leave no usable arena interior")
    if gate_width >= 2.0 * inner_edge:
        raise ValueError("gate_width must be smaller than the arena interior width")
    if inner_wall_offset + interior_wall_thickness / 2.0 >= inner_edge:
        raise ValueError("interior barriers do not fit inside the outer walls")

    spawn_y = inner_edge - spawn_wall_margin
    if spawn_y <= inner_wall_offset + interior_wall_thickness / 2.0 + spawn_jitter_y:
        raise ValueError("spawn regions overlap the interior barriers")
    if spawn_jitter_x >= inner_edge:
        raise ValueError("spawn_jitter_x reaches an outer wall")
    minimum_separation = 2.0 * (spawn_y - spawn_jitter_y)
    if min_goal_distance > minimum_separation + 1.0e-6:
        raise ValueError("min_goal_distance is impossible for the configured spawn regions")


def s_corridor_wall_specs(
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    inner_wall_offset: float,
    gate_width: float,
    mirrored: bool = False,
) -> list[tuple[str, float, float, float, float]]:
    """Return outer walls and two alternating barriers.

    Each tuple is ``(name, center_x, center_y, size_x, size_y)``. The lower
    barrier leaves its opening at the left wall and the upper barrier leaves
    its opening at the right wall. ``mirrored`` swaps those openings.
    """

    if arena_size <= 0.0 or wall_thickness <= 0.0 or interior_wall_thickness <= 0.0:
        raise ValueError("wall dimensions and arena_size must be positive")
    inner_edge = arena_size / 2.0 - wall_thickness
    barrier_length = 2.0 * inner_edge - gate_width
    if barrier_length <= 0.0:
        raise ValueError("gate_width leaves no interior barrier")
    direction = -1.0 if mirrored else 1.0
    lower_x = direction * gate_width / 2.0
    upper_x = -direction * gate_width / 2.0
    return _outer_wall_specs(arena_size, wall_thickness) + [
        (
            "LowerRightBarrier",
            lower_x,
            -inner_wall_offset,
            barrier_length,
            interior_wall_thickness,
        ),
        (
            "UpperLeftBarrier",
            upper_x,
            inner_wall_offset,
            barrier_length,
            interior_wall_thickness,
        ),
    ]


def sample_s_corridor_spawn_pairs(
    num_envs: int,
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    inner_wall_offset: float,
    gate_width: float,
    spawn_wall_margin: float,
    spawn_jitter_x: float,
    spawn_jitter_y: float,
    min_goal_distance: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample opposite end points and random initial headings for the corridor."""

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    validate_s_corridor_geometry(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        interior_wall_thickness=interior_wall_thickness,
        inner_wall_offset=inner_wall_offset,
        gate_width=gate_width,
        spawn_wall_margin=spawn_wall_margin,
        spawn_jitter_x=spawn_jitter_x,
        spawn_jitter_y=spawn_jitter_y,
        min_goal_distance=min_goal_distance,
    )
    dtype = torch.float32
    end_y = arena_size / 2.0 - wall_thickness - spawn_wall_margin
    emma_end = torch.randint(
        0,
        2,
        (num_envs,),
        device=device,
        generator=generator,
    )
    goal_end = 1 - emma_end
    end_sign = torch.where(emma_end == 0, -1.0, 1.0)

    def jitter(scale: float) -> torch.Tensor:
        return (
            torch.rand(num_envs, device=device, dtype=dtype, generator=generator)
            * 2.0
            - 1.0
        ) * scale

    emma = torch.stack(
        (jitter(spawn_jitter_x), end_sign * end_y + jitter(spawn_jitter_y)),
        dim=-1,
    )
    goal = torch.stack(
        (jitter(spawn_jitter_x), -end_sign * end_y + jitter(spawn_jitter_y)),
        dim=-1,
    )
    pair_distance = torch.linalg.norm(goal - emma, dim=-1)
    if (pair_distance < min_goal_distance).any():
        # Validation guarantees a feasible region. Resample only the small
        # endpoint jitter until every requested pair satisfies the threshold.
        pending = pair_distance < min_goal_distance
        for _ in range(128):
            if not pending.any():
                break
            count = int(pending.sum().item())
            emma[pending, 0] = (
                torch.rand(count, device=device, generator=generator) * 2.0 - 1.0
            ) * spawn_jitter_x
            emma[pending, 1] = (
                end_sign[pending] * end_y
                + (torch.rand(count, device=device, generator=generator) * 2.0 - 1.0)
                * spawn_jitter_y
            )
            goal[pending, 0] = (
                torch.rand(count, device=device, generator=generator) * 2.0 - 1.0
            ) * spawn_jitter_x
            goal[pending, 1] = (
                -end_sign[pending] * end_y
                + (torch.rand(count, device=device, generator=generator) * 2.0 - 1.0)
                * spawn_jitter_y
            )
            pair_distance = torch.linalg.norm(goal - emma, dim=-1)
            pending = pair_distance < min_goal_distance
        if pending.any():
            raise RuntimeError("unable to sample S-corridor spawn pairs")

    yaw = torch.rand(num_envs, device=device, dtype=dtype, generator=generator) * (
        2.0 * math.pi
    )
    return emma, yaw, goal, emma_end, goal_end
