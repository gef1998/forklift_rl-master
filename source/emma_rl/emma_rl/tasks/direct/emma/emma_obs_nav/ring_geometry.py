"""Pure Torch geometry helpers for the bounded Emma maze world.

The runtime mode remains named ``ring`` for CLI/checkpoint compatibility, but
the generated scene is a small local-navigation maze rather than a radial
obstacle ring.  This module intentionally has no Isaac Sim dependency so its
geometry invariants can be tested with ordinary PyTorch.
"""

from __future__ import annotations

import math

import torch


# (name, center_x, center_y, size_x, size_y)
WallSpec = tuple[str, float, float, float, float]


def alternating_obstacle_types(count: int) -> tuple[str, ...]:
    """Return a deterministic 50/50 cylinder/cube template sequence."""

    if count < 0:
        raise ValueError("count must be non-negative")
    return tuple("cylinder" if index % 2 == 0 else "cube" for index in range(count))


def validate_maze_geometry(
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    corridor_width: float,
    wall_tip_offset: float,
    spawn_jitter: float,
    min_passage_width: float,
) -> None:
    """Validate the dimensions shared by wall, spawn, and obstacle sampling."""

    values = (
        arena_size,
        wall_thickness,
        interior_wall_thickness,
        corridor_width,
        wall_tip_offset,
        spawn_jitter,
        min_passage_width,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("maze geometry values must be finite")
    if arena_size <= 0.0 or wall_thickness <= 0.0:
        raise ValueError("arena_size and wall_thickness must be positive")
    if interior_wall_thickness <= 0.0:
        raise ValueError("interior_wall_thickness must be positive")
    if corridor_width <= 0.0:
        raise ValueError("corridor_width must be positive")
    if spawn_jitter < 0.0:
        raise ValueError("spawn_jitter must be non-negative")
    if min_passage_width <= 0.0:
        raise ValueError("min_passage_width must be positive")

    inner_edge = arena_size / 2.0 - wall_thickness
    interior_offset = (
        inner_edge - corridor_width - interior_wall_thickness / 2.0
    )
    if interior_offset <= 0.0:
        raise ValueError("corridor leaves no room for the interior maze")
    if wall_tip_offset < 0.0:
        raise ValueError("wall_tip_offset must be non-negative")
    segment_length = arena_size / 4.0 + wall_tip_offset
    segment_tip = -inner_edge + segment_length
    if not -interior_offset < segment_tip < interior_offset:
        raise ValueError(
            "the inner-wall tip must lie inside the central maze opening"
        )

    spawn_center = inner_edge - corridor_width / 2.0
    # 任意 yaw 出生时，走廊中心到两侧墙面必须容纳机器人半对角线。
    if spawn_center <= spawn_jitter:
        raise ValueError("spawn jitter does not fit inside the corner corridor")
    if min_passage_width >= corridor_width:
        raise ValueError("min_passage_width must be smaller than corridor_width")


def maze_wall_specs(
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    corridor_width: float,
    wall_tip_offset: float,
) -> tuple[WallSpec, ...]:
    """Return four boundary walls and four staggered interior walls.

    The distance from each outer wall's inner face to its parallel interior
    wall is exactly ``corridor_width``.  Alternating wall tips reproduce the
    sketched winding layout and force temporary motion away from some goals.
    """

    validate_maze_geometry(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        interior_wall_thickness=interior_wall_thickness,
        corridor_width=corridor_width,
        wall_tip_offset=wall_tip_offset,
        spawn_jitter=0.0,
        min_passage_width=min(0.5 * corridor_width, corridor_width - 1.0e-6),
    )
    arena_half = arena_size / 2.0
    outer_center = arena_half - wall_thickness / 2.0
    inner_edge = arena_half - wall_thickness
    interior_offset = (
        inner_edge - corridor_width - interior_wall_thickness / 2.0
    )
    # 内墙从外墙内沿开始，基础长度取场地边长的 1/4，再由 offset 控制
    # 向中央延伸量。这样缩小场地或 offset 时，绕行距离会同步缩短。
    segment_length = arena_size / 4.0 + wall_tip_offset
    segment_center = -inner_edge + segment_length / 2.0

    return (
        ("TopWall", 0.0, outer_center, arena_size, wall_thickness),
        ("BottomWall", 0.0, -outer_center, arena_size, wall_thickness),
        ("LeftWall", -outer_center, 0.0, wall_thickness, arena_size),
        ("RightWall", outer_center, 0.0, wall_thickness, arena_size),
        (
            "UpperLeftWall",
            segment_center,
            interior_offset,
            segment_length,
            interior_wall_thickness,
        ),
        (
            "UpperRightWall",
            interior_offset,
            -segment_center,
            interior_wall_thickness,
            segment_length,
        ),
        (
            "LowerLeftWall",
            -interior_offset,
            segment_center,
            interior_wall_thickness,
            segment_length,
        ),
        (
            "LowerRightWall",
            -segment_center,
            -interior_offset,
            segment_length,
            interior_wall_thickness,
        ),
    )


def maze_corner_anchors(
    *,
    arena_size: float,
    wall_thickness: float,
    corridor_width: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the centers of the four safe corner corridor ends."""

    center = arena_size / 2.0 - wall_thickness - corridor_width / 2.0
    if center <= 0.0:
        raise ValueError("corner anchors do not fit inside the arena")
    return torch.tensor(
        (
            (-center, center),   # upper-left
            (center, center),    # upper-right
            (-center, -center),  # lower-left
            (center, -center),   # lower-right
        ),
        dtype=torch.float32,
        device=device,
    )


def maze_gate_centers(
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    corridor_width: float,
    wall_tip_offset: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the four interior wall tips whose openings must stay traversable."""

    inner_edge = arena_size / 2.0 - wall_thickness
    offset = inner_edge - corridor_width - interior_wall_thickness / 2.0
    segment_length = arena_size / 4.0 + wall_tip_offset
    tip = -inner_edge + segment_length
    return torch.tensor(
        (
            (tip, offset),
            (offset, -tip),
            (-offset, tip),
            (-tip, -offset),
        ),
        dtype=torch.float32,
        device=device,
    )


def sample_maze_spawn_pairs(
    count: int,
    *,
    arena_size: float,
    wall_thickness: float,
    corridor_width: float,
    spawn_jitter: float,
    min_goal_distance: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Emma and goal in different safe corner corridor ends.

    Returns ``(emma_xy, emma_yaw, goal_xy, emma_corner, goal_corner)``.
    Emma yaw remains fully random so the policy cannot associate a corner with
    one preferred initial heading.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if spawn_jitter < 0.0:
        raise ValueError("spawn_jitter must be non-negative")
    if min_goal_distance <= 0.0:
        raise ValueError("min_goal_distance must be positive")

    anchors = maze_corner_anchors(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        corridor_width=corridor_width,
        device=device,
    )
    if count == 0:
        empty_xy = anchors.new_empty((0, 2))
        empty_scalar = anchors.new_empty((0,))
        empty_index = torch.empty((0,), dtype=torch.long, device=device)
        return empty_xy, empty_scalar, empty_xy.clone(), empty_index, empty_index.clone()

    # 先确认阈值在几何上可达。抖动不是扩大任务距离的必要条件，因此这里
    # 使用无抖动锚点的最大间距作为稳定、保守的可行性上界。
    anchor_distances = torch.cdist(anchors, anchors)
    max_anchor_distance = float(anchor_distances.max().item())
    if min_goal_distance > max_anchor_distance + 1.0e-6:
        raise ValueError(
            "min_goal_distance exceeds the maximum corner-anchor distance "
            f"({min_goal_distance:.3f}m > {max_anchor_distance:.3f}m)"
        )

    emma_corner = torch.randint(
        0, 4, (count,), device=device, generator=generator
    )
    emma_jitter = (
        torch.rand((count, 2), device=device, generator=generator) * 2.0 - 1.0
    ) * spawn_jitter
    emma_xy = anchors[emma_corner] + emma_jitter

    goal_corner = torch.empty(
        (count,), dtype=torch.long, device=device
    )
    goal_xy = anchors.new_empty((count, 2))
    unresolved = torch.ones((count,), dtype=torch.bool, device=device)

    # 不满足最短任务距离时只重采样对应环境，不能让单个随机样本终止整批
    # 1024 个环境。角落编号仍从另外三个角落中均匀抽取。
    for _ in range(32):
        unresolved_ids = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        if unresolved_ids.numel() == 0:
            break
        candidate_corner = (
            emma_corner[unresolved_ids]
            + torch.randint(
                1,
                4,
                (unresolved_ids.numel(),),
                device=device,
                generator=generator,
            )
        ) % 4
        candidate_jitter = (
            torch.rand(
                (unresolved_ids.numel(), 2),
                device=device,
                generator=generator,
            )
            * 2.0
            - 1.0
        ) * spawn_jitter
        candidate_xy = anchors[candidate_corner] + candidate_jitter
        accepted = (
            torch.linalg.norm(candidate_xy - emma_xy[unresolved_ids], dim=-1)
            >= min_goal_distance
        )
        accepted_ids = unresolved_ids[accepted]
        goal_corner[accepted_ids] = candidate_corner[accepted]
        goal_xy[accepted_ids] = candidate_xy[accepted]
        unresolved[accepted_ids] = False

    if unresolved.any():
        # 极端随机序列下使用对角角落的中心作为确定性回退。它仍然与出生
        # 角落不同，并保留最长的局部导航任务。
        unresolved_ids = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        opposite_corner = 3 - emma_corner[unresolved_ids]
        opposite_xy = anchors[opposite_corner]
        fallback_distance = torch.linalg.norm(
            opposite_xy - emma_xy[unresolved_ids], dim=-1
        )
        accepted = fallback_distance >= min_goal_distance
        accepted_ids = unresolved_ids[accepted]
        goal_corner[accepted_ids] = opposite_corner[accepted]
        goal_xy[accepted_ids] = opposite_xy[accepted]
        unresolved[accepted_ids] = False

    if unresolved.any():
        unresolved_ids = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        best_available = torch.linalg.norm(
            anchors[3 - emma_corner[unresolved_ids]]
            - emma_xy[unresolved_ids],
            dim=-1,
        ).max()
        raise ValueError(
            "min_goal_distance is incompatible with the sampled Emma jitter; "
            f"requested={min_goal_distance:.3f}m, "
            f"best_available={float(best_available.item()):.3f}m"
        )

    yaw = torch.rand(count, device=device, generator=generator) * (2.0 * math.pi)
    return emma_xy, yaw, goal_xy, emma_corner, goal_corner


def _point_to_wall_distance(
    points: torch.Tensor,
    wall_specs: tuple[WallSpec, ...],
) -> torch.Tensor:
    """Return point-to-axis-aligned-rectangle distances with shape ``(N, W)``."""

    wall_values = points.new_tensor(
        [(cx, cy, sx, sy) for _, cx, cy, sx, sy in wall_specs]
    )
    delta = (
        (points.unsqueeze(1) - wall_values[None, :, :2]).abs()
        - wall_values[None, :, 2:] / 2.0
    )
    return torch.linalg.norm(torch.relu(delta), dim=-1)


def _corridor_passage_is_valid(
    candidates: torch.Tensor,
    bound_radius: torch.Tensor,
    *,
    inner_edge: float,
    corridor_width: float,
    min_passage_width: float,
) -> torch.Tensor:
    """保证走廊障碍至少有一侧留下满足配置的净通道。"""

    x = candidates[:, 0]
    y = candidates[:, 1]
    radius = bound_radius
    band_edge = inner_edge - corridor_width
    safe = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)

    top = y >= band_edge
    top_gaps = torch.stack(
        (inner_edge - y - radius, y - radius - band_edge), dim=-1
    )
    safe &= (~top) | (top_gaps.amax(dim=-1) >= min_passage_width)

    bottom = y <= -band_edge
    bottom_gaps = torch.stack(
        (y + inner_edge - radius, -band_edge - y - radius), dim=-1
    )
    safe &= (~bottom) | (bottom_gaps.amax(dim=-1) >= min_passage_width)

    right = x >= band_edge
    right_gaps = torch.stack(
        (inner_edge - x - radius, x - radius - band_edge), dim=-1
    )
    safe &= (~right) | (right_gaps.amax(dim=-1) >= min_passage_width)

    left = x <= -band_edge
    left_gaps = torch.stack(
        (x + inner_edge - radius, -band_edge - x - radius), dim=-1
    )
    safe &= (~left) | (left_gaps.amax(dim=-1) >= min_passage_width)
    return safe


def sample_maze_obstacle_positions(
    num_envs: int,
    bound_radii: torch.Tensor,
    *,
    arena_size: float,
    wall_thickness: float,
    interior_wall_thickness: float,
    corridor_width: float,
    wall_tip_offset: float,
    obstacle_wall_clearance: float,
    min_passage_width: float,
    spawn_exclusion_radius: float,
    gate_exclusion_radius: float,
    min_edge_spacing: float,
    candidate_pool_size: int = 24,
    max_attempts: int = 512,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample mixed obstacles without blocking the maze's required passages.

    Bounding circles make the checks conservative for arbitrarily rotated
    cubes. Corner spawn pockets and wall-tip gates remain obstacle-free.
    """

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    if bound_radii.ndim != 1 or (bound_radii <= 0.0).any():
        raise ValueError("bound_radii must be a positive one-dimensional tensor")
    for name, value in (
        ("obstacle_wall_clearance", obstacle_wall_clearance),
        ("min_passage_width", min_passage_width),
        ("spawn_exclusion_radius", spawn_exclusion_radius),
        ("gate_exclusion_radius", gate_exclusion_radius),
        ("min_edge_spacing", min_edge_spacing),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be positive")

    wall_specs = maze_wall_specs(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        interior_wall_thickness=interior_wall_thickness,
        corridor_width=corridor_width,
        wall_tip_offset=wall_tip_offset,
    )
    anchors = maze_corner_anchors(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        corridor_width=corridor_width,
        device=bound_radii.device,
    )
    gates = maze_gate_centers(
        arena_size=arena_size,
        wall_thickness=wall_thickness,
        interior_wall_thickness=interior_wall_thickness,
        corridor_width=corridor_width,
        wall_tip_offset=wall_tip_offset,
        device=bound_radii.device,
    )
    inner_edge = arena_size / 2.0 - wall_thickness
    positions = bound_radii.new_zeros((num_envs, bound_radii.numel(), 2))
    if num_envs == 0 or bound_radii.numel() == 0:
        return positions

    for obstacle_index, radius in enumerate(bound_radii):
        valid = torch.zeros(num_envs, dtype=torch.bool, device=bound_radii.device)
        sample_limit = inner_edge - float(radius.item()) - obstacle_wall_clearance
        if sample_limit <= 0.0:
            raise ValueError("an obstacle does not fit inside the maze")

        for _ in range(max_attempts):
            pending = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if pending.numel() == 0:
                break
            candidate = (
                torch.rand(
                    (pending.numel(), candidate_pool_size, 2),
                    device=bound_radii.device,
                    generator=generator,
                )
                * 2.0
                - 1.0
            ) * sample_limit
            flat_candidate = candidate.reshape(-1, 2)

            required_wall = radius + obstacle_wall_clearance
            wall_safe = (
                _point_to_wall_distance(flat_candidate, wall_specs) >= required_wall
            ).all(dim=-1).reshape(pending.numel(), candidate_pool_size)
            spawn_safe = (
                torch.linalg.norm(
                    candidate.unsqueeze(2) - anchors.view(1, 1, 4, 2), dim=-1
                )
                >= spawn_exclusion_radius + radius
            ).all(dim=-1)
            gate_safe = (
                torch.linalg.norm(
                    candidate.unsqueeze(2) - gates.view(1, 1, 4, 2), dim=-1
                )
                >= gate_exclusion_radius + radius
            ).all(dim=-1)
            passage_safe = _corridor_passage_is_valid(
                flat_candidate,
                radius.expand(flat_candidate.shape[0]),
                inner_edge=inner_edge,
                corridor_width=corridor_width,
                min_passage_width=min_passage_width,
            ).reshape(pending.numel(), candidate_pool_size)
            safe = wall_safe & spawn_safe & gate_safe & passage_safe

            if obstacle_index:
                previous = positions[pending, :obstacle_index]
                separation = torch.linalg.norm(
                    candidate.unsqueeze(2) - previous.unsqueeze(1), dim=-1
                )
                required = (
                    radius + bound_radii[:obstacle_index] + min_edge_spacing
                )
                safe &= (separation >= required.unsqueeze(0)).all(dim=-1)
                # 在一批合法候选中选择离已有障碍最远的位置，形成近似蓝噪声
                # 分布，减少普通均匀随机采样产生的局部扎堆。
                score = (
                    separation - radius - bound_radii[:obstacle_index]
                ).amin(dim=-1)
            else:
                score = torch.rand(
                    (pending.numel(), candidate_pool_size),
                    device=bound_radii.device,
                    generator=generator,
                )

            has_safe_candidate = safe.any(dim=-1)
            best_index = score.masked_fill(~safe, -torch.inf).argmax(dim=-1)
            accepted_rows = has_safe_candidate.nonzero(
                as_tuple=False
            ).squeeze(-1)
            accepted = pending[accepted_rows]
            positions[accepted, obstacle_index] = candidate[
                accepted_rows, best_index[accepted_rows]
            ]
            valid[accepted] = True

        if not valid.all():
            failed = int((~valid).sum().item())
            raise RuntimeError(
                "could not sample a traversable maze layout for "
                f"{failed}/{num_envs} environments at obstacle {obstacle_index}; "
                "reduce obstacle count/spacing or exclusion radii"
            )

    return positions
