"""10 m 随机圆柱/方柱课程场景的纯 Torch 几何采样。"""

from __future__ import annotations

import math

import torch


def alternating_obstacle_types(num_obstacles: int) -> list[str]:
    """返回数量尽量均衡、顺序交替的圆柱和方柱模板。"""

    if num_obstacles < 0:
        raise ValueError("num_obstacles must be non-negative")
    return ["cylinder" if index % 2 == 0 else "cube" for index in range(num_obstacles)]


def outer_wall_specs(
    arena_size: float,
    wall_thickness: float,
) -> list[tuple[str, float, float, float, float]]:
    """返回方形场景的四面薄外墙 ``(name, cx, cy, sx, sy)``。"""

    if arena_size <= 0.0 or wall_thickness <= 0.0:
        raise ValueError("arena_size and wall_thickness must be positive")
    if 2.0 * wall_thickness >= arena_size:
        raise ValueError("outer walls leave no usable arena interior")
    half = arena_size / 2.0
    center = half - wall_thickness / 2.0
    return [
        ("TopWall", 0.0, center, arena_size, wall_thickness),
        ("BottomWall", 0.0, -center, arena_size, wall_thickness),
        ("LeftWall", -center, 0.0, wall_thickness, arena_size),
        ("RightWall", center, 0.0, wall_thickness, arena_size),
    ]


def sample_random_obstacle_positions(
    num_envs: int,
    bound_radii: torch.Tensor,
    *,
    arena_size: float,
    wall_thickness: float,
    min_edge_spacing: float,
    obstacle_wall_clearance: float,
    candidate_pool_size: int = 32,
    max_attempts: int = 512,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """采样近似蓝噪声障碍布局，保证几何边缘间距和外墙净空。

    方柱使用任意朝向下的平面外接圆，因此旋转后仍满足约束。每轮从一批
    合法候选中选择离既有障碍最远的位置，减少均匀随机采样产生的扎堆。
    """

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    if bound_radii.ndim != 1 or (bound_radii <= 0.0).any():
        raise ValueError("bound_radii must be a positive one-dimensional tensor")
    for name, value in (
        ("arena_size", arena_size),
        ("wall_thickness", wall_thickness),
        ("min_edge_spacing", min_edge_spacing),
        ("obstacle_wall_clearance", obstacle_wall_clearance),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if arena_size <= 2.0 * wall_thickness:
        raise ValueError("outer walls leave no usable arena interior")
    if candidate_pool_size < 1 or max_attempts < 1:
        raise ValueError("candidate_pool_size and max_attempts must be positive")

    positions = bound_radii.new_zeros((num_envs, bound_radii.numel(), 2))
    if num_envs == 0 or bound_radii.numel() == 0:
        return positions

    inner_half = arena_size / 2.0 - wall_thickness
    for obstacle_index, radius_tensor in enumerate(bound_radii):
        radius = float(radius_tensor.item())
        sample_limit = inner_half - obstacle_wall_clearance - radius
        if sample_limit <= 0.0:
            raise ValueError("an obstacle does not fit inside the arena")
        valid = torch.zeros(num_envs, dtype=torch.bool, device=bound_radii.device)

        for _ in range(max_attempts):
            pending = (~valid).nonzero(as_tuple=False).squeeze(-1)
            if pending.numel() == 0:
                break
            candidates = (
                torch.rand(
                    (pending.numel(), candidate_pool_size, 2),
                    device=bound_radii.device,
                    generator=generator,
                )
                * 2.0
                - 1.0
            ) * sample_limit
            safe = torch.ones(
                (pending.numel(), candidate_pool_size),
                dtype=torch.bool,
                device=bound_radii.device,
            )

            if obstacle_index:
                previous = positions[pending, :obstacle_index]
                center_distance = torch.linalg.norm(
                    candidates.unsqueeze(2) - previous.unsqueeze(1), dim=-1
                )
                required = (
                    radius + bound_radii[:obstacle_index] + min_edge_spacing
                )
                clearance = center_distance - radius - bound_radii[:obstacle_index]
                safe &= (center_distance >= required.view(1, 1, -1)).all(dim=-1)
                score = clearance.amin(dim=-1)
            else:
                score = torch.rand(
                    (pending.numel(), candidate_pool_size),
                    device=bound_radii.device,
                    generator=generator,
                )

            has_safe = safe.any(dim=-1)
            best = score.masked_fill(~safe, -torch.inf).argmax(dim=-1)
            accepted_rows = has_safe.nonzero(as_tuple=False).squeeze(-1)
            accepted_envs = pending[accepted_rows]
            positions[accepted_envs, obstacle_index] = candidates[
                accepted_rows, best[accepted_rows]
            ]
            valid[accepted_envs] = True

        if not valid.all():
            failed = int((~valid).sum().item())
            raise RuntimeError(
                "unable to sample obstacle layout with configured "
                f"{min_edge_spacing:.3f}m edge spacing for {failed} environments"
            )

    return positions


def _candidate_is_obstacle_safe(
    candidates: torch.Tensor,
    obstacle_positions: torch.Tensor,
    obstacle_radii: torch.Tensor,
    clearance_radius: float,
) -> torch.Tensor:
    """检查候选点与所有活动障碍外接圆的安全距离。"""

    if obstacle_positions.shape[1] == 0:
        return torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    distance = torch.linalg.norm(
        candidates.unsqueeze(2) - obstacle_positions.unsqueeze(1), dim=-1
    )
    required = clearance_radius + obstacle_radii.view(1, 1, -1)
    return (distance >= required).all(dim=-1)


def sample_safe_spawn_pairs(
    obstacle_positions: torch.Tensor,
    obstacle_radii: torch.Tensor,
    *,
    arena_size: float,
    wall_thickness: float,
    emma_clearance_radius: float,
    goal_clearance_radius: float,
    min_goal_distance: float,
    candidate_pool_size: int = 96,
    max_attempts: int = 256,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在当前随机布局中联合采样安全 Emma/Goal，并保证任务长度。

    Emma 使用包含任意出生朝向的保守半径；Goal 使用成功区域和额外余量组成
    的安全半径。两者都不会贴外墙或障碍出生，且欧氏距离严格满足配置下限。
    最后两个返回量是便于调试的象限编号。
    """

    if obstacle_positions.ndim != 3 or obstacle_positions.shape[-1] != 2:
        raise ValueError("obstacle_positions must have shape (num_envs, num_obstacles, 2)")
    if obstacle_radii.ndim != 1 or obstacle_radii.numel() != obstacle_positions.shape[1]:
        raise ValueError("obstacle_radii must match the obstacle dimension")
    for name, value in (
        ("emma_clearance_radius", emma_clearance_radius),
        ("goal_clearance_radius", goal_clearance_radius),
        ("min_goal_distance", min_goal_distance),
    ):
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive")
    if candidate_pool_size < 1 or max_attempts < 1:
        raise ValueError("candidate_pool_size and max_attempts must be positive")

    num_envs = obstacle_positions.shape[0]
    device = obstacle_positions.device
    dtype = obstacle_positions.dtype
    inner_half = arena_size / 2.0 - wall_thickness
    emma_limit = inner_half - emma_clearance_radius
    goal_limit = inner_half - goal_clearance_radius
    if emma_limit <= 0.0 or goal_limit <= 0.0:
        raise ValueError("spawn clearance leaves no usable arena interior")

    emma = torch.zeros(num_envs, 2, device=device, dtype=dtype)
    goal = torch.zeros_like(emma)
    valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    for _ in range(max_attempts):
        pending = (~valid).nonzero(as_tuple=False).squeeze(-1)
        if pending.numel() == 0:
            break
        emma_candidates = (
            torch.rand(
                (pending.numel(), candidate_pool_size, 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 2.0
            - 1.0
        ) * emma_limit
        goal_candidates = (
            torch.rand(
                (pending.numel(), candidate_pool_size, 2),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 2.0
            - 1.0
        ) * goal_limit
        active_positions = obstacle_positions[pending]
        emma_safe = _candidate_is_obstacle_safe(
            emma_candidates,
            active_positions,
            obstacle_radii,
            emma_clearance_radius,
        )
        goal_safe = _candidate_is_obstacle_safe(
            goal_candidates,
            active_positions,
            obstacle_radii,
            goal_clearance_radius,
        )
        pair_distance = torch.linalg.norm(goal_candidates - emma_candidates, dim=-1)
        safe = emma_safe & goal_safe & (pair_distance > min_goal_distance)
        has_safe = safe.any(dim=-1)
        # 在所有合法点对中随机选择，既严格满足 5.2m 下限，也避免总把任务
        # 推到场景对角线附近而导致课程距离单一且偏长。
        selection_score = torch.rand(
            pair_distance.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        best = selection_score.masked_fill(~safe, -torch.inf).argmax(dim=-1)
        accepted_rows = has_safe.nonzero(as_tuple=False).squeeze(-1)
        accepted_envs = pending[accepted_rows]
        emma[accepted_envs] = emma_candidates[accepted_rows, best[accepted_rows]]
        goal[accepted_envs] = goal_candidates[accepted_rows, best[accepted_rows]]
        valid[accepted_envs] = True

    if not valid.all():
        failed = int((~valid).sum().item())
        raise RuntimeError(f"unable to sample safe Emma/Goal pairs for {failed} environments")

    yaw = torch.rand(num_envs, device=device, dtype=dtype, generator=generator) * (
        2.0 * math.pi
    )
    emma_region = (emma[:, 0] >= 0.0).long() + 2 * (emma[:, 1] < 0.0).long()
    goal_region = (goal[:, 0] >= 0.0).long() + 2 * (goal[:, 1] < 0.0).long()
    return emma, yaw, goal, emma_region, goal_region
