"""局部导航奖励及尺度计算所需的纯 Torch 辅助函数。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """将任意弧度角连续映射到 ``[-pi, pi]``。"""

    return torch.atan2(torch.sin(angle), torch.cos(angle))


def vfh_passage_mask(
    sector_ranges: torch.Tensor,
    lookahead_distance: torch.Tensor,
    *,
    min_passage_width: float,
) -> torch.Tensor:
    """根据物理通道宽度生成 VFH+ 可通行方向掩码。

    每个近场 LiDAR 命中点按 ``min_passage_width / 2`` 做角向膨胀。
    某个候选方向落入任意膨胀区域时，意味着沿该方向延伸的直线路径不能
    保证所需车宽与侧向余量，因此从 VFH 候选中剔除。

    ``lookahead_distance`` 可随环境取 ``min(固定前视距离, Goal距离)``，
    避免 Goal 后方的墙影响到达判定。假定输入是均匀的 360° 扇区。
    """

    if sector_ranges.ndim != 2 or sector_ranges.shape[1] < 4:
        raise ValueError(
            "sector_ranges must have shape (num_envs, num_sectors>=4)"
        )
    if lookahead_distance.shape != sector_ranges.shape[:1]:
        raise ValueError("lookahead_distance must match the environment dimension")
    if min_passage_width <= 0.0 or not math.isfinite(min_passage_width):
        raise ValueError("min_passage_width must be finite and positive")

    num_sectors = sector_ranges.shape[-1]
    sector_width = 2.0 * math.pi / num_sectors
    safe_ranges = sector_ranges.clamp_min(1.0e-4)
    obstacle_source = (
        torch.isfinite(sector_ranges)
        & (sector_ranges >= 0.0)
        & (sector_ranges < lookahead_distance.clamp_min(0.0).unsqueeze(-1))
    )

    half_width = 0.5 * min_passage_width
    half_angle = torch.asin((half_width / safe_ranges).clamp(0.0, 1.0))
    inflation_bins = torch.ceil(half_angle / sector_width).to(torch.long)
    inflation_bins = inflation_bins.clamp(max=num_sectors // 4)

    # 用三圈差分数组一次标记所有循环区间，避免按膨胀半径逐 bin 启动 GPU
    # kernel。取中间一圈时，左右两圈自然提供跨越 ±pi 边界的区间。
    env_count = sector_ranges.shape[0]
    indices = torch.arange(
        num_sectors, device=sector_ranges.device, dtype=torch.long
    ).unsqueeze(0).expand(env_count, -1)
    updates = obstacle_source.to(dtype=sector_ranges.dtype)
    difference = torch.zeros(
        (env_count, 3 * num_sectors + 1),
        device=sector_ranges.device,
        dtype=sector_ranges.dtype,
    )
    for circle_offset in (0, num_sectors, 2 * num_sectors):
        centers = indices + circle_offset
        starts = (centers - inflation_bins).clamp_min(0)
        ends = (centers + inflation_bins + 1).clamp_max(3 * num_sectors)
        difference.scatter_add_(1, starts, updates)
        difference.scatter_add_(1, ends, -updates)
    blocked_triple = difference[:, :-1].cumsum(dim=-1) > 0.0
    blocked = blocked_triple[:, num_sectors : 2 * num_sectors]
    return ~blocked


def bounded_smoothstep_blockage(
    clearance: torch.Tensor,
    *,
    blocked_distance: float,
    free_distance: float,
) -> torch.Tensor:
    """将净空连续映射为有限支撑的远场阻塞程度。

    ``clearance <= blocked_distance`` 时严格为 1，
    ``clearance >= free_distance`` 时严格为 0，中间使用 smoothstep。
    相比 sigmoid，它不会在安全远场留下长尾，同时保留阈值附近的连续变化。
    """

    if blocked_distance < 0.0 or not math.isfinite(blocked_distance):
        raise ValueError("blocked_distance must be finite and non-negative")
    if free_distance <= blocked_distance or not math.isfinite(free_distance):
        raise ValueError("free_distance must be finite and greater than blocked_distance")

    finite_clearance = torch.nan_to_num(
        clearance,
        nan=0.0,
        posinf=free_distance,
        neginf=0.0,
    )
    x = ((free_distance - finite_clearance) / (free_distance - blocked_distance)).clamp(
        0.0, 1.0
    )
    return x.square() * (3.0 - 2.0 * x)


def _circular_window_mean(values: torch.Tensor, radius_bins: int) -> torch.Tensor:
    """对均匀 360° 扇区执行循环滑窗平均。"""

    if radius_bins <= 0:
        return values
    extended = torch.cat(
        (values[:, -radius_bins:], values, values[:, :radius_bins]), dim=-1
    )
    return F.avg_pool1d(
        extended.unsqueeze(1), kernel_size=2 * radius_bins + 1, stride=1
    ).squeeze(1)


def _circular_window_min(values: torch.Tensor, radius_bins: int) -> torch.Tensor:
    """对均匀 360° 扇区执行循环滑窗最小值。"""

    if radius_bins <= 0:
        return values
    extended = torch.cat(
        (values[:, -radius_bins:], values, values[:, :radius_bins]), dim=-1
    )
    return -F.max_pool1d(
        -extended.unsqueeze(1), kernel_size=2 * radius_bins + 1, stride=1
    ).squeeze(1)


def vfh_opening_quality(
    clearance: torch.Tensor,
    passage_mask: torch.Tensor,
    *,
    route_lookahead: float,
    valley_radius_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算每个候选方向所在开阔通道的宽度与可见深度质量。

    物理车宽仍由 ``passage_mask`` 硬约束。这里在候选方向左右的固定角窗内
    聚合可通行比例，使宽通道中心得分高、窄缝和通道边缘得分低。深度质量
    同时考虑窗口最小净空和平均净空，避免单根穿过缝隙的远射线把死缝误判
    为深走廊。两项输出均位于 ``[0, 1]``。
    """

    if clearance.ndim != 2:
        raise ValueError("clearance must have shape (num_envs, num_sectors)")
    if passage_mask.shape != clearance.shape or passage_mask.dtype != torch.bool:
        raise ValueError("passage_mask must be boolean and match clearance")
    if route_lookahead <= 0.0 or not math.isfinite(route_lookahead):
        raise ValueError("route_lookahead must be finite and positive")
    if valley_radius_bins < 0 or valley_radius_bins >= clearance.shape[-1] // 2:
        raise ValueError("valley_radius_bins is outside the supported circular window")

    width_quality = _circular_window_mean(
        passage_mask.to(dtype=clearance.dtype), valley_radius_bins
    )
    finite_clearance = torch.nan_to_num(
        clearance,
        nan=0.0,
        posinf=route_lookahead,
        neginf=0.0,
    )
    normalized_depth = (finite_clearance / route_lookahead).clamp(0.0, 1.0)
    depth_floor = _circular_window_min(normalized_depth, valley_radius_bins)
    depth_mean = _circular_window_mean(normalized_depth, valley_radius_bins)
    depth_quality = (0.6 * depth_floor + 0.4 * depth_mean) * width_quality
    return width_quality, depth_quality


def vfh_reference_angle(
    obstacle_density: torch.Tensor,
    sector_angles: torch.Tensor,
    goal_angle: torch.Tensor,
    previous_reference: torch.Tensor,
    previous_valid: torch.Tensor,
    *,
    smoothing_radius_bins: int,
    blocked_threshold: float,
    goal_cost: float,
    obstacle_cost: float,
    blocked_goal_cost: float,
    blocked_obstacle_cost: float,
    continuity_cost: float,
    switch_margin: float,
    max_reference_delta_rad: float,
    passage_mask: torch.Tensor | None = None,
    opening_width_quality: torch.Tensor | None = None,
    opening_depth_quality: torch.Tensor | None = None,
    opening_width_cost: float = 0.0,
    opening_depth_cost: float = 0.0,
    previous_blockage: torch.Tensor | None = None,
    blockage_smoothing_alpha: float = 1.0,
    valley_keep_half_angle_rad: float = math.pi,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """用平滑阻塞权重和通道质量生成稳定 VFH 参考航向。

    Goal 阻塞程度连续插值 Goal/障碍权重，避免阈值处硬切换。物理不可通行
    方向始终由布尔掩码剔除；宽度和深度只在可行方向之间提供连续偏好。
    迟滞跟踪的是上一航向附近的通道，并在通道内部重新寻找安全中心，避免
    长时间冻结一个已经偏离入口中心的旧角度。
    """

    if obstacle_density.ndim != 2:
        raise ValueError("obstacle_density must have shape (num_envs, num_sectors)")
    if sector_angles.shape != obstacle_density.shape:
        if sector_angles.ndim == 1 and sector_angles.shape[0] == obstacle_density.shape[1]:
            sector_angles = sector_angles.unsqueeze(0).expand_as(obstacle_density)
        else:
            raise ValueError("sector_angles must match obstacle_density")
    if goal_angle.shape != obstacle_density.shape[:1]:
        raise ValueError("goal_angle must match the environment dimension")
    if previous_reference.shape != goal_angle.shape:
        raise ValueError("previous_reference must match goal_angle")
    if previous_valid.shape != goal_angle.shape or previous_valid.dtype != torch.bool:
        raise ValueError("previous_valid must be a boolean tensor matching goal_angle")
    if passage_mask is not None:
        if passage_mask.shape != obstacle_density.shape:
            raise ValueError("passage_mask must match obstacle_density")
        if passage_mask.dtype != torch.bool:
            raise ValueError("passage_mask must be a boolean tensor")
    for name, quality in (
        ("opening_width_quality", opening_width_quality),
        ("opening_depth_quality", opening_depth_quality),
    ):
        if quality is not None and quality.shape != obstacle_density.shape:
            raise ValueError(f"{name} must match obstacle_density")
    if previous_blockage is not None and previous_blockage.shape != goal_angle.shape:
        raise ValueError("previous_blockage must match goal_angle")
    if smoothing_radius_bins < 0:
        raise ValueError("smoothing_radius_bins must be non-negative")
    if not 0.0 < blocked_threshold < 1.0:
        raise ValueError("blocked_threshold must be in (0, 1)")
    for name, value in (
        ("goal_cost", goal_cost),
        ("obstacle_cost", obstacle_cost),
        ("blocked_goal_cost", blocked_goal_cost),
        ("blocked_obstacle_cost", blocked_obstacle_cost),
        ("continuity_cost", continuity_cost),
        ("switch_margin", switch_margin),
        ("max_reference_delta_rad", max_reference_delta_rad),
        ("opening_width_cost", opening_width_cost),
        ("opening_depth_cost", opening_depth_cost),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 < blockage_smoothing_alpha <= 1.0:
        raise ValueError("blockage_smoothing_alpha must be in (0, 1]")
    if not 0.0 < valley_keep_half_angle_rad <= math.pi:
        raise ValueError("valley_keep_half_angle_rad must be in (0, pi]")

    density = obstacle_density.clamp(0.0, 1.0)
    radius = min(smoothing_radius_bins, max(0, density.shape[-1] // 2 - 1))
    if radius > 0:
        extended = torch.cat((density[:, -radius:], density, density[:, :radius]), dim=-1)
        inflated = F.max_pool1d(
            extended.unsqueeze(1), kernel_size=2 * radius + 1, stride=1
        ).squeeze(1)
        offsets = torch.arange(
            -radius, radius + 1, device=density.device, dtype=density.dtype
        )
        kernel = (radius + 1 - offsets.abs()).clamp_min(0.0)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        smoothed = F.conv1d(extended.unsqueeze(1), kernel).squeeze(1)
        histogram = 0.7 * inflated + 0.3 * smoothed
    else:
        histogram = density

    active_previous = torch.where(previous_valid, previous_reference, goal_angle)
    goal_error = wrap_to_pi(sector_angles - goal_angle.unsqueeze(-1)).abs() / math.pi
    continuity_error = (
        wrap_to_pi(sector_angles - active_previous.unsqueeze(-1)).abs() / math.pi
    )

    # Goal 阻塞程度由远场直方图和物理通道共同决定。物理不可通行严格给 1，
    # 其余情况保留连续值，再按控制周期做一阶平滑，避免阈值附近反复切模式。
    goal_index = goal_error.argmin(dim=-1)
    goal_histogram = histogram.gather(1, goal_index.unsqueeze(-1)).squeeze(-1)
    raw_free_mask = histogram < blocked_threshold
    if passage_mask is None:
        passage_mask = torch.ones_like(raw_free_mask)
    # 远场直方图只连续改变代价，不再用阈值硬删除候选；真正的硬约束仅来自
    # 近场物理通道掩码。这样墙刚进入远场时参考航向可以逐步偏向入口，等墙
    # 进入车体通行前视范围后才严格禁止对应方向。
    passable_mask = passage_mask
    has_passable = passable_mask.any(dim=-1)
    # 极端包围状态下退回原始自由扇区，仍允许策略原地转向脱困；正常情况下
    # 只在满足物理宽度的方向中选取候选。
    candidate_mask = torch.where(
        has_passable.unsqueeze(-1), passable_mask, raw_free_mask
    )
    goal_passable = passage_mask.gather(
        1, goal_index.unsqueeze(-1)
    ).squeeze(-1)
    goal_blockage_raw = torch.maximum(
        goal_histogram, (~goal_passable).to(dtype=goal_histogram.dtype)
    )
    if previous_blockage is None:
        goal_blockage = goal_blockage_raw
    else:
        smoothed_blockage = previous_blockage + blockage_smoothing_alpha * (
            goal_blockage_raw - previous_blockage
        )
        goal_blockage = torch.where(
            previous_valid, smoothed_blockage, goal_blockage_raw
        )
    goal_blockage = goal_blockage.clamp(0.0, 1.0)
    goal_blocked = goal_blockage >= blocked_threshold
    effective_goal_cost = goal_cost + goal_blockage * (
        blocked_goal_cost - goal_cost
    )
    effective_obstacle_cost = obstacle_cost + goal_blockage * (
        blocked_obstacle_cost - obstacle_cost
    )

    if opening_width_quality is None:
        opening_width_quality = torch.zeros_like(histogram)
    if opening_depth_quality is None:
        opening_depth_quality = torch.zeros_like(histogram)
    opening_bonus = goal_blockage.unsqueeze(-1) * (
        opening_width_cost * opening_width_quality.clamp(0.0, 1.0)
        + opening_depth_cost * opening_depth_quality.clamp(0.0, 1.0)
    )
    sector_cost = (
        effective_goal_cost.unsqueeze(-1) * goal_error
        + effective_obstacle_cost.unsqueeze(-1) * histogram
        + continuity_cost * continuity_error
        - opening_bonus
    )

    has_candidate = candidate_mask.any(dim=-1)
    masked_cost = torch.where(
        candidate_mask, sector_cost, torch.full_like(sector_cost, float("inf"))
    )
    candidate_index = torch.where(
        has_candidate, masked_cost.argmin(dim=-1), sector_cost.argmin(dim=-1)
    )
    candidate_angle = sector_angles.gather(1, candidate_index.unsqueeze(-1)).squeeze(-1)
    candidate_score = sector_cost.gather(
        1, candidate_index.unsqueeze(-1)
    ).squeeze(-1)

    # 在上一参考角附近重新寻找当前通道中心，而不是原样保持旧角度。只有全局
    # 最优通道的代价至少改善 switch_margin 时才跨通道切换。
    same_valley_mask = candidate_mask & (
        wrap_to_pi(sector_angles - active_previous.unsqueeze(-1)).abs()
        <= valley_keep_half_angle_rad
    )
    has_same_valley = same_valley_mask.any(dim=-1)
    same_valley_cost = torch.where(
        same_valley_mask, sector_cost, torch.full_like(sector_cost, float("inf"))
    )
    same_valley_index = same_valley_cost.argmin(dim=-1)
    same_valley_score = sector_cost.gather(
        1, same_valley_index.unsqueeze(-1)
    ).squeeze(-1)
    same_valley_angle = sector_angles.gather(
        1, same_valley_index.unsqueeze(-1)
    ).squeeze(-1)
    keep_previous = (
        previous_valid
        & has_same_valley
        & (candidate_score + switch_margin >= same_valley_score)
    )
    selected_index = torch.where(
        keep_previous, same_valley_index, candidate_index
    )
    selected_angle = torch.where(keep_previous, same_valley_angle, candidate_angle)
    selected_width_quality = opening_width_quality.gather(
        1, selected_index.unsqueeze(-1)
    ).squeeze(-1)
    selected_depth_quality = opening_depth_quality.gather(
        1, selected_index.unsqueeze(-1)
    ).squeeze(-1)

    delta = wrap_to_pi(selected_angle - active_previous)
    if max_reference_delta_rad > 0.0:
        delta = delta.clamp(
            min=-max_reference_delta_rad, max=max_reference_delta_rad
        )
    reference = torch.where(
        previous_valid,
        wrap_to_pi(active_previous + delta),
        candidate_angle,
    )
    diagnostics = {
        "histogram": histogram,
        "free_mask": raw_free_mask,
        "passage_mask": passage_mask,
        "passable_mask": passable_mask,
        "passage_fallback": ~has_passable,
        "goal_histogram": goal_histogram,
        "goal_blocked": goal_blocked,
        "goal_blockage_raw": goal_blockage_raw,
        "goal_blockage": goal_blockage,
        "effective_goal_cost": effective_goal_cost,
        "effective_obstacle_cost": effective_obstacle_cost,
        "opening_width_quality": opening_width_quality,
        "opening_depth_quality": opening_depth_quality,
        "opening_bonus": opening_bonus,
        "selected_width_quality": selected_width_quality,
        "selected_depth_quality": selected_depth_quality,
        "candidate_angle": candidate_angle,
        "kept_previous": keep_previous,
        "reference_delta": wrap_to_pi(reference - active_previous),
    }
    return reference, diagnostics


def step_turn_direction_reward(
    reference_angle: torch.Tensor,
    angular_velocity: torch.Tensor,
    *,
    max_angular_velocity: float,
    scale_per_step: float,
    deadband_rad: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐控制步奖励朝 VFH 参考航向正确旋转。

    奖励直接按实际角速度相对上限归一化，不再乘 ``dt``。参考角在死区内
    时不奖励转向；误差增大到 90° 后使用完整权重。朝正确方向旋转为正，
    朝反方向旋转为负，原地不转严格为零。

    返回 ``(reward, directional_score)``，后者位于 ``[-1, 1]``。
    """

    if max_angular_velocity <= 0 or not math.isfinite(max_angular_velocity):
        raise ValueError("max_angular_velocity must be finite and positive")
    if scale_per_step < 0 or not math.isfinite(scale_per_step):
        raise ValueError("scale_per_step must be finite and non-negative")
    if (
        deadband_rad < 0
        or deadband_rad >= math.pi / 2.0
        or not math.isfinite(deadband_rad)
    ):
        raise ValueError("deadband_rad must be finite and in [0, pi / 2)")
    if reference_angle.shape != angular_velocity.shape:
        raise ValueError("reference_angle and angular_velocity must have matching shapes")

    error = wrap_to_pi(reference_angle)
    turn_need = (
        (error.abs() - deadband_rad) / (math.pi / 2.0 - deadband_rad)
    ).clamp(0.0, 1.0)
    correct_turn_rate = (
        torch.sign(error) * angular_velocity / max_angular_velocity
    ).clamp(-1.0, 1.0)
    directional_score = turn_need * correct_turn_rate
    return scale_per_step * directional_score, directional_score


def step_safe_forward_reward(
    forward_speed: torch.Tensor,
    reference_angle: torch.Tensor,
    *,
    max_forward_speed: float,
    scale_per_step: float,
    full_reward_angle_rad: float,
    cutoff_angle_rad: float,
) -> torch.Tensor:
    """逐控制步奖励沿参考安全航向前进。

    ``full_reward_angle_rad`` 内给予完整的速度比例奖励；随后使用有限支撑
    smoothstep，在 ``cutoff_angle_rad`` 处平滑降为严格的零。这里不使用
    余弦或 sigmoid，避免较大航向误差仍保留过多奖励或在截止角外留下长尾。
    静止与倒车不获得该项，倒车由独立惩罚处理。
    """

    if max_forward_speed <= 0 or not math.isfinite(max_forward_speed):
        raise ValueError("max_forward_speed must be finite and positive")
    if scale_per_step < 0 or not math.isfinite(scale_per_step):
        raise ValueError("scale_per_step must be finite and non-negative")
    if (
        full_reward_angle_rad < 0.0
        or not math.isfinite(full_reward_angle_rad)
    ):
        raise ValueError("full_reward_angle_rad must be finite and non-negative")
    if (
        cutoff_angle_rad <= full_reward_angle_rad
        or cutoff_angle_rad > math.pi
        or not math.isfinite(cutoff_angle_rad)
    ):
        raise ValueError(
            "cutoff_angle_rad must be finite, greater than the full-reward "
            "angle, and no greater than pi"
        )
    if forward_speed.shape != reference_angle.shape:
        raise ValueError("forward_speed and reference_angle must have matching shapes")

    error_abs = wrap_to_pi(reference_angle).abs()
    transition = (
        (cutoff_angle_rad - error_abs)
        / (cutoff_angle_rad - full_reward_angle_rad)
    ).clamp(0.0, 1.0)
    alignment = transition.square() * (3.0 - 2.0 * transition)
    forward_ratio = (forward_speed / max_forward_speed).clamp(0.0, 1.0)
    return scale_per_step * forward_ratio * alignment


def step_idle_penalty(
    forward_speed: torch.Tensor,
    angular_velocity: torch.Tensor,
    reference_angle: torch.Tensor,
    *,
    max_forward_speed: float,
    max_angular_velocity: float,
    scale_per_step: float,
    turn_deadband_rad: float,
    forward_full_reward_angle_rad: float,
    forward_cutoff_angle_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐步惩罚既不正确转向、也不沿参考航向前进的停滞状态。

    当参考角较大时，正确方向的旋转可以消除 idle 惩罚；参考角接近零时，
    前进才是有效活动。错误旋转和背离参考方向的运动不计为有效活动。
    返回 ``(penalty, task_activity)``。
    """

    if max_forward_speed <= 0 or not math.isfinite(max_forward_speed):
        raise ValueError("max_forward_speed must be finite and positive")
    if max_angular_velocity <= 0 or not math.isfinite(max_angular_velocity):
        raise ValueError("max_angular_velocity must be finite and positive")
    if scale_per_step < 0 or not math.isfinite(scale_per_step):
        raise ValueError("scale_per_step must be finite and non-negative")
    if not 0.0 <= turn_deadband_rad < math.pi / 2.0:
        raise ValueError("turn_deadband_rad must be in [0, pi / 2)")
    if (
        forward_full_reward_angle_rad < 0.0
        or not math.isfinite(forward_full_reward_angle_rad)
    ):
        raise ValueError(
            "forward_full_reward_angle_rad must be finite and non-negative"
        )
    if (
        forward_cutoff_angle_rad <= forward_full_reward_angle_rad
        or forward_cutoff_angle_rad > math.pi
        or not math.isfinite(forward_cutoff_angle_rad)
    ):
        raise ValueError(
            "forward_cutoff_angle_rad must be finite, greater than the "
            "full-reward angle, and no greater than pi"
        )
    if (
        forward_speed.shape != angular_velocity.shape
        or forward_speed.shape != reference_angle.shape
    ):
        raise ValueError("speed and reference tensors must have matching shapes")

    error = wrap_to_pi(reference_angle)
    turn_need = (
        (error.abs() - turn_deadband_rad)
        / (math.pi / 2.0 - turn_deadband_rad)
    ).clamp(0.0, 1.0)
    correct_turn_activity = (
        torch.sign(error) * angular_velocity / max_angular_velocity
    ).clamp(0.0, 1.0) * turn_need
    forward_transition = (
        (forward_cutoff_angle_rad - error.abs())
        / (forward_cutoff_angle_rad - forward_full_reward_angle_rad)
    ).clamp(0.0, 1.0)
    forward_alignment = forward_transition.square() * (
        3.0 - 2.0 * forward_transition
    )
    forward_activity = (
        (forward_speed / max_forward_speed).clamp(0.0, 1.0)
        * forward_alignment
    )
    task_activity = torch.maximum(correct_turn_activity, forward_activity)
    return -scale_per_step * (1.0 - task_activity), task_activity


def goal_aware_obstacle_relevance(
    clearance: torch.Tensor,
    sector_angles: torch.Tensor,
    goal_angle: torch.Tensor,
    goal_distance: torch.Tensor,
    *,
    goal_reach_dist: float,
    collision_clearance_threshold: float,
    transition_width: float = 0.1,
) -> torch.Tensor:
    """判断每个扇区中的障碍是否位于到达 Goal 所需路径之前。

    只有射线会穿过 Goal 成功圆时，才比较障碍净空与进入成功圆所需的
    行驶距离。位于成功圆之后的墙或障碍会被平滑降权；其他方向以及
    Goal 之前的障碍保持完整权重。该权重只用于参考航向和斥力奖励，
    不参与碰撞检测，因此不会放宽物理安全边界。
    """

    if clearance.ndim != 2:
        raise ValueError("clearance must have shape (num_envs, num_sectors)")
    if sector_angles.shape != clearance.shape:
        if sector_angles.ndim == 1 and sector_angles.shape[0] == clearance.shape[1]:
            sector_angles = sector_angles.unsqueeze(0).expand_as(clearance)
        else:
            raise ValueError("sector_angles must match clearance")
    if goal_angle.shape != clearance.shape[:1]:
        raise ValueError("goal_angle must match the environment dimension")
    if goal_distance.shape != clearance.shape[:1]:
        raise ValueError("goal_distance must match the environment dimension")
    if goal_reach_dist <= 0.0:
        raise ValueError("goal_reach_dist must be positive")
    if collision_clearance_threshold < 0.0:
        raise ValueError("collision_clearance_threshold must be non-negative")
    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive")

    angle_error = wrap_to_pi(
        sector_angles - goal_angle.unsqueeze(-1)
    )
    distance = goal_distance.unsqueeze(-1)
    forward_projection = distance * torch.cos(angle_error)
    lateral_distance = distance * torch.sin(angle_error).abs()

    # 射线与 Goal 成功圆的近端交点，即沿该方向最少需要行驶的距离。
    intersects_goal = (
        (goal_distance > goal_reach_dist).unsqueeze(-1)
        & (forward_projection > 0.0)
        & (lateral_distance < goal_reach_dist)
    )
    chord = torch.sqrt(
        (goal_reach_dist**2 - lateral_distance.square()).clamp_min(0.0)
    )
    travel_to_goal = (forward_projection - chord).clamp_min(0.0)

    # clearance 已经扣除了车体外形投影，因此只需额外保留碰撞阈值。
    blocking_boundary = travel_to_goal + collision_clearance_threshold
    before_goal_weight = torch.sigmoid(
        (blocking_boundary - clearance) / transition_width
    )
    return torch.where(
        intersects_goal,
        before_goal_weight,
        torch.ones_like(clearance),
    )


def closing_repulse_penalty(
    clearance: torch.Tensor,
    sector_angles: torch.Tensor,
    forward_speed: torch.Tensor,
    angular_speed: torch.Tensor,
    *,
    start_clearance: float,
    max_speed: float,
    scale_per_step: float,
    robot_half_length: float,
    robot_half_width: float,
    lateral_weight_floor: float = 0.3,
    rotation_weight: float = 1.0,
    motion_floor: float = 0.0,
    obstacle_relevance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """惩罚近障平移和旋转扫掠，同时降低非行进方向障碍的减速影响。

    ``danger`` 使用带 0.15 m 偏置和上限的反比例函数。相比原来的二次
    形状，它在很小净空处增长更快，同时在 ``start_clearance`` 处连续为零。
    平移风险随实际速度线性增长：当前行进方向上的障碍使用完整权重，侧面
    以及正在远离的方向仍保留 ``lateral_weight_floor``，从而不会重新产生
    贴障零代价。旋转项继续按矩形车体扫掠速度对所有方向生效，用于约束
    车头和车尾扫碰。最终惩罚直接使用逐控制步 scale，不再乘 ``dt``。
    """

    if clearance.ndim != 2:
        raise ValueError("clearance must have shape (num_envs, num_sectors)")
    if sector_angles.shape != clearance.shape:
        if sector_angles.ndim == 1 and sector_angles.shape[0] == clearance.shape[1]:
            sector_angles = sector_angles.unsqueeze(0).expand_as(clearance)
        else:
            raise ValueError("sector_angles must match clearance")
    if forward_speed.shape != clearance.shape[:1]:
        raise ValueError("forward_speed must match the environment dimension")
    if angular_speed.shape != clearance.shape[:1]:
        raise ValueError("angular_speed must match the environment dimension")
    if start_clearance <= 0 or max_speed <= 0:
        raise ValueError("clearance and speed limits must be positive")
    if scale_per_step < 0 or not math.isfinite(scale_per_step):
        raise ValueError("scale_per_step must be finite and non-negative")
    if robot_half_length <= 0 or robot_half_width <= 0:
        raise ValueError("robot half extents must be positive")
    if not 0.0 <= lateral_weight_floor <= 1.0:
        raise ValueError("lateral_weight_floor must be in [0, 1]")
    if rotation_weight < 0 or not math.isfinite(rotation_weight):
        raise ValueError("rotation_weight must be finite and non-negative")
    if not 0.0 <= motion_floor <= 1.0 or not math.isfinite(motion_floor):
        raise ValueError("motion_floor must be finite and in [0, 1]")
    if obstacle_relevance is not None:
        if obstacle_relevance.shape != clearance.shape:
            raise ValueError("obstacle_relevance must match clearance")

    # 重构前使用的有界反比例净空风险。先把穿入车体 footprint 的负净空
    # 截到零，既保留最大危险度，又避免分母接近零或变成负数。
    inverse_offset = 0.15
    danger = (
        1.0 / (clearance.clamp_min(0.0) + inverse_offset)
        - 1.0 / (start_clearance + inverse_offset)
    ).clamp(min=0.0, max=50.0)
    if obstacle_relevance is not None:
        danger = danger * obstacle_relevance

    cos_angles = torch.cos(sector_angles)
    abs_cos = cos_angles.abs()
    abs_sin = torch.sin(sector_angles).abs()

    # 行进方向障碍获得完整平移权重；侧面和正在远离的方向仅保留基础斥力。
    # forward_speed 的符号负责在前进与倒车时交换强风险的前后方向。
    motion_sign = torch.sign(forward_speed).unsqueeze(-1)
    approaching_alignment = torch.relu(motion_sign * cos_angles)
    translation_angle_weight = (
        lateral_weight_floor
        + (1.0 - lateral_weight_floor) * approaching_alignment
    )
    translation_sweep_speed = (
        forward_speed.abs().unsqueeze(-1) * translation_angle_weight
    )

    # 旋转时矩形车身各方向都可能发生车头或车尾扫碰，因此保持全向风险。
    rotation_sweep_speed = angular_speed.abs().unsqueeze(-1) * (
        robot_half_length * abs_sin + robot_half_width * abs_cos
    )
    motion_ratio = (
        (
            translation_sweep_speed
            + rotation_weight * rotation_sweep_speed
        )
        / max_speed
    ).clamp(min=motion_floor, max=1.0)
    risk = (danger * motion_ratio).amax(dim=-1)
    penalty = -scale_per_step * risk
    return penalty, risk


def action_jitter_penalty(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    linear_scale: float,
    angular_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """惩罚相邻控制步之间的策略原始归一化动作突变。

    该信号在动作延迟、执行器噪声和安全层之前计算，避免策略为无法控制的
    域随机化噪声受罚。线速度变化使用较小权重，允许正常起步和制动；
    角速度变化使用较大权重，重点抑制会传递到轮速的左右抖动。
    返回 ``(penalty, mean_squared_delta)``。
    """

    if action.shape != previous_action.shape:
        raise ValueError("action tensors must have matching shapes")
    if action.ndim < 2 or action.shape[-1] != 2:
        raise ValueError("actions must end with two normalized values")
    for name, value in (
        ("linear_scale", linear_scale),
        ("angular_scale", angular_scale),
    ):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")

    delta = action - previous_action
    mean_squared_delta = delta.square().mean(dim=-1)
    penalty = -(
        linear_scale * delta[..., 0].square()
        + angular_scale * delta[..., 1].square()
    )
    return penalty, mean_squared_delta


def step_progress_reward(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    *,
    full_progress_distance: float,
    scale_per_step: float,
) -> torch.Tensor:
    """逐步单边 Goal 进度奖励，输出固定的逐 step 量级。

    每一步比较当前距离与上一步距离，只保留正向接近量。这样绕墙远离 Goal
    时不会被压制。达到 ``full_progress_distance`` 时获得完整逐步 scale，
    更大的单步变化截断为完整奖励。该项只作弱辅助，主要行为反馈来自正确
    转向和沿 VFH 参考方向前进。
    """

    if previous_distance.shape != current_distance.shape:
        raise ValueError("distance tensors must have matching shapes")
    if full_progress_distance <= 0 or not math.isfinite(full_progress_distance):
        raise ValueError("full_progress_distance must be finite and positive")
    if scale_per_step < 0 or not math.isfinite(scale_per_step):
        raise ValueError("scale_per_step must be finite and non-negative")

    improvement = torch.relu(previous_distance - current_distance)
    progress_ratio = (improvement / full_progress_distance).clamp(0.0, 1.0)
    return progress_ratio * scale_per_step
