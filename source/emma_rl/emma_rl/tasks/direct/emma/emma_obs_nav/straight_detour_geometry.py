"""Pure NumPy geometry for dedicated multi-lane straight detour environments."""

from __future__ import annotations

import math

import numpy as np


def _range(name, values, *, integer=False):
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low, high = values
    if integer and (not isinstance(low, int) or not isinstance(high, int)):
        raise ValueError(f"{name} must contain integers")
    if not integer and (not math.isfinite(low) or not math.isfinite(high)):
        raise ValueError(f"{name} must be finite")
    if low > high:
        raise ValueError(f"{name} must be ordered")
    return low, high


def _distance_to_polyline(point, waypoints):
    start, end = waypoints[:-1], waypoints[1:]
    delta = end - start
    length_sq = np.square(delta).sum(axis=-1).clip(1.0e-12)
    t = ((point - start) * delta).sum(axis=-1) / length_sq
    nearest = start + t.clip(0.0, 1.0)[:, None] * delta
    return np.linalg.norm(nearest - point, axis=-1).min()


def _point_clearance(point, positions, radii, active, wall_poses,
                     wall_active, wall_lengths, wall_width, inner_half):
    result = inner_half - np.abs(point).max()
    for center, radius in zip(positions[active], radii[active]):
        result = min(result, np.linalg.norm(point - center) - radius)
    for (x, y, yaw), length in zip(
            wall_poses[wall_active], wall_lengths[wall_active]):
        delta = point - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array((c * delta[0] + s * delta[1],
                          -s * delta[0] + c * delta[1]))
        q = np.abs(local) - (length / 2.0, wall_width / 2.0)
        distance = np.linalg.norm(np.maximum(q, 0.0)) + min(q.max(), 0.0)
        result = min(result, distance)
    return result


def sample_straight_detour_layout(
    obstacle_radii,
    wall_lengths,
    *,
    seed,
    arena_size=10.0,
    outer_wall_thickness=0.15,
    wall_width=0.15,
    base_wall_slot_count=16,
    lane_count=3,
    passage_width_range=(0.60, 1.55),
    curriculum_passage_width_ranges=(
            (2.0, 2.5),
            (1.6, 2.2),
            (1.6, 1.9),
            (1.3, 1.7),
            (1.0, 1.5),
            (0.7, 1.2),
    ),
    passage_width_samples_per_level=3,
    barrier_count_range=(3, 3),
    route_barrier_crossings=(1, 2, 3),
    barrier_y_span=2.70,
    extra_obstacle_count_range=(3, 6),
    obstacle_wall_clearance=0.10,
    robot_half_width=0.275,
    robot_half_length=0.425,
    collision_clearance=0.01,
    min_obstacle_spacing=0.60,
    navigation_clearance=0.52,
    emma_clearance=0.70,
    goal_clearance=0.50,
    min_goal_distance=5.20,
    spawn_bank_size=128,
    spawn_y=4.05,
    spawn_lateral_jitter=0.25,
    curriculum_levels=6,
):
    """Build several sealed, parallel detour lanes in one square arena."""

    radii = np.asarray(obstacle_radii, dtype=np.float64)
    lengths = np.asarray(wall_lengths, dtype=np.float64)
    if radii.ndim != 1 or (radii <= 0.0).any():
        raise ValueError("obstacle_radii must be a positive one-dimensional array")
    if lengths.ndim != 1 or (lengths <= 0.0).any():
        raise ValueError("wall_lengths must be a positive one-dimensional array")
    passage_min, passage_max = _range("passage_width_range", passage_width_range)
    level_ranges = np.asarray(
        curriculum_passage_width_ranges, dtype=np.float64
    )
    barrier_min, barrier_max = _range(
        "barrier_count_range", barrier_count_range, integer=True)
    obstacle_min, obstacle_max = _range(
        "extra_obstacle_count_range", extra_obstacle_count_range, integer=True)
    if lane_count < 1 or obstacle_min < lane_count:
        raise ValueError("detour requires lanes and one obstacle per lane")
    if curriculum_levels != 2 * lane_count:
        raise ValueError("six-level detour requires two curriculum levels per lane")
    crossings = tuple(route_barrier_crossings)
    if len(crossings) != lane_count:
        raise ValueError("one route-crossing count per lane is required")
    if any(not isinstance(value, int) for value in crossings):
        raise ValueError("route barrier crossings must be integers")
    if any(not 1 <= value <= 3 for value in crossings):
        raise ValueError("route barrier crossings must lie between one and three")
    if any(crossings[index] > crossings[index + 1]
           for index in range(lane_count - 1)):
        raise ValueError("route barrier crossings must be non-decreasing across lanes")
    if (not isinstance(passage_width_samples_per_level, int)
            or passage_width_samples_per_level < 1):
        raise ValueError("passage_width_samples_per_level must be positive")
    if level_ranges.shape != (curriculum_levels, 2):
        raise ValueError("one passage-width range is required per curriculum level")
    if (not np.isfinite(level_ranges).all()
            or (level_ranges[:, 0] <= 0.0).any()
            or (level_ranges[:, 0] > level_ranges[:, 1]).any()
            or (level_ranges < passage_min - 1.0e-9).any()
            or (level_ranges > passage_max + 1.0e-9).any()):
        raise ValueError("invalid curriculum passage-width ranges")
    if ((level_ranges[:-1, 0] < level_ranges[1:, 0]).any()
            or (level_ranges[:-1, 1] < level_ranges[1:, 1]).any()):
        raise ValueError("curriculum passage widths must run easiest to hardest")
    # if (barrier_min, barrier_max) != (3, 3):
    #     raise ValueError("six-level detour requires exactly three barriers per lane")
    if min(robot_half_width, robot_half_length, collision_clearance) <= 0.0:
        raise ValueError("robot footprint and collision clearance must be positive")
    minimum_physical_passage = 2.0 * (
        robot_half_width + collision_clearance
    )
    if passage_min < minimum_physical_passage:
        raise ValueError(
            "minimum passage must fit the robot width and collision clearance"
        )
    if min(arena_size, outer_wall_thickness, wall_width, barrier_y_span,
           spawn_y, spawn_bank_size, curriculum_levels) <= 0:
        raise ValueError("detour geometry dimensions and counts must be positive")

    inner_half = arena_size / 2.0 - outer_wall_thickness
    full_segment_length = (arena_size - 2.0 * outer_wall_thickness) / 3.0
    divider_slots = (lane_count - 1) * 3
    slots_per_lane = 2 * passage_width_samples_per_level + 1
    barrier_slots = lane_count * slots_per_lane
    if base_wall_slot_count < divider_slots:
        raise ValueError("base wall slots cannot hold the lane dividers")
    if lengths.size < base_wall_slot_count + barrier_slots:
        raise ValueError("insufficient dedicated barrier wall slots")
    if divider_slots and not np.allclose(
            lengths[:divider_slots], full_segment_length, atol=1.0e-6):
        raise ValueError("lane dividers require full cell wall slots")
    if spawn_y + max(emma_clearance, goal_clearance) > inner_half:
        raise ValueError("spawn_y leaves insufficient clearance from the end walls")
    if obstacle_max > radii.size:
        raise ValueError("extra obstacle count exceeds available obstacle slots")

    gate_guard = max(
        navigation_clearance + wall_width / 2.0 + 0.05,
        robot_half_length + wall_width / 2.0 + collision_clearance + 0.05,
    )
    if barrier_y_span + 0.10 + gate_guard >= spawn_y:
        raise ValueError("barriers leave no complete alignment segment before spawn")
    if barrier_max > 1:
        minimum_barrier_spacing = (
            2.0 * barrier_y_span / (barrier_max - 1) - 0.20
        )
        if minimum_barrier_spacing <= 2.0 * gate_guard:
            raise ValueError("barrier alignment zones may overlap")

    edges = np.linspace(-inner_half, inner_half, lane_count + 1)
    lane_bounds = np.empty((lane_count, 2), dtype=np.float64)
    lane_centers = np.empty(lane_count, dtype=np.float64)
    for lane in range(lane_count):
        left = edges[lane] + (wall_width / 2.0 if lane else 0.0)
        right = edges[lane + 1] - (
            wall_width / 2.0 if lane + 1 < lane_count else 0.0)
        lane_bounds[lane] = (left, right)
        lane_centers[lane] = (left + right) / 2.0

    rng = np.random.default_rng(seed)
    wall_poses = np.zeros((lengths.size, 3), dtype=np.float64)
    wall_active = np.zeros(lengths.size, dtype=bool)
    slot = 0
    for divider_x in edges[1:-1]:
        for segment in range(3):
            wall_poses[slot] = (
                divider_x, (segment - 1) * full_segment_length,
                math.pi / 2.0)
            wall_active[slot] = True
            slot += 1

    barrier_counts = np.empty(lane_count, dtype=np.int64)
    passage_widths = []
    route_waypoints = []
    gate_centers = []
    curriculum_endpoint_ys = []
    lane_route_floors = []
    outer_centers = (-arena_size / 2.0 + outer_wall_thickness / 2.0,
                     arena_size / 2.0 - outer_wall_thickness / 2.0)
    for lane in range(lane_count):
        count = 3
        barrier_counts[lane] = count
        y_positions = np.linspace(-barrier_y_span, barrier_y_span, count)
        y_positions += rng.uniform(-0.05, 0.05, count)
        lane_slot_start = base_wall_slot_count + lane * slots_per_lane
        lower_slot = lane_slot_start + int(rng.integers(
            0, passage_width_samples_per_level
        ))
        middle_slot = lane_slot_start + passage_width_samples_per_level
        upper_slot = middle_slot + 1 + int(rng.integers(
            0, passage_width_samples_per_level
        ))
        selected = np.array((lower_slot, middle_slot, upper_slot))
        from_left = (np.arange(count) + int(rng.integers(0, 2))) % 2 == 0
        gate_x = np.empty(count, dtype=np.float64)
        actual_widths = np.empty(count, dtype=np.float64)
        left_inner, right_inner = lane_bounds[lane]
        left_center = outer_centers[0] if lane == 0 else edges[lane]
        right_center = outer_centers[1] if lane + 1 == lane_count else edges[lane + 1]
        for index, (wall_slot, y, anchor_left) in enumerate(
                zip(selected, y_positions, from_left)):
            length = lengths[wall_slot]
            if anchor_left:
                center_x = left_center + length / 2.0
                tip = left_center + length
                actual_widths[index] = right_inner - tip
                gate_x[index] = (tip + right_inner) / 2.0
            else:
                center_x = right_center - length / 2.0
                tip = right_center - length
                actual_widths[index] = tip - left_inner
                gate_x[index] = (tip + left_inner) / 2.0
            wall_poses[wall_slot] = (center_x, y, 0.0)
            wall_active[wall_slot] = True
        if ((actual_widths < passage_min - 1.0e-6)
                | (actual_widths > passage_max + 1.0e-6)).any():
            raise ValueError("dedicated barrier slot length violates passage range")
        passage_widths.append(actual_widths)

        order = np.argsort(y_positions)
        # Keep the vehicle parallel to the corridor while its complete body is
        # crossing a tight gate. The larger circular navigation clearance is
        # still used below to keep movable clutter away from the whole route.
        route = [[lane_centers[lane], -spawn_y]]
        ordered_gates = []
        for barrier_index in order:
            x, y = gate_x[barrier_index], y_positions[barrier_index]
            ordered_gates.append([x, y])
            route.extend(([x, y - gate_guard], [x, y],
                          [x, y + gate_guard]))
        route.append([lane_centers[lane], spawn_y])
        route_waypoints.append(np.asarray(route, dtype=np.float64))
        ordered_gates = np.asarray(ordered_gates, dtype=np.float64)
        gate_centers.append(ordered_gates)

        endpoint_margin = 0.03
        endpoint_guard = max(
            gate_guard,
            max(emma_clearance, goal_clearance) + wall_width / 2.0,
        )
        lower_inner_y = ordered_gates[0, 1] + endpoint_guard + endpoint_margin
        upper_inner_y = ordered_gates[2, 1] - endpoint_guard - endpoint_margin
        middle_lower_y = ordered_gates[1, 1] - endpoint_guard - endpoint_margin
        middle_upper_y = ordered_gates[1, 1] + endpoint_guard + endpoint_margin
        crossing_count = int(route_barrier_crossings[lane])
        if crossing_count < 3 and (
                middle_lower_y <= ordered_gates[0, 1] + gate_guard + endpoint_margin
                or middle_upper_y >= ordered_gates[2, 1] - gate_guard - endpoint_margin):
            raise ValueError("curriculum routes have no alignment room around middle gate")
        if crossing_count == 1:
            # Easiest lane: each level crosses only its own level-specific gate
            # and stops clear of the shared middle gate's alignment zone.
            endpoints = ((-spawn_y, middle_lower_y), (middle_upper_y, spawn_y))
            route_floor = 2.0 * gate_guard + endpoint_guard
        elif crossing_count == 2:
            endpoints = ((-spawn_y, upper_inner_y), (lower_inner_y, spawn_y))
            route_floor = min_goal_distance
        else:
            endpoints = ((-spawn_y, spawn_y), (-spawn_y, spawn_y))
            route_floor = min_goal_distance
        if any(top - bottom <= route_floor for bottom, top in endpoints):
            raise ValueError("curriculum route violates min_goal_distance")
        curriculum_endpoint_ys.append(endpoints)
        lane_route_floors.append(route_floor)

    positions = np.zeros((radii.size, 2), dtype=np.float64)
    obstacle_active = np.zeros(radii.size, dtype=bool)
    obstacle_yaws = np.zeros(radii.size, dtype=np.float64)
    obstacle_lane_by_slot = np.full(radii.size, -1, dtype=np.int64)
    requested_obstacles = int(rng.integers(obstacle_min, obstacle_max + 1))
    first_lanes = rng.permutation(lane_count)
    obstacle_lanes = [int(first_lanes[i]) if i < lane_count
                      else int(rng.integers(0, lane_count))
                      for i in range(requested_obstacles)]
    # Prefer the compact obstacle assets here.  A 0.40 m cylinder cannot leave
    # the configured 0.52 m robot clearance beside the certified centreline in
    # a 3.08 m lane once wall clearance is included; the 0.318 m boxes can.
    # Slots remain randomized among equal-size assets so the visible clutter is
    # not tied to a fixed obstacle index.
    shuffled_slots = rng.permutation(radii.size)
    obstacle_slots = sorted(
        shuffled_slots, key=lambda index: radii[index])[:requested_obstacles]
    for obstacle_order, (obstacle_slot, lane) in enumerate(
            zip(obstacle_slots, obstacle_lanes)):
        radius = radii[obstacle_slot]
        left_inner, right_inner = lane_bounds[lane]
        x_low = left_inner + radius + obstacle_wall_clearance
        x_high = right_inner - radius - obstacle_wall_clearance
        placed = False
        for _ in range(8192):
            candidate = np.array((rng.uniform(x_low, x_high),
                                  rng.uniform(-2.85, 2.85)))
            spawn_dx = max(
                lane_centers[lane] - spawn_lateral_jitter - candidate[0],
                0.0,
                candidate[0] - lane_centers[lane] - spawn_lateral_jitter,
            )
            endpoint_clearance = min(
                math.hypot(spawn_dx, candidate[1] - endpoint_y)
                for endpoints in curriculum_endpoint_ys[lane]
                for endpoint_y in endpoints
            )
            if endpoint_clearance < radius + max(emma_clearance, goal_clearance):
                continue
            clearance = _point_clearance(
                candidate, positions, radii, obstacle_active, wall_poses,
                wall_active, lengths, wall_width, inner_half)
            if clearance < radius + obstacle_wall_clearance:
                continue
            if (_distance_to_polyline(candidate, route_waypoints[lane])
                    < radius + navigation_clearance + 0.08):
                continue
            same_lane = obstacle_active & (obstacle_lane_by_slot == lane)
            if same_lane.any():
                other = np.linalg.norm(
                    positions[same_lane] - candidate, axis=-1)
                required = radii[same_lane] + radius + min_obstacle_spacing
                if (other < required).any():
                    continue
            positions[obstacle_slot] = candidate
            obstacle_active[obstacle_slot] = True
            obstacle_lane_by_slot[obstacle_slot] = lane
            obstacle_yaws[obstacle_slot] = rng.uniform(0.0, 2.0 * math.pi)
            placed = True
            break
        if not placed and obstacle_order < lane_count:
            raise RuntimeError(
                "could not place mandatory dedicated-detour clutter "
                f"(seed={seed}, lane={lane}, slot={obstacle_slot}, "
                f"radius={radius:.3f}, requested={requested_obstacles})")

    def make_spawn_pairs(lane_ids, bottom_y=-spawn_y, top_y=spawn_y):
        pairs = np.empty((spawn_bank_size, 2, 2), dtype=np.float64)
        for index, lane in enumerate(lane_ids):
            bottom = np.array((lane_centers[lane] + rng.uniform(
                -spawn_lateral_jitter, spawn_lateral_jitter), bottom_y))
            top = np.array((lane_centers[lane] + rng.uniform(
                -spawn_lateral_jitter, spawn_lateral_jitter), top_y))
            pairs[index] = (
                (bottom, top) if rng.random() < 0.5 else (top, bottom)
            )
        return pairs

    spawn_lane_ids = np.arange(spawn_bank_size, dtype=np.int64) % lane_count
    rng.shuffle(spawn_lane_ids)
    spawn_pairs = make_spawn_pairs(spawn_lane_ids)
    if (np.linalg.norm(spawn_pairs[:, 1] - spawn_pairs[:, 0], axis=-1)
            <= min_goal_distance).any():
        raise RuntimeError("dedicated-detour spawn bank violates min_goal_distance")

    # Two levels share each fixed lane; the number of short walls their
    # certified routes cross grows with the lane (one, two, then all three).
    # Both levels keep their own level-specific gate on the route, so the six
    # width distributions survive without moving geometry between resets.
    curriculum_lane_ids = np.arange(curriculum_levels, dtype=np.int64) // 2
    curriculum_route_floors = np.asarray(
        lane_route_floors, dtype=np.float64
    )[curriculum_lane_ids]
    curriculum_passage_widths = np.empty(curriculum_levels, dtype=np.float64)
    curriculum_pairs = np.empty(
        (curriculum_levels, spawn_bank_size, 2, 2), dtype=np.float64
    )
    curriculum_spawn_lane_ids = np.empty(
        (curriculum_levels, spawn_bank_size), dtype=np.int64
    )
    for level, lane in enumerate(curriculum_lane_ids):
        lane_ids = np.full(spawn_bank_size, lane, dtype=np.int64)
        curriculum_spawn_lane_ids[level] = lane_ids
        bottom_y, top_y = curriculum_endpoint_ys[lane][level % 2]
        curriculum_pairs[level] = make_spawn_pairs(
            lane_ids, bottom_y=bottom_y, top_y=top_y
        )
        curriculum_passage_widths[level] = passage_widths[lane][
            0 if level % 2 == 0 else 2
        ]
    pair_lengths = np.linalg.norm(
        curriculum_pairs[:, :, 1] - curriculum_pairs[:, :, 0], axis=-1
    )
    if (pair_lengths <= curriculum_route_floors[:, None]).any():
        raise RuntimeError(
            "dedicated-detour curriculum bank violates min_goal_distance"
        )

    return {
        "positions": positions,
        "active": obstacle_active,
        "yaws": obstacle_yaws,
        "wall_poses": wall_poses,
        "wall_active": wall_active,
        "spawn_pairs": spawn_pairs,
        "curriculum_pairs": curriculum_pairs,
        "spawn_lane_ids": spawn_lane_ids,
        "curriculum_lane_ids": curriculum_lane_ids,
        "curriculum_spawn_lane_ids": curriculum_spawn_lane_ids,
        "curriculum_passage_widths": curriculum_passage_widths,
        "curriculum_route_floors": curriculum_route_floors,
        "curriculum_endpoint_ys": np.asarray(curriculum_endpoint_ys),
        "passage_widths": tuple(passage_widths),
        "barrier_counts": barrier_counts,
        "barrier_count": int(barrier_counts.sum()),
        "route_waypoints": tuple(route_waypoints),
        "gate_centers": tuple(gate_centers),
        "gate_guard": gate_guard,
        "minimum_gate_clearance": robot_half_width + collision_clearance,
        "lane_bounds": lane_bounds,
        "lane_centers": lane_centers,
    }
