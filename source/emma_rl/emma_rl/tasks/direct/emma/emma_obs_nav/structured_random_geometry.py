"""Connected random wall graphs plus clutter; no Isaac Sim dependency.

Layouts are generated once per parallel environment. A randomized spanning tree
opens a square cell graph, then independently sampled extra edges create loops.
There are no S/ring/straight templates or scene labels. Normal wall jitter and
clutter continuously vary geometry within this distribution.
"""

from collections import deque
import math

import numpy as np


def wall_graph(arena_size, wall_thickness, grid_size):
    """Return cell edges and (x, y, yaw) for identical full-cell wall slots."""
    pitch = (arena_size - 2 * wall_thickness) / grid_size
    half = arena_size / 2 - wall_thickness
    edges, poses = [], []
    for row in range(grid_size):
        for col in range(grid_size):
            cell = row * grid_size + col
            if col + 1 < grid_size:
                edges.append((cell, cell + 1))
                poses.append((-half + (col + 1) * pitch,
                              -half + (row + .5) * pitch, math.pi / 2))
            if row + 1 < grid_size:
                edges.append((cell, cell + grid_size))
                poses.append((-half + (col + .5) * pitch,
                              -half + (row + 1) * pitch, 0.0))
    return np.asarray(edges), np.asarray(poses, dtype=np.float64), pitch


def point_clearance(points, positions, radii, active, wall_poses,
                    wall_active, wall_length, wall_width, inner_half):
    """Distance to outer walls, conservative clutter circles and exact wall OBBs."""
    result = inner_half - np.abs(points).max(axis=-1)
    for center, radius in zip(positions[active], radii[active]):
        result = np.minimum(result, np.linalg.norm(points - center, axis=-1) - radius)
    wall_lengths = np.broadcast_to(np.asarray(wall_length, dtype=np.float64),
                                   (len(wall_poses),))
    for (x, y, yaw), length in zip(wall_poses[wall_active], wall_lengths[wall_active]):
        delta = points - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.stack((c * delta[..., 0] + s * delta[..., 1],
                          -s * delta[..., 0] + c * delta[..., 1]), axis=-1)
        q = np.abs(local) - (length / 2, wall_width / 2)
        distance = np.linalg.norm(np.maximum(q, 0), axis=-1)
        distance += np.minimum(q.max(axis=-1), 0)
        result = np.minimum(result, distance)
    return result


def largest_component(mask):
    """Four-neighbour flood fill, avoiding diagonal corner cutting."""
    unseen = mask.copy()
    best = []
    rows, cols = mask.shape
    for row, col in zip(*np.nonzero(mask)):
        if not unseen[row, col]:
            continue
        unseen[row, col] = False
        queue = deque([(row, col)])
        component = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and unseen[nr, nc]:
                    unseen[nr, nc] = False
                    queue.append((nr, nc))
        if len(component) > len(best):
            best = component
    return np.asarray(best, dtype=np.int64).reshape(-1, 2)


def straight_path_is_clear(pairs, positions, radii, active, wall_poses,
                           wall_active, wall_length, wall_width, clearance):
    """Conservative swept-disc LOS check: circles and inflated wall boxes."""
    start, end = pairs[:, 0], pairs[:, 1]
    delta = end - start
    length_sq = np.square(delta).sum(axis=-1).clip(1e-12)
    safe = np.ones(len(pairs), dtype=bool)
    for center, radius in zip(positions[active], radii[active]):
        t = ((center - start) * delta).sum(axis=-1) / length_sq
        nearest = start + t.clip(0, 1)[:, None] * delta
        safe &= np.linalg.norm(nearest - center, axis=-1) >= radius + clearance
    wall_lengths = np.broadcast_to(np.asarray(wall_length, dtype=np.float64),
                                   (len(wall_poses),))
    for (x, y, yaw), length in zip(wall_poses[wall_active], wall_lengths[wall_active]):
        half = np.array([length / 2, wall_width / 2]) + clearance
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        local_start = (start - (x, y)) @ rotation
        local_delta = delta @ rotation
        parallel = np.abs(local_delta) < 1e-10
        parallel_outside = (parallel & (np.abs(local_start) > half)).any(axis=-1)
        divisor = np.where(parallel, 1.0, local_delta)
        t1, t2 = (-half - local_start) / divisor, (half - local_start) / divisor
        enter = np.where(parallel, -np.inf, np.minimum(t1, t2)).max(axis=-1)
        leave = np.where(parallel, np.inf, np.maximum(t1, t2)).min(axis=-1)
        intersects = ~parallel_outside & (enter <= leave) & (leave >= 0) & (enter <= 1)
        safe &= ~intersects
    return safe


def static_clutter_path_density(pairs, positions, radii, active, influence_radius=1.25):
    """Score how strongly static clutter surrounds each start-to-goal chord.

    This is a task-bank ranking metric, not a collision check. Each obstacle
    contributes most when its surface intersects the chord and decays linearly
    to zero at ``influence_radius``. All selected pairs are still certified to
    lie in the same collision-free connected component by the layout sampler.
    """
    pairs = np.asarray(pairs, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if pairs.ndim != 3 or pairs.shape[1:] != (2, 2):
        raise ValueError("pairs must have shape (N, 2, 2)")
    if not math.isfinite(influence_radius) or influence_radius <= 0:
        raise ValueError("influence_radius must be positive")
    if not active.any():
        return np.zeros(len(pairs), dtype=np.float64)
    start, end = pairs[:, 0], pairs[:, 1]
    delta = end - start
    length_sq = np.square(delta).sum(axis=-1).clip(1e-12)
    centers = positions[active]
    projection = ((centers[None] - start[:, None]) * delta[:, None]).sum(axis=-1)
    projection /= length_sq[:, None]
    nearest = start[:, None] + projection.clip(0, 1)[..., None] * delta[:, None]
    edge_distance = np.linalg.norm(nearest - centers[None], axis=-1) - radii[active]
    contribution = np.clip(1.0 - np.maximum(edge_distance, 0.0) / influence_radius, 0.0, 1.0)
    return contribution.sum(axis=-1)


def _curriculum_banks(starts, goals, full_pairs, rng, geometry, distance_ranges,
                      min_goal_distance, dense_influence_radius,
                      ultra_dense_influence_radius):
    banks = []
    count = len(full_pairs)
    for level, (low, high) in enumerate(distance_ranges):
        collected = []
        remaining = count
        for _ in range(32):
            a = starts[rng.integers(len(starts), size=count * 16)]
            b = goals[rng.integers(len(goals), size=len(a))]
            distance = np.linalg.norm(a - b, axis=-1)
            valid = (distance >= low) & (distance <= high)
            pairs = np.stack((a[valid], b[valid]), axis=1)
            clear = straight_path_is_clear(pairs, **geometry)
            if level == 0:
                pairs = pairs[clear]
            else:
                # Prefer direct-line obstruction when available, but keep open
                # layouts valid: geometry is never fabricated to fill a quota.
                pairs = np.concatenate((pairs[~clear], pairs[clear]))
            pairs = pairs[:remaining]
            collected.append(pairs)
            remaining -= len(pairs)
            if not remaining:
                banks.append(np.concatenate(collected))
                break
        if remaining:
            return None

    # Levels 2 and 3 retain the original long-range task bank. Level 3 raises
    # noise only. Levels 4 and 5 rank a much larger candidate set by nearby
    # static clutter at progressively tighter radii, so they traverse the dense
    # parts of the same immutable map. This preserves fast Warp LiDAR: no
    # geometry is moved or rebuilt at reset.
    candidate_count = max(count * 64, count)
    a = starts[rng.integers(len(starts), size=candidate_count)]
    b = goals[rng.integers(len(goals), size=candidate_count)]
    candidates = np.stack((a, b), axis=1)
    distance = np.linalg.norm(a - b, axis=-1)
    candidates = candidates[distance > min_goal_distance]
    if len(candidates) < count:
        return None
    density = static_clutter_path_density(
        candidates,
        geometry["positions"],
        geometry["radii"],
        geometry["active"],
        dense_influence_radius,
    )
    # Stable sorting keeps seeded generation reproducible when scores tie.
    dense_pairs = candidates[np.argsort(-density, kind="stable")[:count]]
    ultra_density = static_clutter_path_density(
        candidates,
        geometry["positions"],
        geometry["radii"],
        geometry["active"],
        ultra_dense_influence_radius,
    )
    ultra_dense_pairs = candidates[
        np.argsort(-ultra_density, kind="stable")[:count]
    ]
    return np.stack((
        *banks,
        full_pairs,
        full_pairs,
        dense_pairs,
        ultra_dense_pairs,
    ))


def sample_connected_layout(radii, *, seed, arena_size=10.0, wall_thickness=.15,
                            grid_size=3, interior_wall_thickness=.15,
                            wall_jitter=.20, loop_probability_range=(.0, .65),
                            clutter_density_range=(.35, 1.0),
                            min_obstacle_spacing=1.5, obstacle_wall_clearance=.30,
                            navigation_clearance=.52, grid_resolution=.15,
                            emma_clearance=.70, goal_clearance=.60,
                            min_goal_distance=5.2, spawn_bank_size=128,
                            max_layout_attempts=32, include_curriculum_banks=False,
                            curriculum_distance_ranges=((1.5, 3.0), (3.0, 5.2)),
                            narrow_passage_probability=.85,
                            narrow_passage_width_range=(1.20, 1.50),
                            max_narrow_passages=2,
                            narrow_pair_fraction=.50,
                            dense_clutter_influence_radius=1.25,
                            ultra_dense_clutter_influence_radius=.70,
                            min_connected_free_fraction=.85):
    """Sample one joint layout and bank of safe, connected start/goal pairs.

    Fixed-size slots simplify physics cloning. Inactive clutter/walls are hidden
    underground by the manager and excluded from both ray meshes and clearance.
    The grid is inflated by half its diagonal in addition to the robot radius:
    adjacent free grid nodes therefore certify a continuous collision-free path.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("radii must be a finite positive vector")
    if len(curriculum_distance_ranges) != 2 or any(
        len(r) != 2 or not 0 < r[0] < r[1] or not all(map(math.isfinite, r))
        for r in curriculum_distance_ranges
    ):
        raise ValueError("two positive curriculum distance ranges are required")
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    for value in (arena_size, wall_thickness, interior_wall_thickness,
                  navigation_clearance, grid_resolution, emma_clearance,
                  goal_clearance, min_goal_distance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("geometry dimensions and clearances must be positive")
    for limits in (loop_probability_range, clutter_density_range):
        if len(limits) != 2 or not 0 <= limits[0] <= limits[1] <= 1:
            raise ValueError("probability/density ranges must lie in [0, 1]")
    if not 0 <= narrow_passage_probability <= 1:
        raise ValueError("narrow_passage_probability must lie in [0, 1]")
    if (len(narrow_passage_width_range) != 2
            or not 0 < narrow_passage_width_range[0] <= narrow_passage_width_range[1]
            or not all(map(math.isfinite, narrow_passage_width_range))):
        raise ValueError("narrow_passage_width_range must contain two positive bounds")
    if not isinstance(max_narrow_passages, int) or max_narrow_passages < 0:
        raise ValueError("max_narrow_passages must be a non-negative integer")
    if not 0 <= narrow_pair_fraction <= 1:
        raise ValueError("narrow_pair_fraction must lie in [0, 1]")
    if (not math.isfinite(dense_clutter_influence_radius)
            or dense_clutter_influence_radius <= 0):
        raise ValueError("dense_clutter_influence_radius must be positive")
    if (not math.isfinite(ultra_dense_clutter_influence_radius)
            or not 0 < ultra_dense_clutter_influence_radius
            < dense_clutter_influence_radius):
        raise ValueError(
            "ultra_dense_clutter_influence_radius must lie between zero and "
            "dense_clutter_influence_radius"
        )
    if (not math.isfinite(min_connected_free_fraction)
            or not 0 < min_connected_free_fraction <= 1):
        raise ValueError("min_connected_free_fraction must lie in (0, 1]")
    if (not math.isfinite(wall_jitter) or wall_jitter < 0
            or min_obstacle_spacing < 0 or obstacle_wall_clearance < 0
            or spawn_bank_size < 1 or max_layout_attempts < 1):
        raise ValueError("invalid sampling limits")
    inner_half = arena_size / 2 - wall_thickness
    edges, base_poses, pitch = wall_graph(arena_size, wall_thickness, grid_size)
    if narrow_passage_width_range[1] >= pitch:
        raise ValueError("narrow passages must be shorter than a cell edge")
    gate_segment_length = (pitch - narrow_passage_width_range[1]) / 2
    wall_lengths = np.concatenate((
        np.full(len(edges), pitch),
        np.full(2 * max_narrow_passages, gate_segment_length),
    ))
    # Keep room cores wide enough. Jitter changes actual junction geometry, so
    # reachability below is checked against the boxes, not just the cell graph.
    if pitch - 2 * wall_jitter - interior_wall_thickness <= 2 * navigation_clearance:
        raise ValueError("cells are too narrow for the navigation clearance")
    if min_goal_distance >= 2 * math.sqrt(2) * (inner_half - max(emma_clearance, goal_clearance)):
        raise ValueError("goal distance cannot fit inside the arena")
    axis = np.linspace(-inner_half, inner_half,
                       math.ceil(2 * inner_half / grid_resolution) + 1)
    step = float(axis[1] - axis[0])
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack((xx, yy), axis=-1)
    grid_margin = step / math.sqrt(2)
    rng = np.random.default_rng(seed)

    for _ in range(max_layout_attempts):
        # Random Kruskal tree, followed by independent loop creation.
        parent = list(range(grid_size ** 2))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        wall_active = np.ones(len(edges), dtype=bool)
        for index in rng.permutation(len(edges)):
            a, b = (root(int(i)) for i in edges[index])
            if a != b:
                parent[a] = b
                wall_active[index] = False
        loop_probability = rng.uniform(*loop_probability_range)
        wall_active &= rng.random(len(edges)) >= loop_probability
        graph_wall_poses = base_poses.copy()
        jitter = rng.uniform(-wall_jitter, wall_jitter, len(edges))
        graph_wall_poses[:, 0] -= np.sin(graph_wall_poses[:, 2]) * jitter
        graph_wall_poses[:, 1] += np.cos(graph_wall_poses[:, 2]) * jitter

        # Replace selected open graph edges with two short wall segments. Gates
        # are chosen from graph bridges so some start/goal pairs must traverse
        # them; they remain part of this procedural wall graph distribution.
        open_indices = np.flatnonzero(~wall_active)
        bridges = []
        bridge_components = {}
        for candidate in open_indices:
            start, target = map(int, edges[candidate])
            neighbours = [[] for _ in range(grid_size ** 2)]
            for edge_index in open_indices:
                if edge_index == candidate:
                    continue
                a, b = map(int, edges[edge_index])
                neighbours[a].append(b)
                neighbours[b].append(a)
            stack, seen = [start], {start}
            while stack:
                node = stack.pop()
                for neighbour in neighbours[node]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            if target not in seen:
                bridges.append(candidate)
                bridge_components[candidate] = seen

        gate_edges = []
        if (max_narrow_passages and open_indices.size
                and rng.random() < narrow_passage_probability):
            candidates = bridges if bridges else open_indices.tolist()
            count = min(max_narrow_passages, len(candidates))
            if count > 1:
                count = 1 + int(rng.random() < .5)
            gate_edges = rng.choice(candidates, size=count, replace=False).tolist()
        wall_poses = np.concatenate((
            graph_wall_poses,
            np.zeros((2 * max_narrow_passages, 3), dtype=np.float64),
        ))
        slot_active = np.concatenate((
            wall_active,
            np.zeros(2 * max_narrow_passages, dtype=bool),
        ))
        gate_widths = []
        for gate_index, edge_index in enumerate(gate_edges):
            width = rng.uniform(*narrow_passage_width_range)
            gate_widths.append(width)
            center = graph_wall_poses[edge_index, :2]
            yaw = graph_wall_poses[edge_index, 2]
            tangent = np.array([math.cos(yaw), math.sin(yaw)])
            offset = (width + gate_segment_length) / 2
            first = len(edges) + 2 * gate_index
            wall_poses[first] = (*tuple(center - tangent * offset), yaw)
            wall_poses[first + 1] = (*tuple(center + tangent * offset), yaw)
            slot_active[first:first + 2] = True
        gate_components = [bridge_components[index] for index in gate_edges
                           if index in bridge_components]
        positions = np.zeros((len(radii), 2))
        active = np.zeros(len(radii), dtype=bool)
        density = rng.uniform(*clutter_density_range)
        requested = rng.random(len(radii)) < density
        for index in rng.permutation(len(radii)):
            if not requested[index]:
                continue
            radius = radii[index]
            limit = inner_half - radius - obstacle_wall_clearance
            if limit <= 0:
                raise ValueError("clutter does not fit inside the arena")
            candidates = rng.uniform(-limit, limit, (128, 2))
            wall_clear = point_clearance(
                candidates, positions, radii, np.zeros_like(active), wall_poses,
                slot_active, wall_lengths, interior_wall_thickness, inner_half)
            safe = wall_clear >= radius + obstacle_wall_clearance
            if active.any():
                distance = np.linalg.norm(candidates[:, None] - positions[active], axis=-1)
                safe &= (distance >= radius + radii[active] + min_obstacle_spacing).all(axis=-1)
            valid = np.flatnonzero(safe)
            if len(valid):
                positions[index] = candidates[rng.choice(valid)]
                active[index] = True

        clearance = point_clearance(points, positions, radii, active, wall_poses,
                                    slot_active, wall_lengths, interior_wall_thickness, inner_half)
        free = clearance >= navigation_clearance + grid_margin
        component = largest_component(free)
        # Reject fragmented layouts instead of silently training only in a tiny room.
        if (len(component) < min_connected_free_fraction * np.count_nonzero(free)
                or len(component) < 2):
            continue
        rr, cc = component.T
        safe_points = points[rr, cc]
        clear = clearance[rr, cc]
        starts = safe_points[clear >= emma_clearance]
        goals = safe_points[clear >= max(goal_clearance, navigation_clearance)]
        if not len(starts) or not len(goals):
            continue
        # Bounded rejection; the bank is drawn from one certified component.
        forced_target = round(spawn_bank_size * narrow_pair_fraction) if gate_components else 0
        regular_target = spawn_bank_size - forced_target
        forced_banks, regular_banks = [], []
        forced_remaining, regular_remaining = forced_target, regular_target
        for _ in range(32):
            a = starts[rng.integers(len(starts), size=spawn_bank_size * 16)]
            b = goals[rng.integers(len(goals), size=len(a))]
            valid = np.linalg.norm(a - b, axis=-1) > min_goal_distance
            pairs = np.stack((a[valid], b[valid]), axis=1)
            forced = np.zeros(len(pairs), dtype=bool)
            if gate_components and len(pairs):
                def cell_index(point):
                    col = np.floor((point[:, 0] + inner_half) / pitch).astype(int)
                    row = np.floor((point[:, 1] + inner_half) / pitch).astype(int)
                    return (row.clip(0, grid_size - 1) * grid_size
                            + col.clip(0, grid_size - 1))
                cell_a, cell_b = cell_index(pairs[:, 0]), cell_index(pairs[:, 1])
                for component in gate_components:
                    forced |= (np.isin(cell_a, list(component))
                               != np.isin(cell_b, list(component)))
            take_forced = pairs[forced][:forced_remaining]
            take_regular = pairs[~forced][:regular_remaining]
            forced_banks.append(take_forced)
            regular_banks.append(take_regular)
            forced_remaining -= len(take_forced)
            regular_remaining -= len(take_regular)
            if not forced_remaining and not regular_remaining:
                forced_pairs = (np.concatenate(forced_banks) if forced_target
                                else np.empty((0, 2, 2)))
                regular_pairs = np.concatenate(regular_banks)
                spawn_pairs = np.concatenate((forced_pairs, regular_pairs))
                order = rng.permutation(spawn_bank_size)
                spawn_pairs = spawn_pairs[order]
                narrow_pair_mask = np.concatenate((
                    np.ones(forced_target, dtype=bool),
                    np.zeros(regular_target, dtype=bool),
                ))[order]
                layout = dict(positions=positions, active=active, wall_poses=wall_poses,
                              wall_active=slot_active, wall_lengths=wall_lengths,
                              spawn_pairs=spawn_pairs,
                              open_edges=edges[~wall_active], grid_step=step,
                              navigation_clearance=navigation_clearance,
                              narrow_gate_edges=edges[gate_edges],
                              narrow_gate_widths=np.asarray(gate_widths),
                              narrow_pair_mask=narrow_pair_mask)
                if include_curriculum_banks:
                    course_banks = _curriculum_banks(
                        starts, goals, layout["spawn_pairs"], rng,
                        dict(positions=positions, radii=radii, active=active,
                             wall_poses=wall_poses, wall_active=slot_active,
                             wall_length=wall_lengths, wall_width=interior_wall_thickness,
                             clearance=navigation_clearance), curriculum_distance_ranges,
                        min_goal_distance, dense_clutter_influence_radius,
                        ultra_dense_clutter_influence_radius)
                    if course_banks is None:
                        break  # Retry the joint map, never silently weaken stage constraints.
                    layout["curriculum_pairs"] = course_banks
                return layout
    raise RuntimeError("unable to generate connected random layout; relax density/clearances")
