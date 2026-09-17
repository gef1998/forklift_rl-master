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
    for x, y, yaw in wall_poses[wall_active]:
        delta = points - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.stack((c * delta[..., 0] + s * delta[..., 1],
                          -s * delta[..., 0] + c * delta[..., 1]), axis=-1)
        q = np.abs(local) - (wall_length / 2, wall_width / 2)
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


def sample_connected_layout(radii, *, seed, arena_size=10.0, wall_thickness=.15,
                            grid_size=3, interior_wall_thickness=.15,
                            wall_jitter=.20, loop_probability_range=(.0, .65),
                            clutter_density_range=(.35, 1.0),
                            min_obstacle_spacing=1.5, obstacle_wall_clearance=.30,
                            navigation_clearance=.60, grid_resolution=.15,
                            emma_clearance=.70, goal_clearance=.60,
                            min_goal_distance=5.2, spawn_bank_size=128,
                            max_layout_attempts=32):
    """Sample one joint layout and bank of safe, connected start/goal pairs.

    Fixed-size slots simplify physics cloning. Inactive clutter/walls are hidden
    underground by the manager and excluded from both ray meshes and clearance.
    The grid is inflated by half its diagonal in addition to the robot radius:
    adjacent free grid nodes therefore certify a continuous collision-free path.
    """
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("radii must be a finite positive vector")
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
    if (not math.isfinite(wall_jitter) or wall_jitter < 0
            or min_obstacle_spacing < 0 or obstacle_wall_clearance < 0
            or spawn_bank_size < 1 or max_layout_attempts < 1):
        raise ValueError("invalid sampling limits")
    inner_half = arena_size / 2 - wall_thickness
    edges, base_poses, pitch = wall_graph(arena_size, wall_thickness, grid_size)
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
        wall_poses = base_poses.copy()
        jitter = rng.uniform(-wall_jitter, wall_jitter, len(edges))
        wall_poses[:, 0] -= np.sin(wall_poses[:, 2]) * jitter
        wall_poses[:, 1] += np.cos(wall_poses[:, 2]) * jitter
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
                wall_active, pitch, interior_wall_thickness, inner_half)
            safe = wall_clear >= radius + obstacle_wall_clearance
            if active.any():
                distance = np.linalg.norm(candidates[:, None] - positions[active], axis=-1)
                safe &= (distance >= radius + radii[active] + min_obstacle_spacing).all(axis=-1)
            valid = np.flatnonzero(safe)
            if len(valid):
                positions[index] = candidates[rng.choice(valid)]
                active[index] = True

        clearance = point_clearance(points, positions, radii, active, wall_poses,
                                    wall_active, pitch, interior_wall_thickness, inner_half)
        free = clearance >= navigation_clearance + grid_margin
        component = largest_component(free)
        # Reject fragmented layouts instead of silently training only in a tiny room.
        if len(component) < .90 * np.count_nonzero(free) or len(component) < 2:
            continue
        rr, cc = component.T
        safe_points = points[rr, cc]
        clear = clearance[rr, cc]
        starts = safe_points[clear >= emma_clearance]
        goals = safe_points[clear >= max(goal_clearance, navigation_clearance)]
        if not len(starts) or not len(goals):
            continue
        # Bounded rejection; the bank is drawn from one certified component.
        banks = []
        remaining = spawn_bank_size
        for _ in range(16):
            a = starts[rng.integers(len(starts), size=spawn_bank_size * 16)]
            b = goals[rng.integers(len(goals), size=len(a))]
            valid = np.linalg.norm(a - b, axis=-1) > min_goal_distance
            pairs = np.stack((a[valid], b[valid]), axis=1)[:remaining]
            banks.append(pairs)
            remaining -= len(pairs)
            if not remaining:
                return dict(positions=positions, active=active, wall_poses=wall_poses,
                            wall_active=wall_active, spawn_pairs=np.concatenate(banks),
                            open_edges=edges[~wall_active], grid_step=step,
                            navigation_clearance=navigation_clearance)
    raise RuntimeError("unable to generate connected random layout; relax density/clearances")
