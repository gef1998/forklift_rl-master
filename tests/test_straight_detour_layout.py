"""Geometry tests for reserved three-lane straight detour environments."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav"
    / "straight_detour_geometry.py"
)
spec = importlib.util.spec_from_file_location("straight_detour_geometry", MODULE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)

FULL_LENGTH = (10.0 - 2 * 0.15) / 3
BASE_WALL_LENGTHS = [FULL_LENGTH] * 12 + [(FULL_LENGTH - 1.5) / 2] * 4
LEVEL_RANGES = (
    (1.10, 1.55),
    (1.10, 1.55),
    (0.85, 1.30),
    (0.85, 1.30),
    (0.60, 1.00),
    (0.60, 1.00),
)
DEDICATED_PATTERNS = []
WIDTH_SAMPLES = 3
for lane in range(3):
    available = FULL_LENGTH - (.15 / 2 if lane == 1 else 0.0)
    lower_range = LEVEL_RANGES[2 * lane]
    upper_range = LEVEL_RANGES[2 * lane + 1]
    DEDICATED_PATTERNS.extend(
        available - width
        for width in np.linspace(*lower_range, WIDTH_SAMPLES)
    )
    DEDICATED_PATTERNS.append(
        available - max(lower_range[1], upper_range[1])
    )
    DEDICATED_PATTERNS.extend(
        available - width
        for width in np.linspace(*upper_range, WIDTH_SAMPLES)
    )
WALL_LENGTHS = np.array(BASE_WALL_LENGTHS + DEDICATED_PATTERNS)
RADII = np.array([0.40 if i % 2 == 0 else math.hypot(.45, .45) / 2
                  for i in range(20)])


class StraightDetourLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layouts = [geometry.sample_straight_detour_layout(
            RADII, WALL_LENGTHS, seed=seed) for seed in range(96)]

    def test_seed_reproducibility_and_randomized_difficulty(self):
        first = geometry.sample_straight_detour_layout(
            RADII, WALL_LENGTHS, seed=0)
        for name in ("positions", "active", "yaws", "wall_poses",
                     "wall_active", "spawn_pairs", "barrier_counts"):
            np.testing.assert_array_equal(first[name], self.layouts[0][name])
        widths = np.concatenate([
            layout["curriculum_passage_widths"] for layout in self.layouts
        ])
        self.assertGreater(np.ptp(widths), .85)
        self.assertTrue(((.60 - 1e-6 <= widths)
                         & (widths <= 1.55 + 1e-6)).all())
        self.assertLessEqual(widths.min(), .61)
        self.assertGreaterEqual(
            (widths / 2).min(),
            self.layouts[0]["minimum_gate_clearance"] - 1e-6,
        )
        counts = np.concatenate([layout["barrier_counts"] for layout in self.layouts])
        self.assertEqual(set(counts), {3})

    def test_passage_width_is_the_only_dedicated_curriculum_variable(self):
        expected_level_lanes = np.array((0, 0, 1, 1, 2, 2))
        for layout in self.layouts:
            np.testing.assert_array_equal(
                layout["curriculum_lane_ids"], expected_level_lanes
            )
            for level, lane in enumerate(expected_level_lanes):
                low, high = LEVEL_RANGES[level]
                width = layout["curriculum_passage_widths"][level]
                self.assertGreaterEqual(width, low - 1e-6)
                self.assertLessEqual(width, high + 1e-6)
                lane_ids = layout["curriculum_spawn_lane_ids"][level]
                self.assertTrue((lane_ids == lane).all())
                pairs = layout["curriculum_pairs"][level]
                left, right = layout["lane_bounds"][lane]
                self.assertTrue(((left <= pairs[:, :, 0])
                                 & (pairs[:, :, 0] <= right)).all())
                y_low, y_high = layout["curriculum_endpoint_ys"][lane, level % 2]
                self.assertTrue(np.allclose(
                    np.sort(pairs[:, :, 1], axis=1), (y_low, y_high)
                ))
                gate_y = layout["gate_centers"][lane][:, 1]
                crossed = np.flatnonzero((y_low < gate_y) & (gate_y < y_high))
                np.testing.assert_array_equal(
                    crossed, (0, 1) if level % 2 == 0 else (1, 2)
                )

    def test_three_lanes_are_sealed_and_each_has_multiple_barriers(self):
        inner_half = 5.0 - .15
        expected_dividers = np.linspace(-inner_half, inner_half, 4)[1:-1]
        for layout in self.layouts:
            poses = layout["wall_poses"]
            self.assertEqual(len(layout["route_waypoints"]), 3)
            self.assertTrue((layout["barrier_counts"] == 3).all())
            for divider, x in enumerate(expected_dividers):
                segments = poses[divider * 3:(divider + 1) * 3]
                self.assertTrue(np.allclose(segments[:, 0], x))
                self.assertTrue(np.allclose(segments[:, 1],
                                            [-FULL_LENGTH, 0, FULL_LENGTH]))
                self.assertTrue(np.allclose(segments[:, 2], math.pi / 2))
                self.assertAlmostEqual(
                    abs(segments[0, 1]) + FULL_LENGTH / 2, inner_half)

    def test_all_three_certified_routes_and_spawn_pairs_are_safe(self):
        inner_half = 5.0 - .15
        for seed, layout in enumerate(self.layouts):
            for lane, route in enumerate(layout["route_waypoints"]):
                samples = []
                for start, end in zip(route[:-1], route[1:]):
                    samples.append(start + (end - start)
                                   * np.linspace(0, 1, 81)[:, None])
                samples = np.concatenate(samples)
                clearance = np.array([geometry._point_clearance(
                    point, layout["positions"], RADII, layout["active"],
                    layout["wall_poses"], layout["wall_active"],
                    WALL_LENGTHS, .15, inner_half) for point in samples])
                minimum_index = int(clearance.argmin())
                self.assertGreaterEqual(
                    clearance[minimum_index],
                    layout["minimum_gate_clearance"] - 1e-6,
                    f"seed={seed} lane={lane} point={samples[minimum_index]}",
                )

                gates = layout["gate_centers"][lane]
                guard = layout["gate_guard"]
                for gate_index, (gate_x, gate_y) in enumerate(gates):
                    triplet = route[1 + 3 * gate_index:4 + 3 * gate_index]
                    np.testing.assert_allclose(
                        triplet,
                        [[gate_x, gate_y - guard], [gate_x, gate_y],
                         [gate_x, gate_y + guard]],
                    )
            pairs = np.concatenate((
                layout["spawn_pairs"][None], layout["curriculum_pairs"]
            ))
            self.assertEqual(layout["spawn_pairs"].shape, (128, 2, 2))
            self.assertEqual(pairs.shape, (7, 128, 2, 2))
            self.assertEqual(set(layout["spawn_lane_ids"]), {0, 1, 2})
            self.assertTrue((np.linalg.norm(pairs[:, :, 1] - pairs[:, :, 0], axis=-1)
                             > 5.20).all())
            endpoint_clearance = np.array([geometry._point_clearance(
                point, layout["positions"], RADII, layout["active"],
                layout["wall_poses"], layout["wall_active"],
                WALL_LENGTHS, .15, inner_half)
                for point in pairs.reshape(-1, 2)])
            self.assertGreaterEqual(endpoint_clearance.min(), .70 - 1e-6)

    def test_at_least_three_obstacles_with_one_in_each_lane(self):
        inner_half = 5.0 - .15
        for layout in self.layouts:
            active = layout["active"]
            self.assertGreaterEqual(active.sum(), 3)
            self.assertLessEqual(active.sum(), 6)
            lane_counts = np.zeros(3, dtype=int)
            for index in np.flatnonzero(active):
                point = layout["positions"][index]
                lane = np.flatnonzero(
                    (layout["lane_bounds"][:, 0] <= point[0])
                    & (point[0] <= layout["lane_bounds"][:, 1]))
                self.assertEqual(len(lane), 1)
                lane_counts[lane[0]] += 1
                other_active = active.copy()
                other_active[index] = False
                clearance = geometry._point_clearance(
                    point, layout["positions"], RADII, other_active,
                    layout["wall_poses"], layout["wall_active"], WALL_LENGTHS,
                    .15, inner_half)
                self.assertGreaterEqual(clearance, RADII[index] + .10 - 1e-6)
                route_distance = geometry._distance_to_polyline(
                    point, layout["route_waypoints"][lane[0]])
                self.assertGreaterEqual(route_distance, RADII[index] + .60 - 1e-6)
            self.assertTrue((lane_counts >= 1).all())
        self.assertTrue(any(layout["active"].sum() > 3 for layout in self.layouts))

    def test_invalid_counts_and_too_narrow_passage_are_rejected(self):
        for options in (
            {"passage_width_range": (.56, 1.2)},
            {"barrier_count_range": (2, 3)},
            {"extra_obstacle_count_range": (2, 5)},
            {"curriculum_passage_width_ranges": LEVEL_RANGES[:-1]},
        ):
            with self.assertRaises(ValueError):
                geometry.sample_straight_detour_layout(
                    RADII, WALL_LENGTHS, seed=1, **options)


if __name__ == "__main__":
    unittest.main()
