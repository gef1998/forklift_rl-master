"""Check the geometry of task difficulty, independently of the simulator."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/structured_random_geometry.py"
spec = importlib.util.spec_from_file_location("structured_random_geometry", MODULE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
RADII = np.array([.50 if i % 2 == 0 else math.hypot(.5, .5)/2 for i in range(15)])


class CurriculumGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layouts = [g.sample_connected_layout(RADII, seed=i, include_curriculum_banks=True)
                       for i in range(64)]

    def test_difficulty_distances_and_seed(self):
        for layout in self.layouts:
            pairs = layout['curriculum_pairs']
            self.assertEqual(pairs.shape, (6, 128, 2, 2))
            distance = np.linalg.norm(pairs[:, :, 0] - pairs[:, :, 1], axis=-1)
            self.assertTrue(((distance[0] >= 1.5) & (distance[0] <= 3)).all())
            self.assertTrue(((distance[1] >= 3) & (distance[1] <= 5.2)).all())
            self.assertTrue((distance[2:] > 5.2).all())
            np.testing.assert_array_equal(pairs[2], layout['spawn_pairs'])
            np.testing.assert_array_equal(pairs[3], layout['spawn_pairs'])
        again = g.sample_connected_layout(RADII, seed=0, include_curriculum_banks=True)
        np.testing.assert_array_equal(again['curriculum_pairs'], self.layouts[0]['curriculum_pairs'])

    def test_easy_lines_are_clear_medium_prefers_detours(self):
        obstructed = total = 0
        for layout in self.layouts:
            pairs = layout['curriculum_pairs'][:2]
            lines = pairs[:, :, :1] + (pairs[:, :, 1:] - pairs[:, :, :1]) * np.linspace(0, 1, 129)[None, None, :, None]
            clear = g.point_clearance(lines, layout['positions'], RADII, layout['active'],
                                     layout['wall_poses'], layout['wall_active'],
                                     layout['wall_lengths'], .15, 4.85).min(axis=-1)
            self.assertTrue((clear[0] >= layout['navigation_clearance'] - 1e-9).all())
            obstructed += (clear[1] < layout['navigation_clearance']).sum()
            total += len(clear[1])
        self.assertGreater(obstructed / total, .70)

    def test_final_level_selects_denser_static_clutter_routes(self):
        baseline_scores, dense_scores = [], []
        previous_local_scores, final_local_scores = [], []
        for layout in self.layouts:
            args = (layout['positions'], RADII, layout['active'])
            baseline_scores.extend(g.static_clutter_path_density(
                layout['curriculum_pairs'][2], *args))
            dense_scores.extend(g.static_clutter_path_density(
                layout['curriculum_pairs'][4], *args))
            previous_local_scores.extend(g.static_clutter_path_density(
                layout['curriculum_pairs'][4], *args, influence_radius=.70))
            final_local_scores.extend(g.static_clutter_path_density(
                layout['curriculum_pairs'][5], *args, influence_radius=.70))
        self.assertGreater(np.mean(dense_scores), np.mean(baseline_scores) * 1.35)
        self.assertGreater(
            np.mean(final_local_scores), np.mean(previous_local_scores) * 1.05)

    def test_dense_training_configuration_remains_generatable(self):
        # Use maximum configured cylinder/cuboid bounds to stress the 18-slot
        # training layout rather than relying on one random template draw.
        radii = np.array([
            .40 if i % 2 == 0 else math.hypot(.45, .45) / 2 for i in range(20)
        ])
        layouts = [g.sample_connected_layout(
            radii,
            seed=seed,
            clutter_density_range=(.65, .90),
            min_obstacle_spacing=.60,
            loop_probability_range=(0.0, .40),
            emma_clearance=.70,
            goal_clearance=.50,
            include_curriculum_banks=True,
            max_layout_attempts=64,
            min_connected_free_fraction=.60,
        ) for seed in range(16)]
        self.assertGreater(np.mean([x['active'].sum() for x in layouts]), 12.0)
        self.assertTrue(all(x['curriculum_pairs'].shape == (6, 128, 2, 2)
                            for x in layouts))

    def test_all_levels_connected_with_start_and_goal_clearance(self):
        axis = np.linspace(-4.85, 4.85, math.ceil(9.7/.15)+1)
        step = axis[1] - axis[0]
        xx, yy = np.meshgrid(axis, axis)
        grid = np.stack((xx, yy), axis=-1)
        for layout in self.layouts:
            args = (layout['positions'], RADII, layout['active'], layout['wall_poses'],
                    layout['wall_active'], layout['wall_lengths'], .15, 4.85)
            pairs = layout['curriculum_pairs'].reshape(-1, 2, 2)
            clearance = g.point_clearance(pairs, *args)
            self.assertTrue((clearance[:, 0] >= .70).all())
            self.assertTrue((clearance[:, 1] >= .60).all())
            free = (g.point_clearance(grid, *args)
                    >= layout['navigation_clearance'] + step/math.sqrt(2))
            indices = np.rint((pairs[..., ::-1] + 4.85)/step).astype(int)
            reached = np.zeros_like(free)
            reached[tuple(indices[0, 0])] = True
            while True:
                expanded = reached.copy()
                expanded[1:] |= reached[:-1]
                expanded[:-1] |= reached[1:]
                expanded[:, 1:] |= reached[:, :-1]
                expanded[:, :-1] |= reached[:, 1:]
                expanded &= free
                if np.array_equal(expanded, reached):
                    break
                reached = expanded
            self.assertTrue(reached[indices[..., 0], indices[..., 1]].all())

    def test_empty_clutter_and_invalid_ranges(self):
        layout = g.sample_connected_layout([], seed=4, include_curriculum_banks=True)
        self.assertEqual(layout['curriculum_pairs'].shape, (6, 128, 2, 2))
        with self.assertRaises(ValueError):
            g.sample_connected_layout(RADII, seed=1, curriculum_distance_ranges=((1, 0), (3, 5)))


if __name__ == '__main__':
    unittest.main()
