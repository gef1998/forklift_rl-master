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
            self.assertEqual(pairs.shape, (3, 128, 2, 2))
            distance = np.linalg.norm(pairs[:, :, 0] - pairs[:, :, 1], axis=-1)
            self.assertTrue(((distance[0] >= 1.5) & (distance[0] <= 3)).all())
            self.assertTrue(((distance[1] >= 3) & (distance[1] <= 5.2)).all())
            self.assertTrue((distance[2] > 5.2).all())
            np.testing.assert_array_equal(pairs[2], layout['spawn_pairs'])
        again = g.sample_connected_layout(RADII, seed=0, include_curriculum_banks=True)
        np.testing.assert_array_equal(again['curriculum_pairs'], self.layouts[0]['curriculum_pairs'])

    def test_easy_lines_are_clear_medium_prefers_detours(self):
        obstructed = total = 0
        for layout in self.layouts:
            pairs = layout['curriculum_pairs'][:2]
            lines = pairs[:, :, :1] + (pairs[:, :, 1:] - pairs[:, :, :1]) * np.linspace(0, 1, 129)[None, None, :, None]
            clear = g.point_clearance(lines, layout['positions'], RADII, layout['active'],
                                     layout['wall_poses'], layout['wall_active'], 9.7/3, .15, 4.85).min(axis=-1)
            self.assertTrue((clear[0] >= .60 - 1e-9).all())
            obstructed += (clear[1] < .60).sum()
            total += len(clear[1])
        self.assertGreater(obstructed / total, .70)

    def test_all_levels_connected_with_start_and_goal_clearance(self):
        axis = np.linspace(-4.85, 4.85, math.ceil(9.7/.15)+1)
        step = axis[1] - axis[0]
        xx, yy = np.meshgrid(axis, axis)
        grid = np.stack((xx, yy), axis=-1)
        for layout in self.layouts:
            args = (layout['positions'], RADII, layout['active'], layout['wall_poses'],
                    layout['wall_active'], 9.7/3, .15, 4.85)
            pairs = layout['curriculum_pairs'].reshape(-1, 2, 2)
            clearance = g.point_clearance(pairs, *args)
            self.assertTrue((clearance[:, 0] >= .70).all())
            self.assertTrue((clearance[:, 1] >= .60).all())
            free = g.point_clearance(grid, *args) >= .60 + step/math.sqrt(2)
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
        self.assertEqual(layout['curriculum_pairs'].shape, (3, 128, 2, 2))
        with self.assertRaises(ValueError):
            g.sample_connected_layout(RADII, seed=1, curriculum_distance_ranges=((1, 0), (3, 5)))


if __name__ == '__main__':
    unittest.main()
