"""Pure geometry checks; renderer/PNG are exercised by the preview command."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

root = Path(__file__).resolve().parents[1] / 'source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav'
spec = importlib.util.spec_from_file_location('reference_path_visualization', root/'reference_path_visualization.py')
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


class ReferenceVisualizationTests(unittest.TestCase):
    def test_reverse_color_uses_arriving_segment_and_rotation_is_not_translation(self):
        plan = SimpleNamespace(success=True, poses=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1]]),
                               directions=np.array([0,1,0,-1]), arc_lengths=np.array([0,1,1,2]))
        lines = vis.reference_lines(plan, [0,0,0], [0,0], [])
        np.testing.assert_array_equal(lines['forward'], [[[0,0],[1,0]]])
        np.testing.assert_array_equal(lines['reverse'], [[[1,0],[0,0]]])
        failed = SimpleNamespace(success=False)
        lines = vis.reference_lines(failed, [0,0,0], [2,0], [])
        self.assertEqual(lines['forward'].shape, (0,2,2))
        self.assertEqual(lines['reverse'].shape, (0,2,2))
        self.assertGreater(len(lines['goal']), 0)

    def test_rotated_rectangle_has_correct_world_extent(self):
        corners = vis.rectangle([2,3,np.pi/2], .425, .275)
        np.testing.assert_allclose(corners.min(0), [2-.275,3-.425])
        np.testing.assert_allclose(corners.max(0), [2+.275,3+.425])


if __name__ == '__main__':
    unittest.main()
