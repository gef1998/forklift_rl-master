"""Tests for privileged reference-path projection."""

import importlib.util
from pathlib import Path
import unittest

import torch


MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/reference_path.py"
)
spec = importlib.util.spec_from_file_location("reference_path", MODULE)
reference_path = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_path)


class ReferencePathTests(unittest.TestCase):
    def test_projects_onto_polyline_arc_length(self):
        paths = torch.tensor([
            [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ])
        counts = torch.tensor([3, 2])
        points = torch.tensor([[1.0, 2.5], [0.0, -0.3]])
        progress, error = reference_path.project_points_to_polylines(
            points, paths, counts
        )
        torch.testing.assert_close(progress, torch.tensor([3.0, 1.0]))
        torch.testing.assert_close(error, torch.tensor([0.5, 0.3]))

    def test_unavailable_path_returns_zero_diagnostics(self):
        paths = torch.zeros(2, 3, 2)
        counts = torch.tensor([0, 1])
        points = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        progress, error = reference_path.project_points_to_polylines(
            points, paths, counts
        )
        torch.testing.assert_close(progress, torch.zeros(2))
        torch.testing.assert_close(error, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
