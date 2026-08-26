#!/usr/bin/env python3
"""Pure Torch tests for the deployment LiDAR Costmap contract."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/emma_rl/emma_rl/tasks/direct/emma/emma_costmap_nav/lidar_costmap.py"
)
SPEC = importlib.util.spec_from_file_location("lidar_costmap", MODULE_PATH)
lidar_costmap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = lidar_costmap
SPEC.loader.exec_module(lidar_costmap)

CostmapSpec = lidar_costmap.CostmapSpec
lidar_ranges_to_costmap = lidar_costmap.lidar_ranges_to_costmap


def _spec(inflation: float = 0.0) -> CostmapSpec:
    return CostmapSpec(inflation_radius_m=inflation)


def _angles(spec: CostmapSpec, rays: int = 220) -> torch.Tensor:
    step = spec.angle_span_rad / rays
    return spec.angle_min_rad + (torch.arange(rays) + 0.5) * step


def test_shape_and_front_only_observed_mask():
    spec = _spec()
    ranges = torch.full((2, 220), spec.lidar_max_range_m)
    costmap = lidar_ranges_to_costmap(ranges, _angles(spec), spec)
    assert costmap.shape == (2, 2, 48, 48)
    occupancy, observed = costmap[:, 0], costmap[:, 1]
    assert not occupancy.any()

    # Far rear cells are outside the 220-degree sensor sector and stay unknown.
    coords = (torch.arange(48) + 0.5) * spec.cell_size_m - spec.grid_size_m / 2.0
    rear_cols = coords < -1.0
    assert not observed[:, :, rear_cols].any()
    assert observed.any()


def test_first_return_marks_endpoint_and_occludes_cells_behind_it():
    spec = _spec()
    ranges = torch.full((1, 220), spec.lidar_max_range_m)
    # A short wall-like return over the central forward beams.
    ranges[:, 105:115] = 2.0
    costmap = lidar_ranges_to_costmap(ranges, _angles(spec), spec)
    occupancy, observed = costmap[0, 0], costmap[0, 1]
    assert occupancy.any()

    coords = (torch.arange(48) + 0.5) * spec.cell_size_m - spec.grid_size_m / 2.0
    center_row = torch.argmin(coords.abs())
    free_col = torch.argmin((coords - 1.0).abs())
    hidden_col = torch.argmin((coords - 4.0).abs())
    assert observed[center_row, free_col] == 1
    assert occupancy[center_row, free_col] == 0
    assert observed[center_row, hidden_col] == 0


def test_inflation_is_applied_once_and_expands_occupancy():
    ranges = torch.full((1, 220), 20.0)
    ranges[:, 105:115] = 2.0
    no_inflation = _spec(0.0)
    inflated = _spec(0.35)
    raw = lidar_ranges_to_costmap(ranges, _angles(no_inflation), no_inflation)
    expanded = lidar_ranges_to_costmap(ranges, _angles(inflated), inflated)
    assert expanded[:, 0].sum() > raw[:, 0].sum()


def test_invalid_beam_remains_unknown_instead_of_free():
    spec = _spec()
    ranges = torch.full((1, 220), spec.lidar_max_range_m)
    ranges[:, 100:120] = float("nan")
    costmap = lidar_ranges_to_costmap(ranges, _angles(spec), spec)
    coords = (torch.arange(48) + 0.5) * spec.cell_size_m - spec.grid_size_m / 2.0
    center_row = torch.argmin(coords.abs())
    front_col = torch.argmin((coords - 2.0).abs())
    assert costmap[0, 1, center_row, front_col] == 0


if __name__ == "__main__":
    tests = [
        test_shape_and_front_only_observed_mask,
        test_first_return_marks_endpoint_and_occludes_cells_behind_it,
        test_inflation_is_applied_once_and_expands_occupancy,
        test_invalid_beam_remains_unknown_instead_of_free,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
