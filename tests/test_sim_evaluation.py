#!/usr/bin/env python3
"""Pure Python tests for simulation-evaluation aggregation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "skrl" / "evaluation_utils.py"
SPEC = importlib.util.spec_from_file_location("evaluation_utils", MODULE_PATH)
evaluation_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation_utils)


def test_width_sequence_is_descending_inclusive_and_decimal_exact():
    assert evaluation_utils.width_sequence(0.7, 1.0, 0.1) == [1.0, 0.9, 0.8, 0.7]
    assert evaluation_utils.width_sequence(0.75, 1.0, 0.1) == [1.0, 0.9, 0.8, 0.75]


def test_three_consecutive_successes_are_required():
    assert not evaluation_utils.width_passed([True, True, False, True], 3)
    assert evaluation_utils.width_passed([False, True, True, True], 3)
    assert evaluation_utils.longest_success_streak([True, True, False, True]) == 2


def test_minimum_width_uses_only_tested_passing_rows():
    rows = [
        {"passage_width_m": 1.5, "passed": True},
        {"passage_width_m": 1.4, "passed": False},
        {"passage_width_m": 1.3, "passed": True},
    ]
    assert evaluation_utils.minimum_passing_width(rows) == 1.3
    assert evaluation_utils.minimum_passing_width([]) is None


def test_termination_reason_mapping_is_stable():
    expected = {
        1: "timeout",
        2: "fell",
        3: "drifted",
        4: "collision",
        5: "goal_reached",
    }
    assert {
        code: evaluation_utils.termination_reason_name(code) for code in expected
    } == expected
    assert evaluation_utils.termination_reason_name(99) == "unknown"


def test_straight_return_requires_final_post_obstacle_centerline_entry():
    returned, final_error = evaluation_utils.straight_returned_to_centerline(
        [(-2.0, 0.0), (0.0, 0.8), (1.0, 0.4), (2.0, 0.2)],
        travel_direction=1.0,
        obstacle_half_length=0.5,
        robot_half_length=0.42,
        tolerance=0.3,
    )
    assert returned
    assert final_error == 0.2
    returned, final_error = evaluation_utils.straight_returned_to_centerline(
        [(-2.0, 0.0), (1.0, 0.2), (2.0, 0.4)],
        travel_direction=1.0,
        obstacle_half_length=0.5,
        robot_half_length=0.42,
        tolerance=0.3,
    )
    assert not returned and final_error == 0.4
    returned, final_error = evaluation_utils.straight_returned_to_centerline(
        [(-2.0, 0.0), (0.0, 0.8), (0.8, 0.4)],
        travel_direction=1.0,
        obstacle_half_length=0.5,
        robot_half_length=0.42,
        tolerance=0.3,
    )
    assert not returned and final_error is None


def test_width_summary_is_json_serializable():
    episodes = [
        {
            "success": value,
            "termination_reason": "goal_reached" if value else "collision",
            "elapsed_time_s": 10.0,
            "path_length_m": 5.0,
            "mean_forward_speed_mps": 0.5,
            "minimum_clearance_m": 0.1,
        }
        for value in (True, True, True)
    ]
    summary = evaluation_utils.summarize_width("ring", 1.4, episodes, 3)
    assert summary["passed"]
    assert summary["longest_consecutive_successes"] == 3
    json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
