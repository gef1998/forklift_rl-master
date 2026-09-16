"""Pure helpers shared by the skrl simulation evaluation entry point and tests."""

from __future__ import annotations

from decimal import Decimal
import math
from statistics import fmean
from typing import Any, Iterable


TERMINATION_REASONS = {
    0: "running",
    1: "timeout",
    2: "fell",
    3: "drifted",
    4: "collision",
    5: "goal_reached",
}


def width_sequence(width_min: float, width_max: float, width_step: float) -> list[float]:
    """Build an inclusive, descending decimal width scan without float drift."""

    low = Decimal(str(width_min))
    high = Decimal(str(width_max))
    step = Decimal(str(width_step))
    if not all(value.is_finite() for value in (low, high, step)):
        raise ValueError("width values must be finite")
    if low <= 0 or high <= 0 or step <= 0:
        raise ValueError("width values and step must be positive")
    if low > high:
        raise ValueError("width_min must not exceed width_max")
    widths: list[float] = []
    current = high
    while current >= low:
        widths.append(float(current))
        current -= step
    if Decimal(str(widths[-1])) != low:
        widths.append(float(low))
    return widths


def longest_success_streak(results: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for result in results:
        current = current + 1 if result else 0
        longest = max(longest, current)
    return longest


def width_passed(results: Iterable[bool], required_consecutive: int = 3) -> bool:
    if required_consecutive <= 0:
        raise ValueError("required_consecutive must be positive")
    return longest_success_streak(results) >= required_consecutive


def minimum_passing_width(width_rows: Iterable[dict[str, Any]]) -> float | None:
    passed = [float(row["passage_width_m"]) for row in width_rows if row["passed"]]
    return min(passed) if passed else None


def termination_reason_name(code: int) -> str:
    return TERMINATION_REASONS.get(int(code), "unknown")


def straight_returned_to_centerline(
    samples: Iterable[tuple[float, float]],
    *,
    travel_direction: float,
    obstacle_half_length: float,
    robot_half_length: float,
    tolerance: float,
) -> tuple[bool, float | None]:
    """Check whether the path re-enters the centerline band after the obstacle."""

    if travel_direction == 0.0:
        raise ValueError("travel_direction must be non-zero")
    if min(obstacle_half_length, robot_half_length, tolerance) < 0.0:
        raise ValueError("straight return dimensions must be non-negative")
    threshold = obstacle_half_length + robot_half_length
    post_errors = [
        abs(float(y))
        for x, y in samples
        if travel_direction * float(x) >= threshold
    ]
    if not post_errors:
        return False, None
    final_error = post_errors[-1]
    return final_error <= tolerance, final_error


def summarize_width(
    scenario: str,
    passage_width_m: float,
    episodes: list[dict[str, Any]],
    required_consecutive: int = 3,
) -> dict[str, Any]:
    """Aggregate one tested width while retaining the ordered trial outcomes."""

    outcomes = [bool(row["success"]) for row in episodes]
    reasons: dict[str, int] = {}
    for row in episodes:
        reason = str(row["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in episodes if row.get(field) is not None]
        return fmean(values) if values else None

    finite_clearances = [
        float(row["minimum_clearance_m"])
        for row in episodes
        if row.get("minimum_clearance_m") is not None
        and math.isfinite(float(row["minimum_clearance_m"]))
    ]
    return {
        "scenario": scenario,
        "passage_width_m": float(passage_width_m),
        "trials": len(episodes),
        "trial_successes": outcomes,
        "success_count": sum(outcomes),
        "success_rate": sum(outcomes) / len(outcomes) if outcomes else 0.0,
        "longest_consecutive_successes": longest_success_streak(outcomes),
        "required_consecutive_successes": required_consecutive,
        "passed": width_passed(outcomes, required_consecutive),
        "termination_counts": reasons,
        "mean_elapsed_time_s": mean("elapsed_time_s"),
        "mean_path_length_m": mean("path_length_m"),
        "mean_forward_speed_mps": mean("mean_forward_speed_mps"),
        "minimum_clearance_m": min(finite_clearances) if finite_clearances else None,
    }
