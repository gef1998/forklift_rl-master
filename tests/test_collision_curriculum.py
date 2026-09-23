"""Tests for curriculum-dependent collision tolerance without Isaac Sim."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "emma_rl"
    / "emma_rl"
    / "tasks"
    / "direct"
    / "emma"
    / "emma_obs_nav"
    / "collision_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("collision_curriculum", MODULE_PATH)
collision_curriculum = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collision_curriculum)


def _step(
    clearance: float,
    active: bool,
    duration: float,
    events: int,
    *,
    terminate_immediately: bool = False,
    budget: int = 3,
    step_dt: float = 0.02,
):
    return collision_curriculum.collision_curriculum_step(
        torch.tensor([clearance]),
        torch.tensor([active]),
        torch.tensor([duration]),
        torch.tensor([events], dtype=torch.long),
        torch.tensor([terminate_immediately]),
        torch.tensor([budget], dtype=torch.long),
        step_dt=step_dt,
        collision_clearance=0.01,
        release_clearance=0.03,
        severe_clearance=-0.05,
        max_continuous_collision_time_s=0.75,
    )


def test_level_zero_allows_three_events_and_terminates_on_the_fourth():
    active, duration, events = False, 0.0, 0
    for expected_count in range(1, 5):
        result = _step(0.0, active, duration, events, budget=3)
        assert result["collision_event"].item()
        assert result["episode_event_count"].item() == expected_count
        assert result["terminal_collision"].item() == (expected_count == 4)
        active = result["contact_active"].item()
        duration = result["contact_duration_s"].item()
        events = result["episode_event_count"].item()

        released = _step(0.04, active, duration, events, budget=3)
        active = released["contact_active"].item()
        duration = released["contact_duration_s"].item()
        events = released["episode_event_count"].item()
        assert not active


def test_high_level_terminates_on_first_mild_collision():
    result = _step(
        0.0,
        False,
        0.0,
        0,
        terminate_immediately=True,
        budget=0,
    )
    assert result["collision_event"].item()
    assert result["terminal_collision"].item()
    assert not result["nonterminal_collision_event"].item()


def test_hysteresis_does_not_recount_one_flickering_contact():
    first = _step(0.0, False, 0.0, 0)
    near = _step(
        0.02,
        first["contact_active"].item(),
        first["contact_duration_s"].item(),
        first["episode_event_count"].item(),
    )
    second_sample = _step(
        0.0,
        near["contact_active"].item(),
        near["contact_duration_s"].item(),
        near["episode_event_count"].item(),
    )
    assert not near["collision"].item()
    assert near["contact_active"].item()
    assert not second_sample["collision_event"].item()
    assert second_sample["episode_event_count"].item() == 1


def test_severe_or_sustained_collision_is_always_terminal():
    severe = _step(-0.06, False, 0.0, 0, budget=99)
    sustained = _step(
        0.0,
        True,
        0.74,
        1,
        budget=99,
        step_dt=0.02,
    )
    assert severe["severe_collision"].item()
    assert severe["terminal_collision"].item()
    assert sustained["sustained_collision"].item()
    assert sustained["terminal_collision"].item()
