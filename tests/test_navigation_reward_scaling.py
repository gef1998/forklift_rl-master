"""Behavioral reward regression checks; runnable without Isaac Sim or pytest."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

MODULE_PATH = (Path(__file__).resolve().parents[1]
               / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/navigation_rewards.py")
SPEC = importlib.util.spec_from_file_location("navigation_rewards", MODULE_PATH)
rewards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rewards)


def terms(previous, current, *, clearance=1.0, dt=.02, failure=False, goal=False,
          action=(0.0, 0.0), previous_action=(0.0, 0.0), has_previous=True):
    previous = torch.atleast_1d(torch.as_tensor(previous, dtype=torch.float64))
    current = torch.as_tensor(current, dtype=torch.float64).expand_as(previous)
    n = previous.numel()
    def vector(value, dtype):
        return torch.as_tensor(value, dtype=dtype).expand(n)
    return rewards.navigation_reward_terms(
        previous, current, vector(clearance, torch.float64),
        torch.tensor(action, dtype=torch.float64).expand(n, 2),
        torch.tensor(previous_action, dtype=torch.float64).expand(n, 2),
        vector(has_previous, torch.bool), vector(goal, torch.bool), vector(failure, torch.bool),
        step_dt=dt, progress_per_meter=5.0, clearance_threshold=.35,
        collision_clearance=.05, proximity_cost_per_second=.25,
        time_cost_per_second=.10, action_change_scale=.002,
        goal_bonus=25.0, failure_penalty=-30.0,
    )


class NavigationRewardTests(unittest.TestCase):
    def test_progress_rewards_approach_and_charges_retreat_symmetrically(self):
        r = terms([5., 4.99, 5.], [4.99, 5., 5.])["progress"]
        torch.testing.assert_close(r, torch.tensor([.05, -.05, 0.], dtype=torch.float64))

    def test_closed_distance_cycles_cannot_collect_positive_only_progress(self):
        distances = torch.tensor([5., 4., 4.5, 3., 5.], dtype=torch.float64)
        r = terms(distances[:-1], distances[1:])
        self.assertAlmostEqual(r["progress"].sum().item(), 0.0)
        self.assertLess(sum(r.values()).sum().item(), 0.0)

    def test_equal_endpoints_have_equal_undiscounted_progress(self):
        direct = terms([5., 4.], [4., 3.])["progress"].sum()
        detour = terms([5., 6., 7., 6.], [6., 7., 6., 3.])["progress"].sum()
        torch.testing.assert_close(direct, detour)

    def test_stop_and_spin_have_no_motion_bonus(self):
        stopped = sum(terms(5., 5.).values())
        # Once a constant angular command is established it has no smoothness cost.
        spinning = sum(terms(5., 5., action=(0., 1.), previous_action=(0., 1.)).values())
        torch.testing.assert_close(stopped, spinning)
        self.assertLess(spinning.item(), 0)

    def test_reverse_motion_toward_goal_is_not_penalized_for_its_sign(self):
        forward = sum(terms(5., 4.99, action=(1., 0.), previous_action=(1., 0.)).values())
        reverse = sum(terms(5., 4.99, action=(-1., 0.), previous_action=(-1., 0.)).values())
        torch.testing.assert_close(forward, reverse)
        self.assertGreater(reverse.item(), 0)

    def test_running_reward_units_are_consistent_across_control_rates(self):
        totals = []
        for steps in (25, 50, 100):
            distances = torch.linspace(5., 4.5, steps + 1, dtype=torch.float64)
            totals.append(sum(terms(distances[:-1], distances[1:], clearance=.20,
                                   dt=1./steps).values()).sum())
        for total in totals[1:]:
            torch.testing.assert_close(total, totals[0])

    def test_proximity_is_bounded_and_zero_outside_near_collision_band(self):
        r = terms([5.] * 6, [5.] * 6,
                  clearance=[10., .5, .35, .20, .05, -.5])["proximity"]
        self.assertTrue((r[:3] == 0).all())
        self.assertTrue((r[3:] < 0).all())
        self.assertTrue((r >= -.25 * .02).all())
        self.assertEqual(r[-1], r[-2])

    def test_safe_corridor_does_not_receive_the_old_long_range_repulsion(self):
        self.assertEqual(terms(5., 4.99, clearance=.50)["proximity"].item(), 0)
        self.assertGreater(sum(terms(5., 4.99, clearance=.50).values()).item(), 0)

    def test_first_action_is_not_compared_to_an_artificial_reset_zero(self):
        first = terms(5., 5., action=(1., -1.), has_previous=False)
        next_step = terms(5., 5., action=(1., -1.), has_previous=True)
        self.assertEqual(first["action_change"].item(), 0)
        self.assertLess(next_step["action_change"].item(), 0)

    def test_action_smoothness_is_weak_and_independent_of_action_sign(self):
        r = terms(5., 4.99, action=(1., 1.), previous_action=(-1., -1.))
        self.assertAlmostEqual(r["action_change"].item(), -.008)
        self.assertGreater(sum(r.values()).item(), 0)

    def test_failure_beats_success_and_suppresses_progress_on_failure_step(self):
        r = terms([5., 5., 5.], [0., 0., 0.], failure=[True, True, False], goal=True)
        self.assertTrue((r["goal"][:2] == 0).all())
        self.assertTrue((r["progress"][:2] == 0).all())
        self.assertTrue((r["failure"][:2] == -30).all())
        self.assertEqual(r["goal"][2], 25)

    def test_invalid_failure_pose_does_not_poison_reward(self):
        r = terms(5., float("nan"), clearance=float("nan"), failure=True)
        self.assertTrue(torch.isfinite(sum(r.values())).all())
        self.assertLess(sum(r.values()).item(), -30)

    def test_near_obstacle_waiting_for_full_episode_is_cheaper_than_suicide(self):
        stationary_return = 3000 * sum(terms(5., 5., clearance=.05).values()).item()
        failure_return = sum(terms(5., 5., clearance=.05, failure=True).values()).item()
        self.assertGreater(stationary_return, failure_return)
        self.assertLess(stationary_return, 0)

    def test_goal_terminal_credit_is_larger_than_one_second_of_free_progress(self):
        walking = 50 * sum(terms(5., 4.99).values()).item()
        reached = sum(terms(.31, .29, goal=True).values()).item()
        self.assertGreater(reached, walking)


if __name__ == "__main__":
    unittest.main()
