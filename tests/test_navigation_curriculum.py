"""Outcome-based curriculum regression tests; no simulator required."""
import importlib.util
from pathlib import Path
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / "source/emma_rl/emma_rl/tasks/direct/emma/emma_obs_nav/navigation_curriculum.py"
spec = importlib.util.spec_from_file_location("navigation_curriculum", MODULE)
course = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = course
spec.loader.exec_module(course)


class CurriculumTests(unittest.TestCase):
    def make(self, **kwargs):
        options = dict(window_size=10, required_windows=2, min_level_steps=0)
        options.update(kwargs)
        return course.NavigationCurriculum(course.NavigationCurriculumCfg(**options))

    def outcomes(self, c, success=10, failure=0, **kwargs):
        n = kwargs.pop("count", 10)
        return c.record([kwargs.get("level", c.level)] * n,
                        [kwargs.get("generation", c.generation)] * n,
                        [i < success for i in range(n)],
                        [success <= i < success + failure for i in range(n)])

    def test_two_independent_windows_and_no_clock_promotion(self):
        c = self.make()
        for _ in range(10000):
            c.advance()
        self.assertEqual(c.level, 0)
        self.assertFalse(self.outcomes(c, 8, 1))
        self.assertEqual(c.promote_streak, 1)
        self.assertFalse(c.record([], [], [], []))
        self.assertTrue(self.outcomes(c, 8, 1))
        self.assertEqual((c.level, c.generation, c.steps_since_change), (1, 1, 0))

    def test_replay_old_generation_and_cooldown_excluded(self):
        c = self.make(initial_level=2, min_level_steps=3)
        self.outcomes(c)
        self.assertEqual(len(c.window), 0)
        for _ in range(3):
            c.advance()
        self.outcomes(c, level=1)
        self.outcomes(c, generation=99)
        self.assertEqual(len(c.window), 0)
        self.outcomes(c)
        self.assertTrue(self.outcomes(c))
        self.assertEqual(c.level, 3)
        self.outcomes(c, generation=0)
        self.assertEqual(len(c.window), 0)

    def test_demotion_and_neutral_window_reset_streak(self):
        c = self.make(initial_level=2)
        self.outcomes(c, 3)
        self.outcomes(c, 6, 2)
        self.assertEqual(c.demote_streak, 0)
        self.outcomes(c, 5, 4)
        self.assertTrue(self.outcomes(c, 5, 4))
        self.assertEqual(c.level, 1)

    def test_level_bounds_and_failure_over_success(self):
        c = self.make(initial_level=5)
        for _ in range(3):
            self.outcomes(c)
        self.assertEqual(c.level, 5)
        c.record([5]*20, [0]*20, [True]*20, [True]*20)
        self.assertEqual(c.level, 4)
        self.assertEqual(c.last_success_rate, 0)
        low = self.make()
        for _ in range(3):
            self.outcomes(low, 0, 10)
        self.assertEqual(low.level, 0)

    def test_transition_discards_remaining_old_batch(self):
        c = self.make()
        self.outcomes(c, count=40, success=40)
        self.assertEqual((c.level, c.generation, len(c.window)), (1, 1, 0))

    def test_checkpoint_restores_partial_window_and_streak(self):
        c = self.make(initial_level=2)
        self.outcomes(c)
        self.outcomes(c, count=4, success=4)
        d = self.make()
        d.load_state_dict(c.state_dict())
        self.assertEqual(d.state_dict(), c.state_dict())
        self.assertTrue(self.outcomes(d, count=6, success=6))
        self.assertEqual(d.level, 3)
        changed = self.make(window_size=20)
        changed.load_state_dict(c.state_dict())
        self.assertEqual((len(changed.window), changed.promote_streak), (0, 0))

    def test_evaluation_is_fixed_even_after_loading_training_state(self):
        c = self.make(enabled=False)
        c.load_state_dict(self.make(initial_level=1).state_dict())
        c.advance()
        self.outcomes(c, 0, 10)
        self.assertEqual((c.level, c.steps_since_change), (5, 0))
        unsupported = course.NavigationCurriculum(course.NavigationCurriculumCfg(), supported=False)
        self.assertFalse(unsupported.enabled)

    def test_invalid_configuration_and_state(self):
        for options in (dict(window_size=0), dict(required_windows=0), dict(min_level_steps=-1),
                        dict(initial_level=6), dict(replay_probability=1),
                        dict(promote_failure_rate=.5), dict(evaluation_noise_scale=float('nan')),
                        dict(noise_ranges=((0, 1),))):
            with self.assertRaises(ValueError):
                self.make(**options)
        c = self.make()
        with self.assertRaises(ValueError):
            c.record([0], [], [], [])
        for change in (dict(level=9), dict(window=[(True,)]), dict(generation=-1)):
            state = c.state_dict()
            state.update(change)
            with self.assertRaises(ValueError):
                c.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
