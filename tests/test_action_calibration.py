from __future__ import print_function
import imp
import math
import os
import tempfile
import unittest

core = imp.load_source("emma_action_calibration", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "emma_action_calibration.py"))


class ActionCalibrationTest(unittest.TestCase):
    def test_profile_endpoints_order_and_acceleration(self):
        dt, acceleration = .02, .75
        samples = core.generate_profile(acceleration, dt, .1)
        values = [item[0] for item in samples]
        self.assertEqual(values[0], 0.)
        self.assertAlmostEqual(max(values), 1.)
        self.assertAlmostEqual(min(values), -1.)
        self.assertEqual(values[-1], 0.)
        positive = values.index(max(values))
        negative = values.index(min(values))
        self.assertLess(positive, negative)
        discrete_peak = max(abs(values[i] - values[i-1]) / dt
                            for i in range(1, len(values)))
        self.assertLessEqual(discrete_peak, acceleration * 1.002)

    def test_physical_acceleration_is_converted_to_action_rate(self):
        runs = core.build_runs([.33], [.75], .02, .1, .66, 1.5)
        self.assertAlmostEqual(runs[0]["action_acceleration_per_s"], .5)
        self.assertAlmostEqual(runs[1]["action_acceleration_per_s"], .5)

    def test_body_twist_and_wheel_kinematics(self):
        straight = core.action_to_motion(1., 0., .66, 1.5, .05, .43, 20.)
        self.assertAlmostEqual(straight["vx_mps"], .66)
        self.assertAlmostEqual(straight["left_radps"], 13.2)
        self.assertAlmostEqual(straight["right_radps"], 13.2)
        spin = core.action_to_motion(0., 1., .66, 1.5, .05, .43, 20.)
        self.assertAlmostEqual(spin["wz_radps"], 1.5)
        self.assertAlmostEqual(spin["left_radps"], -6.45)
        self.assertAlmostEqual(spin["right_radps"], 6.45)

    def test_lag_estimator(self):
        try:
            __import__("numpy")
        except ImportError:
            self.skipTest("numpy is unavailable")
        run = {"run_id": "x", "axis": "linear", "acceleration": 1.}
        command, feedback = [], []
        for i in range(300):
            t = i * .02
            value = math.sin(t * 2.)
            command.append(core.stream_row(run, i, "", t, {"vx_mps": value}))
            feedback.append(core.stream_row(run, i, "", t + .12, {"vx_mps": value}))
        result = core.estimate_lag(command, feedback, "vx_mps", .4)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["lag_ms"], 120., delta=6.)


if __name__ == "__main__":
    unittest.main()
