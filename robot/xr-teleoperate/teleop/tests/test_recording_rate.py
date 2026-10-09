# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teleop_hand_and_arm as teleop_app


class PeriodicRateGateTest(unittest.TestCase):
    def test_records_at_15_hz_inside_a_30_hz_loop(self):
        gate = teleop_app.PeriodicRateGate(15.0, 30.0)
        loop_period_ns = round(1_000_000_000 / 30.0)

        selected = [
            index
            for index in range(31)
            if gate.due(index * loop_period_ns)
        ]

        self.assertEqual(selected, list(range(0, 31, 2)))

    def test_reset_samples_immediately(self):
        gate = teleop_app.PeriodicRateGate(15.0)
        self.assertTrue(gate.due(1_000_000_000))
        self.assertFalse(gate.due(1_010_000_000))
        gate.reset()
        self.assertTrue(gate.due(1_010_000_000))


class RecordingFrequencyArgumentsTest(unittest.TestCase):
    @staticmethod
    def parse(*arguments):
        parser = teleop_app.build_arg_parser()
        args = parser.parse_args(list(arguments))
        return teleop_app.validate_args(parser, args)

    def test_accepts_recording_slower_than_control(self):
        args = self.parse("--frequency", "30", "--record-frequency", "15")
        self.assertEqual(args.frequency, 30.0)
        self.assertEqual(args.record_frequency, 15.0)

    def test_rejects_recording_faster_than_control(self):
        with self.assertRaises(SystemExit):
            self.parse("--frequency", "15", "--record-frequency", "30")

    def test_r1_waist_control_is_disabled_by_default(self):
        args = self.parse("--arm", "R1_A7", "--sim")
        self.assertFalse(args.enable_r1_waist_control)

    def test_r1_waist_control_requires_explicit_opt_in(self):
        with self.assertRaises(SystemExit):
            self.parse(
                "--arm",
                "R1_A7",
                "--sim",
                "--r1-waist-velocity-limit",
                "1.0",
            )
        args = self.parse(
            "--arm",
            "R1_A7",
            "--sim",
            "--enable-r1-waist-control",
            "--r1-waist-velocity-limit",
            "1.0",
        )
        self.assertTrue(args.enable_r1_waist_control)


if __name__ == "__main__":
    unittest.main()
