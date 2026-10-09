# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teleop_hand_and_arm as teleop_app


class Dex1ServiceEnvironmentTest(unittest.TestCase):
    def test_prepends_sibling_library_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin" / "dex1_1_gripper_server"
            binary.parent.mkdir()
            binary.touch()
            library_dir = root / "lib"
            library_dir.mkdir()

            env = teleop_app.dex1_service_environment(
                binary,
                {"LD_LIBRARY_PATH": f"/existing{os.pathsep}{library_dir}"},
            )

            self.assertEqual(
                env["LD_LIBRARY_PATH"].split(os.pathsep),
                [str(library_dir), "/existing"],
            )

    def test_leaves_environment_unchanged_without_sibling_library_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "bin" / "custom_server"
            binary.parent.mkdir()
            binary.touch()
            original = {"LD_LIBRARY_PATH": "/existing", "OTHER": "value"}

            self.assertEqual(
                teleop_app.dex1_service_environment(binary, original), original
            )

    def test_start_passes_sibling_library_directory_to_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "bin" / "dex1_1_gripper_server"
            binary.parent.mkdir()
            binary.touch(mode=0o755)
            library_dir = root / "lib"
            library_dir.mkdir()

            with (
                mock.patch.object(teleop_app.subprocess, "Popen") as popen,
                mock.patch.object(teleop_app.time, "sleep"),
            ):
                popen.return_value.poll.return_value = None
                teleop_app.start_dex1_service(binary, "test0")

            child_env = popen.call_args.kwargs["env"]
            self.assertEqual(
                child_env["LD_LIBRARY_PATH"].split(os.pathsep)[0],
                str(library_dir),
            )


class Dex1ForceControlArgumentsTest(unittest.TestCase):
    @staticmethod
    def parse(*arguments):
        parser = teleop_app.build_arg_parser()
        args = parser.parse_args(list(arguments))
        return teleop_app.validate_args(parser, args)

    def test_uses_unitree_stiffness_and_relaxed_step_defaults(self):
        args = self.parse("--ee", "dex1")
        self.assertEqual(args.dex1_kp, 5.0)
        self.assertEqual(args.dex1_kd, 0.05)
        self.assertEqual(args.dex1_command_step, 0.36)
        self.assertIsNone(args.dex1_soft_torque_limit)
        self.assertIsNone(args.dex1_hard_torque_limit)

    def test_accepts_paired_torque_limits(self):
        args = self.parse(
            "--ee", "dex1",
            "--dex1-soft-torque-limit", "1.0",
            "--dex1-hard-torque-limit", "1.5",
        )
        self.assertEqual(args.dex1_soft_torque_limit, 1.0)
        self.assertEqual(args.dex1_hard_torque_limit, 1.5)

    def test_rejects_missing_or_reversed_torque_limit(self):
        with self.assertRaises(SystemExit):
            self.parse("--ee", "dex1", "--dex1-soft-torque-limit", "1.0")
        with self.assertRaises(SystemExit):
            self.parse(
                "--ee", "dex1",
                "--dex1-soft-torque-limit", "2.0",
                "--dex1-hard-torque-limit", "1.0",
            )

    def test_rejects_external_torque_limits_for_internal_dex1(self):
        with self.assertRaises(SystemExit):
            self.parse(
                "--ee", "dex1_internal",
                "--dex1-soft-torque-limit", "1.0",
                "--dex1-hard-torque-limit", "1.5",
            )


if __name__ == "__main__":
    unittest.main()
