from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CHECK_PATH = SCRIPT_DIR / "check_r1a7_setup.py"
TRAIN_PATH = SCRIPT_DIR / "train-pi05-full.sh"


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_r1a7_setup", CHECK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CHECK_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load_check_module()


class TrainingOnlyPreflightTest(unittest.TestCase):
    def test_training_probe_has_no_control_or_cuda_runtime_queries(self):
        forbidden = (
            "torch.cuda",
            "lerobot_robot_unitree_r1a7",
            "unitree_sdk2py",
            "cyclonedds",
            "cv2",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, check.TRAINING_ENV_PROBE)

    def test_skip_control_alias_selects_training_only_mode(self):
        args = check.parse_args(["--skip-control"])
        self.assertTrue(args.training_only)

    def test_training_only_rejects_control_requirements(self):
        for option in ("--require-legacy-pi", "--require-plugin", "--require-hardware"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                check.parse_args(["--training-only", option])

    def test_training_only_main_never_calls_control_or_gpu_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            conda_root = Path(temp_dir) / "conda"
            new_python = conda_root / "envs/lerobot/bin/python"
            new_python.parent.mkdir(parents=True)
            new_python.touch()

            environment = {
                "python": "3.12.0",
                "packages": {
                    "lerobot": "0.6.2",
                    "transformers": "5.4.0",
                    "draccus": "0.11.6",
                },
                "lerobot_module_version": "0.6.2",
                "lerobot_file": "/new/source/lerobot/__init__.py",
                "pi05_import": "PI05Config",
            }

            with (
                mock.patch.object(check, "_check_environment", return_value=environment) as check_env,
                mock.patch.object(check, "_check_dataset") as check_dataset,
                mock.patch.object(check, "_check_model") as check_model,
                mock.patch.object(check, "_find_plugin_root") as find_plugin,
                mock.patch.object(check, "_check_plugin") as check_plugin,
            ):
                result = check.main(
                    [
                        "--training-only",
                        "--conda-root",
                        str(conda_root),
                        "--new-source",
                        "/new/source",
                        "--source-dataset-root",
                        "/data/source",
                        "--dataset-root",
                        "/data/training",
                        "--base-checkpoint",
                        "/models/pi05",
                        "--tokenizer-root",
                        "/models/tokenizer",
                    ]
                )

            self.assertEqual(result, 0)
            check_env.assert_called_once()
            _, env_kwargs = check_env.call_args
            self.assertIs(env_kwargs["probe"], check.TRAINING_ENV_PROBE)
            self.assertFalse(env_kwargs["check_control_dependencies"])
            check_dataset.assert_called_once()
            self.assertEqual(check_dataset.call_args.args[3], "dataset.training")
            check_model.assert_called_once()
            self.assertEqual(check_model.call_args.kwargs["config_device"], "cpu")
            find_plugin.assert_not_called()
            check_plugin.assert_not_called()

    def test_training_launcher_uses_training_only_mode_and_no_torch_cuda_probe(self):
        launcher = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('"$ROOT_DIR/scripts/lerobot/check_r1a7_setup.py" \\\n  --training-only', launcher)
        self.assertNotIn("torch.cuda.device_count", launcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=''", launcher)


if __name__ == "__main__":
    unittest.main()
