from __future__ import annotations

import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = SCRIPT_DIR / "train-pi05-unitree-legacy.sh"


class LegacyTrainingLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    def test_uses_unitree_environment_and_vendored_source(self):
        self.assertIn("envs/unitree_lerobot/bin/lerobot-train", self.launcher)
        self.assertIn("unitree_lerobot/lerobot/src", self.launcher)
        self.assertIn('PYTHONPATH="$OLD_SOURCE:$PACKAGE_ROOT"', self.launcher)
        self.assertIn('getattr(lerobot, "__version__", None) != "0.4.1"', self.launcher)

    def test_defaults_to_physical_gpu_one_and_bf16(self):
        self.assertIn('R1A7_LEGACY_CUDA_DEVICE:-1', self.launcher)
        self.assertIn('CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"', self.launcher)
        self.assertIn("ACCELERATE_MIXED_PRECISION=bf16", self.launcher)
        self.assertIn("--policy.dtype=bfloat16", self.launcher)
        self.assertNotIn("torchrun", self.launcher)

    def test_fixes_r1a7_pi05_training_contract(self):
        required = (
            "load_dataset_info(dataset_root)",
            "--dataset.video_backend=pyav",
            "--dataset.use_imagenet_stats=false",
            "--policy.type=pi05",
            "--policy.n_action_steps=10",
            "--policy.gradient_checkpointing=true",
            "--policy.push_to_hub=false",
            "--save_checkpoint=true",
            "R1A7_PI05_SAVE_FREQ:-4000",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, self.launcher)

    def test_never_imports_or_authorizes_hardware_control(self):
        forbidden = (
            "unitree_sdk2py",
            "cyclonedds",
            "LEROBOT_R1A7_ALLOW_REAL",
            "load_hardened_controller_types",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, self.launcher)

    def test_supports_smoke_production_resume_and_dry_run(self):
        for token in ("smoke", "production", "resume", "--checkpoint", "--dry-run"):
            with self.subTest(token=token):
                self.assertIn(token, self.launcher)


if __name__ == "__main__":
    unittest.main()
