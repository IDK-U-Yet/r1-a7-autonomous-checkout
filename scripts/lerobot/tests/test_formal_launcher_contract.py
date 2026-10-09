from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
FORMAL_LAUNCHER = SCRIPT_DIR / "train-pi05-4x4090-formal.sh"


class FormalLauncherContractTest(unittest.TestCase):
    """Static contract checks: the launcher runs on the training machine, not here."""

    def test_launcher_exists_and_is_executable(self):
        self.assertTrue(FORMAL_LAUNCHER.is_file())
        self.assertTrue(FORMAL_LAUNCHER.stat().st_mode & 0o111)

    def test_bash_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(FORMAL_LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pins_lerobot_commit_and_split_sha(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('262d2fd07e32224d46dca8ba8bcdef5a0b3a9c3b', text)
        self.assertIn(
            "53c12e7768cf53db0201641ba77147590dcb546a9130baf7d15b5fc87efe487b",
            text,
        )
        self.assertIn(
            "589182c4a4aa4ab0236417fea30c8c232570bf4a13a90e5e35f0503809c946b7",
            text,
        )

    def test_requires_all_four_patches_reverse_checked(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        for patch in (
            "262d2fd-pi05-fsdp2-forward-hooks.patch",
            "262d2fd-fsdp2-lazy-optimizer-resume.patch",
            "262d2fd-training-filter.patch",
            "262d2fd-episode-split.patch",
            "262d2fd-episode-eval-filter.patch",
            "262d2fd-checkpoint-retention.patch",
        ):
            self.assertIn(patch, text)
        self.assertGreaterEqual(text.count("apply --reverse --check"), 1)

    def test_consumes_episode_split_and_val_loss(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--dataset.episode_split_path=", text)
        self.assertIn("--dataset.episode_split_name=train", text)
        self.assertIn("--dataset.episode_eval_split_name=val", text)
        self.assertIn("--eval_steps=", text)

    def test_explicitly_records_pi05_scheduler_contract(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        for option in (
            "--policy.optimizer_lr=",
            "--policy.scheduler_warmup_steps=",
            "--policy.scheduler_decay_steps=",
            "--policy.scheduler_decay_lr=",
        ):
            self.assertIn(option, text)
        self.assertIn("--peak-lr", text)
        self.assertIn("--warmup-steps", text)
        self.assertIn("STEPS=24000", text)
        self.assertIn("DECAY_STEPS=24000", text)

    def test_validation_filter_patch_is_required(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("262d2fd-episode-eval-filter.patch", text)

    def test_checkpoint_retention_is_enabled(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("262d2fd-checkpoint-retention.patch", text)
        self.assertIn("--checkpoint_keep_last=", text)
        self.assertIn("--checkpoint_keep_steps=", text)
        self.assertIn("CHECKPOINT_KEEP_LAST=3", text)

    def test_keeps_verified_fsdp2_topology_and_memory_contract(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--parallelism.dp_shard=4", text)
        self.assertIn("--accelerator.fsdp.reshard_after_forward=true", text)
        self.assertIn('"SiglipEncoderLayer","Linear","GemmaTextScaledWordEmbedding"', text)
        self.assertIn("--policy.gradient_checkpointing=true", text)
        self.assertIn("--checkpoint_format=dcp", text)

    def test_has_disk_and_gpu_preflights(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("min-free-gb", text)
        self.assertIn("skip-gpu-idle-check", text)
        self.assertIn("df -Pk", text)

    def test_resume_maps_to_config_path_flag(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("--resume=true", text)
        self.assertIn("--config_path=", text)
        self.assertIn("Resume contract mismatch", text)
        self.assertIn("accelerator.gradient_accumulation.steps", text)


if __name__ == "__main__":
    unittest.main()
