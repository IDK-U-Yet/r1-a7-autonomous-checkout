from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PRUNE_PATH = SCRIPT_DIR / "prune_fsdp_checkpoints.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prune_fsdp_checkpoints", PRUNE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PRUNE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prune = _load_module()


def _complete_checkpoint(checkpoints: Path, step: int) -> Path:
    path = checkpoints / f"{step:06d}"
    optimizer_dir = path / "training_state" / "optimizer_0"
    optimizer_dir.mkdir(parents=True)
    (path / "training_state" / "training_step.json").write_text("{}\n", encoding="utf-8")
    (optimizer_dir / "__0_0.distcp").write_bytes(b"optimizer")
    model_dir = path / "pretrained_model" / "pytorch_model_fsdp_0"
    model_dir.mkdir(parents=True)
    (model_dir / "__0_0.distcp").write_bytes(b"model")
    (path / "pretrained_model" / "train_config.json").write_text("{}\n", encoding="utf-8")
    return path


class PruneFSDPCheckpointsTest(unittest.TestCase):
    def test_keeps_latest_two_and_requested_milestone(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            for step in (2000, 4000, 6000, 8000):
                _complete_checkpoint(checkpoints, step)
            (checkpoints / "last").symlink_to("008000")
            found = prune.discover(Path(tmp))
            keep, delete = prune.plan(found, keep_last=2, keep_steps={2000})
            self.assertEqual([item.step for item in keep], [2000, 6000, 8000])
            self.assertEqual([item.step for item in delete], [4000])

    def test_last_symlink_protects_current_and_newer_unpublished_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            old = _complete_checkpoint(checkpoints, 2000)
            _complete_checkpoint(checkpoints, 4000)
            _complete_checkpoint(checkpoints, 6000)
            (checkpoints / "last").symlink_to(old.name)
            found = prune.discover(Path(tmp))
            keep, delete = prune.plan(found, keep_last=1, keep_steps=set())
            self.assertEqual([item.step for item in keep], [2000])
            self.assertEqual(delete, [])
            self.assertFalse(next(item for item in found if item.step == 6000).complete)

    def test_incomplete_checkpoint_is_never_deletion_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            checkpoints.mkdir()
            _complete_checkpoint(checkpoints, 2000)
            incomplete = checkpoints / "004000"
            incomplete.mkdir()
            _complete_checkpoint(checkpoints, 6000)
            (checkpoints / "last").symlink_to("006000")
            found = prune.discover(Path(tmp))
            _, delete = prune.plan(found, keep_last=1, keep_steps=set())
            self.assertEqual([item.step for item in delete], [2000])
            self.assertFalse(next(item for item in found if item.step == 4000).complete)


if __name__ == "__main__":
    unittest.main()
