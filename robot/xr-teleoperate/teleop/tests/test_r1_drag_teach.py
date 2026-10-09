# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "r1_drag_teach.py"
SPEC = importlib.util.spec_from_file_location("r1_drag_teach", MODULE_PATH)
teach = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teach
SPEC.loader.exec_module(teach)


class DragTeachDataTest(unittest.TestCase):
    def test_sample_uses_measured_q_as_replay_action(self):
        q = np.arange(14, dtype=np.float64) / 10.0
        dq = np.arange(14, dtype=np.float64) / 100.0
        tau = np.arange(14, dtype=np.float64) / 20.0
        sample = teach.make_sample(3, q, dq, tau, 123, 456)

        self.assertEqual(sample["idx"], 3)
        self.assertEqual(sample["states"]["left_arm"]["qpos"], q[:7].tolist())
        self.assertEqual(sample["actions"]["right_arm"]["qpos"], q[7:].tolist())
        self.assertEqual(sample["actions"]["left_arm"]["torque"], tau[:7].tolist())
        self.assertEqual(sample["metadata"]["recording_mode"], "r1_drag_teach")

    def test_episode_numbering_and_atomic_save(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory) / "task"
            first = teach.next_episode_dir(task_dir)
            second = teach.next_episode_dir(task_dir)
            self.assertEqual(first.name, "episode_0000")
            self.assertEqual(second.name, "episode_0001")

            args = SimpleNamespace(
                frequency=100.0,
                enabled_arms="left",
                teach_kp=0.0,
                teach_kd=2.0,
                gravity_scale=1.0,
                transition_duration=2.0,
                task_goal="test",
            )
            q = np.zeros(14)
            samples = [
                teach.make_sample(index, q, q, q, 100 + index, 200 + index)
                for index in range(2)
            ]
            path = teach.save_episode(first, samples, args)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["info"]["recording_mode"], "r1_drag_teach")
        self.assertEqual(payload["info"]["teach_parameters"]["enabled_arms"], "left")
        self.assertEqual(len(payload["data"]), 2)


if __name__ == "__main__":
    unittest.main()
