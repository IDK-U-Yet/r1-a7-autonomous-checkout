# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "r1_joint_replay.py"
SPEC = importlib.util.spec_from_file_location("r1_joint_replay", MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


def make_sample(index, *, include_dynamics=True):
    q_state = np.linspace(0.0, 0.13, 14) + index * 0.01
    q_target = q_state + 0.001

    def arm_fields(values, start):
        fields = {"qpos": values[start:start + 7].tolist()}
        if include_dynamics:
            fields["qvel"] = [0.1] * 7
            fields["torque"] = [0.2] * 7
        else:
            fields["qvel"] = []
            fields["torque"] = []
        return fields

    return {
        "idx": index,
        "states": {
            "left_arm": arm_fields(q_state, 0),
            "right_arm": arm_fields(q_state, 7),
        },
        "actions": {
            "left_arm": arm_fields(q_target, 0),
            "right_arm": arm_fields(q_target, 7),
        },
        "metadata": {"record_monotonic_ns": 1_000_000_000 + index * 40_000_000},
    }


class JointReplayTest(unittest.TestCase):
    def write_episode(self, root, *, include_dynamics=True, recording_mode=None):
        episode_dir = root / "episode_0001"
        episode_dir.mkdir()
        info = {"image": {"fps": 25}}
        if recording_mode is not None:
            info.update(
                {
                    "recording_mode": recording_mode,
                    "teach_parameters": {"enabled_arms": "left"},
                }
            )
        payload = {
            "info": info,
            "data": [
                make_sample(0, include_dynamics=include_dynamics),
                make_sample(1, include_dynamics=include_dynamics),
                make_sample(2, include_dynamics=include_dynamics),
            ],
        }
        path = episode_dir / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_joint_dynamics_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_episode(Path(directory))
            trajectory = replay.load_trajectory(path.parent)

        self.assertEqual(trajectory.q_target.shape, (3, 14))
        self.assertEqual(trajectory.dq_state.shape, (3, 14))
        self.assertTrue(trajectory.recorded_torque)
        np.testing.assert_allclose(trajectory.time_s, [0.0, 0.04, 0.08])

    def test_legacy_empty_dynamics_use_zero_feedforward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_episode(Path(directory), include_dynamics=False)
            trajectory = replay.load_trajectory(path)

        self.assertFalse(trajectory.recorded_torque)
        np.testing.assert_array_equal(trajectory.tau_target, np.zeros((3, 14)))
        np.testing.assert_array_equal(trajectory.dq_state, np.zeros((3, 14)))

    def test_summary_limits_metrics_to_selected_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_episode(Path(directory))
            trajectory = replay.load_trajectory(path)
            summary = replay.summarize_trajectory(trajectory, "left")

        metrics = summary["recorded_same_sample_encoder_error"]
        self.assertIn("left_shoulder_pitch", metrics)
        self.assertNotIn("right_shoulder_pitch", metrics)
        self.assertAlmostEqual(metrics["all_selected"]["max_abs_deg"], np.rad2deg(0.001))

    def test_task_directory_with_multiple_episodes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_episode(root)
            second_dir = root / "episode_0002"
            second_dir.mkdir()
            (second_dir / "data.json").write_text(first.read_text(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "select one episode"):
                replay.resolve_episode(root)

    def test_real_replay_rejects_unmarked_episode_before_hardware_init(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_episode(Path(directory))
            trajectory = replay.load_trajectory(path)
            with self.assertRaisesRegex(ValueError, "dedicated R1-A7 recording tool"):
                replay.run_replay(trajectory, SimpleNamespace())

    def test_drag_replay_requires_matching_arm_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_episode(
                Path(directory), recording_mode="r1_drag_teach"
            )
            trajectory = replay.load_trajectory(path)
            with self.assertRaisesRegex(ValueError, "recorded='left'"):
                replay.run_replay(
                    trajectory, SimpleNamespace(enabled_arms="right")
                )


if __name__ == "__main__":
    unittest.main()
