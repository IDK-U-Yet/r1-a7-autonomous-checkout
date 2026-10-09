# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import builtins
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from unitree_lerobot.r1a7_contract import (
    CAMERA_KEYS,
    CAMERA_SHAPES,
    JOINT_NAMES,
    ROBOT_TYPE,
)
from unitree_lerobot.r1a7_hardware_gateway import (
    REAL_ROBOT_CONFIRMATION,
    RealRobotAuthorization,
    RealRobotAuthorizationError,
    load_hardened_controller_types,
)
from unitree_lerobot.r1a7_shadow_inference import main


HARDWARE_MODULE_PREFIXES = (
    "unitree_sdk2py",
    "teleop",
    "teleimager",
    "unitree_lerobot.eval_robot",
)


def _make_dataset(root):
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [16],
            "names": [list(JOINT_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": [16],
            "names": [list(JOINT_NAMES)],
        },
    }
    for key in CAMERA_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": list(CAMERA_SHAPES[key]),
            "names": ["height", "width", "channel"],
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": 1,
        "total_frames": 1,
        "total_tasks": 1,
        "fps": 15,
        "features": features,
    }
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")

    action = np.zeros(16, dtype=np.float32)
    action[14:] = [5.4, 2.5]
    table = pa.table(
        {
            "observation.state": [np.zeros(16, dtype=np.float32).tolist()],
            "action": [action.tolist()],
            "episode_index": [3],
            "frame_index": [7],
            "index": [0],
            "task_index": [9],
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": [9],
                "task": ["scan the product"],
            }
        ),
        meta / "tasks.parquet",
    )
    return action


def test_dry_run_outputs_valid_action_without_importing_hardware(tmp_path, capsys, monkeypatch):
    expected_action = _make_dataset(tmp_path)
    imported_before = set(sys.modules)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(HARDWARE_MODULE_PREFIXES):
            raise AssertionError(f"dry-run attempted hardware import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    exit_code = main(
        [
            "--dataset-root",
            str(tmp_path),
            "--episode",
            "3",
            "--frame",
            "7",
            "--policy-path",
            str(tmp_path / "does-not-exist"),
            "--dry-run",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["hardware_commands_sent"] is False
    assert payload["mode"] == "dry-run-dataset-reference"
    assert payload["task"] == "scan the product"
    np.testing.assert_allclose(payload["action"], expected_action)
    newly_imported = set(sys.modules) - imported_before
    assert not any(name.startswith(HARDWARE_MODULE_PREFIXES) for name in newly_imported)


def test_hardware_gateway_refuses_before_import(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("importlib.import_module", lambda name: calls.append(name))

    with pytest.raises(RealRobotAuthorizationError):
        load_hardened_controller_types(
            tmp_path,
            RealRobotAuthorization(
                allow_real_robot=True,
                confirmation="wrong confirmation",
            ),
        )
    with pytest.raises(RealRobotAuthorizationError):
        load_hardened_controller_types(
            tmp_path,
            RealRobotAuthorization(
                allow_real_robot=False,
                confirmation=REAL_ROBOT_CONFIRMATION,
            ),
        )
    assert calls == []
