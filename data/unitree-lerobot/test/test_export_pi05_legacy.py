# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import os
from pathlib import Path

import pytest

from unitree_lerobot import export_pi05_legacy as exporter
from unitree_lerobot.r1a7_contract import CAMERA_KEYS, CAMERA_SHAPES, JOINT_NAMES, ROBOT_TYPE


def _dataset_info():
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
    return {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "features": features,
    }


def _make_dataset(root):
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps(_dataset_info()), encoding="utf-8")
    vector_stats = {
        name: [0.0] * 16
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    vector_stats["count"] = [1]
    stats = {
        "observation.state": vector_stats,
        "action": vector_stats,
    }
    (meta / "stats.json").write_text(json.dumps(stats), encoding="utf-8")


def _latest_config(**overrides):
    config = {
        "type": "pi05",
        "n_obs_steps": 1,
        "device": "cuda",
        "use_amp": False,
        "use_peft": False,
        "pretrained_revision": "new-only-field",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "float32",
        "chunk_size": 50,
        "n_action_steps": 8,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "use_visual_memory": False,
        "use_proprioceptive_memory": False,
        "use_relative_actions": False,
        "relative_exclude_joints": ["gripper"],
        "action_feature_names": list(JOINT_NAMES),
        "rtc_config": None,
        "rtc_training_max_delay": 0,
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "tokenizer_max_length": 200,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_visual_memory", True),
        ("use_proprioceptive_memory", True),
        ("use_relative_actions", True),
        ("rtc_config", {"enabled": True}),
        ("rtc_training_max_delay", 3),
        ("use_peft", True),
    ],
)
def test_export_rejects_latest_only_runtime_features(field, value):
    with pytest.raises(exporter.LegacyExportError):
        exporter.make_legacy_config(
            _latest_config(**{field: value}),
            _dataset_info(),
            output_path=Path("/tmp/export"),
            device="cpu",
            tokenizer_name=None,
            legacy_fields=exporter.LEGACY_PI05_CONFIG_FIELDS,
        )


def test_small_fake_checkpoint_exports_layout_and_filters_config(tmp_path, monkeypatch):
    source = tmp_path / "latest"
    dataset = tmp_path / "dataset"
    output = tmp_path / "legacy"
    source.mkdir()
    _make_dataset(dataset)
    (source / "config.json").write_text(json.dumps(_latest_config()), encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"tiny fake model weights")

    monkeypatch.setattr(
        exporter,
        "_legacy_dataclass_fields",
        lambda: exporter.LEGACY_PI05_CONFIG_FIELDS,
    )
    monkeypatch.setattr(
        exporter,
        "_write_stats_safetensors",
        lambda stats, path: path.write_bytes(b"tiny fake processor state"),
    )
    validated = []
    monkeypatch.setattr(exporter, "_validate_legacy_config", validated.append)

    manifest = exporter.export_legacy_checkpoint(
        source,
        dataset,
        output,
        weights_mode="hardlink",
        device="cpu",
        tokenizer_name="/local/paligemma-tokenizer",
    )

    assert validated
    assert manifest["model_weights_loaded"] is False
    assert os.stat(source / "model.safetensors").st_ino == os.stat(
        output / "model.safetensors"
    ).st_ino
    assert {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        exporter.PREPROCESSOR_STATE_FILE,
        exporter.POSTPROCESSOR_STATE_FILE,
        "legacy_export.json",
    } <= {path.name for path in output.iterdir()}

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert "use_relative_actions" not in config
    assert "rtc_config" not in config
    assert "pretrained_revision" not in config
    assert set(config["input_features"]) == set(CAMERA_KEYS) | {"observation.state"}
    assert config["output_features"]["action"]["shape"] == [16]
    assert config["tokenizer_name"] == "/local/paligemma-tokenizer"
