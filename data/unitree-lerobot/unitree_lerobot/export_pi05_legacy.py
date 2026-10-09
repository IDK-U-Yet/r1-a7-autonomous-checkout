# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Export a latest-LeRobot full PI0.5 checkpoint for the vendored legacy runtime.

The exporter never loads model weights.  It filters configuration through the
legacy PI05Config dataclass field set, rebuilds legacy processor files from a
strictly validated R1-A7 dataset, and links or copies ``model.safetensors``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from unitree_lerobot.r1a7_contract import CAMERA_KEYS, load_dataset_info


class LegacyExportError(ValueError):
    """Raised when a new checkpoint cannot be represented by legacy PI0.5."""


# Snapshot of dataclasses.fields(legacy PI05Config).  At export time it is
# checked against the imported legacy class before it is used as a whitelist.
LEGACY_PI05_CONFIG_FIELDS = frozenset(
    {
        "n_obs_steps",
        "input_features",
        "output_features",
        "device",
        "use_amp",
        "push_to_hub",
        "repo_id",
        "private",
        "tags",
        "license",
        "pretrained_path",
        "paligemma_variant",
        "action_expert_variant",
        "dtype",
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "num_inference_steps",
        "time_sampling_beta_alpha",
        "time_sampling_beta_beta",
        "time_sampling_scale",
        "time_sampling_offset",
        "min_period",
        "max_period",
        "image_resolution",
        "empty_cameras",
        "tokenizer_max_length",
        "tokenizer_name",
        "normalization_mapping",
        "gradient_checkpointing",
        "compile_model",
        "compile_mode",
        "optimizer_lr",
        "optimizer_betas",
        "optimizer_eps",
        "optimizer_weight_decay",
        "optimizer_grad_clip_norm",
        "scheduler_warmup_steps",
        "scheduler_decay_steps",
        "scheduler_decay_lr",
    }
)

NORMALIZATION_MAPPING = {
    "VISUAL": "IDENTITY",
    "STATE": "QUANTILES",
    "ACTION": "QUANTILES",
}
PREPROCESSOR_STATE_FILE = "policy_preprocessor_step_2_normalizer_processor.safetensors"
POSTPROCESSOR_STATE_FILE = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyExportError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyExportError(f"expected a JSON object in {path}")
    return value


def _legacy_dataclass_fields() -> frozenset[str]:
    try:
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except ImportError as exc:
        raise RuntimeError(
            "the vendored legacy LeRobot package must be importable to export PI0.5"
        ) from exc
    actual = frozenset(field.name for field in dataclasses.fields(PI05Config))
    if actual != LEGACY_PI05_CONFIG_FIELDS:
        missing = sorted(actual - LEGACY_PI05_CONFIG_FIELDS)
        stale = sorted(LEGACY_PI05_CONFIG_FIELDS - actual)
        raise RuntimeError(
            "legacy PI05Config fields changed; update the exporter whitelist before use: "
            f"missing_from_whitelist={missing}, stale_in_whitelist={stale}"
        )
    return actual


def _reject_unsupported_latest_features(config: Mapping[str, Any]) -> None:
    enabled_flags = (
        "use_visual_memory",
        "use_proprioceptive_memory",
        "use_relative_actions",
        "use_peft",
    )
    enabled = [name for name in enabled_flags if config.get(name) is True]
    enabled.extend(
        name
        for name, value in config.items()
        if name.startswith("lora_") and value is True
    )
    if enabled:
        raise LegacyExportError(
            "checkpoint enables features unsupported by legacy PI0.5: "
            f"{sorted(enabled)}"
        )

    rtc_config = config.get("rtc_config")
    if rtc_config is not None:
        if not isinstance(rtc_config, Mapping) or rtc_config.get("enabled", True):
            raise LegacyExportError("enabled RTC cannot be exported to legacy PI0.5")
    rtc_training_delay = config.get("rtc_training_max_delay", 0)
    if rtc_training_delay not in (None, 0):
        raise LegacyExportError(
            "RTC-trained checkpoints cannot be exported to legacy PI0.5: "
            f"rtc_training_max_delay={rtc_training_delay!r}"
        )


def _dataset_policy_features(info: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    features = info["features"]
    input_features: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        height, width, channels = features[key]["shape"]
        input_features[key] = {
            "type": "VISUAL",
            "shape": [channels, height, width],
        }
    input_features["observation.state"] = {"type": "STATE", "shape": [16]}
    output_features = {"action": {"type": "ACTION", "shape": [16]}}
    return input_features, output_features


def make_legacy_config(
    latest_config: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
    *,
    output_path: Path,
    device: str,
    tokenizer_name: str | None,
    legacy_fields: frozenset[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Filter latest config and replace all dataset-dependent feature fields."""

    if latest_config.get("type") != "pi05":
        raise LegacyExportError(
            f"source policy type must be 'pi05', got {latest_config.get('type')!r}"
        )
    _reject_unsupported_latest_features(latest_config)
    whitelist = _legacy_dataclass_fields() if legacy_fields is None else legacy_fields
    filtered = {
        key: value for key, value in latest_config.items() if key in whitelist
    }
    dropped = sorted(
        key for key in latest_config if key != "type" and key not in whitelist
    )

    input_features, output_features = _dataset_policy_features(dataset_info)
    filtered.update(
        {
            "type": "pi05",
            "n_obs_steps": 1,
            "input_features": input_features,
            "output_features": output_features,
            "device": device,
            "use_amp": bool(filtered.get("use_amp", False)),
            "push_to_hub": False,
            "pretrained_path": str(output_path),
            "empty_cameras": 0,
            "normalization_mapping": dict(NORMALIZATION_MAPPING),
        }
    )
    resolved_tokenizer = tokenizer_name or filtered.get("tokenizer_name")
    if not isinstance(resolved_tokenizer, str) or not resolved_tokenizer:
        resolved_tokenizer = "google/paligemma-3b-pt-224"
    filtered["tokenizer_name"] = resolved_tokenizer

    unknown_output_fields = set(filtered) - whitelist - {"type"}
    if unknown_output_fields:
        raise AssertionError(f"exporter emitted non-legacy fields: {unknown_output_fields}")
    return filtered, dropped


def _load_and_validate_stats(dataset_root: Path) -> dict[str, dict[str, Any]]:
    stats_path = dataset_root / "meta" / "stats.json"
    stats = _read_json(stats_path)
    for feature_name, expected_size in (("observation.state", 16), ("action", 16)):
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, Mapping):
            raise LegacyExportError(f"stats are missing {feature_name!r}")
        for stat_name in ("q01", "q99"):
            values = np.asarray(feature_stats.get(stat_name), dtype=np.float32)
            if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
                raise LegacyExportError(
                    f"{feature_name}.{stat_name} must be a finite ({expected_size},) vector"
                )
    return stats


def _flatten_stats(stats: Mapping[str, Any]) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for feature_name, feature_stats in stats.items():
        if not isinstance(feature_stats, Mapping):
            raise LegacyExportError(f"invalid stats entry for {feature_name!r}")
        for stat_name, values in feature_stats.items():
            array = np.asarray(values, dtype=np.float32)
            if not np.all(np.isfinite(array)):
                raise LegacyExportError(
                    f"stats contain non-finite values in {feature_name}.{stat_name}"
                )
            tensors[f"{feature_name}.{stat_name}"] = np.ascontiguousarray(array)
    return tensors


def _write_stats_safetensors(stats: Mapping[str, Any], path: Path) -> None:
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to rebuild legacy PI0.5 processors"
        ) from exc
    save_file(_flatten_stats(stats), path)


def _processor_features(config: Mapping[str, Any]) -> dict[str, Any]:
    return {**config["input_features"], **config["output_features"]}


def _write_legacy_processors(
    output_dir: Path,
    config: Mapping[str, Any],
    stats: Mapping[str, Any],
) -> None:
    all_features = _processor_features(config)
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}},
            },
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": all_features,
                    "norm_map": dict(NORMALIZATION_MAPPING),
                },
                "state_file": PREPROCESSOR_STATE_FILE,
            },
            {
                "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                "config": {
                    "max_state_dim": config.get("max_state_dim", 32),
                    "task_key": "task",
                },
            },
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": config.get("tokenizer_max_length", 200),
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": config["tokenizer_name"],
                },
            },
            {
                "registry_name": "device_processor",
                "config": {"device": config["device"], "float_dtype": None},
            },
        ],
    }
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-8,
                    "features": config["output_features"],
                    "norm_map": dict(NORMALIZATION_MAPPING),
                },
                "state_file": POSTPROCESSOR_STATE_FILE,
            },
            {
                "registry_name": "device_processor",
                "config": {"device": "cpu", "float_dtype": None},
            },
        ],
    }
    (output_dir / "policy_preprocessor.json").write_text(
        json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "policy_postprocessor.json").write_text(
        json.dumps(postprocessor, indent=2) + "\n", encoding="utf-8"
    )
    _write_stats_safetensors(stats, output_dir / PREPROCESSOR_STATE_FILE)
    _write_stats_safetensors(stats, output_dir / POSTPROCESSOR_STATE_FILE)


def _place_weights(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise LegacyExportError(
                "cannot hardlink model.safetensors; use --weights-mode copy when source "
                f"and output are on different filesystems: {exc}"
            ) from exc
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported weights mode: {mode!r}")
    if destination.stat().st_size != source.stat().st_size:
        raise LegacyExportError("exported model.safetensors size does not match source")


def _validate_legacy_config(output_dir: Path) -> None:
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except ImportError as exc:
        raise RuntimeError(
            "cannot import the legacy PI05Config parser for export validation"
        ) from exc
    parsed = PreTrainedConfig.from_pretrained(output_dir)
    if not isinstance(parsed, PI05Config):
        raise LegacyExportError(
            f"legacy parser returned {type(parsed).__name__}, expected PI05Config"
        )
    if set(parsed.input_features) != set(CAMERA_KEYS) | {"observation.state"}:
        raise LegacyExportError("legacy parser did not preserve the R1-A7 input feature set")
    if parsed.output_features["action"].shape != (16,):
        raise LegacyExportError("legacy parser did not preserve the 16D action feature")


def export_legacy_checkpoint(
    source_policy: str | Path,
    dataset_root: str | Path,
    output: str | Path,
    *,
    weights_mode: str = "hardlink",
    device: str = "cuda",
    tokenizer_name: str | None = None,
) -> dict[str, Any]:
    source = Path(source_policy).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"output already exists and will not be overwritten: {destination}"
        )
    config_path = source / "config.json"
    weights_path = source / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"source checkpoint must contain config.json: {source}"
        )

    latest_config = _read_json(config_path)
    _reject_unsupported_latest_features(latest_config)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"full source checkpoint must contain model.safetensors: {source}"
        )
    dataset_info = load_dataset_info(dataset)
    stats = _load_and_validate_stats(dataset)
    legacy_config, dropped_fields = make_legacy_config(
        latest_config,
        dataset_info,
        output_path=destination,
        device=device,
        tokenizer_name=tokenizer_name,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.export-", dir=destination.parent)
    )
    try:
        (temporary / "config.json").write_text(
            json.dumps(legacy_config, indent=4) + "\n", encoding="utf-8"
        )
        _write_legacy_processors(temporary, legacy_config, stats)
        _place_weights(weights_path, temporary / "model.safetensors", weights_mode)
        _validate_legacy_config(temporary)
        manifest = {
            "format": "lerobot-pi05-legacy-export-v1",
            "source_policy": str(source),
            "dataset_root": str(dataset),
            "weights_mode": weights_mode,
            "model_weights_loaded": False,
            "dropped_latest_config_fields": dropped_fields,
        }
        (temporary / "legacy_export.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a full latest-LeRobot PI0.5 checkpoint for legacy LeRobot."
    )
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights-mode", choices=("hardlink", "copy"), default="hardlink"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer-name", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_legacy_checkpoint(
        args.source_policy,
        args.dataset_root,
        args.output,
        weights_mode=args.weights_mode,
        device=args.device,
        tokenizer_name=args.tokenizer_name,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
