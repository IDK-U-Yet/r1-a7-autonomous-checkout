#!/usr/bin/env python3
"""Read-only preflight checks for R1-A7 LeRobot training and inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDA_ROOT = Path("/home/jieguo/anaconda3")
DEFAULT_OLD_SOURCE = ROOT / "data/unitree-lerobot/unitree_lerobot/lerobot"
DEFAULT_NEW_SOURCE = Path("/media/jieguo/disk/zyc/lerobot")
DEFAULT_PLUGIN_ROOT = DEFAULT_NEW_SOURCE / "integrations/lerobot_robot_unitree_r1a7"
DEFAULT_SOURCE_DATASET = Path(
    "/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30"
)
DEFAULT_TRAIN_DATASET = Path(
    "/home/jieguo/.cache/huggingface/lerobot/local/"
    "r1a7_black_truffle_4cam_30_pi05_stats"
)
DEFAULT_BASE = DEFAULT_NEW_SOURCE / "pretrained/pi05_base"
DEFAULT_TOKENIZER = DEFAULT_NEW_SOURCE / "pretrained/google/paligemma-3b-pt-224"
EXPECTED_CAMERAS = {
    "observation.images.cam_global": (360, 640, 3),
    "observation.images.cam_left_wrist": (480, 640, 3),
    "observation.images.cam_right_wrist": (480, 640, 3),
    "observation.images.cam_scanner": (480, 640, 3),
}


ENV_PROBE = r"""
import importlib.metadata as metadata
import importlib
import json
import sys
from pathlib import Path


def dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


result = {
    "python": sys.version.split()[0],
    "prefix": sys.prefix,
    "packages": {
        name: dist_version(name)
        for name in (
            "lerobot",
            "torch",
            "torchvision",
            "transformers",
            "datasets",
            "pyarrow",
            "av",
            "torchcodec",
            "accelerate",
            "huggingface-hub",
            "cyclonedds",
            "draccus",
            "unitree-sdk2py",
            "lerobot_robot_unitree_r1a7",
        )
    },
}

try:
    import lerobot

    result["lerobot_file"] = str(Path(lerobot.__file__).resolve())
    result["lerobot_module_version"] = getattr(lerobot, "__version__", None)
except Exception as exc:
    result["lerobot_error"] = f"{type(exc).__name__}: {exc}"

try:
    import torch

    def gpu_info(index):
        with torch.cuda.device(index):
            bf16 = bool(torch.cuda.is_bf16_supported())
        properties = torch.cuda.get_device_properties(index)
        return {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory": properties.total_memory,
            "bf16": bf16,
        }

    result["torch"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [gpu_info(index) for index in range(torch.cuda.device_count())],
    }
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"

try:
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    result["pi05_import"] = PI05Config.__name__
except Exception as exc:
    result["pi05_error"] = f"{type(exc).__name__}: {exc}"

try:
    import lerobot_robot_unitree_r1a7

    result["r1a7_plugin_file"] = str(Path(lerobot_robot_unitree_r1a7.__file__).resolve())
except Exception as exc:
    result["r1a7_plugin_error"] = f"{type(exc).__name__}: {exc}"

result["hardware_imports"] = {}
for module_name in ("unitree_sdk2py", "cyclonedds", "cv2"):
    try:
        module = importlib.import_module(module_name)
        result["hardware_imports"][module_name] = {
            "file": str(Path(module.__file__).resolve()),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        result["hardware_imports"][module_name] = {
            "error": f"{type(exc).__name__}: {exc}"
        }

print(json.dumps(result))
"""


# Keep the training preflight independent from the robot-control stack. Importing
# torch is safe here, but touching torch.cuda would create CUDA contexts in a
# process whose only job is to validate the Python installation.
TRAINING_ENV_PROBE = r"""
import importlib.metadata as metadata
import json
import sys
from pathlib import Path


def dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


result = {
    "python": sys.version.split()[0],
    "prefix": sys.prefix,
    "packages": {
        name: dist_version(name)
        for name in (
            "lerobot",
            "torch",
            "torchvision",
            "transformers",
            "datasets",
            "pyarrow",
            "av",
            "accelerate",
            "huggingface-hub",
            "draccus",
        )
    },
}

try:
    import lerobot

    result["lerobot_file"] = str(Path(lerobot.__file__).resolve())
    result["lerobot_module_version"] = getattr(lerobot, "__version__", None)
except Exception as exc:
    result["lerobot_error"] = f"{type(exc).__name__}: {exc}"

try:
    import torch

    result["torch"] = {
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"

try:
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    result["pi05_import"] = PI05Config.__name__
except Exception as exc:
    result["pi05_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result))
"""


DATASET_PROBE = r"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


root = Path(sys.argv[1]).expanduser().resolve()
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
data_files = sorted((root / "data").glob("*/*.parquet"))
episode_files = sorted((root / "meta/episodes").glob("*/*.parquet"))


def read_vectors(feature):
    parts = []
    for path in data_files:
        values = pq.read_table(path, columns=[feature]).column(feature).to_pylist()
        if values:
            parts.append(np.asarray(values, dtype=np.float64))
    if not parts:
        raise ValueError(f"No rows for {feature}")
    return np.concatenate(parts, axis=0)


def quantile_errors(feature, values):
    errors = {}
    for name, quantile in (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)):
        stored = np.asarray(stats[feature][name], dtype=np.float64)
        exact = np.quantile(values, quantile, axis=0)
        errors[name] = float(np.max(np.abs(stored - exact)))
    return errors


state = read_vectors("observation.state")
action = read_vectors("action")
data_rows = 0
episode_indices = []
for path in data_files:
    table = pq.read_table(path, columns=["episode_index"])
    data_rows += table.num_rows
    episode_indices.extend(table.column("episode_index").to_pylist())

episode_rows = sum(pq.read_metadata(path).num_rows for path in episode_files)
task_rows = pq.read_metadata(root / "meta/tasks.parquet").num_rows

videos = {}
for key, feature in info["features"].items():
    if feature.get("dtype") != "video":
        continue
    paths = sorted((root / "videos" / key).glob("*/*.mp4"))
    streams = []
    for path in paths:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate
            streams.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "codec": stream.codec_context.codec.name,
                    "width": stream.width,
                    "height": stream.height,
                    "frames": stream.frames,
                    "fps": float(Fraction(rate)) if rate is not None else None,
                }
            )
    videos[key] = streams

result = {
    "root": str(root),
    "codebase_version": info.get("codebase_version"),
    "robot_type": info.get("robot_type"),
    "fps": info.get("fps"),
    "total_frames": info.get("total_frames"),
    "total_episodes": info.get("total_episodes"),
    "total_tasks": info.get("total_tasks"),
    "data_rows": data_rows,
    "episode_rows": episode_rows,
    "task_rows": task_rows,
    "episode_count_in_data": len(set(episode_indices)),
    "state_shape": list(state.shape),
    "action_shape": list(action.shape),
    "state_finite": bool(np.isfinite(state).all()),
    "action_finite": bool(np.isfinite(action).all()),
    "feature_metadata": info.get("features", {}),
    "quantile_errors": {
        "observation.state": quantile_errors("observation.state", state),
        "action": quantile_errors("action", action),
    },
    "videos": videos,
    "exact_stats_audit": (root / "meta/r1a7_exact_stats_audit.json").is_file(),
}
print(json.dumps(result))
"""


MODEL_PROBE = r"""
import json
import sys
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


checkpoint = Path(sys.argv[1]).expanduser().resolve()
tokenizer_root = Path(sys.argv[2]).expanduser().resolve()
config_device = sys.argv[3] if len(sys.argv) > 3 else "cuda"
config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
processor = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
tokenizer_names = [
    step.get("config", {}).get("tokenizer_name")
    for step in processor.get("steps", [])
    if step.get("registry_name") == "tokenizer_processor"
]

with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as tensors:
    keys = list(tensors.keys())
    first_shape = list(tensors.get_slice(keys[0]).get_shape()) if keys else None

tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
resolved = PreTrainedConfig.from_pretrained(
    checkpoint,
    cli_overrides=[
        "--input_features=null",
        f"--device={config_device}",
        "--dtype=bfloat16",
        "--freeze_vision_encoder=false",
        "--train_expert_only=false",
        "--gradient_checkpointing=true",
        "--push_to_hub=false",
    ],
)

print(
    json.dumps(
        {
            "config_type": config.get("type"),
            "checkpoint_device": config.get("device"),
            "checkpoint_dtype": config.get("dtype"),
            "max_state_dim": config.get("max_state_dim"),
            "max_action_dim": config.get("max_action_dim"),
            "checkpoint_image_keys": [
                key
                for key, value in config.get("input_features", {}).items()
                if value.get("type") == "VISUAL"
            ],
            "model_size": (checkpoint / "model.safetensors").stat().st_size,
            "tensor_keys": len(keys),
            "first_tensor_shape": first_shape,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "processor_tokenizer_names": tokenizer_names,
            "four_camera_override": {
                "input_features_is_none": resolved.input_features is None,
                "device": resolved.device,
                "dtype": resolved.dtype,
                "freeze_vision_encoder": resolved.freeze_vision_encoder,
                "train_expert_only": resolved.train_expert_only,
                "gradient_checkpointing": resolved.gradient_checkpointing,
                "push_to_hub": resolved.push_to_hub,
            },
        }
    )
)
"""


@dataclass
class Finding:
    status: str
    name: str
    message: str


class Report:
    def __init__(self, json_output: bool = False) -> None:
        self.json_output = json_output
        self.findings: list[Finding] = []

    def add(self, status: str, name: str, message: str) -> None:
        self.findings.append(Finding(status, name, message))
        if not self.json_output:
            print(f"[{status:<4}] {name}: {message}")

    def passed(self, name: str, message: str) -> None:
        self.add("PASS", name, message)

    def warn(self, name: str, message: str) -> None:
        self.add("WARN", name, message)

    def fail(self, name: str, message: str) -> None:
        self.add("FAIL", name, message)

    @property
    def failure_count(self) -> int:
        return sum(finding.status == "FAIL" for finding in self.findings)

    def finish(self) -> int:
        if self.json_output:
            print(
                json.dumps(
                    {
                        "ok": self.failure_count == 0,
                        "failure_count": self.failure_count,
                        "findings": [asdict(finding) for finding in self.findings],
                    },
                    indent=2,
                )
            )
        elif self.failure_count:
            print(f"R1-A7 setup check failed with {self.failure_count} required item(s).")
        else:
            print("R1-A7 setup check passed.")
        return 1 if self.failure_count else 0


def _read_project_version(source: Path) -> str | None:
    pyproject = source / "pyproject.toml"
    if not pyproject.is_file():
        return None
    text = pyproject.read_text(encoding="utf-8")
    project = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    if project is None:
        return None
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project.group(1), re.MULTILINE)
    return match.group(1) if match else None


def _is_within(path: str | None, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _version_tuple(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _run_probe(python: Path, code: str, args: list[str] | None = None) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [str(python), "-c", code, *(args or [])]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"probe exited {completed.returncode}: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("probe returned no JSON")
    return json.loads(lines[-1])


def _check_environment(
    report: Report,
    name: str,
    python: Path,
    source: Path,
    expected_python: tuple[int, int],
    require_pi05: bool,
    require_legacy_pi05: bool = False,
    require_hardware: bool = False,
    probe: str = ENV_PROBE,
    check_control_dependencies: bool = True,
) -> dict[str, Any] | None:
    if not python.is_file():
        report.fail(name, f"missing environment Python: {python}")
        return None
    if not source.is_dir():
        report.fail(name, f"missing LeRobot source: {source}")
        return None
    try:
        result = _run_probe(python, probe)
    except Exception as exc:  # noqa: BLE001 - turn all probe failures into findings
        report.fail(name, str(exc))
        return None

    installed = result.get("packages", {}).get("lerobot")
    source_version = _read_project_version(source)
    module_version = result.get("lerobot_module_version")
    python_version = tuple(int(part) for part in result["python"].split(".")[:2])
    details = (
        f"Python {result['python']}, LeRobot dist={installed}, module={module_version}, "
        f"source={source_version}"
    )
    if python_version != expected_python:
        report.fail(name, f"{details}; expected Python {expected_python[0]}.{expected_python[1]}")
    elif not installed or not source_version or installed != source_version or module_version != source_version:
        report.fail(name, f"{details}; editable install is not synchronized")
    elif not _is_within(result.get("lerobot_file"), source):
        report.fail(name, f"{details}; import resolves to {result.get('lerobot_file')}")
    elif require_pi05 and not result.get("pi05_import"):
        report.fail(name, f"{details}; PI0.5 import failed: {result.get('pi05_error')}")
    elif require_pi05 and not result.get("packages", {}).get("transformers"):
        report.fail(name, f"{details}; transformers is required for PI0.5")
    elif require_pi05 and not (
        (0, 11, 6) <= _version_tuple(result.get("packages", {}).get("draccus")) < (0, 12, 0)
    ):
        report.fail(
            name,
            f"{details}; draccus={result.get('packages', {}).get('draccus')} is outside "
            "the required >=0.11.6,<0.12 range",
        )
    else:
        report.passed(name, details)

    if not check_control_dependencies:
        return result

    hardware_finding = report.fail if require_hardware else report.warn
    if not result.get("packages", {}).get("unitree-sdk2py"):
        hardware_finding(
            f"{name}.unitree",
            "unitree-sdk2py is not installed in this environment; training-only checks can continue",
        )
    elif result.get("hardware_imports", {}).get("unitree_sdk2py", {}).get("error"):
        hardware_finding(
            f"{name}.unitree",
            "unitree_sdk2py import failed: "
            f"{result['hardware_imports']['unitree_sdk2py']['error']}",
        )
    if not result.get("packages", {}).get("cyclonedds"):
        hardware_finding(
            f"{name}.dds",
            "cyclonedds is not installed in this environment; training-only checks can continue",
        )
    elif result.get("hardware_imports", {}).get("cyclonedds", {}).get("error"):
        hardware_finding(
            f"{name}.dds",
            f"cyclonedds import failed: {result['hardware_imports']['cyclonedds']['error']}",
        )
    if result.get("hardware_imports", {}).get("cv2", {}).get("error"):
        hardware_finding(
            f"{name}.opencv",
            f"cv2 import failed: {result['hardware_imports']['cv2']['error']}",
        )
    if not require_pi05:
        legacy_ready = bool(
            result.get("packages", {}).get("transformers") and result.get("pi05_import")
        )
        if require_legacy_pi05 and not legacy_ready:
            report.fail(
                f"{name}.pi05",
                "legacy PI0.5 requires the fork's custom Transformers dependency and PI05 import",
            )
        elif not legacy_ready:
            report.warn(
                f"{name}.pi05",
                "transformers is absent; legacy dataset tools work, but legacy PI0.5 execution does not",
            )
    return result


def _max_quantile_error(result: dict[str, Any]) -> float:
    return max(
        error
        for feature_errors in result["quantile_errors"].values()
        for error in feature_errors.values()
    )


def _check_dataset(
    report: Report,
    python: Path,
    root: Path,
    name: str,
    require_exact_stats: bool,
    quantile_atol: float,
) -> dict[str, Any] | None:
    if not root.is_dir():
        report.fail(name, f"missing dataset: {root}")
        return None
    try:
        result = _run_probe(python, DATASET_PROBE, [str(root)])
    except Exception as exc:  # noqa: BLE001
        report.fail(name, str(exc))
        return None

    failures: list[str] = []
    if result["codebase_version"] != "v3.0":
        failures.append(f"format={result['codebase_version']!r}")
    if result["robot_type"] != "Unitree_R1_A7_Dex1_4Cam":
        failures.append(f"robot_type={result['robot_type']!r}")
    if result["fps"] != 15:
        failures.append(f"fps={result['fps']!r}")
    if result["state_shape"] != [result["total_frames"], 16]:
        failures.append(f"state_shape={result['state_shape']}")
    if result["action_shape"] != [result["total_frames"], 16]:
        failures.append(f"action_shape={result['action_shape']}")
    if not result["state_finite"] or not result["action_finite"]:
        failures.append("state/action contains NaN or Inf")
    if result["data_rows"] != result["total_frames"]:
        failures.append(f"rows={result['data_rows']} total_frames={result['total_frames']}")
    if result["episode_rows"] != result["total_episodes"]:
        failures.append(
            f"episode rows={result['episode_rows']} total_episodes={result['total_episodes']}"
        )
    if result["episode_count_in_data"] != result["total_episodes"]:
        failures.append(f"episode ids={result['episode_count_in_data']}")
    if result["task_rows"] != result["total_tasks"]:
        failures.append(f"task rows={result['task_rows']} total_tasks={result['total_tasks']}")

    metadata = result["feature_metadata"]
    actual_cameras = {
        key: tuple(value.get("shape", ()))
        for key, value in metadata.items()
        if value.get("dtype") == "video"
    }
    if actual_cameras != EXPECTED_CAMERAS:
        failures.append(f"camera features={actual_cameras}")
    for key, expected_shape in EXPECTED_CAMERAS.items():
        streams = result["videos"].get(key, [])
        if not streams:
            failures.append(f"missing video for {key}")
            continue
        if sum(stream["frames"] for stream in streams) != result["total_frames"]:
            failures.append(f"video frame count mismatch for {key}")
        for stream in streams:
            # PyAV reports the selected AV1 decoder (usually libdav1d), while
            # ffprobe and LeRobot metadata report the codec family as av1.
            if stream["codec"] not in {"av1", "libdav1d", "libaom-av1"}:
                failures.append(f"{key} codec={stream['codec']}")
            if (stream["height"], stream["width"], 3) != expected_shape:
                failures.append(
                    f"{key} dimensions={(stream['height'], stream['width'], 3)}"
                )
            if stream["fps"] is None or abs(stream["fps"] - 15.0) > 1e-6:
                failures.append(f"{key} fps={stream['fps']}")
            if stream["size"] <= 0:
                failures.append(f"{key} has an empty video file")

    quantile_error = _max_quantile_error(result)
    if require_exact_stats:
        if not result["exact_stats_audit"]:
            failures.append("missing meta/r1a7_exact_stats_audit.json")
        if quantile_error > quantile_atol:
            failures.append(
                f"stored numeric quantiles differ from full-Parquet values by {quantile_error:.6g}"
            )

    summary = (
        f"v3.0, {result['total_episodes']} episodes, {result['total_frames']} frames, "
        f"16D state/action, 4 AV1 cameras, max quantile error={quantile_error:.6g}"
    )
    if failures:
        report.fail(name, f"{summary}; " + "; ".join(failures))
    else:
        report.passed(name, summary)
    if not require_exact_stats and quantile_error > quantile_atol:
        report.warn(
            f"{name}.stats",
            f"original stats are intentionally not used for PI0.5 (max error={quantile_error:.6g})",
        )
    return result


def _check_model(
    report: Report,
    python: Path,
    checkpoint: Path,
    tokenizer: Path,
    config_device: str = "cuda",
) -> None:
    required_checkpoint_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required_checkpoint_files if not (checkpoint / name).is_file()]
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    missing_tokenizer = [name for name in tokenizer_files if not (tokenizer / name).is_file()]
    if missing or missing_tokenizer:
        report.fail(
            "pi05.assets",
            f"missing checkpoint files={missing}, tokenizer files={missing_tokenizer}",
        )
        return
    try:
        result = _run_probe(
            python,
            MODEL_PROBE,
            [str(checkpoint), str(tokenizer), config_device],
        )
    except Exception as exc:  # noqa: BLE001
        report.fail("pi05.assets", str(exc))
        return

    failures: list[str] = []
    if result["config_type"] != "pi05":
        failures.append(f"config type={result['config_type']!r}")
    if result["model_size"] < 14_000_000_000:
        failures.append(f"model is unexpectedly small ({result['model_size']} bytes)")
    if result["tensor_keys"] < 800:
        failures.append(f"only {result['tensor_keys']} tensor keys")
    if result["max_state_dim"] != 32 or result["max_action_dim"] != 32:
        failures.append("PI0.5 padding dimensions are not 32")
    if result["tokenizer_vocab_size"] != 257152:
        failures.append(f"tokenizer vocab={result['tokenizer_vocab_size']}")

    override = result["four_camera_override"]
    expected_override = {
        "input_features_is_none": True,
        "device": config_device,
        "dtype": "bfloat16",
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "gradient_checkpointing": True,
        "push_to_hub": False,
    }
    if override != expected_override:
        failures.append(f"full-finetune override did not resolve as expected: {override}")

    processor_names = [name for name in result["processor_tokenizer_names"] if name]
    if not processor_names:
        failures.append("preprocessor has no tokenizer path")
    elif not all(Path(name).expanduser().resolve() == tokenizer.resolve() for name in processor_names):
        failures.append(f"preprocessor tokenizer path={processor_names}")

    summary = (
        f"{result['model_size']} bytes, {result['tensor_keys']} tensor keys, "
        f"{result['tokenizer_class']} vocab={result['tokenizer_vocab_size']}"
    )
    if failures:
        report.fail("pi05.assets", f"{summary}; " + "; ".join(failures))
    else:
        report.passed("pi05.assets", summary)
    if len(result["checkpoint_image_keys"]) == 3:
        report.warn(
            "pi05.camera_contract",
            "base checkpoint declares 3 cameras; training must set policy.input_features=null "
            "to infer all 4 R1-A7 cameras",
        )


def _find_plugin_root(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = (
        DEFAULT_PLUGIN_ROOT,
        ROOT / "integrations/lerobot-r1a7",
        ROOT / "integrations/lerobot_r1a7",
        ROOT / "plugins/lerobot-r1a7",
        ROOT / "plugins/lerobot_r1a7",
    )
    return next((path for path in candidates if (path / "pyproject.toml").is_file()), None)


def _check_plugin(
    report: Report,
    plugin_root: Path | None,
    environments: dict[str, dict[str, Any] | None],
    required: bool,
) -> None:
    if plugin_root is None or not (plugin_root / "pyproject.toml").is_file():
        message = "R1-A7 LeRobot plugin source was not found; pass --plugin-root"
        if required:
            report.fail("r1a7.plugin", message)
        else:
            report.warn("r1a7.plugin", message)
        return
    version = _read_project_version(plugin_root)
    if not version:
        report.fail("r1a7.plugin", f"cannot read project version from {plugin_root}")
        return
    for environment_name, environment in environments.items():
        installed = (environment or {}).get("packages", {}).get("lerobot_robot_unitree_r1a7")
        import_file = (environment or {}).get("r1a7_plugin_file")
        finding_name = f"r1a7.plugin.{environment_name}"
        if installed != version:
            report.fail(
                finding_name,
                f"source version={version}, installed distribution={installed}; "
                "run setup-r1a7-envs.sh",
            )
        elif not _is_within(import_file, plugin_root):
            report.fail(
                finding_name,
                f"import resolves to {import_file!r}, expected source below {plugin_root}",
            )
        else:
            report.passed(
                finding_name,
                f"source={plugin_root}, version={version}, import={import_file}",
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-root", type=Path, default=DEFAULT_CONDA_ROOT)
    parser.add_argument("--old-source", type=Path, default=DEFAULT_OLD_SOURCE)
    parser.add_argument("--new-source", type=Path, default=DEFAULT_NEW_SOURCE)
    parser.add_argument("--source-dataset-root", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("R1A7_PI05_DATASET_ROOT", DEFAULT_TRAIN_DATASET)),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path(os.environ.get("R1A7_PI05_BASE", DEFAULT_BASE)),
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path(os.environ.get("R1A7_PI05_TOKENIZER", DEFAULT_TOKENIZER)),
    )
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument(
        "--training-only",
        "--skip-control",
        dest="training_only",
        action="store_true",
        help=(
            "Check only the new LeRobot training environment and assets. Skip the legacy "
            "environment, robot plugin, hardware dependencies, and all CUDA runtime queries."
        ),
    )
    parser.add_argument(
        "--require-legacy-pi",
        action="store_true",
        help="Require the unitree_lerobot environment to import legacy PI0.5 and Transformers.",
    )
    parser.add_argument(
        "--require-plugin",
        action="store_true",
        help="Treat a missing R1-A7 plugin source as a required-item failure.",
    )
    parser.add_argument(
        "--require-hardware",
        action="store_true",
        help="Require Unitree SDK, CycloneDDS, and OpenCV imports in both environments.",
    )
    parser.add_argument("--quantile-atol", type=float, default=1e-8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.training_only and (args.require_legacy_pi or args.require_plugin or args.require_hardware):
        parser.error(
            "--training-only/--skip-control cannot be combined with --require-legacy-pi, "
            "--require-plugin, or --require-hardware"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report(json_output=args.json)
    conda_root = args.conda_root.expanduser().resolve()
    old_python = conda_root / "envs/unitree_lerobot/bin/python"
    new_python = conda_root / "envs/lerobot/bin/python"

    old_environment = None
    if not args.training_only:
        old_environment = _check_environment(
            report,
            "env.unitree_lerobot",
            old_python,
            args.old_source.expanduser().resolve(),
            (3, 10),
            require_pi05=False,
            require_legacy_pi05=args.require_legacy_pi,
            require_hardware=args.require_hardware,
        )
    new_environment = _check_environment(
        report,
        "env.lerobot",
        new_python,
        args.new_source.expanduser().resolve(),
        (3, 12),
        require_pi05=True,
        require_hardware=args.require_hardware,
        probe=TRAINING_ENV_PROBE if args.training_only else ENV_PROBE,
        check_control_dependencies=not args.training_only,
    )

    if args.training_only:
        report.passed(
            "mode.training_only",
            "legacy control environment, robot plugin, hardware imports, and CUDA queries skipped",
        )
    elif new_environment is not None:
        torch_info = new_environment.get("torch", {})
        devices = torch_info.get("devices", [])
        if torch_info.get("cuda_available") and len(devices) >= 2 and all(
            device.get("bf16") for device in devices[:2]
        ):
            summary = ", ".join(
                f"GPU {device['index']} {device['name']} ({device['total_memory'] / 2**30:.1f} GiB)"
                for device in devices[:2]
            )
            report.passed("gpu", f"{summary}; BF16 supported")
        else:
            report.fail("gpu", f"two BF16-capable CUDA GPUs are required: {torch_info}")

    source_dataset = args.source_dataset_root.expanduser().resolve()
    train_dataset = args.dataset_root.expanduser().resolve()
    if source_dataset == train_dataset:
        report.fail("dataset.separation", "training dataset must be a derived copy, not the source")
    else:
        report.passed("dataset.separation", f"source={source_dataset}; training={train_dataset}")

    if new_python.is_file():
        if not args.training_only:
            _check_dataset(
                report,
                new_python,
                source_dataset,
                "dataset.source",
                require_exact_stats=False,
                quantile_atol=args.quantile_atol,
            )
        _check_dataset(
            report,
            new_python,
            train_dataset,
            "dataset.training",
            require_exact_stats=True,
            quantile_atol=args.quantile_atol,
        )
        _check_model(
            report,
            new_python,
            args.base_checkpoint.expanduser().resolve(),
            args.tokenizer_root.expanduser().resolve(),
            config_device="cpu" if args.training_only else "cuda",
        )

    if not args.training_only:
        plugin_root = _find_plugin_root(args.plugin_root)
        _check_plugin(
            report,
            plugin_root,
            {"unitree_lerobot": old_environment, "lerobot": new_environment},
            required=args.require_plugin,
        )
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
