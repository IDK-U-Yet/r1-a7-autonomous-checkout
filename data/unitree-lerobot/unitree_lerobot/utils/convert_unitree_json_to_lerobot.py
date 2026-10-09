# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \
    --push_to_hub
"""

import argparse
import dataclasses
import importlib.metadata
import inspect
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tqdm

from lerobot.utils.constants import HF_LEROBOT_HOME

try:
    from lerobot.datasets import LeRobotDataset
except ImportError:  # Keep dataset-inspection unit tests importable with the legacy vendored source.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.utils.constants import ROBOT_CONFIGS
from unitree_lerobot.utils.reviewed_dagger import validate_reviewed_export


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


EXPECTED_LEROBOT_VERSION = "0.6.2"


def assert_lerobot_version(version: str) -> None:
    if version != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"This converter requires LeRobot {EXPECTED_LEROBOT_VERSION}, got {version}. "
            "Run it with the project-pinned runtime/envs/pi05-262d2fd environment."
        )


def verify_lerobot_runtime() -> None:
    try:
        installed_version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("LeRobot is not installed in the active Python environment.") from exc

    assert_lerobot_version(installed_version)

    module_path = Path(inspect.getfile(LeRobotDataset)).resolve()
    vendored_source = Path(__file__).resolve().parents[1] / "lerobot"
    try:
        module_path.relative_to(vendored_source.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "LeRobot 0.6.2 metadata is installed, but Python imported the repository's "
            f"legacy vendored source: {module_path}. Remove that source from PYTHONPATH."
        )


def discover_episode_jsons(raw_dir: Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    direct_episodes = sorted(
        path for path in raw_dir.glob("*/data.json") if path.parent.name != "rejected"
    )
    if direct_episodes:
        return direct_episodes
    return sorted(
        path
        for path in raw_dir.glob("*/*/data.json")
        if "rejected" not in path.relative_to(raw_dir).parts
    )


def infer_dataset_fps(raw_dir: Path) -> int:
    json_paths = discover_episode_jsons(raw_dir)
    if not json_paths:
        raise ValueError(f"No episode data.json files found under {raw_dir}")

    frequencies = set()
    for json_path in json_paths:
        with json_path.open(encoding="utf-8") as stream:
            episode = json.load(stream)
        frequency = episode.get("info", {}).get("image", {}).get("fps")
        if frequency is None:
            raise ValueError(f"Missing info.image.fps in {json_path}")
        frequency = float(frequency)
        if frequency <= 0 or not frequency.is_integer():
            raise ValueError(f"Invalid integer FPS {frequency!r} in {json_path}")
        frequencies.add(int(frequency))

    if len(frequencies) != 1:
        raise ValueError(f"Episodes use inconsistent FPS values: {sorted(frequencies)}")
    return frequencies.pop()


class JsonDataset:
    def __init__(
        self,
        data_dirs: Path,
        robot_type: str,
        episode_annotations_path: Path | None = None,
    ) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.

        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = "data.json"

        # Initialize paths and cache
        self._init_paths()
        self.episode_annotations = (
            load_episode_annotations(episode_annotations_path, self.data_dirs, self.episode_paths)
            if episode_annotations_path is not None
            else {}
        )
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

    def _init_paths(self) -> None:
        """Initialize episode and task paths."""
        json_paths = discover_episode_jsons(Path(self.data_dirs))
        self.episode_paths = [str(path.parent) for path in json_paths]
        self.task_paths = sorted({str(path.parent.parent) for path in json_paths})
        self.episode_ids = list(range(len(self.episode_paths)))

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.segment_refs)

    def _init_cache(self) -> list:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            with open(json_path, encoding="utf-8") as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        self.segment_refs = []
        reviewed_paths = validate_reviewed_export(self.data_dirs, self.episode_paths)
        for episode_index, episode_data in enumerate(self.episodes_data_cached):
            info = episode_data.get("info") or {}
            reviewed = str(Path(self.episode_paths[episode_index]).resolve().relative_to(self.data_dirs.resolve()) / "data.json") in reviewed_paths
            if info.get("dagger_reviewed_export") and not reviewed:
                raise ValueError("Incomplete reviewed DAgger export: missing approval manifest")
            if info.get("dagger_audit_only"):
                raise ValueError("DAgger audit rollout cannot be trained directly; export HUMAN segments first")
            if info.get("dagger_expert_export") and not (self.data_dirs / "expert_export.json").is_file():
                raise ValueError("Incomplete DAgger expert export: missing expert_export.json")
            samples = episode_data.get("data") or []
            if not samples:
                raise ValueError(f"Episode contains no samples: {self.episode_paths[episode_index]}")
            if not reviewed and any("control_source" in (sample.get("metadata") or {}) for sample in samples):
                if not all((s.get("metadata") or {}).get("control_source") == "human"
                           and (s.get("metadata") or {}).get("sample_valid") is True for s in samples):
                    raise ValueError("Only validated HUMAN DAgger targets may enter training")

            active_tasks = [
                (sample.get("metadata") or {}).get("active_task")
                for sample in samples
            ]
            has_task_labels = any(task is not None for task in active_tasks)
            if has_task_labels and any(
                not isinstance(task, str) or not task.strip() for task in active_tasks
            ):
                raise ValueError(
                    f"active_task must be present on every frame: {self.episode_paths[episode_index]}"
                )

            if not has_task_labels:
                active_tasks = [episode_data.get("text", {}).get("goal", "")] * len(samples)

            start = 0
            task_runs = []
            for sample_index in range(1, len(active_tasks) + 1):
                if (
                    sample_index == len(active_tasks)
                    or active_tasks[sample_index] != active_tasks[start]
                ):
                    task_runs.append((start, sample_index, active_tasks[start]))
                    start = sample_index
            source_episode_path = str(Path(self.episode_paths[episode_index]).resolve())
            annotation = self.episode_annotations.get(source_episode_path)
            if annotation is not None:
                if len(task_runs) != 2:
                    raise ValueError(
                        "Episode annotations require exactly one item_accepted task transition: "
                        f"{self.episode_paths[episode_index]}"
                    )
                task_runs[0] = (
                    task_runs[0][0],
                    task_runs[0][1],
                    annotation["task_before_accept"],
                )

            if len(task_runs) > 2:
                raise ValueError(
                    f"Expected at most one task transition, found {len(task_runs) - 1}: "
                    f"{self.episode_paths[episode_index]}"
                )
            if len({task for _, _, task in task_runs}) != len(task_runs):
                raise ValueError(
                    f"A task label reappears after a transition: {self.episode_paths[episode_index]}"
                )
            for start, end, task in task_runs:
                # Reviewed pause cuts stay in the same raw/replay episode, but
                # must not share a future action chunk across the edit boundary.
                boundaries = [start]
                if reviewed:
                    for i in range(start + 1, end):
                        current = samples[i].get("metadata") or {}
                        previous = samples[i - 1].get("metadata") or {}
                        if current.get("edit_discontinuity_before") or current.get("continuity_id") != previous.get("continuity_id"):
                            boundaries.append(i)
                boundaries.append(end)
                for segment_start, segment_end in zip(boundaries, boundaries[1:]):
                    self._append_segment(episode_index, samples, segment_start, segment_end, task, annotation, reviewed)

        print(
            f"==> Cached {len(self.episodes_data_cached)} raw episodes; "
            f"prepared {len(self.segment_refs)} task episodes"
        )
        return self.episodes_data_cached

    def _append_segment(self, episode_index, samples, start, end, task, annotation, reviewed):
        segment = {
            "source_episode_index": episode_index,
            "source_episode_path": self.episode_paths[episode_index],
            "start_frame": start,
            "end_frame_exclusive": end,
            "task": task,
        }
        parent_trial = (samples[start].get("metadata") or {}).get("source_trial_id")
        if parent_trial:
            segment["source_trial_id"] = parent_trial
        if reviewed:
            segment["human_reviewed_export"] = True
            segment["continuity_id"] = (samples[start].get("metadata") or {}).get("continuity_id")
            segment["boundary_policy"] = "no_cross_boundary_action_chunks"
        if annotation is not None:
            segment["source_episode_id"] = annotation["episode_id"]
            segment["episode_annotation"] = annotation
        self.segment_refs.append(segment)

    def _extract_data(self, episode_data: dict, key: str, parts: list[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.

        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')

        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data["data"]:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                key_parts = part.split(".")
                qpos = None
                for key_part in key_parts:
                    if qpos is None and key_part in sample_data[key] and sample_data[key][key_part] is not None:
                        qpos = sample_data[key][key_part]
                    else:
                        if qpos is None:
                            raise ValueError(f"qpos is None for part: {part}")
                        qpos = qpos[key_part]
                if qpos is None:
                    raise ValueError(f"qpos is None for part: {part}")
                if isinstance(qpos, list):
                    qpos = np.array(qpos, dtype=np.float32).flatten()
                else:
                    qpos = np.array([qpos], dtype=np.float32).flatten()
                data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)

    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        keys = episode_data["data"][0]["colors"].keys()
        cameras = [key for key in keys if "depth" not in key]

        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data["data"]:
                relative_path = sample_data["colors"].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images

    def get_item(
        self,
        index: int | None = None,
    ) -> dict:
        """Get a training sample from the dataset."""

        if index is None:
            index = int(np.random.randint(len(self.segment_refs)))
        segment = self.segment_refs[index]
        source_episode_index = segment["source_episode_index"]
        file_path = self.episode_paths[source_episode_index]
        source_episode = self.episodes_data_cached[source_episode_index]
        episode_data = {
            **source_episode,
            "data": source_episode["data"][
                segment["start_frame"]:segment["end_frame_exclusive"]
            ],
        }

        # Load state and action data
        action = self._extract_data(episode_data, "actions", self.json_action_data_name)
        state = self._extract_data(episode_data, "states", self.json_state_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = segment["task"]

        # Load camera images
        cameras = self._parse_images(file_path, episode_data)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }

        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "task": task,
            "data_cfg": data_cfg,
            "source_segment": dict(segment),
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    fps: int = 30,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    robot_config = ROBOT_CONFIGS[robot_type]
    motors = robot_config.motors
    cameras = robot_config.cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        camera_shape = robot_config.camera_shapes.get(cam, (480, 640, 3))
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": camera_shape,
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    *,
    episode_annotations_path: Path | None = None,
) -> LeRobotDataset:
    json_dataset = JsonDataset(raw_dir, robot_type, episode_annotations_path)
    source_segments = []
    for output_episode_index in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(output_episode_index)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for frame_index in range(num_frames):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[frame_index]

            frame["task"] = task

            dataset.add_frame(frame)
        dataset.save_episode()
        source_segments.append(
            {
                "output_episode_index": output_episode_index,
                **episode["source_segment"],
            }
        )

    manifest_path = Path(dataset.root) / "meta" / "source_segments.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as stream:
        for segment in source_segments:
            stream.write(json.dumps(segment, ensure_ascii=False) + "\n")

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  # e.g., Unitree_Z1_Single, Unitree_Z1_Dual, Unitree_G1_Dex1, Unitree_G1_Dex3, Unitree_G1_Brainco, Unitree_G1_Inspire
    *,
    fps: int | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    episode_annotations_path: Path | None = None,
):
    verify_lerobot_runtime()
    effective_fps = infer_dataset_fps(raw_dir) if fps is None else fps
    if effective_fps <= 0:
        raise ValueError("fps must be positive")

    # Validate annotations and task boundaries before replacing an existing output.
    if episode_annotations_path is not None:
        JsonDataset(raw_dir, robot_type, episode_annotations_path)

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        fps=effective_fps,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        episode_annotations_path=episode_annotations_path,
    )
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


def local_push_to_hub(
    repo_id: str,
    root_path: Path,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Unitree JSON episodes to LeRobot v3 format.")
    parser.add_argument("--raw-dir", "--raw_dir", required=True, type=Path)
    parser.add_argument("--repo-id", "--repo_id", required=True)
    parser.add_argument("--robot-type", "--robot_type", required=True)
    parser.add_argument(
        "--episode-annotations",
        type=Path,
        help="JSONL sidecar with one post-capture annotation per accepted raw episode.",
    )
    parser.add_argument("--fps", type=int)
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true")
    parser.add_argument("--mode", choices=("video", "image"), default="video")
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DATASET_CONFIG.use_videos,
    )
    parser.add_argument("--tolerance-s", type=float, default=DEFAULT_DATASET_CONFIG.tolerance_s)
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=DEFAULT_DATASET_CONFIG.image_writer_processes,
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=DEFAULT_DATASET_CONFIG.image_writer_threads,
    )
    parser.add_argument("--video-backend", default=DEFAULT_DATASET_CONFIG.video_backend)
    args = parser.parse_args()

    dataset_config = DatasetConfig(
        use_videos=args.use_videos,
        tolerance_s=args.tolerance_s,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
    )
    json_to_lerobot(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        mode=args.mode,
        dataset_config=dataset_config,
        episode_annotations_path=args.episode_annotations,
    )



def load_episode_annotations(annotation_path: Path, raw_dir: Path, episode_paths: list[str]) -> dict[str, dict]:
    """Load a complete JSONL annotation manifest keyed to accepted raw episodes."""
    annotation_path = Path(annotation_path)
    raw_dir = Path(raw_dir).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Episode annotation manifest not found: {annotation_path}")

    paths_by_id: dict[str, list[str]] = defaultdict(list)
    paths_by_relative: dict[str, str] = {}
    for episode_path in episode_paths:
        resolved_path = Path(episode_path).resolve()
        relative_path = resolved_path.relative_to(raw_dir).as_posix()
        paths_by_id[resolved_path.name].append(str(resolved_path))
        paths_by_relative[relative_path] = str(resolved_path)

    annotations: dict[str, dict] = {}
    with annotation_path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on annotation line {line_number}: {exc.msg}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"Annotation line {line_number} must be a JSON object.")

            raw_episode_id = entry.get("episode_id")
            if isinstance(raw_episode_id, bool):
                raise ValueError(f"Annotation line {line_number}: invalid episode_id.")
            if isinstance(raw_episode_id, int):
                episode_id = f"episode_{raw_episode_id:04d}"
            elif isinstance(raw_episode_id, str):
                episode_id = raw_episode_id.strip()
                if episode_id.isdigit():
                    episode_id = f"episode_{int(episode_id):04d}"
            else:
                raise ValueError(
                    f"Annotation line {line_number}: episode_id must be an integer or episode_NNNN."
                )
            suffix = episode_id.removeprefix("episode_") if episode_id.startswith("episode_") else ""
            if not suffix.isdigit():
                raise ValueError(
                    f"Annotation line {line_number}: episode_id must look like 'episode_0000'."
                )

            episode_type = entry.get("episode_type")
            task_before_accept = entry.get("task_before_accept")
            if not isinstance(episode_type, str) or not episode_type.strip():
                raise ValueError(f"Annotation line {line_number}: episode_type is required.")
            if not isinstance(task_before_accept, str) or not task_before_accept.strip():
                raise ValueError(f"Annotation line {line_number}: task_before_accept is required.")

            episode_path_value = entry.get("episode_path")
            if episode_path_value is not None:
                if not isinstance(episode_path_value, str) or not episode_path_value.strip():
                    raise ValueError(
                        f"Annotation line {line_number}: episode_path must be a relative directory path."
                    )
                relative_path = Path(episode_path_value)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"Annotation line {line_number}: episode_path must stay under raw-dir.")
                normalized_path = relative_path.as_posix().rstrip("/")
                resolved_episode = paths_by_relative.get(normalized_path)
                if resolved_episode is None:
                    raise ValueError(
                        f"Annotation line {line_number}: episode_path {normalized_path!r} "
                        "does not match an accepted episode."
                    )
                if Path(resolved_episode).name != episode_id:
                    raise ValueError(f"Annotation line {line_number}: episode_id does not match episode_path.")
            else:
                matches = paths_by_id.get(episode_id, [])
                if len(matches) > 1:
                    raise ValueError(
                        f"Annotation line {line_number}: {episode_id} is ambiguous; "
                        "provide episode_path relative to raw-dir."
                    )
                if not matches:
                    raise ValueError(
                        f"Annotation line {line_number}: {episode_id} does not match an accepted episode."
                    )
                resolved_episode = matches[0]

            if resolved_episode in annotations:
                raise ValueError(f"Duplicate annotation for {episode_id}.")
            labels = entry.get("labels", {})
            if not isinstance(labels, dict):
                raise ValueError(f"Annotation line {line_number}: labels must be a JSON object.")

            annotation = {
                "episode_id": episode_id,
                "episode_type": episode_type.strip(),
                "task_before_accept": task_before_accept.strip(),
                "labels": labels,
            }
            if episode_path_value is not None:
                annotation["episode_path"] = Path(episode_path_value).as_posix().rstrip("/")
            annotations[resolved_episode] = annotation

    missing = [
        Path(path).resolve().relative_to(raw_dir).as_posix()
        for path in episode_paths
        if str(Path(path).resolve()) not in annotations
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"Episode annotation manifest is incomplete: {len(missing)} accepted episode(s) "
            f"have no annotation: {preview}{suffix}"
        )
    return annotations


if __name__ == "__main__":
    try:
        verify_lerobot_runtime()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    main()
