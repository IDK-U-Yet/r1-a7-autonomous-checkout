# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Bounded audit recorder and expert-only raw export (no hardware imports)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid
import cv2
import numpy as np
from unitree_lerobot.r1a7_contract import CAMERA_KEYS, validate_action_vector, validate_state_vector


def parts(vector):
    return {"left_arm": {"qpos": vector[:7]}, "right_arm": {"qpos": vector[7:14]},
            "left_ee": {"qpos": vector[14:15]}, "right_ee": {"qpos": vector[15:16]}}


def collection_path(root, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name or ""):
        raise ValueError("collection-name must be a safe explicit batch name, not a path")
    return Path(root) / name


def collection_name(dataset_root, override=None):
    """Default to the supplied dataset folder label; retain explicit overrides."""
    name = override if override is not None else Path(dataset_root).name
    collection_path(Path("."), name)  # Apply the same path safety contract.
    return name


class DaggerRecorder:
    def __init__(self, root, checkpoint, *, fps=15, queue_size=30, source_metadata=None):
        if fps <= 0 or queue_size <= 0:
            raise ValueError("fps and queue_size must be positive")
        self.trial_id = "dagger_" + uuid.uuid4().hex
        self.path = Path(root) / self.trial_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.storage_root = self.path.parent.resolve()
        trial_stat = self.path.stat()
        self.directory_identity = (trial_stat.st_dev, trial_stat.st_ino)
        (self.path / "colors").mkdir()
        self.checkpoint = str(checkpoint)
        self.source_metadata = dict(source_metadata or {})
        self.fps = fps
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.count = 0
        self.closed = False
        self.finalized = False
        self.continuity = 0
        self.worker = threading.Thread(target=self._write, daemon=True)
        self.worker.start()
        self.event("trial_started", checkpoint=self.checkpoint)

    def discard(self):
        """Delete only this recorder's closed, unchanged UUID trial directory."""
        if not self.closed or self.worker.is_alive():
            raise RuntimeError("cannot delete a trial while its writer is active")
        if not self.finalized or self.error is not None:
            raise RuntimeError("cannot delete a trial whose finalization failed")
        if (self.path.is_symlink() or self.path.parent.resolve() != self.storage_root
                or self.path.name != self.trial_id
                or not re.fullmatch(r"dagger_[0-9a-f]{32}", self.trial_id)):
            raise RuntimeError("refusing deletion outside the current trial directory")
        trial_stat = self.path.stat()
        if (trial_stat.st_dev, trial_stat.st_ino) != self.directory_identity:
            raise RuntimeError("current trial directory was replaced; refusing deletion")
        shutil.rmtree(self.path)

    def check(self):
        if self.error is not None:
            raise RuntimeError("DAgger recording failed; stop hardware") from self.error
        if self.closed:
            raise RuntimeError("DAgger recorder closed")

    def event(self, event, **fields):
        self.check()
        payload = {"event": event, "source_trial_id": self.trial_id,
                   "monotonic_ns": time.monotonic_ns(), **fields}
        try:
            self.queue.put_nowait(("event", payload))
        except queue.Full as exc:
            self.error = exc
            raise RuntimeError("recorder event queue full; refusing silent event loss") from exc

    def gap(self, reason):
        self.continuity += 1
        self.event("recording_gap", reason=str(reason), continuity_id=self.continuity)

    def add(self, state, images, action, metadata):
        self.check()
        state = validate_state_vector(state).tolist()
        action = validate_action_vector(action).tolist()
        if set(images) != set(CAMERA_KEYS):
            raise ValueError("four cameras required")
        frame = {"idx": self.count, "states": parts(state), "actions": parts(action),
                 "metadata": {**self.source_metadata, **metadata, "source_trial_id": self.trial_id,
                              "checkpoint": self.checkpoint, "continuity_id": self.continuity}}
        pixels = {k: np.asarray(v).copy() for k, v in images.items()}
        try:
            self.queue.put_nowait((frame, pixels))
        except queue.Full as exc:
            self.error = exc
            raise RuntimeError("recorder queue full; refusing silent data loss") from exc
        self.count += 1

    def _write(self):
        try:
            with (self.path / "frames.jsonl").open("x", encoding="utf-8") as output, \
                    (self.path / "events.jsonl").open("x", encoding="utf-8") as events:
                while True:
                    if self.error is not None:
                        break
                    item = self.queue.get()
                    if item is None:
                        break
                    if isinstance(item[0], str) and item[0] == "event":
                        events.write(json.dumps(item[1]) + "\n")
                        continue
                    frame, images = item
                    frame["colors"] = {}
                    for i, key in enumerate(CAMERA_KEYS):
                        relative = f"colors/{frame['idx']:07d}_color_{i}.jpg"
                        rgb_chw = images[key]
                        bgr = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0)[:, :, ::-1])
                        if not cv2.imwrite(str(self.path / relative), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise OSError(f"failed writing {relative}")
                        frame["colors"][f"color_{i}"] = relative
                    output.write(json.dumps(frame) + "\n")
                output.flush()
                os.fsync(output.fileno())
                events.flush()
                os.fsync(events.fileno())
        except Exception as exc:
            self.error = exc

    def finish(self, *, completed=False, accepted_event_id=None):
        if self.closed:
            return
        # Bounded wait: if storage is stuck leave a clearly incomplete audit.
        deadline = time.monotonic() + 10
        finished = ("event", {"event": "trial_finished", "source_trial_id": self.trial_id,
                    "monotonic_ns": time.monotonic_ns(), "completed": completed,
                    "accepted_event_id": accepted_event_id})
        pending = [finished, None]
        while self.worker.is_alive():
            if self.error is not None:
                break
            try:
                self.queue.put(pending[0], timeout=0.1)
                pending.pop(0)
                if not pending:
                    break
            except queue.Full:
                if time.monotonic() >= deadline:
                    self.error = TimeoutError("recorder did not drain in 10 seconds")
                    break
        self.worker.join(timeout=max(0, deadline - time.monotonic()))
        if self.worker.is_alive() and self.error is None:
            self.error = TimeoutError("recorder storage worker stuck")
        self.closed = True
        if self.error is not None:
            raise RuntimeError(f"incomplete recording retained at {self.path}") from self.error
        frames = [json.loads(line) for line in (self.path / "frames.jsonl").read_text().splitlines()]
        if len(frames) != self.count:
            raise RuntimeError("recorder frame count mismatch")
        manifest = {**self.source_metadata, "source_trial_id": self.trial_id, "checkpoint": self.checkpoint,
                    "completed": bool(completed), "accepted_event_id": accepted_event_id,
                    "frames": self.count, "fps": self.fps, "schema": "r1a7_hg_dagger_v1"}
        payload = {"info": {"image": {"fps": self.fps}, "dagger_audit_only": True}, "text": {}, "data": frames}
        for name, value in (("data.json", payload), ("manifest.json", manifest)):
            temp = self.path / (name + ".partial")
            with temp.open("x", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(self.path / name)
        self.finalized = True


def expert_segments(frames, *, max_gap_ns=120_000_000):
    segments, active = [], []
    previous = None
    for frame in frames:
        meta = frame["metadata"]
        key = (meta.get("owner_epoch"), meta.get("active_task"), meta.get("continuity_id"))
        stamp = int(meta["record_monotonic_ns"])
        valid = (meta.get("control_source") == "human" and meta.get("sample_valid") is True
                 and meta.get("expert_action_valid", True) is True
                 and meta.get("active_task") and meta.get("owner_epoch") is not None)
        consecutive = (previous is not None and key == previous[0]
                       and 0 < stamp - previous[1] <= max_gap_ns)
        if active and (not valid or not consecutive):
            segments.append(active)
            active = []
        if valid:
            active.append(frame)
        previous = (key, stamp) if valid else None
    if active:
        segments.append(active)
    return segments


def export_expert(trials_root, output, *, min_frames=50):
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError("expert output exists; choose a new directory")
    trial_paths = sorted(Path(trials_root).glob("dagger_*/manifest.json"))
    if not trial_paths:
        raise ValueError("no finalized DAgger trials")
    output.mkdir(parents=True)
    mapping = []
    for path in trial_paths:
        manifest = json.loads(path.read_text())
        if not manifest["completed"] or not manifest.get("accepted_event_id"):
            continue
        raw = json.loads((path.parent / "data.json").read_text())
        for frames in expert_segments(raw["data"]):
            if len(frames) < min_frames:
                continue
            episode = output / f"episode_{len(mapping):04d}"
            episode.mkdir()
            (episode / "colors").mkdir()
            exported = []
            for idx, original in enumerate(frames):
                frame = json.loads(json.dumps(original))
                frame["metadata"]["source_frame_idx"] = frame["idx"]
                frame["idx"] = idx
                for relative in frame["colors"].values():
                    src, dst = path.parent / relative, episode / relative
                    if not src.is_file():
                        raise FileNotFoundError(src)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copyfile(src, dst)
                exported.append(frame)
            with (episode / "data.json").open("x") as stream:
                info = {**raw["info"], "dagger_audit_only": False, "dagger_expert_export": True}
                json.dump({"info": info, "text": raw.get("text", {}), "data": exported}, stream)
            mapping.append({"episode": episode.name, "source_trial_id": manifest["source_trial_id"],
                            "source_frame_start": frames[0]["idx"], "frames": len(frames),
                            "task": frames[0]["metadata"]["active_task"]})
    with (output / "expert_export.json").open("x") as stream:
        json.dump({"segments": mapping, "split_group_key": "source_trial_id",
                   "note": "Group all sibling segments by source_trial_id when splitting."}, stream, indent=2)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Export only completed, scanner-accepted HUMAN suffixes")
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=50)
    args = parser.parse_args()
    result = export_expert(args.trials_root, args.output, min_frames=args.min_frames)
    print(json.dumps({"expert_segments": len(result), "output": str(args.output)}))


if __name__ == "__main__":
    main()
