#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read-only four-camera reviewer for intentional-error recovery intervals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


CAMERAS = (
    ("color_0", "Global"),
    ("color_1", "Left wrist"),
    ("color_2", "Right wrist"),
    ("color_3", "Scanner"),
)


@dataclass(frozen=True)
class ReviewEpisode:
    accepted_index: int
    ordinal: int
    episode_id: int
    path: Path
    frames: list[dict]
    scan_end_exclusive: int

    @property
    def name(self) -> str:
        return self.path.name


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"no event found in {path}")


def load_review_episodes(
    raw_dir: Path,
    last_accepted: int | None = None,
    ordinal_start: int | None = None,
    ordinal_end: int | None = None,
) -> tuple[str, list[ReviewEpisode]]:
    state_path = raw_dir / "batch_state.json"
    state = read_json(state_path)
    accepted = state.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError(f"batch_state has no accepted episodes: {state_path}")
    using_range = ordinal_start is not None or ordinal_end is not None
    if using_range:
        if last_accepted is not None:
            raise ValueError("use either --last-accepted or --ordinal-start/--ordinal-end")
        if ordinal_start is None or ordinal_end is None:
            raise ValueError("--ordinal-start and --ordinal-end must be provided together")
        if ordinal_start <= 0 or ordinal_end < ordinal_start:
            raise ValueError("invalid accepted ordinal range")
        selected = [
            (index, entry)
            for index, entry in enumerate(accepted)
            if ordinal_start <= int(entry.get("ordinal", index + 1)) <= ordinal_end
        ]
        if not selected:
            raise ValueError(f"no accepted episodes in ordinal range {ordinal_start}..{ordinal_end}")
    else:
        count = 10 if last_accepted is None else last_accepted
        if count <= 0 or count > len(accepted):
            raise ValueError(f"--last-accepted must be within 1..{len(accepted)}")
        start = len(accepted) - count
        selected = list(enumerate(accepted[start:], start=start))

    episodes: list[ReviewEpisode] = []
    for accepted_index, entry in selected:
        path = Path(entry["episode_path"]).resolve()
        if not path.is_dir():
            fallback = raw_dir / f"episode_{int(entry['episode_id']):04d}"
            path = fallback.resolve()
        payload = read_json(path / "data.json")
        frames = payload.get("data")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"episode has no frames: {path}")
        event = read_first_jsonl(path / "events.jsonl")
        boundary = int(event["last_enqueued_frame_index"]) + 1
        if not 0 < boundary < len(frames):
            raise ValueError(
                f"invalid scan boundary {boundary} for {path.name} with {len(frames)} frames"
            )
        episodes.append(
            ReviewEpisode(
                accepted_index=accepted_index,
                ordinal=int(entry.get("ordinal", accepted_index + 1)),
                episode_id=int(entry["episode_id"]),
                path=path,
                frames=frames,
                scan_end_exclusive=boundary,
            )
        )
    return str(state.get("batch_id", raw_dir.name)), episodes


def exclusion_interval(
    failure_start: int,
    recovery_start: int,
    chunk_size: int,
    segment_start: int = 0,
) -> tuple[int, int]:
    if failure_start < 0:
        raise ValueError("failure start must be non-negative")
    if recovery_start <= failure_start:
        raise ValueError("recovery start must be after failure start")
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if segment_start < 0 or failure_start < segment_start:
        raise ValueError("failure start must be within the current task segment")
    return max(segment_start, failure_start - chunk_size + 1), recovery_start


def reviewed_trajectory_type(intervals: list[dict]) -> str:
    return "intentional_error_recovery" if intervals else "reviewed_normal"


def default_output_path(raw_dir: Path) -> Path:
    runtime_dir = next((parent for parent in raw_dir.parents if parent.name == "runtime"), None)
    base = runtime_dir if runtime_dir is not None else raw_dir.parent
    return base / "annotations" / raw_dir.name / "recovery_intervals.json"


class CameraPanel(QLabel):
    def __init__(self, title: str):
        super().__init__(f"{title}\nNo frame")
        self.title = title
        self.original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setStyleSheet(
            "background:#080b0a;color:#aeb9b3;border:1px solid #2b3531;font-size:16px;"
        )

    def set_image(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self.original = QPixmap()
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\nMissing frame")
            return
        self.original = QPixmap(str(path))
        self.setText("")
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if not self.original.isNull():
            self.setPixmap(
                self.original.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class RecoveryAnnotator(QMainWindow):
    def __init__(
        self,
        batch_id: str,
        episodes: list[ReviewEpisode],
        output_path: Path,
        chunk_size: int,
        fps: float,
    ):
        super().__init__()
        self.batch_id = batch_id
        self.episodes = episodes
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.fps = fps
        self.episode_index = 0
        self.pending_failure: int | None = None
        self.annotations = self._load_annotations()
        self._normalize_existing_intervals()

        self.setWindowTitle("R1-A7 recovery interval annotation (read only)")
        self.resize(1500, 980)
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#101512;color:#e7ece9;}"
            "QPushButton{background:#26332d;border:1px solid #405047;padding:9px 14px;font-size:15px;}"
            "QPushButton:hover{background:#314139;}"
            "QLabel{font-size:15px;}"
            "QListWidget{background:#0c100e;border:1px solid #2b3531;font-size:14px;}"
        )

        self.header = QLabel()
        self.header.setStyleSheet("font-size:20px;font-weight:600;padding:6px;")
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("font-size:17px;")
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color:#f0c36a;font-size:16px;padding:5px;")

        self.panels = {key: CameraPanel(title) for key, title in CAMERAS}
        camera_grid = QGridLayout()
        camera_grid.setSpacing(6)
        for index, (key, _) in enumerate(CAMERAS):
            camera_grid.addWidget(self.panels[key], index // 2, index % 2)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)

        self.intervals = QListWidget()
        self.intervals.setMinimumWidth(390)

        buttons = QHBoxLayout()
        for label, callback in (
            ("Previous episode", self.previous_episode),
            ("-1 frame", lambda: self.seek_relative(-1)),
            ("Play / pause", self.toggle_play),
            ("+1 frame", lambda: self.seek_relative(1)),
            ("Next episode", self.next_episode),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)

        marks = QHBoxLayout()
        self.failure_button = QPushButton("F: mark wrong action start")
        self.failure_button.clicked.connect(self.mark_failure)
        self.recovery_button = QPushButton("R: mark recovery start")
        self.recovery_button.clicked.connect(self.mark_recovery)
        undo = QPushButton("Undo last interval")
        undo.clicked.connect(self.undo_interval)
        save_next = QPushButton("Save and next")
        save_next.clicked.connect(self.save_and_next)
        for button in (self.failure_button, self.recovery_button, undo, save_next):
            marks.addWidget(button)

        left = QVBoxLayout()
        left.addWidget(self.header)
        left.addLayout(camera_grid, 1)
        left.addWidget(self.frame_label)
        left.addWidget(self.slider)
        left.addLayout(buttons)
        left.addLayout(marks)
        left.addWidget(self.status_label)

        right = QVBoxLayout()
        guide = QLabel(
            "1. Pause at the first intentional wrong command and press F.\n"
            "2. Move to the first deliberate recovery command and press R.\n"
            "3. Both marks must stay in the same SCAN or PLACE phase.\n"
            "4. The tool expands the exclusion by chunk_size automatically.\n"
            "Raw data is never modified."
        )
        guide.setWordWrap(True)
        guide.setStyleSheet("font-size:16px;line-height:1.4;padding:8px;")
        right.addWidget(guide)
        right.addWidget(self.intervals, 1)

        content = QHBoxLayout()
        content.addLayout(left, 1)
        content.addLayout(right)
        root = QWidget()
        root.setLayout(content)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.setInterval(max(1, round(1000 / fps)))
        self.timer.timeout.connect(lambda: self.seek_relative(1, wrap=True))

        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("F"), self, activated=self.mark_failure)
        QShortcut(QKeySequence("R"), self, activated=self.mark_recovery)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.seek_relative(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.seek_relative(1))

        saved = self.annotations.get("episodes", {})
        initial_index = next(
            (
                index
                for index, episode in enumerate(self.episodes)
                if not saved.get(episode.name, {}).get("reviewed", False)
            ),
            0,
        )
        self.load_episode(initial_index)

    def _load_annotations(self) -> dict:
        if self.output_path.is_file():
            payload = read_json(self.output_path)
            if payload.get("batch_id") != self.batch_id:
                raise ValueError(f"annotation batch mismatch: {self.output_path}")
            return payload
        return {
            "version": 1,
            "batch_id": self.batch_id,
            "fps": self.fps,
            "action_chunk_size": self.chunk_size,
            "policy": (
                "exclude every training start whose future action chunk intersects an intentional error; "
                "intervals cannot cross the scan/place task boundary"
            ),
            "episodes": {},
        }

    def _normalize_existing_intervals(self) -> None:
        by_name = {episode.name: episode for episode in self.episodes}
        for name, annotation in self.annotations.get("episodes", {}).items():
            episode = by_name.get(name)
            if episode is None:
                continue
            for interval in annotation.get("intervals", []):
                failure = int(interval["failure_action_start_frame"])
                recovery = int(interval["recovery_start_frame"])
                phase = "scan" if failure < episode.scan_end_exclusive else "place"
                segment_start = 0 if phase == "scan" else episode.scan_end_exclusive
                excluded_start, excluded_end = exclusion_interval(
                    failure, recovery, self.chunk_size, segment_start
                )
                interval.update(
                    {
                        "task_phase": phase,
                        "task_segment_start_frame": segment_start,
                        "excluded_training_start_frame": excluded_start,
                        "excluded_training_end_frame_exclusive": excluded_end,
                        "segment_local_excluded_start_frame": excluded_start - segment_start,
                        "segment_local_excluded_end_frame_exclusive": excluded_end - segment_start,
                    }
                )
            annotation["trajectory_type"] = reviewed_trajectory_type(annotation.get("intervals", []))

    @property
    def episode(self) -> ReviewEpisode:
        return self.episodes[self.episode_index]

    def episode_annotation(self) -> dict:
        episodes = self.annotations.setdefault("episodes", {})
        return episodes.setdefault(
            self.episode.name,
            {
                "accepted_index": self.episode.accepted_index,
                "ordinal": self.episode.ordinal,
                "episode_id": self.episode.episode_id,
                "scan_end_frame_exclusive": self.episode.scan_end_exclusive,
                "total_frames": len(self.episode.frames),
                "trajectory_type": "unreviewed",
                "reviewed": False,
                "intervals": [],
            },
        )

    def load_episode(self, index: int) -> None:
        self.timer.stop()
        self.pending_failure = None
        self.episode_index = max(0, min(index, len(self.episodes) - 1))
        episode = self.episode
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(episode.frames) - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.header.setText(
            f"Accepted ordinal {episode.ordinal} | {episode.name} | "
            f"review {self.episode_index + 1}/{len(self.episodes)}"
        )
        self.refresh_intervals()
        self.show_frame(0)

    def show_frame(self, frame_index: int) -> None:
        frame = self.episode.frames[frame_index]
        colors = frame.get("colors", {})
        for key, _ in CAMERAS:
            relative = colors.get(key)
            self.panels[key].set_image(self.episode.path / relative if relative else None)
        phase = "SCAN" if frame_index < self.episode.scan_end_exclusive else "PLACE"
        seconds = frame_index / self.fps
        self.frame_label.setText(
            f"Frame {frame_index}/{len(self.episode.frames) - 1} | {seconds:.2f}s | {phase} | "
            f"item_accepted boundary={self.episode.scan_end_exclusive}"
        )
        if self.pending_failure is not None:
            self.status_label.setText(
                f"Wrong action starts at frame {self.pending_failure}; seek to the first recovery command and press R."
            )
        else:
            self.status_label.setText("No pending mark. Press F at the first intentional wrong command.")

    def seek_relative(self, delta: int, wrap: bool = False) -> None:
        value = self.slider.value() + delta
        if value >= self.slider.maximum() and wrap:
            value = 0
        else:
            value = max(0, min(value, self.slider.maximum()))
        self.slider.setValue(value)

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()

    def mark_failure(self) -> None:
        self.timer.stop()
        frame = self.slider.value()
        self.pending_failure = frame
        self.show_frame(frame)

    def mark_recovery(self) -> None:
        self.timer.stop()
        if self.pending_failure is None:
            QMessageBox.warning(self, "Missing mark", "Mark the wrong action start with F first.")
            return
        recovery = self.slider.value()
        failure_phase = (
            "scan" if self.pending_failure < self.episode.scan_end_exclusive else "place"
        )
        recovery_phase = "scan" if recovery < self.episode.scan_end_exclusive else "place"
        if failure_phase != recovery_phase:
            QMessageBox.warning(
                self,
                "Invalid interval",
                "Wrong-action and recovery marks must be in the same SCAN or PLACE phase.",
            )
            return
        segment_start = 0 if failure_phase == "scan" else self.episode.scan_end_exclusive
        try:
            excluded_start, excluded_end = exclusion_interval(
                self.pending_failure, recovery, self.chunk_size, segment_start
            )
        except ValueError as error:
            QMessageBox.warning(self, "Invalid interval", str(error))
            return
        self.episode_annotation()["intervals"].append(
            {
                "failure_action_start_frame": self.pending_failure,
                "recovery_start_frame": recovery,
                "task_phase": failure_phase,
                "task_segment_start_frame": segment_start,
                "excluded_training_start_frame": excluded_start,
                "excluded_training_end_frame_exclusive": excluded_end,
                "segment_local_excluded_start_frame": excluded_start - segment_start,
                "segment_local_excluded_end_frame_exclusive": excluded_end - segment_start,
            }
        )
        self.pending_failure = None
        self.save()
        self.refresh_intervals()
        self.show_frame(recovery)

    def undo_interval(self) -> None:
        intervals = self.episode_annotation()["intervals"]
        if intervals:
            intervals.pop()
            self.save()
        self.pending_failure = None
        self.refresh_intervals()
        self.show_frame(self.slider.value())

    def refresh_intervals(self) -> None:
        annotation = self.episode_annotation()
        self.intervals.clear()
        self.intervals.addItem(
            f"{self.episode.name}: {len(annotation['intervals'])} intentional-error interval(s)"
        )
        for index, interval in enumerate(annotation["intervals"], start=1):
            self.intervals.addItem(
                f"{index}. {interval.get('task_phase', 'unknown').upper()}  "
                f"wrong={interval['failure_action_start_frame']}  "
                f"recovery={interval['recovery_start_frame']}\n"
                f"   raw exclude starts [{interval['excluded_training_start_frame']}, "
                f"{interval['excluded_training_end_frame_exclusive']})\n"
                f"   segment-local [{interval.get('segment_local_excluded_start_frame')}, "
                f"{interval.get('segment_local_excluded_end_frame_exclusive')})"
            )
        self.intervals.addItem(f"Reviewed: {annotation['reviewed']}")

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.annotations, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(self.output_path)

    def save_and_next(self) -> None:
        if self.pending_failure is not None:
            QMessageBox.warning(self, "Incomplete interval", "Finish the pending interval with R first.")
            return
        annotation = self.episode_annotation()
        if not annotation["intervals"]:
            answer = QMessageBox.question(
                self,
                "No interval",
                "No intentional-error interval was marked. Save this episode as reviewed anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        annotation["reviewed"] = True
        annotation["trajectory_type"] = reviewed_trajectory_type(annotation["intervals"])
        self.save()
        self.refresh_intervals()
        if self.episode_index + 1 < len(self.episodes):
            self.load_episode(self.episode_index + 1)
        else:
            QMessageBox.information(self, "Review complete", f"Annotations saved to:\n{self.output_path}")

    def previous_episode(self) -> None:
        self.load_episode(self.episode_index - 1)

    def next_episode(self) -> None:
        self.load_episode(self.episode_index + 1)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.save()
        super().closeEvent(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--last-accepted", type=int)
    parser.add_argument("--ordinal-start", type=int, help="First accepted ordinal to review, inclusive")
    parser.add_argument("--ordinal-end", type=int, help="Last accepted ordinal to review, inclusive")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = args.raw_dir.resolve()
    batch_id, episodes = load_review_episodes(
        raw_dir,
        last_accepted=args.last_accepted,
        ordinal_start=args.ordinal_start,
        ordinal_end=args.ordinal_end,
    )
    output = args.output.resolve() if args.output else default_output_path(raw_dir)
    app = QApplication(sys.argv)
    window = RecoveryAnnotator(batch_id, episodes, output, args.chunk_size, args.fps)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
