# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Persistent batch state for supervised cashier data collection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


STATE_VERSION = 1
VALID_STATES = {
    "ready",
    "recording",
    "saving",
    "aborting",
    "resetting",
    "complete",
}


class CollectionBatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectionSlot:
    product_index: int
    barcode: str
    sku: str
    name: str
    ordinal: int
    target_episodes: int

    def as_dict(self):
        return {
            "product_index": self.product_index,
            "barcode": self.barcode,
            "sku": self.sku,
            "name": self.name,
            "ordinal": self.ordinal,
            "target_episodes": self.target_episodes,
        }


def _load_plan(path):
    path = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise CollectionBatchError("collection plan version must be 1")
    products = raw.get("products")
    if not isinstance(products, list) or not products:
        raise CollectionBatchError("collection plan requires a non-empty products list")
    seen_barcodes = set()
    normalized_products = []
    for index, product in enumerate(products):
        if not isinstance(product, dict):
            raise CollectionBatchError(f"products[{index}] must be an object")
        barcode = str(product.get("barcode", "")).strip()
        sku = str(product.get("sku", "")).strip()
        name = str(product.get("name", "")).strip()
        target = product.get("target_episodes")
        if not barcode or not sku or not name:
            raise CollectionBatchError(
                f"products[{index}] requires barcode, sku and name"
            )
        if barcode in seen_barcodes:
            raise CollectionBatchError(f"duplicate barcode in collection plan: {barcode}")
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise CollectionBatchError(
                f"products[{index}].target_episodes must be a positive integer"
            )
        seen_barcodes.add(barcode)
        normalized_products.append(
            {
                "barcode": barcode,
                "sku": sku,
                "name": name,
                "target_episodes": target,
            }
        )
    reset_seconds = raw.get("reset_seconds", 30.0)
    if (
        isinstance(reset_seconds, bool)
        or not isinstance(reset_seconds, (int, float))
        or not math.isfinite(reset_seconds)
        or reset_seconds < 0
    ):
        raise CollectionBatchError("reset_seconds must be finite and non-negative")
    plan = {
        "version": 1,
        "batch_id": str(raw.get("batch_id", path.stem)).strip() or path.stem,
        "reset_seconds": float(reset_seconds),
        "products": normalized_products,
    }
    selection_mode = raw.get("selection_mode")
    if selection_mode is not None:
        selection_mode = str(selection_mode).strip().lower()
        if selection_mode not in {"ordered", "flexible"}:
            raise CollectionBatchError(
                "collection plan selection_mode must be ordered or flexible"
            )
        plan["selection_mode"] = selection_mode
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return path, plan, hashlib.sha256(canonical).hexdigest()


def _plan_sha256(plan):
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class CollectionBatch:
    """Track accepted attempts and restore progress after a process restart."""

    def __init__(self, task_dir, plan_path, *, reset_seconds=None, wall_clock=time.time):
        self.task_dir = Path(task_dir).expanduser().resolve()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.plan_path, self.plan, self.plan_sha256 = _load_plan(plan_path)
        self.wall_clock = wall_clock
        if reset_seconds is not None:
            if not math.isfinite(reset_seconds) or reset_seconds < 0:
                raise CollectionBatchError(
                    "reset seconds override must be finite and non-negative"
                )
            self.plan["reset_seconds"] = float(reset_seconds)
            self.plan_sha256 = _plan_sha256(self.plan)
        self.state_path = self.task_dir / "batch_state.json"
        self.rejected_dir = self.task_dir / "rejected"
        self._resumed = self.state_path.exists()
        if self._resumed:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._validate_state()
            self._reconcile_episode_paths()
            self._recover_interrupted_attempt()
        else:
            legacy = sorted(self.task_dir.glob("episode_[0-9][0-9][0-9][0-9]"))
            if legacy:
                raise CollectionBatchError(
                    "batch_state.json is missing but episode directories already exist; "
                    "archive the old batch before starting a managed collection"
                )
            self.state = {
                "version": STATE_VERSION,
                "batch_id": self.plan["batch_id"],
                "plan_path": str(self.plan_path),
                "plan_sha256": self.plan_sha256,
                "status": "ready",
                "active_attempt": None,
                "accepted": [],
                "rejected": [],
                "last_finalized": None,
                "reset_deadline_wall_time_ns": None,
                "created_wall_time_ns": self._wall_time_ns(),
                "updated_wall_time_ns": self._wall_time_ns(),
            }
            self._save()
        self.refresh()

    @property
    def resumed(self):
        return self._resumed

    @contextmanager
    def transaction(self):
        """Serialize recorder state changes with incoming scanner events."""
        with self._lock:
            yield

    @property
    def status(self):
        return self.state["status"]

    @property
    def total_target(self):
        return sum(product["target_episodes"] for product in self.plan["products"])

    @property
    def accepted_count(self):
        return len(self.state["accepted"])

    @property
    def rejected_count(self):
        return len(self.state["rejected"])

    @property
    def selection_mode(self):
        return self.plan.get("selection_mode", "ordered")

    def _validate_state(self):
        if self.state.get("version") != STATE_VERSION:
            raise CollectionBatchError("unsupported batch_state.json version")
        if self.state.get("plan_sha256") != self.plan_sha256:
            raise CollectionBatchError(
                "collection plan differs from the plan recorded in batch_state.json"
            )
        if self.state.get("status") not in VALID_STATES:
            raise CollectionBatchError("batch_state.json contains an invalid status")
        if not isinstance(self.state.get("accepted"), list) or not isinstance(
            self.state.get("rejected"), list
        ):
            raise CollectionBatchError("batch_state.json history is invalid")

    def _save(self):
        self.state["updated_wall_time_ns"] = self._wall_time_ns()
        temporary = self.state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.state_path)

    def _wall_time_ns(self):
        return int(self.wall_clock() * 1_000_000_000)

    def _move_to_rejected(self, attempt, reason):
        source_text = attempt.get("episode_path")
        rejected_path = None
        if source_text:
            source = Path(source_text)
            if source.exists():
                self.rejected_dir.mkdir(parents=True, exist_ok=True)
                destination = self.rejected_dir / source.name
                if destination.exists():
                    raise CollectionBatchError(
                        f"rejected episode destination already exists: {destination}"
                    )
                shutil.move(str(source), str(destination))
                rejected_path = str(destination)
            else:
                recovered_destination = self.rejected_dir / source.name
                if recovered_destination.exists():
                    rejected_path = str(recovered_destination)
        record = {
            **attempt,
            "status": "rejected",
            "rejection_reason": reason,
            "rejected_wall_time_ns": self._wall_time_ns(),
        }
        if rejected_path is not None:
            record["episode_path"] = rejected_path
        self.state["rejected"].append(record)
        return record

    def _recover_interrupted_attempt(self):
        if self.state["status"] not in {"recording", "saving", "aborting"}:
            return
        attempt = self.state.get("active_attempt")
        if attempt:
            self._move_to_rejected(attempt, "interrupted_before_batch_commit")
        self.state["active_attempt"] = None
        self.state["status"] = "ready"
        self.state["reset_deadline_wall_time_ns"] = None
        self._save()

    def _reconcile_episode_paths(self):
        """Repair a crash between moving an episode and committing batch_state.json."""
        changed = False
        accepted = []
        for attempt in self.state["accepted"]:
            source = Path(attempt["episode_path"])
            if source.exists():
                accepted.append(attempt)
                continue
            rejected_path = self.rejected_dir / source.name
            if not rejected_path.exists():
                raise CollectionBatchError(
                    f"accepted episode directory is missing: {source}"
                )
            recovered = {
                **attempt,
                "status": "rejected",
                "episode_path": str(rejected_path),
                "rejection_reason": "recovered_interrupted_operator_rejection",
                "rejected_wall_time_ns": self._wall_time_ns(),
            }
            self.state["rejected"].append(recovered)
            if (
                (self.state.get("last_finalized") or {}).get("episode_id")
                == attempt.get("episode_id")
            ):
                self.state["last_finalized"] = {
                    "episode_id": attempt["episode_id"],
                    "status": "rejected",
                }
            changed = True
        self.state["accepted"] = accepted

        for attempt in self.state["rejected"]:
            source = Path(attempt["episode_path"])
            if source.exists():
                continue
            recovered_path = self.rejected_dir / source.name
            if recovered_path.exists():
                attempt["episode_path"] = str(recovered_path)
                changed = True
        if changed:
            self._save()

    def _counts_by_barcode(self):
        counts = {product["barcode"]: 0 for product in self.plan["products"]}
        for attempt in self.state["accepted"]:
            barcode = attempt.get("barcode")
            if barcode in counts:
                counts[barcode] += 1
        return counts

    def next_slot(self):
        counts = self._counts_by_barcode()
        for index, product in enumerate(self.plan["products"]):
            count = counts[product["barcode"]]
            if count < product["target_episodes"]:
                return CollectionSlot(
                    product_index=index,
                    barcode=product["barcode"],
                    sku=product["sku"],
                    name=product["name"],
                    ordinal=count + 1,
                    target_episodes=product["target_episodes"],
                )
        return None

    def _slot_for_barcode(self, barcode):
        barcode = str(barcode).strip()
        counts = self._counts_by_barcode()
        for index, product in enumerate(self.plan["products"]):
            if product["barcode"] != barcode:
                continue
            count = counts[barcode]
            if count >= product["target_episodes"]:
                raise CollectionBatchError(
                    f"product quota is complete for barcode {barcode}: "
                    f"{count}/{product['target_episodes']}"
                )
            return CollectionSlot(
                product_index=index,
                barcode=barcode,
                sku=product["sku"],
                name=product["name"],
                ordinal=count + 1,
                target_episodes=product["target_episodes"],
            )
        raise CollectionBatchError(f"barcode is not in the collection plan: {barcode}")

    def _attempt_number(self, slot):
        return 1 + sum(
            1
            for attempt in self.state["accepted"] + self.state["rejected"]
            if attempt.get("barcode") == slot.barcode
            and attempt.get("ordinal") == slot.ordinal
        )

    def refresh(self):
        if self.state["status"] == "resetting":
            deadline = self.state.get("reset_deadline_wall_time_ns")
            if deadline is None or self._wall_time_ns() >= deadline:
                self.state["status"] = (
                    "complete" if self.next_slot() is None else "ready"
                )
                self.state["reset_deadline_wall_time_ns"] = None
                self._save()
        elif self.next_slot() is None and self.state["status"] == "ready":
            self.state["status"] = "complete"
            self._save()
        return self.state["status"]

    def reset_remaining_seconds(self):
        if self.state["status"] != "resetting":
            return 0.0
        deadline = self.state.get("reset_deadline_wall_time_ns")
        if deadline is None:
            return 0.0
        return max(0.0, (deadline - self._wall_time_ns()) / 1_000_000_000.0)

    def start_attempt(self, episode_id, episode_path):
        self.refresh()
        if self.state["status"] == "resetting":
            self.state["status"] = "ready"
            self.state["reset_deadline_wall_time_ns"] = None
        if self.state["status"] != "ready":
            raise CollectionBatchError(
                f"cannot start an episode while batch status is {self.state['status']}"
            )
        suggested_slot = self.next_slot()
        if suggested_slot is None:
            self.state["status"] = "complete"
            self._save()
            raise CollectionBatchError("collection batch is already complete")
        if self.selection_mode == "flexible":
            slot_data = {
                "product_index": None,
                "barcode": None,
                "sku": None,
                "name": None,
                "ordinal": None,
                "target_episodes": None,
            }
            attempt_number = None
        else:
            slot_data = suggested_slot.as_dict()
            attempt_number = self._attempt_number(suggested_slot)
        self.state["active_attempt"] = {
            **slot_data,
            "attempt_number": attempt_number,
            "episode_id": int(episode_id),
            "episode_path": str(Path(episode_path).resolve()),
            "started_wall_time_ns": self._wall_time_ns(),
            "item_accepted_event_id": None,
        }
        self.state["status"] = "recording"
        self._save()
        return dict(self.state["active_attempt"])

    def validate_barcode(self, barcode):
        if self.state["status"] != "recording" or not self.state["active_attempt"]:
            raise CollectionBatchError("no managed episode is currently recording")
        barcode = str(barcode).strip()
        attempt = self.state["active_attempt"]
        expected = attempt["barcode"]
        if self.selection_mode == "flexible" and expected is None:
            slot = self._slot_for_barcode(barcode)
            attempt.update(slot.as_dict())
            attempt["attempt_number"] = self._attempt_number(slot)
            self._save()
            return dict(attempt)
        if barcode != expected:
            raise CollectionBatchError(
                f"wrong product: expected barcode {expected}, received {barcode}"
            )
        return dict(attempt)

    def mark_item_accepted(self, event_id):
        if self.state["status"] != "recording" or not self.state["active_attempt"]:
            raise CollectionBatchError("no managed episode is currently recording")
        self.state["active_attempt"]["item_accepted_event_id"] = event_id
        self._save()
        return dict(self.state["active_attempt"])

    def mark_saving(self):
        if self.state["status"] != "recording":
            raise CollectionBatchError("batch is not recording")
        self.state["status"] = "saving"
        self._save()

    def mark_aborting(self, reason="operator_aborted_during_recording"):
        if self.state["status"] != "recording" or not self.state["active_attempt"]:
            raise CollectionBatchError("no managed episode is currently recording")
        self.state["active_attempt"]["abort_reason"] = str(reason)
        self.state["status"] = "aborting"
        self._save()

    def finish_abort(self):
        if self.state["status"] != "aborting" or not self.state["active_attempt"]:
            raise CollectionBatchError("batch is not waiting for an aborted episode save")
        attempt = self.state["active_attempt"]
        result = self._move_to_rejected(
            attempt,
            attempt.get("abort_reason", "operator_aborted_during_recording"),
        )
        self.state["last_finalized"] = {
            "episode_id": result["episode_id"],
            "status": "rejected",
        }
        self.state["active_attempt"] = None
        self.state["status"] = "resetting"
        self.state["reset_deadline_wall_time_ns"] = self._wall_time_ns() + int(
            self.plan["reset_seconds"] * 1_000_000_000
        )
        self._save()
        return result

    def finish_save(self, accepted_event):
        if self.state["status"] != "saving" or not self.state["active_attempt"]:
            raise CollectionBatchError("batch is not waiting for an episode save")
        attempt = self.state["active_attempt"]
        expected_barcode = attempt["barcode"]
        event_barcode = None if accepted_event is None else str(
            accepted_event.get("barcode", "")
        ).strip()
        if not expected_barcode or event_barcode != expected_barcode:
            reason = (
                "missing_item_accepted"
                if not event_barcode
                else f"accepted_wrong_barcode:{event_barcode}"
            )
            result = self._move_to_rejected(attempt, reason)
            accepted = False
        else:
            result = {
                **attempt,
                "status": "accepted",
                "item_accepted_event_id": accepted_event.get("event_id"),
                "saved_wall_time_ns": self._wall_time_ns(),
            }
            self.state["accepted"].append(result)
            accepted = True
        self.state["last_finalized"] = {
            "episode_id": result["episode_id"],
            "status": "accepted" if accepted else "rejected",
        }
        self.state["active_attempt"] = None
        self.state["status"] = "resetting"
        self.state["reset_deadline_wall_time_ns"] = self._wall_time_ns() + int(
            self.plan["reset_seconds"] * 1_000_000_000
        )
        if self.next_slot() is None and self.plan["reset_seconds"] == 0:
            self.state["status"] = "complete"
            self.state["reset_deadline_wall_time_ns"] = None
        self._save()
        return accepted, result

    def reject_last_accepted(self):
        if self.state["status"] not in {"ready", "resetting", "complete"}:
            raise CollectionBatchError(
                "the last episode can only be rejected before the next recording starts"
            )
        if not self.state["accepted"]:
            raise CollectionBatchError("there is no accepted episode to reject")
        last_finalized = self.state.get("last_finalized")
        if not last_finalized or last_finalized.get("status") != "accepted":
            raise CollectionBatchError("the most recently saved episode is not accepted")
        attempt = self.state["accepted"][-1]
        if attempt["episode_id"] != last_finalized.get("episode_id"):
            raise CollectionBatchError("batch history is inconsistent")
        result = self._move_to_rejected(attempt, "operator_rejected")
        self.state["accepted"].pop()
        self.state["last_finalized"] = {
            "episode_id": result["episode_id"],
            "status": "rejected",
        }
        self.state["status"] = "resetting"
        if self.state.get("reset_deadline_wall_time_ns") is None:
            self.state["reset_deadline_wall_time_ns"] = self._wall_time_ns() + int(
                self.plan["reset_seconds"] * 1_000_000_000
            )
        self._save()
        return result

    def progress_summary(self):
        slot = self.next_slot()
        return {
            "batch_id": self.plan["batch_id"],
            "selection_mode": self.selection_mode,
            "status": self.state["status"],
            "accepted": self.accepted_count,
            "target": self.total_target,
            "rejected": self.rejected_count,
            "next_slot": None if slot is None else slot.as_dict(),
            "reset_remaining_seconds": self.reset_remaining_seconds(),
        }
