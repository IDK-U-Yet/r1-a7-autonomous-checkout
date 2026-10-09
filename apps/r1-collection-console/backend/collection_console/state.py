from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCANNER_EVENT_LEVELS = {
    "item_accepted": "success",
    "delivery_failed": "error",
    "unknown_barcode": "warning",
    "episode_duplicate_suppressed": "info",
    "duplicate_suppressed": "info",
    "scanner_ready": "success",
    "scanner_error": "error",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_last_json_lines(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    """Read the last complete JSONL records without loading a large log."""
    if limit <= 0 or not path.is_file():
        return []
    chunk_size = 64 * 1024
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        data = b""
        while position > 0 and data.count(b"\n") <= limit:
            size = min(chunk_size, position)
            position -= size
            stream.seek(position)
            data = stream.read(size) + data

    records: list[dict[str, Any]] = []
    for raw_line in data.splitlines()[-limit:]:
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def read_last_json_record(path: Path) -> dict[str, Any] | None:
    records = read_last_json_lines(path, limit=2)
    return records[-1] if records else None


def _attempt_timestamp_ns(attempt: dict[str, Any]) -> int:
    for key in (
        "saved_wall_time_ns",
        "rejected_wall_time_ns",
        "started_wall_time_ns",
    ):
        value = attempt.get(key)
        if isinstance(value, int):
            return value
    return 0


def _batch_event(attempt: dict[str, Any], status: str) -> dict[str, Any]:
    episode_id = attempt.get("episode_id")
    timestamp_ns = _attempt_timestamp_ns(attempt)
    accepted = status == "accepted"
    reason = attempt.get("rejection_reason")
    return {
        "id": f"batch:{status}:{episode_id}:{timestamp_ns}",
        "source": "collection",
        "type": f"episode_{status}",
        "level": "success" if accepted else "warning",
        "wall_time_ns": timestamp_ns,
        "episode_id": episode_id,
        "barcode": attempt.get("barcode"),
        "title": (
            f"episode_{int(episode_id):04d} 已接收"
            if accepted and isinstance(episode_id, int)
            else f"episode_{int(episode_id):04d} 已废弃"
            if isinstance(episode_id, int)
            else "Episode 状态已更新"
        ),
        "detail": (
            f"{attempt.get('name') or '未绑定商品'} · 第 {attempt.get('ordinal') or '-'} 条"
            if accepted
            else str(reason or "rejected")
        ),
    }


def _empty_batch(task_dir: Path) -> dict[str, Any]:
    return {
        "available": False,
        "task_dir": str(task_dir),
        "batch_id": task_dir.name,
        "selection_mode": None,
        "status": "waiting",
        "phase": "waiting",
        "accepted": 0,
        "target": 0,
        "rejected": 0,
        "remaining": 0,
        "reset_remaining_seconds": 0.0,
        "active_attempt": None,
        "last_finalized": None,
        "products": [],
        "history": [],
        "error": None,
    }


def read_batch_snapshot(task_dir: Path, now_wall_time_ns: int | None = None) -> dict[str, Any]:
    now_wall_time_ns = now_wall_time_ns or time.time_ns()
    snapshot = _empty_batch(task_dir)
    state_path = task_dir / "batch_state.json"
    if not state_path.is_file():
        return snapshot

    try:
        state = _read_json(state_path)
        plan_path = Path(state["plan_path"]).expanduser()
        plan = _read_json(plan_path)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        snapshot["error"] = f"无法读取批次状态: {exc}"
        return snapshot

    accepted_attempts = state.get("accepted", [])
    rejected_attempts = state.get("rejected", [])
    products = plan.get("products", [])
    counts = Counter(
        str(attempt.get("barcode"))
        for attempt in accepted_attempts
        if attempt.get("barcode") is not None
    )
    product_rows = [
        {
            "barcode": str(product.get("barcode", "")),
            "sku": str(product.get("sku", "")),
            "name": str(product.get("name", "")),
            "accepted": counts[str(product.get("barcode", ""))],
            "target": int(product.get("target_episodes", 0)),
        }
        for product in products
    ]
    target = sum(product["target"] for product in product_rows)
    status = str(state.get("status", "unknown"))
    active_attempt = state.get("active_attempt")
    if status == "recording" and active_attempt:
        phase = (
            "placing"
            if active_attempt.get("item_accepted_event_id")
            else "scanning"
        )
    elif status in {"saving", "aborting", "resetting", "complete", "ready"}:
        phase = status
    else:
        phase = "unknown"

    reset_remaining = 0.0
    deadline = state.get("reset_deadline_wall_time_ns")
    if status == "resetting" and isinstance(deadline, int):
        reset_remaining = max(0.0, (deadline - now_wall_time_ns) / 1_000_000_000)

    history = [
        *(_batch_event(item, "accepted") for item in accepted_attempts),
        *(_batch_event(item, "rejected") for item in rejected_attempts),
    ]
    history.sort(key=lambda item: item["wall_time_ns"], reverse=True)

    snapshot.update(
        {
            "available": True,
            "batch_id": str(state.get("batch_id") or plan.get("batch_id") or task_dir.name),
            "selection_mode": str(plan.get("selection_mode", "ordered")),
            "status": status,
            "phase": phase,
            "accepted": len(accepted_attempts),
            "target": target,
            "rejected": len(rejected_attempts),
            "remaining": max(0, target - len(accepted_attempts)),
            "reset_remaining_seconds": reset_remaining,
            "active_attempt": active_attempt,
            "last_finalized": state.get("last_finalized"),
            "products": product_rows,
            "history": history,
            "updated_wall_time_ns": state.get("updated_wall_time_ns"),
        }
    )
    return snapshot


def _scanner_title(record: dict[str, Any]) -> tuple[str, str]:
    event = str(record.get("event", "unknown"))
    barcode = str(record.get("barcode", ""))
    product = record.get("product") if isinstance(record.get("product"), dict) else {}
    product_name = str(product.get("name", ""))
    if event == "item_accepted":
        return f"扫码成功 {barcode}", product_name or "商品已进入放置阶段"
    if event == "delivery_failed":
        return f"扫码事件投递失败 {barcode}", str(record.get("error") or record.get("message") or "采集器未接收")
    if event == "unknown_barcode":
        return f"未知条码 {barcode}", "商品数据库中没有有效记录"
    if event == "episode_duplicate_suppressed":
        return f"重复扫码已忽略 {barcode}", "当前 Episode 已经接受过条码"
    if event == "duplicate_suppressed":
        return f"冷却期扫码已忽略 {barcode}", "没有重复投递"
    if event == "scanner_ready":
        return "扫码器已就绪", "正在等待条码"
    return event.replace("_", " "), barcode


def read_scanner_snapshot(scanner_log: Path, now_wall_time_ns: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    now_wall_time_ns = now_wall_time_ns or time.time_ns()
    records = read_last_json_lines(scanner_log, limit=400)
    latest_health_record = next(
        (
            record
            for record in reversed(records)
            if record.get("event") in {"scanner_health", "scanner_ready"}
            and isinstance(record.get("scanner_health"), dict)
        ),
        None,
    )
    health_data = latest_health_record.get("scanner_health", {}) if latest_health_record else {}
    health_wall_time_ns = latest_health_record.get("wall_time_ns") if latest_health_record else None
    health_age_seconds = (
        max(0.0, (now_wall_time_ns - health_wall_time_ns) / 1_000_000_000)
        if isinstance(health_wall_time_ns, int)
        else None
    )
    online = bool(health_data.get("ready")) and health_age_seconds is not None and health_age_seconds <= 10.0
    scanner = {
        "available": scanner_log.is_file(),
        "online": online,
        "state": str(health_data.get("state") or ("waiting" if not records else "unknown")),
        "device": health_data.get("device"),
        "resolved_device": health_data.get("resolved_device"),
        "exclusive": health_data.get("exclusive"),
        "last_scan_at": health_data.get("last_scan_at"),
        "health_age_seconds": health_age_seconds,
        "error_code": health_data.get("error_code"),
        "message": health_data.get("message"),
        "log_path": str(scanner_log),
    }

    events: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        event_type = str(record.get("event", ""))
        if event_type not in SCANNER_EVENT_LEVELS or event_type == "scanner_ready":
            continue
        title, detail = _scanner_title(record)
        reply = record.get("recorder_reply") if isinstance(record.get("recorder_reply"), dict) else {}
        wall_time_ns = record.get("wall_time_ns")
        events.append(
            {
                "id": str(record.get("event_id") or f"scanner:{wall_time_ns}:{index}"),
                "source": "scanner",
                "type": event_type,
                "level": SCANNER_EVENT_LEVELS[event_type],
                "wall_time_ns": wall_time_ns if isinstance(wall_time_ns, int) else 0,
                "episode_id": reply.get("episode_id"),
                "barcode": record.get("barcode"),
                "title": title,
                "detail": detail,
                "product": record.get("product"),
            }
        )
    events.sort(key=lambda item: item["wall_time_ns"], reverse=True)
    return scanner, events[:200]


def read_collector_snapshot(diagnostics_path: Path, now_wall_time: float | None = None) -> dict[str, Any]:
    now_wall_time = now_wall_time or time.time()
    record = read_last_json_record(diagnostics_path)
    if record is None:
        return {
            "available": diagnostics_path.is_file(),
            "online": False,
            "state": "waiting",
            "xr_age_ms": None,
            "fault_reason": None,
            "diagnostics_path": str(diagnostics_path),
            "age_seconds": None,
        }
    try:
        age_seconds = max(0.0, now_wall_time - diagnostics_path.stat().st_mtime)
    except OSError:
        age_seconds = None
    xr_age = record.get("xr_age")
    return {
        "available": True,
        "online": age_seconds is not None and age_seconds <= 2.0,
        "state": str(record.get("state", "unknown")),
        "xr_age_ms": float(xr_age) * 1000.0 if isinstance(xr_age, (int, float)) else None,
        "fault_reason": record.get("fault_reason"),
        "diagnostics_path": str(diagnostics_path),
        "age_seconds": age_seconds,
    }


def read_disk_snapshot(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(probe),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "warning": usage.free < 50 * 1024**3,
    }


def build_file_snapshot(task_dir: Path, scanner_log: Path, diagnostics_path: Path) -> dict[str, Any]:
    now_ns = time.time_ns()
    batch = read_batch_snapshot(task_dir, now_ns)
    scanner, scanner_events = read_scanner_snapshot(scanner_log, now_ns)
    collector = read_collector_snapshot(diagnostics_path)
    collection_events = batch.get("history", [])
    events = sorted(
        [*scanner_events, *collection_events],
        key=lambda item: item.get("wall_time_ns", 0),
        reverse=True,
    )
    return {
        "server_wall_time_ns": now_ns,
        "batch": batch,
        "scanner": scanner,
        "collector": collector,
        "disk": read_disk_snapshot(task_dir),
        "events": events[:300],
    }
