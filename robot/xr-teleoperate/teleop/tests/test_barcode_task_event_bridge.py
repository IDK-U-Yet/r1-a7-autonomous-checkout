# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from teleop.tools.barcode_task_event_bridge import (
    BarcodeScanOnly,
    BarcodeTaskEventBridge,
    DesktopInputIsolationError,
    DuplicateWindow,
    ProductCatalog,
    TaskEventClient,
    TaskEventDeliveryError,
    build_parser,
    block_x11_desktop_input,
    is_capslock_storm,
)
from teleop.utils.task_events import TaskEventServer


class MemoryAuditLog:
    def __init__(self):
        self.records = []

    def write(self, event, **fields):
        self.records.append({"event": event, **fields})


class FakeClient:
    def __init__(self, error=None, replies=None):
        self.error = error
        self.replies = list(replies or [])
        self.payloads = []

    def send(self, payload):
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        return {"status": "ok", "event_id": payload["event_id"], "episode_id": 4}


def create_catalog(tmp_path):
    path = tmp_path / "supermarket.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            sku TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE product_barcodes (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            barcode TEXT NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO products VALUES (1, 'SKU-1', 'Active product', 'ACTIVE');
        INSERT INTO products VALUES (2, 'SKU-2', 'Inactive product', 'INACTIVE');
        INSERT INTO product_barcodes VALUES (1, 1, '6941750501192', 'ACTIVE');
        INSERT INTO product_barcodes VALUES (2, 2, '690000000001', 'ACTIVE');
        """
    )
    connection.commit()
    connection.close()
    return ProductCatalog(path)


def test_catalog_only_returns_active_products(tmp_path):
    catalog = create_catalog(tmp_path)
    try:
        assert catalog.lookup("6941750501192").sku == "SKU-1"
        assert catalog.lookup("690000000001") is None
        assert catalog.lookup("missing") is None
    finally:
        catalog.close()


def test_same_barcode_is_suppressed_for_configured_window(tmp_path):
    now = [100.0]
    catalog = create_catalog(tmp_path)
    client = FakeClient()
    audit = MemoryAuditLog()
    bridge = BarcodeTaskEventBridge(
        catalog,
        client,
        DuplicateWindow(30.0, clock=lambda: now[0]),
        audit,
        session_id="test-session",
    )
    try:
        assert bridge.handle_barcode("6941750501192")["status"] == "accepted"
        now[0] += 29.0
        assert bridge.handle_barcode("6941750501192")["status"] == "duplicate_suppressed"
        now[0] += 1.1
        assert bridge.handle_barcode("6941750501192")["status"] == "accepted"
        assert len(client.payloads) == 2
    finally:
        catalog.close()


def test_failed_delivery_does_not_start_duplicate_window(tmp_path):
    catalog = create_catalog(tmp_path)
    client = FakeClient(TaskEventDeliveryError("recorder offline"))
    bridge = BarcodeTaskEventBridge(
        catalog,
        client,
        DuplicateWindow(30.0),
        MemoryAuditLog(),
        session_id="test-session",
    )
    try:
        assert bridge.handle_barcode("6941750501192")["status"] == "delivery_failed"
        assert bridge.handle_barcode("6941750501192")["status"] == "delivery_failed"
        assert len(client.payloads) == 2
    finally:
        catalog.close()


def test_recorder_episode_duplicate_does_not_increment_accept_count(tmp_path):
    catalog = create_catalog(tmp_path)
    client = FakeClient(
        replies=[
            {"status": "ok", "episode_id": 4, "duplicate": False},
            {
                "status": "ok",
                "episode_id": 4,
                "duplicate": True,
                "duplicate_scope": "episode",
                "accepted_barcode": "6941750501192",
            },
        ]
    )
    audit = MemoryAuditLog()
    bridge = BarcodeTaskEventBridge(
        catalog,
        client,
        DuplicateWindow(0.0),
        audit,
        session_id="test-session",
    )
    try:
        first = bridge.handle_barcode("6941750501192")
        duplicate = bridge.handle_barcode("6941750501192")
        assert first["status"] == "accepted"
        assert duplicate["status"] == "episode_duplicate"
        assert duplicate["accepted_count"] == 1
        assert audit.records[-1]["event"] == "episode_duplicate_suppressed"
    finally:
        catalog.close()


def test_event_id_retry_is_reported_as_accepted_not_episode_duplicate(tmp_path):
    catalog = create_catalog(tmp_path)
    client = FakeClient(
        replies=[
            {
                "status": "ok",
                "episode_id": 4,
                "duplicate": True,
                "duplicate_scope": "event",
            }
        ]
    )
    bridge = BarcodeTaskEventBridge(
        catalog,
        client,
        DuplicateWindow(0.0),
        MemoryAuditLog(),
        session_id="test-session",
    )
    try:
        result = bridge.handle_barcode("6941750501192")
        assert result["status"] == "accepted"
        assert result["accepted_count"] == 1
    finally:
        catalog.close()


def test_unknown_barcode_is_not_sent(tmp_path):
    catalog = create_catalog(tmp_path)
    client = FakeClient()
    bridge = BarcodeTaskEventBridge(
        catalog,
        client,
        DuplicateWindow(30.0),
        MemoryAuditLog(),
    )
    try:
        assert bridge.handle_barcode("750082125063")["status"] == "unknown_barcode"
        assert client.payloads == []
    finally:
        catalog.close()


def test_scan_only_identifies_products_without_event_client(tmp_path):
    catalog = create_catalog(tmp_path)
    audit = MemoryAuditLog()
    scanner = BarcodeScanOnly(catalog, DuplicateWindow(2.0), audit)
    try:
        result = scanner.handle_barcode("6941750501192", source_wall_time_ns=123)
        assert result["status"] == "scanned"
        assert result["product"].name == "Active product"
        assert result["scan_count"] == 1
        assert audit.records[-1]["event"] == "scan_detected"
        assert audit.records[-1]["source_wall_time_ns"] == 123
    finally:
        catalog.close()


def test_scan_only_suppresses_continuous_unknown_barcode_output(tmp_path):
    catalog = create_catalog(tmp_path)
    scanner = BarcodeScanOnly(catalog, DuplicateWindow(2.0), MemoryAuditLog())
    try:
        assert scanner.handle_barcode("unknown")["status"] == "unknown_barcode"
        assert scanner.handle_barcode("unknown")["status"] == "duplicate_suppressed"
    finally:
        catalog.close()


def test_task_event_client_uses_recorder_protocol():
    received = []

    def accept(event):
        received.append(event)
        return {"status": "ok", "episode_id": 7}

    endpoint = f"inproc://barcode-bridge-{uuid.uuid4().hex}"
    server = TaskEventServer(endpoint, accept)
    server.start()
    payload = {
        "event_id": "item_accepted:test:scan-1",
        "event": "item_accepted",
        "barcode": "6941750501192",
        "transaction_id": "test",
        "scan_id": "scan-1",
        "source_wall_time_ns": 123,
    }
    try:
        reply = TaskEventClient(endpoint, attempts=1).send(payload)
    finally:
        server.stop()

    assert reply["status"] == "ok"
    assert reply["episode_id"] == 7
    assert received[0]["barcode"] == "6941750501192"


def test_scanner_prebuffer_default_cannot_discard_every_scan():
    args = build_parser().parse_args([])
    assert args.scanner_prebuffer_seconds > 0.0
    assert args.scan_only is False
    assert args.duplicate_window is None


def test_detects_capslock_storm_without_misclassifying_digit_input():
    storm = SimpleNamespace(partial_length=0, recent_key_codes=(58,) * 32)
    digits = SimpleNamespace(
        partial_length=4,
        recent_key_codes=(58,) * 32 + (7, 10, 4, 28),
    )

    assert is_capslock_storm(storm)
    assert not is_capslock_storm(digits)


def test_floats_only_xinput_device_matching_scanner_node(tmp_path):
    scanner = tmp_path / "event26"
    scanner.touch()
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command == ["xinput", "list", "--short"]:
            scanner_state = (
                "Scanner id=22 [floating slave]"
                if ["xinput", "float", "22"] in calls
                else "Scanner id=22 [slave keyboard (3)]"
            )
            return SimpleNamespace(stdout=f"Keyboard id=14\n{scanner_state}\n")
        if command == ["xinput", "list-props", "14"]:
            return SimpleNamespace(
                stdout='Device Enabled (146): 1\nDevice Node (271): "/dev/input/event4"\n'
            )
        if command == ["xinput", "list-props", "22"]:
            return SimpleNamespace(
                stdout=(
                    "Device Enabled (146): 1\n"
                    f'Device Node (271): "{scanner}"\n'
                )
            )
        if command in (["xinput", "enable", "22"], ["xinput", "float", "22"]):
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    device_id = block_x11_desktop_input(
        scanner,
        environ={"XDG_SESSION_TYPE": "x11", "DISPLAY": ":1"},
        runner=run,
    )

    assert device_id == "22"
    assert ["xinput", "enable", "22"] in calls
    assert ["xinput", "float", "22"] in calls
    assert ["xinput", "float", "14"] not in calls


def test_desktop_input_isolation_refuses_non_x11_session(tmp_path):
    scanner = tmp_path / "event26"
    scanner.touch()
    with pytest.raises(DesktopInputIsolationError, match="requires an X11 desktop"):
        block_x11_desktop_input(
            scanner,
            environ={"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":1"},
        )
