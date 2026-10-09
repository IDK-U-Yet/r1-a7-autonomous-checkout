#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read a HID barcode scanner and emit an ``item_accepted`` task event."""

import argparse
import json
import math
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import zmq


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SUPERMARKET_ROOT = WORKSPACE_ROOT / "apps" / "r1-supermarket"
DEFAULT_DATABASE = DEFAULT_SUPERMARKET_ROOT / "data" / "supermarket.db"
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "runtime" / "logs"


class TaskEventDeliveryError(RuntimeError):
    pass


class DesktopInputIsolationError(RuntimeError):
    pass


def block_x11_desktop_input(device_path, *, environ=None, runner=subprocess.run):
    """Keep the scanner enabled but detach it from the X11 core keyboard."""
    environment = os.environ if environ is None else environ
    if environment.get("XDG_SESSION_TYPE") != "x11" or not environment.get("DISPLAY"):
        raise DesktopInputIsolationError(
            "automatic scanner isolation requires an X11 desktop and DISPLAY"
        )

    target = str(Path(device_path).expanduser().resolve(strict=True))
    try:
        listing = runner(
            ["xinput", "list", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DesktopInputIsolationError(f"cannot list XInput devices: {exc}") from exc

    candidate_ids = []
    for match in re.finditer(r"\bid=(\d+)\b", listing.stdout):
        device_id = match.group(1)
        try:
            properties = runner(
                ["xinput", "list-props", device_id],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue
        node_match = re.search(
            r'Device Node \(\d+\):\s+"([^"]+)"', properties.stdout
        )
        if node_match and str(Path(node_match.group(1)).resolve()) == target:
            candidate_ids.append(device_id)

    if len(candidate_ids) != 1:
        raise DesktopInputIsolationError(
            f"expected one XInput device for {target}, found {len(candidate_ids)}"
        )

    device_id = candidate_ids[0]
    try:
        # Some HID scanners use keyboard LED state while decoding. Disabling the
        # XInput device breaks that exchange, so leave it enabled and float it.
        runner(["xinput", "enable", device_id], check=True)
        runner(["xinput", "float", device_id], check=True)
        properties = runner(
            ["xinput", "list-props", device_id],
            check=True,
            capture_output=True,
            text=True,
        )
        isolated_listing = runner(
            ["xinput", "list", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DesktopInputIsolationError(
            f"cannot float XInput device {device_id}: {exc}"
        ) from exc
    if not re.search(r"Device Enabled \(\d+\):\s+1(?:\s|$)", properties.stdout):
        raise DesktopInputIsolationError(
            f"XInput device {device_id} is not enabled after isolation"
        )
    floating_pattern = rf"^.*\bid={re.escape(device_id)}\b.*\[floating slave\]\s*$"
    if not re.search(floating_pattern, isolated_listing.stdout, re.MULTILINE):
        raise DesktopInputIsolationError(
            f"XInput device {device_id} is still attached to a desktop master"
        )
    return device_id


@dataclass(frozen=True)
class Product:
    product_id: int
    sku: str
    name: str
    barcode: str


class ProductCatalog:
    """Read active barcode mappings from the supermarket SQLite database."""

    def __init__(self, database_path):
        path = Path(database_path).expanduser().resolve(strict=True)
        self.database_path = path
        self._connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=2.0,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only = ON")
        self._connection.execute(
            "SELECT 1 FROM products JOIN product_barcodes "
            "ON products.id = product_barcodes.product_id LIMIT 1"
        ).fetchone()

    def lookup(self, barcode):
        row = self._connection.execute(
            """
            SELECT p.id AS product_id, p.sku, p.name, pb.barcode
            FROM product_barcodes AS pb
            JOIN products AS p ON p.id = pb.product_id
            WHERE pb.barcode = ?
              AND pb.status = 'ACTIVE'
              AND p.status = 'ACTIVE'
            LIMIT 1
            """,
            (barcode,),
        ).fetchone()
        if row is None:
            return None
        return Product(
            product_id=int(row["product_id"]),
            sku=str(row["sku"]),
            name=str(row["name"]),
            barcode=str(row["barcode"]),
        )

    def close(self):
        self._connection.close()


class DuplicateWindow:
    def __init__(self, seconds, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("duplicate window must be finite and non-negative")
        self.seconds = float(seconds)
        self.clock = clock
        self._accepted_at = {}

    def remaining(self, barcode):
        accepted_at = self._accepted_at.get(barcode)
        if accepted_at is None:
            return 0.0
        remaining = self.seconds - (self.clock() - accepted_at)
        if remaining <= 0:
            self._accepted_at.pop(barcode, None)
            return 0.0
        return remaining

    def mark_accepted(self, barcode):
        self._accepted_at[barcode] = self.clock()


class TaskEventClient:
    def __init__(
        self,
        endpoint,
        *,
        token=None,
        timeout_ms=700,
        attempts=3,
        retry_delays=(0.1, 0.3),
    ):
        if not endpoint.strip():
            raise ValueError("task event endpoint must not be empty")
        if timeout_ms <= 0 or attempts <= 0:
            raise ValueError("timeout_ms and attempts must be positive")
        self.endpoint = endpoint.strip()
        self.token = token or None
        self.timeout_ms = int(timeout_ms)
        self.attempts = int(attempts)
        self.retry_delays = tuple(retry_delays)
        self._context = zmq.Context.instance()

    def _send_once(self, payload):
        request = dict(payload)
        if self.token:
            request["token"] = self.token
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send_json(request)
            reply = socket.recv_json()
        except zmq.Again as exc:
            raise TaskEventDeliveryError(
                f"task event request timed out after {self.timeout_ms} ms"
            ) from exc
        except zmq.ZMQError as exc:
            raise TaskEventDeliveryError(f"task event transport failed: {exc}") from exc
        finally:
            socket.close(linger=0)

        if not isinstance(reply, dict):
            raise TaskEventDeliveryError("task event receiver returned a non-object reply")
        if reply.get("event_id") not in (None, payload["event_id"]):
            raise TaskEventDeliveryError("task event receiver replied with a different event_id")
        if reply.get("status") != "ok":
            message = reply.get("message") or reply.get("error") or "receiver returned status:error"
            raise TaskEventDeliveryError(str(message))
        return reply

    def send(self, payload):
        last_error = None
        for attempt in range(self.attempts):
            try:
                return self._send_once(payload)
            except TaskEventDeliveryError as exc:
                last_error = exc
                if attempt + 1 < self.attempts:
                    delay_index = min(attempt, len(self.retry_delays) - 1)
                    delay = self.retry_delays[delay_index] if self.retry_delays else 0.0
                    if delay > 0:
                        time.sleep(delay)
        raise last_error


class JsonlAuditLog:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event, **fields):
        record = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class BarcodeTaskEventBridge:
    def __init__(
        self,
        catalog,
        event_client,
        duplicate_window,
        audit_log,
        *,
        session_id=None,
    ):
        self.catalog = catalog
        self.event_client = event_client
        self.duplicate_window = duplicate_window
        self.audit_log = audit_log
        self.session_id = session_id or f"direct_{uuid.uuid4().hex}"
        self.accepted_count = 0

    def handle_barcode(self, barcode, source_wall_time_ns=None):
        barcode = str(barcode).strip()
        remaining = self.duplicate_window.remaining(barcode)
        if remaining > 0:
            self.audit_log.write(
                "duplicate_suppressed",
                barcode=barcode,
                remaining_seconds=remaining,
            )
            return {"status": "duplicate_suppressed", "remaining_seconds": remaining}

        product = self.catalog.lookup(barcode)
        if product is None:
            self.audit_log.write("unknown_barcode", barcode=barcode)
            return {"status": "unknown_barcode", "barcode": barcode}

        scan_id = f"direct_scan_{uuid.uuid4().hex}"
        payload = {
            "event_id": f"item_accepted:{self.session_id}:{scan_id}",
            "event": "item_accepted",
            "barcode": barcode,
            "transaction_id": self.session_id,
            "scan_id": scan_id,
            "source_wall_time_ns": source_wall_time_ns or time.time_ns(),
        }
        try:
            reply = self.event_client.send(payload)
        except TaskEventDeliveryError as exc:
            self.audit_log.write(
                "delivery_failed",
                barcode=barcode,
                product=asdict(product),
                event_id=payload["event_id"],
                error=str(exc),
            )
            return {
                "status": "delivery_failed",
                "barcode": barcode,
                "product": product,
                "error": str(exc),
            }

        if reply.get("duplicate", False) and reply.get("duplicate_scope") != "event":
            self.audit_log.write(
                "episode_duplicate_suppressed",
                barcode=barcode,
                product=asdict(product),
                event_id=payload["event_id"],
                recorder_reply=reply,
            )
            return {
                "status": "episode_duplicate",
                "barcode": barcode,
                "product": product,
                "reply": reply,
                "accepted_count": self.accepted_count,
            }

        self.duplicate_window.mark_accepted(barcode)
        self.accepted_count += 1
        self.audit_log.write(
            "item_accepted",
            barcode=barcode,
            product=asdict(product),
            event_id=payload["event_id"],
            recorder_reply=reply,
            accepted_count=self.accepted_count,
        )
        return {
            "status": "accepted",
            "barcode": barcode,
            "product": product,
            "reply": reply,
            "accepted_count": self.accepted_count,
        }


class BarcodeScanOnly:
    """Identify scanned products without contacting a recorder."""

    def __init__(self, catalog, duplicate_window, audit_log):
        self.catalog = catalog
        self.duplicate_window = duplicate_window
        self.audit_log = audit_log
        self.scan_count = 0

    def handle_barcode(self, barcode, source_wall_time_ns=None):
        barcode = str(barcode).strip()
        remaining = self.duplicate_window.remaining(barcode)
        if remaining > 0:
            self.audit_log.write(
                "duplicate_suppressed",
                barcode=barcode,
                remaining_seconds=remaining,
                mode="scan_only",
            )
            return {"status": "duplicate_suppressed", "remaining_seconds": remaining}

        product = self.catalog.lookup(barcode)
        self.duplicate_window.mark_accepted(barcode)
        if product is None:
            self.audit_log.write(
                "unknown_barcode",
                barcode=barcode,
                mode="scan_only",
                source_wall_time_ns=source_wall_time_ns,
            )
            return {"status": "unknown_barcode", "barcode": barcode}

        self.scan_count += 1
        self.audit_log.write(
            "scan_detected",
            barcode=barcode,
            product=asdict(product),
            scan_count=self.scan_count,
            mode="scan_only",
            source_wall_time_ns=source_wall_time_ns,
        )
        return {
            "status": "scanned",
            "barcode": barcode,
            "product": product,
            "scan_count": self.scan_count,
        }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Send item_accepted directly from a HID barcode scanner to a compatible "
            "recording or inference task-event receiver."
        )
    )
    parser.add_argument("--device", default="/dev/supermarket-scanner")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--supermarket-root", type=Path, default=DEFAULT_SUPERMARKET_ROOT)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:55580")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Identify barcodes and products without connecting to a recorder.",
    )
    parser.add_argument("--token", default=os.environ.get("R1_TASK_EVENT_TOKEN"))
    parser.add_argument(
        "--duplicate-window",
        type=float,
        default=None,
        help=(
            "Optional time suppression for repeated HID emissions. Defaults to 0 in "
            "task-event mode because the recorder deduplicates by Episode, and 2 seconds "
            "in scan-only mode."
        ),
    )
    parser.add_argument("--delivery-timeout-ms", type=int, default=700)
    parser.add_argument("--delivery-attempts", type=int, default=3)
    parser.add_argument("--scanner-prebuffer-seconds", type=float, default=2.0)
    parser.add_argument("--scanner-ready-timeout", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--grab-device", action="store_true")
    parser.add_argument(
        "--allow-desktop-input",
        action="store_true",
        help=(
            "Do not block the scanner from typing into X11 applications. By default "
            "the matching XInput device stays enabled but is detached from the X11 "
            "core keyboard while direct evdev reads remain active."
        ),
    )
    parser.add_argument(
        "--allow-shared-device",
        action="store_true",
        help="Allow startup while the supermarket scanner service is running (debug only).",
    )
    parser.add_argument("--log-path", type=Path)
    return parser


def find_scanner_service_processes():
    matches = []
    proc = Path("/proc")
    if not proc.is_dir():
        return matches
    for process_dir in proc.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, ValueError):
            continue
        if "services.scanner_service" in command:
            matches.append((int(process_dir.name), command.strip()))
    return matches


def wait_for_scanner_ready(scanner, timeout, clock=time.monotonic):
    deadline = clock() + timeout
    while clock() < deadline:
        health = scanner.health()
        if health.ready:
            return health
        if health.error_code:
            raise RuntimeError(health.message or health.error_code)
        time.sleep(0.05)
    health = scanner.health()
    raise RuntimeError(
        f"scanner did not become ready within {timeout:.1f}s "
        f"(state={health.state}, error={health.message or 'none'})"
    )


def is_capslock_storm(scanner_health, minimum_events=32):
    """Detect a scanner keyboard-mode loop that cannot contain a barcode."""
    codes = tuple(scanner_health.recent_key_codes)
    return (
        scanner_health.partial_length == 0
        and len(codes) >= minimum_events
        and all(code == 58 for code in codes[-minimum_events:])
    )


def main():
    args = build_parser().parse_args()
    if args.duplicate_window is None:
        args.duplicate_window = 2.0 if args.scan_only else 0.0
    if not math.isfinite(args.duplicate_window) or args.duplicate_window < 0:
        raise SystemExit("--duplicate-window must be finite and non-negative")
    for name in (
        "scanner_prebuffer_seconds",
        "scanner_ready_timeout",
        "health_interval",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    conflicts = find_scanner_service_processes()
    if conflicts and not args.allow_shared_device:
        processes = ", ".join(str(pid) for pid, _command in conflicts)
        raise SystemExit(
            "supermarket scanner service is still running "
            f"(PID {processes}); stop it before starting the standalone bridge"
        )

    supermarket_root = args.supermarket_root.expanduser().resolve(strict=True)
    if str(supermarket_root) not in sys.path:
        sys.path.insert(0, str(supermarket_root))
    from services.scanner_service.hid import HIDBarcodeScanner, HIDConfig, HIDScannerError

    log_path = args.log_path
    if log_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_prefix = "barcode_scan_test" if args.scan_only else "barcode_task_event"
        log_path = DEFAULT_LOG_DIR / f"{log_prefix}_{timestamp}.jsonl"

    catalog = ProductCatalog(args.database)
    audit_log = JsonlAuditLog(log_path)
    duplicate_window = DuplicateWindow(args.duplicate_window)
    if args.scan_only:
        bridge = BarcodeScanOnly(catalog, duplicate_window, audit_log)
    else:
        client = TaskEventClient(
            args.endpoint,
            token=args.token,
            timeout_ms=args.delivery_timeout_ms,
            attempts=args.delivery_attempts,
        )
        bridge = BarcodeTaskEventBridge(
            catalog,
            client,
            duplicate_window,
            audit_log,
        )
    desktop_input_id = None
    if not args.allow_desktop_input:
        try:
            desktop_input_id = block_x11_desktop_input(args.device)
        except DesktopInputIsolationError as exc:
            catalog.close()
            raise SystemExit(
                f"cannot block scanner desktop input: {exc}. Use "
                "--allow-desktop-input only if keyboard injection is intentional."
            ) from exc

    scanner = HIDBarcodeScanner(
        HIDConfig(
            device=args.device,
            grab=args.grab_device,
            prebuffer_seconds=args.scanner_prebuffer_seconds,
            same_code_release_seconds=0.0,
        )
    )

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    print(f"Database: {catalog.database_path}")
    print(f"Scanner: {args.device}")
    print(f"Mode: {'SCAN ONLY' if args.scan_only else 'TASK EVENT'}")
    if desktop_input_id is not None:
        print(f"Desktop input: BLOCKED (XInput id {desktop_input_id})")
    if not args.scan_only:
        print(f"Task event receiver: {args.endpoint}")
        print("Run policy: first accepted barcode wins; later scans are ignored")
    print(f"Duplicate window: {args.duplicate_window:.1f} seconds")
    print(f"Audit log: {log_path}")
    print("Opening barcode scanner...", flush=True)

    try:
        scanner_health = wait_for_scanner_ready(scanner, args.scanner_ready_timeout)
        audit_log.write("scanner_ready", scanner_health=asdict(scanner_health))
        print(
            f"SCANNER READY {scanner_health.resolved_device} | "
            f"exclusive={scanner_health.exclusive}",
            flush=True,
        )
        print("Waiting for barcode scans. Press Ctrl-C to stop.", flush=True)
        last_health_log = time.monotonic()
        last_raw_key_count = scanner_health.raw_key_count
        capslock_storm_reported = False
        while not stop_requested:
            try:
                observation = scanner.read(timeout=0.5)
            except HIDScannerError as exc:
                print(f"SCANNER ERROR: {exc}", flush=True)
                time.sleep(1.0)
                continue
            scanner_health = scanner.health()
            now = time.monotonic()
            capslock_storm = is_capslock_storm(scanner_health)
            if capslock_storm and not capslock_storm_reported:
                message = (
                    "scanner is sending only KEY_CAPSLOCK instead of barcode digits; "
                    "continuous-scan mode alone does not explain this. "
                    f"exclusive={scanner_health.exclusive}; desktop isolation keeps the "
                    "XInput device enabled and does not grab evdev. Verify the scanner's "
                    "CapsLock adaptation setting, or configure it for standard USB HID "
                    "keyboard/USB serial output."
                )
                audit_log.write(
                    "scanner_input_anomaly",
                    anomaly="capslock_storm",
                    message=message,
                    scanner_health=asdict(scanner_health),
                )
                print(f"SCANNER INPUT ERROR: {message}", flush=True)
                capslock_storm_reported = True
            elif not capslock_storm:
                capslock_storm_reported = False
            if (
                scanner_health.raw_key_count != last_raw_key_count
                or now - last_health_log >= args.health_interval
            ):
                audit_log.write("scanner_health", scanner_health=asdict(scanner_health))
                last_health_log = now
                last_raw_key_count = scanner_health.raw_key_count
            if observation is None:
                continue
            result = bridge.handle_barcode(
                observation.value,
                source_wall_time_ns=int(observation.captured_wall_time * 1_000_000_000),
            )
            status = result["status"]
            if status == "scanned":
                product = result["product"]
                print(
                    f"SCANNED {observation.value} | {product.sku} | {product.name} | "
                    f"count={result['scan_count']}",
                    flush=True,
                )
            elif status == "accepted":
                product = result["product"]
                print(
                    f"ACCEPTED {observation.value} | {product.sku} | {product.name} | "
                    f"episode={result['reply'].get('episode_id')}",
                    flush=True,
                )
            elif status == "episode_duplicate":
                print(
                    f"IGNORED {observation.value} | episode "
                    f"{result['reply'].get('episode_id')} already accepted "
                    f"{result['reply'].get('accepted_barcode', 'a barcode')}",
                    flush=True,
                )
            elif status == "duplicate_suppressed":
                print(
                    f"DUPLICATE {observation.value} | suppressed for "
                    f"{result['remaining_seconds']:.1f}s",
                    flush=True,
                )
            elif status == "unknown_barcode":
                print(f"UNKNOWN {observation.value} | not in active product catalog", flush=True)
            else:
                print(f"DELIVERY FAILED {observation.value} | {result['error']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        scanner.close()
        catalog.close()
    print("Barcode task event bridge stopped.")


if __name__ == "__main__":
    main()
