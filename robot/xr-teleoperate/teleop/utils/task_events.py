# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import secrets
import threading
import time

import logging_mp
import zmq


logger_mp = logging_mp.getLogger(__name__)


class TaskEventServer:
    """Receive idempotent business events from the checkout system over ZMQ."""

    def __init__(self, endpoint, on_item_accepted, token=None):
        if not callable(on_item_accepted):
            raise ValueError("on_item_accepted callback must be callable")
        self.endpoint = endpoint
        self.on_item_accepted = on_item_accepted
        self.token = token or None
        self._running = False
        self._thread = None
        self._successful_replies = {}
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)

    def _error(self, event_id, message):
        return {
            "event_id": event_id,
            "status": "error",
            "message": message,
        }

    def handle_message(self, message):
        if not isinstance(message, dict):
            return self._error(None, "request must be a JSON object")

        event_id = message.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            return self._error(event_id, "event_id must be a non-empty string")
        event_id = event_id.strip()

        cached = self._successful_replies.get(event_id)
        if cached is not None:
            return {**cached, "duplicate": True, "duplicate_scope": "event"}

        if self.token is not None:
            supplied_token = message.get("token")
            if not isinstance(supplied_token, str) or not secrets.compare_digest(
                supplied_token, self.token
            ):
                return self._error(event_id, "invalid task event token")

        if message.get("event") != "item_accepted":
            return self._error(event_id, "only item_accepted is supported")
        barcode = message.get("barcode")
        if not isinstance(barcode, str) or not barcode.strip():
            return self._error(event_id, "barcode must be a non-empty string")

        optional_strings = ("transaction_id", "order_line_id", "scan_id")
        for field in optional_strings:
            value = message.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                return self._error(event_id, f"{field} must be a non-empty string or null")
        source_wall_time_ns = message.get("source_wall_time_ns")
        if source_wall_time_ns is not None and not isinstance(source_wall_time_ns, int):
            return self._error(event_id, "source_wall_time_ns must be an integer or null")

        normalized = {
            "event_id": event_id,
            "event": "item_accepted",
            "barcode": barcode.strip(),
            "transaction_id": message.get("transaction_id"),
            "order_line_id": message.get("order_line_id"),
            "scan_id": message.get("scan_id"),
            "source_wall_time_ns": source_wall_time_ns,
            "receiver_monotonic_ns": time.monotonic_ns(),
            "receiver_wall_time_ns": time.time_ns(),
        }
        try:
            reply = dict(self.on_item_accepted(normalized) or {})
        except Exception as exc:
            logger_mp.exception("[TaskEventServer] item_accepted callback failed")
            return self._error(event_id, str(exc))

        reply.setdefault("event_id", event_id)
        reply.setdefault("status", "ok")
        reply.setdefault("duplicate", False)
        if reply.get("duplicate", False):
            reply.setdefault("duplicate_scope", "episode")
        if reply.get("status") == "ok":
            if reply.get("duplicate_scope") != "episode":
                self._successful_replies[event_id] = dict(reply)
            if reply.get("duplicate", False):
                logger_mp.info(
                    "[TaskEventServer] Ignored scan %s because episode %s already accepted %s",
                    event_id,
                    reply.get("episode_id"),
                    reply.get("accepted_barcode", "a barcode"),
                )
            else:
                logger_mp.info(
                    "[TaskEventServer] Accepted item_accepted event %s for barcode %s",
                    event_id,
                    normalized["barcode"],
                )
        else:
            logger_mp.warning(
                "[TaskEventServer] Rejected event %s: %s",
                event_id,
                reply.get("message", "unknown reason"),
            )
        return reply

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                ready = dict(poller.poll(100))
                if self._socket not in ready:
                    continue
                message = self._socket.recv_json()
                self._socket.send_json(self.handle_message(message))
            except zmq.error.ContextTerminated:
                break
            except Exception as exc:
                logger_mp.error(f"[TaskEventServer] request failed: {exc}")
                try:
                    self._socket.send_json(self._error(None, str(exc)))
                except Exception:
                    pass

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._serve,
            name="task-event-server",
            daemon=True,
        )
        self._thread.start()
        logger_mp.info(f"[TaskEventServer] Listening at {self.endpoint}")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._socket.close(linger=0)
