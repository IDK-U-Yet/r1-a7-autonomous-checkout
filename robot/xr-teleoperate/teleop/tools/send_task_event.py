#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import argparse
import json
import os
import uuid

import zmq


def main():
    parser = argparse.ArgumentParser(description="Send an item_accepted event to the recorder.")
    parser.add_argument("barcode")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:55580")
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--transaction-id")
    parser.add_argument("--order-line-id")
    parser.add_argument("--scan-id")
    parser.add_argument("--token", default=os.environ.get("R1_TASK_EVENT_TOKEN"))
    parser.add_argument("--timeout-ms", type=int, default=2000)
    args = parser.parse_args()

    payload = {
        "event_id": args.event_id or f"item_{uuid.uuid4().hex}",
        "event": "item_accepted",
        "barcode": args.barcode,
        "transaction_id": args.transaction_id,
        "order_line_id": args.order_line_id,
        "scan_id": args.scan_id,
    }
    if args.token:
        payload["token"] = args.token

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.connect(args.endpoint)
    try:
        socket.send_json(payload)
        reply = socket.recv_json()
    except zmq.Again as exc:
        raise SystemExit(f"task event request timed out: {args.endpoint}") from exc
    finally:
        socket.close(linger=0)

    print(json.dumps(reply, ensure_ascii=False, indent=2))
    raise SystemExit(0 if reply.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
