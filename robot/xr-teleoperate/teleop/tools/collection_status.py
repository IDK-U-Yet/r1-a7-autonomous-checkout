#!/usr/bin/env python3
# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Print managed collection progress without changing batch state."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Show persistent collection batch progress.")
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()

    state_path = args.task_dir.expanduser().resolve() / "batch_state.json"
    if not state_path.is_file():
        parser.error(f"batch state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan_path = Path(state["plan_path"])
    if not plan_path.is_file():
        parser.error(f"collection plan does not exist: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    counts = Counter(attempt["barcode"] for attempt in state.get("accepted", []))
    print(f"Batch: {state['batch_id']}")
    print(f"Selection: {plan.get('selection_mode', 'ordered')}")
    print(f"Status: {state['status']}")
    total = sum(product["target_episodes"] for product in plan["products"])
    print(
        f"Progress: {len(state.get('accepted', []))}/{total} accepted, "
        f"{len(state.get('rejected', []))} rejected"
    )
    deadline = state.get("reset_deadline_wall_time_ns")
    if state["status"] == "resetting" and deadline is not None:
        remaining = max(0.0, deadline / 1_000_000_000.0 - time.time())
        print(f"Reset remaining: {remaining:.1f}s")
    for product in plan["products"]:
        print(
            f"  {product['sku']} | {product['name']} | {product['barcode']} | "
            f"{counts[product['barcode']]}/{product['target_episodes']}"
        )
    active = state.get("active_attempt")
    if active:
        if active.get("barcode"):
            print(
                f"Active: episode_{active['episode_id']:04d} | {active['name']} | "
                f"item {active['ordinal']}/{active['target_episodes']} | "
                f"attempt {active['attempt_number']}"
            )
        else:
            print(
                f"Active: episode_{active['episode_id']:04d} | "
                "waiting for the first accepted barcode"
            )


if __name__ == "__main__":
    main()
