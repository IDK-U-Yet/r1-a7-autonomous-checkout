# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Read-only RTC log analysis. Target continuity is not manipulation success."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

STAGES = ("grasp", "handover_flip", "barcode_presentation", "scan", "place")
FIELDS = ("model_latency_ms", "chunk_processing_ms", "ceil_model_delay_steps",
          "ceil_processing_delay_steps", "conditioned_delay_steps",
          "consumed_delay_steps", "reserved_delay_steps")


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    def percentile(q):
        i = (len(values) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return {"count": len(values), **({"p50": percentile(.5), "p95": percentile(.95),
                                    "max": values[-1]} if values else {})}


def summarize(path):
    counts, metrics = Counter(), {k: [] for k in FIELDS}
    ready, previous, seam = {}, None, False
    jumps, intervals = [], []
    resets = {"rtc_paused", "rtc_resumed", "rtc_restarted", "rtc_queue_wait",
              "rtc_camera_sync_wait", "rtc_action_rejected", "rtc_fault", "rtc_item_accepted"}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON; copy a complete log") from exc
            event = row.get("event", "unknown")
            counts[event] += 1
            if event == "rtc_ready":
                ready = row
            if event in resets or event.startswith("manual_cycle"):
                previous, seam = None, False
            if event == "rtc_chunk_merged":
                guided = row.get("conditioned_delay_steps", row.get("guidance_delay_steps", 0)) > 0
                if not guided:
                    previous, seam = None, False
                else:
                    for key in FIELDS:
                        value = row.get(key)
                        # Old logs only measured the whole worker, not isolated model latency.
                        if key == "chunk_processing_ms":
                            value = row.get(key, row.get("inference_ms"))
                        if key == "conditioned_delay_steps":
                            value = row.get(key, row.get("guidance_delay_steps"))
                        metrics[key].append(value)
                    seam = True
            if event == "rtc_command":
                if previous is not None:
                    same_task = row.get("task") == previous.get("task")
                    if same_task:
                        if seam:
                            jumps.append(max(abs(a-b) for a, b in zip(
                                row["executed_action"][:14], previous["executed_action"][:14])))
                        clock = "monotonic_ns" if "monotonic_ns" in row and "monotonic_ns" in previous else "wall_time_ns"
                        intervals.append((row[clock] - previous[clock]) / 1e6)
                previous, seam = row, False
    return {"log_path": str(Path(path).resolve()), "parameters": ready, "events": dict(counts),
            "guided_metrics": {k: distribution(v) for k, v in metrics.items()},
            "command_interval_ms": distribution(intervals),
            "seam_max_arm_target_jump_rad": distribution(jumps)}


def outcomes(rows, log_path):
    selected = [r for r in rows if str(Path(r["log_path"]).resolve()) == log_path]
    identities = [r["trial_id"] for r in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate trial_id for the same log")
    for row in selected:
        if any(row[k] not in {"success", "failure", "not_reached", "unknown", ""} for k in STAGES):
            raise ValueError("stage must be success/failure/not_reached/unknown")
        if row["human_intervention"] not in {"yes", "no", "unknown", ""}:
            raise ValueError("human_intervention must be yes/no/unknown")
    reviewed = [r for r in selected if all(r[k] in {"success", "failure", "not_reached"} for k in STAGES)
                and r["human_intervention"] in {"yes", "no"}]
    stages = {}
    for stage in STAGES:
        attempted = [r for r in selected if r[stage] in {"success", "failure"}]
        success = sum(r[stage] == "success" for r in attempted)
        stages[stage] = {"attempted": len(attempted), "successful": success,
                         "conditional_rate": success / len(attempted) if attempted else None,
                         "all_reviewed_trial_rate": sum(r[stage] == "success" for r in reviewed) / len(reviewed) if reviewed else None}
    unassisted = sum(all(r[k] == "success" for k in STAGES) and r["human_intervention"] == "no" for r in reviewed)
    return {"template_rows": len(selected), "fully_reviewed_trials": len(reviewed),
            "unassisted_successes": unassisted,
            "unassisted_success_rate": unassisted / len(reviewed) if reviewed else None,
            "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--template-trials", type=int, default=10)
    args = parser.parse_args()
    if args.template_trials < 0:
        parser.error("template-trials must be nonnegative")
    summaries = [summarize(p) for p in args.logs]
    if args.outcomes:
        with args.outcomes.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for summary in summaries:
            summary["outcomes"] = outcomes(rows, summary["log_path"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["# RTC 对比", "", "连续性指下发目标的跳变，不代表真机反馈或任务成功。缺失指标不估算；bootstrap 不计入带前缀延迟。", "",
             "| 日志 | s | β | 去噪 n | 模型 P50/P95 ms | 整块 P50/P95 ms | 交接跳变 P95 rad | 等待 ticks | 故障 |", "|---|---:|---:|---:|---|---|---:|---:|---:|"]
    def pair(metric):
        return f"{metric['p50']:.1f}/{metric['p95']:.1f}" if metric["count"] else "未记录"
    for s in summaries:
        m, e = s["guided_metrics"], s["events"]
        jump = s["seam_max_arm_target_jump_rad"].get("p95")
        table.append(f"| {Path(s['log_path']).name} | {s['parameters'].get('execution_steps', '?')} | {s['parameters'].get('max_guidance_weight', '?')} | {s['parameters'].get('num_inference_steps', '?')} | {pair(m['model_latency_ms'])} | {pair(m['chunk_processing_ms'])} | {f'{jump:.4f}' if jump is not None else '无'} | {e.get('rtc_queue_wait', 0)} | {e.get('rtc_fault', 0)} |")
    if args.outcomes:
        table.extend(["", "## 人工试验结果", "", "未审核试验不计入完整成功率，阶段条件成功率见 summary.json。", "",
                      "| 日志 | 完整已审核试验 | 无介入成功次数 | 无介入完整成功率 |", "|---|---:|---:|---:|"])
        for s in summaries:
            o = s["outcomes"]
            rate = o["unassisted_success_rate"]
            table.append(f"| {Path(s['log_path']).name} | {o['fully_reviewed_trials']} | {o['unassisted_successes']} | {f'{rate:.1%}' if rate is not None else '未审核'} |")
    (args.output_dir / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    template = args.output_dir / "trial_outcomes.csv"
    # Never overwrite the operator's annotations on re-analysis.
    if not template.exists():
        with template.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["log_path", "trial_id", "orientation", *STAGES, "human_intervention", "notes"])
            writer.writeheader()
            for summary in summaries:
                for i in range(1, args.template_trials + 1):
                    writer.writerow({"log_path": summary["log_path"], "trial_id": i,
                                     **{k: "unknown" for k in (*STAGES, "human_intervention")}})
    print(f"Comparison saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
