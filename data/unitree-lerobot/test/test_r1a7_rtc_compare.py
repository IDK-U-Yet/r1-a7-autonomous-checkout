# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import pytest
from unitree_lerobot.r1a7_rtc_compare import summarize, outcomes, STAGES


def test_old_logs_do_not_invent_model_latency_and_reset_seams(tmp_path):
    path = tmp_path / "log.jsonl"
    cmd = lambda x, t: {"event": "rtc_command", "executed_action": [x]*16,
                        "task": {"phase": "scan"}, "wall_time_ns": t}
    rows = [cmd(0, 0), {"event": "rtc_chunk_merged", "guidance_delay_steps": 7, "inference_ms": 368},
            cmd(.1, 100000000), {"event": "rtc_queue_wait"}, cmd(1, 200000000)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    result = summarize(path)
    assert result["guided_metrics"]["model_latency_ms"]["count"] == 0
    assert result["guided_metrics"]["chunk_processing_ms"]["p50"] == 368
    assert result["seam_max_arm_target_jump_rad"]["max"] == .1
    assert result["command_interval_ms"]["count"] == 1


def test_unknown_outcomes_are_not_failures(tmp_path):
    path = str((tmp_path / "log.jsonl").resolve())
    row = {"log_path": path, "trial_id": "1", **{k: "success" for k in STAGES}, "human_intervention": "no"}
    unknown = {**row, "trial_id": "2", **{k: "unknown" for k in STAGES}}
    result = outcomes([row, unknown], path)
    assert result["fully_reviewed_trials"] == 1
    assert result["unassisted_success_rate"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        outcomes([row, row], path)
