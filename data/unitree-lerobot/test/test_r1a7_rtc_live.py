# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import pytest
import json
import sys
from types import SimpleNamespace
import numpy as np
import unitree_lerobot.r1a7_rtc_live as live
from unitree_lerobot.r1a7_rtc_live import DispatchGate, build_parser


def test_task_ack_invalidates_dispatch():
    gate = DispatchGate()
    revision = gate.revision
    called = []
    gate.invalidate()
    assert gate.dispatch(revision, lambda: called.append(True)) is None
    assert called == []


def test_pause_resume_does_not_revive_old_actions():
    gate = DispatchGate()
    old = gate.revision
    gate.invalidate(paused=True)
    assert gate.dispatch(gate.revision, lambda: True) is None
    gate.invalidate(paused=False)
    assert gate.dispatch(old, lambda: True) is None
    assert gate.dispatch(gate.revision, lambda: True) is True


def test_real_cli_requires_confirmation():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("feedback_fault", [False, True])
@pytest.mark.parametrize("gripper_roundoff", [False, True])
def test_mock_main_dispatch_or_fault_always_disarms(monkeypatch, tmp_path, feedback_fault, gripper_roundoff):
    calls = []

    class Commands:
        def __init__(self, endpoint):
            pass
        def command(self, action):
            calls.append("command")
            return {"executed_action": action.tolist(), "hardware_commands_sent": True}
        def hold(self):
            return {"hardware_commands_sent": True}
        def disarm(self):
            calls.append("disarm")
        def close(self):
            calls.append("close")

    class State:
        def __init__(self, endpoint, timeout_ms):
            assert timeout_ms == 1000
        def read(self):
            if feedback_fault:
                raise RuntimeError("lowstate timeout")
            return {"state": np.zeros(16)}
        def close(self):
            pass

    class Session:
        chunk_size = 50
        def __init__(self, **kwargs):
            pass
        def reset(self):
            pass
        def predict_normalized(self, *args, **kwargs):
            return np.zeros((50,16))
        def to_hardware_action(self, normalized):
            result = np.r_[np.zeros(14), [5.4000001,5.4000001] if gripper_roundoff else [1,1]]
            if gripper_roundoff:
                result[5] = 1.61440003  # projected wrist boundary, float32 roundoff
            return result

    monkeypatch.setenv("R1A7_REAL_INFERENCE", "YES")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(live, "RTCConsole", lambda: SimpleNamespace(poll=lambda: []))
    monkeypatch.setattr(live, "RealCommandClient", Commands)
    monkeypatch.setattr(live, "StateBridgeClient", State)
    monkeypatch.setattr(live, "RTCSession", Session)
    monkeypatch.setattr(live, "ImageClient", lambda **_: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(live, "TaskEventServer", lambda *_: SimpleNamespace(start=lambda: None, stop=lambda: None))
    monkeypatch.setattr(live, "load_task_start_reference", lambda *_: (np.zeros(16), 160))
    monkeypatch.setattr(live, "return_to_scan_start_pose", lambda *_: np.zeros(16))
    monkeypatch.setattr(live, "_read_cameras", lambda *_: {})
    monkeypatch.setattr(live, "assemble_live_observation", lambda *_, **__: (np.zeros(16), {}, {}))
    path = tmp_path / "mock.jsonl"
    monkeypatch.setattr(sys, "argv", ["rtc", "--dataset-root", str(tmp_path),
        "--policy-path", str(tmp_path), "--tokenizer-path", str(tmp_path),
        "--output", str(path), "--max-commands", "20", "--confirm-real-robot", live.REAL_ROBOT_CONFIRMATION])
    if feedback_fault:
        with pytest.raises(RuntimeError, match="lowstate timeout"):
            live.main()
        assert "command" not in calls
    else:
        live.main()
        assert calls.count("command") == 20
    assert calls[-2:] == ["disarm", "close"]
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    assert any(row["event"] == ("rtc_fault" if feedback_fault else "rtc_command") for row in rows)
    if not feedback_fault:
        names = [row["event"] for row in rows]
        assert names.index("rtc_warmup_complete") < names.index("rtc_ready") < names.index("rtc_command")
        assert next(row for row in rows if row["event"] == "rtc_warmup_complete")["policy_commands_sent"] is False
        assert any(row.get("guidance_delay_steps", 0) > 0 for row in rows)
        assert all(row.get("generation", 0) == 0 for row in rows if row["event"] == "rtc_chunk_merged")


def test_startup_camera_retry_preserves_limits_and_reads_state_first(monkeypatch):
    import threading
    calls, records = [], []
    state = SimpleNamespace(read=lambda: calls.append("state") or {})
    commands = SimpleNamespace(hold=lambda: calls.append("hold"))
    monkeypatch.setattr(live, "_read_cameras", lambda _: calls.append("frames") or {})
    attempts = []
    def assemble(*args, **kwargs):
        assert kwargs["max_camera_age_ms"] == 250
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("stale camera frame observation.images.cam_global: age_ms=692")
        return np.zeros(16), {}, {}
    monkeypatch.setattr(live, "assemble_live_observation", assemble)
    result = live.wait_fresh_observation(state, None, commands, threading.Event(), records.append)
    assert calls == ["hold", "state", "frames", "hold", "state", "frames"]
    assert result[2]["timing"]["state_read_ms"] >= 0
    assert records[0]["event"] == "rtc_observation_error"


@pytest.mark.parametrize("reason", ["stale robot state: lowstate", "camera shape mismatch for global"])
def test_startup_does_not_retry_feedback_fault_or_invalid_camera(monkeypatch, reason):
    import threading
    def fail(*args):
        raise RuntimeError(reason)
    monkeypatch.setattr(live, "read_observation", fail)
    with pytest.raises(RuntimeError, match=reason):
        live.wait_fresh_observation(None, None, SimpleNamespace(hold=lambda: None),
                                   threading.Event(), lambda _: None)


@pytest.mark.parametrize("reason", ["cross-camera skew too large: 194.977 ms",
    "stale camera frame observation.images.cam_scanner: age_ms=251.282"])
def test_runtime_camera_fault_holds_and_discards_before_fresh_observation(monkeypatch, reason):
    import threading
    calls, records = [], []
    attempts = []
    def read(*args):
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError(reason)
        assert calls[0] == "invalidate"
        assert calls[1] == "hold"
        return np.zeros(16), {}, {}
    monkeypatch.setattr(live, "read_observation", read)
    result = live.read_rtc_observation(None, None,
        SimpleNamespace(hold=lambda: calls.append("hold")),
        SimpleNamespace(invalidate=lambda: calls.append("invalidate")),
        threading.Event(), records.append)
    assert result is not None
    assert calls.count("invalidate") == 1
    assert [r["event"] for r in records] == ["rtc_camera_sync_wait", "rtc_camera_sync_restored"]


@pytest.mark.parametrize("reason", ["lowstate timeout", "stale robot state: lowstate",
    "camera shape mismatch for global", "camera timestamp unavailable: global"])
def test_runtime_non_recoverable_fault_is_not_retried(monkeypatch, reason):
    import threading
    def read(*args):
        raise RuntimeError(reason)
    monkeypatch.setattr(live, "read_observation", read)
    with pytest.raises(RuntimeError, match=reason):
        live.read_rtc_observation(None, None, None, None, threading.Event(), lambda _: None)


@pytest.mark.parametrize("reason", ["cross-camera skew too large: 195 ms",
    "stale camera frame observation.images.cam_scanner: age_ms=251.282"])
def test_persistent_camera_sync_fault_times_out(monkeypatch, reason):
    import threading
    def read(*args):
        raise RuntimeError(reason)
    monkeypatch.setattr(live, "read_observation", read)
    with pytest.raises(RuntimeError, match="cross-camera skew|stale camera frame"):
        live.wait_fresh_observation(None, None, SimpleNamespace(hold=lambda: None),
            threading.Event(), lambda _: None, timeout=0)


def test_stale_recovery_has_half_second_budget_and_propagates_feedback_fault(monkeypatch):
    import threading
    calls, records = [], []
    def read(*args):
        raise RuntimeError("stale camera frame observation.images.cam_scanner: age_ms=251.282")
    def wait(*args, timeout):
        assert timeout == .5
        assert calls[:2] == ["invalidate", "hold"]
        raise RuntimeError("stale robot state: lowstate")
    monkeypatch.setattr(live, "read_observation", read)
    monkeypatch.setattr(live, "wait_fresh_observation", wait)
    with pytest.raises(RuntimeError, match="stale robot state"):
        live.read_rtc_observation(None, None,
            SimpleNamespace(hold=lambda: calls.append("hold")),
            SimpleNamespace(invalidate=lambda: calls.append("invalidate")),
            threading.Event(), records.append)
    assert len(records) == 1 and records[0]["timeout_ms"] == 500
