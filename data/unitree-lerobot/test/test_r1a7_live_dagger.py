# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""End-to-end coordinator test with no sockets, model assets or hardware."""
import json
import time
from pathlib import Path
import numpy as np
import pytest
from unitree_lerobot.r1a7_contract import CAMERA_KEYS
import unitree_lerobot.r1a7_live_dagger as live


def test_blend_duration_cli_rejects_nonfinite_and_negative():
    parser = live.build_dagger_parser()
    for value in ("nan", "inf", "-1"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--takeover-blend-seconds", value])
    assert parser.get_default("takeover_blend_seconds") == 3.0


@pytest.mark.parametrize("during_ready", [False, True])
def test_b_during_restart_returns_to_waiting_hold(monkeypatch, during_ready):
    import io
    import threading
    from types import SimpleNamespace
    reply = {"status": "rejected", "fault_locked": False, "control_source": "hold",
             "owner_epoch": 84, "takeover_reason": "operator B"}
    requests = []
    def request(payload):
        requests.append(payload["request"])
        if payload["request"] == "dagger_start_ready":
            return {**reply, "status": "ok", "fault_reason": None}
        assert payload["request"] == "dagger_between_trials"
        return {"status": "ok", "owner_epoch": 85, "control_source": "hold"}
    commands = SimpleNamespace(strict_rejections=False, owner_epoch=83, _request=request)
    def returning(*args):
        assert commands.strict_rejections
        if not during_ready:
            raise live.ReturnInterrupted(reply)
    monkeypatch.setattr(live, "return_to_scan_start_pose", returning)
    audit = io.StringIO()
    assert not live.return_for_restart(commands, None, None, None, audit, threading.Event())
    assert requests[-1] == "dagger_between_trials"
    assert commands.owner_epoch == 85 and not commands.strict_rejections
    assert json.loads(audit.getvalue())["event"] == "dagger_return_interrupted"


@pytest.mark.parametrize("reply", [
    {"status": "rejected", "fault_locked": True, "control_source": "hold", "takeover_reason": "operator B"},
    {"status": "rejected", "fault_locked": False, "control_source": "hold", "takeover_reason": "stale XR"},
])
def test_unrelated_return_rejections_still_raise(reply):
    from types import SimpleNamespace
    assert not live.operator_b_hold(reply)
    client = live.DaggerClient.__new__(live.DaggerClient)
    client.owner_epoch = 83
    client.strict_rejections = True
    client._socket = SimpleNamespace(send_json=lambda _: None, recv_json=lambda: reply)
    with pytest.raises(RuntimeError) as exc:
        client._request({"request": "command"})
    assert not isinstance(exc.value, live.ReturnInterrupted)


def test_strict_b_rejection_is_typed_interruption():
    from types import SimpleNamespace
    client = live.DaggerClient.__new__(live.DaggerClient)
    client.owner_epoch = 83
    client.strict_rejections = True
    client._socket = SimpleNamespace(send_json=lambda _: None, recv_json=lambda: {
        "status": "rejected", "fault_locked": False, "control_source": "hold",
        "owner_epoch": 84, "takeover_reason": "operator B"})
    with pytest.raises(live.ReturnInterrupted):
        client._request({"request": "command"})


@pytest.mark.parametrize("finish", ["console", "S", "Q", "grip_start", "skip_invalid", "two_cycles", "scan_later", "grip_restart"])
def test_model_rollout_human_takeover_record_and_complete(tmp_path, monkeypatch, finish):
    commands_after_takeover = []
    clients = []
    event_handler = []
    delivered = []
    signal_handlers = {}
    human_at = 7 if finish == "skip_invalid" else 5
    end_at = human_at + 4

    class Client:
        def __init__(self, *args):
            self.owner_epoch = 0
            self.strict_rejections = True
            self.cycles = 0
            self.disarmed = False
            self.finished = False
            self.restarts = 0
            self.completion_refused = False
            self.restart_sequence = 0
            self.restart_ready = False
            clients.append(self)

        def snapshot(self):
            now = time.monotonic_ns()
            human = self.cycles >= human_at
            return {"status": "ok", "owner_epoch": 2 if human else 0,
                    "control_source": "hold" if self.finished else "human" if human else "policy",
                    "command_source": "human" if human else "policy",
                    "command_owner_epoch": 2 if human else 0,
                    "commanded_action": [0.] * 16, "requested_action": [0.] * 16,
                    "command_monotonic_ns": now, "state": [0.] * 16,
                    "sample_monotonic_ns": {"lowstate": now, "left_gripper": now, "right_gripper": now},
                    "trigger_pickup": [True, True], "xr_ready": True, "fault_reason": None,
                    "restart_press_sequence": self.restart_sequence,
                    "operator_finish": ("S" if finish in {"grip_start", "skip_invalid", "two_cycles", "scan_later", "grip_restart"} else finish)
                    if self.cycles >= end_at and finish != "console" and not self.finished else None}

        def _request(self, payload):
            if payload["request"] == "dagger_ack_finish":
                self.completion_refused = True
            if payload["request"] == "dagger_between_trials":
                self.finished = True
                self.restart_ready = False
            if payload["request"] == "dagger_prepare_restart":
                self.restarts += 1
                self.finished = False
                self.cycles = 1
            if payload["request"] == "dagger_heartbeat":
                self.cycles += 1
                if finish == "grip_restart" and self.finished and self.restart_ready and not self.restarts:
                    self.restart_sequence = 1  # Short press occurs after finalization.
                if (self.cycles == 4 and finish != "scan_later") or (finish == "scan_later" and self.cycles == end_at + 1):
                    event_handler[0]({"barcode": "test", "event_id": f"accepted-test-{self.restarts}",
                                      "source_wall_time_ns": time.time_ns()})
            return self.snapshot()

        def command(self, action, **kwargs):
            delivered.append(np.asarray(action).copy())
            if self.cycles >= human_at:
                commands_after_takeover.append(action)
            return {"status": "ok"}

        def hold(self):
            return {"status": "ok"}

        def disarm(self):
            self.disarmed = True

        def close(self):
            pass

    class State:
        def __init__(self, *args):
            self.reads = 0

        def read(self):
            self.reads += 1
            if clients[0].finished:
                clients[0].restart_ready = True
            result = clients[0].snapshot()
            result["right_squeeze"] = self.reads >= 4
            result["start_press_sequence"] = self.reads if finish == "grip_start" else 0
            return result

        def close(self):
            pass

    class Console:
        def poll(self):
            if clients[0].finished:
                if finish == "grip_restart" and not clients[0].restarts:
                    return None  # Restart must come from grip, not the keyboard.
                if finish == "two_cycles" and not clients[0].restarts:
                    # Scanner input between trials must not arm the next trial.
                    reply = event_handler[0]({"barcode": "test", "event_id": "stale",
                                             "source_wall_time_ns": time.time_ns()})
                    assert reply["status"] == "rejected"
                    return "R"
                signal_handlers[live.signal.SIGINT]()
                return None
            return "G" if clients[0].cycles == 0 and finish != "grip_start" else "S" if clients[0].cycles >= end_at and finish == "console" else None

    class Session:
        n_action_steps = 10
        chunk_size = 50

        def __init__(self, **kwargs):
            pass

        def reset(self):
            pass

        def predict_chunk(self, *args):
            time.sleep(0.005)
            result = np.zeros((50, 16))
            result[:, 14:] = 5.4224  # Real model's slight grip overshoot must clip before blending.
            if finish == "skip_invalid":
                result[0, 0] = 100.0  # Finite overshoot now clips instead of skipping/HOLD.
            return result

    class Images:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class Events:
        def __init__(self, endpoint, handler):
            event_handler.append(handler)

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setenv("R1A7_REAL_INFERENCE", "YES")
    monkeypatch.setattr(live, "DaggerClient", Client)
    monkeypatch.setattr(live, "StateBridgeClient", State)
    monkeypatch.setattr(live, "Console", Console)
    monkeypatch.setattr(live, "PI05Session", Session)
    monkeypatch.setattr(live, "ImageClient", Images)
    monkeypatch.setattr(live, "TaskEventServer", Events)
    monkeypatch.setattr(live, "load_task_start_reference", lambda *args: (np.zeros(16), 80))
    monkeypatch.setattr(live, "return_to_scan_start_pose", lambda *args: np.zeros(16))
    monkeypatch.setattr(live, "_read_cameras", lambda *args: {})
    monkeypatch.setattr(live, "assemble_live_observation", lambda *args, **kwargs: (
        np.zeros(16), {k: np.zeros((3, 16, 16), dtype=np.uint8) for k in CAMERA_KEYS}, {}))
    monkeypatch.setattr(live.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(live.signal, "signal", lambda sig, handler: signal_handlers.update({sig: handler}))
    argv = [
        "--dataset-root", str(tmp_path / "dataset"), "--policy-path", str(tmp_path / "model"),
        "--tokenizer-path", str(tmp_path / "tokenizer"), "--expected-barcode", "test",
        "--output", str(tmp_path / "audit.jsonl"), "--dagger-record-dir", str(tmp_path / "raw"),
        "--confirm-real-robot", live.REAL_ROBOT_CONFIRMATION,
    ]
    if finish == "console":
        argv += ["--collection-name", "cashier_oreo_16d_v3_standard_80"]
    assert live.main(argv) == 0
    assert not commands_after_takeover
    assert clients[0].disarmed
    if finish == "scan_later":
        assert clients[0].completion_refused
    manifests = list((tmp_path / "raw").glob("*/*/manifest.json"))
    if finish == "Q":
        assert not manifests
        assert not list((tmp_path / "raw/dataset").glob("dagger_*"))
        return
    assert len(manifests) == (2 if finish in {"two_cycles", "grip_restart"} else 1)
    if finish in {"two_cycles", "grip_restart"}:
        assert {json.loads(p.read_text())["accepted_event_id"] for p in manifests} == {"accepted-test-0", "accepted-test-1"}
        assert len({json.loads(p.read_text())["source_trial_id"] for p in manifests}) == 2
    manifest = json.loads(manifests[0].read_text())
    assert manifest["completed"] is (finish != "Q")
    assert manifest["accepted_event_id"].startswith("accepted-test-")
    assert manifest["collection_name"] == ("cashier_oreo_16d_v3_standard_80" if finish == "console" else "dataset")
    data = json.loads((manifests[0].parent / "data.json").read_text())
    sources = {f["metadata"]["control_source"] for f in data["data"]}
    assert sources == {"policy", "human"}
    human = [f for f in data["data"] if f["metadata"]["control_source"] == "human"]
    assert len(human) >= 3
    if finish == "skip_invalid":
        records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
        assert not any(r["event"] == "dagger_prediction_rejected" for r in records)
        assert any(r["event"] == "dagger_model_command" and
                   any(c["joint"] == "kLeftShoulderPitch" for c in r["clipping"])
                   for r in records)
        assert any(r["event"] == "dagger_model_command" for r in records)
        assert all(abs(row[0]) < 100 for row in delivered)
