# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import io
import json
import numpy as np

from unitree_lerobot.r1a7_rtc_offline import run_async_probe


def test_async_probe_uses_no_hardware_interfaces():
    class FakeSession:
        chunk_size = 50
        def reset(self):
            pass
        def predict_normalized(self, *args, **kwargs):
            return np.zeros((50,16))
        def to_hardware_action(self, action):
            return action.copy()
    log = io.StringIO()
    run_async_probe(FakeSession(), np.zeros(16), {}, "scan", log, 2)
    rows = [json.loads(row) for row in log.getvalue().splitlines()]
    assert rows[-1]["event"] == "rtc_offline_async_complete"
    assert rows[-1]["simulated_actions"] == 2
    assert all(row["hardware_commands_sent"] is False for row in rows)
