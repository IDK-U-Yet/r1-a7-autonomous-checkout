# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import math
from pathlib import Path
import time

import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class DiagnosticsLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, **values):
        record = {"monotonic_time": time.monotonic()}
        record.update({key: _json_value(value) for key, value in values.items()})
        self._stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

    def close(self):
        self._stream.close()
