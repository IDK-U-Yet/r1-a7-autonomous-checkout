# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "lerobot" / "prepare_r1a7_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_r1a7_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_allowed_mask_validates_counts():
    payload = {
        "total_frames": 6,
        "allowed_training_starts": 4,
        "excluded_ranges": [{"dataset_start": 2, "dataset_end_exclusive": 4}],
    }
    assert MODULE._allowed_mask(payload, 6).tolist() == [True, True, False, False, True, True]

    payload["allowed_training_starts"] = 5
    with pytest.raises(ValueError, match="ranges produce 4"):
        MODULE._allowed_mask(payload, 6)


def test_prepare_dataset_uses_only_allowed_numeric_rows(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    values = [np.array([float(index), float(index + 10)], dtype=np.float32) for index in range(6)]
    pd.DataFrame({"observation.state": values, "action": values}).to_parquet(
        source / "data" / "chunk-000" / "file-000.parquet"
    )
    info = {
        "codebase_version": "v3.0",
        "total_frames": 6,
        "features": {
            "observation.state": {"shape": [2]},
            "action": {"shape": [2]},
        },
    }
    (source / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (source / "meta" / "stats.json").write_text(
        json.dumps({"observation.state": {}, "action": {}}), encoding="utf-8"
    )
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(
        json.dumps(
            {
                "total_frames": 6,
                "allowed_training_starts": 4,
                "excluded_ranges": [{"dataset_start": 2, "dataset_end_exclusive": 4}],
            }
        ),
        encoding="utf-8",
    )

    audit = MODULE.prepare_dataset(source, output, training_filter=filter_path)
    stats = json.loads((output / "meta" / "stats.json").read_text(encoding="utf-8"))

    assert audit["sample_selection"]["selected_frames"] == 4
    assert audit["sample_selection"]["excluded_frames"] == 2
    assert stats["action"]["count"] == [4]
    assert stats["action"]["mean"] == pytest.approx([2.5, 12.5])
    assert (output / "meta" / "training_filter.json").is_file()
