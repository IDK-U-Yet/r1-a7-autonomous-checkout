# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from __future__ import annotations

import numpy as np
import pytest

from unitree_lerobot.r1a7_live_action_conditioning import (
    ActionChunkBlender,
    build_start_pose_command,
    interpolate_start_pose,
    prepare_safe_real_action,
    start_pose_errors,
)


def test_chunk_boundary_blends_three_commands_from_previous_target():
    blender = ActionChunkBlender(steps=3)
    first = np.zeros(16, dtype=np.float64)
    target = np.full(16, 0.1, dtype=np.float64)
    target[14:] = 3.0

    output, metadata = blender.apply(first, new_chunk=True)
    np.testing.assert_allclose(output, first)
    assert metadata["active"] is False

    outputs = []
    metadata = []
    for index in range(3):
        output, details = blender.apply(target, new_chunk=index == 0)
        outputs.append(output)
        metadata.append(details)

    np.testing.assert_allclose(outputs[0], target / 3)
    np.testing.assert_allclose(outputs[1], target * 2 / 3)
    np.testing.assert_allclose(outputs[2], target)
    assert [item["alpha"] for item in metadata] == pytest.approx([1 / 3, 2 / 3, 1])


def test_zero_chunk_blend_steps_disables_blending():
    blender = ActionChunkBlender(steps=0)
    blender.apply(np.zeros(16), new_chunk=True)
    target = np.full(16, 0.1, dtype=np.float64)
    target[14:] = 1.0
    output, metadata = blender.apply(target, new_chunk=True)

    np.testing.assert_allclose(output, target)
    assert metadata == {"new_chunk": True, "active": False, "alpha": 1.0, "steps": 0}


def test_start_pose_command_opens_grippers():
    measured = np.zeros(16, dtype=np.float64)
    reference = np.full(16, 0.1, dtype=np.float64)
    reference[14:] = (5.29, 5.34)

    target = build_start_pose_command(measured, reference)

    np.testing.assert_allclose(target[:14], reference[:14])
    np.testing.assert_allclose(target[14:], 5.4)
    halfway = interpolate_start_pose(measured, target, 0.5)
    np.testing.assert_allclose(halfway, (measured + target) / 2)
    assert start_pose_errors(reference, reference) == (0.0, 0.0)


def test_real_action_clips_small_arm_overshoot_but_rejects_large_one():
    raw = np.zeros(16, dtype=np.float64)
    raw[1] = -0.24
    raw[15] = 5.45

    safe, clipping = prepare_safe_real_action(raw, max_arm_overshoot=0.05)

    assert safe[1] == pytest.approx(-0.22689)
    assert safe[15] == pytest.approx(5.4)
    assert [item["joint"] for item in clipping] == [
        "kLeftShoulderRoll",
        "kRightGripper",
    ]

    raw[1] = -0.30
    with pytest.raises(ValueError, match="maximum clippable overshoot"):
        prepare_safe_real_action(raw, max_arm_overshoot=0.05)


def test_dagger_finite_overshoot_clips_without_rejecting():
    raw = np.zeros(16)
    raw[1] = -0.30
    safe, clipping = prepare_safe_real_action(raw, clip_arm_overshoot=True)
    assert safe[1] == pytest.approx(-0.22689)
    assert clipping[0]["joint"] == "kLeftShoulderRoll"
    raw[1] = np.nan
    with pytest.raises(ValueError):
        prepare_safe_real_action(raw, clip_arm_overshoot=True)
