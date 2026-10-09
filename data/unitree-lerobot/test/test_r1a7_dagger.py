# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import numpy as np
import pytest
from unitree_lerobot.r1a7_dagger import Ownership, OwnershipRejected, RelativeAlignment


def test_late_model_action_rejected_even_after_new_alignment():
    owner = Ownership()
    epoch = owner.epoch
    owner.transition("hold")
    owner.transition("human")
    with pytest.raises(OwnershipRejected):
        owner.check("policy", epoch)
    owner.transition("hold")
    owner.transition("human")
    with pytest.raises(OwnershipRejected):
        owner.check("human", 2)
    with pytest.raises(OwnershipRejected):
        owner.transition("policy")


def test_alignment_has_no_pose_jump_and_preserves_grip():
    robot = np.eye(4)
    robot[:3, 3] = [0.2, 0.3, 0.4]
    xr = np.eye(4)
    xr[:3, 3] = [-0.2, 1, 0.1]
    alignment = RelativeAlignment([robot, robot], [xr, xr], [1.1, 4.0])
    for pose in alignment.poses([xr, xr]):
        np.testing.assert_allclose(pose, robot)
    np.testing.assert_allclose(alignment.grippers([10, 10]), [1.1, 4])
    np.testing.assert_allclose(alignment.grippers([0, 0]), [0, 0])
    np.testing.assert_allclose(alignment.grippers([10, 10]), [5.4, 5.4])


def test_restart_epoch_never_reuses_old_owner():
    owner = Ownership()
    old = owner.epoch
    owner.transition("human")
    owner.transition("hold")
    owner.restart_trial()
    assert owner.source == "policy" and not owner.human_started
    assert owner.epoch > old
    with pytest.raises(OwnershipRejected):
        owner.check("policy", old)


def test_fading_alignment_starts_at_robot_and_finishes_at_ordinary_xr():
    from scipy.spatial.transform import Rotation
    robot, xr = np.eye(4), np.eye(4)
    robot[:3, 3] = [0.25, 0.15, 0.3]
    robot[:3, :3] = Rotation.from_rotvec([0.1, 0.2, -0.1]).as_matrix()
    xr[:3, 3] = [0.35, 0.15, 0.3]
    xr[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    align = RelativeAlignment([robot] * 2, [xr] * 2, [2, 3], blend_seconds=3, now=0)
    np.testing.assert_allclose(align.poses([xr] * 2, now=0)[0], robot, atol=1e-12)
    assert not align.blend_snapshot()["expert_action_valid"]
    previous = robot
    for tick in range(1, 93):
        current = align.poses([xr] * 2, now=tick / 30)[0]
        assert np.linalg.norm(current[:3, 3] - previous[:3, 3]) <= 0.1 / 30 + 1e-10
        angle = Rotation.from_matrix(previous[:3, :3].T @ current[:3, :3]).magnitude()
        assert angle <= 0.6 / 30 + 1e-10
        np.testing.assert_allclose(current[:3, :3].T @ current[:3, :3], np.eye(3), atol=1e-12)
        previous = current
    np.testing.assert_allclose(current, xr, atol=1e-12)
    assert align.blend_snapshot()["expert_action_valid"]
    moved = xr.copy()
    moved[:3, 3] += [0.05, 0, 0]
    moved[:3, :3] = Rotation.from_rotvec([0, 0.1, 0]).as_matrix() @ xr[:3, :3]
    np.testing.assert_allclose(align.poses([moved] * 2, now=4)[0], moved, atol=1e-12)


def test_fading_duration_extends_for_offset_and_stalls_cannot_jump():
    from scipy.spatial.transform import Rotation
    robot, xr = np.eye(4), np.eye(4)
    xr[0, 3] = 0.5
    xr[:3, :3] = Rotation.from_rotvec([0, np.pi, 0]).as_matrix()
    align = RelativeAlignment([robot] * 2, [xr] * 2, [2, 3], blend_seconds=3, now=0)
    assert align.blend_duration >= 1.875 * 0.5 / 0.1
    assert align.blend_duration >= 1.875 * np.pi / 0.6
    align.poses([xr] * 2, now=100)
    assert align.blend_elapsed == pytest.approx(1 / 30)
    assert align.blend_weight < 0.001
    with pytest.raises(ValueError, match="clock"):
        align.poses([xr] * 2, now=99)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_bad_blend_duration_rejected(value):
    from unitree_lerobot.r1a7_dagger import takeover_blend_seconds
    with pytest.raises(ValueError):
        takeover_blend_seconds(value)
    with pytest.raises(ValueError):
        RelativeAlignment([np.eye(4)] * 2, [np.eye(4)] * 2, [2, 3], blend_seconds=value)


def test_expert_export_excludes_automatic_fade_but_keeps_legacy_frames():
    from unitree_lerobot.r1a7_dagger_recording import expert_segments
    frames = [{"metadata": {"control_source": "human", "owner_epoch": 1,
               "active_task": "scan", "sample_valid": True,
               "record_monotonic_ns": i * 66_666_667}} for i in range(4)]
    frames[0]["metadata"]["expert_action_valid"] = False
    frames[1]["metadata"]["expert_action_valid"] = False
    frames[2]["metadata"]["expert_action_valid"] = True
    assert expert_segments(frames) == [frames[2:]]


@pytest.mark.parametrize("name", ["", "..", "/tmp/out", "a/b", "../other"])
def test_batch_label_cannot_escape_root(tmp_path, name):
    from unitree_lerobot.r1a7_dagger_recording import collection_path
    with pytest.raises(ValueError):
        collection_path(tmp_path, name)


def test_batch_name_defaults_to_dataset_folder():
    from unitree_lerobot.r1a7_dagger_recording import collection_name
    name = "r1a7_oreo_4cam_16d_v2v3_standard160_train"
    assert collection_name(f"/data/datasets/{name}/") == name
    assert collection_name(f"/data/datasets/{name}", "custom_batch") == "custom_batch"
    with pytest.raises(ValueError):
        collection_name("/data/datasets/example", "../escape")


def test_discard_only_removes_closed_current_trial(tmp_path):
    from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder
    old = tmp_path / "dagger_old"
    old.mkdir()
    (old / "keep.txt").write_text("keep")
    recorder = DaggerRecorder(tmp_path, "test")
    with pytest.raises(RuntimeError, match="writer is active"):
        recorder.discard()
    recorder.finish(completed=False)
    recorder.discard()
    assert not recorder.path.exists()
    assert (old / "keep.txt").read_text() == "keep"


def test_discard_refuses_redirected_path(tmp_path):
    from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder
    recorder = DaggerRecorder(tmp_path / "batch", "test")
    recorder.finish(completed=False)
    original = recorder.path
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("keep")
    recorder.path = elsewhere
    with pytest.raises(RuntimeError, match="refusing deletion"):
        recorder.discard()
    assert original.exists() and (elsewhere / "keep.txt").exists()


def test_discard_retains_failed_recording(tmp_path):
    from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder
    recorder = DaggerRecorder(tmp_path, "test")
    recorder.finish(completed=False)
    recorder.finalized = False
    with pytest.raises(RuntimeError, match="finalization failed"):
        recorder.discard()
    assert recorder.path.exists()


def test_pico_restart_requires_between_trials_fresh_xr_and_no_fault():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_restart_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    pico.owner = Ownership()
    pico.owner.transition("human")
    pico.disarmed = False
    pico.controller = SimpleNamespace(fault_reason=None)
    pico.between_trials = False
    pico.xr_ready = True
    with pytest.raises(OwnershipRejected):
        pico.prepare_restart()
    pico.between_trials = True
    pico.xr_ready = False
    with pytest.raises(OwnershipRejected):
        pico.prepare_restart()
    pico.xr_ready = True
    pico.controller.fault_reason = "lost feedback"
    with pytest.raises(OwnershipRejected):
        pico.prepare_restart()
    pico.controller.fault_reason = None
    pico.prepare_restart()
    assert pico.owner.source == "policy" and pico.initializing
    assert not pico.between_trials and pico.alignment is None


def test_relative_translation_in_robot_basis():
    robot, xr = np.eye(4), np.eye(4)
    robot[:3, 3] = [0.1, 0.2, 0.3]
    align = RelativeAlignment([robot, robot], [xr, xr], [5.4, 5.4])
    current = xr.copy()
    current[2, 3] = -0.05
    np.testing.assert_allclose(align.poses([current, current])[0][:3, 3], [0.1, 0.2, 0.25])
    current[2, 3] = -1
    with pytest.raises(ValueError):
        align.poses([current, current])


def test_bad_trigger_and_pose_rejected():
    align = RelativeAlignment([np.eye(4)] * 2, [np.eye(4)] * 2, [5.4, 5.4])
    with pytest.raises(ValueError):
        align.grippers([float("nan"), 10])
    with pytest.raises(ValueError):
        align.poses([np.zeros((4, 4)), np.eye(4)])


def test_recorder_and_expert_export_keep_only_human(tmp_path):
    import cv2
    import json
    from unitree_lerobot.r1a7_contract import CAMERA_KEYS
    from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder, export_expert
    images = {key: np.zeros((3, 16, 16), dtype=np.uint8) for key in CAMERA_KEYS}
    for image in images.values():
        image[0] = 255
    recorder = DaggerRecorder(tmp_path / "raw", "checkpoint-test")
    states = np.r_[np.zeros(14), [2, 3]]
    for i, source in enumerate(["policy", "hold", "human", "human", "human"]):
        recorder.add(states, images, states, {"control_source": source, "owner_epoch": 2,
                     "active_task": "scan", "sample_valid": True,
                     "record_monotonic_ns": 1_000_000_000 + i * 66_666_667})
    recorder.finish(completed=True, accepted_event_id="event-test")
    result = export_expert(tmp_path / "raw", tmp_path / "expert", min_frames=2)
    assert len(result) == 1 and result[0]["frames"] == 3
    episode = json.loads((tmp_path / "expert/episode_0000/data.json").read_text())
    assert all(f["metadata"]["control_source"] == "human" for f in episode["data"])
    assert episode["data"][0]["metadata"]["source_frame_idx"] == 2
    pixel = cv2.imread(str(tmp_path / "expert/episode_0000" / episode["data"][0]["colors"]["color_0"]))[0, 0]
    assert pixel[2] > 250 and pixel[0] < 5  # RGB red remains BGR red, no channel swap.
    with pytest.raises(FileExistsError):
        export_expert(tmp_path / "raw", tmp_path / "expert", min_frames=2)


def test_export_boundaries_and_incomplete_trials(tmp_path):
    from unitree_lerobot.r1a7_dagger_recording import expert_segments, DaggerRecorder, export_expert
    frames = []
    for i, (source, epoch, task, continuity) in enumerate([
        ("human", 2, "scan", 0), ("human", 2, "scan", 0),
        ("hold", 3, "scan", 0), ("human", 4, "scan", 0),
        ("human", 4, "place", 0), ("human", 4, "place", 1),
    ]):
        frames.append({"metadata": {"control_source": source, "owner_epoch": epoch,
                      "active_task": task, "continuity_id": continuity, "sample_valid": True,
                      "record_monotonic_ns": i * 66_666_667}})
    assert [len(s) for s in expert_segments(frames)] == [2, 1, 1, 1]
    incomplete = DaggerRecorder(tmp_path / "raw", "model")
    incomplete.finish(completed=False)
    assert export_expert(tmp_path / "raw", tmp_path / "expert", min_frames=1) == []


def test_storage_error_not_silently_accepted(tmp_path, monkeypatch):
    import time
    from unitree_lerobot.r1a7_dagger_recording import DaggerRecorder
    from unitree_lerobot.r1a7_contract import CAMERA_KEYS
    import unitree_lerobot.r1a7_dagger_recording as recording
    monkeypatch.setattr(recording.cv2, "imwrite", lambda *a, **kw: False)
    recorder = DaggerRecorder(tmp_path, "model")
    recorder.add(np.zeros(16), {key: np.zeros((3, 4, 4), dtype=np.uint8) for key in CAMERA_KEYS},
                 np.zeros(16), {})
    deadline = time.monotonic() + 2
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(RuntimeError):
        recorder.check()
    with pytest.raises(RuntimeError):
        recorder.finish()
    assert not (recorder.path / "manifest.json").exists()


def test_pico_ik_uses_collection_result_without_residual_hold():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import time
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_ik_collection_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    tele = SimpleNamespace(motion_data_ready=True, motion_data_timestamp=time.monotonic(),
                           right_ctrl_aButton=False, right_ctrl_bButton=False,
                           left_ctrl_aButton=False, left_ctrl_bButton=False,
                           left_wrist_pose=np.eye(4), right_wrist_pose=np.eye(4),
                           left_ctrl_triggerValue=0, right_ctrl_triggerValue=0)
    pico.xr = SimpleNamespace(get_tele_data=lambda: tele)
    pico.disarmed = False
    pico.enabled = True
    pico.initializing = False
    pico.between_trials = False
    pico.next_tick = 0
    pico.last_host_heartbeat = time.monotonic()
    pico.last_buttons = pico.last_finish_buttons = (False, False)
    pico.operator_finish = None
    pico.update_start_button = lambda *args: None
    pico.owner = Ownership()
    pico.owner.transition("human")
    goal = np.eye(4)
    goal[:3, 3] = [1, 1, 1]  # Deliberately unreachable by this fake solver.
    pico.alignment = SimpleNamespace(poses=lambda _: [goal, goal],
                                     grippers=lambda _: np.array([2., 3.]))
    solved = np.zeros(14)
    pico.ik = SimpleNamespace(solve_ik=lambda *args: (solved, solved))
    executed, held = [], []
    pico.controller = SimpleNamespace(fault_reason=None,
        arm=SimpleNamespace(get_current_dual_arm_q=lambda: solved,
                            get_current_dual_arm_dq=lambda: solved),
        execute=lambda action, **kw: executed.append(action))
    pico.hold = held.append
    def no_residual_fk(_):
        raise AssertionError("Extra FK residual gate must not run")
    pico.fk = no_residual_fk
    pico.tick()
    assert not held and pico.owner.source == "human"
    np.testing.assert_allclose(executed[0], np.r_[solved, 2, 3])


def test_pico_ik_commit_rechecks_freshness_and_disarm():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import time
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_pico_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    tele = SimpleNamespace(motion_data_ready=True, motion_data_timestamp=time.monotonic(),
                           right_ctrl_aButton=False, right_ctrl_bButton=False,
                           left_ctrl_aButton=False, left_ctrl_bButton=False)
    pico.xr = SimpleNamespace(get_tele_data=lambda: tele)
    pico.disarmed = False
    pico.last_host_heartbeat = time.monotonic()
    pico.last_buttons = (False, False)
    held, faults = [], []
    pico.controller = SimpleNamespace(fault_reason=None, latch_fault=faults.append)
    pico.hold = held.append
    assert pico.commit_ready(tele)
    def updated_during_read():
        tele.motion_data_timestamp = time.monotonic()
        return tele
    pico.xr.get_tele_data = updated_during_read
    assert pico.commit_ready(tele)  # Validation clock must be sampled after XR read.
    pico.last_buttons = (True, False)
    assert pico.commit_ready(tele)  # A release during IK must not force HOLD.
    pico.last_buttons = (False, False)
    old = SimpleNamespace(motion_data_timestamp=time.monotonic() - 0.3)
    assert not pico.commit_ready(old)
    assert held
    tele.right_ctrl_bButton = True
    assert not pico.commit_ready(tele)
    pico.last_host_heartbeat = time.monotonic() - 1
    assert not pico.commit_ready(tele)
    assert faults
    pico.disarmed = True
    assert not pico.commit_ready(tele)


@pytest.mark.parametrize("x,y,expected", [(True, False, "S"), (False, True, "Q"), (True, True, "Q")])
def test_pico_completion_buttons_hold_and_latch(x, y, expected):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    import time
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_finish_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    pico.disarmed = False
    pico.next_tick = 0
    pico.last_buttons = pico.last_finish_buttons = (False, False)
    pico.operator_finish = None
    pico.enabled = True
    pico.initializing = False
    pico.last_host_heartbeat = time.monotonic()
    tele = SimpleNamespace(motion_data_ready=True, motion_data_timestamp=time.monotonic(),
                           right_ctrl_aButton=False, right_ctrl_bButton=False,
                           left_ctrl_aButton=x, left_ctrl_bButton=y, right_ctrl_squeeze=False)
    def updated_during_read():
        tele.motion_data_timestamp = time.monotonic()
        return tele
    pico.xr = SimpleNamespace(get_tele_data=updated_during_read)
    held = []
    pico.hold = held.append
    pico.controller = SimpleNamespace(hold_current=lambda **kwargs: None)
    pico.tick()
    assert pico.operator_finish == expected
    assert len(held) == 1
    pico.next_tick = 0
    pico.tick()
    assert len(held) == 1  # Holding the button is not a second request.


def test_start_grip_latches_analog_edge_without_held_start():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_start_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    pico.enabled = False
    pico.initializing = True
    pico.right_squeeze = False
    pico.start_press_sequence = 0
    pico.previous_start_squeeze = None
    tele = SimpleNamespace(right_ctrl_squeeze=False, right_ctrl_squeezeValue=1.0)
    pico.update_start_button(tele, True)
    assert pico.start_press_sequence == 0
    tele.right_ctrl_squeezeValue = 0.0
    pico.update_start_button(tele, True)
    tele.right_ctrl_squeezeValue = 1.0
    pico.update_start_button(tele, True)
    tele.right_ctrl_squeezeValue = 0.0
    pico.update_start_button(tele, True)
    assert pico.start_press_sequence == 1  # Short press survives release until coordinator reads.
    pico.enabled = True
    tele.right_ctrl_squeezeValue = 1.0
    pico.update_start_button(tele, True)
    assert pico.start_press_sequence == 1


def test_right_grip_restart_only_on_fresh_hold_edge():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[3] / "robot/xr-teleoperate/teleop/tools/r1a7_dagger_pico.py"
    spec = importlib.util.spec_from_file_location("dagger_restart_button_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pico = module.DaggerPico.__new__(module.DaggerPico)
    pico.enabled = True
    pico.initializing = False
    pico.between_trials = False
    pico.owner = Ownership()
    pico.previous_start_squeeze = False
    pico.right_squeeze = False
    pico.restart_press_sequence = 0
    tele = SimpleNamespace(right_ctrl_squeeze=False, right_ctrl_squeezeValue=1.0)
    pico.update_start_button(tele, True)  # POLICY press is ignored.
    assert pico.restart_press_sequence == 0
    pico.owner.transition("hold")
    pico.update_start_button(tele, True)  # Held across HOLD transition is not a new press.
    assert pico.restart_press_sequence == 0
    tele.right_ctrl_squeezeValue = 0.0
    pico.update_start_button(tele, True)
    tele.right_ctrl_squeezeValue = 1.0
    pico.update_start_button(tele, True)
    pico.update_start_button(tele, True)
    assert pico.restart_press_sequence == 1
    tele.right_ctrl_squeezeValue = 0.0
    pico.update_start_button(tele, True)
    tele.right_ctrl_squeezeValue = 1.0
    pico.update_start_button(tele, False)  # Stale stream cannot request restart.
    assert pico.restart_press_sequence == 1
    pico.update_start_button(tele, True)  # Fresh-but-held stream still cannot.
    assert pico.restart_press_sequence == 1


def test_bounded_queue_backpressure_stops_instead_of_dropping(tmp_path, monkeypatch):
    import threading
    import time
    import unitree_lerobot.r1a7_dagger_recording as recording
    from unitree_lerobot.r1a7_contract import CAMERA_KEYS
    entered, release = threading.Event(), threading.Event()
    original = recording.cv2.imwrite

    def stalled(*args, **kwargs):
        entered.set()
        release.wait(2)
        return original(*args, **kwargs)

    monkeypatch.setattr(recording.cv2, "imwrite", stalled)
    recorder = recording.DaggerRecorder(tmp_path, "checkpoint", queue_size=1)
    deadline = time.monotonic() + 1
    while not recorder.queue.empty() and time.monotonic() < deadline:
        time.sleep(0.001)
    images = {k: np.zeros((3, 4, 4), dtype=np.uint8) for k in CAMERA_KEYS}
    try:
        recorder.add(np.zeros(16), images, np.zeros(16), {})
        assert entered.wait(1)
        recorder.add(np.zeros(16), images, np.zeros(16), {})
        with pytest.raises(RuntimeError, match="queue full"):
            recorder.add(np.zeros(16), images, np.zeros(16), {})
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="incomplete recording"):
            recorder.finish(completed=True, accepted_event_id="accepted")
        assert not (recorder.path / "manifest.json").exists()
