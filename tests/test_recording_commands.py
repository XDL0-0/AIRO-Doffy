"""Regression checks for ordered recording mutations and camera-lock ownership."""

from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from doffy_teleop.media.udp import UDPManagerCore
from doffy_teleop.media.webrtc import WebRTCUDPManagerCore
from doffy_teleop.recording_control import RecordingControl, ManagerRecordingControl
from doffy_teleop.config import Config
from doffy_teleop.recording.service import DataRecordingService, RecordingFrame
from doffy_teleop.recording.dataset import DatasetRecorder


class MemoryDataset:
    def __init__(self):
        self.recorded_episodes = 0
        self.collect_step = 0
        self.events = []
        self.closed = False

    def data_collection(self, **kwargs):
        self.collect_step += 1

    def data_export(self, _context):
        self.events.append("Stop")
        self.recorded_episodes += 1

    def _reset_data_dict(self):
        self.collect_step = 0

    def rollback_last_episode(self):
        self.events.append("Undo")
        if self.collect_step:
            self.collect_step = 0
        else:
            self.recorded_episodes = max(0, self.recorded_episodes - 1)
        return True

    def close(self):
        self.closed = True


def frame():
    return RecordingFrame(np.zeros(7), np.ones(7), {})


def service_for(dataset=None, control=None):
    return DataRecordingService(dataset or MemoryDataset(), frame, 30, control=control)


def test_second_undo_arriving_during_io_is_not_erased():
    dataset = MemoryDataset()
    dataset.recorded_episodes = 2
    service = service_for(dataset)
    entered, release = threading.Event(), threading.Event()
    rollback = dataset.rollback_last_episode

    def delayed_rollback():
        entered.set()
        assert release.wait(2)
        return rollback()

    dataset.rollback_last_episode = delayed_rollback
    service.request_rollback()
    worker = threading.Thread(target=service.process_pending_once, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        service.request_rollback()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert service.control.snapshot()[2]
    assert service.process_pending_once()
    assert dataset.events == ["Undo", "Undo"]
    assert dataset.recorded_episodes == 0
    assert not service.process_pending_once()
    service.close()


def test_restart_during_export_waits_and_does_not_mix_episodes():
    service = service_for()
    entered, release = threading.Event(), threading.Event()
    export = service.dataset.data_export

    def delayed_export(context):
        entered.set()
        assert release.wait(2)
        export(context)

    service.dataset.data_export = delayed_export
    assert service.start_recording()
    assert service.collect_once()
    assert service.stop_recording()
    worker = threading.Thread(target=service.process_pending_once, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        assert service.start_recording()
        assert not service.collecting
        assert not service.collect_once()
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    assert service.collecting
    assert service.dataset.collect_step == 0
    assert service.collect_once()
    assert service.dataset.collect_step == 1
    service.close()


def test_stop_undo_start_are_applied_in_order_before_new_frames():
    service = service_for()
    service.start_recording()
    service.collect_once()
    service.stop_recording()
    service.request_rollback()
    service.start_recording()
    assert not service.collect_once()
    assert service.process_pending_once()
    assert not service.collect_once()
    assert service.process_pending_once()
    assert service.dataset.events == ["Stop", "Undo"]
    assert service.dataset.recorded_episodes == 0
    assert service.collect_once()
    service.close()


def test_close_drains_all_already_accepted_requests():
    service = service_for()
    service.start_recording()
    service.collect_once()
    service.stop_recording()
    service.request_rollback()
    service.close()
    service.close()
    assert service.dataset.events == ["Stop", "Undo"]
    assert service.dataset.recorded_episodes == 0
    assert service.dataset.closed
    assert not service.start_recording()
    assert not service.control.start_recording()
    assert not service.control.request_rollback()
    assert not service.collect_once()


class StubSocket:
    def __init__(self, **kwargs):
        self.messages = []
        self.closed = False

    def read_all(self):
        result, self.messages = self.messages, []
        return result

    def read(self):
        return self.messages.pop(0) if self.messages else None

    def close(self):
        self.closed = True


def manager_with_plain_camera_lock(cls):
    camera = SimpleNamespace(
        _lock=threading.Lock(), camera_num=0, camera_list={}, camera_images={},
        camera_image_timestamps_ns={}, depth_mode=False, depth_images={},
        depth_timestamps_ns={}, realsense_resolution=(640, 480), realsense_fps=30,
        start=lambda: None, close=lambda: None,
    )
    return cls(Config(TACTILE_TRANSFER=False), camera_manager=camera,
               socket_module=SimpleNamespace(UdpComms=StubSocket))


@pytest.mark.parametrize("manager_class", [UDPManagerCore, WebRTCUDPManagerCore])
def test_real_manager_undo_finishes_with_nonreentrant_camera_lock(tmp_path, manager_class):
    manager = manager_with_plain_camera_lock(manager_class)
    dataset = DatasetRecorder(
        0, robot_dof=7, robot_type="realman", gripper=False,
        force_collect=False, torque_collect=False,
        config=Config(DATASET_TYPE="a", DATA_TYPE="qpos", PUSH_TO_HUB=False),
        dataset_root=tmp_path / "episodes",
    )
    service = service_for(dataset, ManagerRecordingControl(manager))
    service.export_context = manager
    errors = []

    def run():
        try:
            for _ in range(2):
                assert service.start_recording()
                assert service.collect_once()
                assert service.stop_recording()
                assert service.process_pending_once()
            manager._apply_record_control("Undo")
            assert service.process_pending_once()
            service.close()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    try:
        worker.start()
        worker.join(3)
        assert not worker.is_alive(), "recording control deadlocked on the camera lock"
        assert not errors, errors
        assert dataset.recorded_episodes == 1
        assert (Path(dataset.dataset_dir) / "episode_0.hdf5").exists()
        assert not (Path(dataset.dataset_dir) / "episode_1.hdf5").exists()
        assert not manager.data_rollback_state
        assert not service.pause_event.is_set()
    finally:
        manager.close()


@pytest.mark.parametrize("manager_class", [UDPManagerCore, WebRTCUDPManagerCore])
def test_receiver_preserves_record_command_received_before_worker_start(manager_class):
    manager = manager_with_plain_camera_lock(manager_class)
    manager.socket_list["socket_1"].messages.append("Start")
    started = threading.Event()
    original = manager._apply_record_control

    def accept(command):
        original(command)
        started.set()

    manager._apply_record_control = accept
    worker = threading.Thread(target=manager._vr_receive_thread, daemon=True)
    try:
        worker.start()
        assert started.wait(1), "startup discarded the first recording command"
        assert manager.data_collecting_state
    finally:
        manager.running = False
        worker.join(1)
        manager.close()


def test_export_exception_finishes_only_its_own_request():
    service = service_for()
    service.start_recording()
    service.collect_once()
    service.stop_recording()
    service.request_rollback()

    def fail(_context):
        raise RuntimeError("disk failure")

    service.dataset.data_export = fail
    with pytest.raises(RuntimeError, match="disk failure"):
        service.process_pending_once()
    assert service.control.snapshot()[2]
    assert not service.pause_event.is_set()
    assert service.process_pending_once()
    service.close()


@pytest.mark.parametrize("failed_command", ["Stop", "Undo"])
def test_close_drains_later_requests_and_closes_after_operation_failure(failed_command):
    service = service_for()
    service.start_recording()
    service.collect_once()

    def fail(*_args):
        service.dataset.events.append(f"failed {failed_command}")
        raise RuntimeError(f"{failed_command} disk failure")

    if failed_command == "Stop":
        service.dataset.data_export = fail
        service.stop_recording()
        service.request_rollback()
        following_command = "Undo"
    else:
        service.dataset.rollback_last_episode = fail
        service.request_rollback()
        service.stop_recording()
        following_command = "Stop"

    with pytest.raises(RuntimeError, match=f"{failed_command} disk failure"):
        service.close()
    assert service.dataset.events == [f"failed {failed_command}", following_command]
    assert service.dataset.closed
    assert service.control.snapshot() == (False, False, False)
    assert not service.pause_event.is_set()
    assert not service.collect_once()
    service.close()


def test_close_releases_dataset_when_saving_active_episode_fails():
    service = service_for()
    service.start_recording()
    service.collect_once()

    def fail(_context):
        raise RuntimeError("exit export failure")

    service.dataset.data_export = fail
    with pytest.raises(RuntimeError, match="exit export failure"):
        service.close()
    assert service.dataset.closed
    assert not service.pause_event.is_set()
    assert not service.control.start_recording()
    service.close()


def test_export_worker_reports_shutdown_failure_without_leaving_dataset_open():
    errors = []
    service = DataRecordingService(MemoryDataset(), frame, 30, background_errors=errors)
    service.request_rollback()

    def fail():
        raise RuntimeError("exit rollback failure")

    service.dataset.rollback_last_episode = fail
    stop_event = threading.Event()
    stop_event.set()
    service._export_loop(stop_event)
    assert service.dataset.closed
    assert errors == ["dataset shutdown failed: exit rollback failure"]
