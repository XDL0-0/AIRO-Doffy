"""Config wiring and startup cleanup for the classic teleop runtime."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import doffy_teleop.runtime.classic as classic
import doffy_teleop.runtime.publisher as publisher
import doffy_teleop.robots.teleop as robot_teleop
from doffy_teleop.config import Config
from doffy_teleop.visualization.config import VisualizerConfig


def test_classic_and_publisher_camera_factories_forward_config(monkeypatch) -> None:
    config = SimpleNamespace(VIDEO_TRANSPORT="webrtc")
    captured = []

    def make_camera(*, config):
        captured.append(config)
        return object()

    monkeypatch.setattr(classic, "WebRTCUDPManager", make_camera)
    assert classic._create_camera_manager(config) is not None

    import doffy_teleop.media.webrtc_manager as WebRTC_udp

    monkeypatch.setattr(WebRTC_udp, "WebRTCUDPManager", make_camera)
    assert publisher._create_camera_manager(config) is not None
    assert captured == [config, config]


@pytest.mark.parametrize("transport", ["udp", "webrtc"])
def test_classic_camera_factory_forwards_config_for_each_transport(
    monkeypatch, transport
) -> None:
    config = SimpleNamespace(VIDEO_TRANSPORT=transport)
    captured = []

    class FakeCamera:
        def __init__(self, *, config):
            captured.append(config)

    monkeypatch.setattr(classic, "UDPManager", FakeCamera)
    monkeypatch.setattr(classic, "WebRTCUDPManager", FakeCamera)
    classic._create_camera_manager(config)
    assert captured == [config]


def test_webrtc_factories_keep_no_argument_monkeypatch_compatibility(
    monkeypatch,
) -> None:
    config = SimpleNamespace(VIDEO_TRANSPORT="webrtc")
    classic_result = object()
    publisher_result = object()
    monkeypatch.setattr(
        classic,
        "WebRTCUDPManager",
        lambda: classic_result,
    )
    assert classic._create_camera_manager(config) is classic_result

    import doffy_teleop.media.webrtc_manager as WebRTC_udp

    monkeypatch.setattr(
        WebRTC_udp,
        "WebRTCUDPManager",
        lambda: publisher_result,
    )
    assert publisher._create_camera_manager(config) is publisher_result


def test_legacy_robot_factory_receives_entry_config(monkeypatch) -> None:
    config = object()
    captured = []

    def factory(received_config):
        captured.append(received_config)
        raise RuntimeError("stop after checking config")

    monkeypatch.setattr(robot_teleop, "make_robot_backend", factory)
    with pytest.raises(RuntimeError, match="checking config"):
        robot_teleop.RobotTeleop([], cfg=config)
    assert captured == [config]


def test_legacy_robot_closes_backend_if_initialization_fails() -> None:
    from doffy_teleop.robots.legacy.runtime import RobotTeleop

    events = []

    class FakeBackend:
        robot = None
        ik_solver = None
        gripper = None
        hand = None
        dof = 6
        tcp_tool = "None"
        dataset_robot_type = "fake"

        def initial_joint_configuration(self, _joint):
            return [0.0] * self.dof

        def reset(self, _joint):
            raise RuntimeError("robot reset failed")

        def cleanup(self):
            events.append("backend-closed")

    with pytest.raises(RuntimeError, match="reset failed"):
        RobotTeleop([], cfg=Config(), backend=FakeBackend())
    assert events == ["backend-closed"]


class _FakeManager:
    def __init__(self, events, failure=None):
        self.events = events
        self.failure = failure
        self.camera_num = 1
        self._lock = threading.Lock()
        self.data = None
        self.hand_data = None
        self.data_collecting_state = False
        self.data_export_state = False
        self.data_rollback_state = False

    def test_connection(self):
        self.events.append("connection")
        if self.failure == "connection":
            raise RuntimeError("connection failed")
        return []

    def start_comms_threads(self):
        self.events.append("comms-started")

    def close(self):
        self.events.append("camera-closed")


class _FakeTeleop:
    def __init__(self, events):
        self.events = events
        self.dof = 6
        self.backend = SimpleNamespace(
            dataset_robot_type="fake", supports_force=False
        )
        self.gripper_enabled = False
        self.wrench_mode = False
        self.reset_sign = False

    def step(self, *_args, **_kwargs):
        self.events.append("step")

    def close(self):
        self.events.append("teleop-closed")


class _FakeDataset:
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append("dataset-closed")


class _FakeRecordingService:
    def __init__(self, events, *, fail=False, stop_on_start=False):
        self.events = events
        self.fail = fail
        self.stop_on_start = stop_on_start

    def start(self, stop_event):
        self.events.append("recording-started")
        if self.fail:
            raise RuntimeError("recording failed")
        if self.stop_on_start:
            stop_event.set()
        return []

    def handle_visualizer_commands(self, _handle):
        pass

    def close(self):
        self.events.append("recording-closed")


def _config(*, visualizer_enabled=False):
    config = Config()
    config.VIDEO_TRANSPORT = "udp"
    config.beaver_enable = False
    config.TACTILE_ENABLE = False
    config.TACTILE_TRANSFER = False
    config.FORCE_COLLECT = False
    config.TORQUE_COLLECT = False
    return config, VisualizerConfig(ENABLED=visualizer_enabled)


@pytest.mark.parametrize(
    "failure", ["connection", "robot", "recording", "visualizer"]
)
def test_startup_failure_closes_every_acquired_resource(monkeypatch, failure) -> None:
    events = []
    config, viz_cfg = _config(visualizer_enabled=failure == "visualizer")
    manager = _FakeManager(events, failure=failure)
    dataset = _FakeDataset(events)
    recording = _FakeRecordingService(events, fail=failure == "recording")

    monkeypatch.setattr(classic, "_create_camera_manager", lambda _cfg: manager)

    def make_teleop(_initial_data, *, cfg, visualizer_config):
        assert cfg is config
        assert visualizer_config is viz_cfg
        events.append("robot-created")
        if failure == "robot":
            raise RuntimeError("robot failed")
        return _FakeTeleop(events)

    monkeypatch.setattr(classic, "RobotTeleop", make_teleop)
    monkeypatch.setattr(classic, "DatasetRecorder", lambda *_a, **_kw: dataset)

    def make_recording(*_args, **_kwargs):
        events.append("recording-created")
        if failure == "recording":
            raise RuntimeError("recording construction failed")
        return recording

    monkeypatch.setattr(classic, "DataRecordingService", make_recording)
    monkeypatch.setattr(classic.time, "sleep", lambda _seconds: None)

    if failure == "visualizer":
        import doffy_teleop.visualization.dashboard as visualizer

        class FakeVisualizer:
            def close(self):
                events.append("visualizer-closed")

        class FailingThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("visualizer thread failed")

            def join(self, **_kwargs):
                raise RuntimeError("visualizer thread was not started")

        monkeypatch.setattr(
            visualizer,
            "start_visualizer",
            lambda **_kwargs: FakeVisualizer(),
        )
        monkeypatch.setattr(classic.threading, "Thread", FailingThread)

    with pytest.raises(RuntimeError):
        classic.main(config=config, visualizer_config=viz_cfg)

    assert "camera-closed" in events
    assert ("teleop-closed" in events) == (
        failure in {"recording", "visualizer"}
    )
    if failure == "recording":
        assert "dataset-closed" in events
    if failure == "visualizer":
        assert "recording-closed" in events
        assert "visualizer-closed" in events


def test_stop_event_from_recording_service_exits_main_loop(monkeypatch) -> None:
    events = []
    config, viz_cfg = _config()
    manager = _FakeManager(events)
    teleop = _FakeTeleop(events)
    dataset = _FakeDataset(events)
    recording = _FakeRecordingService(events, stop_on_start=True)

    monkeypatch.setattr(classic, "_create_camera_manager", lambda _cfg: manager)
    monkeypatch.setattr(
        classic,
        "RobotTeleop",
        lambda *_args, **_kwargs: teleop,
    )
    monkeypatch.setattr(classic, "DatasetRecorder", lambda *_a, **_kw: dataset)
    monkeypatch.setattr(classic, "DataRecordingService", lambda *_a, **_kw: recording)
    monkeypatch.setattr(classic.time, "sleep", lambda _seconds: None)

    classic.main(config=config, visualizer_config=viz_cfg)

    assert "step" not in events
    assert "camera-closed" in events
    assert "recording-closed" in events
    assert "teleop-closed" in events


def test_keyboard_interrupt_runs_normal_cleanup(monkeypatch) -> None:
    events = []
    config, viz_cfg = _config()
    manager = _FakeManager(events)
    teleop = _FakeTeleop(events)
    dataset = _FakeDataset(events)
    recording = _FakeRecordingService(events)
    sleep_calls = 0

    def sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(classic, "_create_camera_manager", lambda _cfg: manager)
    monkeypatch.setattr(
        classic,
        "RobotTeleop",
        lambda *_args, **_kwargs: teleop,
    )
    monkeypatch.setattr(classic, "DatasetRecorder", lambda *_a, **_kw: dataset)
    monkeypatch.setattr(classic, "DataRecordingService", lambda *_a, **_kw: recording)
    monkeypatch.setattr(classic.time, "sleep", sleep)

    classic.main(config=config, visualizer_config=viz_cfg)

    assert "camera-closed" in events
    assert "recording-closed" in events
    assert "teleop-closed" in events


def test_control_loop_period_uses_injected_rate(monkeypatch) -> None:
    events = []
    config, viz_cfg = _config()
    config.UR_CTRL_RATE = 50
    manager = _FakeManager(events)
    teleop = _FakeTeleop(events)
    dataset = _FakeDataset(events)
    recording = _FakeRecordingService(events)
    loop_sleeps = []

    def sleep(duration):
        if duration != 5:
            loop_sleeps.append(duration)
            raise KeyboardInterrupt

    monkeypatch.setattr(classic, "_create_camera_manager", lambda _cfg: manager)
    monkeypatch.setattr(
        classic,
        "RobotTeleop",
        lambda *_args, **_kwargs: teleop,
    )
    monkeypatch.setattr(classic, "DatasetRecorder", lambda *_a, **_kw: dataset)
    monkeypatch.setattr(classic, "DataRecordingService", lambda *_a, **_kw: recording)
    monkeypatch.setattr(classic, "MIN_DT", 0.5)
    monkeypatch.setattr(classic.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(classic.time, "sleep", sleep)

    classic.main(config=config, visualizer_config=viz_cfg)

    assert loop_sleeps == pytest.approx([0.02])


def test_root_main_forwards_replaced_configs_and_factory_points(monkeypatch) -> None:
    import doffy_teleop.runtime.classic_entrypoint as main

    config, viz_cfg = _config()
    captured = {}

    monkeypatch.setattr(main, "cfg", config)
    monkeypatch.setattr(main, "viz_cfg", viz_cfg)
    monkeypatch.setattr(
        main._classic,
        "main",
        lambda **kwargs: captured.update(kwargs),
    )

    main.main()

    assert captured == {"config": config, "visualizer_config": viz_cfg}
