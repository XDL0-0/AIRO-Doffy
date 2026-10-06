"""Import and compatibility checks for the modular PC teleop runtime."""

from __future__ import annotations

import numpy as np

import doffy_teleop.runtime.classic_entrypoint as main
import doffy_teleop.runtime.realman_cli as realman_teleop
import doffy_teleop.robots.backend_api as robot_backend
import doffy_teleop.robots.teleop as robot_teleop
from doffy_teleop.control.canfd_loop import CanfdCommandLoop as PackageCanfdLoop
from doffy_teleop.control.contracts import (
    CanfdLoopSnapshot as PackageCanfdSnapshot,
    QuestTcpStateSender as PackageQuestSender,
)
from doffy_teleop.control.realman_qp import RealManRemoteIkSolver as PackageRemoteIk
from doffy_teleop.robots.backends import RealManBackend as PackageRealManBackend
from doffy_teleop.robots.contracts import RobotBackend as PackageRobotBackend
from doffy_teleop.robots.legacy.runtime import RobotTeleop as PackageRobotTeleop
from doffy_teleop.runtime.classic_helpers import (
    TactileDataHolder as PackageTactileDataHolder,
    _visualizer_image as package_visualizer_image,
)
from doffy_teleop.runtime.realman import RealManTeleop as PackageRealManTeleop
from doffy_teleop.recording.service import RecordingFrame


def test_legacy_root_modules_resolve_to_package_implementations() -> None:
    """Historical imports remain backed by the focused package modules."""
    assert issubclass(realman_teleop.RealManTeleop, PackageRealManTeleop)
    assert issubclass(robot_teleop.RobotTeleop, PackageRobotTeleop)
    assert robot_teleop.URTeleop is robot_teleop.RobotTeleop
    assert robot_backend.RobotBackend is PackageRobotBackend
    assert robot_backend.RealManBackend is PackageRealManBackend

    assert realman_teleop.CanfdCommandLoop is PackageCanfdLoop
    assert realman_teleop.CanfdLoopSnapshot is PackageCanfdSnapshot
    assert realman_teleop.QuestTcpStateSender is PackageQuestSender
    assert realman_teleop.RealManRemoteIkSolver is PackageRemoteIk


def test_classic_root_exports_keep_public_helpers_without_hardware() -> None:
    """Classic helpers can be imported and exercised without robot devices."""
    assert main.TactileDataHolder is PackageTactileDataHolder
    assert main.RecordingFrame is RecordingFrame

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    preview = main._visualizer_image(image)
    expected = package_visualizer_image(image)
    np.testing.assert_array_equal(preview, expected)
    assert preview is not image


def test_controller_reset_helper_uses_configured_threshold() -> None:
    data = [None, {
        "Joystick_Press": True,
        "IndexTrigger": main.cfg.CONTROLLER_RESET_TRIGGER_THRESHOLD,
    }]
    assert main.controller_reset_requested(data)
    assert not main.controller_reset_requested(None)


def test_controller_reset_helper_follows_replaced_root_config(monkeypatch) -> None:
    """The historical ``main.cfg`` replacement point remains live."""
    from types import SimpleNamespace

    replacement = SimpleNamespace(CONTROLLER_RESET_TRIGGER_THRESHOLD=1.0)
    monkeypatch.setattr(main, "cfg", replacement)
    below_threshold = [None, {"Joystick_Press": True, "IndexTrigger": 0.9}]
    at_threshold = [None, {"Joystick_Press": True, "IndexTrigger": 1.0}]

    assert not main.controller_reset_requested(below_threshold)
    assert main.controller_reset_requested(at_threshold)


def test_realman_root_main_forwards_legacy_replacement_points(monkeypatch) -> None:
    """Root-level integration patches reach the modular composition root."""
    import doffy_teleop.runtime.realman_entrypoint as entrypoint

    captured = {}

    def fake_run_main(**dependencies) -> None:
        captured.update(dependencies)

    monkeypatch.setattr(entrypoint, "main", fake_run_main)
    replacements = {
        "Config": object(),
        "VisualizerConfig": object(),
        "_create_camera_manager": object(),
        "RealManTeleop": object(),
        "RealManEpisodeRecorder": object(),
        "QuestTcpStateSender": object(),
        "visualizer_publish_loop": object(),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(realman_teleop, name, value)

    realman_teleop.main()

    assert captured == {
        "config_factory": replacements["Config"],
        "visualizer_config_factory": replacements["VisualizerConfig"],
        "camera_manager_factory": replacements["_create_camera_manager"],
        "teleop_factory": replacements["RealManTeleop"],
        "recorder_factory": replacements["RealManEpisodeRecorder"],
        "tcp_state_sender_factory": replacements["QuestTcpStateSender"],
        "visualizer_publish_loop_fn": replacements["visualizer_publish_loop"],
    }


def test_realman_composition_root_uses_injected_dependencies() -> None:
    """The explicit composition parameters work without opening hardware."""
    import threading
    from types import SimpleNamespace

    import doffy_teleop.runtime.realman_entrypoint as entrypoint

    calls = []
    cfg = SimpleNamespace(
        TELEOP_COMMAND_MODE="joint",
        REALMAN_CTRL_RATE=200.0,
        VIDEO_TRANSPORT="udp",
        WRM_enable=False,
        beaver_enable=False,
        FORCE_ENABLE=False,
    )
    viz_cfg = SimpleNamespace(ENABLED=False)

    class FakeCamera:
        camera_num = 0

        def __init__(self) -> None:
            self._lock = threading.Lock()

        def test_connection(self):
            calls.append("camera.test_connection")
            return [None, {"Position": [0.0, 0.0, 0.0], "Rotation": [0.0, 0.0, 0.0, 1.0]}]

        def start_comms_threads(self) -> None:
            calls.append("camera.start")

        def close(self) -> None:
            calls.append("camera.close")

    class FakeCanfd:
        def wait_until_healthy(self, _stop_event):
            return SimpleNamespace(achieved_hz=200.0)

        def snapshot(self):
            return SimpleNamespace(error="")

    class FakeTeleop:
        def __init__(self, initial_data, *, cfg) -> None:
            calls.append(("teleop", initial_data, cfg))
            self.canfd = FakeCanfd()

        def start(self, stop_event):
            calls.append("teleop.start")
            stop_event.set()
            return []

        def sdk_worker_threads(self):
            return ()

        def live_sdk_workers(self):
            return []

        def close(self) -> None:
            calls.append("teleop.close")

    class FakeRecorder:
        def __init__(self, *args, **kwargs) -> None:
            calls.append(("recorder", args, kwargs))

        def start(self, _stop_event):
            calls.append("recorder.start")
            return []

        def close(self) -> None:
            calls.append("recorder.close")

    entrypoint.main(
        config_factory=lambda: cfg,
        visualizer_config_factory=lambda: viz_cfg,
        camera_manager_factory=lambda _cfg: FakeCamera(),
        teleop_factory=FakeTeleop,
        recorder_factory=FakeRecorder,
        tcp_state_sender_factory=object,
        visualizer_publish_loop_fn=lambda *_args: None,
    )

    assert calls[0] == "camera.test_connection"
    assert calls[1][0] == "teleop"
    assert calls[1][2] is cfg
    assert "camera.start" in calls
    assert "teleop.start" in calls
    assert "recorder.start" in calls
    assert calls[-3:] == ["camera.close", "recorder.close", "teleop.close"]


def test_invalid_realman_config_does_not_open_robot_connection(monkeypatch) -> None:
    """Reject an incompatible profile before any backend/hardware side effect."""
    from types import SimpleNamespace
    import pytest

    opened = []
    monkeypatch.setattr(realman_teleop, "make_robot_backend", lambda cfg: opened.append(cfg))
    for cfg in (
        SimpleNamespace(ROBOT_TYPE="ur"),
        SimpleNamespace(ROBOT_TYPE="realman", GRIPPER=True),
        SimpleNamespace(ROBOT_TYPE="realman", GRIPPER=False, TACTILE_TRANSFER=True),
    ):
        with pytest.raises(ValueError):
            realman_teleop.RealManTeleop([], cfg=cfg)
    assert not opened
