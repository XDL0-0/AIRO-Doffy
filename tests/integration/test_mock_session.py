"""Complete short data-collection session composed only from mocks."""

from __future__ import annotations

import struct
import time
import unittest
from pathlib import Path

from airo_doffy.config import CameraConfig, VRConfig, VideoStreamingConfig
from airo_doffy.core import (
    ClockDomain,
    ControllerState,
    EncodedFrame,
    HandSide,
    RobotAction,
    RobotCommandType,
    VideoCodec,
    VRInputMode,
    VRInputState,
)
from airo_doffy.devices.cameras import MockCameraSource
from airo_doffy.devices.tactile import MockTactileSensor
from airo_doffy.devices.vr import MockVRInputSource
from airo_doffy.recording import (
    FrozenArray,
    NamedArray,
    RecordingSample,
    build_recording_schema,
)
from airo_doffy.robots import LatestActionExecutor, MockRobotBackend
from airo_doffy.runtime import (
    DataCollectionSession,
    RecordingCycleExtension,
    TeleopCycle,
    TeleopSession,
)
from airo_doffy.streaming.video import (
    LatestVideoEncodingPipeline,
    PackedFrameProcessor,
)
from airo_doffy.visualization import (
    MemorySnapshotRenderer,
    TypedSnapshotConsumer,
    VisualizationSnapshot,
)


def _vr_sample(sequence: int) -> VRInputState:
    controllers = tuple(
        ControllerState(
            sequence=sequence,
            source_timestamp_ns=100 + sequence,
            side=side,
            position_m=(0.01 * sequence, 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            joystick_xy=(0.0, 0.0),
            index_trigger=0.0,
            grip_trigger=1.0 if side is HandSide.RIGHT else 0.0,
            buttons=frozenset(),
        )
        for side in (HandSide.LEFT, HandSide.RIGHT)
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=100 + sequence,
        mode=VRInputMode.CONTROLLERS,
        controllers=controllers,
    )


class _JointMapping:
    def map_input(self, vr_input, robot_state, dt_s: float) -> RobotAction:
        del robot_state, dt_s
        value = 0.01 * (vr_input.sequence + 1)
        return RobotAction(
            sequence=vr_input.sequence,
            source_timestamp_ns=vr_input.source_timestamp_ns,
            clock_domain=ClockDomain.MONOTONIC,
            command_type=RobotCommandType.JOINT_POSITION,
            values=(value,) * 6,
            gripper_width_m=0.03,
        )


class _AllowAll:
    def apply(self, action, robot_state, now_ns: int):
        del robot_state, now_ns
        return action


class _MemoryEncoder:
    def __init__(self) -> None:
        self.closed = False

    def start(self) -> None:
        pass

    def encode(self, frame) -> EncodedFrame:
        return EncodedFrame(
            sequence=frame.sequence,
            source_timestamp_ns=frame.source_timestamp_ns,
            receive_timestamp_ns=frame.receive_timestamp_ns,
            clock_domain=frame.clock_domain,
            stream_id=frame.stream_id,
            data=b"h264:" + frame.data,
            codec=VideoCodec.H264,
            width=frame.shape[1],
            height=frame.shape[0],
            encoded_timestamp_ns=frame.processing_timestamp_ns,
        )

    def close(self) -> None:
        self.closed = True


class _MemoryTransport:
    def __init__(self) -> None:
        self.frames: list[EncodedFrame] = []
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def send(self, frame: EncodedFrame) -> None:
        if not self.started:
            raise RuntimeError("transport has not started")
        self.frames.append(frame)

    def close(self) -> None:
        self.started = False
        self.closed = True


class _MemoryWriter:
    def __init__(self) -> None:
        self.episodes = []
        self.closed = False

    def write(self, episode):
        self.episodes.append(episode)
        return Path(f"episode_{episode.index}.mock")

    def close(self) -> None:
        self.closed = True


class _MockPeripherals:
    """Test-only lifecycle adapter for camera, video, tactile, and display."""

    def __init__(self) -> None:
        self.camera = MockCameraSource(
            CameraConfig(resolution=(2, 1), fps=30)
        )
        self.processor = PackedFrameProcessor()
        self.encoder = _MemoryEncoder()
        self.video = LatestVideoEncodingPipeline(
            self.encoder,
            VideoStreamingConfig(
                input_queue_capacity=1,
                output_queue_capacity=1,
            ),
        )
        self.transport = _MemoryTransport()
        self.tactile = MockTactileSensor()
        self.renderer = MemorySnapshotRenderer()
        self.visualizer = TypedSnapshotConsumer(
            self.renderer,
            wait_timeout_s=0.01,
        )
        self.latest_frame = None
        self.latest_tactile = None

    def start(self) -> None:
        self.camera.start()
        self.tactile.start()
        self.video.start()
        self.transport.start()
        self.visualizer.start()

    def on_cycle(self, cycle: object) -> None:
        if not isinstance(cycle, TeleopCycle):
            raise TypeError("mock peripherals require TeleopCycle")
        source = self.camera.read_latest()
        if source is None:
            raise RuntimeError("mock camera unexpectedly returned no frame")
        self.latest_frame = self.processor.process(source)
        self.latest_tactile = self.tactile.read_latest()
        self.video.submit(self.latest_frame)
        encoded = self._wait_for_encoded(self.latest_frame.sequence)
        self.transport.send(encoded)
        snapshot = VisualizationSnapshot(
            sequence=cycle.sequence,
            source_timestamp_ns=cycle.timestamp_ns,
            robot=cycle.robot_state,
            frames=(self.latest_frame,),
            tactile=self.latest_tactile,
        )
        if not self.visualizer.publish(snapshot):
            raise RuntimeError("mock visualizer rejected a snapshot")

    def _wait_for_encoded(self, sequence: int) -> EncodedFrame:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            encoded = self.video.read_latest()
            if encoded is not None and encoded.sequence == sequence:
                return encoded
            time.sleep(0.001)
        raise AssertionError(f"video frame {sequence} was not encoded")

    def check_health(self) -> None:
        if self.video.health_error is not None:
            raise RuntimeError("mock encoder failed") from self.video.health_error
        error = self.visualizer.metrics().last_error
        if error is not None:
            raise RuntimeError(error)

    def close(self) -> None:
        self.visualizer.close()
        self.transport.close()
        self.video.close()
        self.tactile.close()
        self.camera.close()


def _recording_sample(
    peripherals: _MockPeripherals,
    cycle: TeleopCycle,
) -> RecordingSample:
    frame = peripherals.latest_frame
    tactile = peripherals.latest_tactile
    action = cycle.safe_action
    if frame is None or tactile is None or action is None:
        raise RuntimeError("sensor extension must run before recording")
    tactile_data = struct.pack(
        "<12f",
        *(value for taxel in tactile.values for value in taxel),
    )
    return RecordingSample(
        state=(*cycle.robot_state.joints_rad, 0.03),
        action=(*action.values, 0.03),
        timestamps_ns=(
            cycle.timestamp_ns,
            cycle.robot_state.source_timestamp_ns,
            action.source_timestamp_ns,
            cycle.vr_input.source_timestamp_ns,
            tactile.source_timestamp_ns,
            frame.source_timestamp_ns,
        ),
        tactile=FrozenArray(
            data=tactile_data,
            shape=(4, 3),
            dtype="float32",
        ),
        images=(
            NamedArray(
                name="camera_0",
                value=FrozenArray(
                    data=frame.data,
                    shape=frame.shape,
                    dtype="uint8",
                ),
            ),
        ),
    )


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class CompleteMockSessionTest(unittest.TestCase):
    def test_robot_vr_camera_video_tactile_recording_and_visualizer(self) -> None:
        robot = MockRobotBackend()
        executor = LatestActionExecutor(robot, target_hz=500)
        peripherals = _MockPeripherals()
        writer = _MemoryWriter()
        recording = RecordingCycleExtension(
            schema=build_recording_schema(
                data_type="qpos",
                robot_dof=6,
                camera_count=1,
                resolution=(2, 1),
                tactile_shape=(4, 3),
            ),
            task="mock integration",
            sample_factory=lambda cycle: _recording_sample(
                peripherals,
                cycle,
            ),
            writer=writer,
        )
        teleop = TeleopSession(
            vr_source=MockVRInputSource(
                VRConfig(),
                script=tuple(_vr_sample(index) for index in range(3)),
            ),
            state_source=robot,
            mapping=_JointMapping(),
            action_filter=_AllowAll(),
            executor=executor,
            target_hz=200,
            extensions=(peripherals, recording),
        )
        session = DataCollectionSession(teleop, recording)

        session.start()
        try:
            session.recording.start_recording()
            session.run(max_cycles=3)
            session.recording.stop_recording()
            self.assertTrue(
                _wait_until(
                    lambda: session.recording.status().state.value == "idle"
                )
            )
            self.assertTrue(
                _wait_until(
                    lambda: bool(peripherals.renderer.snapshots)
                    and peripherals.renderer.snapshots[-1].sequence == 2
                )
            )
        finally:
            session.close()

        self.assertEqual(session.teleop.metrics().cycles, 3)
        self.assertGreaterEqual(len(robot.captured_actions), 1)
        self.assertEqual(
            tuple(frame.sequence for frame in peripherals.transport.frames),
            (0, 1, 2),
        )
        self.assertTrue(peripherals.encoder.closed)
        self.assertTrue(peripherals.transport.closed)
        self.assertEqual(len(writer.episodes), 1)
        self.assertEqual(len(writer.episodes[0].samples), 3)
        self.assertEqual(
            writer.episodes[0].samples[-1].images[0].value.shape,
            (1, 2, 3),
        )
        self.assertEqual(
            writer.episodes[0].samples[-1].tactile.shape,
            (4, 3),
        )
        self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
