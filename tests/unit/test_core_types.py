"""Tests for immutable core samples, commands, events, and clocks."""

from __future__ import annotations

import dataclasses
import pickle
import unittest

from airo_doffy.core import (
    CameraFrame,
    Clock,
    ControllerButton,
    ControllerState,
    EncodedFrame,
    HandSide,
    HandState,
    ModelValidationError,
    MonotonicClock,
    Observation,
    PixelFormat,
    ProcessedFrame,
    RobotAction,
    RobotCommandType,
    RobotState,
    RuntimeCommand,
    RuntimeCommandType,
    RuntimeEvent,
    RuntimeEventSeverity,
    RuntimeEventType,
    TactileSample,
    VideoCodec,
    VRInputMode,
    VRInputState,
    WallClock,
    WrenchSample,
)


def controller(side: HandSide, sequence: int = 1) -> ControllerState:
    return ControllerState(
        side=side,
        position_m=(0, 0, 0),
        orientation_xyzw=(0, 0, 0, 1),
        joystick_xy=(0, 0),
        index_trigger=0,
        grip_trigger=0,
        buttons=frozenset({ControllerButton.JOYSTICK}),
        sequence=sequence,
        source_timestamp_ns=10,
    )


class ImageModelTest(unittest.TestCase):
    def test_camera_frame_copies_mutable_payload_and_is_frozen(self) -> None:
        payload = bytearray(range(12))
        frame = CameraFrame(
            stream_id="camera_0",
            data=payload,
            shape=(2, 2, 3),
            pixel_format="rgb8",
            sequence=1,
            source_timestamp_ns=100,
        )
        payload[0] = 255
        self.assertEqual(frame.data[0], 0)
        self.assertEqual(frame.pixel_format, PixelFormat.RGB8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            frame.sequence = 2  # type: ignore[misc]

    def test_camera_frame_rejects_shape_and_payload_mismatch(self) -> None:
        with self.assertRaises(ModelValidationError):
            CameraFrame(
                stream_id="camera_0",
                data=b"short",
                shape=(2, 2, 3),
                pixel_format=PixelFormat.RGB8,
                sequence=1,
                source_timestamp_ns=100,
            )

    def test_processed_and_encoded_frame_metadata(self) -> None:
        processed = ProcessedFrame(
            stream_id="camera_0",
            data=b"\x00" * 4,
            shape=(2, 2),
            pixel_format=PixelFormat.GRAY8,
            sequence=4,
            source_timestamp_ns=100,
            processing_timestamp_ns=110,
        )
        encoded = EncodedFrame(
            stream_id=processed.stream_id,
            data=b"\x00\x00\x01",
            codec=VideoCodec.H264,
            width=2,
            height=2,
            sequence=processed.sequence,
            source_timestamp_ns=processed.source_timestamp_ns,
            encoded_timestamp_ns=120,
            keyframe=True,
        )
        self.assertTrue(encoded.keyframe)
        self.assertEqual(encoded.sequence, 4)


class SensorAndInputModelTest(unittest.TestCase):
    def test_tactile_and_wrench_shapes(self) -> None:
        tactile = TactileSample(
            values=[[0, 0, 0] for _ in range(4)],
            sequence=1,
            source_timestamp_ns=10,
        )
        wrench = WrenchSample(
            values=[1, 2, 3, 0.1, 0.2, 0.3],
            sequence=1,
            source_timestamp_ns=10,
        )
        self.assertEqual(len(tactile.values), 4)
        self.assertEqual(wrench.values[3:], (0.1, 0.2, 0.3))
        with self.assertRaises(ModelValidationError):
            TactileSample(
                values=[[0, 0, 0] for _ in range(41)],
                sequence=1,
                source_timestamp_ns=10,
            )
        with self.assertRaises(ModelValidationError):
            WrenchSample(
                values=[0, 0, float("nan"), 0, 0, 0],
                sequence=1,
                source_timestamp_ns=10,
            )

    def test_controller_and_vr_controller_mode(self) -> None:
        left = controller(HandSide.LEFT)
        right = controller(HandSide.RIGHT)
        state = VRInputState(
            mode=VRInputMode.CONTROLLERS,
            controllers=(left, right),
            sequence=1,
            source_timestamp_ns=10,
        )
        self.assertEqual(state.controllers, (left, right))
        with self.assertRaises(ModelValidationError):
            VRInputState(
                mode=VRInputMode.CONTROLLERS,
                controllers=(left, left),
                sequence=1,
                source_timestamp_ns=10,
            )

    def test_hand_shape_and_wrist_pair(self) -> None:
        hand = HandState(
            side=HandSide.LEFT,
            joints_m=[(0, 0, 0)] * 26,
            sequence=1,
            source_timestamp_ns=10,
        )
        state = VRInputState(
            mode=VRInputMode.HANDS,
            hands=(hand,),
            sequence=1,
            source_timestamp_ns=10,
        )
        self.assertEqual(len(state.hands[0].joints_m), 26)
        with self.assertRaises(ModelValidationError):
            HandState(
                side=HandSide.LEFT,
                joints_m=[(0, 0, 0)] * 25,
                wrist_position_m=(0, 0, 0),
                sequence=1,
                source_timestamp_ns=10,
            )


class RobotAndObservationModelTest(unittest.TestCase):
    def robot_state(self, sequence: int = 1) -> RobotState:
        return RobotState(
            joints_rad=[0] * 6,
            tcp_pose=[
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            sequence=sequence,
            source_timestamp_ns=10,
        )

    def test_robot_shapes_and_action_modes(self) -> None:
        state = self.robot_state()
        action = RobotAction(
            command_type=RobotCommandType.JOINT_POSITION,
            values=[0] * 6,
            duration_s=0.01,
            sequence=1,
            source_timestamp_ns=10,
        )
        stop = RobotAction(
            command_type=RobotCommandType.STOP,
            values=(),
            sequence=2,
            source_timestamp_ns=11,
        )
        self.assertEqual(len(state.tcp_pose), 4)
        self.assertEqual(len(action.values), 6)
        self.assertEqual(stop.values, ())
        with self.assertRaises(ModelValidationError):
            RobotAction(
                command_type=RobotCommandType.STOP,
                values=(1,),
                sequence=3,
                source_timestamp_ns=12,
            )

    def test_observation_rejects_duplicate_streams(self) -> None:
        frame = ProcessedFrame(
            stream_id="camera_0",
            data=b"\x00",
            shape=(1, 1),
            pixel_format=PixelFormat.GRAY8,
            sequence=1,
            source_timestamp_ns=10,
            processing_timestamp_ns=11,
        )
        with self.assertRaises(ModelValidationError):
            Observation(
                frames=(frame, frame),
                sequence=1,
                source_timestamp_ns=12,
            )


class RuntimeMessageAndClockTest(unittest.TestCase):
    def test_typed_command_value_rules(self) -> None:
        command = RuntimeCommand(
            kind=RuntimeCommandType.SET_VIDEO_PROFILE,
            value="low-latency",
            origin="unity",
            sequence=1,
            source_timestamp_ns=10,
        )
        self.assertEqual(command.kind, RuntimeCommandType.SET_VIDEO_PROFILE)
        self.assertEqual(pickle.loads(pickle.dumps(command)), command)
        for kind, value in (
            (RuntimeCommandType.SET_TELEOP_MODE, "hand"),
            (RuntimeCommandType.SET_CAMERA_ZOOM, "1.5"),
            (RuntimeCommandType.SET_CAMERA_RESOLUTION, "1280x720"),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(
                    RuntimeCommand(
                        kind=kind,
                        value=value,
                        sequence=2,
                        source_timestamp_ns=11,
                    ).value,
                    value,
                )
        with self.assertRaises(ModelValidationError):
            RuntimeCommand(
                kind=RuntimeCommandType.START_RECORDING,
                value="unexpected",
                sequence=2,
                source_timestamp_ns=11,
            )

    def test_runtime_event_details_are_immutable(self) -> None:
        details = [["reason", "stale-input"]]
        event = RuntimeEvent(
            kind=RuntimeEventType.WATCHDOG_TRIPPED,
            severity=RuntimeEventSeverity.ERROR,
            component="watchdog",
            details=details,
            sequence=1,
            timestamp_ns=10,
        )
        details[0][1] = "changed"
        self.assertEqual(event.details, (("reason", "stale-input"),))

    def test_standard_clocks_satisfy_protocol(self) -> None:
        monotonic = MonotonicClock()
        wall = WallClock()
        self.assertIsInstance(monotonic, Clock)
        self.assertGreater(monotonic.now_ns(), 0)
        self.assertGreater(wall.now_ns(), 0)


if __name__ == "__main__":
    unittest.main()
