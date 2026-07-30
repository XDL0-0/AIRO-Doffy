"""Episode state machine and immutable buffer tests."""

from __future__ import annotations

import unittest

from airo_doffy.core.errors import LifecycleError, ModelValidationError
from airo_doffy.recording import (
    EpisodeState,
    EpisodeStateMachine,
    FrozenArray,
    NamedArray,
    RecordingSample,
    SampleBuffer,
    build_recording_schema,
)


def _array(shape: tuple[int, ...], dtype: str, fill: int = 0) -> FrozenArray:
    sizes = {"float32": 4, "uint8": 1}
    length = sizes[dtype]
    for dimension in shape:
        length *= dimension
    return FrozenArray(data=bytes([fill]) * length, shape=shape, dtype=dtype)


class SampleBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_recording_schema(
            data_type="both",
            robot_dof=6,
            camera_count=1,
            resolution=(2, 1),
            tactile_shape=(4, 3),
            force_enabled=True,
            torque_enabled=True,
            depth_enabled=True,
        )

    def sample(self) -> RecordingSample:
        return RecordingSample(
            state=(0.0,) * 7,
            action=(1.0,) * 7,
            timestamps_ns=(1,) * 6,
            tcp_pose=(0.0,) * 7,
            force=(0.0,) * 3,
            torque=(0.0,) * 3,
            tactile=_array((4, 3), "float32"),
            images=(
                NamedArray(
                    name="camera_0",
                    value=_array((1, 2, 3), "uint8"),
                ),
            ),
            depths=(
                NamedArray(
                    name="camera_0",
                    value=_array((1, 2), "float32"),
                ),
            ),
        )

    def test_buffer_validates_and_seals_an_immutable_episode(self) -> None:
        buffer = SampleBuffer(self.schema)
        sample = self.sample()
        buffer.append(sample)
        episode = buffer.seal(index=3, task="pick")
        self.assertEqual(episode.index, 3)
        self.assertEqual(episode.samples, (sample,))
        self.assertTrue(buffer.is_sealed)
        with self.assertRaises(LifecycleError):
            buffer.append(sample)

    def test_buffer_rejects_missing_modalities_and_wrong_order(self) -> None:
        buffer = SampleBuffer(self.schema)
        sample = self.sample()
        with self.assertRaises(ModelValidationError):
            buffer.append(
                RecordingSample(
                    state=sample.state,
                    action=sample.action,
                    timestamps_ns=sample.timestamps_ns,
                    tcp_pose=sample.tcp_pose,
                    force=sample.force,
                    torque=sample.torque,
                    tactile=sample.tactile,
                    images=(),
                    depths=sample.depths,
                )
            )
        with self.assertRaises(ModelValidationError):
            FrozenArray(data=b"\x00", shape=(2,), dtype="float32")

    def test_capacity_is_explicit_and_bounded(self) -> None:
        buffer = SampleBuffer(self.schema, capacity=1)
        buffer.append(self.sample())
        with self.assertRaises(BufferError):
            buffer.append(self.sample())


class EpisodeStateMachineTest(unittest.TestCase):
    def test_successful_export_advances_index_only_after_completion(self) -> None:
        lifecycle = EpisodeStateMachine(next_episode_index=4)
        self.assertEqual(lifecycle.start_episode(), 4)
        lifecycle.note_sample()
        self.assertEqual(lifecycle.request_finish(), 4)
        self.assertEqual(lifecycle.snapshot().state, EpisodeState.EXPORT_PENDING)
        self.assertEqual(lifecycle.snapshot().next_episode_index, 4)
        lifecycle.export_succeeded(4)
        self.assertEqual(lifecycle.snapshot().next_episode_index, 5)
        self.assertEqual(lifecycle.snapshot().state, EpisodeState.IDLE)

    def test_failed_export_requires_retry_or_explicit_discard(self) -> None:
        lifecycle = EpisodeStateMachine()
        lifecycle.start_episode()
        lifecycle.note_sample()
        lifecycle.request_finish()
        lifecycle.export_failed(0, "disk full")
        self.assertEqual(lifecycle.snapshot().state, EpisodeState.EXPORT_FAILED)
        self.assertEqual(lifecycle.retry_export(), 0)
        lifecycle.export_failed(0, "still full")
        self.assertEqual(lifecycle.discard_failed_export(), 0)
        self.assertEqual(lifecycle.snapshot().next_episode_index, 0)

    def test_rollback_discards_active_before_touching_storage(self) -> None:
        lifecycle = EpisodeStateMachine(next_episode_index=2)
        lifecycle.start_episode()
        lifecycle.note_sample()
        request = lifecycle.request_rollback()
        self.assertTrue(request.discard_active)
        self.assertIsNone(request.episode_index)
        self.assertEqual(lifecycle.snapshot().next_episode_index, 2)

        request = lifecycle.request_rollback()
        self.assertFalse(request.discard_active)
        self.assertEqual(request.episode_index, 1)
        lifecycle.rollback_succeeded(1)
        self.assertEqual(lifecycle.snapshot().next_episode_index, 1)

    def test_invalid_transitions_and_empty_finish_are_rejected(self) -> None:
        lifecycle = EpisodeStateMachine()
        with self.assertRaises(LifecycleError):
            lifecycle.note_sample()
        lifecycle.start_episode()
        with self.assertRaises(LifecycleError):
            lifecycle.request_finish()


if __name__ == "__main__":
    unittest.main()
