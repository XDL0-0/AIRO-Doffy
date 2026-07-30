"""Data collection composition tests without a duplicated teleop loop."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

from airo_doffy.config import VRConfig
from airo_doffy.core import RobotCommandType
from airo_doffy.devices.vr import MockVRInputSource
from airo_doffy.recording import RecordingSample, build_recording_schema
from airo_doffy.runtime import (
    DataCollectionSession,
    RecordingCycleExtension,
    TeleopSession,
)
from tests.unit.test_mock_vr import controller_state
from tests.unit.test_runtime_session import (
    FakeClock,
    FakeExecutor,
    FakeFilter,
    FakeMapping,
    FakeStateSource,
)


class FakeWriter:
    def __init__(self) -> None:
        self.episodes = []
        self.closed = False

    def write(self, episode):
        self.episodes.append(episode)
        return Path(f"episode_{episode.index}.data")

    def close(self) -> None:
        self.closed = True


class FakeRollback:
    def __init__(self) -> None:
        self.indices = []

    def rollback(self, episode_index: int) -> bool:
        self.indices.append(episode_index)
        return True


def sample_from_cycle(cycle) -> RecordingSample:
    action = cycle.safe_action
    values = (
        action.values
        if action is not None and action.command_type is RobotCommandType.JOINT_POSITION
        else (0.0,) * 6
    )
    return RecordingSample(
        state=(*cycle.robot_state.joints_rad, 0.0),
        action=(*values, 0.0),
        timestamps_ns=(
            cycle.timestamp_ns,
            cycle.robot_state.source_timestamp_ns,
            0 if action is None else action.source_timestamp_ns,
            0 if cycle.vr_input is None else cycle.vr_input.source_timestamp_ns,
            0,
        ),
    )


def wait_for_state(recording: RecordingCycleExtension, value: str) -> None:
    deadline = time.monotonic() + 1.0
    while recording.status().state.value != value and time.monotonic() < deadline:
        time.sleep(0.005)
    if recording.status().state.value != value:
        raise AssertionError(
            f"recording state is {recording.status().state.value}, expected {value}"
        )


class DataCollectionSessionTest(unittest.TestCase):
    def create_session(self):
        clock = FakeClock()
        calls: list[str] = []
        writer = FakeWriter()
        rollback = FakeRollback()
        recording = RecordingCycleExtension(
            schema=build_recording_schema(
                data_type="qpos",
                robot_dof=6,
                camera_count=0,
                resolution=(2, 1),
            ),
            task="pick",
            sample_factory=sample_from_cycle,
            writer=writer,
            rollback=rollback,
        )
        teleop = TeleopSession(
            vr_source=MockVRInputSource(
                VRConfig(),
                script=(
                    controller_state(0),
                    controller_state(1),
                    controller_state(2),
                ),
            ),
            state_source=FakeStateSource(clock),
            mapping=FakeMapping(),
            action_filter=FakeFilter(),
            executor=FakeExecutor(calls),
            target_hz=1000,
            extensions=(recording,),
            clock=clock,
        )
        return DataCollectionSession(teleop, recording), writer, rollback

    def test_recording_observes_the_existing_teleop_loop(self) -> None:
        session, writer, _rollback = self.create_session()
        session.start()
        session.recording.start_recording()
        session.run(max_cycles=2)
        session.recording.stop_recording()
        wait_for_state(session.recording, "idle")
        self.assertEqual(len(writer.episodes), 1)
        self.assertEqual(len(writer.episodes[0].samples), 2)
        self.assertEqual(session.teleop.metrics().cycles, 2)
        self.assertEqual(session.recording.status().next_episode_index, 1)
        session.close()
        self.assertTrue(writer.closed)

    def test_rollback_discards_active_episode_before_storage(self) -> None:
        session, writer, rollback = self.create_session()
        session.start()
        session.recording.start_recording()
        session.run(max_cycles=1)
        self.assertEqual(
            session.recording.rollback_last_episode(),
            "discarded active episode",
        )
        self.assertEqual(writer.episodes, [])
        self.assertEqual(rollback.indices, [])
        session.close()

    def test_completed_episode_rollback_reuses_index(self) -> None:
        session, _writer, rollback = self.create_session()
        session.start()
        session.recording.start_recording()
        session.run(max_cycles=1)
        session.recording.stop_recording()
        wait_for_state(session.recording, "idle")
        session.recording.rollback_last_episode()
        wait_for_state(session.recording, "idle")
        self.assertEqual(rollback.indices, [0])
        self.assertEqual(session.recording.status().next_episode_index, 0)
        session.close()


if __name__ == "__main__":
    unittest.main()
