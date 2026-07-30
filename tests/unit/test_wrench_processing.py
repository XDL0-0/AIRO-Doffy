"""Hardware-free wrench source, gravity compensation, and pipeline tests."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from airo_doffy.core import ClockDomain, ModelValidationError, RobotState, WrenchSample
from airo_doffy.devices.wrench import (
    GravityCompensator,
    RobotStateWrenchSource,
    WrenchFilter,
    WrenchProcessor,
    WrenchSource,
)
from airo_doffy.robots import MockRobotBackend

IDENTITY_3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def add(left, right):
    return tuple(a + b for a, b in zip(left, right, strict=True))


class WrenchProcessingTest(unittest.TestCase):
    def test_robot_state_source_preserves_raw_metadata(self) -> None:
        state = RobotState(
            sequence=7,
            source_timestamp_ns=50,
            receive_timestamp_ns=60,
            clock_domain=ClockDomain.MONOTONIC,
            joints_rad=(0.0,) * 6,
            tcp_pose=IDENTITY_4,
            wrench=(1, 2, 3, 4, 5, 6),
        )
        backend = MockRobotBackend(initial_state=state)
        backend.start()
        source = RobotStateWrenchSource(backend, frame_id="robot_base")
        self.assertIsInstance(source, WrenchSource)
        sample = source.read_latest()
        self.assertEqual(sample.values, state.wrench)
        self.assertEqual(sample.sequence, 7)
        self.assertEqual(sample.source_timestamp_ns, 50)
        self.assertEqual(sample.receive_timestamp_ns, 60)
        self.assertEqual(sample.frame_id, "robot_base")
        backend.close()

    def test_gravity_bias_calibration_and_explicit_reset(self) -> None:
        compensator = GravityCompensator(
            2.0,
            (0.1, 0.0, 0.0),
            filter_alpha=1.0,
        )
        gravity = compensator.gravity_wrench(IDENTITY_3)
        self.assertEqual(gravity[:3], (0.0, 0.0, -19.62))
        self.assertAlmostEqual(gravity[4], 1.962)
        bias = (1.0, -2.0, 0.5, 0.1, 0.2, -0.3)
        unloaded = add(gravity, bias)
        compensator.add_calibration_sample(unloaded, IDENTITY_3)
        compensator.add_calibration_sample(unloaded, IDENTITY_3)
        for actual, expected in zip(
            compensator.finish_calibration(),
            bias,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        contact = (2.0, 4.0, 6.0, 0.2, 0.4, 0.6)
        for actual, expected in zip(
            compensator.compensate(add(unloaded, contact), IDENTITY_3),
            contact,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

        compensator.reset_baseline((0.0,) * 6)
        self.assertTrue(compensator.calibrated)
        for actual, expected in zip(
            compensator.compensate(add(gravity, contact), IDENTITY_3),
            contact,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        compensator.reset_baseline()
        self.assertFalse(compensator.calibrated)

    def test_pipeline_preserves_metadata_and_requires_rotation(self) -> None:
        compensator = GravityCompensator(1.0, (0.0, 0.0, 0.0), filter_alpha=1.0)
        processor = WrenchProcessor(
            WrenchFilter(force_deadband=1.0, torque_deadband=0.1),
            gravity_compensator=compensator,
        )
        raw = WrenchSample(
            sequence=4,
            source_timestamp_ns=123,
            receive_timestamp_ns=130,
            values=(2.0, 0.0, -9.81, 0.2, 0.0, 0.0),
            frame_id="robot_base",
        )
        with self.assertRaises(ModelValidationError):
            processor.process(raw)
        processed = processor.process(raw, rotation_tool_to_base=IDENTITY_3)
        self.assertEqual(processed.values, (1.0, 0.0, 0.0, 0.1, 0.0, 0.0))
        self.assertEqual(processed.sequence, raw.sequence)
        self.assertEqual(processed.source_timestamp_ns, raw.source_timestamp_ns)
        self.assertEqual(processed.receive_timestamp_ns, raw.receive_timestamp_ns)
        self.assertEqual(processed.frame_id, raw.frame_id)

    def test_compensation_rejects_invalid_shapes_and_nonfinite_values(self) -> None:
        compensator = GravityCompensator(1.0, (0.0, 0.0, 0.0))
        with self.assertRaises(ModelValidationError):
            compensator.gravity_wrench(((1, 0),) * 3)
        with self.assertRaises(ModelValidationError):
            compensator.compensate((1, 2, 3, 4, 5, float("nan")), IDENTITY_3)

    def test_utils_adapter_preserves_numpy_contract(self) -> None:
        code = """
import sys
import types
import numpy as np
scipy = types.ModuleType("scipy")
scipy.__path__ = []
spatial = types.ModuleType("scipy.spatial")
spatial.__path__ = []
transform = types.ModuleType("scipy.spatial.transform")
transform.Rotation = object
distance = types.ModuleType("scipy.spatial.distance")
distance.cdist = lambda *_args, **_kwargs: None
sys.modules.update({
    "scipy": scipy,
    "scipy.spatial": spatial,
    "scipy.spatial.transform": transform,
    "scipy.spatial.distance": distance,
})
import utils
compensator = utils.GravityCompensator(1.0, np.zeros(3), filter_alpha=1.0)
gravity = compensator._gravity_wrench(np.eye(3))
bias = np.array([1, 2, 3, 0.1, 0.2, 0.3], dtype=float)
compensator.add_calibration_sample(gravity + bias, np.eye(3))
np.testing.assert_allclose(compensator.finish_calibration(), bias)
np.testing.assert_allclose(
    compensator.compensate(gravity + bias + 1.0, np.eye(3)),
    np.ones(6),
)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
