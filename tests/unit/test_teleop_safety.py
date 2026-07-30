"""Tests for composable teleoperation action safety filters."""

from __future__ import annotations

import math
import unittest

from airo_doffy.core import (
    ClockDomain,
    RobotAction,
    RobotCommandType,
    RobotState,
)
from airo_doffy.teleop import (
    ActionFreshnessFilter,
    ActionRateLimitFilter,
    CartesianVelocityLimitFilter,
    InverseKinematicsFilter,
    JointAccelerationLimitFilter,
    JointLimitsFilter,
    JointVelocityLimitFilter,
    SafetyFilterChain,
    WorkspaceBoundsFilter,
)
from airo_doffy.teleop.transforms import flatten_transform, transform

IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def state(*, joints=(0.0,) * 6, position=(0.0, 0.0, 0.0)) -> RobotState:
    return RobotState(
        sequence=1,
        source_timestamp_ns=1,
        receive_timestamp_ns=2,
        clock_domain=ClockDomain.MONOTONIC,
        joints_rad=joints,
        tcp_pose=transform(IDENTITY, position),
    )


def action(
    command_type=RobotCommandType.JOINT_POSITION,
    values=(0.0,) * 6,
    *,
    duration_s=0.1,
    source_timestamp_ns=1_000_000_000,
    receive_timestamp_ns=None,
    clock_domain=ClockDomain.MONOTONIC,
) -> RobotAction:
    return RobotAction(
        sequence=1,
        source_timestamp_ns=source_timestamp_ns,
        receive_timestamp_ns=receive_timestamp_ns,
        clock_domain=clock_domain,
        command_type=command_type,
        values=values,
        duration_s=duration_s,
    )


class TeleopSafetyFilterTest(unittest.TestCase):
    def test_workspace_rejects_pose_and_predicted_twist_outside_box(self) -> None:
        bounds = WorkspaceBoundsFilter((-1, -1, 0), (1, 1, 1))
        inside = action(
            RobotCommandType.TCP_POSE,
            flatten_transform(transform(IDENTITY, (0.5, 0, 0.5))),
        )
        outside = action(
            RobotCommandType.TCP_POSE,
            flatten_transform(transform(IDENTITY, (1.1, 0, 0.5))),
        )
        twist = action(
            RobotCommandType.TCP_TWIST,
            (0, 0, -1, 0, 0, 0),
            duration_s=0.1,
        )
        robot_state = state(position=(0, 0, 0.05))
        self.assertIs(bounds.apply(inside, robot_state, 0), inside)
        self.assertIsNone(bounds.apply(outside, robot_state, 0))
        self.assertIsNone(bounds.apply(twist, robot_state, 0))

    def test_joint_limits_reject_position_and_predicted_velocity(self) -> None:
        limits = JointLimitsFilter((-1,) * 6, (1,) * 6)
        self.assertIsNotNone(limits.apply(action(values=(0.5,) * 6), state(), 0))
        self.assertIsNone(limits.apply(action(values=(1.1,) * 6), state(), 0))
        velocity = action(
            RobotCommandType.JOINT_VELOCITY,
            (2.0,) * 6,
            duration_s=1,
        )
        self.assertIsNone(limits.apply(velocity, state(), 0))
        mismatched = action(values=(0.0,) * 6)
        self.assertIsNone(
            JointLimitsFilter((-1,) * 7, (1,) * 7).apply(
                mismatched,
                state(joints=(0.0,) * 7),
                0,
            )
        )

    def test_joint_velocity_limits_position_step_and_explicit_velocity(self) -> None:
        limiter = JointVelocityLimitFilter((1.0,) * 6, default_dt_s=0.1)
        position = limiter.apply(action(values=(1.0,) * 6), state(), 0)
        velocity = limiter.apply(
            action(RobotCommandType.JOINT_VELOCITY, (2.0,) * 6),
            state(),
            0,
        )
        self.assertEqual(position.values, (0.1,) * 6)
        self.assertEqual(velocity.values, (1.0,) * 6)

    def test_cartesian_velocity_limits_pose_and_twist_norms(self) -> None:
        limiter = CartesianVelocityLimitFilter(
            max_linear_m_s=1.0,
            max_angular_rad_s=1.0,
            default_dt_s=0.1,
        )
        rotation = (
            (math.cos(1), -math.sin(1), 0.0),
            (math.sin(1), math.cos(1), 0.0),
            (0.0, 0.0, 1.0),
        )
        pose = action(
            RobotCommandType.TCP_POSE,
            flatten_transform(transform(rotation, (1, 0, 0))),
        )
        limited = limiter.apply(pose, state(), 0)
        self.assertAlmostEqual(limited.values[3], 0.1)
        angle = math.atan2(limited.values[4], limited.values[0])
        self.assertAlmostEqual(angle, 0.1)
        twist = limiter.apply(
            action(RobotCommandType.TCP_TWIST, (3, 4, 0, 0, 0, 2)),
            state(),
            0,
        )
        self.assertAlmostEqual(math.hypot(*twist.values[:2]), 1.0)
        self.assertAlmostEqual(twist.values[5], 1.0)

    def test_joint_acceleration_limits_consecutive_velocity_changes(self) -> None:
        limiter = JointAccelerationLimitFilter(
            (2.0,) * 6,
            default_dt_s=0.1,
        )
        desired = action(RobotCommandType.JOINT_VELOCITY, (1.0,) * 6)
        first = limiter.apply(desired, state(), 100_000_000)
        second = limiter.apply(desired, state(), 200_000_000)
        self.assertEqual(first.values, (0.2,) * 6)
        self.assertEqual(second.values, (0.4,) * 6)
        limiter.reset()
        position = limiter.apply(action(values=(1.0,) * 6), state(), 300_000_000)
        self.assertAlmostEqual(position.values[0], 0.02)
        self.assertIsNone(limiter.apply(desired, state(), 200_000_000))

    def test_freshness_uses_receive_timestamp_or_monotonic_source(self) -> None:
        freshness = ActionFreshnessFilter(0.25, future_tolerance_s=0.01)
        received = action(
            source_timestamp_ns=1,
            receive_timestamp_ns=900_000_000,
            clock_domain=ClockDomain.DEVICE,
        )
        self.assertIsNotNone(freshness.apply(received, state(), 1_000_000_000))
        self.assertIsNone(freshness.apply(received, state(), 1_200_000_000))
        incomparable = action(
            source_timestamp_ns=900_000_000,
            clock_domain=ClockDomain.DEVICE,
        )
        self.assertIsNone(freshness.apply(incomparable, state(), 1_000_000_000))
        future = action(source_timestamp_ns=1_020_000_000)
        self.assertIsNone(freshness.apply(future, state(), 1_000_000_000))

    def test_rate_limit_is_deterministic_and_never_blocks_hold(self) -> None:
        limiter = ActionRateLimitFilter(10)
        active = action()
        self.assertIs(limiter.apply(active, state(), 0), active)
        self.assertIsNone(limiter.apply(active, state(), 50_000_000))
        self.assertIs(limiter.apply(active, state(), 100_000_000), active)
        hold = action(RobotCommandType.HOLD, (), duration_s=None)
        self.assertIs(limiter.apply(hold, state(), 100_000_001), hold)

    def test_ik_filter_converts_or_rejects_without_backend_access(self) -> None:
        class Solver:
            def __init__(self) -> None:
                self.solution = (0.1,) * 6

            def solve(self, _pose, _seed):
                return self.solution

        solver = Solver()
        ik_filter = InverseKinematicsFilter(solver)
        tcp = action(
            RobotCommandType.TCP_POSE,
            flatten_transform(transform(IDENTITY, (0.1, 0, 0))),
        )
        selected = ik_filter.apply(tcp, state(), 0)
        self.assertEqual(selected.command_type, RobotCommandType.JOINT_POSITION)
        solver.solution = None
        self.assertIsNone(ik_filter.apply(tcp, state(), 0))
        self.assertEqual(ik_filter.rejection_count, 1)

    def test_filter_chain_orders_transformations_and_reports_rejection(self) -> None:
        chain = SafetyFilterChain(
            (
                JointVelocityLimitFilter((1.0,) * 6),
                JointLimitsFilter((-0.05,) * 6, (0.05,) * 6),
            )
        )
        rejected = chain.apply(action(values=(1.0,) * 6), state(), 0)
        self.assertIsNone(rejected)
        self.assertEqual(chain.metrics.processed, 1)
        self.assertEqual(chain.metrics.rejected, 1)
        self.assertEqual(chain.metrics.last_rejected_by, "JointLimitsFilter")


if __name__ == "__main__":
    unittest.main()
