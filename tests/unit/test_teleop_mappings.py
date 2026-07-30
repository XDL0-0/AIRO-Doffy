"""Tests for pure controller, hand, gripper, and command-mode mappings."""

from __future__ import annotations

import unittest

from airo_doffy.config import TeleopConfig
from airo_doffy.core import (
    ClockDomain,
    ControllerState,
    HandSide,
    HandState,
    ModelValidationError,
    RobotCommandType,
    RobotState,
    VRInputMode,
    VRInputState,
)
from airo_doffy.teleop import (
    CommandModeSelector,
    ControllerPoseMapping,
    HandPoseMapping,
    ModeAwareTeleopMapping,
    TeleopMapping,
)

IDENTITY_POSE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def robot(*, gripper: float | None = 0.04, dof: int = 6) -> RobotState:
    return RobotState(
        sequence=1,
        source_timestamp_ns=90,
        receive_timestamp_ns=95,
        clock_domain=ClockDomain.MONOTONIC,
        joints_rad=(0.0,) * dof,
        tcp_pose=IDENTITY_POSE,
        gripper_width_m=gripper,
    )


def controller_vr(
    *,
    sequence: int,
    position=(0.0, 0.0, 0.0),
    grip: float = 1.0,
    joystick_y: float = 0.0,
) -> VRInputState:
    controllers = tuple(
        ControllerState(
            sequence=sequence,
            source_timestamp_ns=100 + sequence,
            receive_timestamp_ns=200 + sequence,
            clock_domain=ClockDomain.DEVICE,
            side=side,
            position_m=position if side is HandSide.RIGHT else (0, 0, 0),
            orientation_xyzw=(0, 0, 0, 1),
            joystick_xy=(0, joystick_y if side is HandSide.RIGHT else 0),
            index_trigger=0,
            grip_trigger=grip if side is HandSide.RIGHT else 0,
            buttons=frozenset(),
        )
        for side in (HandSide.LEFT, HandSide.RIGHT)
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=100 + sequence,
        receive_timestamp_ns=200 + sequence,
        clock_domain=ClockDomain.DEVICE,
        mode=VRInputMode.CONTROLLERS,
        controllers=controllers,
    )


def hand_vr(
    *,
    sequence: int,
    wrist=(0.0, 0.0, 0.0),
    finger_distance: float = 0.04,
    with_wrist: bool = True,
) -> VRInputState:
    joints = [(0.0, 0.0, 0.0) for _index in range(26)]
    joints[0] = wrist
    joints[5] = (0.0, 0.0, 0.0)
    joints[10] = (finger_distance, 0.0, 0.0)
    hand = HandState(
        sequence=sequence,
        source_timestamp_ns=100 + sequence,
        receive_timestamp_ns=200 + sequence,
        clock_domain=ClockDomain.DEVICE,
        side=HandSide.RIGHT,
        joints_m=tuple(joints),
        wrist_position_m=wrist if with_wrist else None,
        wrist_orientation_xyzw=(0, 0, 0, 1) if with_wrist else None,
    )
    return VRInputState(
        sequence=sequence,
        source_timestamp_ns=100 + sequence,
        receive_timestamp_ns=200 + sequence,
        clock_domain=ClockDomain.DEVICE,
        mode=VRInputMode.HANDS,
        hands=(hand,),
    )


class _Ik:
    def __init__(self, solution=(0.1,) * 6) -> None:
        self.solution = solution
        self.calls = []

    def solve(self, tcp_pose, seed_joints_rad):
        self.calls.append((tcp_pose, seed_joints_rad))
        return self.solution


class TeleopMappingTest(unittest.TestCase):
    def test_tcp_selector_and_controller_reference_engagement(self) -> None:
        config = TeleopConfig(command_mode="tcp", freeze_rotation=True)
        mapping = ControllerPoseMapping(config, CommandModeSelector("tcp"))
        self.assertIsInstance(mapping, TeleopMapping)
        first = mapping.map_input(controller_vr(sequence=1), robot(), 0.1)
        second = mapping.map_input(
            controller_vr(sequence=2, position=(0.1, 0, 0)),
            robot(),
            0.1,
        )
        self.assertEqual(first.command_type, RobotCommandType.HOLD)
        self.assertEqual(second.command_type, RobotCommandType.TCP_POSE)
        self.assertAlmostEqual(second.values[3], -0.1)
        self.assertEqual(second.receive_timestamp_ns, 202)
        self.assertEqual(mapping.metrics.mapped, 1)

    def test_grip_release_rebases_and_joystick_maps_gripper_only(self) -> None:
        config = TeleopConfig(command_mode="tcp")
        mapping = ControllerPoseMapping(config, CommandModeSelector("tcp"))
        mapping.map_input(controller_vr(sequence=1), robot(), 0.1)
        released = mapping.map_input(
            controller_vr(
                sequence=2,
                position=(0.2, 0, 0),
                grip=0,
                joystick_y=-1,
            ),
            robot(),
            0.1,
        )
        resumed = mapping.map_input(
            controller_vr(sequence=3, position=(0.3, 0, 0)),
            robot(),
            0.1,
        )
        self.assertEqual(released.command_type, RobotCommandType.HOLD)
        self.assertAlmostEqual(released.gripper_width_m, 0.05)
        self.assertAlmostEqual(resumed.values[3], -0.1)

    def test_gripper_target_integrates_while_measurement_lags(self) -> None:
        mapping = ControllerPoseMapping(
            TeleopConfig(command_mode="tcp"),
            CommandModeSelector("tcp"),
        )
        first = mapping.map_input(
            controller_vr(sequence=1, grip=0, joystick_y=-1),
            robot(),
            0.1,
        )
        second = mapping.map_input(
            controller_vr(sequence=2, grip=0, joystick_y=-1),
            robot(),
            0.1,
        )
        held = mapping.map_input(
            controller_vr(sequence=3, grip=0, joystick_y=0),
            robot(),
            0.1,
        )
        self.assertAlmostEqual(first.gripper_width_m, 0.05)
        self.assertAlmostEqual(second.gripper_width_m, 0.06)
        self.assertAlmostEqual(held.gripper_width_m, 0.04)

    def test_fine_mode_edge_rebases_then_scales_translation(self) -> None:
        mapping = ControllerPoseMapping(
            TeleopConfig(command_mode="tcp"),
            CommandModeSelector("tcp"),
        )
        mapping.map_input(controller_vr(sequence=1), robot(), 0.1)
        mapping.set_fine_mode(True)
        rebased = mapping.map_input(
            controller_vr(sequence=2, position=(0.1, 0, 0)),
            robot(),
            0.1,
        )
        moved = mapping.map_input(
            controller_vr(sequence=3, position=(0.2, 0, 0)),
            robot(),
            0.1,
        )
        self.assertEqual(rebased.command_type, RobotCommandType.HOLD)
        self.assertAlmostEqual(moved.values[3], -0.03)

    def test_joint_mode_uses_injected_ik_and_holds_on_rejection(self) -> None:
        solver = _Ik()
        selector = CommandModeSelector("joint", ik_solver=solver)
        mapping = ControllerPoseMapping(
            TeleopConfig(command_mode="joint"),
            selector,
        )
        mapping.map_input(controller_vr(sequence=1), robot(), 0.1)
        action = mapping.map_input(
            controller_vr(sequence=2, position=(0.1, 0, 0)),
            robot(),
            0.1,
        )
        self.assertEqual(action.command_type, RobotCommandType.JOINT_POSITION)
        self.assertEqual(action.values, (0.1,) * 6)
        self.assertEqual(len(solver.calls), 1)
        solver.solution = None
        rejected = mapping.map_input(
            controller_vr(sequence=3, position=(0.2, 0, 0)),
            robot(),
            0.1,
        )
        self.assertEqual(rejected.command_type, RobotCommandType.HOLD)
        self.assertEqual(selector.metrics.ik_rejected, 1)

    def test_hand_reference_motion_gripper_thresholds_and_jump_rejection(self) -> None:
        mapping = HandPoseMapping(
            TeleopConfig(command_mode="tcp"),
            CommandModeSelector("tcp"),
        )
        first = mapping.map_input(hand_vr(sequence=1), robot(), 0.1)
        opened = mapping.map_input(
            hand_vr(sequence=2, wrist=(0.01, 0, 0), finger_distance=0.07),
            robot(),
            0.1,
        )
        jumped = mapping.map_input(
            hand_vr(sequence=3, wrist=(0.3, 0, 0)),
            robot(),
            0.1,
        )
        recovered = mapping.map_input(
            hand_vr(sequence=4, wrist=(0.31, 0, 0), with_wrist=False),
            robot(),
            0.1,
        )
        self.assertEqual(first.command_type, RobotCommandType.HOLD)
        self.assertEqual(opened.command_type, RobotCommandType.TCP_POSE)
        self.assertAlmostEqual(opened.gripper_width_m, 0.05)
        self.assertEqual(jumped.command_type, RobotCommandType.HOLD)
        self.assertEqual(recovered.command_type, RobotCommandType.HOLD)
        self.assertEqual(mapping.metrics.jump_rejections, 1)

    def test_mode_aware_mapping_selects_typed_input_mode(self) -> None:
        config = TeleopConfig(command_mode="tcp")
        mapping = ModeAwareTeleopMapping(
            ControllerPoseMapping(config, CommandModeSelector("tcp")),
            HandPoseMapping(config, CommandModeSelector("tcp")),
        )
        self.assertEqual(
            mapping.map_input(controller_vr(sequence=1), robot(), 0.1).command_type,
            RobotCommandType.HOLD,
        )
        self.assertEqual(
            mapping.map_input(hand_vr(sequence=2), robot(), 0.1).command_type,
            RobotCommandType.HOLD,
        )

    def test_invalid_dt_and_joint_mode_without_ik_are_rejected(self) -> None:
        with self.assertRaises(ModelValidationError):
            CommandModeSelector("joint")
        mapping = ControllerPoseMapping(
            TeleopConfig(command_mode="tcp"),
            CommandModeSelector("tcp"),
        )
        with self.assertRaises(ModelValidationError):
            mapping.map_input(controller_vr(sequence=1), robot(), 0)


if __name__ == "__main__":
    unittest.main()
