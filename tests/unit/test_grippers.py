"""Hardware-free gripper contract and Robotiq socket tests."""

from __future__ import annotations

import unittest

from airo_doffy.core import LifecycleError, ModelValidationError
from airo_doffy.robots.grippers import Gripper, NullGripper, Robotiq2F85Gripper


class _FakeSocket:
    def __init__(self, responses=(), *, fail_send: bool = False) -> None:
        self.responses = list(responses)
        self.fail_send = fail_send
        self.sent: list[bytes] = []
        self.close_count = 0

    def sendall(self, data: bytes) -> None:
        if self.fail_send:
            self.fail_send = False
            raise OSError("disconnected")
        self.sent.append(data)

    def recv(self, _size: int) -> bytes:
        return self.responses.pop(0)

    def close(self) -> None:
        self.close_count += 1


class GripperTest(unittest.TestCase):
    def test_null_gripper_clamps_width_and_has_explicit_lifecycle(self) -> None:
        gripper = NullGripper(max_width_m=0.1)
        self.assertIsInstance(gripper, Gripper)
        with self.assertRaises(LifecycleError):
            gripper.read_width()
        gripper.start()
        gripper.move(0.2)
        self.assertEqual(gripper.read_width(), 0.1)
        gripper.move(-1)
        self.assertEqual(gripper.read_width(), 0.0)
        gripper.close()
        gripper.close()

    def test_robotiq_mapping_and_commands(self) -> None:
        connection = _FakeSocket(
            [
                b"ack\n",
                b"SPE 255\n",
                b"ack\n",
                b"ack\n",
                b"ack\n",
                b"POS 115\n",
            ]
        )
        calls: list[tuple[str, int, float]] = []

        def factory(host: str, port: int, timeout: float):
            calls.append((host, port, timeout))
            return connection

        gripper = Robotiq2F85Gripper(
            "192.0.2.2",
            max_width_m=0.1,
            socket_factory=factory,
        )
        self.assertIsInstance(gripper, Gripper)
        gripper.start()
        gripper.open()
        gripper.move(0.0)
        gripper.move(0.05)
        self.assertAlmostEqual(gripper.read_width(), 0.05)
        self.assertEqual(
            connection.sent,
            [
                b"SET SPE 255\n",
                b"GET SPE\n",
                b"SET POS 0\n",
                b"SET POS 230\n",
                b"SET POS 115\n",
                b"GET POS\n",
            ],
        )
        self.assertEqual(calls, [("192.0.2.2", 63352, 0.02)])
        gripper.close()
        gripper.close()
        self.assertEqual(connection.close_count, 1)

    def test_socket_failure_reconnects_once(self) -> None:
        first = _FakeSocket(fail_send=True)
        second = _FakeSocket([b"ack\n", b"SPE 255\n"])
        connections = iter((first, second))
        gripper = Robotiq2F85Gripper(
            "192.0.2.2",
            socket_factory=lambda host, port, timeout: next(connections),
        )
        gripper.start()
        self.assertEqual(first.close_count, 1)
        self.assertEqual(second.sent, [b"SET SPE 255\n", b"GET SPE\n"])
        gripper.close()

    def test_validation_and_bad_response(self) -> None:
        with self.assertRaises(ModelValidationError):
            Robotiq2F85Gripper("")
        with self.assertRaises(ModelValidationError):
            NullGripper(max_width_m=0)
        connection = _FakeSocket([b"ack\n", b"SPE 255\n", b"invalid\n"])
        gripper = Robotiq2F85Gripper(
            "192.0.2.2",
            socket_factory=lambda host, port, timeout: connection,
        )
        gripper.start()
        with self.assertRaisesRegex(RuntimeError, "invalid Robotiq"):
            gripper.read_width()
        gripper.close()


if __name__ == "__main__":
    unittest.main()
