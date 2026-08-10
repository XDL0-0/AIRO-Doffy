#!/usr/bin/env python3
"""
Keyboard teleoperation for a BrainCo Revo2 hand connected through a
RealMan RM75 controller using the RM_ARM+ protocol.

Key mapping:
    q / a   Thumb flexion      increase / decrease
    w / s   Index finger       increase / decrease
    e / d   Middle finger      increase / decrease
    r / f   Ring finger        increase / decrease
    t / g   Little finger      increase / decrease
    y / h   Thumb rotation     increase / decrease
2
    1       Open
    2       Grab: close four fingers -> rotate thumb -> close thumb
    4       Point
    5       Pinch: rotate thumb -> position index -> close thumb
    6       Tripod pinch

    [ / ]   Decrease / increase teleoperation step
    i       Read and print the current Revo2 state
    v       Synchronize the command target to the measured position
    SPACE   Resend/hold the current target
    ?       Print help
    ESC     Quit

Safety:
- Start with the hand clear of people and objects.
- The command target is initialized from the measured hand position.
- Every preset stage sends one target only; motion smoothing is left to the
  Revo2/RealMan controller.
- Preset values are conservative examples. Adjust the pose fractions below
  after checking the actual hand geometry and the object being grasped.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


DEFAULT_ROBOT_IP = "192.168.1.18"
DEFAULT_ROBOT_PORT = 8080
DEFAULT_BAUDRATE = 460800

# RealMan RM_ARM+ DoF order:
# [thumb flexion, index, middle, ring, little finger, thumb rotation]
THUMB_FLEX = 0
INDEX = 1
MIDDLE = 2
RING = 3
PINKY = 4
THUMB_ROTATE = 5

DOF_NAMES = (
    "Thumb flex",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
    "Thumb rotate",
)

DOF_KEYS: dict[str, tuple[int, int]] = {
    "q": (THUMB_FLEX, +1),
    "a": (THUMB_FLEX, -1),
    "w": (INDEX, +1),
    "s": (INDEX, -1),
    "e": (MIDDLE, +1),
    "d": (MIDDLE, -1),
    "r": (RING, +1),
    "f": (RING, -1),
    "t": (PINKY, +1),
    "g": (PINKY, -1),
    "y": (THUMB_ROTATE, +1),
    "h": (THUMB_ROTATE, -1),
}

# Avoid issuing two API commands in the same 20 ms control interval.
# Presets are not interpolated in Python: every stage sends exactly one target
# and lets the Revo2/RealMan controller generate the smooth motion.
MIN_SEND_INTERVAL_S = 1.0 / 50.0

# ---------------------------------------------------------------------------
# Preset tuning parameters
# ---------------------------------------------------------------------------
# Each value is a fraction of the position range reported by RM_ARM+.
# Start conservatively. Use the individual teleoperation keys to find better
# values for your specific Revo2 hand, then update these constants.

GRAB_FOUR_FINGER_CLOSE = 0.90
GRAB_THUMB_ROTATE = 0.8
GRAB_THUMB_CLOSE = 0.88

PINCH_THUMB_ROTATE = 0.65
PINCH_INDEX_CLOSE = 0.72
PINCH_THUMB_CLOSE = 0.68

TRIPOD_THUMB_ROTATE = 0.65
TRIPOD_THUMB_CLOSE = 0.70
TRIPOD_INDEX_CLOSE = 0.68
TRIPOD_MIDDLE_CLOSE = 0.68

HELP_TEXT = """
====================== Revo2 keyboard teleoperation ======================

Finger control; upper key increases position, lower key decreases it:

    q / a   Thumb flexion
    w / s   Index finger
    e / d   Middle finger
    r / f   Ring finger
    t / g   Little finger
    y / h   Thumb rotation/opposition

Presets:
    1       Open
    2       Grab: four fingers -> thumb rotate -> thumb close
    4       Point
    5       Pinch: thumb rotate -> index -> thumb close
    6       Tripod pinch

Other:
    [ / ]   Decrease / increase step size
    i       Print measured hand state
    v       Synchronize target with measured position
    SPACE   Resend the current target
    ?       Show this help
    ESC     Quit

Each preset stage sends one target; there is no Python-side interpolation.
The program starts from the measured Revo2 position.
=========================================================================
""".strip()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def check_result(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed, error code: {code}")


def get_rm_plus_state_with_retry(
    arm: RoboticArm,
    retries: int = 10,
    delay: float = 0.25,
) -> dict[str, Any] | None:
    """Read RM_ARM+ state, retrying while the end-effector initializes."""
    last_result: int | None = None
    last_state: dict[str, Any] | None = None

    for attempt in range(1, retries + 1):
        result, state_info = arm.rm_get_rm_plus_state_info()
        last_result = result
        last_state = state_info

        if result == 0:
            return state_info

        print(
            f"[WARN] rm_get_rm_plus_state_info failed: "
            f"result={result}, attempt={attempt}/{retries}"
        )
        time.sleep(delay)

    print(
        "[WARN] RM_ARM+ basic communication works, but real-time state "
        "is currently unavailable."
    )
    print(f"[WARN] Last result: {last_result}")
    print(f"[WARN] Last state: {last_state}")
    return None


def extract_limits(base_info: dict[str, Any]) -> tuple[list[int], list[int]]:
    """Read six-DoF position limits from RM_ARM+ base information."""
    low = base_info.get("pos_low")
    high = base_info.get("pos_up")

    if (
        not isinstance(low, list)
        or not isinstance(high, list)
        or len(low) < 6
        or len(high) < 6
    ):
        print("[WARN] Invalid position limits in base info; using 0..1000.")
        return [0] * 6, [1000] * 6

    lower = [int(value) for value in low[:6]]
    upper = [int(value) for value in high[:6]]

    for index in range(6):
        if lower[index] > upper[index]:
            lower[index], upper[index] = upper[index], lower[index]

    return lower, upper


def measured_position(
    arm: RoboticArm,
    lower: Sequence[int],
    upper: Sequence[int],
) -> list[int] | None:
    state = get_rm_plus_state_with_retry(arm, retries=6, delay=0.2)
    if state is None:
        return None

    pos = state.get("pos")
    if not isinstance(pos, list) or len(pos) < 6:
        print(f"[WARN] State contains no valid six-DoF position: {pos!r}")
        return None

    return [
        clamp(int(pos[index]), int(lower[index]), int(upper[index]))
        for index in range(6)
    ]


def print_hand_state(arm: RoboticArm) -> None:
    state = get_rm_plus_state_with_retry(arm, retries=5, delay=0.2)
    if state is None:
        return

    print("\nMeasured Revo2 state:")
    print(f"  sys_state:       {state.get('sys_state')}")
    print(f"  dof_state:       {state.get('dof_state')}")
    print(f"  dof_err:         {state.get('dof_err')}")
    print(f"  pos:             {state.get('pos')}")
    print(f"  angle:           {state.get('angle')}")
    print(f"  speed:           {state.get('speed')}")
    print(f"  current:         {state.get('current')}")
    print(f"  force:           {state.get('force')}")
    print()


@contextlib.contextmanager
def cbreak_terminal() -> Iterator[None]:
    """Temporarily enable immediate single-key input on a POSIX terminal."""
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard teleoperation requires an interactive terminal.")

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


@dataclass
class Revo2Teleop:
    arm: RoboticArm
    lower: list[int]
    upper: list[int]
    target: list[int]
    step: int = 50
    last_send_time: float = 0.0

    def _bounded(self, values: Sequence[int]) -> list[int]:
        if len(values) != 6:
            raise ValueError(f"Expected 6 positions, received {len(values)}")

        return [
            clamp(int(values[index]), self.lower[index], self.upper[index])
            for index in range(6)
        ]

    def frac(self, index: int, fraction: float) -> int:
        """Convert a normalized preset value into the reported motor range."""
        fraction = max(0.0, min(1.0, float(fraction)))
        low = self.lower[index]
        high = self.upper[index]
        return int(round(low + fraction * (high - low)))

    def pose_from_fractions(self, fractions: Sequence[float]) -> list[int]:
        if len(fractions) != 6:
            raise ValueError(f"Expected 6 fractions, received {len(fractions)}")
        return [self.frac(index, fraction) for index, fraction in enumerate(fractions)]

    def send(
        self,
        values: Sequence[int],
        label: str = "command",
        *,
        show_status: bool = True,
    ) -> bool:
        """
        Send a position target through RM_ARM+.

        RealMan documents block=True for rm_set_hand_follow_pos() as returning
        after the command has been sent. Preset stages therefore use one target
        command followed by an explicit wait before the next stage.
        """
        candidate = self._bounded(values)

        elapsed = time.monotonic() - self.last_send_time
        if elapsed < MIN_SEND_INTERVAL_S:
            time.sleep(MIN_SEND_INTERVAL_S - elapsed)

        result = self.arm.rm_set_hand_follow_pos(candidate, True)
        self.last_send_time = time.monotonic()

        if result != 0:
            print(
                f"\n[ERROR] rm_set_hand_follow_pos failed for {label}: "
                f"error code {result}"
            )
            return False

        self.target = candidate
        if show_status:
            self.print_status(label)
        return True

    def run_stages(
        self,
        stages: Sequence[tuple[str, Sequence[int], float]],
        action_name: str,
    ) -> bool:
        """Send one target per stage and wait before sending the next one."""
        print(f"\nRunning {action_name}...")

        for stage_name, target, wait_after_s in stages:
            label = f"{action_name}: {stage_name}"
            if not self.send(target, label):
                print(f"\n[ERROR] {action_name} stopped during: {stage_name}")
                return False

            # The API returns after accepting the target, not after reaching it.
            # This wait preserves the stage order without Python-side streaming.
            if wait_after_s > 0.0:
                time.sleep(wait_after_s)

        measured = measured_position(self.arm, self.lower, self.upper)
        if measured is not None:
            self.target = measured

        self.print_status(f"{action_name} complete")
        return True

    def adjust_dof(self, dof_index: int, direction: int) -> None:
        candidate = self.target.copy()
        candidate[dof_index] += direction * self.step
        self.send(candidate, DOF_NAMES[dof_index])

    def set_step(self, new_step: int) -> None:
        self.step = clamp(new_step, 1, 500)
        self.print_status("step changed")

    def sync_from_state(self) -> None:
        position = measured_position(self.arm, self.lower, self.upper)
        if position is None:
            print("\n[WARN] Could not synchronize from measured state.")
            return

        self.target = position
        self.print_status("target synced")

    def print_status(self, label: str = "") -> None:
        text = (
            f"target={self.target}  step={self.step}"
            + (f"  [{label}]" if label else "")
        )
        sys.stdout.write("\r\033[2K" + text)
        sys.stdout.flush()

    def open_hand(self) -> bool:
        open_pose = self.pose_from_fractions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return self.send(open_pose, "open")

    def grab(self) -> bool:
        """
        Grab sequence:
        1. Rotate/oppose the thumb while all fingers remain open.
        2. Close index, middle, ring, and little finger while maintaining
        the thumb rotation.
        3. Flex/close the thumb to complete the grasp.
        """

        # Stage 1: rotate the thumb while all fingers remain open.
        stage_thumb_rotate = self.pose_from_fractions(
            [
                0.00,               # Thumb flex
                0.00,               # Index
                0.00,               # Middle
                0.00,               # Ring
                0.00,               # Pinky
                GRAB_THUMB_ROTATE,  # Thumb rotation
            ]
        )

        # Stage 2: close the four fingers while keeping the thumb rotated.
        stage_four_fingers = stage_thumb_rotate.copy()
        stage_four_fingers[THUMB_FLEX] = self.frac(
            THUMB_FLEX, GRAB_THUMB_CLOSE-0.6
        )
        stage_four_fingers[INDEX] = self.frac(
            INDEX, GRAB_FOUR_FINGER_CLOSE
        )
        stage_four_fingers[MIDDLE] = self.frac(
            MIDDLE, GRAB_FOUR_FINGER_CLOSE
        )
        stage_four_fingers[RING] = self.frac(
            RING, GRAB_FOUR_FINGER_CLOSE
        )
        stage_four_fingers[PINKY] = self.frac(
            PINKY, GRAB_FOUR_FINGER_CLOSE
        )

        # Stage 3: close the thumb while preserving all previous positions.
        stage_thumb_close = stage_four_fingers.copy()
        stage_thumb_close[THUMB_FLEX] = self.frac(
            THUMB_FLEX, GRAB_THUMB_CLOSE
        )

        return self.run_stages(
            [
                ("rotate thumb", stage_thumb_rotate, 0.80),
                ("close four fingers", stage_four_fingers, 0.80),
                ("close thumb", stage_thumb_close, 0.90),
            ],
            "grab",
        )

    def point(self) -> bool:
        point_pose = self.pose_from_fractions(
            [0.75, 0.00, 0.90, 0.90, 0.90, 0.55]
        )
        return self.send(point_pose, "point")

    def notcool(self) -> bool:
        stage_four_fingers = self.pose_from_fractions(
            [
                0.00,               # Thumb flex
                GRAB_FOUR_FINGER_CLOSE,               # Index
                0.00,               # Middle
                GRAB_FOUR_FINGER_CLOSE,               # Ring
                GRAB_FOUR_FINGER_CLOSE,               # Pinky
                GRAB_FOUR_FINGER_CLOSE,  # Thumb rotation
            ]
        )

        # stage_four_fingers[THUMB_FLEX] = self.frac(
        #     THUMB_FLEX, GRAB_FOUR_FINGER_CLOSE
        # )
        # stage_four_fingers[INDEX] = self.frac(
        #     INDEX, GRAB_FOUR_FINGER_CLOSE
        # )
        # stage_four_fingers[RING] = self.frac(
        #     RING, GRAB_FOUR_FINGER_CLOSE
        # )
        # stage_four_fingers[PINKY] = self.frac(
        #     PINKY, GRAB_FOUR_FINGER_CLOSE
        # )
        stage_thumb_rotate = stage_four_fingers.copy()
        stage_four_fingers[THUMB_FLEX] = self.frac(
            THUMB_FLEX, GRAB_FOUR_FINGER_CLOSE
        )
        return self.run_stages(
            [
                ("rotate thumb", stage_thumb_rotate, 0.50),
                ("close four fingers", stage_four_fingers, 0.50),
            ],
            "grab",
        )


    def pinch(self) -> bool:
        """
        Pinch without sending thumb and index toward one another at once.

        The old preset commanded thumb flexion, index flexion, and thumb
        rotation simultaneously. Depending on the Revo2 calibration, their
        paths can intersect before either motor reaches its final command. This
        version first establishes thumb opposition, then positions the index,
        and finally closes the thumb onto the stationary index/object.
        """
        stage_thumb_rotate = self.pose_from_fractions(
            [0.00, 0.00, 0.00, 0.00, 0.00, PINCH_THUMB_ROTATE]
        )
        stage_index = stage_thumb_rotate.copy()
        stage_index[INDEX] = self.frac(INDEX, PINCH_INDEX_CLOSE)
        stage_thumb_close = stage_index.copy()
        stage_thumb_close[THUMB_FLEX] = self.frac(
            THUMB_FLEX, PINCH_THUMB_CLOSE
        )

        return self.run_stages(
            [
                ("rotate thumb with fingers open", stage_thumb_rotate, 0.80),
                ("position index", stage_index, 0.85),
                ("close thumb slowly", stage_thumb_close, 1.00),
            ],
            "pinch",
        )

    def tripod_pinch(self) -> bool:
        tripod_pose = self.pose_from_fractions(
            [
                TRIPOD_THUMB_CLOSE,
                TRIPOD_INDEX_CLOSE,
                TRIPOD_MIDDLE_CLOSE,
                0.00,
                0.00,
                TRIPOD_THUMB_ROTATE,
            ]
        )
        return self.send(tripod_pose, "tripod pinch")

    def run(self) -> None:
        print(HELP_TEXT)
        print()
        self.print_status("ready")

        with cbreak_terminal():
            while True:
                key = sys.stdin.read(1)

                if key == "\x1b":  # ESC
                    print("\nExiting keyboard teleoperation.")
                    return

                if key in DOF_KEYS:
                    dof_index, direction = DOF_KEYS[key]
                    self.adjust_dof(dof_index, direction)
                    continue

                if key == "1":
                    self.open_hand()
                    continue

                if key == "2":
                    self.grab()
                    continue

                if key == "3":
                    self.notcool()
                    continue

                if key == "4":
                    self.point()
                    continue

                if key == "5":
                    self.pinch()
                    continue

                if key == "6":
                    self.tripod_pinch()
                    continue

                if key == "[":
                    self.set_step(max(1, self.step // 2))
                    continue

                if key == "]":
                    self.set_step(min(500, self.step * 2))
                    continue

                if key == "i":
                    print_hand_state(self.arm)
                    self.print_status("state read")
                    continue

                if key == "v":
                    self.sync_from_state()
                    continue

                if key == " ":
                    self.send(self.target, "hold")
                    continue

                if key == "?":
                    print("\n")
                    print(HELP_TEXT)
                    print()
                    self.print_status("ready")
                    continue

                # Ignore Enter, the removed key 3, and unknown keys.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for a Revo2 hand through RM_ARM+."
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_ROBOT_IP,
        help=f"RM controller IP address (default: {DEFAULT_ROBOT_IP})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_ROBOT_PORT,
        help=f"RM controller API port (default: {DEFAULT_ROBOT_PORT})",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        choices=(9600, 115200, 256000, 460800),
        help=f"RM_ARM+ baud rate (default: {DEFAULT_BAUDRATE})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Initial position increment for each key press (default: 50)",
    )
    parser.add_argument(
        "--no-change-mode",
        action="store_true",
        help="Fail instead of changing RM_ARM+ mode when the baud rate differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if os.name != "posix":
        raise RuntimeError("This version uses termios and requires Linux or macOS.")

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

    try:
        handle = arm.rm_create_robot_arm(args.ip, args.port)
        if handle.id < 0:
            raise RuntimeError(
                f"Could not connect to RM controller at "
                f"{args.ip}:{args.port}; handle={handle.id}"
            )

        print(f"Connected to RM75, handle={handle.id}")

        result, current_mode = arm.rm_get_rm_plus_mode()
        check_result(result, "rm_get_rm_plus_mode")

        if current_mode != args.baudrate:
            if args.no_change_mode:
                raise RuntimeError(
                    f"RM_ARM+ is configured for {current_mode} bps, "
                    f"but {args.baudrate} bps was requested."
                )

            print(
                f"Changing RM_ARM+ mode from {current_mode} "
                f"to {args.baudrate} bps..."
            )
            result = arm.rm_set_rm_plus_mode(args.baudrate)
            check_result(result, "rm_set_rm_plus_mode")
            time.sleep(2.0)

        print(f"RM_ARM+ mode: {args.baudrate} bps")

        result, base_info = arm.rm_get_rm_plus_base_info()
        check_result(result, "rm_get_rm_plus_base_info")

        print("Base information:")
        print(base_info)

        if int(base_info.get("type", -1)) != 2:
            raise RuntimeError(
                "The RM_ARM+ end-effector is not reported as a dexterous hand "
                f"(type={base_info.get('type')!r})."
            )

        if int(base_info.get("dof", -1)) != 6:
            raise RuntimeError(
                "This teleoperation mapping expects a six-DoF hand "
                f"(reported dof={base_info.get('dof')!r})."
            )

        lower, upper = extract_limits(base_info)
        print(f"Position lower limits: {lower}")
        print(f"Position upper limits: {upper}")

        # Initialize from the measured position to avoid a startup jump.
        time.sleep(0.5)
        initial_target = measured_position(arm, lower, upper)
        if initial_target is None:
            raise RuntimeError(
                "Could not read the initial Revo2 position. "
                "Refusing to start teleoperation with an unknown target."
            )

        print(f"Initial measured position: {initial_target}")

        teleop = Revo2Teleop(
            arm=arm,
            lower=lower,
            upper=upper,
            target=initial_target,
            step=clamp(args.step, 1, 500),
        )
        teleop.run()
        return 0

    finally:
        arm.rm_delete_robot_arm()
        print("Disconnected from RM75.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nUser interrupted.")
        raise SystemExit(130)
    except Exception as exception:
        print(f"\nERROR: {exception}", file=sys.stderr)
        raise SystemExit(1)
