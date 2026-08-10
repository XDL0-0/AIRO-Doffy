import math
import time

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_thread_mode_e,
)


ROBOT_IP = "192.168.1.18"
ROBOT_PORT = 8080

CONTROL_DT = 0.005       # 5 ms, 200 Hz
DURATION = 12.0         # Total motion duration
RAMP_TIME = 2.0         # Startup and shutdown ramp duration
PERIOD = 4.0            # Duration of one round trip
AMPLITUDE_DEG = 45.0     # Oscillation amplitude: +/-45 degrees
JOINT_INDEX = 3         # Python index 3 is the fourth RM75 joint


def smoothstep(x: float) -> float:
    """Return a smooth 0-to-1 transition with zero endpoint velocity."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def envelope(t: float) -> float:
    """Ramp the amplitude up at the start and down at the end."""
    ramp_up = smoothstep(t / RAMP_TIME)
    ramp_down = smoothstep((DURATION - t) / RAMP_TIME)
    return min(ramp_up, ramp_down)


def send_for_duration(
    arm: RoboticArm,
    target: list[float],
    duration: float,
) -> None:
    """Resend one target continuously to prevent pass-through interruption."""
    end_time = time.perf_counter() + duration
    next_time = time.perf_counter()

    while time.perf_counter() < end_time:
        ret = arm.rm_movej_canfd(
            target,
            True,   # High-follow mode
            0,      # No expansion axis
            0,      # Full pass-through
            0,
        )

        if ret != 0:
            raise RuntimeError(f"rm_movej_canfd failed: {ret}")

        next_time += CONTROL_DT
        sleep_time = next_time - time.perf_counter()

        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_time = time.perf_counter()


def main() -> None:
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(ROBOT_IP, ROBOT_PORT)

    if handle.id < 0:
        raise ConnectionError(
            f"Failed to connect to the robot arm; handle.id={handle.id}"
        )

    print(f"Connected to the robot arm; ID={handle.id}")

    last_target = None

    try:
        # Read the current robot-arm state.
        ret, state = arm.rm_get_current_arm_state()

        if ret != 0:
            raise RuntimeError(
                f"Failed to read the robot-arm state; error code: {ret}"
            )

        start_joint = list(state["joint"])

        if len(start_joint) != 7:
            raise RuntimeError(
                "The RM75 must return seven joint angles; "
                f"received: {start_joint}"
            )

        print("Current joint angles:")
        print([round(q, 3) for q in start_joint])

        print(
            f"Joint {JOINT_INDEX + 1} will oscillate by "
            f"±{AMPLITUDE_DEG}°"
        )
        print("Press Ctrl+C to stop")

        # Hold the current pose briefly before moving.
        send_for_duration(arm, start_joint, 0.5)

        start_time = time.perf_counter()
        next_time = start_time

        while True:
            now = time.perf_counter()
            t = now - start_time

            if t >= DURATION:
                break

            # The amplitude envelope ensures a smooth start and stop.
            scale = envelope(t)

            # Apply sinusoidal back-and-forth motion.
            offset = (
                AMPLITUDE_DEG
                * scale
                * math.sin(2.0 * math.pi * t / PERIOD)
            )

            target = start_joint.copy()
            target[JOINT_INDEX] += offset
            target[JOINT_INDEX+1] += offset
            last_target = target

            ret = arm.rm_movej_canfd(
                target,
                True,   # High-follow mode
                0,
                0,      # Full pass-through mode
                0,
            )

            if ret != 0:
                raise RuntimeError(
                    f"rm_movej_canfd failed: {ret}"
                )

            next_time += CONTROL_DT
            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(
                    "Warning: control cycle exceeded its deadline by "
                    f"{-sleep_time * 1000:.2f} ms"
                )
                next_time = time.perf_counter()

        # The envelope reaches zero, so the target is near the start position.
        send_for_duration(arm, start_joint, 0.5)
        print("Motion complete")

    except KeyboardInterrupt:
        print("\nCtrl+C received; holding the current position")

        if last_target is not None:
            # Hold the final target instead of returning abruptly to the start.
            send_for_duration(arm, last_target, 0.3)

    finally:
        arm.rm_delete_robot_arm()
        print("Disconnected from the robot arm")


if __name__ == "__main__":
    main()
