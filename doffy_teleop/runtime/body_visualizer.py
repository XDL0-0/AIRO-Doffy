"""Show Quest human-body poses without connecting to a robot by default.

Use ``--teleop realman`` or ``--teleop classic`` to also run teleoperation.
Robot imports stay in a child process. Demo and preview modes never start it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TELEOP_SCRIPTS = {"realman": "realman_teleop.py", "classic": "main.py"}
INTERRUPT_TIMEOUT = 10.0
TERMINATE_TIMEOUT = 3.0
KILL_TIMEOUT = 3.0


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def _udp_port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be a UDP port between 1 and 65535")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    robot_mode = parser.add_mutually_exclusive_group()
    robot_mode.add_argument(
        "--teleop", choices=TELEOP_SCRIPTS, default=None,
        help="opt in to robot teleoperation (default: visualization only)",
    )
    robot_mode.add_argument(
        "--visualization-only", action="store_true",
        help="receive body poses without robot teleoperation (the default)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="show an animated example body without launching a robot",
    )
    parser.add_argument(
        "--bind-ip", default="0.0.0.0",
        help="local address for the body UDP receiver (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--body-port", type=_udp_port, default=8015,
        help="dedicated body UDP port (default: 8015)",
    )
    parser.add_argument(
        "--hz", type=_positive_float, default=30.0,
        help="visualization refresh rate (default: 30)",
    )
    parser.add_argument(
        "--stale-after", type=_positive_float, default=0.5,
        help="seconds before a body pose is shown as stale (default: 0.5)",
    )
    parser.add_argument(
        "--save-preview", metavar="PATH",
        help="save a preview image without launching a robot",
    )
    return parser


def _exit_status(returncode: int) -> int:
    """Convert subprocess signal exits to conventional shell exit statuses."""
    return returncode if returncode >= 0 else 128 - returncode


def _shutdown_teleop(process: subprocess.Popen) -> None:
    """Give the runtime time to close cameras, recordings, and robot control."""
    if process.poll() is not None:
        return

    actions = (
        (lambda: process.send_signal(signal.SIGINT), INTERRUPT_TIMEOUT),
        (process.terminate, TERMINATE_TIMEOUT),
        (process.kill, KILL_TIMEOUT),
    )
    for action, timeout in actions:
        try:
            action()
        except ProcessLookupError:
            # It exited between poll() and sending the signal; still reap it.
            pass
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("teleoperation did not stop after SIGKILL")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    launch_robot = args.teleop is not None and not (
        args.demo or args.save_preview is not None
    )
    process: subprocess.Popen | None = None
    teleop_returncode: int | None = None
    status = 0

    def on_ready() -> None:
        nonlocal process
        if launch_robot:
            entry_point = REPOSITORY_ROOT / TELEOP_SCRIPTS[args.teleop]
            # A separate session leaves Ctrl-C handling and orderly cleanup to
            # this launcher instead of interrupting both runtimes at once.
            process = subprocess.Popen(
                [sys.executable, str(entry_point)],
                cwd=str(REPOSITORY_ROOT),
                start_new_session=True,
            )

    def should_stop() -> bool:
        nonlocal teleop_returncode
        if process is None:
            return False
        returncode = process.poll()
        if returncode is None:
            return False
        if teleop_returncode is None:
            teleop_returncode = returncode
            print(
                f"Teleoperation exited with status {_exit_status(returncode)}; "
                "closing the body visualizer.",
                file=sys.stderr,
            )
        return True

    try:
        # This import follows argparse so --help works without GUI packages or
        # any robot, camera, or recording dependencies installed.
        from doffy_teleop.runtime.body_viewer import run_visualizer

        run_visualizer(
            bind_ip=args.bind_ip,
            port=args.body_port,
            hz=args.hz,
            stale_after=args.stale_after,
            demo=args.demo,
            save_preview=args.save_preview,
            on_ready=on_ready,
            should_stop=should_stop,
        )
        # Catch an exit coinciding with window closure before initiating our
        # own shutdown. Shutdown-triggered child statuses are not failures.
        should_stop()
    except KeyboardInterrupt:
        status = 130
    except Exception as exc:
        print(f"Unable to run teleoperation/body visualizer: {exc}", file=sys.stderr)
        status = 1
    finally:
        if process is not None:
            try:
                _shutdown_teleop(process)
            except KeyboardInterrupt:
                # A second Ctrl-C requests immediate cleanup escalation.
                status = 130
                try:
                    process.kill()
                    process.wait(timeout=KILL_TIMEOUT)
                except Exception as exc:
                    print(f"Unable to stop teleoperation: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"Unable to stop teleoperation: {exc}", file=sys.stderr)
                if status == 0:
                    status = 1

    if status == 0 and teleop_returncode is not None:
        status = _exit_status(teleop_returncode)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
