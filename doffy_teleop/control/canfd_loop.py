"""Dedicated RealMan high-follow CAN-FD command loop."""

from __future__ import annotations

from typing import Any, Callable
import threading
import time

import numpy as np

import doffy_teleop.utils as utils
from .contracts import CanfdLoopSnapshot
from .canfd_setpoints import CanfdSetpointMixin


class CanfdCommandLoop(CanfdSetpointMixin):
    """Own every high-follow SDK call on one measured command clock."""

    def __init__(
        self,
        arm: Any,
        *,
        control_mode: str,
        dof: int,
        target_hz: float,
        minimum_hz: float,
        rate_check_window: float,
        maximum_failure_windows: int,
        trajectory_mode: int = 1,
        radio: int = 60,
        joint_speed_limits: float | np.ndarray | None = None,
        joint_acceleration_limits: float | np.ndarray | None = None,
        linear_speed_limit: float | None = None,
        linear_acceleration_limit: float | None = None,
        angular_speed_limit: float | None = None,
        angular_acceleration_limit: float | None = None,
        heartbeat_timeout: float = 0.05,
    ) -> None:
        if control_mode not in {"joint", "tcp"}:
            raise ValueError(f"Unsupported CAN-FD control mode: {control_mode}")
        if target_hz <= minimum_hz:
            raise ValueError(
                f"CAN-FD target rate ({target_hz:g} Hz) must be strictly above "
                f"the minimum ({minimum_hz:g} Hz)."
            )
        if minimum_hz < 100.0:
            raise ValueError("CAN-FD minimum rate must be at least 100 Hz.")
        if rate_check_window <= 0.0:
            raise ValueError("CAN-FD rate check window must be positive.")
        if maximum_failure_windows < 1:
            raise ValueError("maximum_failure_windows must be at least 1.")
        if trajectory_mode not in {0, 1, 2}:
            raise ValueError("trajectory_mode must be 0, 1, or 2.")
        if radio < 0:
            raise ValueError("radio cannot be negative.")
        if trajectory_mode == 1 and radio > 100:
            raise ValueError("curve-fit CAN-FD radio must be between 0 and 100.")
        if trajectory_mode == 2 and radio > 999:
            raise ValueError("filter CAN-FD radio must be between 0 and 999.")

        required_method = "rm_movej_canfd" if control_mode == "joint" else "rm_movep_canfd"
        if not callable(getattr(arm, required_method, None)):
            raise TypeError(f"RealMan SDK object does not expose {required_method}().")

        self.arm = arm
        self.control_mode = control_mode
        self.dof = int(dof)
        self.target_hz = float(target_hz)
        self.minimum_hz = float(minimum_hz)
        self.period_s = 1.0 / self.target_hz
        self.rate_check_window = float(rate_check_window)
        self.maximum_failure_windows = int(maximum_failure_windows)
        self.trajectory_mode = int(trajectory_mode)
        self.radio = int(radio)
        if joint_speed_limits is None:
            self.joint_speed_limits = np.full(self.dof, np.inf)
        else:
            limits = np.asarray(joint_speed_limits, dtype=float)
            if limits.ndim == 0:
                limits = np.full(self.dof, float(limits))
            if limits.shape != (self.dof,) or np.any(limits <= 0.0):
                raise ValueError(
                    f"joint_speed_limits must be positive with shape ({self.dof},)."
                )
            self.joint_speed_limits = limits
        if joint_acceleration_limits is None:
            self.joint_acceleration_limits = np.full(self.dof, np.inf)
        else:
            acceleration_limits = np.asarray(
                joint_acceleration_limits,
                dtype=float,
            )
            if acceleration_limits.ndim == 0:
                acceleration_limits = np.full(
                    self.dof,
                    float(acceleration_limits),
                )
            if (
                acceleration_limits.shape != (self.dof,)
                or np.any(acceleration_limits <= 0.0)
            ):
                raise ValueError(
                    "joint_acceleration_limits must be positive with shape "
                    f"({self.dof},)."
                )
            self.joint_acceleration_limits = acceleration_limits
        self.linear_speed_limit = (
            np.inf if linear_speed_limit is None else float(linear_speed_limit)
        )
        self.linear_acceleration_limit = (
            np.inf
            if linear_acceleration_limit is None
            else float(linear_acceleration_limit)
        )
        self.angular_speed_limit = (
            np.inf if angular_speed_limit is None else float(angular_speed_limit)
        )
        self.angular_acceleration_limit = (
            np.inf
            if angular_acceleration_limit is None
            else float(angular_acceleration_limit)
        )
        if self.linear_speed_limit <= 0.0:
            raise ValueError("linear_speed_limit must be positive.")
        if self.linear_acceleration_limit <= 0.0:
            raise ValueError("linear_acceleration_limit must be positive.")
        if self.angular_speed_limit <= 0.0:
            raise ValueError("angular_speed_limit must be positive.")
        if self.angular_acceleration_limit <= 0.0:
            raise ValueError("angular_acceleration_limit must be positive.")
        self.heartbeat_timeout = float(heartbeat_timeout)
        if self.heartbeat_timeout <= 0.01:
            raise ValueError("heartbeat_timeout must be greater than 10 ms.")

        self._target_lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._setpoint: np.ndarray | None = None
        self._joint_velocity = np.zeros(self.dof)
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._planned_joint_velocity = np.zeros(self.dof)
        self._planned_linear_velocity = np.zeros(3)
        self._planned_angular_velocity = np.zeros(3)
        self._hold_requested = False
        self._pending_lock = threading.Lock()
        self._pending_joint_target: (
            tuple[np.ndarray, float, int] | None
        ) = None
        self._joint_request_generation = 0
        self._joint_target_resolver: (
            Callable[[np.ndarray, float], np.ndarray | None] | None
        ) = None
        self._continuous_joint_resolution = False
        self._maintenance_callback: Callable[[], None] | None = None
        self._maintenance_period_ns = 0

        self._stats_lock = threading.Lock()
        self._achieved_hz: float | None = None
        self._total_commands = 0
        self._deadline_misses = 0
        self._high_follow_gap_violations = 0
        self._sdk_call_overruns = 0
        self._max_gap_ms = 0.0
        self._max_sdk_call_ms = 0.0
        self._last_command_start_ns = 0
        self._last_command_success_ns = 0
        self._completed_timing_windows = 0
        self._consecutive_timing_failure_windows = 0
        self._timing_verified = False
        self._running = False
        self._error = ""

    def send_once(self) -> None:
        """Send one high-follow packet. Exposed for hardware-adapter tests."""

        target = self._next_setpoint()
        if self.control_mode == "joint":
            result = self.arm.rm_movej_canfd(
                np.degrees(target).tolist(),
                True,
                0,
                self.trajectory_mode,
                self.radio,
            )
            method_name = "rm_movej_canfd"
        else:
            result = self.arm.rm_movep_canfd(
                target.tolist(),
                True,
                self.trajectory_mode,
                self.radio,
            )
            method_name = "rm_movep_canfd"

        if result != 0:
            raise RuntimeError(f"{method_name} failed with RealMan error code {result}.")
        self._commit_setpoint(target)

    def _record_command_gap(
        self,
        previous_start_ns: int | None,
        start_ns: int,
    ) -> tuple[bool, int]:
        if previous_start_ns is None:
            return False, 0
        gap_ns = start_ns - previous_start_ns
        gap_ms = gap_ns / 1e6
        violated = gap_ms > 10.0
        with self._stats_lock:
            self._max_gap_ms = max(self._max_gap_ms, gap_ms)
            # High-follow requires no more than 10 ms between packets.
            if violated:
                self._high_follow_gap_violations += 1
        return violated, gap_ns

    def _record_sdk_call_duration(self, duration_ns: int) -> bool:
        duration_ms = duration_ns / 1e6
        violated = duration_ms > 10.0
        with self._stats_lock:
            self._max_sdk_call_ms = max(self._max_sdk_call_ms, duration_ms)
            if violated:
                self._sdk_call_overruns += 1
        return violated

    def _finish_rate_window(
        self,
        interval_count: int,
        elapsed_s: float,
        gap_violations: int,
    ) -> tuple[bool, bool]:
        achieved_hz = interval_count / elapsed_s
        with self._stats_lock:
            self._completed_timing_windows += 1
            first_measurement = self._achieved_hz is None
            self._achieved_hz = achieved_hz
            rate_too_slow = achieved_hz <= self.minimum_hz
            if rate_too_slow or gap_violations:
                self._consecutive_timing_failure_windows += 1
            else:
                self._consecutive_timing_failure_windows = 0
            timing_failed = (
                self._consecutive_timing_failure_windows
                >= self.maximum_failure_windows
            )

        if first_measurement:
            utils.logger.info(
                f"RealMan CAN-FD measured rate: {achieved_hz:.1f} Hz "
                f"(target {self.target_hz:g} Hz)"
            )
        if rate_too_slow:
            utils.logger.warning(
                f"RealMan CAN-FD rate is too slow: {achieved_hz:.1f} Hz "
                f"(minimum must be > {self.minimum_hz:g} Hz)."
            )
        if gap_violations:
            utils.logger.warning(
                f"RealMan CAN-FD had {gap_violations} packet gap(s) above 10 ms "
                "in the latest timing window."
            )
        return timing_failed, rate_too_slow

    def run(self, stop_event: threading.Event) -> None:
        """Run until stopped, or stop the application after sustained low rate."""

        period_ns = max(1, round(self.period_s * 1e9))
        window_ns = max(1, round(self.rate_check_window * 1e9))
        previous_start_ns: int | None = None
        window_start_ns = time.perf_counter_ns()
        window_intervals = 0
        window_interval_ns = 0
        window_gap_violations = 0
        next_tick_ns = window_start_ns
        next_maintenance_ns = window_start_ns

        with self._stats_lock:
            self._running = True
            self._error = ""
            self._timing_verified = False

        try:
            while not stop_event.is_set():
                command_start_ns = time.perf_counter_ns()
                gap_violated, gap_ns = self._record_command_gap(
                    previous_start_ns,
                    command_start_ns,
                )
                with self._stats_lock:
                    timing_verified = self._timing_verified
                if timing_verified and gap_violated:
                    raise RuntimeError(
                        "CAN-FD packet gap exceeded the 10 ms high-follow "
                        f"limit ({gap_ns / 1e6:.2f} ms)."
                    )
                with self._stats_lock:
                    self._last_command_start_ns = command_start_ns
                # The teleoperation QP solver is a continuous controller: use
                # the latest Cartesian target at this loop's fixed dT before
                # emitting each joint packet. Legacy one-shot IK remains after
                # the packet so existing adapters do not change behavior.
                if self._continuous_joint_resolution:
                    self.resolve_pending_target()
                self.send_once()
                command_complete_ns = time.perf_counter_ns()
                with self._stats_lock:
                    self._last_command_success_ns = command_complete_ns
                call_violated = self._record_sdk_call_duration(
                    command_complete_ns - command_start_ns
                )
                window_gap_violations += int(gap_violated or call_violated)
                if timing_verified and call_violated:
                    raise RuntimeError(
                        "CAN-FD SDK call exceeded the 10 ms high-follow limit "
                        f"({(command_complete_ns - command_start_ns) / 1e6:.2f} ms)."
                    )
                if gap_ns:
                    window_intervals += 1
                    window_interval_ns += gap_ns
                previous_start_ns = command_start_ns

                with self._stats_lock:
                    self._total_commands += 1

                now_ns = time.perf_counter_ns()
                window_elapsed_ns = now_ns - window_start_ns
                if window_elapsed_ns >= window_ns and window_intervals:
                    interval_elapsed_s = window_interval_ns / 1e9
                    timing_failed, rate_too_slow = self._finish_rate_window(
                        window_intervals,
                        interval_elapsed_s,
                        window_gap_violations,
                    )
                    if timing_failed:
                        snapshot = self.snapshot()
                        if rate_too_slow:
                            reason = (
                                f"send rate remained at or below {self.minimum_hz:g} Hz "
                                f"(latest: {snapshot.achieved_hz:.1f} Hz)"
                            )
                        else:
                            reason = "packet timing repeatedly exceeded the 10 ms high-follow limit"
                        raise RuntimeError(
                            f"CAN-FD {reason} for "
                            f"{snapshot.consecutive_timing_failure_windows} "
                            "consecutive timing windows."
                        )
                    window_start_ns = now_ns
                    window_intervals = 0
                    window_interval_ns = 0
                    window_gap_violations = 0

                if not self._continuous_joint_resolution:
                    self.resolve_pending_target()
                if (
                    self._maintenance_callback is not None
                    and time.perf_counter_ns() >= next_maintenance_ns
                ):
                    self._maintenance_callback()
                    next_maintenance_ns += self._maintenance_period_ns
                    now_ns = time.perf_counter_ns()
                    if next_maintenance_ns <= now_ns:
                        next_maintenance_ns = (
                            now_ns + self._maintenance_period_ns
                        )

                next_tick_ns += period_ns
                remaining_s = (next_tick_ns - time.perf_counter_ns()) / 1e9
                if remaining_s > 0.0:
                    stop_event.wait(remaining_s)
                else:
                    with self._stats_lock:
                        self._deadline_misses += 1
                    # Do not issue a burst of stale packets after an overrun.
                    next_tick_ns = time.perf_counter_ns()

        except Exception as exc:
            with self._stats_lock:
                self._error = str(exc)
            utils.logger.exception("RealMan CAN-FD loop stopped")
            stop_event.set()
        finally:
            with self._stats_lock:
                self._running = False

    def snapshot(self) -> CanfdLoopSnapshot:
        with self._stats_lock:
            return CanfdLoopSnapshot(
                target_hz=self.target_hz,
                achieved_hz=self._achieved_hz,
                total_commands=self._total_commands,
                deadline_misses=self._deadline_misses,
                high_follow_gap_violations=self._high_follow_gap_violations,
                sdk_call_overruns=self._sdk_call_overruns,
                max_gap_ms=self._max_gap_ms,
                max_sdk_call_ms=self._max_sdk_call_ms,
                last_command_start_ns=self._last_command_start_ns,
                last_command_success_ns=self._last_command_success_ns,
                completed_timing_windows=self._completed_timing_windows,
                consecutive_timing_failure_windows=(
                    self._consecutive_timing_failure_windows
                ),
                timing_verified=self._timing_verified,
                running=self._running,
                error=self._error,
            )

    def heartbeat_error(self) -> str:
        snapshot = self.snapshot()
        now_ns = time.perf_counter_ns()
        timeout_ns = self.heartbeat_timeout * 1e9
        if (
            snapshot.last_command_start_ns > snapshot.last_command_success_ns
            and now_ns - snapshot.last_command_start_ns > timeout_ns
        ):
            return (
                "CAN-FD SDK call has not completed for "
                f"{(now_ns - snapshot.last_command_start_ns) / 1e6:.1f} ms."
            )
        if (
            snapshot.running
            and snapshot.last_command_success_ns
            and now_ns - snapshot.last_command_success_ns > timeout_ns
        ):
            return (
                "CAN-FD has not completed a successful packet for "
                f"{(now_ns - snapshot.last_command_success_ns) / 1e6:.1f} ms."
            )
        return ""

    def report_external_failure(
        self,
        message: str,
        stop_event: threading.Event,
    ) -> None:
        with self._stats_lock:
            if not self._error:
                self._error = message
        stop_event.set()

    def wait_until_healthy(
        self,
        stop_event: threading.Event,
        timeout: float | None = None,
    ) -> CanfdLoopSnapshot:
        """Wait for a new clean timing window before enabling motion input."""

        if timeout is None:
            timeout = self.rate_check_window * (self.maximum_failure_windows + 2)
        deadline = time.monotonic() + timeout
        initial_window_count = self.snapshot().completed_timing_windows

        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.error:
                raise RuntimeError(snapshot.error)
            heartbeat_error = self.heartbeat_error()
            if heartbeat_error:
                self.report_external_failure(heartbeat_error, stop_event)
                raise RuntimeError(heartbeat_error)
            if (
                snapshot.completed_timing_windows > initial_window_count
                and snapshot.achieved_hz is not None
                and snapshot.achieved_hz > self.minimum_hz
                and snapshot.consecutive_timing_failure_windows == 0
            ):
                with self._stats_lock:
                    self._timing_verified = True
                return self.snapshot()
            if stop_event.wait(0.01):
                snapshot = self.snapshot()
                if snapshot.error:
                    raise RuntimeError(snapshot.error)
                raise RuntimeError("CAN-FD timing check stopped before it completed.")

        raise TimeoutError(
            f"CAN-FD did not produce a clean >{self.minimum_hz:g} Hz timing "
            f"window within {timeout:.1f} seconds."
        )
