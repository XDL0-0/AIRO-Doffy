"""RealMan sensor cache, realtime state push and lifecycle ownership."""

from __future__ import annotations

import threading
import time

import numpy as np
from airo_spatial_algebra.se3 import SE3Container

import doffy_teleop.utils as utils
from ..control.contracts import RealManStateSnapshot


_UNCLOSED_REALMAN_TELEOPS: list[object] = []


class StateLifecycleMixin:
    """Own sensor callbacks, worker lifecycle and safe SDK teardown."""

    def _refresh_sensors(self) -> None:
        state_error = ""
        force_error = ""
        joints: np.ndarray | None = None
        tcp_pose: np.ndarray | None = None
        wrench: np.ndarray | None = None

        try:
            joints = np.asarray(self.backend.get_joint_configuration(), dtype=float)
            tcp_pose = np.asarray(self.backend.get_tcp_pose(), dtype=float)
            self._validate_sensor_state(joints, tcp_pose)
        except Exception as exc:
            state_error = str(exc)

        try:
            raw_wrench = self.backend.get_tcp_force()
            if raw_wrench is None:
                raise RuntimeError("RealMan force sensor returned no wrench.")
            raw_wrench = np.asarray(raw_wrench, dtype=float)
            self._validate_wrench(raw_wrench)
            wrench = self.wrench_filter.process(raw_wrench)
        except Exception as exc:
            force_error = str(exc)

        now_ns = time.monotonic_ns()
        with self._sensor_lock:
            previous_state_error = self._state_error
            previous_force_error = self._force_error
            if joints is not None and tcp_pose is not None:
                self._joints = joints
                self._tcp_pose = tcp_pose
                self._state_timestamp_ns = now_ns
            if wrench is not None:
                self._wrench = wrench
                self._force_timestamp_ns = now_ns
            self._state_error = state_error
            self._force_error = force_error

        if state_error and state_error != previous_state_error:
            utils.logger.warning(f"RealMan state read failed: {state_error}")
        if force_error and force_error != previous_force_error:
            utils.logger.warning(f"RealMan force read failed: {force_error}")

    def _validate_sensor_state(
        self,
        joints: np.ndarray,
        tcp_pose: np.ndarray,
    ) -> None:
        if joints.shape != (self.dof,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(
                f"RealMan returned invalid joints with shape {joints.shape}."
            )
        if tcp_pose.shape != (4, 4) or not np.all(np.isfinite(tcp_pose)):
            raise RuntimeError(
                f"RealMan returned an invalid TCP pose with shape {tcp_pose.shape}."
            )

    @staticmethod
    def _validate_wrench(wrench: np.ndarray) -> None:
        if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
            raise RuntimeError(
                f"RealMan returned an invalid wrench with shape {wrench.shape}."
            )

    def _handle_realtime_state(self, state) -> None:
        """Copy one vendor UDP state-push callback into the local cache."""

        try:
            if int(state.errCode) != 0:
                raise RuntimeError(
                    f"RealMan realtime state parse error {int(state.errCode)}."
                )

            joints = np.radians(
                np.asarray(
                    list(state.joint_status.joint_position)[: self.dof],
                    dtype=float,
                )
            )
            waypoint = state.waypoint
            robot_tcp_pose = (
                SE3Container.from_euler_angles_and_translation(
                    np.array(
                        [
                            waypoint.euler.rx,
                            waypoint.euler.ry,
                            waypoint.euler.rz,
                        ],
                        dtype=float,
                    ),
                    np.array(
                        [
                            waypoint.position.x,
                            waypoint.position.y,
                            waypoint.position.z,
                        ],
                        dtype=float,
                    ),
                ).homogeneous_matrix
            )
            tcp_pose = robot_tcp_pose @ self.tcp_transform
            raw_wrench = np.asarray(
                list(state.force_sensor.zero_force),
                dtype=float,
            )
            force_coordinate = int(
                getattr(
                    state.force_sensor,
                    "coordinate",
                    self.cfg.REALMAN_FORCE_COORDINATE,
                )
            )
            if force_coordinate != self.cfg.REALMAN_FORCE_COORDINATE:
                raise RuntimeError(
                    "RealMan force frame mismatch: expected "
                    f"{self.cfg.REALMAN_FORCE_COORDINATE}, got "
                    f"{force_coordinate}."
                )
            self._validate_sensor_state(joints, tcp_pose)
            self._validate_wrench(raw_wrench)
            if not self._state_push_received.is_set():
                # The synchronous startup read is in the sensor frame. Reset
                # before the first pushed sample so work/tool-frame filtering
                # never mixes wrenches expressed in different coordinates.
                self.wrench_filter.reset()
            wrench = self.wrench_filter.process(raw_wrench)

            now_ns = time.monotonic_ns()
            with self._sensor_lock:
                self._joints = joints
                self._tcp_pose = tcp_pose
                self._wrench = wrench
                self._state_timestamp_ns = now_ns
                self._force_timestamp_ns = now_ns
                self._state_error = ""
                self._force_error = ""
            self._state_push_received.set()

        except Exception as exc:
            message = str(exc)
            with self._sensor_lock:
                self._state_error = message
                self._force_error = message

    def _state_push_callback_entry(self, state) -> None:
        """Reject late packets during shutdown and track callbacks in flight."""

        with self._state_callback_condition:
            if not self._accept_state_callbacks:
                return
            self._state_callbacks_in_flight += 1
        try:
            self._handle_realtime_state(state)
        finally:
            with self._state_callback_condition:
                self._state_callbacks_in_flight -= 1
                if self._state_callbacks_in_flight == 0:
                    self._state_callback_condition.notify_all()

    def _stop_accepting_state_callbacks(self) -> None:
        deadline = time.monotonic() + self.cfg.REALMAN_STATE_PUSH_TIMEOUT
        with self._state_callback_condition:
            self._accept_state_callbacks = False
            while self._state_callbacks_in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Timed out waiting for a RealMan realtime state "
                        "callback to finish."
                    )
                self._state_callback_condition.wait(remaining)

    def _start_realtime_state_push(self) -> None:
        api = self._realman_api
        if api is None:
            raise RuntimeError(
                "Installed airo-robots RealMan wrapper does not expose its SDK module."
            )
        callback_type = getattr(api, "rm_realtime_arm_state_callback_ptr", None)
        config_type = getattr(api, "rm_realtime_push_config_t", None)
        if callback_type is None or config_type is None:
            raise RuntimeError(
                "Installed RealMan SDK does not support realtime UDP state push."
            )
        if not callable(
            getattr(self._raw_arm, "rm_realtime_arm_state_call_back", None)
        ) or not callable(getattr(self._raw_arm, "rm_set_realtime_push", None)):
            raise RuntimeError(
                "RealMan SDK arm does not expose realtime state-push methods."
            )

        self._state_push_received.clear()
        with self._state_callback_condition:
            self._accept_state_callbacks = True
        self._state_push_callback = callback_type(
            self._state_push_callback_entry
        )
        self._raw_arm.rm_realtime_arm_state_call_back(
            self._state_push_callback
        )
        config = config_type(
            # The SDK/wire value counts 5 ms intervals, not milliseconds.
            cycle=self.cfg.REALMAN_STATE_PUSH_CYCLE_MS // 5,
            enable=True,
            port=self.cfg.REALMAN_STATE_PUSH_PORT,
            force_coordinate=self.cfg.REALMAN_FORCE_COORDINATE,
            ip=getattr(self.cfg, "REALMAN_STATE_PUSH_IP", None) or self.cfg.PC_IP,
        )
        result = self._raw_arm.rm_set_realtime_push(config)
        if result != 0:
            self._stop_accepting_state_callbacks()
            raise RuntimeError(
                "rm_set_realtime_push failed with RealMan error code "
                f"{result}."
            )
        self._state_push_active = True

        if not self._state_push_received.wait(self.cfg.REALMAN_STATE_PUSH_TIMEOUT):
            self._stop_realtime_state_push()
            raise TimeoutError(
                "No RealMan realtime state packet arrived. Check Config.REALMAN_STATE_PUSH_IP "
                "(or PC_IP when unset), "
                f"UDP port {self.cfg.REALMAN_STATE_PUSH_PORT}, and the firewall."
            )
        utils.logger.info(
            "RealMan realtime joint/TCP/force push enabled at "
            f"{self.cfg.REALMAN_STATE_PUSH_CYCLE_MS} ms."
        )

    def _stop_realtime_state_push(self) -> None:
        disable_error: Exception | None = None
        if self._state_push_active:
            try:
                config_type = getattr(
                    self._realman_api,
                    "rm_realtime_push_config_t",
                )
                config = config_type(
                    cycle=self.cfg.REALMAN_STATE_PUSH_CYCLE_MS // 5,
                    enable=False,
                    port=self.cfg.REALMAN_STATE_PUSH_PORT,
                    force_coordinate=self.cfg.REALMAN_FORCE_COORDINATE,
                    ip=getattr(self.cfg, "REALMAN_STATE_PUSH_IP", None) or self.cfg.PC_IP,
                )
                result = self._raw_arm.rm_set_realtime_push(config)
                if result != 0:
                    disable_error = RuntimeError(
                        "Disabling RealMan realtime state push failed with "
                        f"error code {result}."
                    )
            except Exception as exc:
                disable_error = exc
            finally:
                self._state_push_active = False

        self._stop_accepting_state_callbacks()
        if disable_error is not None:
            utils.logger.warning(
                f"Could not disable RealMan realtime state push: "
                f"{disable_error}"
            )

    def start(self, stop_event: threading.Event) -> list[threading.Thread]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Cannot start a closed RealMan teleoperation.")
            if self._quarantined:
                raise RuntimeError(
                    "Cannot restart a quarantined RealMan teleoperation."
                )
            if self._started:
                raise RuntimeError(
                    "RealMan teleoperation has already been started."
                )
            self._started = True

            if self.cfg.REALMAN_REALTIME_STATE_PUSH:
                self._start_realtime_state_push()
            else:
                utils.logger.warning(
                    "REALMAN_REALTIME_STATE_PUSH is disabled; synchronous "
                    "state reads are serialized on the CAN-FD owner thread "
                    "and may consume the timing budget."
                )

            command_thread = threading.Thread(
                target=self.canfd.run,
                args=(stop_event,),
                name="realman-canfd",
                daemon=True,
            )
            started: list[threading.Thread] = []
            self._sdk_worker_threads = started
            try:
                command_thread.start()
                started.append(command_thread)
            except Exception:
                stop_event.set()
                for thread in started:
                    thread.join(timeout=1.0)
                if not self.live_sdk_workers():
                    self._stop_realtime_state_push()
                raise
            return list(started)

    @property
    def sensor_stale_after_s(self) -> float:
        if self.cfg.REALMAN_REALTIME_STATE_PUSH:
            push_period_s = self.cfg.REALMAN_STATE_PUSH_CYCLE_MS / 1000.0
            return max(0.1, 4.0 * push_period_s)
        return max(0.2, 3.0 / self.cfg.REALMAN_SENSOR_RATE)

    def live_sdk_workers(self) -> list[str]:
        return [
            thread.name
            for thread in self._sdk_worker_threads
            if thread.is_alive()
        ]

    def sdk_worker_threads(self) -> tuple[threading.Thread, ...]:
        return tuple(self._sdk_worker_threads)

    def state_snapshot(self) -> RealManStateSnapshot:
        with self._sensor_lock:
            joints = self._joints.copy()
            tcp_pose = self._tcp_pose.copy()
            wrench = self._wrench.copy()
            state_timestamp_ns = self._state_timestamp_ns
            force_timestamp_ns = self._force_timestamp_ns
            state_error = self._state_error
            force_error = self._force_error
        with self._control_lock:
            input_stale = self._input_stale
        return RealManStateSnapshot(
            joints=joints,
            tcp_pose=tcp_pose,
            wrench=wrench,
            state_timestamp_ns=state_timestamp_ns,
            force_timestamp_ns=force_timestamp_ns,
            state_error=state_error,
            force_error=force_error,
            input_stale=input_stale,
        )

    def recording_snapshot(
        self,
    ) -> tuple[RealManStateSnapshot, np.ndarray, np.ndarray, int]:
        """Copy cached measured state and the latest teleoperation targets."""

        state = self.state_snapshot()
        with self._control_lock:
            if self.control_mode == "joint":
                action_joints = self._last_joint_target.copy()
            else:
                # Direct Cartesian CAN-FD has no host-side joint target. The
                # measured joints provide a replayable joint demonstration
                # when DATA_TYPE is qpos/both; DATA_TYPE=tcp records the actual
                # Cartesian command below.
                action_joints = state.joints.copy()
            action_tcp = self._last_tcp_target.copy()
            action_timestamp_ns = self._action_timestamp_ns
        return state, action_joints, action_tcp, action_timestamp_ns

    def reset_active(self) -> bool:
        with self._control_lock:
            return self._reset_in_progress

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            live_workers = self.live_sdk_workers()
            if live_workers:
                raise RuntimeError(
                    "Refusing to close the RealMan SDK handle while worker(s) "
                    f"are still alive: {', '.join(live_workers)}."
                )
            self._stop_realtime_state_push()
            self.backend.cleanup()
            self._closed = True
            _UNCLOSED_REALMAN_TELEOPS[:] = [
                teleop
                for teleop in _UNCLOSED_REALMAN_TELEOPS
                if teleop is not self
            ]

    def quarantine_without_sdk_cleanup(self) -> None:
        """Keep callback memory alive when an in-flight SDK call prevents close."""

        callback_error: Exception | None = None
        with self._lifecycle_lock:
            if self._closed:
                return
            self._quarantined = True
            try:
                # This only changes Python-side callback state. It deliberately
                # makes no SDK call while another thread may be stuck inside one.
                self._stop_accepting_state_callbacks()
            except Exception as exc:
                callback_error = exc
            finally:
                if not any(
                    retained is self
                    for retained in _UNCLOSED_REALMAN_TELEOPS
                ):
                    _UNCLOSED_REALMAN_TELEOPS.append(self)
        if callback_error is not None:
            raise callback_error
