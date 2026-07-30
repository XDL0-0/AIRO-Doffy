import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from airo_spatial_algebra.se3 import SE3Container
from scipy.spatial.transform import Rotation

from config import Config
from realman_teleop import (
    _UNCLOSED_REALMAN_TELEOPS,
    CanfdCommandLoop,
    RealManTeleop,
    visualizer_publish_loop,
)
from visualizer_config import VisualizerConfig


class FakeRealManArm:
    def __init__(self, delay_s: float = 0.0, result: int = 0) -> None:
        self.delay_s = delay_s
        self.result = result
        self.joint_calls: list[tuple] = []
        self.tcp_calls: list[tuple] = []
        self.realtime_callback = None
        self.push_configs: list[object] = []

    def rm_movej_canfd(
        self,
        joints,
        follow,
        expand=0,
        trajectory_mode=0,
        radio=0,
    ):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.joint_calls.append(
            (joints, follow, expand, trajectory_mode, radio)
        )
        return self.result

    def rm_movep_canfd(
        self,
        pose,
        follow,
        trajectory_mode=0,
        radio=0,
    ):
        if self.delay_s:
            time.sleep(self.delay_s)
        self.tcp_calls.append((pose, follow, trajectory_mode, radio))
        return self.result

    def rm_realtime_arm_state_call_back(self, callback) -> None:
        self.realtime_callback = callback

    def rm_set_realtime_push(self, config) -> int:
        self.push_configs.append(config)
        if config.enable and self.realtime_callback is not None:
            self.realtime_callback(realtime_state())
        return 0


class FakeRobotWrapper:
    def __init__(self, arm: FakeRealManArm, api=None) -> None:
        self.robot = arm
        self._api = api
        self._joint_lower_limits = np.full(7, -np.pi)
        self._joint_upper_limits = np.full(7, np.pi)


class FakeRealManBackend:
    name = "realman"
    supports_force = True
    dof = 7

    def __init__(self, arm: FakeRealManArm, api=None) -> None:
        self.robot = FakeRobotWrapper(arm, api)
        self.joints = np.zeros(7)
        self.tcp_pose = np.eye(4)
        self.tcp_pose[:3, 3] = [0.4, 0.0, 0.3]
        self.cleaned = False
        self.ik_calls: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.sensor_read_thread_ids: list[int] = []

    def initial_joint_configuration(self, configured):
        return np.asarray(configured, dtype=float)

    def reset(self, joints) -> None:
        self.joints = np.asarray(joints, dtype=float).copy()

    def get_joint_configuration(self):
        self.sensor_read_thread_ids.append(threading.get_ident())
        return self.joints.copy()

    def get_tcp_pose(self):
        self.sensor_read_thread_ids.append(threading.get_ident())
        return self.tcp_pose.copy()

    def get_tcp_force(self):
        self.sensor_read_thread_ids.append(threading.get_ident())
        return np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    def solve_tcp_ik(self, tcp_pose, seed=None):
        seed = self.joints if seed is None else np.asarray(seed, dtype=float)
        self.ik_calls.append(
            (
                np.asarray(tcp_pose, dtype=float).copy(),
                seed.copy(),
                threading.get_ident(),
            )
        )
        return seed + 0.01

    def is_joint_target_safe(
        self,
        joints,
        previous_joints,
        tcp_position,
        joint_threshold,
    ):
        del previous_joints, tcp_position, joint_threshold
        return np.asarray(joints).shape == (7,)

    def cleanup(self) -> None:
        self.cleaned = True


def controller_data(*, x: float = 0.0, grip: bool = False) -> list[dict]:
    empty = {
        "Position": (0.0, 0.0, 0.0),
        "Rotation": (0.0, 0.0, 0.0, 1.0),
        "GripTrigger": 0.0,
    }
    right = dict(empty)
    right["Position"] = (x, 0.0, 0.0)
    right["GripTrigger"] = float(grip)
    return [empty, right]


def make_loop(
    arm: FakeRealManArm,
    *,
    mode: str = "joint",
    target_hz: float = 200.0,
    minimum_hz: float = 100.0,
    window_s: float = 0.05,
    maximum_failure_windows: int = 2,
    joint_speed_limits: float | np.ndarray | None = None,
    linear_speed_limit: float | None = None,
    angular_speed_limit: float | None = None,
    heartbeat_timeout: float = 0.05,
) -> CanfdCommandLoop:
    return CanfdCommandLoop(
        arm,
        control_mode=mode,
        dof=7,
        target_hz=target_hz,
        minimum_hz=minimum_hz,
        rate_check_window=window_s,
        maximum_failure_windows=maximum_failure_windows,
        trajectory_mode=0,
        radio=0,
        joint_speed_limits=joint_speed_limits,
        linear_speed_limit=linear_speed_limit,
        angular_speed_limit=angular_speed_limit,
        heartbeat_timeout=heartbeat_timeout,
    )


def realtime_state(
    *,
    joints_degrees: list[float] | None = None,
    translation: tuple[float, float, float] = (0.45, -0.1, 0.25),
    euler: tuple[float, float, float] = (0.1, -0.2, 0.3),
    wrench: list[float] | None = None,
    error_code: int = 0,
):
    if joints_degrees is None:
        joints_degrees = [0.0, 10.0, -20.0, 30.0, -40.0, 50.0, -60.0]
    if wrench is None:
        wrench = [4.0, 5.0, 6.0, 0.4, 0.5, 0.6]
    return SimpleNamespace(
        errCode=error_code,
        joint_status=SimpleNamespace(joint_position=joints_degrees),
        waypoint=SimpleNamespace(
            position=SimpleNamespace(
                x=translation[0],
                y=translation[1],
                z=translation[2],
            ),
            euler=SimpleNamespace(
                rx=euler[0],
                ry=euler[1],
                rz=euler[2],
            ),
        ),
        force_sensor=SimpleNamespace(zero_force=wrench),
    )


class FakeRealtimePushConfig:
    def __init__(
        self,
        *,
        cycle: int,
        enable: bool,
        port: int,
        force_coordinate: int,
        ip: str,
    ) -> None:
        self.cycle = cycle
        self.enable = enable
        self.port = port
        self.force_coordinate = force_coordinate
        self.ip = ip


def fake_realtime_api():
    return SimpleNamespace(
        rm_realtime_arm_state_callback_ptr=lambda callback: callback,
        rm_realtime_push_config_t=FakeRealtimePushConfig,
    )


def realman_config(**overrides) -> Config:
    values = {
        "ROBOT_TYPE": "realman",
        "ROBOT_IP": "192.0.2.1",
        "FREEZE_ROTATION": False,
    }
    values.update(overrides)
    return Config(**values)


class CanfdCommandLoopTest(unittest.TestCase):
    def test_config_rate_is_strictly_above_100_hz(self) -> None:
        cfg = realman_config()
        self.assertGreater(cfg.REALMAN_CTRL_RATE, 100.0)
        self.assertGreater(cfg.REALMAN_CTRL_RATE, cfg.REALMAN_MIN_CANFD_RATE)

        with self.assertRaises(ValueError):
            make_loop(
                FakeRealManArm(),
                target_hz=100.0,
                minimum_hz=100.0,
            )

    def test_joint_target_is_converted_to_degrees_and_high_follow(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm)
        joints_degrees = np.array([0.0, 10.0, -20.0, 30.0, -40.0, 50.0, -60.0])

        loop.set_joint_target(np.radians(joints_degrees))
        loop.send_once()

        values, follow, expand, trajectory_mode, radio = arm.joint_calls[-1]
        np.testing.assert_allclose(values, joints_degrees)
        self.assertTrue(follow)
        self.assertEqual((expand, trajectory_mode, radio), (0, 0, 0))

    def test_tcp_target_uses_movep_canfd_and_high_follow(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm, mode="tcp")
        pose = np.array([0.35, -0.1, 0.42, 0.2, -0.3, 0.4])

        loop.set_tcp_target(pose)
        loop.send_once()

        values, follow, trajectory_mode, radio = arm.tcp_calls[-1]
        np.testing.assert_allclose(values, pose)
        self.assertTrue(follow)
        self.assertEqual((trajectory_mode, radio), (0, 0))

    def test_joint_slew_is_limited_on_every_canfd_packet(self) -> None:
        arm = FakeRealManArm()
        speed_limits = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        loop = make_loop(arm, joint_speed_limits=speed_limits)
        initial = np.zeros(7)
        direction = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        loop.set_joint_target(initial)
        loop.set_joint_target(direction * 0.5)

        loop.send_once()
        first = np.radians(np.asarray(arm.joint_calls[-1][0], dtype=float))
        expected_step = direction * speed_limits / loop.target_hz
        np.testing.assert_allclose(first, expected_step, atol=1e-12)

        loop.send_once()
        second = np.radians(np.asarray(arm.joint_calls[-1][0], dtype=float))
        np.testing.assert_allclose(second, 2.0 * expected_step, atol=1e-12)
        np.testing.assert_allclose(second - first, expected_step, atol=1e-12)

    def test_bounded_joint_interpolation_does_not_wrap_across_pi(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm, joint_speed_limits=1.0)
        initial = np.full(7, 3.0)
        target = np.full(7, -3.0)
        loop.set_joint_target(initial)
        loop.set_joint_target(target)

        loop.send_once()

        sent = np.radians(np.asarray(arm.joint_calls[-1][0], dtype=float))
        np.testing.assert_allclose(
            sent,
            initial - 1.0 / loop.target_hz,
            atol=1e-12,
        )

    def test_hold_during_an_inflight_packet_does_not_reverse(self) -> None:
        arm = FakeRealManArm()
        packet_started = threading.Event()
        release_packet = threading.Event()

        def blocked_movej(
            joints,
            follow,
            expand=0,
            trajectory_mode=0,
            radio=0,
        ):
            packet_started.set()
            release_packet.wait(1.0)
            arm.joint_calls.append(
                (joints, follow, expand, trajectory_mode, radio)
            )
            return 0

        arm.rm_movej_canfd = blocked_movej
        loop = make_loop(arm, joint_speed_limits=0.2)
        loop.set_joint_target(np.zeros(7))
        loop.set_joint_target(np.ones(7))
        sender = threading.Thread(target=loop.send_once, daemon=True)
        sender.start()
        self.assertTrue(packet_started.wait(1.0))

        loop.hold_current_setpoint()
        release_packet.set()
        sender.join(timeout=1.0)
        self.assertFalse(sender.is_alive())
        loop.send_once()

        first = np.radians(np.asarray(arm.joint_calls[-2][0], dtype=float))
        second = np.radians(np.asarray(arm.joint_calls[-1][0], dtype=float))
        np.testing.assert_allclose(second, first, atol=1e-12)

    def test_tcp_slew_limits_translation_and_rotation_per_packet(self) -> None:
        arm = FakeRealManArm()
        linear_limit = 0.2
        angular_limit = 0.4
        loop = make_loop(
            arm,
            mode="tcp",
            linear_speed_limit=linear_limit,
            angular_speed_limit=angular_limit,
        )
        loop.set_tcp_target(np.zeros(6))
        target = np.array([0.03, 0.04, 0.0, 0.0, 0.0, 0.5])
        loop.set_tcp_target(target)

        loop.send_once()
        first = np.asarray(arm.tcp_calls[-1][0], dtype=float)
        expected_translation_step = (
            target[:3] / np.linalg.norm(target[:3])
            * linear_limit
            / loop.target_hz
        )
        np.testing.assert_allclose(
            first[:3],
            expected_translation_step,
            atol=1e-12,
        )
        first_pose = SE3Container.from_euler_angles_and_translation(
            first[3:],
            first[:3],
        )
        first_rotation_vector, _ = cv2.Rodrigues(first_pose.rotation_matrix)
        self.assertAlmostEqual(
            float(np.linalg.norm(first_rotation_vector)),
            angular_limit / loop.target_hz,
            places=10,
        )

        loop.send_once()
        second = np.asarray(arm.tcp_calls[-1][0], dtype=float)
        np.testing.assert_allclose(
            second[:3],
            2.0 * expected_translation_step,
            atol=1e-12,
        )
        second_pose = SE3Container.from_euler_angles_and_translation(
            second[3:],
            second[:3],
        )
        second_rotation_vector, _ = cv2.Rodrigues(second_pose.rotation_matrix)
        self.assertAlmostEqual(
            float(np.linalg.norm(second_rotation_vector)),
            2.0 * angular_limit / loop.target_hz,
            places=10,
        )

    def test_pending_ik_uses_only_latest_request_on_owner_thread(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm)
        loop.set_joint_target(np.zeros(7))
        resolver_called = threading.Event()
        resolver_calls: list[tuple[np.ndarray, float, int]] = []
        resolved_target = np.full(7, 0.02)

        def resolver(tcp_pose: np.ndarray, dt: float) -> np.ndarray:
            resolver_calls.append(
                (tcp_pose.copy(), dt, threading.get_ident())
            )
            resolver_called.set()
            return resolved_target

        loop.set_joint_target_resolver(resolver)
        first_pose = np.eye(4)
        first_pose[0, 3] = 0.1
        latest_pose = np.eye(4)
        latest_pose[1, 3] = -0.2
        loop.request_joint_target(first_pose, 0.01)
        loop.request_joint_target(latest_pose, 0.02)

        stop_event = threading.Event()
        thread = threading.Thread(target=loop.run, args=(stop_event,), daemon=True)
        thread.start()
        self.assertTrue(resolver_called.wait(1.0))

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if any(
                np.any(np.abs(np.asarray(call[0], dtype=float)) > 1e-9)
                for call in arm.joint_calls
            ):
                break
            time.sleep(0.005)
        stop_event.set()
        thread.join(timeout=1.0)

        self.assertEqual(len(resolver_calls), 1)
        resolved_pose, resolved_dt, resolver_thread_id = resolver_calls[0]
        np.testing.assert_allclose(resolved_pose, latest_pose)
        self.assertEqual(resolved_dt, 0.02)
        self.assertEqual(resolver_thread_id, thread.ident)
        self.assertNotEqual(resolver_thread_id, threading.get_ident())
        self.assertTrue(
            any(
                np.allclose(np.radians(np.asarray(call[0])), resolved_target)
                for call in arm.joint_calls
            )
        )

    def test_hold_cancels_an_ik_result_that_is_already_inflight(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm, joint_speed_limits=1.0)
        loop.set_joint_target(np.zeros(7))
        resolver_started = threading.Event()
        release_resolver = threading.Event()

        def resolver(_tcp_pose: np.ndarray, _dt: float) -> np.ndarray:
            resolver_started.set()
            release_resolver.wait(1.0)
            return np.ones(7)

        loop.set_joint_target_resolver(resolver)
        loop.request_joint_target(np.eye(4), 0.01)
        resolve_result: list[bool] = []
        resolver_thread = threading.Thread(
            target=lambda: resolve_result.append(
                loop.resolve_pending_target()
            ),
            daemon=True,
        )
        resolver_thread.start()
        self.assertTrue(resolver_started.wait(1.0))

        loop.hold_current_setpoint()
        release_resolver.set()
        resolver_thread.join(timeout=1.0)
        self.assertFalse(resolver_thread.is_alive())
        self.assertEqual(resolve_result, [False])

        loop.send_once()
        np.testing.assert_allclose(
            np.radians(np.asarray(arm.joint_calls[-1][0], dtype=float)),
            np.zeros(7),
            atol=1e-12,
        )

    def test_heartbeat_reports_inflight_and_missing_success_stalls(self) -> None:
        loop = make_loop(
            FakeRealManArm(),
            heartbeat_timeout=0.05,
        )
        with loop._stats_lock:
            loop._running = True
            loop._last_command_start_ns = 1_000_000_000
            loop._last_command_success_ns = 900_000_000

        with patch(
            "realman_teleop.time.perf_counter_ns",
            return_value=1_060_000_000,
        ):
            inflight_error = loop.heartbeat_error()
        self.assertIn("SDK call has not completed", inflight_error)
        self.assertIn("60.0 ms", inflight_error)

        with loop._stats_lock:
            loop._last_command_start_ns = 1_000_000_000
            loop._last_command_success_ns = 1_000_000_000
        with patch(
            "realman_teleop.time.perf_counter_ns",
            return_value=1_060_000_000,
        ):
            missing_success_error = loop.heartbeat_error()
        self.assertIn("successful packet", missing_success_error)
        self.assertIn("60.0 ms", missing_success_error)

        stop_event = threading.Event()
        loop.report_external_failure(missing_success_error, stop_event)
        self.assertTrue(stop_event.is_set())
        self.assertEqual(loop.snapshot().error, missing_success_error)

    def test_fast_adapter_measures_more_than_100_hz(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm)
        loop.set_joint_target(np.zeros(7))
        stop_event = threading.Event()
        thread = threading.Thread(target=loop.run, args=(stop_event,), daemon=True)
        thread.start()

        measured = loop.wait_until_healthy(stop_event, timeout=1.0)
        stop_event.set()
        thread.join(timeout=1.0)

        snapshot = loop.snapshot()
        self.assertGreater(measured.achieved_hz, 100.0)
        self.assertGreater(snapshot.achieved_hz, 100.0)
        self.assertFalse(snapshot.error)

    def test_single_timing_violation_after_readiness_stops_immediately(self) -> None:
        arm = FakeRealManArm()
        loop = make_loop(arm, maximum_failure_windows=10)
        loop.set_joint_target(np.zeros(7))
        stop_event = threading.Event()
        thread = threading.Thread(target=loop.run, args=(stop_event,), daemon=True)
        thread.start()

        measured = loop.wait_until_healthy(stop_event, timeout=1.0)
        self.assertTrue(measured.timing_verified)
        arm.delay_s = 0.012
        thread.join(timeout=1.0)

        snapshot = loop.snapshot()
        self.assertFalse(thread.is_alive())
        self.assertTrue(stop_event.is_set())
        self.assertIn("SDK call exceeded the 10 ms", snapshot.error)
        self.assertEqual(
            snapshot.consecutive_timing_failure_windows,
            0,
        )

    def test_sustained_rate_below_100_hz_stops_the_loop(self) -> None:
        arm = FakeRealManArm(delay_s=0.012)
        loop = make_loop(arm, maximum_failure_windows=1)
        loop.set_joint_target(np.zeros(7))
        stop_event = threading.Event()
        thread = threading.Thread(target=loop.run, args=(stop_event,), daemon=True)
        thread.start()
        thread.join(timeout=1.0)

        snapshot = loop.snapshot()
        self.assertTrue(stop_event.is_set())
        self.assertFalse(thread.is_alive())
        self.assertIn("at or below", snapshot.error)
        self.assertLessEqual(snapshot.achieved_hz, 100.0)

    def test_sdk_error_stops_the_loop(self) -> None:
        arm = FakeRealManArm(result=-1)
        loop = make_loop(arm)
        loop.set_joint_target(np.zeros(7))
        stop_event = threading.Event()

        loop.run(stop_event)

        self.assertTrue(stop_event.is_set())
        self.assertIn("error code -1", loop.snapshot().error)


class RealManTeleopTest(unittest.TestCase):
    def test_constructor_failure_cleans_an_internally_created_backend(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        with patch(
            "realman_teleop.make_robot_backend",
            return_value=backend,
        ):
            with self.assertRaises(IndexError):
                RealManTeleop([{}], cfg=realman_config())
        self.assertTrue(backend.cleaned)

    def test_realtime_push_enable_callback_disable_lifecycle(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm, api=fake_realtime_api())
        cfg = realman_config(
            REALMAN_STATE_PUSH_TIMEOUT=0.1,
            FORCE_MOVING_AVERAGE_WINDOW=1,
            FORCE_LOW_PASS_ALPHA=1.0,
        )
        teleop = RealManTeleop(
            controller_data(),
            cfg=cfg,
            backend=backend,
        )
        stop_event = threading.Event()
        threads = teleop.start(stop_event)

        snapshot = teleop.state_snapshot()
        np.testing.assert_allclose(
            snapshot.joints,
            np.radians([0.0, 10.0, -20.0, 30.0, -40.0, 50.0, -60.0]),
        )
        self.assertEqual(len(arm.push_configs), 1)
        enabled = arm.push_configs[0]
        self.assertTrue(enabled.enable)
        self.assertEqual(enabled.cycle, cfg.REALMAN_STATE_PUSH_CYCLE_MS)
        self.assertEqual(enabled.port, cfg.REALMAN_STATE_PUSH_PORT)
        self.assertEqual(enabled.force_coordinate, cfg.REALMAN_FORCE_COORDINATE)
        self.assertEqual(enabled.ip, cfg.PC_IP)

        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        teleop.close()

        self.assertEqual(len(arm.push_configs), 2)
        self.assertFalse(arm.push_configs[-1].enable)
        self.assertTrue(backend.cleaned)

        before_late_packet = snapshot.joints.copy()
        arm.realtime_callback(
            realtime_state(joints_degrees=[90.0] * 7)
        )
        np.testing.assert_allclose(
            teleop.state_snapshot().joints,
            before_late_packet,
        )
        with self.assertRaisesRegex(RuntimeError, "closed"):
            teleop.start(threading.Event())

    def test_repeated_start_is_rejected(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm, api=fake_realtime_api())
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(REALMAN_STATE_PUSH_TIMEOUT=0.1),
            backend=backend,
        )
        stop_event = threading.Event()
        threads = teleop.start(stop_event)
        try:
            with self.assertRaisesRegex(RuntimeError, "already been started"):
                teleop.start(stop_event)
        finally:
            stop_event.set()
            for thread in threads:
                thread.join(timeout=1.0)
            teleop.close()

    def test_thread_start_failure_leaves_no_unstarted_worker_to_join(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm, api=fake_realtime_api())
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(REALMAN_STATE_PUSH_TIMEOUT=0.1),
            backend=backend,
        )
        with patch(
            "realman_teleop.threading.Thread.start",
            side_effect=RuntimeError("thread start failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                teleop.start(threading.Event())

        self.assertEqual(teleop.sdk_worker_threads(), ())
        self.assertFalse(arm.push_configs[-1].enable)
        teleop.close()
        self.assertTrue(backend.cleaned)

    def test_polling_fallback_keeps_all_sdk_reads_on_owner_thread(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(
                REALMAN_REALTIME_STATE_PUSH=False,
                REALMAN_RATE_CHECK_WINDOW=0.05,
                REALMAN_RATE_FAILURE_WINDOWS=2,
            ),
            backend=backend,
        )
        backend.sensor_read_thread_ids.clear()
        stop_event = threading.Event()
        threads = teleop.start(stop_event)
        self.assertEqual(len(threads), 1)

        measured = teleop.canfd.wait_until_healthy(stop_event, timeout=1.0)
        self.assertGreater(measured.achieved_hz, 100.0)
        deadline = time.monotonic() + 1.0
        while (
            not backend.sensor_read_thread_ids
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)

        stop_event.set()
        for thread in threads:
            thread.join(timeout=1.0)
        self.assertTrue(backend.sensor_read_thread_ids)
        self.assertEqual(
            set(backend.sensor_read_thread_ids),
            {threads[0].ident},
        )
        teleop.close()

    def test_close_refuses_to_invalidate_a_live_sdk_worker(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(),
            backend=backend,
        )
        release = threading.Event()
        worker = threading.Thread(
            target=release.wait,
            name="realman-canfd",
            daemon=True,
        )
        teleop._sdk_worker_threads = [worker]
        worker.start()

        with self.assertRaisesRegex(RuntimeError, "still alive"):
            teleop.close()
        self.assertFalse(backend.cleaned)
        teleop.quarantine_without_sdk_cleanup()
        self.assertTrue(
            any(
                retained is teleop
                for retained in _UNCLOSED_REALMAN_TELEOPS
            )
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            teleop.start(threading.Event())

        release.set()
        worker.join(timeout=1.0)
        teleop.close()
        self.assertTrue(backend.cleaned)
        self.assertFalse(
            any(
                retained is teleop
                for retained in _UNCLOSED_REALMAN_TELEOPS
            )
        )

    def test_joint_mode_uses_cached_sensors_and_canfd_adapter(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        cfg = realman_config(TELEOP_COMMAND_MODE="joint")
        teleop = RealManTeleop(controller_data(), cfg=cfg, backend=backend)

        self.assertFalse(teleop.process_controller(controller_data(), None, 0.01))
        self.assertTrue(
            teleop.process_controller(
                controller_data(x=0.01, grip=True),
                None,
                0.01,
            )
        )
        self.assertTrue(teleop.canfd.resolve_pending_target())
        teleop.canfd.send_once()

        self.assertEqual(len(arm.joint_calls), 1)
        self.assertEqual(len(backend.ik_calls), 1)
        np.testing.assert_allclose(
            teleop.state_snapshot().wrench,
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        )
        teleop.close()
        self.assertTrue(backend.cleaned)

    def test_tcp_mode_uses_movep_canfd(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        cfg = realman_config(TELEOP_COMMAND_MODE="tcp")
        teleop = RealManTeleop(controller_data(), cfg=cfg, backend=backend)

        teleop.process_controller(controller_data(), None, 0.01)
        self.assertTrue(
            teleop.process_controller(
                controller_data(x=0.01, grip=True),
                None,
                0.01,
            )
        )
        teleop.canfd.send_once()

        self.assertEqual(len(arm.tcp_calls), 1)
        self.assertEqual(len(arm.tcp_calls[-1][0]), 6)
        teleop.close()

    def test_noncommuting_controller_rotation_composes_in_robot_local_frame(
        self,
    ) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(TELEOP_COMMAND_MODE="tcp"),
            backend=backend,
        )
        try:
            controller_reference_rotation = Rotation.from_euler(
                "z",
                40.0,
                degrees=True,
            ).as_matrix()
            local_controller_delta = Rotation.from_euler(
                "x",
                30.0,
                degrees=True,
            ).as_matrix()
            controller_rotation = (
                controller_reference_rotation @ local_controller_delta
            )
            robot_reference_rotation = Rotation.from_euler(
                "y",
                55.0,
                degrees=True,
            ).as_matrix()

            teleop._controller_reference = (
                SE3Container.from_rotation_matrix_and_translation(
                    controller_reference_rotation,
                    np.zeros(3),
                )
            )
            teleop._robot_reference = (
                SE3Container.from_rotation_matrix_and_translation(
                    robot_reference_rotation,
                    np.zeros(3),
                )
            )
            teleop._position_filter.reset()
            teleop._rotation_filter.reset()
            controller = SE3Container.from_rotation_matrix_and_translation(
                controller_rotation,
                np.zeros(3),
            )

            target = teleop._target_from_controller(controller, dt=0.01)
            expected = robot_reference_rotation @ local_controller_delta
            incorrect_left_composition = (
                local_controller_delta @ robot_reference_rotation
            )
            np.testing.assert_allclose(
                target[:3, :3],
                expected,
                atol=1e-10,
            )
            self.assertFalse(
                np.allclose(
                    target[:3, :3],
                    incorrect_left_composition,
                    atol=1e-6,
                )
            )
        finally:
            teleop.close()

    def test_realtime_callback_parses_units_frames_and_wrench(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        cfg = realman_config(
            TELEOP_COMMAND_MODE="joint",
            TCP_POSE=np.array([0.08, -0.02, 0.03, 0.0, 0.0, 0.0]),
            FORCE_MOVING_AVERAGE_WINDOW=1,
            FORCE_LOW_PASS_ALPHA=0.0,
        )
        teleop = RealManTeleop(controller_data(), cfg=cfg, backend=backend)
        try:
            pushed = realtime_state()
            teleop._state_push_received.clear()
            teleop._handle_realtime_state(pushed)
            snapshot = teleop.state_snapshot()

            expected_joints = np.radians(
                np.asarray(pushed.joint_status.joint_position, dtype=float)
            )
            expected_robot_pose = (
                SE3Container.from_euler_angles_and_translation(
                    np.array(
                        [
                            pushed.waypoint.euler.rx,
                            pushed.waypoint.euler.ry,
                            pushed.waypoint.euler.rz,
                        ]
                    ),
                    np.array(
                        [
                            pushed.waypoint.position.x,
                            pushed.waypoint.position.y,
                            pushed.waypoint.position.z,
                        ]
                    ),
                ).homogeneous_matrix
            )
            np.testing.assert_allclose(snapshot.joints, expected_joints)
            np.testing.assert_allclose(
                snapshot.tcp_pose,
                expected_robot_pose @ cfg.TCP_TRANSFORM,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                snapshot.wrench,
                pushed.force_sensor.zero_force,
            )
            self.assertEqual(snapshot.state_error, "")
            self.assertEqual(snapshot.force_error, "")
            self.assertTrue(teleop._state_push_received.is_set())
        finally:
            teleop.close()

    def test_realtime_callback_rejects_parse_and_nonfinite_data(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(
                FORCE_MOVING_AVERAGE_WINDOW=1,
                FORCE_LOW_PASS_ALPHA=0.0,
            ),
            backend=backend,
        )
        try:
            original = teleop.state_snapshot()

            teleop._state_push_received.clear()
            teleop._handle_realtime_state(realtime_state(error_code=-3))
            parse_error = teleop.state_snapshot()
            self.assertIn("parse error -3", parse_error.state_error)
            self.assertEqual(parse_error.force_error, parse_error.state_error)
            self.assertFalse(teleop._state_push_received.is_set())
            np.testing.assert_allclose(parse_error.joints, original.joints)
            np.testing.assert_allclose(parse_error.tcp_pose, original.tcp_pose)
            np.testing.assert_allclose(parse_error.wrench, original.wrench)

            teleop._handle_realtime_state(
                realtime_state(
                    wrench=[1.0, 2.0, np.nan, 0.1, 0.2, 0.3],
                )
            )
            nonfinite_error = teleop.state_snapshot()
            self.assertIn("invalid wrench", nonfinite_error.state_error)
            self.assertEqual(
                nonfinite_error.force_error,
                nonfinite_error.state_error,
            )
            self.assertFalse(teleop._state_push_received.is_set())

            teleop._handle_realtime_state(
                realtime_state(joints_degrees=[0.0] * 6)
            )
            shape_error = teleop.state_snapshot()
            self.assertIn("invalid joints with shape (6,)", shape_error.state_error)
            self.assertEqual(shape_error.force_error, shape_error.state_error)
            self.assertFalse(teleop._state_push_received.is_set())
        finally:
            teleop.close()

    def test_visualizer_payload_contains_only_cached_realman_data(self) -> None:
        arm = FakeRealManArm()
        backend = FakeRealManBackend(arm)
        teleop = RealManTeleop(
            controller_data(),
            cfg=realman_config(TELEOP_COMMAND_MODE="joint"),
            backend=backend,
        )
        stop_event = threading.Event()

        class FakeCameraManager:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self.camera_data = {
                    "camera_0": np.zeros((24, 32, 3), dtype=np.uint8)
                }
                self.camera_data_timestamps_ns = {
                    "camera_0": time.monotonic_ns()
                }
                self.camera_images = {}
                self.camera_image_timestamps_ns = {}
                self.camera_num = 1
                self.realsense_fps = 60.0

        class CapturingHandle:
            sample = None

            def publish(self, sample) -> None:
                self.sample = sample
                stop_event.set()

        handle = CapturingHandle()
        visualizer_publish_loop(
            handle,
            teleop,
            FakeCameraManager(),
            VisualizerConfig(),
            stop_event,
        )

        self.assertIsNotNone(handle.sample)
        self.assertEqual(handle.sample["joints"].shape, (7,))
        np.testing.assert_allclose(
            handle.sample["tcp_translation"],
            [0.4, 0.0, 0.3],
        )
        self.assertNotIn("dataset", handle.sample)
        self.assertNotIn("tactile", handle.sample)
        self.assertIn("camera_0", handle.sample["images"])
        teleop.close()


class ConfigRuntimeDefaultsTest(unittest.TestCase):
    def test_runtime_ur_defaults_use_ur_axes_joints_and_ip_fallback(self) -> None:
        cfg = Config(
            ROBOT_TYPE="ur3e",
            ROBOT_IP=None,
            VR_TO_ROBOT_AXES=None,
            INITIAL_JOINT=None,
        )

        self.assertEqual(cfg.ROBOT_IP, cfg.UR_IP)
        self.assertEqual(cfg.INITIAL_JOINT.shape, (6,))
        np.testing.assert_allclose(
            cfg.INITIAL_JOINT,
            [1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
        )
        np.testing.assert_allclose(
            cfg.VR_TO_ROBOT_AXES,
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
        )

    def test_runtime_realman_defaults_remain_seven_dof(self) -> None:
        cfg = Config(
            ROBOT_TYPE="realman",
            ROBOT_IP="192.0.2.1",
            VR_TO_ROBOT_AXES=None,
            INITIAL_JOINT=None,
        )

        self.assertEqual(cfg.INITIAL_JOINT.shape, (7,))
        np.testing.assert_allclose(
            cfg.VR_TO_ROBOT_AXES,
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )


if __name__ == "__main__":
    unittest.main()
