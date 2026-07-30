"""Explicit stale-input watchdog, zero-velocity stop, hold, and recovery state."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum

from ...config.models import TeleopConfig
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import (
    ClockDomain,
    RobotAction,
    RobotCommandType,
    RobotState,
    VRInputState,
)

_MAX_SEQUENCE = (1 << 64) - 1
_VELOCITY_COMMANDS = {
    RobotCommandType.JOINT_VELOCITY,
    RobotCommandType.TCP_TWIST,
}


class WatchdogState(str, Enum):
    """Teleoperation permission state."""

    HOLDING = "holding"
    RECOVERING = "recovering"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class WatchdogDecision:
    """One freshness evaluation and its required control response."""

    state: WatchdogState
    generation: int
    reason: str
    tripped: bool
    reference_required: bool
    reference_ready: bool
    recovery_samples: int
    vr_age_ns: int | None
    robot_age_ns: int | None
    vr_sequence: int | None
    robot_sequence: int | None


@dataclass(frozen=True, slots=True)
class WatchdogMetrics:
    """Snapshot of evaluations, trips, recoveries, and safety outputs."""

    evaluations: int
    trips: int
    recoveries: int
    hold_actions: int
    zero_velocity_actions: int


def _timestamp_age(
    sample: VRInputState | RobotState,
    *,
    now_ns: int,
    max_age_ns: int,
    future_tolerance_ns: int,
    name: str,
) -> tuple[int | None, str | None]:
    if sample.receive_timestamp_ns is not None:
        timestamp_ns = sample.receive_timestamp_ns
    elif sample.clock_domain is ClockDomain.MONOTONIC:
        timestamp_ns = sample.source_timestamp_ns
    else:
        return None, f"{name}_clock_incomparable"
    age_ns = now_ns - timestamp_ns
    if age_ns < -future_tolerance_ns:
        return age_ns, f"{name}_timestamp_in_future"
    if age_ns > max_age_ns:
        return age_ns, f"{name}_stale"
    return age_ns, None


class TeleopWatchdog:
    """Gate actions until VR and robot state are fresh and references are rebuilt."""

    def __init__(
        self,
        config: TeleopConfig,
        *,
        initially_active: bool = False,
    ) -> None:
        if not isinstance(config, TeleopConfig):
            raise ModelValidationError("watchdog config must be TeleopConfig")
        self._vr_max_age_ns = round(config.vr_input_timeout_s * 1_000_000_000)
        self._robot_max_age_ns = round(config.robot_state_timeout_s * 1_000_000_000)
        self._future_tolerance_ns = round(
            config.watchdog_future_tolerance_s * 1_000_000_000
        )
        self._required_samples = config.watchdog_recovery_samples
        self._lock = threading.RLock()
        self._state = (
            WatchdogState.ACTIVE if initially_active else WatchdogState.HOLDING
        )
        self._reason = "active" if initially_active else "initial_reference_required"
        self._recovery_samples = 0
        self._last_recovery_pair: tuple[int, int] | None = None
        self._latest_fresh_pair: tuple[int, int] | None = None
        self._last_output_sequence: int | None = None
        self._generation = 0
        self._evaluations = 0
        self._trips = 0
        self._recoveries = 0
        self._hold_actions = 0
        self._zero_velocity_actions = 0

    @property
    def state(self) -> WatchdogState:
        with self._lock:
            return self._state

    @property
    def metrics(self) -> WatchdogMetrics:
        with self._lock:
            return WatchdogMetrics(
                evaluations=self._evaluations,
                trips=self._trips,
                recoveries=self._recoveries,
                hold_actions=self._hold_actions,
                zero_velocity_actions=self._zero_velocity_actions,
            )

    def _decision(
        self,
        *,
        tripped: bool,
        vr_age_ns: int | None,
        robot_age_ns: int | None,
        vr_sequence: int | None,
        robot_sequence: int | None,
    ) -> WatchdogDecision:
        ready = (
            self._state is WatchdogState.RECOVERING
            and self._recovery_samples >= self._required_samples
        )
        return WatchdogDecision(
            state=self._state,
            generation=self._generation,
            reason=self._reason,
            tripped=tripped,
            reference_required=self._state is not WatchdogState.ACTIVE,
            reference_ready=ready,
            recovery_samples=self._recovery_samples,
            vr_age_ns=vr_age_ns,
            robot_age_ns=robot_age_ns,
            vr_sequence=vr_sequence,
            robot_sequence=robot_sequence,
        )

    def evaluate(
        self,
        vr_input: VRInputState | None,
        robot_state: RobotState | None,
        *,
        now_ns: int,
    ) -> WatchdogDecision:
        """Evaluate freshness without sending or mutating robot state."""

        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ModelValidationError("now_ns must be a non-negative integer")
        if vr_input is not None and not isinstance(vr_input, VRInputState):
            raise ModelValidationError("vr_input must be VRInputState or None")
        if robot_state is not None and not isinstance(robot_state, RobotState):
            raise ModelValidationError("robot_state must be RobotState or None")
        vr_age = None
        robot_age = None
        problem = None
        if vr_input is None:
            problem = "missing_vr_input"
        else:
            vr_age, problem = _timestamp_age(
                vr_input,
                now_ns=now_ns,
                max_age_ns=self._vr_max_age_ns,
                future_tolerance_ns=self._future_tolerance_ns,
                name="vr_input",
            )
        if problem is None:
            if robot_state is None:
                problem = "missing_robot_state"
            else:
                robot_age, problem = _timestamp_age(
                    robot_state,
                    now_ns=now_ns,
                    max_age_ns=self._robot_max_age_ns,
                    future_tolerance_ns=self._future_tolerance_ns,
                    name="robot_state",
                )
        with self._lock:
            self._evaluations += 1
            self._generation += 1
            tripped = False
            if problem is not None:
                tripped = self._state is WatchdogState.ACTIVE
                if tripped:
                    self._trips += 1
                self._state = WatchdogState.HOLDING
                self._reason = problem
                self._recovery_samples = 0
                self._last_recovery_pair = None
                self._latest_fresh_pair = None
            else:
                assert vr_input is not None and robot_state is not None
                pair = (vr_input.sequence, robot_state.sequence)
                self._latest_fresh_pair = pair
                if self._state is not WatchdogState.ACTIVE:
                    self._state = WatchdogState.RECOVERING
                    if (
                        self._last_recovery_pair is None
                        or (
                            pair[0] != self._last_recovery_pair[0]
                            and pair[1] != self._last_recovery_pair[1]
                        )
                    ):
                        self._recovery_samples += 1
                        self._last_recovery_pair = pair
                    self._reason = (
                        "reference_required"
                        if self._recovery_samples >= self._required_samples
                        else "recovery_samples_required"
                    )
                else:
                    self._reason = "active"
            return self._decision(
                tripped=tripped,
                vr_age_ns=vr_age,
                robot_age_ns=robot_age,
                vr_sequence=None if vr_input is None else vr_input.sequence,
                robot_sequence=None if robot_state is None else robot_state.sequence,
            )

    def acknowledge_reference(
        self,
        *,
        vr_sequence: int,
        robot_sequence: int,
    ) -> None:
        """Activate only after mapping captured the latest approved references."""

        with self._lock:
            if (
                self._state is not WatchdogState.RECOVERING
                or self._recovery_samples < self._required_samples
            ):
                raise LifecycleError("watchdog recovery is not ready for reference")
            if self._latest_fresh_pair != (vr_sequence, robot_sequence):
                raise LifecycleError("reference acknowledgement is not for latest samples")
            self._state = WatchdogState.ACTIVE
            self._reason = "active"
            self._recoveries += 1
            self._generation += 1

    def force_hold(self, reason: str = "safe_hold_requested") -> None:
        """Enter hold due to an explicit command and require normal recovery."""

        if not isinstance(reason, str) or not reason.strip():
            raise ModelValidationError("watchdog hold reason must be non-empty")
        with self._lock:
            if self._state is WatchdogState.ACTIVE:
                self._trips += 1
            self._state = WatchdogState.HOLDING
            self._reason = reason
            self._recovery_samples = 0
            self._last_recovery_pair = None
            self._latest_fresh_pair = None
            self._generation += 1

    def _output_sequence(self, candidate: int) -> int:
        if self._last_output_sequence is None or candidate > self._last_output_sequence:
            result = candidate
        else:
            if self._last_output_sequence >= _MAX_SEQUENCE:
                raise LifecycleError("watchdog action sequence exhausted uint64")
            result = self._last_output_sequence + 1
        self._last_output_sequence = result
        return result

    def guard(
        self,
        action: RobotAction,
        decision: WatchdogDecision,
    ) -> RobotAction:
        """Apply a decision, issuing one zero-velocity stop before steady hold."""

        if not isinstance(action, RobotAction) or not isinstance(
            decision,
            WatchdogDecision,
        ):
            raise ModelValidationError("watchdog guard requires action and decision")
        with self._lock:
            if (
                decision.generation != self._generation
                or decision.state is not self._state
            ):
                raise LifecycleError("watchdog decision is no longer current")
            sequence = self._output_sequence(action.sequence)
            if action.command_type is RobotCommandType.STOP:
                return replace(action, sequence=sequence)
            if decision.state is WatchdogState.ACTIVE:
                return replace(action, sequence=sequence)
            if decision.tripped and action.command_type in _VELOCITY_COMMANDS:
                self._zero_velocity_actions += 1
                return replace(
                    action,
                    sequence=sequence,
                    values=(0.0,) * len(action.values),
                    gripper_width_m=None,
                )
            self._hold_actions += 1
            return RobotAction(
                sequence=sequence,
                source_timestamp_ns=action.source_timestamp_ns,
                receive_timestamp_ns=action.receive_timestamp_ns,
                clock_domain=action.clock_domain,
                command_type=RobotCommandType.HOLD,
                values=(),
            )
