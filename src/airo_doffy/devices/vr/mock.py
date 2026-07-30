"""Deterministic VR input source with scriptable fault injection."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable

from ...config.models import NetworkConfig, VRConfig
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import (
    ClockDomain,
    ControllerState,
    HandSide,
    HandState,
    VRInputMode,
    VRInputState,
)


def _reorder_adjacent(states: tuple[VRInputState, ...]) -> tuple[VRInputState, ...]:
    reordered = list(states)
    for index in range(0, len(reordered) - 1, 2):
        reordered[index], reordered[index + 1] = (
            reordered[index + 1],
            reordered[index],
        )
    return tuple(reordered)


class MockVRInputSource:
    """On-demand controller or hand source with deterministic failure modes.

    Scripted states are returned unchanged, including their sequence numbers,
    so pair reordering and loops can intentionally expose stale input to a
    downstream receiver or watchdog. Without a script, valid monotonic states
    are generated from the configured tracking mode.
    """

    def __init__(
        self,
        config: VRConfig,
        *,
        script: Iterable[VRInputState] | None = None,
        loop_script: bool = False,
        reorder_pairs: bool = False,
        stale_after_s: float | None = None,
        artificial_delay_s: float = 0.0,
        drop_every: int | None = None,
        clock: Clock | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            delay = float(artificial_delay_s)
            stale_after = (
                None if stale_after_s is None else float(stale_after_s)
            )
        except (TypeError, ValueError) as exc:
            raise ModelValidationError("mock VR delays must be numbers") from exc
        if not math.isfinite(delay) or delay < 0:
            raise ModelValidationError(
                "artificial_delay_s must be finite and non-negative"
            )
        if stale_after is not None and (
            not math.isfinite(stale_after) or stale_after < 0
        ):
            raise ModelValidationError(
                "stale_after_s must be finite and non-negative"
            )
        if drop_every is not None and (
            isinstance(drop_every, bool)
            or not isinstance(drop_every, int)
            or drop_every < 1
        ):
            raise ModelValidationError("drop_every must be an integer >= 1")

        states = None if script is None else tuple(script)
        if states is not None and not states:
            raise ModelValidationError("script must contain at least one VR state")
        if states is not None and any(
            not isinstance(state, VRInputState) for state in states
        ):
            raise ModelValidationError("script must contain VRInputState values")
        expected_mode = (
            VRInputMode.CONTROLLERS
            if config.tracking_mode == "controller"
            else VRInputMode.HANDS
        )
        if states is not None and any(state.mode is not expected_mode for state in states):
            raise ModelValidationError(
                "script state modes must match VRConfig.tracking_mode"
            )
        if reorder_pairs and states is None:
            raise ModelValidationError("reorder_pairs requires a script")
        if reorder_pairs:
            states = _reorder_adjacent(states)

        self._mode = expected_mode
        self._script = states
        self._loop_script = bool(loop_script)
        self._stale_after_ns = (
            None if stale_after is None else round(stale_after * 1_000_000_000)
        )
        self._delay_s = delay
        self._drop_every = drop_every
        self._clock = clock or MonotonicClock()
        self._sleep = sleep
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._forced_stale = False
        self._started_at_ns = 0
        self._attempt = 0
        self._script_index = 0

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start a closed mock VR source")
            if self._started:
                raise LifecycleError("mock VR source is already started")
            self._started_at_ns = self._clock.now_ns()
            self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("mock VR source is closed")
        if not self._started:
            raise LifecycleError("mock VR source has not been started")

    def set_stale(self, stale: bool) -> None:
        """Force or clear an immediate no-fresh-input condition."""

        with self._lock:
            if self._closed:
                raise LifecycleError("mock VR source is closed")
            self._forced_stale = bool(stale)

    def _default_state(self, sequence: int, timestamp_ns: int) -> VRInputState:
        if self._mode is VRInputMode.CONTROLLERS:
            controllers = tuple(
                ControllerState(
                    sequence=sequence,
                    source_timestamp_ns=timestamp_ns,
                    receive_timestamp_ns=timestamp_ns,
                    clock_domain=ClockDomain.MONOTONIC,
                    side=side,
                    position_m=(0.0, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                    joystick_xy=(0.0, 0.0),
                    index_trigger=0.0,
                    grip_trigger=0.0,
                    buttons=frozenset(),
                )
                for side in (HandSide.LEFT, HandSide.RIGHT)
            )
            return VRInputState(
                sequence=sequence,
                source_timestamp_ns=timestamp_ns,
                receive_timestamp_ns=timestamp_ns,
                clock_domain=ClockDomain.MONOTONIC,
                mode=self._mode,
                controllers=controllers,
            )
        hand = HandState(
            sequence=sequence,
            source_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns,
            clock_domain=ClockDomain.MONOTONIC,
            side=HandSide.LEFT,
            joints_m=((0.0, 0.0, 0.0),) * 26,
        )
        return VRInputState(
            sequence=sequence,
            source_timestamp_ns=timestamp_ns,
            receive_timestamp_ns=timestamp_ns,
            clock_domain=ClockDomain.MONOTONIC,
            mode=self._mode,
            hands=(hand,),
        )

    def _next_state(self, timestamp_ns: int) -> VRInputState | None:
        if self._script is None:
            return self._default_state(self._attempt, timestamp_ns)
        if self._script_index >= len(self._script):
            if not self._loop_script:
                return None
            self._script_index = 0
        state = self._script[self._script_index]
        self._script_index += 1
        return state

    def read_latest(self) -> VRInputState | None:
        with self._lock:
            self._require_started()
            if self._delay_s:
                self._sleep(self._delay_s)
            timestamp_ns = self._clock.now_ns()
            if self._forced_stale or (
                self._stale_after_ns is not None
                and timestamp_ns - self._started_at_ns >= self._stale_after_ns
            ):
                return None
            state = self._next_state(timestamp_ns)
            self._attempt += 1
            if (
                self._drop_every is not None
                and self._attempt % self._drop_every == 0
            ):
                return None
            return state

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._started = False
            self._closed = True


def create_mock_vr(
    config: VRConfig,
    _network: NetworkConfig,
) -> MockVRInputSource:
    """Create an unstarted mock VR source for the configured tracking mode."""

    return MockVRInputSource(config)
