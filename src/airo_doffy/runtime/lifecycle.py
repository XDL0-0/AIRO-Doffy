"""Deterministic resource and worker-thread lifecycle management."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from threading import Event, RLock, Thread

from ..core.errors import LifecycleError, ModelValidationError
from ..core.interfaces import Lifecycle


class LifecycleManagerState(str, Enum):
    """Lifecycle manager state visible to health/status code."""

    NEW = "new"
    STARTING = "starting"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LifecycleManagerSnapshot:
    """Immutable manager status."""

    state: LifecycleManagerState
    configured: tuple[str, ...]
    started: tuple[str, ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Immutable managed-thread health status."""

    name: str
    running: bool
    started: bool
    closed: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class _NamedLifecycle:
    name: str
    component: Lifecycle


class LifecycleManager:
    """Start resources in order and close successful starts in reverse order."""

    def __init__(
        self,
        components: Iterable[tuple[str, Lifecycle]] = (),
    ) -> None:
        checked = []
        names = set()
        for name, component in components:
            if not isinstance(name, str) or not name.strip():
                raise ModelValidationError("lifecycle names must be non-empty strings")
            if name in names:
                raise ModelValidationError(f"duplicate lifecycle name: {name}")
            if not isinstance(component, Lifecycle):
                raise ModelValidationError(
                    f"component {name!r} must satisfy Lifecycle"
                )
            names.add(name)
            checked.append(_NamedLifecycle(name=name, component=component))
        self._configured = tuple(checked)
        self._started: list[_NamedLifecycle] = []
        self._state = LifecycleManagerState.NEW
        self._error: str | None = None
        self._lock = RLock()

    def snapshot(self) -> LifecycleManagerSnapshot:
        with self._lock:
            return LifecycleManagerSnapshot(
                state=self._state,
                configured=tuple(item.name for item in self._configured),
                started=tuple(item.name for item in self._started),
                error=self._error,
            )

    def start(self) -> None:
        with self._lock:
            if self._state is not LifecycleManagerState.NEW:
                raise LifecycleError(
                    f"lifecycle manager cannot start from {self._state.value}"
                )
            self._state = LifecycleManagerState.STARTING
        current: _NamedLifecycle | None = None
        try:
            for current in self._configured:
                current.component.start()
                with self._lock:
                    self._started.append(current)
        except BaseException as exc:
            cleanup_errors = self._close_started()
            detail = f"{type(exc).__name__}: {exc}"
            if cleanup_errors:
                detail += "; cleanup: " + "; ".join(cleanup_errors)
            with self._lock:
                self._state = LifecycleManagerState.FAILED
                self._error = detail
            failed_name = "unknown" if current is None else current.name
            raise LifecycleError(
                f"failed to start component {failed_name}: {detail}"
            ) from exc
        with self._lock:
            self._state = LifecycleManagerState.RUNNING

    def close(self) -> None:
        with self._lock:
            if self._state is LifecycleManagerState.CLOSED:
                return
            if self._state is LifecycleManagerState.STARTING:
                raise LifecycleError("cannot close while lifecycle manager is starting")
            self._state = LifecycleManagerState.CLOSING
        errors = self._close_started()
        with self._lock:
            self._state = LifecycleManagerState.CLOSED
            if errors:
                self._error = "; ".join(errors)
        if errors:
            raise LifecycleError(
                "lifecycle cleanup failed: " + "; ".join(errors)
            )

    def _close_started(self) -> list[str]:
        errors = []
        while True:
            with self._lock:
                if not self._started:
                    break
                item = self._started.pop()
            try:
                item.component.close()
            except BaseException as exc:
                errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
        return errors


class ManagedWorker:
    """Lifecycle adapter for one caller-owned, stoppable worker thread."""

    def __init__(
        self,
        target: Callable[[Event], None],
        *,
        name: str,
        join_timeout_s: float = 5.0,
        daemon: bool = False,
    ) -> None:
        if not callable(target):
            raise ModelValidationError("worker target must be callable")
        if not isinstance(name, str) or not name.strip():
            raise ModelValidationError("worker name must be a non-empty string")
        if join_timeout_s < 0:
            raise ModelValidationError("join_timeout_s must be non-negative")
        self._target = target
        self._name = name
        self._join_timeout_s = float(join_timeout_s)
        self._daemon = bool(daemon)
        self._stop = Event()
        self._lock = RLock()
        self._thread: Thread | None = None
        self._started = False
        self._closed = False
        self._error: BaseException | None = None

    @property
    def stop_event(self) -> Event:
        return self._stop

    def snapshot(self) -> WorkerSnapshot:
        with self._lock:
            thread = self._thread
            return WorkerSnapshot(
                name=self._name,
                running=thread is not None and thread.is_alive(),
                started=self._started,
                closed=self._closed,
                error=(
                    None
                    if self._error is None
                    else f"{type(self._error).__name__}: {self._error}"
                ),
            )

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError(f"cannot start closed worker {self._name}")
            if self._started:
                raise LifecycleError(f"worker {self._name} is already started")
            self._started = True
            self._thread = Thread(
                target=self._run,
                name=self._name,
                daemon=self._daemon,
            )
            self._thread.start()

    def check_health(self) -> None:
        with self._lock:
            if self._error is not None:
                raise LifecycleError(
                    f"worker {self._name} failed: "
                    f"{type(self._error).__name__}: {self._error}"
                ) from self._error
            if self._started and not self._closed:
                thread = self._thread
                if thread is not None and not thread.is_alive() and not self._stop.is_set():
                    raise LifecycleError(f"worker {self._name} stopped unexpectedly")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            thread = self._thread
            self._stop.set()
        if thread is not None:
            thread.join(self._join_timeout_s)
            if thread.is_alive():
                raise LifecycleError(
                    f"worker {self._name} did not stop before timeout"
                )
        with self._lock:
            self._closed = True
            self._started = False
            self._thread = None

    def _run(self) -> None:
        try:
            self._target(self._stop)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._stop.set()
