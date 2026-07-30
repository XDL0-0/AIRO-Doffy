"""Bounded background export queue for recording persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Protocol

from ..core.errors import LifecycleError, ModelValidationError
from .errors import ExportQueueFullError
from .samples import Episode
from .writers.base import EpisodeRollback, EpisodeWriter


class ExportTaskKind(str, Enum):
    """Kinds of storage work accepted by the export worker."""

    WRITE = "write"
    ROLLBACK = "rollback"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Terminal result for one queued storage operation."""

    kind: ExportTaskKind
    episode_index: int
    succeeded: bool
    path: Path | None = None
    storage_changed: bool | None = None
    error: BaseException | None = None


class ExportTicket:
    """Waitable result handle returned without blocking the submitter."""

    def __init__(self, *, kind: ExportTaskKind, episode_index: int) -> None:
        self.kind = kind
        self.episode_index = episode_index
        self._event = Event()
        self._result: ExportResult | None = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    def result(self, timeout_s: float | None = None) -> ExportResult:
        if timeout_s is not None and timeout_s < 0:
            raise ModelValidationError("timeout_s must be non-negative or None")
        if not self._event.wait(timeout_s):
            raise TimeoutError(
                f"{self.kind.value} for episode {self.episode_index} is still pending"
            )
        assert self._result is not None
        return self._result

    def _resolve(self, result: ExportResult) -> None:
        self._result = result
        self._event.set()


@dataclass(frozen=True, slots=True)
class ExportWorkerMetrics:
    """Small immutable queue and outcome snapshot."""

    capacity: int
    queued: int
    busy: bool
    submitted: int
    completed: int
    failed: int
    rejected: int


@dataclass(frozen=True, slots=True)
class _ExportTask:
    kind: ExportTaskKind
    episode_index: int
    ticket: ExportTicket
    episode: Episode | None = None


class _Clock(Protocol):
    def __call__(self) -> float: ...


class ExportWorker:
    """Run all disk and dataset conversion work on one bounded worker thread."""

    def __init__(
        self,
        writer: EpisodeWriter,
        *,
        rollback: EpisodeRollback | None = None,
        capacity: int = 2,
        poll_interval_s: float = 0.05,
        thread_name: str = "recording-export",
        clock: _Clock = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ModelValidationError("capacity must be a positive integer")
        if poll_interval_s <= 0:
            raise ModelValidationError("poll_interval_s must be positive")
        if not isinstance(thread_name, str) or not thread_name:
            raise ModelValidationError("thread_name must be a non-empty string")
        self._writer = writer
        self._rollback = rollback
        self._capacity = capacity
        self._poll_interval_s = float(poll_interval_s)
        self._thread_name = thread_name
        self._clock = clock
        self._queue: Queue[_ExportTask] = Queue(maxsize=capacity)
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._accepting = False
        self._drain = True
        self._busy = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise LifecycleError("export worker is already started")
            self._accepting = True
            self._thread = Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def submit_export(self, episode: Episode) -> ExportTicket:
        if not isinstance(episode, Episode):
            raise ModelValidationError("episode must be an Episode")
        return self._submit(
            _ExportTask(
                kind=ExportTaskKind.WRITE,
                episode_index=episode.index,
                episode=episode,
                ticket=ExportTicket(
                    kind=ExportTaskKind.WRITE,
                    episode_index=episode.index,
                ),
            )
        )

    def submit_rollback(self, episode_index: int) -> ExportTicket:
        if self._rollback is None:
            raise LifecycleError("no rollback implementation is configured")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ModelValidationError("episode_index must be a non-negative integer")
        return self._submit(
            _ExportTask(
                kind=ExportTaskKind.ROLLBACK,
                episode_index=episode_index,
                ticket=ExportTicket(
                    kind=ExportTaskKind.ROLLBACK,
                    episode_index=episode_index,
                ),
            )
        )

    def metrics(self) -> ExportWorkerMetrics:
        with self._lock:
            return ExportWorkerMetrics(
                capacity=self._capacity,
                queued=self._queue.qsize(),
                busy=self._busy,
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                rejected=self._rejected,
            )

    def close(self, *, drain: bool = True, timeout_s: float = 10.0) -> None:
        if timeout_s < 0:
            raise ModelValidationError("timeout_s must be non-negative")
        with self._lock:
            thread = self._thread
            if thread is None:
                self._accepting = False
                self._writer.close()
                return
            self._accepting = False
            self._drain = bool(drain)
            self._stop.set()
        thread.join(timeout_s)
        if thread.is_alive():
            raise LifecycleError("export worker did not stop before timeout")
        with self._lock:
            self._thread = None

    def _submit(self, task: _ExportTask) -> ExportTicket:
        with self._lock:
            if not self._accepting:
                raise LifecycleError("export worker is not accepting work")
            try:
                self._queue.put_nowait(task)
            except Full as exc:
                self._rejected += 1
                raise ExportQueueFullError(
                    f"export queue is full (capacity={self._capacity})"
                ) from exc
            self._submitted += 1
        return task.ticket

    def _run(self) -> None:
        try:
            while True:
                if self._stop.is_set() and (not self._drain or self._queue.empty()):
                    break
                try:
                    task = self._queue.get(timeout=self._poll_interval_s)
                except Empty:
                    continue
                with self._lock:
                    self._busy = True
                try:
                    task.ticket._resolve(self._execute(task))
                finally:
                    with self._lock:
                        self._busy = False
                    self._queue.task_done()
            if not self._drain:
                self._cancel_queued()
        finally:
            self._writer.close()

    def _execute(self, task: _ExportTask) -> ExportResult:
        try:
            if task.kind is ExportTaskKind.WRITE:
                assert task.episode is not None
                path = self._writer.write(task.episode)
                result = ExportResult(
                    kind=task.kind,
                    episode_index=task.episode_index,
                    succeeded=True,
                    path=path,
                )
            else:
                assert self._rollback is not None
                changed = self._rollback.rollback(task.episode_index)
                result = ExportResult(
                    kind=task.kind,
                    episode_index=task.episode_index,
                    succeeded=changed,
                    storage_changed=changed,
                )
        except BaseException as exc:
            result = ExportResult(
                kind=task.kind,
                episode_index=task.episode_index,
                succeeded=False,
                error=exc,
            )
        with self._lock:
            if result.succeeded:
                self._completed += 1
            else:
                self._failed += 1
        return result

    def _cancel_queued(self) -> None:
        while True:
            try:
                task = self._queue.get_nowait()
            except Empty:
                return
            error = LifecycleError("export task cancelled during worker shutdown")
            task.ticket._resolve(
                ExportResult(
                    kind=task.kind,
                    episode_index=task.episode_index,
                    succeeded=False,
                    error=error,
                )
            )
            with self._lock:
                self._failed += 1
            self._queue.task_done()
